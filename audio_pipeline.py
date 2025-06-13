#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
audio_pipeline.py
Core audio processing pipeline for the Voice Extractor.
Includes vocal separation, diarization, speaker identification, overlap detection,
verification, transcription, and concatenation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import shutil
import time
import subprocess
import os
import re
import tempfile

os.environ["SPEECHBRAIN_FETCH_LOCAL_STRATEGY"] = "copy"  # For SpeechBrain on Windows

import torch
import torchaudio as ta
import librosa
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio import Model as PyannoteModel
from pyannote.audio.pipelines import OverlappedSpeechDetection as PyannoteOSDPipeline
from pyannote.core import Segment, Timeline, Annotation
import whisper
import ffmpeg
from rich.progress import (
    Progress,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.table import Table

try:
    from transformers import pipeline as transformers_pipeline
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


# Bandit-v2 (Vocal Separation)
HAVE_BANDIT_V2 = os.path.exists(os.environ.get("BANDIT_REPO_PATH", "repos/bandit-v2"))


# WeSpeaker (Speaker Embedding)
try:
    import wespeaker

    HAVE_WESPEAKER = True
except ImportError:
    HAVE_WESPEAKER = False
    wespeaker = None

# SpeechBrain (Speaker Verification - ECAPA-TDNN)
try:
    from speechbrain.inference.speaker import (
        SpeakerRecognition as SpeechBrainSpeakerRecognition,
    )

    HAVE_SPEECHBRAIN = True
except ImportError:
    HAVE_SPEECHBRAIN = False
    SpeechBrainSpeakerRecognition = None


from common import (
    log,
    console,
    DEVICE,
    ff_trim,
    ff_slice_smart,
    cos,
    DEFAULT_MIN_SEGMENT_SEC,
    DEFAULT_MAX_MERGE_GAP,
    ensure_dir_exists,
    safe_filename,
    format_duration,
)

# --- Model Initialization Functions ---
def init_bandit_separator(model_checkpoint_path: Path) -> dict | None:
    """
    Loads and returns the Bandit-v2 model and required components.
    
    This optimized version loads the model once and returns a dict containing:
    - system: The loaded PyTorch Lightning system with the model
    - config: The Hydra configuration
    - repo_path: Path to the Bandit-v2 repository
    
    Usage:
        # Load model once
        bandit_model = init_bandit_separator(checkpoint_path)
        
        # Use multiple times (much faster than CLI method)
        vocals1 = run_bandit_vocal_separation(audio1, bandit_model, output_dir)
        vocals2 = run_bandit_vocal_separation(audio2, bandit_model, output_dir)
        vocals3 = run_bandit_vocal_separation(audio3, bandit_model, output_dir)
    
    Returns:
        dict with loaded model components, or None if loading failed
    """
      # Set up logging fallback
    import logging
    try:
        # Try to use the global log if available
        if 'log' in globals() and log is not None:
            logger = log
        else:
            raise AttributeError("Global log not available")
    except (NameError, AttributeError):
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)
      # Try to get DEVICE, with fallback
    try:
        device = DEVICE
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except NameError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not model_checkpoint_path.exists():
        logger.error(f"Bandit-v2 model checkpoint not found at: {model_checkpoint_path}")
        return None
    
    # Resolve the absolute checkpoint path FIRST, before changing working directory
    absolute_checkpoint_path = model_checkpoint_path.resolve()    
    # Verify that the inference script exists
    bandit_repo_path = Path(os.environ.get("BANDIT_REPO_PATH", "repos/bandit-v2")).resolve()
    inference_script = bandit_repo_path / "inference.py"

    if not inference_script.exists():
        logger.error(f"Bandit-v2 inference.py not found at: {inference_script}")
        return None

    logger.info(f"Loading Bandit-v2 model from checkpoint: {model_checkpoint_path.name}")

    try:
        # Add the bandit repo to sys.path to access its modules
        if str(bandit_repo_path) not in sys.path:
            sys.path.insert(0, str(bandit_repo_path))
        
        # Import required modules from bandit-v2
        from src.system.utils import build_system
        
        # Load and prepare the config
        config_path = bandit_repo_path / "expt" / "inference.yaml"
        
        if not config_path.exists():
            logger.error(f"Bandit-v2 config file not found at: {config_path}")
            return None
            
        # Read the config with Hydra/OmegaConf
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace environment variables in config
        repo_path_url = bandit_repo_path.as_posix()
        config_content = config_content.replace('$REPO_ROOT', repo_path_url)
        config_content = config_content.replace('data: dnr-v3-com-smad-multi-v2', 'data: dnr-v3-com-smad-multi-v2b')
        
        # Create a temporary config file for loading
        temp_config_path = bandit_repo_path / "temp_inference_config.yaml"
        with open(temp_config_path, 'w') as f:
            f.write(config_content)
        
        try:
            # Set environment variables for Hydra  
            os.environ["HYDRA_FULL_ERROR"] = "1"
            os.environ["REPO_ROOT"] = str(bandit_repo_path)
            original_cwd = os.getcwd()
            os.chdir(bandit_repo_path)
            
            # Initialize Hydra and load config
            from hydra import initialize_config_dir, compose
            from hydra.core.global_hydra import GlobalHydra
            
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
            
            # Try to load config with proper fallback handling
            config_dir = (bandit_repo_path / "expt").resolve()
            print("Using config directory:", config_dir)
            try:
                with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                    # First try the original config name
                    cfg = compose(config_name="v3-48k-smad-eng-test")
            except Exception as e:
                logger.warning(f"Could not load v3-48k-smad-eng-test config: {e}")
                try:
                    # Try without the problematic inference defaults
                    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                        cfg = compose(config_name="inference")
                except Exception as e2:
                    logger.warning(f"Could not load any Hydra config: {e2}")
                    # Use direct model loading without build_system
                    logger.info("Using direct model loading approach...")
                      # Import model directly
                    from src.models.bandit.bandit import Bandit
                      # Create model with standard parameters matching the checkpoint
                    model = Bandit(
                        in_channels=1,  # Mono input
                        stems=['speech', 'music', 'sfx'],  # Must match the checkpoint training stems
                        fs=48000,
                        band_type='musical',
                        n_bands=64,
                        normalize_channel_independently=False,
                        treat_channel_as_feature=True,
                        n_sqm_modules=8,
                        emb_dim=128,
                        rnn_dim=256,
                        bidirectional=True,
                        rnn_type='GRU',
                        mlp_dim=512,
                        hidden_activation='Tanh',
                        hidden_activation_kwargs=None,
                        complex_mask=True,
                        use_freq_weights=True,
                        n_fft=2048,
                        win_length=2048,
                        hop_length=512,
                        window_fn='hann_window'
                    )
                      # Load the checkpoint directly into the model
                    logger.info(f"Loading checkpoint weights from: {model_checkpoint_path}")
                    
                    # Use the pre-resolved absolute path for loading
                    checkpoint = torch.load(str(absolute_checkpoint_path), map_location='cpu')
                    
                    # Extract just the model state dict if it's wrapped in a system
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                        # Remove 'model.' prefix if present
                        model_state_dict = {}
                        for key, value in state_dict.items():
                            if key.startswith('model.'):
                                model_state_dict[key[6:]] = value
                            else:
                                model_state_dict[key] = value
                        
                        # Try strict loading first to see what's mismatched
                        try:
                            model.load_state_dict(model_state_dict, strict=True)
                            logger.info("✓ Checkpoint loaded with strict=True")
                        except Exception as strict_error:
                            logger.warning(f"Strict loading failed: {strict_error}")
                            logger.info("Trying with strict=False...")
                            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                            if missing_keys:
                                logger.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                            if unexpected_keys:
                                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                      # Move to device and set to eval mode
                    model.to(device)
                    model.eval()
                    
                    logger.info(f"✓ Bandit-v2 model loaded successfully on {device.type.upper()} (direct loading)")
                    
                    return {
                        "system": model,  # For compatibility, store model as system
                        "config": None,   # No config needed for direct loading
                        "repo_path": bandit_repo_path,
                        "direct_model": True,  # Flag to indicate this is direct loading
                        "checkpoint_path": absolute_checkpoint_path  # Store original path for fallback
                    }
            
            # Override checkpoint path regardless of how we got the config
            cfg.ckpt_path = str(model_checkpoint_path)
            cfg.fs = 48000  # Bandit-v2 typically uses 48kHz
            
            # Build the system
            system = build_system(cfg)
            
            # Load the checkpoint
            logger.info(f"Loading checkpoint weights from: {model_checkpoint_path}")
            checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
            system.load_state_dict(checkpoint["state_dict"], strict=False)
            
            # Move to device and set to eval mode
            system.to(device)
            system.eval()
            
            logger.info(f"✓ Bandit-v2 model loaded successfully on {device.type.upper()}")
            
            return {
                "system": system,
                "config": cfg,
                "repo_path": bandit_repo_path
            }
                
        finally:
            # Cleanup
            os.chdir(original_cwd)
            if temp_config_path.exists():
                temp_config_path.unlink()
                
    except Exception as e:
        logger.error(f"Failed to load Bandit-v2 model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def init_wespeaker_models(
    rvector_id_or_path: str, gemini_id_or_path: str
) -> dict | None:
    """Initializes WeSpeaker models (Deep r-vector and speaker verification)."""
    if not HAVE_WESPEAKER:
        log.error("WeSpeaker library not found. Please ensure it's installed.")
        return None

    models = {"rvector": None, "gemini": None}

    # For automatic model downloading, WeSpeaker uses 'english' or 'chinese'
    # 'english': ResNet221_LM pretrained on VoxCeleb
    # 'chinese': ResNet34_LM pretrained on CnCeleb
    model_configs = {
        "rvector": {"id_or_path": rvector_id_or_path, "desc": "Deep r-vector"},
        "gemini": {"id_or_path": gemini_id_or_path, "desc": "speaker verification"},
    }

    for model_key, config in model_configs.items():
        model_id_or_path = config["id_or_path"]
        model_desc = config["desc"]
        log.info(f"Initializing WeSpeaker {model_desc} model: {model_id_or_path}")

        try:
            # Check if it's a local path with the required files
            local_path = Path(model_id_or_path)
            if (
                local_path.is_dir()
                and (local_path / "avg_model.pt").exists()
                and (local_path / "config.yaml").exists()
            ):
                log.info(
                    f"Loading WeSpeaker {model_desc} from local path: {model_id_or_path}"
                )
                model = wespeaker.load_model_local(str(local_path))
            else:
                # Use the standard load_model function which handles automatic downloading
                log.info(
                    f"Loading WeSpeaker {model_desc} model (auto-download if needed): {model_id_or_path}"
                )

                # WeSpeaker accepts 'english' or 'chinese' as model identifiers
                if model_id_or_path.lower() not in ["english", "chinese"]:
                    log.warning(
                        f"Unknown model identifier '{model_id_or_path}', defaulting to 'english'"
                    )
                    model_id = "english"
                else:
                    model_id = model_id_or_path.lower()

                # Download with retry logic for reliability
                model = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        log.info(
                            f"Downloading WeSpeaker '{model_id}' model (attempt {attempt + 1}/{max_retries})..."
                        )
                        model = wespeaker.load_model(model_id)
                        log.info(
                            f"[green]✓ Successfully loaded WeSpeaker '{model_id}' model[/]"
                        )
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5  # Progressive backoff
                            log.warning(
                                f"Download failed: {e}. Retrying in {wait_time} seconds..."
                            )
                            time.sleep(wait_time)
                        else:
                            log.error(
                                f"Failed to download WeSpeaker '{model_id}' model after {max_retries} attempts: {e}"
                            )
                            raise

                if model is None:
                    raise RuntimeError(f"Failed to load WeSpeaker model '{model_id}'")

            model.set_device(DEVICE.type)
            models[model_key] = model
            log.info(
                f"[green]✓ WeSpeaker {model_desc} model loaded to {DEVICE.type.upper()}.[/]"
            )

        except Exception as e:
            log.error(f"Failed to load WeSpeaker {model_desc} model: {e}")
            log.error("This may be due to network issues during model download.")
            log.error("Please check your internet connection and try again.")
            # For essential models, we should fail here
            if (
                model_key == "rvector"
            ):  # r-vector is critical for speaker identification
                return None

    # If at least the critical r-vector model loaded, we can proceed
    if models["rvector"] is not None:
        if models["gemini"] is None:
            log.warning(
                "Gemini model failed to load. Speaker verification may be less accurate."
            )
            # Both models should use the same one for consistency
            models["gemini"] = models["rvector"]
        return models
    else:
        log.error("Critical r-vector model failed to initialize. Cannot proceed.")
        return None


def init_speechbrain_speaker_recognition_model(
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
):
    """Initializes the SpeechBrain SpeakerRecognition model (ECAPA-TDNN)."""
    if not HAVE_SPEECHBRAIN:
        log.warning(
            "SpeechBrain library not found or import failed. SpeechBrain ECAPA-TDNN verification will be skipped."
        )
        return None

    log.info(f"Initializing SpeechBrain SpeakerRecognition model: {model_source}")
    if os.name == "nt" and os.getenv("SPEECHBRAIN_FETCH_LOCAL_STRATEGY") != "copy":
        log.warning(
            "SPEECHBRAIN_FETCH_LOCAL_STRATEGY is not 'copy'. This may cause issues on Windows with symlinks. "
            "Set environment variable SPEECHBRAIN_FETCH_LOCAL_STRATEGY=copy if errors occur."
        )
    try:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        user_cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
        # Ensure savedir is specific to avoid conflicts if multiple SpeechBrain models are used project-wide
        savedir_name = model_source.replace("/", "_").replace(
            "@", "_"
        )  # Sanitize name for directory
        savedir = user_cache_dir / "voice_extractor_speechbrain_cache" / savedir_name
        ensure_dir_exists(savedir)

        model = SpeechBrainSpeakerRecognition.from_hparams(
            source=model_source, savedir=str(savedir), run_opts={"device": DEVICE.type}
        )
        model.eval()  # Set to evaluation mode
        log.info(
            f"[green]✓ SpeechBrain model '{model_source}' loaded to {DEVICE.type.upper()}.[/]"
        )
        return model
    except Exception as e:
        log.error(
            f"Failed to load SpeechBrain SpeakerRecognition model '{model_source}': {e}"
        )
        return None


# --- Noise Classifier ---
class NoiseClassifier:
    """Classifies audio segments as 'clean' or 'noisy' using a Hugging Face transformer model."""

    def __init__(
        self, model_id: str = "speechbrain/urbansound8k_ecapa", device_to_use=DEVICE
    ):
        self.model_id = model_id
        self.device = device_to_use
        self.classifier = None  # Initialize classifier as None, load on demand
        if not HAVE_TRANSFORMERS:
            log.error(
                "Transformers library not available. NoiseClassifier will not function."
            )

    def load_model(self):
        """Loads the audio classification model into VRAM."""
        if not HAVE_TRANSFORMERS:
            log.debug(
                "Transformers library not found, cannot load noise classification model."
            )
            return

        if self.classifier is None:
            log.info(
                f"Loading noise classification model: {self.model_id} to {self.device.type}"
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()  # Free VRAM before loading
            try:
                self.classifier = transformers_pipeline(
                    "audio-classification", model=self.model_id, device=self.device
                )
                log.info(f"[green]✓ Noise classifier '{self.model_id}' loaded.[/]")
            except Exception as e:
                log.error(f"Failed to load noise classifier '{self.model_id}': {e}")
                self.classifier = None  # Ensure it's None on failure

    def unload_model(self):
        """Unloads the audio classification model from VRAM."""
        if self.classifier is not None:
            log.info(f"Unloading noise classification model: {self.model_id}")
            del self.classifier
            self.classifier = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            log.info(f"[green]✓ Noise classifier '{self.model_id}' unloaded.[/]")

    def classify(self, audio_path: str, confidence_threshold: float = 0.3) -> str:
        """
        Classifies the audio file at audio_path.
        Returns 'noisy', 'clean', or 'unknown'.
        Assumes speechbrain/urbansound8k_ecapa or similar model where all output labels are non-speech sounds.
        """
        if not HAVE_TRANSFORMERS:
            return "unknown"

        self.load_model()  # Ensure model is loaded

        if self.classifier is None:
            log.error("Noise classifier model could not be loaded. Cannot classify.")
            return "unknown"

        try:
            if not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
                log.error(
                    f"Audio file for classification not found or empty: {audio_path}"
                )
                return "unknown"

            result = self.classifier(audio_path)

            if (
                result
                and isinstance(result, list)
                and result[0]
                and "score" in result[0]
                and "label" in result[0]
            ):
                top_result = result[0]
                log.debug(
                    f"Noise classification for {Path(audio_path).name}: Top label '{top_result['label']}' (score: {top_result['score']:.2f})"
                )
                if top_result["score"] >= confidence_threshold:
                    return "noisy"
                else:
                    return "clean"
            else:
                log.warning(
                    f"Noise classification for {Path(audio_path).name} returned empty or unexpected result: {result}"
                )
                return "unknown"
        except Exception as e:
            log.error(
                f"Error during noise classification for {Path(audio_path).name}: {e}"
            )
            return "unknown"


# --- Pipeline Stages ---


def prepare_reference_audio(
    reference_audio_path_arg: Path, tmp_dir: Path, target_name: str
) -> Path:
    log.info(
        f"Preparing reference audio for '{target_name}' from: {reference_audio_path_arg.name}"
    )
    ensure_dir_exists(tmp_dir)
    processed_ref_filename = (
        f"{safe_filename(target_name)}_reference_processed_16k_mono.wav"
    )
    processed_ref_path = tmp_dir / processed_ref_filename
    if not reference_audio_path_arg.exists():
        raise FileNotFoundError(
            f"Reference audio file not found: {reference_audio_path_arg}"
        )
    try:
        # WeSpeaker and SpeechBrain typically expect 16kHz mono
        ff_trim(
            reference_audio_path_arg,
            processed_ref_path,
            0,
            999999,
            target_sr=16000,
            target_ac=1,
        )
        if not processed_ref_path.exists() or processed_ref_path.stat().st_size == 0:
            raise RuntimeError(
                "Processed reference audio file is empty or was not created."
            )
        log.info(
            f"Processed reference audio (16kHz, mono) saved to: {processed_ref_path.name}"
        )
        return processed_ref_path
    except Exception as e:
        log.error(
            f"Failed to process reference audio '{reference_audio_path_arg.name}': {e}"
        )
        raise


def run_bandit_vocal_separation(
    input_audio_file: Path,
    bandit_separator,  # Can be either Path (old way) or dict (new loaded model way)
    output_dir: Path,
    chunk_minutes: float = 5.0,
) -> Path | None:
    """Performs vocal separation using Bandit-v2. 
    
    Args:
        bandit_separator: Either a Path to checkpoint (for subprocess method) 
                         or a dict with loaded model (for direct method)
    """
    
    # Check if we have a loaded model (new way) or checkpoint path (old way)
    if isinstance(bandit_separator, dict) and "system" in bandit_separator:
        # Try the new direct method with loaded model first
        try:
            result = run_bandit_vocal_separation_direct(
                input_audio_file, bandit_separator, output_dir, chunk_minutes
            )
            if result is not None:
                return result
            else:
                log.warning("Direct Bandit method returned None, falling back to subprocess method")
        except Exception as e:
            log.warning(f"Direct Bandit method failed: {e}, falling back to subprocess method")
        
        # If direct method failed, fall back to subprocess method
        # Extract the original checkpoint path for subprocess method
        if "repo_path" in bandit_separator:
            checkpoint_path = bandit_separator.get("checkpoint_path")
            if checkpoint_path is None:
                # Try to find the checkpoint in the expected location
                checkpoint_path = Path("models/bandit_checkpoint_eng.ckpt")
            
            return run_bandit_vocal_separation_subprocess(
                input_audio_file, checkpoint_path, output_dir, chunk_minutes
            )
        else:
            log.error("Failed to find checkpoint path for subprocess fallback")
            return None
            
    elif isinstance(bandit_separator, Path):
        # Use the old subprocess method
        return run_bandit_vocal_separation_subprocess(
            input_audio_file, bandit_separator, output_dir, chunk_minutes
        )
    else:
        log.error("Invalid bandit_separator: must be either a Path or a loaded model dict")
        return None


def run_bandit_vocal_separation_subprocess(
    input_audio_file: Path,
    bandit_separator: Path,  # Checkpoint path for subprocess method
    output_dir: Path,
    chunk_minutes: float = 5.0,
) -> Path | None:
    """Performs vocal separation using Bandit-v2 via subprocess (original method)."""

    checkpoint_path = bandit_separator

    log.info(f"Starting vocal separation with Bandit-v2 (subprocess) for: {input_audio_file.name}")
    ensure_dir_exists(output_dir)

    # Output filename for the vocals stem
    vocals_output_filename = (
        output_dir / f"{input_audio_file.stem}_vocals_bandit_v2.wav"
    )

    if vocals_output_filename.exists() and vocals_output_filename.stat().st_size > 0:
        log.info(
            f"Found existing Bandit-v2 vocals, skipping separation: {vocals_output_filename.name}"
        )
        return vocals_output_filename

    # Get Bandit-v2 paths
    bandit_repo_path = Path(
        os.environ.get("BANDIT_REPO_PATH", "repos/bandit-v2")
    ).resolve()
    inference_script = bandit_repo_path / "inference.py"

    if not inference_script.exists():
        log.error(f"Bandit-v2 inference.py not found at: {inference_script}")
        return None

    # Fix config file
    original_config_path = bandit_repo_path / "expt" / "inference.yaml"
    temp_config_path = bandit_repo_path / "expt" / "inference_temp.yaml"

    try:
        with open(original_config_path, "r", encoding="utf-8") as f:
            config_content = f.read()

        repo_path_url = bandit_repo_path.as_posix()
        config_content = config_content.replace("$REPO_ROOT", repo_path_url)
        config_content = config_content.replace(
            "data: dnr-v3-com-smad-multi-v2", "data: dnr-v3-com-smad-multi-v2b"
        )

        import re

        config_content = re.sub(
            r'file://([^"\']+)',
            lambda m: "file://" + m.group(1).replace("\\", "/"),
            config_content,
        )

        with open(temp_config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        # Check audio duration and decide on processing strategy
        try:
            import torchaudio

            info = torchaudio.info(str(input_audio_file))
            duration_seconds = info.num_frames / info.sample_rate
            duration_minutes = duration_seconds / 60
            sample_rate = info.sample_rate
            num_channels = info.num_channels

            # Determine if chunking is needed based on duration
            if duration_minutes > chunk_minutes:
                log.info(
                    f"Audio is {duration_minutes:.1f} minutes long. Processing in {chunk_minutes}-minute chunks..."
                )

                # Try progressively smaller chunks if memory issues occur
                for attempt_chunk_minutes in [
                    chunk_minutes,
                    chunk_minutes / 2,
                    chunk_minutes / 4,
                ]:
                    log.info(
                        f"Attempting with {attempt_chunk_minutes}-minute chunks..."
                    )

                    result = _process_in_chunks(
                        input_audio_file,
                        checkpoint_path,
                        output_dir,
                        vocals_output_filename,
                        bandit_repo_path,
                        temp_config_path,
                        duration_seconds,
                        sample_rate,
                        num_channels,
                        attempt_chunk_minutes * 60,
                    )

                    if result is not None:
                        return result

                    if attempt_chunk_minutes <= 1.25:  # Don't go below 1.25 minutes
                        log.error(
                            "Even very small chunks are failing. Your audio may be too complex for available GPU memory."
                        )
                        break

                    log.warning(
                        f"{attempt_chunk_minutes}-minute chunks too large, trying smaller..."
                    )

                return None
            else:
                # Single file processing
                log.info(
                    f"Audio is {duration_minutes:.1f} minutes long. Processing as single file..."
                )
                return _process_single_file(
                    input_audio_file,
                    checkpoint_path,
                    output_dir,
                    vocals_output_filename,
                    bandit_repo_path,
                    temp_config_path,
                )

        except Exception as e:
            log.warning(
                f"Could not analyze audio: {e}. Attempting direct processing..."
            )
            return _process_single_file(
                input_audio_file,
                checkpoint_path,
                output_dir,
                vocals_output_filename,
                bandit_repo_path,
                temp_config_path,
            )

    except Exception as e:
        log.error(f"[bold red]Bandit-v2 vocal separation failed: {e}[/]")
        return None
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink(missing_ok=True)


def _process_single_file(
    input_file,
    checkpoint_path,
    output_dir,
    final_output_path,
    bandit_repo_path,
    temp_config_path,
):
    """Process a single file without chunking."""

    cmd = [
        sys.executable,
        "inference.py",
        "--config-name",
        "inference_temp",
        f"ckpt_path={checkpoint_path.resolve()}",
        f"test_audio={input_file.resolve()}",
        f"output_path={output_dir.resolve()}",
        "model_variant=speech",
    ]

    env = os.environ.copy()
    env["REPO_ROOT"] = str(bandit_repo_path)
    env["HYDRA_FULL_ERROR"] = "1"
    if DEVICE.type == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()  # Clear cache before processing

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Bandit-v2 (vocals)...", total=None)
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, cwd=str(bandit_repo_path)
        )
        progress.update(task, completed=1, total=1)

    if result.returncode == 0:
        expected_output = output_dir / "speech_estimate.wav"
        if expected_output.exists():
            log.info(f"Found Bandit output at: {expected_output}")
            shutil.copy(str(expected_output), str(final_output_path))
            log.info(
                f"[green]✓ Bandit-v2 vocal separation completed. Vocals saved to: {final_output_path.name}[/]"
            )
            return final_output_path
        else:
            log.error(f"Expected Bandit output not found at: {expected_output}")
            log.info(f"Contents of output directory {output_dir}:")
            if output_dir.exists():
                for item in output_dir.iterdir():
                    log.info(f"  - {item.name}")
            else:
                log.error(f"Output directory {output_dir} does not exist")
            return None
    else:
        print("\n========== BANDIT FULL ERROR OUTPUT ==========")
        print("STDERR:")
        print(result.stderr)
        print("\nSTDOUT:")
        print(result.stdout)
        print("========== END BANDIT ERROR ==========\n")

        if "CUDA out of memory" in result.stderr:
            log.error("GPU out of memory. File may be too long or complex.")
        return None


def _process_in_chunks(
    input_file,
    checkpoint_path,
    output_dir,
    final_output_path,
    bandit_repo_path,
    temp_config_path,
    duration_seconds,
    sample_rate,
    num_channels,
    chunk_duration,
):
    """Process audio in chunks with crossfading."""

    temp_dir = output_dir / "__temp_chunks"
    temp_dir.mkdir(exist_ok=True)

    try:
        crossfade_duration = 0.5  # 0.5 seconds crossfade
        chunks_processed = []

        chunk_start = 0
        chunk_idx = 0
        failed_chunks = []

        while chunk_start < duration_seconds:
            # Calculate chunk boundaries with crossfade
            actual_start = max(
                0, chunk_start - crossfade_duration if chunk_idx > 0 else chunk_start
            )
            chunk_end = min(duration_seconds, chunk_start + chunk_duration)
            actual_end = min(
                duration_seconds,
                (
                    chunk_end + crossfade_duration
                    if chunk_end < duration_seconds
                    else chunk_end
                ),
            )

            log.info(
                f"Processing chunk {chunk_idx + 1}/{int(duration_seconds/chunk_duration)+1} ({actual_start/60:.1f}-{actual_end/60:.1f} min)"
            )

            # Extract chunk
            chunk_input = temp_dir / f"chunk_{chunk_idx:03d}_input.wav"
            ff_trim(
                input_file,
                chunk_input,
                actual_start,
                actual_end,
                target_sr=sample_rate,
                target_ac=num_channels,
            )

            # Process chunk
            chunk_output_dir = temp_dir / f"chunk_{chunk_idx:03d}_output"
            chunk_output_dir.mkdir(exist_ok=True)

            # Clear GPU cache before each chunk
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            cmd = [
                sys.executable,
                "inference.py",
                "--config-name",
                "inference_temp",
                f"ckpt_path={checkpoint_path.resolve()}",
                f"test_audio={chunk_input.resolve()}",
                f"output_path={chunk_output_dir.resolve()}",
                "model_variant=speech",
                "inference.kwargs.inference_batch_size=1",  # FIX: Override batch size to 1
            ]

            env = os.environ.copy()
            env["REPO_ROOT"] = str(bandit_repo_path)
            env["HYDRA_FULL_ERROR"] = "1"
            if DEVICE.type == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = "0"
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Chunk {chunk_idx + 1}...", total=None)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=str(bandit_repo_path),
                )
                progress.update(task, completed=1, total=1)

            if result.returncode == 0:
                expected_output = chunk_output_dir / "speech_estimate.wav"
                if expected_output.exists():
                    chunk_output = temp_dir / f"chunk_{chunk_idx:03d}_vocals.wav"
                    shutil.copy(str(expected_output), str(chunk_output))
                    chunks_processed.append(
                        {
                            "path": chunk_output,
                            "idx": chunk_idx,
                            "start": actual_start,
                            "chunk_start": chunk_start,
                            "chunk_end": chunk_end,
                            "has_pre_crossfade": chunk_idx > 0,
                            "has_post_crossfade": chunk_end < duration_seconds,
                        }
                    )
                else:
                    log.error(f"Chunk {chunk_idx + 1} output not found")
                    failed_chunks.append(chunk_idx)
            else:
                log.error(f"Chunk {chunk_idx + 1} processing failed")
                print("\n========== BANDIT ERROR OUTPUT ==========")
                print(f"Command: {' '.join(cmd)}")
                print(f"\nSTDERR:\n{result.stderr}")
                print(f"\nSTDOUT:\n{result.stdout}")
                print("========== END ERROR ==========\n")

                if "CUDA out of memory" in result.stderr:
                    log.error("CUDA out of memory detected in error")
                    # Don't continue if memory error - need smaller chunks
                    return None
                failed_chunks.append(chunk_idx)

            # Clean up input
            chunk_input.unlink(missing_ok=True)

            # Next chunk
            chunk_start = chunk_end
            chunk_idx += 1

        if not chunks_processed:
            log.error("No chunks were successfully processed")
            return None

        if failed_chunks:
            log.warning(
                f"Failed to process {len(failed_chunks)} chunks: {failed_chunks}"
            )
            log.warning("Output may have gaps where chunks failed")

        # Concatenate with crossfading
        log.info(f"Concatenating {len(chunks_processed)} chunks...")

        # Simple concatenation for single chunk
        if len(chunks_processed) == 1:
            shutil.copy(str(chunks_processed[0]["path"]), str(final_output_path))
        else:
            # Build concat list with proper ordering
            concat_list = temp_dir / "concat_list.txt"
            with open(concat_list, "w") as f:
                for chunk in sorted(chunks_processed, key=lambda x: x["idx"]):
                    f.write(f"file '{chunk['path'].resolve().as_posix()}'\n")

            # Use ffmpeg to concatenate
            (
                ffmpeg.input(str(concat_list), format="concat", safe=0)
                .output(
                    str(final_output_path),
                    acodec="pcm_s16le",
                    ar=sample_rate,
                    ac=num_channels,
                )
                .overwrite_output()
                .run(quiet=True)
            )

        log.info(
            f"[green]✓ Bandit-v2 vocal separation completed (processed in {len(chunks_processed)} chunks)[/]"
        )
        return final_output_path

    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def diarize_audio(
    input_audio_file: Path,
    tmp_dir: Path,
    huggingface_token: str,
    model_config: dict,
    dry_run: bool = False,
) -> Annotation | None:
    # PyAnnote 3.1 is the target
    model_name = model_config.get("diar_model", "pyannote/speaker-diarization-3.1")
    # Ensure it does not use 3.0, even if specified in args by mistake. Forcing 3.1.
    if "3.0" in model_name:
        log.warning(
            f"Requested diarization model '{model_name}' seems to be v3.0. Upgrading to 'pyannote/speaker-diarization-3.1'."
        )
        model_name = "pyannote/speaker-diarization-3.1"

    hyper_params = model_config.get("diar_hyperparams", {})
    log.info(
        f"Starting speaker diarization for: {input_audio_file.name} (Model: {model_name})"
    )
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    ensure_dir_exists(tmp_dir)
    if hyper_params:
        log.info(f"With diarization hyperparameters: {hyper_params}")

    try:
        pipeline = PyannotePipeline.from_pretrained(
            model_name, use_auth_token=huggingface_token
        )
        if hasattr(pipeline, "to") and callable(getattr(pipeline, "to")):
            pipeline = pipeline.to(DEVICE)
        log.info(f"Diarization model '{model_name}' loaded to {DEVICE.type.upper()}.")
    except Exception as e:
        log.error(f"[bold red]Error loading diarization model '{model_name}': {e}[/]")
        log.error(
            "Please ensure you have accepted the model's terms on Hugging Face and your token is correct."
        )
        return None  # Changed from raise to allow pipeline to potentially continue or handle

    target_audio_for_processing = input_audio_file
    if dry_run:
        cut_audio_file_path = tmp_dir / f"{input_audio_file.stem}_60s_diar_dryrun.wav"
        log.warning(
            f"[DRY-RUN] Using first 60s for diarization. Temp: {cut_audio_file_path.name}"
        )
        try:
            # Diarization models typically expect 16kHz
            ff_trim(input_audio_file, cut_audio_file_path, 0, 60, target_sr=16000)
            target_audio_for_processing = cut_audio_file_path
        except Exception as e:
            log.error(
                f"Failed to create dry-run audio for diarization: {e}. Using full audio."
            )

    log.info(
        f"Running diarization on {DEVICE.type.upper()} for {target_audio_for_processing.name}..."
    )
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Diarizing...", total=None)
            # PyAnnote pipeline expects 'audio' key to be path string
            diarization_result = pipeline(
                {
                    "uri": target_audio_for_processing.stem,
                    "audio": str(target_audio_for_processing),
                },
                **hyper_params,
            )
            progress.update(task, completed=1, total=1)
        num_speakers = len(diarization_result.labels())
        total_speech_duration = diarization_result.get_timeline().duration()
        log.info(
            f"[green]✓ Diarization complete.[/] Found {num_speakers} speaker labels. Total speech: {format_duration(total_speech_duration)}."
        )
        if num_speakers == 0:
            log.warning("Diarization resulted in zero speakers.")
        return diarization_result
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and DEVICE.type == "cuda":
            log.error("[bold red]CUDA out of memory during diarization![/]")
            torch.cuda.empty_cache()
            log.warning("Attempting diarization on CPU (slower)...")
            try:
                pipeline = pipeline.to(torch.device("cpu"))
                log.info("Switched diarization pipeline to CPU.")
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as p_cpu:
                    task_cpu = p_cpu.add_task("Diarizing (CPU)...", total=None)
                    res_cpu = pipeline(
                        {
                            "uri": target_audio_for_processing.stem,
                            "audio": str(target_audio_for_processing),
                        },
                        **hyper_params,
                    )
                    p_cpu.update(task_cpu, completed=1, total=1)
                log.info(
                    f"[green]✓ Diarization (CPU) complete.[/] Found {len(res_cpu.labels())} spk. Total speech: {format_duration(res_cpu.get_timeline().duration())}."
                )
                return res_cpu
            except Exception as cpu_e:
                log.error(
                    f"Diarization failed on GPU (OOM) and subsequently on CPU: {cpu_e}"
                )
                return None
        else:
            log.error(f"Runtime error during diarization: {e}")
            return None
    except Exception as e:
        log.error(f"Unexpected error during diarization: {e}")
        return None


def detect_overlapped_regions(
    input_audio_file: Path,
    tmp_dir: Path,
    huggingface_token: str,
    osd_model_name: str = "pyannote/overlapped-speech-detection",  # Default OSD from original code
    dry_run: bool = False,
) -> Timeline | None:
    log.info(f"Starting OSD for: {input_audio_file.name} (OSD Model: {osd_model_name})")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    ensure_dir_exists(tmp_dir)

    osd_pipeline_instance = None
    # Hyperparameters for OverlappedSpeechDetection from pyannote.audio.pipelines.segmentation.Pipeline
    # These are defaults if segmentation model is used.
    default_osd_hyperparameters = {
        "onset": 0.5,
        "offset": 0.5,
        "min_duration_on": 0.05,
        "min_duration_off": 0.05,
        # For OSD, we are interested in segments with 2 or more speakers.
        # These can be tuned.
        "segmentation_min_duration_off": 0.0,  # from pyannote.audio.pipelines.utils
    }

    try:
        # pyannote/overlapped-speech-detection is a dedicated pipeline
        if osd_model_name == "pyannote/overlapped-speech-detection":
            log.info(f"Loading dedicated OSD pipeline: '{osd_model_name}'...")
            osd_pipeline_instance = PyannotePipeline.from_pretrained(
                osd_model_name, use_auth_token=huggingface_token
            )
        # pyannote/segmentation-3.0 (or similar like voicefixer/mdx23c-segmentation) are base models
        # that can be wrapped by OverlappedSpeechDetection pipeline.
        elif (
            osd_model_name.startswith("pyannote/segmentation")
            or "segmentation" in osd_model_name
        ):
            log.info(
                f"Loading '{osd_model_name}' as base segmentation model for OSD pipeline..."
            )
            segmentation_model = PyannoteModel.from_pretrained(
                osd_model_name, use_auth_token=huggingface_token
            )
            osd_pipeline_instance = PyannoteOSDPipeline(
                segmentation=segmentation_model,
                # device=DEVICE # OSDPipeline takes device here
            )
            # OSDPipeline needs instantiation of params if not set
            osd_pipeline_instance.instantiate(default_osd_hyperparameters)
            log.info(
                f"Instantiated OverlappedSpeechDetection pipeline (from '{osd_model_name}') with parameters: {default_osd_hyperparameters}."
            )
        else:  # Fallback for other potential pipeline types, though less common for OSD
            log.warning(
                f"OSD model string '{osd_model_name}' not recognized as a specific type. "
                "Attempting to load as a generic PyannotePipeline. This may not yield overlap directly."
            )
            osd_pipeline_instance = PyannotePipeline.from_pretrained(
                osd_model_name, use_auth_token=huggingface_token
            )

        if osd_pipeline_instance is None:
            raise RuntimeError(
                f"Failed to load or instantiate OSD pipeline for '{osd_model_name}'. Instance is None."
            )

        # Move to device
        if hasattr(osd_pipeline_instance, "to") and callable(
            getattr(osd_pipeline_instance, "to")
        ):
            log.debug(
                f"Moving OSD pipeline for '{osd_model_name}' to {DEVICE.type.upper()}"
            )
            osd_pipeline_instance = osd_pipeline_instance.to(DEVICE)
        # If it's an OSDPipeline, the model is 'segmentation_model' or 'segmentation' (check pyannote version)
        elif hasattr(osd_pipeline_instance, "segmentation_model") and hasattr(
            osd_pipeline_instance.segmentation_model, "to"
        ):
            log.debug(
                f"Moving OSD pipeline's segmentation_model to {DEVICE.type.upper()}"
            )
            osd_pipeline_instance.segmentation_model = (
                osd_pipeline_instance.segmentation_model.to(DEVICE)
            )
        elif hasattr(osd_pipeline_instance, "segmentation") and hasattr(
            osd_pipeline_instance.segmentation, "to"
        ):  # segmentation is the model instance
            log.debug(
                f"Moving OSD pipeline's segmentation (model) to {DEVICE.type.upper()}"
            )
            osd_pipeline_instance.segmentation = osd_pipeline_instance.segmentation.to(
                DEVICE
            )

        log.info(
            f"OSD model/pipeline '{osd_model_name}' successfully prepared on {DEVICE.type.upper()}."
        )

    except Exception as e:
        log.error(
            f"[bold red]Fatal error loading/instantiating OSD model/pipeline '{osd_model_name}': {type(e).__name__} - {e}[/]"
        )
        # ... (error details from original code) ...
        return None  # Changed from raise

    target_audio_for_processing = input_audio_file
    if dry_run:
        cut_audio_file_path = tmp_dir / f"{input_audio_file.stem}_60s_osd_dryrun.wav"
        log.warning(
            f"[DRY-RUN] Using first 60s for OSD. Temp: {cut_audio_file_path.name}"
        )
        try:
            # OSD models also typically expect 16kHz
            ff_trim(input_audio_file, cut_audio_file_path, 0, 60, target_sr=16000)
            target_audio_for_processing = cut_audio_file_path
        except Exception as e:
            log.error(f"Failed to create dry-run audio for OSD: {e}. Using full audio.")

    log.info(
        f"Running OSD on {DEVICE.type.upper()} for {target_audio_for_processing.name}..."
    )
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Detecting overlaps...", total=None)
            # PyAnnote pipeline expects 'audio' key to be path string
            osd_annotation_or_timeline = osd_pipeline_instance(
                {
                    "uri": target_audio_for_processing.stem,
                    "audio": str(target_audio_for_processing),
                }
            )
            progress.update(task, completed=1, total=1)

        overlap_timeline = Timeline()
        # OSDPipeline directly returns a Timeline of overlapped regions.
        # Generic pipelines return an Annotation.
        if isinstance(osd_annotation_or_timeline, Timeline):
            overlap_timeline = osd_annotation_or_timeline
            log.info(
                "OSD pipeline returned a Timeline directly (expected for OverlappedSpeechDetection)."
            )
        elif isinstance(osd_annotation_or_timeline, Annotation):
            osd_annotation = osd_annotation_or_timeline
            # Logic from original code to extract overlap from Annotation
            if "overlap" in osd_annotation.labels():
                overlap_timeline.update(osd_annotation.label_timeline("overlap"))
            # ... (other label checking logic from original code if 'overlap' not present) ...
            else:  # Try to infer from segmentation model output (e.g., speaker count > 1)
                labels_from_osd = osd_annotation.labels()
                log.debug(
                    f"OSD with '{osd_model_name}' did not directly yield 'overlap' label from Annotation. Checking other labels: {labels_from_osd}"
                )
                found_overlap_in_annotation = False
                for label in labels_from_osd:
                    # For segmentation models (e.g. pyannote/segmentation-3.0), labels might be 'speakerN', 'noise', 'speech'.
                    # Or it might give speaker counts like 'SPEAKER_00+SPEAKER_01', '2speakers'.
                    # This part needs careful checking based on the actual model's output labels.
                    # A common pattern from segmentation models used in OSDPipeline is labels like 'overlap' or counting speakers.
                    if (
                        "overlap" in label.lower()
                    ):  # Check if any label contains 'overlap'
                        overlap_timeline.update(osd_annotation.label_timeline(label))
                        found_overlap_in_annotation = True
                        log.info(f"Using label '{label}' from Annotation as overlap.")
                        break
                    # Try to infer from speaker count in label (e.g. from a segmentation model that counts speakers)
                    try:  # Example: 'speaker_count_2', '2_speakers_MIX', 'INTERSECTION'
                        if (
                            re.search(r"(\d+)\s*speaker", label, re.IGNORECASE)
                            and int(
                                re.search(
                                    r"(\d+)\s*speaker", label, re.IGNORECASE
                                ).group(1)
                            )
                            >= 2
                        ):
                            overlap_timeline.update(
                                osd_annotation.label_timeline(label)
                            )
                            found_overlap_in_annotation = True
                            break
                        if (
                            "+" in label
                            or "intersection" in label.lower()
                            or "overlap" in label.lower()
                        ):  # Heuristic for multi-speaker labels
                            overlap_timeline.update(
                                osd_annotation.label_timeline(label)
                            )
                            found_overlap_in_annotation = True
                            break
                    except (ValueError, AttributeError):
                        pass
                if not found_overlap_in_annotation and labels_from_osd:
                    log.warning(
                        f"Could not determine specific overlap label from Annotation via '{osd_model_name}'. Labels: {labels_from_osd}. No overlap inferred from this Annotation."
                    )

        else:
            log.error(
                f"OSD pipeline returned an unexpected type: {type(osd_annotation_or_timeline)}. Expected Timeline or Annotation."
            )
            return Timeline()  # Return empty timeline

        overlap_timeline = (
            overlap_timeline.support()
        )  # Merge overlapping segments within the timeline
        total_overlap_duration = overlap_timeline.duration()
        log.info(
            f"[green]✓ Overlap detection complete.[/] Total overlap: {format_duration(total_overlap_duration)}."
        )
        if total_overlap_duration == 0:
            log.info(
                "No overlapped speech detected by OSD model or inferred from its output."
            )
        return overlap_timeline

    except RuntimeError as e:  # GPU OOM
        if "CUDA out of memory" in str(e) and DEVICE.type == "cuda":
            log.error("[bold red]CUDA out of memory during OSD![/]")
            torch.cuda.empty_cache()
            log.warning("Attempting OSD on CPU (slower)...")
            cpu_device = torch.device("cpu")
            try:
                osd_pipeline_cpu = None
                # Re-initialize OSD pipeline for CPU
                if osd_model_name == "pyannote/overlapped-speech-detection":
                    osd_pipeline_cpu = PyannotePipeline.from_pretrained(
                        osd_model_name, use_auth_token=huggingface_token
                    ).to(cpu_device)
                elif (
                    osd_model_name.startswith("pyannote/segmentation")
                    or "segmentation" in osd_model_name
                ):
                    segmentation_model_cpu = PyannoteModel.from_pretrained(
                        osd_model_name, use_auth_token=huggingface_token
                    ).to(cpu_device)
                    osd_pipeline_cpu = PyannoteOSDPipeline(
                        segmentation=segmentation_model_cpu
                    )
                    osd_pipeline_cpu.instantiate(default_osd_hyperparameters)
                else:  # Generic
                    osd_pipeline_cpu = PyannotePipeline.from_pretrained(
                        osd_model_name, use_auth_token=huggingface_token
                    ).to(cpu_device)

                if osd_pipeline_cpu is None:
                    raise RuntimeError(
                        "Failed to create OSD pipeline for CPU fallback."
                    )
                log.info("Switched OSD pipeline to CPU.")
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as p_cpu:
                    task_cpu = p_cpu.add_task("Detecting overlaps (CPU)...", total=None)
                    osd_res_cpu = osd_pipeline_cpu(
                        {
                            "uri": target_audio_for_processing.stem,
                            "audio": str(target_audio_for_processing),
                        }
                    )
                    p_cpu.update(task_cpu, completed=1, total=1)

                ov_tl_cpu = Timeline()
                if isinstance(osd_res_cpu, Timeline):
                    ov_tl_cpu = osd_res_cpu
                elif isinstance(osd_res_cpu, Annotation):
                    # Extract from annotation as in GPU block
                    if "overlap" in osd_res_cpu.labels():
                        ov_tl_cpu.update(osd_res_cpu.label_timeline("overlap"))
                    # ... (other label checking) ...
                ov_tl_cpu = ov_tl_cpu.support()
                log.info(
                    f"[green]✓ OSD (CPU) complete.[/] Total overlap: {format_duration(ov_tl_cpu.duration())}."
                )
                return ov_tl_cpu

            except Exception as cpu_e:
                log.error(f"OSD failed on GPU (OOM) and subsequently on CPU: {cpu_e}")
                return Timeline()  # Return empty on error
        else:  # Other runtime errors
            log.error(f"Runtime error during OSD: {e}")
            return Timeline()
    except Exception as e:
        log.error(f"An unexpected error occurred during OSD processing: {e}")
        return Timeline()


def identify_target_speaker(
    annotation: Annotation,
    input_audio_file: Path,  # Audio file from which segments are derived (e.g., bandit output)
    processed_reference_file: Path,  # Reference audio (16kHz mono)
    target_name: str,
    wespeaker_rvector_model,  # WeSpeaker Deep r-vector model instance
) -> str | None:
    log.info(
        f"Identifying '{target_name}' among diarized speakers using WeSpeaker Deep r-vector and reference: {processed_reference_file.name}"
    )

    if wespeaker_rvector_model is None:
        log.error(
            "WeSpeaker r-vector model not available for speaker identification. Cannot proceed."
        )
        return None
    if not processed_reference_file.exists():
        log.error(
            f"Processed reference audio not found: {processed_reference_file}. Cannot ID target."
        )
        return None

    try:
        ref_embedding = wespeaker_rvector_model.extract_embedding(
            str(processed_reference_file)
        )
        log.debug(
            f"Reference embedding for '{target_name}' extracted, shape: {ref_embedding.shape}"
        )
    except Exception as e:
        log.error(
            f"Failed to extract embedding from reference audio '{processed_reference_file.name}' using WeSpeaker: {e}"
        )
        return None

    # Create a temporary directory for speaker segment audio files
    # This is because WeSpeaker model.extract_embedding expects file paths
    with tempfile.TemporaryDirectory(
        prefix="speaker_id_segs_", dir=Path(processed_reference_file).parent
    ) as temp_seg_dir_str:
        temp_seg_dir = Path(temp_seg_dir_str)

        speaker_similarities = {}
        unique_speaker_labels = annotation.labels()
        if not unique_speaker_labels:
            log.error(
                "Diarization produced no speaker labels. Cannot identify target speaker."
            )
            return None

        log.info(
            f"Comparing reference of '{target_name}' with {len(unique_speaker_labels)} diarized speakers using WeSpeaker r-vector."
        )

        # We need to extract audio segments for each speaker.
        # The input_audio_file is the source (e.g., bandit output or original).
        # Segments from diarization are relative to this input_audio_file.
        # WeSpeaker expects 16kHz for its pre-trained models. Ensure segments are 16kHz.
        # The diarization itself should have run on 16kHz audio, so segment times are for that.
        # Bandit output SR might be different, so resampling of segments might be needed if input_audio_file is bandit output.
        # For simplicity, assume input_audio_file is already at a common SR or ff_slice handles it.
        # It's safer to always resample segments to 16kHz for WeSpeaker.

        for spk_label in unique_speaker_labels:
            speaker_segments_timeline = annotation.label_timeline(spk_label)
            if not speaker_segments_timeline:
                log.debug(
                    f"Speaker label '{spk_label}' has no speech segments. Skipping."
                )
                continue            # Concatenate first N seconds of speech for this speaker to create a representative sample
            MAX_EMBED_DURATION_PER_SPEAKER = 20.0  # seconds
            current_duration_for_embedding = 0.0

            temp_speaker_audio_list = []

            for i, seg in enumerate(speaker_segments_timeline):
                if current_duration_for_embedding >= MAX_EMBED_DURATION_PER_SPEAKER:
                    break                # Slice segment from input_audio_file and resample to 16kHz for WeSpeaker
                temp_seg_path = temp_seg_dir / f"{safe_filename(spk_label)}_seg_{i}.wav"
                try:
                    # ff_slice_smart will handle intelligent chunking if needed
                    ff_slice_smart(
                        input_audio_file,
                        temp_seg_path,
                        seg.start,
                        seg.end,
                        target_sr=16000,
                        target_ac=1,
                    )
                    if temp_seg_path.exists() and temp_seg_path.stat().st_size > 0:
                        temp_speaker_audio_list.append(temp_seg_path)
                        current_duration_for_embedding += (
                            seg.duration
                        )  # Using original segment duration for tracking
                    else:
                        log.warning(
                            f"Failed to create/empty slice for speaker ID: {temp_seg_path.name}"
                        )
                except Exception as e_slice:
                    log.warning(
                        f"Slicing segment {i} for speaker {spk_label} failed: {e_slice}"
                    )

            if not temp_speaker_audio_list:
                log.debug(
                    f"No valid audio segments extracted for speaker '{spk_label}' for embedding. Similarity set to 0."
                )
                speaker_similarities[spk_label] = 0.0
                continue

            # Create a single audio file for this speaker by concatenating the temp segments
            speaker_concat_audio_path = (
                temp_seg_dir / f"{safe_filename(spk_label)}_concat_for_embed.wav"
            )
            if (
                len(temp_speaker_audio_list) == 1
            ):  # If only one segment, just use it (rename for consistency)
                shutil.copy(temp_speaker_audio_list[0], speaker_concat_audio_path)
            else:
                concat_list_file = (
                    temp_seg_dir / f"{safe_filename(spk_label)}_concat_list.txt"
                )
                with open(concat_list_file, "w") as f:
                    for p in temp_speaker_audio_list:
                        f.write(f"file '{p.resolve().as_posix()}'\n")
                try:
                    (
                        ffmpeg.input(str(concat_list_file), format="concat", safe=0)
                        .output(
                            str(speaker_concat_audio_path),
                            acodec="pcm_s16le",
                            ar=16000,
                            ac=1,
                        )
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as e_concat:
                    log.warning(
                        f"ffmpeg concat failed for speaker {spk_label} embedding audio: {e_concat.stderr.decode() if e_concat.stderr else 'ffmpeg error'}. Similarity set to 0."
                    )
                    speaker_similarities[spk_label] = 0.0
                    continue

            if (
                speaker_concat_audio_path.exists()
                and speaker_concat_audio_path.stat().st_size > 0
            ):
                try:
                    spk_embedding = wespeaker_rvector_model.extract_embedding(
                        str(speaker_concat_audio_path)
                    )
                    similarity = cos(
                        ref_embedding, spk_embedding
                    )  # Using common.cos for numpy arrays
                    speaker_similarities[spk_label] = similarity
                except Exception as e_embed:
                    log.warning(
                        f"Error extracting WeSpeaker embedding for speaker '{spk_label}': {e_embed}. Similarity set to 0."
                    )
                    speaker_similarities[spk_label] = 0.0
            else:
                log.debug(
                    f"Concatenated audio for speaker '{spk_label}' embedding is missing or empty. Similarity set to 0."
                )
                speaker_similarities[spk_label] = 0.0

    if not speaker_similarities:
        log.error(
            f"Speaker similarity calculation failed for all speakers for '{target_name}'."
        )
        return None

    if all(score == 0.0 for score in speaker_similarities.values()):
        log.error(
            f"[bold red]All WeSpeaker similarity scores are zero for '{target_name}'. Cannot reliably ID target.[/]"
        )
        # Fallback: pick the first speaker label or a placeholder if desired. For now, indicate failure.
        best_match_label = (
            unique_speaker_labels[0] if unique_speaker_labels else "UNKNOWN_SPEAKER"
        )
        max_similarity_score = 0.0
        log.warning(
            f"Arbitrarily assigning '{best_match_label}' due to all zero scores (this is a guess)."
        )
    else:
        best_match_label = max(speaker_similarities, key=speaker_similarities.get)
        max_similarity_score = speaker_similarities[best_match_label]

    log.info(
        f"[green]✓ Identified '{target_name}' as diarization label → [bold]{best_match_label}[/] (WeSpeaker r-vector sim: {max_similarity_score:.4f})[/]"
    )

    sim_table = Table(
        title=f"WeSpeaker r-vector Similarities to '{target_name}' Reference",
        show_lines=True,
        highlight=True,
    )
    sim_table.add_column("Diarized Speaker Label", style="cyan", justify="center")
    sim_table.add_column("Similarity Score", style="magenta", justify="center")
    for spk, score in sorted(
        speaker_similarities.items(), key=lambda item: item[1], reverse=True
    ):
        sim_table.add_row(
            spk,
            f"{score:.4f}",
            style="bold yellow on bright_black" if spk == best_match_label else "",
        )
    console.print(sim_table)

    return best_match_label


def merge_nearby_segments(
    segments_to_merge: list[Segment], max_allowed_gap: float = DEFAULT_MAX_MERGE_GAP
) -> list[Segment]:
    if not segments_to_merge:
        return []
    # Sort segments by start time
    sorted_segments = sorted(list(segments_to_merge), key=lambda s: s.start)
    if not sorted_segments:
        return []  # Should not happen if segments_to_merge was not empty

    merged_timeline = Timeline()
    if not sorted_segments:
        return []

    current_merged_segment = sorted_segments[0]
    for next_segment in sorted_segments[1:]:
        # If next_segment starts within max_allowed_gap of current_merged_segment's end
        if (next_segment.start <= current_merged_segment.end + max_allowed_gap) and (
            next_segment.end > current_merged_segment.end
        ):  # And it extends the current segment
            current_merged_segment = Segment(
                current_merged_segment.start, next_segment.end
            )
        elif (
            next_segment.start > current_merged_segment.end + max_allowed_gap
        ):  # Gap is too large
            merged_timeline.add(current_merged_segment)
            current_merged_segment = next_segment
        # If next_segment is completely within current_merged_segment or starts before but ends earlier, it's usually handled by Timeline.support() or prior logic.
        # This simple merge focuses on extending or starting new.

    merged_timeline.add(current_merged_segment)  # Add the last merged segment
    return list(
        merged_timeline.support()
    )  # .support() merges overlapping segments within the timeline


def filter_segments_by_duration(
    segments_to_filter: list[Segment], min_req_duration: float = DEFAULT_MIN_SEGMENT_SEC
) -> list[Segment]:
    return [seg for seg in segments_to_filter if seg.duration >= min_req_duration]


def check_voice_activity(
    audio_path: Path, min_speech_ratio: float = 0.6, vad_threshold: float = 0.5
) -> bool:
    """Checks voice activity in an audio file using Silero-VAD."""
    try:
        y, sr = librosa.load(
            audio_path, sr=16000, mono=True
        )  # Silero VAD expects 16kHz
    except Exception as e:
        log.debug(
            f"VAD: Librosa load failed for {audio_path.name}: {e}. Assuming no voice activity."
        )
        return False
    if len(y) == 0:
        log.debug(
            f"VAD: Audio file {audio_path.name} is empty. Assuming no voice activity."
        )
        return False
    try:
        # Silero VAD model loading (cached by torch.hub)
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
            onnx=False,
        )
        (get_speech_timestamps, _, read_audio, _, _) = (
            utils  # read_audio is not used here as we load with librosa
        )
        vad_model.to(DEVICE)  # Move model to appropriate device
    except Exception as e:
        log.warning(
            f"VAD: Silero-VAD model loading failed: {e}. Skipping VAD for {audio_path.name}, assuming active speech."
        )
        return True

    try:
        audio_tensor = torch.FloatTensor(y).to(DEVICE)
        # Silero VAD model expects sample rates 16000, 8000 or 48000Hz. We loaded at 16000Hz.
        speech_timestamps = get_speech_timestamps(
            audio_tensor, vad_model, sampling_rate=16000, threshold=vad_threshold
        )

        speech_duration_samples = sum(d["end"] - d["start"] for d in speech_timestamps)
        speech_duration_sec = speech_duration_samples / 16000
        total_duration_sec = len(y) / 16000

        ratio = (
            speech_duration_sec / total_duration_sec if total_duration_sec > 0 else 0.0
        )
        log.debug(
            f"VAD for {audio_path.name}: Speech Ratio {ratio:.2f} (Speech: {speech_duration_sec:.2f}s / Total: {total_duration_sec:.2f}s)"
        )
        return ratio >= min_speech_ratio
    except Exception as e:
        log.warning(
            f"VAD: Error processing {audio_path.name} with Silero-VAD: {e}. Assuming active speech."
        )
        return True


def verify_speaker_segment(
    segment_audio_path: Path,  # Path to the segment to verify (must be 16kHz mono for models)
    reference_audio_path: Path,  # Path to the reference audio (must be 16kHz mono)
    wespeaker_models: dict,  # Dict containing 'rvector' and 'gemini' WeSpeaker model instances
    speechbrain_sb_model,  # SpeechBrain ECAPA-TDNN model instance
    verification_strategy: str = "weighted_average",  # or "sequential_gauntlet" (not fully implemented)
) -> tuple[float, dict]:
    """
    Performs multi-stage speaker verification on an audio segment.
    Ensures input paths (segment_audio_path, reference_audio_path) are 16kHz mono.
    """
    scores = {
        "wespeaker_rvector": 0.0,
        "speechbrain_ecapa": 0.0,
        "wespeaker_gemini": 0.0,
        "voice_activity_factor": 0.1,  # Default to low if VAD fails or no activity
    }
    seg_name = segment_audio_path.name

    # Ensure reference and segment audio are suitable for models (16kHz, mono)
    # This function assumes they are already prepared. If not, they should be converted before calling.

    # --- Stage 1: WeSpeaker Deep r-vector ---
    if wespeaker_models and wespeaker_models.get("rvector"):
        try:
            ws_rvector_model = wespeaker_models["rvector"]
            # WeSpeaker expects file paths.
            ref_emb = ws_rvector_model.extract_embedding(str(reference_audio_path))
            seg_emb = ws_rvector_model.extract_embedding(str(segment_audio_path))
            scores["wespeaker_rvector"] = cos(ref_emb, seg_emb)
            log.debug(
                f"WeSpeaker r-vector score for {seg_name}: {scores['wespeaker_rvector']:.4f}"
            )
        except Exception as e:
            log.warning(f"WeSpeaker r-vector verification failed for {seg_name}: {e}")

    # --- Stage 2: SpeechBrain ECAPA-TDNN ---
    if (
        speechbrain_sb_model and HAVE_SPEECHBRAIN
    ):  # HAVE_SPEECHBRAIN check is redundant if model is passed
        try:
            # SpeechBrain's verify_files loads audio and handles internal resampling if needed.
            # Assumes reference_audio_path and segment_audio_path are valid paths.
            ref_path_str = str(reference_audio_path.resolve()).replace("\\", "/")
            seg_path_str = str(segment_audio_path.resolve()).replace("\\", "/")
            score_tensor, _ = speechbrain_sb_model.verify_files(
                ref_path_str, seg_path_str
            )
            scores["speechbrain_ecapa"] = score_tensor.item()
            log.debug(
                f"SpeechBrain ECAPA-TDNN score for {seg_name}: {scores['speechbrain_ecapa']:.4f}"
            )
        except Exception as e:
            log.warning(
                f"SpeechBrain ECAPA-TDNN verification failed for {seg_name}: {e}"
            )

    # --- Stage 3: WeSpeaker Golden Gemini DF-ResNet ---
    if wespeaker_models and wespeaker_models.get("gemini"):
        try:
            ws_gemini_model = wespeaker_models["gemini"]
            ref_emb_gemini = ws_gemini_model.extract_embedding(
                str(reference_audio_path)
            )
            seg_emb_gemini = ws_gemini_model.extract_embedding(str(segment_audio_path))
            scores["wespeaker_gemini"] = cos(ref_emb_gemini, seg_emb_gemini)
            log.debug(
                f"WeSpeaker Gemini score for {seg_name}: {scores['wespeaker_gemini']:.4f}"
            )
        except Exception as e:
            log.warning(f"WeSpeaker Gemini verification failed for {seg_name}: {e}")

    # --- Voice Activity Check ---
    # VAD runs on segment_audio_path, expects 16kHz mono (librosa handles loading)
    scores["voice_activity_factor"] = (
        1.0 if check_voice_activity(segment_audio_path) else 0.1
    )  # Multiplier

    # --- Combine Scores ---
    # Default: Weighted average. Weights can be tuned.
    # Example weights: r-vector (0.4), ECAPA (0.3), Gemini (0.3)
    # This is a simple combination; more sophisticated fusion could be used.
    # For sequential gauntlet: would involve if score1 > T1 and score2 > T2 ...

    final_score = 0.0
    if verification_strategy == "weighted_average":
        w_rvec = 0.4
        w_ecapa = 0.3
        w_gemini = 0.3
        avg_score = (
            scores["wespeaker_rvector"] * w_rvec
            + scores["speechbrain_ecapa"] * w_ecapa
            + scores["wespeaker_gemini"] * w_gemini
        )
        final_score = avg_score * scores["voice_activity_factor"]
    # Add other strategies if needed
    else:  # Fallback to simple average if strategy not recognized
        valid_scores = [
            s for k, s in scores.items() if k != "voice_activity_factor" and s > 0.0
        ]  # Use only successfully computed scores
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            final_score = avg_score * scores["voice_activity_factor"]
        else:  # No verification model scores available
            final_score = 0.0  # Effectively reject

    log.debug(
        f"Final combined score for {seg_name}: {final_score:.4f}, Details: {scores}"
    )
    return final_score, scores





def slice_classify_clean_and_verify_target_solo_segments(
    diarization_result: Annotation,
    target_speaker_label: str,
    original_audio_file: Path,
    output_dir_target_speaker: Path,
    tmp_dir_segments: Path,
    reference_audio_path_processed: Path,    wespeaker_models: dict,
    speechbrain_sb_model,
    whisper_model_name: str,
    target_name: str,
    min_segment_duration: float,
    max_merge_gap: float,
    verification_threshold: float,
    vad_verification: bool,
    transcribe_verified_segments: bool,
    classify_and_clean: bool,
    noise_classifier_model_id: str | None,
    bandit_model_checkpoint: Path | None,
    bandit_vocals_file: Path | None,
    noise_classification_confidence_threshold: float = 0.3,    skip_verification_if_cleaned: bool = False,
    whisper_model_instance = None,
    language: str = "en",
    max_segment_duration: float = 30.0,
) -> tuple[list[Path], list[dict]]:
    """
    Slices segments for the target speaker, optionally classifies/cleans them,
    verifies them, and optionally transcribes them.
    Manages VRAM by loading/unloading noise classifier when classify_and_clean is active.
    
    Args:
        ...existing args...
        whisper_model_instance: Pre-loaded Whisper model instance for efficient transcription
        language: Language code for transcription    """
    log.info(
        f"Processing segments for target speaker: '{target_name}' (Label: {target_speaker_label})"
    )
    ensure_dir_exists(tmp_dir_segments)
    ensure_dir_exists(output_dir_target_speaker)

    verified_segment_audio_files = []
    all_segment_details = []

    if classify_and_clean:
        audio_source_for_slicing = original_audio_file
        log.info(
            f"Classify & Clean mode: Segments will be sliced from original audio: {original_audio_file.name}"
        )
        if not bandit_model_checkpoint:
            log.warning(
                "Classify & Clean mode is active, but no Bandit model checkpoint provided. Noisy segments cannot be cleaned."
            )
    else:
        if bandit_vocals_file and bandit_vocals_file.exists():
            audio_source_for_slicing = bandit_vocals_file
            log.info(
                f"Standard mode: Segments will be sliced from Bandit-separated vocals: {bandit_vocals_file.name}"
            )
        else:
            audio_source_for_slicing = original_audio_file
            log.info(
                f"Standard mode: No Bandit vocals file. Segments will be sliced from original audio: {original_audio_file.name}"
            )

    target_speaker_segments = list(
        diarization_result.label_timeline(target_speaker_label)
    )
    if not target_speaker_segments:
        log.warning(
            f"No diarized segments found for target speaker '{target_name}' ({target_speaker_label})."
        )
        return [], []

    merged_segments = merge_nearby_segments(target_speaker_segments, max_merge_gap)
    final_segments_to_process = filter_segments_by_duration(
        merged_segments, min_segment_duration
    )
    log.info(
        f"Found {len(target_speaker_segments)} raw segments, merged to {len(merged_segments)}, filtered to {len(final_segments_to_process)} segments for '{target_name}'."
    )

    if not final_segments_to_process:
        log.warning(
            f"No segments remain for '{target_name}' after merging and duration filtering."
        )
        return [], []

    noise_classifier_instance = None
    if classify_and_clean and noise_classifier_model_id and HAVE_TRANSFORMERS:
        noise_classifier_instance = NoiseClassifier(
            model_id=noise_classifier_model_id, device_to_use=DEVICE
        )
    elif classify_and_clean and not HAVE_TRANSFORMERS:
        log.warning(
            "Classify & Clean mode is active, but Transformers library is not available. Noise classification will be skipped."
        )
    elif classify_and_clean and not noise_classifier_model_id:
        log.warning(
            "Classify & Clean mode is active, but no noise_classifier_model_id provided. Noise classification will be skipped."
        )

    for segment_idx, segment in enumerate(final_segments_to_process):
        segment_start_time = segment.start
        segment_end_time = segment.end
        segment_duration = segment.duration
        
        segment_base_name = f"{safe_filename(target_name)}_seg{segment_idx:04d}_{segment_start_time:.2f}s_{segment_end_time:.2f}s"
        log.info(
            f"Processing segment {segment_idx + 1}/{len(final_segments_to_process)} for '{target_name}': {segment_base_name} ({segment_duration:.2f}s long)"
        )

        if classify_and_clean and noise_classifier_instance:
            # Get chunks for classification
            temp_segment_base_path = tmp_dir_segments / f"{segment_base_name}_for_classify_16k_mono.wav"
            
            classification_chunks = ff_slice_smart(
                original_audio_file,
                temp_segment_base_path,
                segment_start_time,
                segment_end_time,
                target_sr=16000,
                target_ac=1,
                max_segment_duration=max_segment_duration,
                return_chunks=True
            )
            
            # Handle both single file and multiple chunks
            if isinstance(classification_chunks, list):
                chunk_paths = classification_chunks
            else:
                chunk_paths = [classification_chunks]
                
            log.info(f"Processing {len(chunk_paths)} chunks for segment {segment_idx+1}")
            
            # Process each chunk separately
            chunk_results = []
            for chunk_idx, chunk_path in enumerate(chunk_paths):
                if not chunk_path or not chunk_path.exists() or chunk_path.stat().st_size == 0:
                    if chunk_path:
                        log.warning(f"Chunk {chunk_idx+1} is missing or empty: {chunk_path.name}")
                    else:
                        log.warning(f"Chunk {chunk_idx+1} is None or failed to create")
                    continue
                    
                chunk_classification = noise_classifier_instance.classify(
                    str(chunk_path),
                    confidence_threshold=noise_classification_confidence_threshold,
                )
                
                log.info(f"Chunk {chunk_idx+1}/{len(chunk_paths)} classified as: {chunk_classification.upper()}")
                  # Process this chunk based on classification
                chunk_audio_path_for_verification = None
                chunk_cleaned_by_bandit = False
                
                if chunk_classification == "noisy" and bandit_model_checkpoint:
                    log.info(f"Chunk {chunk_idx+1} is NOISY. Attempting Bandit-v2 cleaning.")
                    
                    bandit_output_dir_for_chunk = tmp_dir_segments / f"bandit_cleaned_{segment_idx}_chunk{chunk_idx+1}"
                    ensure_dir_exists(bandit_output_dir_for_chunk)
                    
                    cleaned_chunk_path = run_bandit_vocal_separation(
                        input_audio_file=chunk_path,
                        bandit_separator=bandit_model_checkpoint,
                        output_dir=bandit_output_dir_for_chunk,
                    )
                    
                    if cleaned_chunk_path and cleaned_chunk_path.exists():
                        log.info(f"Bandit-v2 cleaning successful for chunk {chunk_idx+1}")
                        chunk_audio_path_for_verification = tmp_dir_segments / f"{segment_base_name}_chunk{chunk_idx+1:02d}_cleaned_16k_mono.wav"
                        # Copy cleaned chunk to verification path with proper format
                        shutil.copy(str(cleaned_chunk_path), str(chunk_audio_path_for_verification))
                        chunk_cleaned_by_bandit = True
                        
                        # Clean up the bandit output directory to save space
                        if bandit_output_dir_for_chunk.exists():
                            shutil.rmtree(bandit_output_dir_for_chunk, ignore_errors=True)
                    else:
                        log.warning(f"Bandit-v2 cleaning failed for chunk {chunk_idx+1}. Using original.")
                        chunk_audio_path_for_verification = chunk_path
                        
                elif chunk_classification == "noisy" and not bandit_model_checkpoint:
                    log.warning(f"Chunk {chunk_idx+1} is NOISY, but no Bandit model provided.")
                    chunk_audio_path_for_verification = chunk_path
                else:  # Clean or Unknown
                    log.info(f"Chunk {chunk_idx+1} is CLEAN or classification UNKNOWN.")
                    chunk_audio_path_for_verification = chunk_path
                
                chunk_results.append({
                    'chunk_idx': chunk_idx,
                    'chunk_path': chunk_path,
                    'verification_path': chunk_audio_path_for_verification,
                    'classification': chunk_classification,
                    'cleaned_by_bandit': chunk_cleaned_by_bandit                })
            # We'll process verification for each chunk later in the verification section

        else:  # Not classify_and_clean, or no classifier instance
            segment_base_path = tmp_dir_segments / f"{segment_base_name}_std_16k_mono.wav"
            
            # Use ff_slice_smart to get either a single file or multiple chunks
            slice_result = ff_slice_smart(
                audio_source_for_slicing,
                segment_base_path,
                segment_start_time,
                segment_end_time,
                target_sr=16000,
                target_ac=1,
                max_segment_duration=max_segment_duration,
                return_chunks=True
            )
            
            # Handle both single file and multiple chunks  
            if isinstance(slice_result, list):
                chunk_paths = slice_result
            else:
                chunk_paths = [slice_result]
                
            log.info(f"Processing {len(chunk_paths)} chunks for segment {segment_idx+1}")
            
            # For standard flow, create chunk results without classification
            chunk_results = []
            for chunk_idx, chunk_path in enumerate(chunk_paths):
                if chunk_path.exists() and chunk_path.stat().st_size > 0:
                    chunk_results.append({
                        'chunk_idx': chunk_idx,
                        'chunk_path': chunk_path,
                        'verification_path': chunk_path,
                        'classification': 'N/A (standard_flow)',
                        'cleaned_by_bandit': False
                    })
                else:
                    if chunk_path:
                        log.warning(f"Chunk {chunk_idx+1} is missing or empty: {chunk_path.name}")
                    else:
                        log.warning(f"Chunk {chunk_idx+1} is None or failed to create")
            
        # --- Verification (Process each chunk separately) ---
        # Now we have chunk_results list with chunks to process
        verified_chunks = []
        
        for chunk_result in chunk_results:
            chunk_idx = chunk_result['chunk_idx']
            chunk_path = chunk_result['verification_path']
            chunk_classification = chunk_result['classification']
            chunk_cleaned_by_bandit = chunk_result['cleaned_by_bandit']
            
            if not chunk_path or not chunk_path.exists() or chunk_path.stat().st_size == 0:
                log.warning(f"Chunk {chunk_idx+1} audio for verification is missing or empty. Skipping.")
                continue
            
            # VAD check for this chunk
            if vad_verification:
                is_active_speech = check_voice_activity(chunk_path)
                if not is_active_speech:
                    log.info(f"Chunk {chunk_idx+1} failed VAD check (low voice activity). Skipping verification.")
                    continue
            
            # Verify this individual chunk
            log.info(f"Verifying chunk {chunk_idx+1}: {chunk_path.name} against reference")
            
            final_score, all_scores = verify_speaker_segment(
                segment_audio_path=chunk_path,
                reference_audio_path=reference_audio_path_processed,
                wespeaker_models=wespeaker_models,
                speechbrain_sb_model=speechbrain_sb_model,
            )
            
            is_verified_chunk = final_score >= verification_threshold
            log.info(f"Chunk {chunk_idx+1} verification score: {final_score:.4f} (Threshold: {verification_threshold}) -> Verified: {is_verified_chunk}")
            
            if is_verified_chunk:                # Create verified chunk filename
                final_verified_chunk_filename = f"{segment_base_name}_chunk{chunk_idx+1:02d}_verified_score{final_score:.2f}.wav"
                final_verified_chunk_path = output_dir_target_speaker / final_verified_chunk_filename
                
                try:
                    shutil.copy(str(chunk_path), str(final_verified_chunk_path))
                    log.info(f"Saved verified chunk to: {final_verified_chunk_path.name}")
                    verified_chunks.append(final_verified_chunk_path)
                    
                    # Note: Transcription will be handled in Stage 7 (batch processing)
                    
                except Exception as e_copy:
                    log.error(f"Error copying verified chunk {chunk_idx+1}: {e_copy}")
            
            # Record details for this chunk
            chunk_details = {
                "index": f"{segment_idx}.{chunk_idx}",
                "start": segment_start_time,
                "end": segment_end_time,
                "duration": segment_duration,
                "chunk_idx": chunk_idx,
                "classification": chunk_classification,
                "cleaned_by_bandit": chunk_cleaned_by_bandit,
                "verified": is_verified_chunk,
                "verification_score": final_score,
                "reason": "Verified" if is_verified_chunk else "Failed verification score",
                "transcript": None,
                "output_file_path": str(final_verified_chunk_path) if is_verified_chunk else None,
            }
            all_segment_details.append(chunk_details)
          # Add verified chunks to the main list
        verified_segment_audio_files.extend(verified_chunks)
        
        log.info(f"Processed {len(chunk_results)} chunks for segment {segment_idx+1}. {len(verified_chunks)} chunks verified.")

        # Skip the old single-segment verification logic since we processed chunks
        continue    # After loop, if noise classifier was used and potentially still loaded, unload it.
    if noise_classifier_instance:
        noise_classifier_instance.unload_model()

    log.info(
        f"Finished processing {len(final_segments_to_process)} segments for '{target_name}'. Found {len(verified_segment_audio_files)} verified segments."
    )
    return verified_segment_audio_files, all_segment_details


def transcribe_audio(audio_path: Path, model_name: str, tmp_dir: Path, 
                    whisper_model_instance=None, language: str = "en") -> str | None:
    """
    Transcribes a given audio file using Whisper ASR model.
    Saves the transcript to a text file and returns the transcript.
    
    Args:
        audio_path: Path to audio file to transcribe
        model_name: Name of Whisper model (used only if whisper_model_instance is None)
        tmp_dir: Directory to save transcript file
        whisper_model_instance: Pre-loaded Whisper model instance (preferred for performance)
        language: Language for transcription
    """
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        log.warning(
            f"Audio file for transcription not found or empty: {audio_path.name}"
        )
        return None

    log.info(
        f"Transcribing audio: {audio_path.name} using Whisper model: {model_name}..."
    )
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Ensure temporary directory for transcription files
    ensure_dir_exists(tmp_dir)

    # Use pre-loaded model if available, otherwise load it
    model = whisper_model_instance
    if model is None:
        try:
            log.info(f"Loading Whisper model '{model_name}'...")
            model = whisper.load_model(model_name, device=DEVICE)
            log.info(f"Whisper model '{model_name}' loaded.")
        except Exception as e:
            log.error(f"Failed to load Whisper model '{model_name}': {e}")
            return None
    else:
        log.debug(f"Using pre-loaded Whisper model for {audio_path.name}")

    # Transcription result
    transcript_text = ""

    # Perform transcription
    try:
        transcribe_kwargs = {"fp16": DEVICE.type == "cuda"}
        if language and language != "auto":
            transcribe_kwargs["language"] = language
        
        result = model.transcribe(str(audio_path), **transcribe_kwargs)
        transcript_text = result["text"].strip() if "text" in result else ""
        log.info(f"Transcription successful for {audio_path.name}.")
    except Exception as e_transcribe:
        log.error(f"Error transcribing {audio_path.name}: {e_transcribe}")
        transcript_text = "[Transcription Error]"

    # Save transcript to a text file
    try:
        # Output TXT file path
        txt_file_path = tmp_dir / f"{safe_filename(audio_path.stem)}_transcript.txt"
        txt_file_path.write_text(transcript_text, encoding="utf-8")
        log.info(f"Transcript saved to: {txt_file_path.name}")
    except Exception as e_save:
        log.error(f"Failed to save transcript for {audio_path.name}: {e_save}")

    return transcript_text


def transcribe_segments(
    segment_paths: list[Path],
    output_dir: Path,
    target_name: str,
    segment_label: str,
    whisper_model_name: str,
    language: str = "en",
    whisper_model_instance = None,
) -> dict[Path, str]:
    """
    Transcribes multiple audio segments using the Whisper ASR model.
    Returns a dictionary mapping segment paths to their transcripts.
    
    Args:
        segment_paths: List of audio file paths to transcribe
        output_dir: Directory to save transcript files
        target_name: Name of target speaker
        segment_label: Label for this batch of segments
        whisper_model_name: Name of Whisper model (used only if whisper_model_instance is None)
        language: Language for transcription
        whisper_model_instance: Pre-loaded Whisper model instance (preferred for performance)
    """
    ensure_dir_exists(output_dir)
    
    log.info(f"Transcribing {len(segment_paths)} {segment_label} segments of '{target_name}' with Whisper...")
    if not segment_paths:
        log.warning(f"No {segment_label} segments to transcribe.")
        return {}

    # Load model once if not provided
    model_to_use = whisper_model_instance
    if model_to_use is None and segment_paths:
        try:
            log.info(f"Loading Whisper model '{whisper_model_name}' for batch transcription...")
            import whisper
            model_to_use = whisper.load_model(whisper_model_name, device=DEVICE)
            log.info(f"Whisper model '{whisper_model_name}' loaded for batch processing.")
        except Exception as e:
            log.error(f"Failed to load Whisper model '{whisper_model_name}': {e}")
            return {}
    elif model_to_use is not None:
        log.info(f"Using pre-loaded Whisper model for batch transcription of {len(segment_paths)} segments.")

    # Use batch transcription for better performance
    log.info(f"Batch transcribing {len(segment_paths)} segments with Whisper...")
    
    transcripts = {}
    
    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb:
        task = pb.add_task("Transcribing segments...", total=len(segment_paths))
        
        for segment_path in segment_paths:
            if not segment_path.exists():
                log.warning(f"Segment file not found: {segment_path}")
                pb.update(task, advance=1)
                continue
                
            try:
                # Use the loaded model for transcription
                result = model_to_use.transcribe(
                    str(segment_path),
                    language=language,
                    verbose=False,
                    word_timestamps=False                )
                
                transcript_text = result.get("text", "").strip()
                if transcript_text:
                    transcripts[segment_path.name] = transcript_text
                    log.debug(f"Transcribed '{segment_path.name}': {transcript_text[:100]}...")
                    
                    # Save transcript to file
                    transcript_filename = f"{segment_path.stem}_transcript.txt"
                    transcript_file_path = output_dir / transcript_filename
                    try:
                        with open(transcript_file_path, "w", encoding="utf-8") as f:
                            f.write(transcript_text)
                        log.debug(f"Saved transcript to: {transcript_filename}")
                    except Exception as save_e:
                        log.error(f"Failed to save transcript for {segment_path.name}: {save_e}")
                else:
                    log.warning(f"Empty transcription for: {segment_path.name}")
                    
            except Exception as e:
                log.error(f"Failed to transcribe {segment_path.name}: {e}")
                
            pb.update(task, advance=1)

    total_transcribed = len(transcripts)
    log.info(f"[green]✓ Transcribed {total_transcribed} of {len(segment_paths)} {segment_label} segments for '{target_name}'.[/]")
    return transcripts


def concatenate_segments(
    audio_segment_paths: list[Path],
    destination_concatenated_file: Path,
    tmp_dir_concat: Path,
    silence_duration: float = 0.5,
    output_sr_concat: int = 44100,
    output_channels_concat: int = 1,
) -> bool:
    if not audio_segment_paths:
        log.warning(
            f"No segments to concatenate for {destination_concatenated_file.name}."
        )
        return False

    ensure_dir_exists(tmp_dir_concat)
    ensure_dir_exists(destination_concatenated_file.parent)

    # Sort segments by original start time parsed from filename
    # Filename pattern: {target_name}_solo_final_{id}_{start_time_str}s_to_{end_time_str}s.wav
    time_pattern_concat = re.compile(r"(\d+p\d+)s_to_")

    def get_sort_key_concat(p: Path):
        try:
            match = time_pattern_concat.search(p.stem)
            if match:
                start_time_str = match.group(1)  # e.g. "0p123"
                return float(
                    start_time_str.replace("p", ".")
                )  # Convert "0p123" to 0.123
            log.debug(
                f"Could not parse start time from {p.name} for sorting concat list. Using 0.0 as sort key."
            )
            return 0.0  # Default sort key if pattern mismatch
        except Exception as e_sort:
            log.debug(f"Error parsing sort key from {p.name}: {e_sort}. Using 0.0.")
            return 0.0

    sorted_audio_paths = sorted(audio_segment_paths, key=get_sort_key_concat)

    silence_file = (
        tmp_dir_concat
        / f"silence_{silence_duration}s_{output_sr_concat}hz_{output_channels_concat}ch.wav"
    )
    if silence_duration > 0:
        try:
            if not silence_file.exists() or silence_file.stat().st_size == 0:
                channel_layout_str = (
                    "mono" if output_channels_concat == 1 else "stereo"
                )  # Adjust if more channels needed
                anullsrc_description = f"anullsrc=channel_layout={channel_layout_str}:sample_rate={output_sr_concat}"
                (
                    ffmpeg.input(
                        anullsrc_description, format="lavfi", t=str(silence_duration)
                    )
                    .output(
                        str(silence_file),
                        acodec="pcm_s16le",
                        ar=str(output_sr_concat),
                        ac=output_channels_concat,
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
        except ffmpeg.Error as e_ff_silence:
            err_msg = (
                e_ff_silence.stderr.decode(errors="ignore")
                if e_ff_silence.stderr
                else "ffmpeg error"
            )
            log.error(f"ffmpeg failed to create silence file: {err_msg}")
            return False

    list_file_path = (
        tmp_dir_concat / f"{destination_concatenated_file.stem}_concat_list.txt"
    )
    concat_lines = []
    valid_segment_count = 0
    for i, audio_path in enumerate(sorted_audio_paths):
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            log.warning(
                f"Segment {audio_path.name} for concatenation is missing or empty. Skipping."
            )
            continue

        if i > 0 and silence_duration > 0 and silence_file.exists():
            concat_lines.append(f"file '{silence_file.resolve().as_posix()}'")
        concat_lines.append(f"file '{audio_path.resolve().as_posix()}'")
        valid_segment_count += 1

    if valid_segment_count == 0:
        log.warning(
            f"No valid segments to concatenate for {destination_concatenated_file.name}."
        )
        return False

    # If only one valid segment and no silence, just copy/re-encode it
    if valid_segment_count == 1 and silence_duration == 0:
        single_valid_path = Path(
            concat_lines[0].split("'")[1]
        )  # Extract path from "file 'path'"
        log.info(
            f"Only one segment to 'concatenate'. Copying/Re-encoding {single_valid_path.name} to {destination_concatenated_file.name}"
        )
        try:
            (
                ffmpeg.input(str(single_valid_path))
                .output(
                    str(destination_concatenated_file),
                    acodec="pcm_s16le",
                    ar=output_sr_concat,
                    ac=output_channels_concat,
                )
                .overwrite_output()
                .run(quiet=True)
            )
            return True
        except ffmpeg.Error as e_ff_single:
            err_msg = (
                e_ff_single.stderr.decode(errors="ignore")
                if e_ff_single.stderr
                else "ffmpeg error"
            )
            log.error(f"ffmpeg single segment copy/re-encode failed: {err_msg}")
            return False

    try:
        list_file_path.write_text("\n".join(concat_lines), encoding="utf-8")
    except Exception as e_write_list:
        log.error(
            f"Failed to write ffmpeg concatenation list file {list_file_path.name}: {e_write_list}"
        )
        return False

    log.info(
        f"Concatenating {valid_segment_count} segments to: {destination_concatenated_file.name}..."
    )
    try:
        (
            ffmpeg.input(
                str(list_file_path), format="concat", safe=0
            )  # safe=0 allows absolute paths
            .output(
                str(destination_concatenated_file),
                acodec="pcm_s16le",
                ar=output_sr_concat,
                ac=output_channels_concat,
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        log.info(
            f"[green]✓ Successfully concatenated segments to: {destination_concatenated_file.name}[/]"
        )
        return True
    except ffmpeg.Error as e_ff_concat:
        err_msg = (
            e_ff_concat.stderr.decode(errors="ignore")
            if e_ff_concat.stderr
            else "ffmpeg error"
        )
        log.error(
            f"ffmpeg concatenation failed for {destination_concatenated_file.name}: {err_msg}"
        )
        log.debug(
            f"Concatenation list file content ({list_file_path.name}):\n"
            + "\n".join(concat_lines)
        )
        return False
    finally:
        # Clean up temporary files
        if list_file_path.exists():
            list_file_path.unlink(missing_ok=True)
        if silence_duration > 0 and silence_file.exists():
            silence_file.unlink(missing_ok=True)





def run_bandit_on_noisy_segments(
    noisy_paths: list[Path], 
    bandit_separator_model,  # Can be either Path or loaded model dict
    output_dir_cleaned: Path,
    tmp_dir: Path
) -> list[Path]:
    """
    Runs Bandit-v2 vocal separation on a list of (small) noisy audio files.
    Can use either subprocess method (with Path) or direct method (with loaded model).
    """
    if not noisy_paths:
        log.info("No noisy segments to process with Bandit-v2.")
        return []

    # Check if we have a loaded model or checkpoint path
    if isinstance(bandit_separator_model, dict) and "system" in bandit_separator_model:
        # Use direct method with loaded model
        return run_bandit_on_noisy_segments_direct(
            noisy_paths, bandit_separator_model, output_dir_cleaned, tmp_dir
        )
    elif isinstance(bandit_separator_model, Path):
        # Use subprocess method
        return run_bandit_on_noisy_segments_subprocess(
            noisy_paths, bandit_separator_model, output_dir_cleaned, tmp_dir
        )
    else:
        log.error("Invalid bandit_separator_model: must be either a Path or a loaded model dict")
        return []


def run_bandit_on_noisy_segments_direct(
    noisy_paths: list[Path], 
    bandit_model: dict,  # Loaded model dict
    output_dir_cleaned: Path,
    tmp_dir: Path
) -> list[Path]:
    """
    Runs Bandit-v2 on noisy segments using the loaded model directly.
    """
    if not bandit_model or "system" not in bandit_model:
        log.error("Bandit model not properly loaded")
        return []
        
    system = bandit_model["system"]
    cfg = bandit_model["config"]
    
    ensure_dir_exists(output_dir_cleaned)
    log.info(f"Running Bandit-v2 (direct) on {len(noisy_paths)} noisy segments...")

    cleaned_segment_paths = []
    
    with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb:
        task = pb.add_task("Cleaning noisy segments (direct)...", total=len(noisy_paths))
        for noisy_file in noisy_paths:
            cleaned_output_path = output_dir_cleaned / f"{noisy_file.stem}_cleaned.wav"
            
            try:
                # Load the noisy audio
                audio, fs = ta.load(str(noisy_file))
                
                # Resample if needed
                if fs != cfg.fs:
                    audio = ta.functional.resample(audio, fs, cfg.fs)
                
                # Ensure correct dimensions
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # Add channel dimension
                
                audio = audio.to(DEVICE)
                
                # Create batch
                batch = {
                    "mixture": {
                        "audio": audio.unsqueeze(0),  # Add batch dimension
                    }
                }
                
                # Clear GPU cache before processing
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                
                # Run inference
                with torch.inference_mode():
                    output = system.inference_handler(batch["mixture"]["audio"], system.model)
                
                # Extract vocals from output
                if "estimates" in output and "speech" in output["estimates"]:
                    vocals_audio = output["estimates"]["speech"]["audio"][0].cpu()
                    
                    # Save the cleaned segment
                    ta.save(str(cleaned_output_path), vocals_audio, cfg.fs)
                    
                    if cleaned_output_path.exists() and cleaned_output_path.stat().st_size > 0:
                        cleaned_segment_paths.append(cleaned_output_path)
                        log.debug(f"Successfully cleaned '{noisy_file.name}' -> '{cleaned_output_path.name}'")
                    else:
                        log.warning(f"Cleaned output for '{noisy_file.name}' was not created or is empty")
                else:
                    log.warning(f"Expected 'speech' stem not found in Bandit output for '{noisy_file.name}'")
                    
            except Exception as e:
                log.error(f"Error cleaning segment '{noisy_file.name}': {e}")
                if "CUDA out of memory" in str(e):
                    log.error("GPU out of memory during segment cleaning")
            
            pb.update(task, advance=1)

    log.info(f"Bandit-v2 direct processing complete. Successfully cleaned {len(cleaned_segment_paths)} segments.")
    return cleaned_segment_paths


def run_bandit_on_noisy_segments_subprocess(
    noisy_paths: list[Path], 
    bandit_separator_model: Path, 
    output_dir_cleaned: Path,
    tmp_dir: Path
) -> list[Path]:
    """
    Runs Bandit-v2 vocal separation on a list of (small) noisy audio files using subprocess.
    """
    if not bandit_separator_model.exists():
        log.error("Bandit-v2 model not available. Cannot clean noisy segments.")
        return []
    
    ensure_dir_exists(output_dir_cleaned)
    log.info(f"Running Bandit-v2 (subprocess) on {len(noisy_paths)} noisy segments...")

    cleaned_segment_paths = []
    
    bandit_repo_path = Path(os.environ.get('BANDIT_REPO_PATH', 'repos/bandit-v2')).resolve()
    original_config_path = bandit_repo_path / "expt" / "inference.yaml"
    temp_config_path = bandit_repo_path / "expt" / "inference_temp_segment_cleaner.yaml"

    try:
        # Use the same approach as the main vocal separation function
        with open(original_config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        repo_path_url = bandit_repo_path.as_posix()
        config_content = config_content.replace('$REPO_ROOT', repo_path_url)
        config_content = config_content.replace('data: dnr-v3-com-smad-multi-v2', 'data: dnr-v3-com-smad-multi-v2b')
        
        import re
        config_content = re.sub(r'file://([^"\']+)', lambda m: 'file://' + m.group(1).replace('\\', '/'), config_content)
        
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        with Progress(*Progress.get_default_columns(), console=console, transient=True) as pb:
            task = pb.add_task("Cleaning noisy segments (subprocess)...", total=len(noisy_paths))
            for noisy_file in noisy_paths:
                segment_temp_output = tmp_dir / f"bandit_out_{noisy_file.stem}"
                ensure_dir_exists(segment_temp_output)
                cleaned_output_path = output_dir_cleaned / f"{noisy_file.stem}_cleaned.wav"
                
                cmd = [
                    sys.executable, "inference.py",
                    "--config-name", temp_config_path.stem,                    f"ckpt_path={bandit_separator_model.resolve()}",
                    f"test_audio={noisy_file.resolve()}",
                    f"--output_path={segment_temp_output.resolve()}",
                    "--model_variant=speech"
                ]
                
                env = os.environ.copy()
                env["REPO_ROOT"] = str(bandit_repo_path)
                env["HYDRA_FULL_ERROR"] = "1"
                if DEVICE.type == "cuda":
                    env["CUDA_VISIBLE_DEVICES"] = "0"

                result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(bandit_repo_path))

                if result.returncode == 0:
                    expected_output = segment_temp_output / "speech_estimate.wav"
                    if expected_output.exists():
                        shutil.copy(str(expected_output), str(cleaned_output_path))
                        cleaned_segment_paths.append(cleaned_output_path)
                        log.debug(f"Successfully cleaned '{noisy_file.name}' -> '{cleaned_output_path.name}'")
                    else:
                        log.warning(f"Bandit ran for '{noisy_file.name}' but output 'speech_estimate.wav' not found.")
                else:
                    log.error(f"Bandit failed for segment '{noisy_file.name}'.")
                    log.error("--- BANDIT STDERR ---")
                    console.print(result.stderr)
                    log.error("--- BANDIT STDOUT ---")
                    console.print(result.stdout)
                    log.error("--- END BANDIT OUTPUT ---")

                shutil.rmtree(segment_temp_output, ignore_errors=True)
                pb.update(task, advance=1)
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()

    log.info(f"Bandit-v2 subprocess processing complete. Successfully cleaned {len(cleaned_segment_paths)} segments.")
    return cleaned_segment_paths


def run_bandit_vocal_separation_direct(
    input_audio_file: Path,
    bandit_model: dict,  # The loaded model dict from init_bandit_separator
    output_dir: Path,
    chunk_minutes: float = 5.0,
) -> Path | None:
    """Performs vocal separation using loaded Bandit-v2 model directly (no subprocess)."""
    
    # Set up logging fallback
    import logging
    try:
        # Try to use the global log if available
        if 'log' in globals() and log is not None:
            logger = log
        else:
            raise AttributeError("Global log not available")
    except (NameError, AttributeError):
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)
    
    if not bandit_model or "system" not in bandit_model:
        logger.error("Bandit model not properly loaded")
        return None
        
    system = bandit_model["system"]
    cfg = bandit_model["config"]

    # If config is None (direct loading), create minimal config
    if cfg is None:
        class MinimalConfig:
            def __init__(self):
                self.fs = 48000  # Bandit-v2 sample rate
        cfg = MinimalConfig()

    logger.info(f"Starting direct vocal separation with Bandit-v2 for: {input_audio_file.name}")
    ensure_dir_exists(output_dir)    # Output filename for the vocals stem
    vocals_output_filename = (
        output_dir / f"{input_audio_file.stem}_vocals_bandit_v2.wav"
    )
    
    if vocals_output_filename.exists() and vocals_output_filename.stat().st_size > 0:
        logger.info(
            f"Found existing Bandit-v2 vocals, skipping separation: {vocals_output_filename.name}"
        )
        return vocals_output_filename

    try:
        # Load audio with torchaudio
        audio, fs = ta.load(str(input_audio_file))
        logger.info(f"Loaded audio: {audio.shape}, fs: {fs}")

        # Resample if needed (Bandit-v2 typically expects 48kHz)
        if fs != cfg.fs:
            logger.info(f"Resampling from {fs}Hz to {cfg.fs}Hz")
            audio = ta.functional.resample(audio, fs, cfg.fs)        # Check if we need to process in chunks
        duration_seconds = audio.shape[-1] / cfg.fs
        duration_minutes = duration_seconds / 60
        
        if duration_minutes > chunk_minutes:
            logger.info(f"Audio is {duration_minutes:.1f} minutes. Processing in chunks...")
            return _process_audio_in_chunks_direct(
                audio, cfg.fs, system, vocals_output_filename, chunk_minutes
            )
        else:
            logger.info(f"Audio is {duration_minutes:.1f} minutes. Processing as single file...")
            return _process_audio_single_direct(audio, cfg.fs, system, vocals_output_filename)

    except Exception as e:
        logger.error(f"Bandit-v2 direct vocal separation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def _process_audio_single_direct(
    audio: torch.Tensor,
    sample_rate: int,
    system,
    output_path: Path
) -> Path | None:
    """Process audio directly with the loaded Bandit model."""
    
    # Set up logging fallback
    import logging
    try:
        # Try to use the global log if available
        if 'log' in globals() and log is not None:
            logger = log
        else:
            raise AttributeError("Global log not available")
    except (NameError, AttributeError):
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)
    
    try:
        # Ensure audio is on the right device and has correct dimensions
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        
        audio = audio.to(DEVICE)
        
        logger.info("Running inference with Bandit-v2 model...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,        ) as progress:
            task = progress.add_task("Bandit-v2 (direct)...", total=None)
            
            with torch.inference_mode():
                # Check if this is a direct model (without system wrapper) or a system
                if hasattr(system, 'inference_handler') and hasattr(system, 'model'):
                    # This is a system with inference handler
                    batch = {
                        "mixture": {
                            "audio": audio.unsqueeze(0),  # Add batch dimension
                        }
                    }
                    output = system.inference_handler(batch["mixture"]["audio"], system.model)
                else:
                    # This is a direct model, call forward method directly
                    # For direct model, need to pass the correct batch format
                    audio_input = audio.unsqueeze(0)  # Shape: [batch=1, channels, time]
                    logger.info(f"Input shape to direct model: {audio_input.shape}")
                    
                    # Bandit model expects batch format: {"mixture": {"audio": tensor}}
                    batch_input = {
                        "mixture": {
                            "audio": audio_input
                        }
                    }
                    
                    try:
                        output = system(batch_input)
                        logger.info(f"Direct model output type: {type(output)}")
                        if isinstance(output, torch.Tensor):
                            logger.info(f"Direct model output shape: {output.shape}")
                        elif isinstance(output, dict):
                            logger.info(f"Direct model output keys: {list(output.keys())}")
                            for key, val in output.items():
                                if isinstance(val, torch.Tensor):
                                    logger.info(f"  {key}: {val.shape}")
                                elif isinstance(val, dict):
                                    logger.info(f"  {key}: dict with keys {list(val.keys())}")
                        elif isinstance(output, (list, tuple)):
                            logger.info(f"Direct model output list/tuple length: {len(output)}")
                            for i, item in enumerate(output):
                                if isinstance(item, torch.Tensor):
                                    logger.info(f"  [{i}]: {item.shape}")
                        else:
                            logger.info(f"Direct model output: {output}")
                    except Exception as model_error:
                        logger.error(f"Error during model forward pass: {model_error}")
                        import traceback
                        logger.error(f"Model error traceback: {traceback.format_exc()}")
                        raise
            progress.update(task, completed=1, total=1)

        # Extract vocals from output
        logger.info(f"DEBUG: Output type: {type(output)}")
        if isinstance(output, torch.Tensor):
            logger.info(f"DEBUG: Output tensor shape: {output.shape}")
        
        if isinstance(output, dict) and "estimates" in output and "speech" in output["estimates"]:
            # System output format
            vocals_audio = output["estimates"]["speech"]["audio"][0].cpu()
        elif isinstance(output, torch.Tensor):
            # Direct model output - debug the actual shape
            logger.info(f"Direct model output shape: {output.shape}")
            
            if output.dim() == 4:  # [batch, stems, channels, time]
                vocals_audio = output[0, 0].cpu()  # First batch, first stem (speech)
            elif output.dim() == 3:  # Could be [stems, channels, time] or [batch, channels, time]
                # Check if this looks like [stems, channels, time] (multiple stems)
                if output.shape[0] == 3:  # 3 stems: speech, music, sfx
                    vocals_audio = output[0].cpu()  # First stem (speech)
                else:
                    # Assume [batch, channels, time] - single stem output
                    vocals_audio = output[0].cpu()  # First batch
            elif output.dim() == 2:  # [channels, time] - single audio
                vocals_audio = output.cpu()
            else:
                logger.error(f"Unexpected output tensor shape: {output.shape}")
                return None
        else:
            logger.error(f"Unexpected output format: {type(output)}")
            if isinstance(output, dict):
                logger.error(f"Output keys: {list(output.keys())}")
            return None

        # Save the vocals
        ta.save(str(output_path), vocals_audio, sample_rate)
        
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"[green]✓ Bandit-v2 direct separation completed: {output_path.name}[/]")
            return output_path
        else:
            logger.error("Output file was not created or is empty")
            return None
            
    except Exception as e:
        logger.error(f"Error in direct Bandit processing: {e}")
        if "CUDA out of memory" in str(e):
            logger.error("GPU out of memory. Try reducing chunk_minutes or using a smaller audio file.")
        return None


def _process_audio_in_chunks_direct(
    audio: torch.Tensor,
    sample_rate: int,
    system,
    output_path: Path,
    chunk_minutes: float
) -> Path | None:
    """Process audio in chunks directly with the loaded Bandit model."""
    
    try:
        # Ensure audio has correct dimensions
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
            
        total_samples = audio.shape[-1]
        chunk_samples = int(chunk_minutes * 60 * sample_rate)
        crossfade_samples = int(0.5 * sample_rate)  # 0.5 second crossfade
        
        chunks_processed = []
        chunk_idx = 0
        
        for start_idx in range(0, total_samples, chunk_samples):
            # Calculate chunk boundaries with crossfade
            actual_start = max(0, start_idx - crossfade_samples if chunk_idx > 0 else start_idx)
            end_idx = min(total_samples, start_idx + chunk_samples)
            actual_end = min(total_samples, end_idx + crossfade_samples if end_idx < total_samples else end_idx)
            
            log.info(f"Processing chunk {chunk_idx + 1} ({actual_start/sample_rate/60:.1f}-{actual_end/sample_rate/60:.1f} min)")
            
            # Extract chunk
            chunk_audio = audio[:, actual_start:actual_end].to(DEVICE)
            
            # Create batch for this chunk
            batch = {
                "mixture": {
                    "audio": chunk_audio.unsqueeze(0),  # Add batch dimension
                }
            }
            
            # Clear GPU cache before processing
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            
            with torch.inference_mode():
                output = system.inference_handler(batch["mixture"]["audio"], system.model)
            
            if "estimates" in output and "speech" in output["estimates"]:
                chunk_vocals = output["estimates"]["speech"]["audio"][0].cpu()
                chunks_processed.append({
                    "audio": chunk_vocals,
                    "start_idx": actual_start,
                    "original_start": start_idx,
                    "original_end": end_idx
                })
            else:
                log.error(f"Chunk {chunk_idx + 1} processing failed - no speech output")
                return None
                
            chunk_idx += 1
        
        if not chunks_processed:
            log.error("No chunks were successfully processed")
            return None
            
        # Concatenate chunks
        log.info(f"Concatenating {len(chunks_processed)} chunks...")
        
        if len(chunks_processed) == 1:
            final_audio = chunks_processed[0]["audio"]
        else:
            # Simple concatenation for now (could add crossfading)
            final_audio = torch.cat([chunk["audio"] for chunk in chunks_processed], dim=-1)
        
        # Save final result
        ta.save(str(output_path), final_audio, sample_rate)
        
        if output_path.exists() and output_path.stat().st_size > 0:
            log.info(f"[green]✓ Bandit-v2 direct chunked separation completed: {output_path.name}[/]")
            return output_path
        else:
            log.error("Output file was not created or is empty")
            return None
            
    except Exception as e:
        log.error(f"Error in chunked direct Bandit processing: {e}")
        if "CUDA out of memory" in str(e):
            log.error("GPU out of memory. Try reducing chunk_minutes.")
        return None
