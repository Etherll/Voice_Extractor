#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_extractor.py - Advanced Voice Extractor

Processes an input audio file to identify, isolate (solo segments),
verify, and transcribe segments of a target speaker.
Uses Bandit-v2 for vocal separation, PyAnnote for diarization/OSD,
WeSpeaker & SpeechBrain for speaker ID/verification, and Whisper for ASR.
"""
import sys
import argparse
from pathlib import Path
import time
import shutil
import logging
import json
import torch
import os

# Parse arguments FIRST to know what components are needed
parser = argparse.ArgumentParser(
    description="Advanced Voice Extractor for TTS data preparation. Uses Bandit-v2, PyAnnote 3.1, WeSpeaker, "
                "SpeechBrain, and Whisper.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Required Arguments
req_group = parser.add_argument_group('Required Arguments')
req_group.add_argument("--input-audio", "-i", type=str, required=True, help="Path to the main input audio file.")
req_group.add_argument("--reference-audio", "-r", type=str, required=True, help="Path to a clean reference audio clip of the target speaker (for speaker ID/verification).")
req_group.add_argument("--target-name", "-n", type=str, required=True, help="A name for the target speaker.")

# Path and Output Arguments
path_group = parser.add_argument_group('Path and Output Arguments')
path_group.add_argument("--output-base-dir", "-o", type=str, default="./output_runs", help="Base directory for all output.")
path_group.add_argument("--output-sr", type=int, default=44100, help="Sample rate for final extracted and concatenated SOLO audio segments (Hz).")
path_group.add_argument("--bandit-repo-path", type=str, default=os.getenv("BANDIT_REPO_PATH", "repos/bandit-v2"), help="Path to the cloned kwatcharasupat/bandit-v2 repository. If not set, attempts to use PYTHONPATH.")

# Authentication Arguments
auth_group = parser.add_argument_group('Authentication Arguments')
auth_group.add_argument("--token", "-t", type=str, default=None, help="HuggingFace User Access Token for PyAnnote models (can also be set via HUGGINGFACE_TOKEN env var).")

# Model Configuration Arguments
model_group = parser.add_argument_group('Model Configuration Arguments')
model_group.add_argument("--bandit-model-path", type=str, default="models/bandit_checkpoint_eng.ckpt", help="Path to the Bandit-v2 .ckpt model file (e.g., checkpoint-eng.ckpt). Required if Bandit is not skipped.")
model_group.add_argument("--wespeaker-rvector-model", type=str, default="english", help="WeSpeaker Deep r-vector model identifier ('english', 'chinese') or local path to model directory with avg_model.pt and config.yaml.")
model_group.add_argument("--wespeaker-gemini-model", type=str, default="english", help="WeSpeaker verification model identifier ('english', 'chinese') or local path to model directory with avg_model.pt and config.yaml.")
model_group.add_argument("--diar-model", type=str, default="pyannote/speaker-diarization-3.1", help="PyAnnote speaker diarization model (forced to v3.1).")
model_group.add_argument("--osd-model", type=str, default="pyannote/overlapped-speech-detection", help="PyAnnote model for Overlapped Speech Detection (e.g., pyannote/overlapped-speech-detection or pyannote/segmentation-3.0).")
model_group.add_argument("--whisper-model", type=str, default="large-v3", help="Whisper model name for transcription (e.g., tiny, base, small, medium, large, large-v2, large-v3).")

# Processing Control Arguments
proc_group = parser.add_argument_group('Processing Control Arguments')
proc_group.add_argument("--language", type=str, default="en", help="Language code for Whisper transcription (e.g., 'en', 'es', 'auto' for Whisper only).")
proc_group.add_argument("--diar-hyperparams", type=str, default="{}", help="JSON string of hyperparameters for PyAnnote diarization pipeline.")
proc_group.add_argument("--skip-bandit", action="store_true", help="Skip Bandit-v2 vocal separation stage (use original audio for downstream).")
proc_group.add_argument("--disable-speechbrain", action="store_true", help="Disable SpeechBrain ECAPA-TDNN for speaker verification (other verification models will still run).")
proc_group.add_argument("--skip-rejected-transcripts", action="store_true", help="Skip transcription of segments that were rejected by speaker verification.")
proc_group.add_argument("--concat-silence", type=float, default=0.25, help="Duration of silence (seconds) between concatenated SOLO verified segments.")
proc_group.add_argument("--preload-whisper", action="store_true", help="Pre-load Whisper model at startup (can save time if RAM is sufficient).")
proc_group.add_argument("--classify-and-clean", action="store_true", help="After verification, classify segments as clean/noisy and run Bandit-v2 on only the noisy ones.")

# Fine-tuning Parameters for SOLO Segments
tune_group = parser.add_argument_group('Fine-tuning Parameters for SOLO Segments')
tune_group.add_argument("--min-duration", type=float, default=1.0, help="Minimum duration (seconds) for a refined SOLO voice segment to be kept.")
tune_group.add_argument("--merge-gap", type=float, default=0.25, help="Maximum gap (seconds) between target speaker's SOLO segments to merge them.")
tune_group.add_argument("--verification-threshold", type=float, default=0.7, help="Minimum combined speaker verification score (0.0-1.0) for a SOLO segment.")
tune_group.add_argument("--noise-threshold", type=float, default=0.7, help="Cleanliness threshold (0-1) for NoisySpeechDetection model when using --classify-and-clean.")

# Debugging and Miscellaneous
debug_group = parser.add_argument_group('Debugging and Miscellaneous')
debug_group.add_argument("--dry-run", "-d", action="store_true", help="Limits diarization/OSD to first 60s of audio for quick testing.")
debug_group.add_argument("--debug", action="store_true", help="Enable verbose DEBUG level logging and potentially more detailed tracebacks.")
debug_group.add_argument("--keep-temp-files", action="store_true", help="Keep temporary processing directory (__tmp_processing).")

args = parser.parse_args()

# Now setup sys.path for toolkits if needed
def add_toolkit_path_to_sys(toolkit_path_str: str | None, toolkit_name: str):
    if toolkit_path_str:
        toolkit_path = Path(toolkit_path_str).resolve()
        if toolkit_path.is_dir():
            sys.path.insert(0, str(toolkit_path))
            print(f"[Setup] Added {toolkit_name} directory to sys.path: {toolkit_path}")
        else:
            print(f"[Setup Warning] {toolkit_name} path provided but not a valid directory: {toolkit_path}")

if args.bandit_repo_path and not args.skip_bandit:
    add_toolkit_path_to_sys(args.bandit_repo_path, "Bandit-v2")

# --- Bootstrap Dependencies with component info ---
try:
    from common import _ensure, REQ, ensure_repositories, ensure_models
    # Pass component usage info to ensure_repositories and ensure_models
    component_usage = {
        'use_bandit': not args.skip_bandit,
        'use_speechbrain': not args.disable_speechbrain,
    }
    ensure_repositories(component_usage)
    ensure_models(component_usage)
    _ensure(REQ)
except ImportError as e_common_bootstrap:
    print(f"FATAL: Could not import 'common' module for bootstrapping: {e_common_bootstrap}")
    print("Ensure common.py is in the same directory or your PYTHONPATH is set correctly.")
    sys.exit(1)

from common import (
    log, console, DEVICE, DEFAULT_OUTPUT_BASE_DIR, get_huggingface_token,
    DEFAULT_MIN_SEGMENT_SEC, DEFAULT_MAX_MERGE_GAP,
    DEFAULT_VERIFICATION_THRESHOLD,
    save_detailed_spectrograms, create_comparison_spectrograms, create_diarization_plot,
    ensure_dir_exists, safe_filename, format_duration, set_args_for_debug,
    torchaudio_version, torchvision_version
)

# Import pipeline functions AFTER setup
try:
    from audio_pipeline import (
        prepare_reference_audio,
        init_bandit_separator, run_bandit_vocal_separation,
        diarize_audio, detect_overlapped_regions,
        init_wespeaker_models, identify_target_speaker,
        init_speechbrain_speaker_recognition_model,
        slice_and_verify_target_solo_segments,
        transcribe_segments,
        classify_segments_for_noise,
        run_bandit_on_noisy_segments,
        concatenate_segments,
        HAVE_BANDIT_V2, HAVE_WESPEAKER, HAVE_SPEECHBRAIN
    )
except ImportError as e_pipeline_import:
    log.error(f"[bold red]FATAL: Failed to import functions from audio_pipeline.py: {e_pipeline_import}[/]")
    log.error("This might be due to errors in audio_pipeline.py, missing toolkit dependencies, "
              "or issues with PYTHONPATH for Bandit-v2 if its repo path was not provided or is incorrect.")
    if args.bandit_repo_path: log.error(f"  Bandit-v2 repo path used: {args.bandit_repo_path}")
    sys.exit(1)

def main(args):
    """Main orchestrator function for the voice extraction pipeline."""
    start_time_total = time.time()
    set_args_for_debug(args)
    
    log.info(f"[bold cyan]===== Voice Extractor Initializing (Device: {DEVICE.type.upper()}) =====[/]")
    log.info("[bold yellow]Strategy: Extract SOLO target speaker segments (overlap removed by splitting), "
             "then verify and transcribe.[/]")

    # --- Validate Core Paths and Toolkits ---
    input_audio_p = Path(args.input_audio)
    reference_audio_p = Path(args.reference_audio)
    target_name_str = args.target_name

    if not input_audio_p.is_file():
        log.error(f"[bold red]Input audio file not found: {input_audio_p}. Exiting.[/]"); sys.exit(1)
    if not reference_audio_p.is_file():
        log.error(f"[bold red]Reference audio file not found: {reference_audio_p}. Exiting.[/]"); sys.exit(1)
    if not target_name_str.strip():
        log.error("[bold red]Target name cannot be empty. Exiting.[/]"); sys.exit(1)

    # Check if critical toolkits are available based on imports and args
    if (not args.skip_bandit or args.classify_and_clean) and not HAVE_BANDIT_V2:
        log.error("[bold red]Bandit-v2 is required for this workflow but its library/repository was not found. Check installation and --bandit-repo-path. Exiting.[/]"); sys.exit(1)
    if not HAVE_WESPEAKER:
        log.error("[bold red]WeSpeaker library not loaded. This is critical for speaker ID/Verification. Check installation. Exiting.[/]"); sys.exit(1)

    # --- Setup Directories ---
    run_output_dir_name = f"{safe_filename(target_name_str)}_{input_audio_p.stem}_extracted"
    output_dir = Path(args.output_base_dir) / run_output_dir_name
    run_tmp_dir = output_dir / "__tmp_processing"
    
    # Specific output subdirectories
    separated_vocals_dir = output_dir / "separated_vocals_bandit"
    segments_base_output_dir = output_dir / "target_segments_solo"
    transcripts_verified_dir = output_dir / "transcripts_solo_verified_whisper"
    transcripts_rejected_dir = output_dir / "transcripts_solo_rejected_whisper"
    concatenated_output_dir = output_dir / "concatenated_audio_solo_verified"
    visualizations_output_dir = output_dir / "visualizations"

    for dir_path in [output_dir, run_tmp_dir, separated_vocals_dir, segments_base_output_dir,
                     transcripts_verified_dir, transcripts_rejected_dir,
                     concatenated_output_dir, visualizations_output_dir]:
        ensure_dir_exists(dir_path)

    log.info(f"Processing input: [bold cyan]{input_audio_p.name}[/]")
    log.info(f"Reference audio for '{target_name_str}': [bold cyan]{reference_audio_p.name}[/]")
    log.info(f"Run output directory: [bold cyan]{output_dir.resolve()}[/]")
    if args.dry_run: log.warning("[DRY-RUN MODE ENABLED] Processing will be limited.")

    log.info(f"[bold cyan]===== Voice Extractor Initializing (Device: {DEVICE.type.upper()}) =====[/]")
    # --- STAGE 0: Initialize Models ---
    log.info("[bold magenta]== STAGE 0: Initializing Models ==[/]")
    bandit_separator_model = None
    if not args.skip_bandit or args.classify_and_clean:
        if not args.bandit_model_path:
            log.error("[bold red]--bandit-model-path is required for this workflow. Exiting.[/]"); sys.exit(1)
        bandit_separator_model = init_bandit_separator(Path(args.bandit_model_path))
        if bandit_separator_model is None: 
            log.error("[bold red]Failed to initialize Bandit-v2 model, which is required for this workflow. Exiting.[/]"); sys.exit(1)
    if not args.wespeaker_rvector_model or not args.wespeaker_gemini_model:
        log.error("[bold red]--wespeaker-rvector-model and --wespeaker-gemini-model are required. Exiting.[/]"); sys.exit(1)
    wespeaker_models = init_wespeaker_models(args.wespeaker_rvector_model, args.wespeaker_gemini_model)
    if wespeaker_models is None or not wespeaker_models.get("rvector") or not wespeaker_models.get("gemini"):
        log.error("[bold red]Failed to initialize one or more WeSpeaker models. Exiting as they are critical.[/]"); sys.exit(1)

    speechbrain_model = None
    if not args.disable_speechbrain:
        if not HAVE_SPEECHBRAIN:
            log.warning("SpeechBrain not disabled, but library not available. ECAPA-TDNN verification will be skipped.")
        else:
            speechbrain_model = init_speechbrain_speaker_recognition_model()
            if speechbrain_model is None:
                log.warning("Failed to initialize SpeechBrain model. ECAPA-TDNN verification will be skipped.")
    else:
        log.info("SpeechBrain ECAPA-TDNN verification is disabled by user.")

    whisper_asr_model = None
    if args.preload_whisper:
        try:
            log.info(f"Pre-loading Whisper model '{args.whisper_model}' to {DEVICE.type.upper()}...")
            import whisper
            whisper_asr_model = whisper.load_model(args.whisper_model, device=DEVICE)
            log.info(f"Whisper model '{args.whisper_model}' pre-loaded.")
        except Exception as e_preload_whisper:
            log.error(f"Failed to pre-load Whisper model: {e_preload_whisper}. Will attempt to load during transcription stage.")
            whisper_asr_model = None

    # --- STAGE 1: Prepare Reference Audio ---
    log.info("[bold magenta]== STAGE 1: Reference Audio Preparation ==[/]")
    processed_reference_file = prepare_reference_audio(reference_audio_p, run_tmp_dir, target_name_str)
    if not processed_reference_file.exists():
        log.error(f"Failed to create processed reference file. Exiting."); sys.exit(1)
    save_detailed_spectrograms(input_audio_p, visualizations_output_dir, "01_Original_InputAudio", target_name_str)
    save_detailed_spectrograms(processed_reference_file, visualizations_output_dir, "00_Processed_ReferenceAudio_16kMono", target_name_str)

    # --- STAGE 2: Vocal Separation (Bandit-v2) ---
    source_for_diarization_osd = input_audio_p
    bandit_vocals_file = None
    if not args.skip_bandit and not args.classify_and_clean and bandit_separator_model:
        log.info("[bold magenta]== STAGE 2: Initial Vocal Separation (Bandit-v2) ==[/]")
        bandit_vocals_file = run_bandit_vocal_separation(input_audio_p, bandit_separator_model, separated_vocals_dir)
        if bandit_vocals_file and bandit_vocals_file.exists():
            save_detailed_spectrograms(bandit_vocals_file, visualizations_output_dir, "02a_BanditV2_Vocals_Only", target_name_str)
            source_for_diarization_osd = bandit_vocals_file
            log.info(f"Using Bandit-v2 output '{bandit_vocals_file.name}' for subsequent diarization and OSD.")
        else:
            log.warning("Bandit-v2 vocal separation failed or produced no output. Using original audio for downstream tasks.")
    else:
        if args.classify_and_clean:
            log.info("[bold yellow]Skipping initial Bandit-v2 run because --classify-and-clean is active. Bandit will be used later on noisy segments.[/]")
        else:
            log.info(f"Skipping Bandit-v2. Using original input '{input_audio_p.name}' for diarization and OSD.")
    # --- STAGE 3: Speaker Diarization (PyAnnote 3.1) ---
    log.info("[bold magenta]== STAGE 3: Speaker Diarization ==[/]")
    diar_model_config = {"diar_model": args.diar_model, "diar_hyperparams": {}}
    if args.diar_hyperparams:
        try: diar_model_config["diar_hyperparams"] = json.loads(args.diar_hyperparams)
        except json.JSONDecodeError: log.error(f"Invalid JSON for diarization hyperparameters: {args.diar_hyperparams}. Using defaults.")
    
    huggingface_auth_token = args.token if args.token else get_huggingface_token()
    diarization_annotation = diarize_audio(source_for_diarization_osd, run_tmp_dir, huggingface_auth_token, diar_model_config, args.dry_run)
    if not diarization_annotation or not diarization_annotation.labels():
        log.error("[bold red]Diarization failed or produced no speaker labels. Cannot proceed. Exiting.[/]")
        if args.dry_run: log.warning("Diarization might be empty due to dry-run mode limiting audio duration.")
        sys.exit(1)

    # --- STAGE 4: Overlapped Speech Detection (PyAnnote) ---
    log.info("[bold magenta]== STAGE 4: Overlapped Speech Detection ==[/]")
    overlap_timeline = detect_overlapped_regions(source_for_diarization_osd, run_tmp_dir, huggingface_auth_token, args.osd_model, args.dry_run)
    if overlap_timeline is None:
        log.warning("Overlap detection failed significantly. Proceeding with an empty overlap timeline, results may be suboptimal.")
        from pyannote.core import Timeline as PyannoteTimeline
        overlap_timeline = PyannoteTimeline() 

    # --- STAGE 5: Identify Target Speaker (WeSpeaker Deep r-vector) ---
    log.info(f"[bold magenta]== STAGE 5: Identifying Target Speaker ('{target_name_str}') ==[/]")
    identified_target_label = identify_target_speaker(
        diarization_annotation, 
        source_for_diarization_osd,
        processed_reference_file, 
        target_name_str, 
        wespeaker_models["rvector"]
    )
    if not identified_target_label:
        log.error(f"[bold red]Failed to identify target speaker '{target_name_str}' in the audio. Exiting.[/]"); sys.exit(1)

    # Visualizations after Diarization/OSD/ID
    create_diarization_plot(diarization_annotation, identified_target_label, target_name_str,
                            visualizations_output_dir,
                            plot_title_prefix=f"03_Diarization_OSD_{safe_filename(target_name_str)}",
                            overlap_timeline=overlap_timeline)
    save_detailed_spectrograms(source_for_diarization_osd, visualizations_output_dir,
                               "04_Source_For_Slicing_with_OverlapMarkings", target_name_str,
                               overlap_timeline=overlap_timeline)

    # --- STAGE 6: Slice & Verify Target's SOLO Segments (Multi-stage Verification) ---
    log.info(f"[bold magenta]== STAGE 6: Slice & Verify '{target_name_str}' SOLO Segments ==[/]")
    slicing_source_audio = bandit_vocals_file if bandit_vocals_file and bandit_vocals_file.exists() else input_audio_p
    log.info(f"Slicing final segments from: {slicing_source_audio.name}")

    verified_solo_paths, rejected_solo_paths = slice_and_verify_target_solo_segments(
        diarization_annotation, identified_target_label, overlap_timeline,
        slicing_source_audio, processed_reference_file, target_name_str,
        segments_base_output_dir, run_tmp_dir,
        args.verification_threshold, args.min_duration, args.merge_gap,
        wespeaker_models, speechbrain_model,
        output_sample_rate=int(args.output_sr), output_channels=1
    )
    
    # --- NEW STAGE 6.5: Classify, Separate, and Re-process Noisy Segments ---
    if args.classify_and_clean and verified_solo_paths:
        log.info(f"[bold magenta]== STAGE 6.5: Classifying and Cleaning Noisy Segments ==[/]")
        
        # 1. Classify segments
        initially_clean_paths, noisy_paths_to_clean = classify_segments_for_noise(
            verified_solo_paths, args.noise_threshold
        )
        
        # 2. Re-organize files into new subdirectories for clarity
        clean_verified_dir = segments_base_output_dir / "clean_verified"
        noisy_originals_dir = segments_base_output_dir / "noisy_originals_for_cleaning"
        cleaned_by_bandit_dir = segments_base_output_dir / "noisy_cleaned_by_bandit"
        ensure_dir_exists(clean_verified_dir)
        ensure_dir_exists(noisy_originals_dir)
        ensure_dir_exists(cleaned_by_bandit_dir)

        # Move files and update path lists
        final_clean_paths = []
        for p in initially_clean_paths:
            dest = clean_verified_dir / p.name
            shutil.move(str(p), str(dest))
            final_clean_paths.append(dest)
        
        moved_noisy_paths = []
        for p in noisy_paths_to_clean:
            dest = noisy_originals_dir / p.name
            shutil.move(str(p), str(dest))
            moved_noisy_paths.append(dest)
        
        # 3. Run Bandit on the noisy segments
        newly_cleaned_paths = run_bandit_on_noisy_segments(
            moved_noisy_paths, bandit_separator_model, cleaned_by_bandit_dir, run_tmp_dir
        )

        # 4. Update the final list of verified paths for downstream tasks
        verified_solo_paths = final_clean_paths + newly_cleaned_paths
        log.info(f"Final dataset for transcription/concatenation consists of {len(final_clean_paths)} initially clean + {len(newly_cleaned_paths)} newly cleaned segments.")
    
    # --- STAGE 7: Transcribe Segments (Whisper) ---
    if verified_solo_paths:
        log.info(f"[bold magenta]== STAGE 7a: Transcribing VERIFIED SOLO Segments ('{target_name_str}') ==[/]")
        transcribe_segments(
            verified_solo_paths, transcripts_verified_dir,
            target_name_str, "solo_verified", args.whisper_model, args.language, whisper_asr_model
        )
    else:
        log.info(f"No verified solo segments of '{target_name_str}' to transcribe.")

    if not args.skip_rejected_transcripts and rejected_solo_paths:
        log.info(f"[bold magenta]== STAGE 7b: Transcribing REJECTED SOLO Segments ('{target_name_str}') ==[/]")
        transcribe_segments(
            rejected_solo_paths, transcripts_rejected_dir,
            target_name_str, "solo_rejected_for_review", args.whisper_model, args.language, whisper_asr_model
        )
    else:
        log.info(f"Skipping transcription of rejected segments for '{target_name_str}'.")

    # --- STAGE 8: Concatenate VERIFIED SOLO Segments ---
    log.info(f"[bold magenta]== STAGE 8: Concatenating VERIFIED SOLO Segments ('{target_name_str}') ==[/]")
    concatenated_solo_verified_file = None
    if verified_solo_paths:
        concat_final_path = concatenated_output_dir / f"{safe_filename(target_name_str)}_solo_verified_concatenated.wav"
        concat_tmp_dir = run_tmp_dir / f"concat_tmp_{safe_filename(target_name_str)}"
        concat_success = concatenate_segments(
            verified_solo_paths, concat_final_path, concat_tmp_dir,
            args.concat_silence, int(args.output_sr)
        )
        if concat_success and concat_final_path.exists():
            concatenated_solo_verified_file = concat_final_path
            save_detailed_spectrograms(concatenated_solo_verified_file, visualizations_output_dir,
                                       "05_Concatenated_Target_SOLO_Verified", target_name_str)
    else:
        log.info(f"No verified solo segments of '{target_name_str}' to concatenate.")

    # --- STAGE 9: Final Comparison Spectrograms ---
    log.info("[bold magenta]== STAGE 9: Generating Final Comparison Spectrograms ==[/]")
    comparison_files_list = [(input_audio_p, "Original Input")]
    overlap_timeline_for_plots = {str(source_for_diarization_osd.resolve()): overlap_timeline}

    if bandit_vocals_file and bandit_vocals_file.exists():
        comparison_files_list.append((bandit_vocals_file, "Bandit-v2 Vocals"))
    if concatenated_solo_verified_file and concatenated_solo_verified_file.exists():
        comparison_files_list.append((concatenated_solo_verified_file, f"{target_name_str} SOLO Verified Concatenated"))
    
    create_comparison_spectrograms(comparison_files_list, visualizations_output_dir, target_name_str,
                                   main_prefix="06_Audio_Processing_Stages_Overview",
                                   overlap_timeline_dict=overlap_timeline_for_plots)

    # --- Finalization ---
    if not args.keep_temp_files and run_tmp_dir.exists():
        log.info(f"Cleaning up temporary directory: {run_tmp_dir.resolve()}")
        try: shutil.rmtree(run_tmp_dir)
        except Exception as e_rm_tmp: log.warning(f"Could not remove temporary directory {run_tmp_dir}: {e_rm_tmp}")
    else:
        log.info(f"Temporary processing files kept at: {run_tmp_dir.resolve()}")

    total_duration_seconds = time.time() - start_time_total
    log.info(f"[bold green]✅ Voice Extractor processing finished successfully for '{target_name_str}'![/]")
    log.info(f"Total processing time: {format_duration(total_duration_seconds)}")
    log.info(f"All output files are located in: [bold cyan]{output_dir.resolve()}[/]")
    log.info(f"  - Verified SOLO Segments: {segments_base_output_dir / (safe_filename(target_name_str) + '_solo_verified')}")
    if (segments_base_output_dir / (safe_filename(target_name_str) + '_solo_rejected_for_review')).exists():
        log.info(f"  - Rejected SOLO Segments (for review): {segments_base_output_dir / (safe_filename(target_name_str) + '_solo_rejected_for_review')}")
    log.info(f"  - Whisper Transcripts (SOLO Verified): {transcripts_verified_dir}")
    if not args.skip_rejected_transcripts and transcripts_rejected_dir.exists() and any(transcripts_rejected_dir.iterdir()):
        log.info(f"  - Whisper Transcripts (SOLO Rejected): {transcripts_rejected_dir}")
    log.info(f"  - Concatenated Audio (SOLO Verified): {concatenated_output_dir}")
    log.info(f"  - Visualizations: {visualizations_output_dir}")


# --- CLI Entry Point ---
if __name__ == "__main__":
    set_args_for_debug(args)

    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("Debug mode enabled. Logging will be verbose.")
        log.debug(f"Python version: {sys.version.split()[0]}")
        log.debug(f"PyTorch version: {torch.__version__}")
        log.debug(f"Torchaudio version: {torchaudio_version}")
        log.debug(f"Torchvision version: {torchvision_version if torchvision_version else 'N/A'}")
        log.debug(f"Device: {DEVICE.type.upper()}")
        if DEVICE.type == "cuda":
            log.debug(f"  CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log.debug(f"    Device {i}: {torch.cuda.get_device_name(i)}")
                try: log.debug(f"      Memory (Allocated/Reserved): {torch.cuda.memory_allocated(i)/1e9:.2f}GB / {torch.cuda.memory_reserved(i)/1e9:.2f}GB")
                except Exception: pass
        log.debug(f"Bandit-v2 available via import: {HAVE_BANDIT_V2}")
        log.debug(f"WeSpeaker available via import: {HAVE_WESPEAKER}")
        log.debug(f"SpeechBrain available via import: {HAVE_SPEECHBRAIN}")
        log.debug(f"Bandit-v2 repo path (from arg or env): {args.bandit_repo_path}")

    # HuggingFace token handling
    if not args.token:
        args.token = get_huggingface_token()

    try:
        main(args)
    except KeyboardInterrupt:
        log.warning("[bold yellow]\nProcess interrupted by user (Ctrl+C). Exiting.[/]")
        if not args.keep_temp_files:
            try:
                _input_audio_p_for_tmp = Path(args.input_audio)
                _target_name_str_for_tmp = args.target_name
                _run_output_dir_name_for_tmp = f"{safe_filename(_target_name_str_for_tmp)}_{_input_audio_p_for_tmp.stem}_Advanced_TTS_Prep"
                _output_dir_for_tmp = Path(args.output_base_dir) / _run_output_dir_name_for_tmp
                tmp_dir_to_clean = _output_dir_for_tmp / "__tmp_processing"
                if tmp_dir_to_clean.exists():
                    log.info(f"Attempting to clean temporary directory on interrupt: {tmp_dir_to_clean.resolve()}")
                    shutil.rmtree(tmp_dir_to_clean, ignore_errors=True)
            except Exception as e_tmp_clean_interrupt:
                log.debug(f"Could not determine or clean tmp_dir path on interrupt: {e_tmp_clean_interrupt}")
        sys.exit(130)
    except FileNotFoundError as e_fnf:
        log.error(f"[bold red][FILE NOT FOUND ERROR] {e_fnf}[/]")
        sys.exit(2)
    except RuntimeError as e_rt:
        log.error(f"[bold red][RUNTIME ERROR] {e_rt}[/]")
        if args.debug: log.exception("Traceback for RuntimeError:")
        sys.exit(1)
    except SystemExit as e_sysexit:
        sys.exit(e_sysexit.code if e_sysexit.code is not None else 1)
    except Exception as e_fatal:
        log.error(f"[bold red][FATAL SCRIPT ERROR] An unexpected error occurred: {e_fatal}[/]")
        if args.debug: log.exception("Full traceback for unexpected error:")
        sys.exit(1)
