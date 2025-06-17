#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
common.py
Shared configurations, constants, utilities, logging, device management,
dependency bootstrapping, and plotting for the Voice Extractor.
"""

import subprocess
import sys
import importlib
import importlib.metadata as md
import os
import getpass
from pathlib import Path
import logging
import shutil
import urllib.request
import hashlib
from tqdm import tqdm
import time
import re
import random

# Pattern to identify Netflix-internal packages
NETFLIX_PRIVATE_PACKAGES = re.compile(
    r'^(baggins|bdp-|jasper|managedbatch|netflix-|nflx[-_]|metatron|obsidian|storage)'
)

# --- Dependency Management ---
REQ = [
    "rich", "ffmpeg-python", "soundfile", "numpy",
    "torch>=2.5.0", "torchaudio>=2.5.0", "torchvision>=0.20.0",  # Changed from >=2.7.0
    "pyannote.audio>=3.3.2",
    "openai-whisper>=20240930",
    "nemo_toolkit[asr]",  # Nvidia Parakeet TDT ASR support
    "matplotlib", "librosa",
    "speechbrain>=1.0.0",
    "torchcrepe>=0.0.21",
    "silero-vad>=5.1.2",
    "pytorch-lightning",
    "hydra-core",
    "pyyaml",
    "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git",
    "scipy",
    "onnx",
    "onnxruntime",
    "fairseq==0.12.2 --no-deps",
    "transformers",
    "audio_separator[gpu]"

]

# Model configurations - will be conditionally downloaded
MODEL_CONFIGS = {
    'bandit_checkpoint_eng': {
        'url': 'https://zenodo.org/records/12701995/files/checkpoint-eng.ckpt?download=1',
        'path': 'models/bandit_checkpoint_eng.ckpt',
        'size': '~450MB',
        'component': 'bandit'  # Which component needs this
    }
}

def _ensure(pkgs):
    """Ensures specified packages are installed, installing them if not."""
    # First, ensure ray[train] is available if ray is in the package list
    ray_packages = [p for p in pkgs if p.startswith("ray")]
    if ray_packages:
        try:
            import ray
            import ray.train
            print("[Setup] Ray with train module already available")
        except ImportError:
            print("[Setup] Installing ray with train extras...")
            env = os.environ.copy()
            env["PIP_INDEX_URL"] = "https://pypi.org/simple"
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "--index-url", "https://pypi.org/simple",
                    "--extra-index-url", "https://download.pytorch.org/whl/cu121",
                    "ray[train]>=2.10.0,<2.20", "pandas", "tensorboard", "tensorboardX"
                ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print("[Setup] ✓ Successfully installed ray[train] and dependencies")
            except subprocess.CalledProcessError:
                print("[Setup] Warning: Could not install ray[train], trying basic ray")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--upgrade",
                        "--index-url", "https://pypi.org/simple",
                        "ray>=2.10.0,<2.20"
                    ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                except:
                    print("[Setup] Warning: Ray installation failed")
    
    # Check if PyTorch is already installed with CUDA support
    pytorch_cuda_available = False
    try:
        import torch
        pytorch_cuda_available = torch.cuda.is_available()
        if pytorch_cuda_available:
            print(f"[Setup] PyTorch {torch.__version__} with CUDA support already installed")
    except ImportError:
        pass
    
    # Continue with the rest of the packages
    installed_pkgs_details = []
    missing_pkgs_to_install_specs = []

    for spec in pkgs:
        # Skip ray packages as they were handled above
        if spec.startswith("ray"):
            continue
        
        # Special handling for PyTorch packages to preserve CUDA support
        if pytorch_cuda_available and any(spec.startswith(p) for p in ["torch>=", "torchaudio>=", "torchvision>="]):
            # Skip version checking for PyTorch packages if CUDA is available
            # This prevents downgrading to CPU-only versions
            try:
                package_name = spec.split(">=")[0].split("==")[0].strip()
                version = md.version(package_name)
                print(f"[Setup] Keeping existing {package_name} {version} with CUDA support")
                continue
            except:
                pass
            
        package_name_for_md_check = spec
        version_spec_part = ""
        is_no_deps_pkg = False

        if ' @ ' in spec:
            package_name_for_md_check = spec.split(' @ ')[0].strip()
        elif " --no-deps" in spec:
            package_name_for_md_check = spec.split(' --no-deps')[0].strip()
            is_no_deps_pkg = True

        if not (' @ ' in spec):
            parts = package_name_for_md_check.split("==") if "==" in package_name_for_md_check else \
                    package_name_for_md_check.split(">=") if ">=" in package_name_for_md_check else \
                    package_name_for_md_check.split("<=") if "<=" in package_name_for_md_check else \
                    package_name_for_md_check.split("!=") if "!=" in package_name_for_md_check else \
                    package_name_for_md_check.split("~=") if "~=" in package_name_for_md_check else [package_name_for_md_check]
            name_for_md_lookup = parts[0].strip()
            if len(parts) > 1 :
                 version_spec_part = package_name_for_md_check
            else:
                 version_spec_part = ""

        try:
            raw_version = md.version(name_for_md_lookup if not (' @ ' in spec) else package_name_for_md_check)
            cleaned_version = raw_version.split('+')[0]
            installed_pkgs_details.append(f"  - Found {name_for_md_lookup if not (' @ ' in spec) else package_name_for_md_check} (version {raw_version}) -> Cleaned for check: {cleaned_version}")

            if version_spec_part and not (' @ ' in spec):
                req_op = ""
                req_ver_str = ""
                current_spec_for_version_extraction = version_spec_part

                for op_char_combo in [">=", "<=", "==", "!=", "~="]:
                    if op_char_combo in current_spec_for_version_extraction:
                        req_op = op_char_combo
                        req_ver_str = current_spec_for_version_extraction.split(op_char_combo, 1)[1]
                        break
                if not req_op and ">" in current_spec_for_version_extraction: req_op = ">"; req_ver_str = current_spec_for_version_extraction.split(">",1)[1]
                if not req_op and "<" in current_spec_for_version_extraction: req_op = "<"; req_ver_str = current_spec_for_version_extraction.split("<",1)[1]

                if req_op and req_ver_str:
                    current_ver_parts = cleaned_version.split('.')
                    req_ver_parts = req_ver_str.split('.')
                    max_len = max(len(current_ver_parts), len(req_ver_parts))
                    current_ver_parts.extend(['0'] * (max_len - len(current_ver_parts)))
                    req_ver_parts.extend(['0'] * (max_len - len(req_ver_parts)))
                    current_ver_tuple = tuple(int(p) if p.isdigit() else 0 for p in current_ver_parts)
                    req_ver_tuple = tuple(int(p) if p.isdigit() else 0 for p in req_ver_parts)

                    compatible = True
                    if req_op == ">=": compatible = current_ver_tuple >= req_ver_tuple
                    elif req_op == "<=": compatible = current_ver_tuple <= req_ver_tuple
                    elif req_op == "==": compatible = current_ver_tuple == req_ver_tuple
                    elif req_op == ">": compatible = current_ver_tuple > req_ver_tuple
                    elif req_op == "<": compatible = current_ver_tuple < req_ver_tuple

                    if not compatible:
                        print(f"[Setup] Package '{name_for_md_lookup if not (' @ ' in spec) else package_name_for_md_check}' version {raw_version} (cleaned: {cleaned_version}) does not meet requirement {spec}. Queuing for update/reinstall.")
                        missing_pkgs_to_install_specs.append(spec)
        except md.PackageNotFoundError:
            print(f"[Setup] Package '{name_for_md_lookup if not (' @ ' in spec) else package_name_for_md_check}' (from spec: {spec}) not found. Queuing for installation.")
            missing_pkgs_to_install_specs.append(spec)
        except ValueError as ve:
            print(f"[Setup] Warning: Could not parse version string for {name_for_md_lookup if not (' @ ' in spec) else package_name_for_md_check} ('{raw_version}' -> '{cleaned_version}'): {ve}. Will rely on pip for this package.")
            if spec not in missing_pkgs_to_install_specs:
                 missing_pkgs_to_install_specs.append(spec)

    if installed_pkgs_details:
        print("[Setup] Checked existing packages:")
        for detail in installed_pkgs_details: print(detail)

    if missing_pkgs_to_install_specs:
        unique_missing_specs = sorted(list(set(missing_pkgs_to_install_specs)))
        
        packages_to_install_normally = []
        packages_to_install_no_deps = [] 

        for spec_str in unique_missing_specs:
            if " --no-deps" in spec_str:
                package_part = spec_str.split(" --no-deps")[0].strip()
                packages_to_install_no_deps.append(package_part)
            else:
                packages_to_install_normally.append(spec_str)
        
        env = os.environ.copy()
        env["PIP_INDEX_URL"] = "https://pypi.org/simple"

        if packages_to_install_normally:
            print(f"[Setup] Installing/updating {len(packages_to_install_normally)} package(s)...")
            
            # Check if we're installing PyTorch packages
            pytorch_packages = [p for p in packages_to_install_normally if any(p.startswith(pt) for pt in ["torch>=", "torchaudio>=", "torchvision>="])]
            other_packages = [p for p in packages_to_install_normally if p not in pytorch_packages]
            
            # Install PyTorch packages with CUDA support if available
            if pytorch_packages:
                try:
                    import torch
                    cuda_available = torch.cuda.is_available()
                except:
                    # If torch is not installed or importable, check for NVIDIA GPU
                    try:
                        subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
                        cuda_available = True
                    except:
                        cuda_available = False
                
                if cuda_available:
                    print("[Setup] Installing PyTorch packages with CUDA support...")
                    pytorch_install_cmd = [
                        sys.executable, "-m", "pip", "install", "--upgrade",
                        "--index-url", "https://download.pytorch.org/whl/cu124"
                    ] + pytorch_packages
                else:
                    print("[Setup] Installing PyTorch packages (CPU only)...")
                    pytorch_install_cmd = [
                        sys.executable, "-m", "pip", "install", "--upgrade",
                        "--index-url", "https://pypi.org/simple"
                    ] + pytorch_packages
                
                try:
                    subprocess.check_call(pytorch_install_cmd, env=env)
                    print(f"[Setup] ✓ Successfully installed/updated PyTorch packages")
                except subprocess.CalledProcessError as e:
                    print(f"[Setup] ERROR: Failed to install/update PyTorch packages. Error: {e}")
                    print("Please try installing them manually. Exiting.")
                    sys.exit(1)
            
            # Install other packages
            if other_packages:
                install_command = [
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "--index-url", "https://pypi.org/simple",
                    "--extra-index-url", "https://download.pytorch.org/whl/cu121"
                ] + other_packages
                try:
                    subprocess.check_call(install_command, env=env)
                    print(f"[Setup] ✓ Successfully installed/updated packages")
                except subprocess.CalledProcessError as e:
                    print(f"[Setup] ERROR: Failed to install/update packages. Error: {e}")
                    print("Please try installing them manually. Exiting.")
                    sys.exit(1)

        if packages_to_install_no_deps:
            print(f"[Setup] Installing {len(packages_to_install_no_deps)} package(s) with --no-deps...")
            install_command_no_deps = [
                sys.executable, "-m", "pip", "install", "--upgrade",
                "--index-url", "https://pypi.org/simple",
                "--extra-index-url", "https://download.pytorch.org/whl/cu121",
                "--no-deps"
            ] + packages_to_install_no_deps
            try:
                subprocess.check_call(install_command_no_deps, env=env)
                print(f"[Setup] ✓ Successfully installed packages with --no-deps")
            except subprocess.CalledProcessError as e:
                print(f"[Setup] ERROR: Failed to install --no-deps packages. Error: {e}")
                print("Please try installing them manually. Exiting.")
                sys.exit(1)
    else:
        print("[Setup] All required packages are already installed.")


# --- End Dependency Management ---

# Delay imports until after _ensure() is called
numpy = None
torch = None
ffmpeg = None
sf = None
Console = None
RichHandler = None
Prompt = None
Confirm = None
matplotlib = None
plt = None
librosa = None

# Placeholder variables
torchaudio_version = "N/A"
torchvision_version = "N/A"
console = None
log = None
DEVICE = None

def _import_dependencies():
    """Import all external dependencies after ensuring they're installed"""
    global numpy, torch, ffmpeg, sf, Console, RichHandler, Prompt, Confirm
    global matplotlib, plt, librosa
    global torchaudio_version, torchvision_version, console, log, DEVICE

    import numpy as np_module
    global numpy
    numpy = np_module

    import torch as _torch
    global torch
    torch = _torch

    import ffmpeg as _ffmpeg
    global ffmpeg
    ffmpeg = _ffmpeg

    import soundfile as _sf
    global sf
    sf = _sf

    from rich.console import Console as _Console
    from rich.logging import RichHandler as _RichHandler
    from rich.prompt import Prompt as _Prompt, Confirm as _Confirm
    global Console, RichHandler, Prompt, Confirm
    Console = _Console
    RichHandler = _RichHandler
    Prompt = _Prompt
    Confirm = _Confirm

    import matplotlib as _matplotlib
    global matplotlib
    matplotlib = _matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as _plt
    global plt
    plt = _plt

    import librosa as _librosa
    global librosa
    librosa = _librosa
    import librosa.display

    global torchaudio
    try:
        import torchaudio as _torchaudio
        torchaudio = _torchaudio
        torchaudio_version = torchaudio.__version__
    except ImportError:
        torchaudio = None
        pass

    global torchvision
    try:
        import torchvision as _torchvision
        torchvision = _torchvision
        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision = None
        pass

    # Now set up console and logging
    console = Console(width=120)
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, console=console, markup=True)]
    )
    log = logging.getLogger("voice_extractor")

    # Now check CUDA (requires torch to be imported)
    DEVICE = check_cuda()


# --- Configuration Constants ---
DEFAULT_OUTPUT_BASE_DIR = Path("./output_runs")
SPECTROGRAM_SEC    = 60
SPEC_FIGSIZE       = (19.2, 6.0)
SPEC_DPI           = 150
HIGH_RES_NFFT      = 4096
FREQ_MAX           = 12000

DEFAULT_MIN_SEGMENT_SEC    = 1.0
DEFAULT_MAX_MERGE_GAP      = 0.25
DEFAULT_VERIFICATION_THRESHOLD = 0.7

SPEECH_BANDS = [
    (0, 300, "Sub-bass / Rumble", "lightgray"),
    (300, 1000, "Vowels & Bass / Warmth", "lightblue"),
    (1000, 3000, "Speech Formants / Clarity", "lightyellow"),
    (3000, 5000, "Consonants & Sibilance / Presence", "lightgreen"),
    (5000, 8000, "Brightness & Harmonics", "lightpink"),
    (8000, 12000, "Air & Detail", "lavender")
]

# --- Device Setup (CPU/CUDA) ---
def check_cuda():
    if torch is None:
        print("PyTorch not imported yet, cannot check CUDA. Call _import_dependencies first.")
        return "cpu"
    if not torch.cuda.is_available():
        print("CUDA is not available. Processing will run on CPU and may be significantly slower.")
        return torch.device("cpu")
    try:
        _ = torch.zeros(1).cuda()
        device_count = torch.cuda.device_count()
        logger_func = log.info if log else print
        logger_func("[bold green]✓ CUDA is available.[/]")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger_func(f"  - Device {i}: {device_name} (Total Memory: {total_mem:.2f} GB)")
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') and torch.version.cuda else "N/A"
        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available() else "N/A"
        logger_func(f"  - PyTorch CUDA version: {cuda_version}")
        logger_func(f"  - PyTorch cuDNN version: {cudnn_version}")
        torch.cuda.empty_cache()
        return torch.device("cuda")
    except Exception as e:
        logger_func = log.error if log else print
        logger_func(f"[bold red]CUDA initialization failed: {e}[/]")
        logger_func("Falling back to CPU. This will be very slow.")
        return torch.device("cpu")

# --- General Utility Functions ---
def to_mono(x):
    """Convert to mono"""
    if numpy is None:
        print("ERROR: numpy not imported yet in to_mono. Ensure _import_dependencies() was called.")
        raise RuntimeError("numpy not imported yet")
    return x.mean(axis=1).astype(numpy.float32) if x.ndim > 1 else x.astype(numpy.float32)

def cos(a, b) -> float:
    """Computes cosine similarity between two numpy arrays."""
    if numpy is None:
        print("ERROR: numpy not imported yet in cos. Ensure _import_dependencies() was called.")
        raise RuntimeError("numpy not imported yet")
    norm_a = numpy.linalg.norm(a)
    norm_b = numpy.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numpy.dot(a, b) / (norm_a * norm_b)


def ff_trim(src_path: Path, dst_path: Path, start_time: float, end_time: float, target_sr: int = 16000, target_ac: int = 1):
    if ffmpeg is None or log is None:
        print("ERROR: ffmpeg or log not initialized in ff_trim.")
        raise RuntimeError("ffmpeg or log not initialized")
    try:
        (
            ffmpeg.input(str(src_path), ss=start_time, to=end_time)
            .output(str(dst_path), acodec="pcm_s16le", ac=target_ac, ar=target_sr)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode('utf8', errors='ignore') if e.stderr else 'Unknown ffmpeg error during trim'
        log.error(f"ffmpeg trim failed for {dst_path.name}: {err_msg}")
        raise

def ff_slice(src_path: Path, dst_path: Path, start_time: float, end_time: float, target_ac: int = 1):
    if ffmpeg is None or log is None:
        print("ERROR: ffmpeg or log not initialized in ff_slice.")
        raise RuntimeError("ffmpeg or log not initialized")
    try:
        (
            ffmpeg.input(str(src_path), ss=start_time, to=end_time)
            .output(str(dst_path), acodec="pcm_s16le", ac=target_ac)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode('utf8', errors='ignore') if e.stderr else 'Unknown ffmpeg error during slice'
        log.error(f"ffmpeg slice failed for {dst_path.name}: {err_msg}")
        raise

def get_huggingface_token(token_arg: str = None) -> str:
    if log is None or console is None or Prompt is None or Confirm is None:
        print("ERROR: log, console, Prompt or Confirm not initialized in get_huggingface_token.")
        raise RuntimeError("log, console, Prompt or Confirm not initialized")
    if token_arg:
        log.info("Using HuggingFace token from command-line argument.")
        return token_arg
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        log.info("Using HuggingFace token from HUGGINGFACE_TOKEN environment variable.")
        return token
    console.print("\n[bold yellow]HuggingFace User Access Token is required for PyAnnote models.[/]")
    console.print("You can create a token at: [link=https://huggingface.co/settings/tokens]https://huggingface.co/settings/tokens[/link] (read permissions are sufficient).")
    try:
        token = Prompt.ask("Enter your HuggingFace token (will not be displayed)", password=True)
    except Exception:
        token = getpass.getpass("Enter your HuggingFace token (input hidden): ")
    if not token or not token.strip():
        log.error("[bold red]No HuggingFace token provided. Exiting.[/]")
        sys.exit(1)
    token = token.strip()
    log.info("HuggingFace token provided by user.")
    try:
        if Confirm.ask("Save this token as environment variable HUGGINGFACE_TOKEN for future use? (Recommended)", default=True):
            env_var_name = "HUGGINGFACE_TOKEN"
            if os.name == "nt":
                try:
                    subprocess.run(["setx", env_var_name, token], check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    console.print(f"[green]Token saved as User environment variable '{env_var_name}'. You may need to restart your terminal/IDE for it to take effect.[/green]")
                except FileNotFoundError:
                    console.print(f"[yellow]'setx' command not found. Please set the environment variable '{env_var_name}' manually.[/yellow]")
                except subprocess.CalledProcessError as e_setx:
                    console.print(f"[red]Failed to save token with 'setx': {e_setx.stderr.decode(errors='ignore') if e_setx.stderr else e_setx}. Please set it manually.[/red]")
            else:
                shell_name = Path(os.environ.get("SHELL", "")).name
                rc_files = {"bash": "~/.bashrc", "zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish"}
                shell_file_path_str = rc_files.get(shell_name)
                if not shell_file_path_str:
                    console.print(f"[yellow]Could not determine shell RC file for '{shell_name}'. Please add/update HUGGINGFACE_TOKEN in your shell's startup file manually.[/yellow]")
                else:
                    shell_file_path = Path(shell_file_path_str).expanduser()
                    try:
                        if shell_file_path.exists() and f'export {env_var_name}=' in shell_file_path.read_text(errors='ignore'):
                             console.print(f"[yellow]HUGGINGFACE_TOKEN already seems to be set in {shell_file_path}. Please update it manually if needed.[/yellow]")
                        else:
                            with open(shell_file_path, "a") as f:
                                f.write(f'\n# Added by Voice Extractor for HuggingFace Token\nexport {env_var_name}="{token}"\n')
                            console.print(f"[green]Token appended to {shell_file_path}. Please restart your terminal or run 'source {shell_file_path_str}' to apply.[/green]")
                    except Exception as e_rc:
                        console.print(f"[red]Failed to write to {shell_file_path}: {e_rc}. Please set HUGGINGFACE_TOKEN manually.[/red]")
    except Exception as e_prompt:
        log.warning(f"Could not prompt to save token due to an interactive console issue: {e_prompt}. Please set HUGGINGFACE_TOKEN environment variable manually if desired.")
    return token

def format_duration(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def safe_filename(name: str, max_length: int = 200) -> str:
    import re
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = name.replace(' ', '_')
    if len(name) > max_length:
        name = name[:max_length]
    return name if name else "unnamed_file"

def ensure_dir_exists(dir_path: Path):
    if log is None:
        print("ERROR: log not initialized in ensure_dir_exists.")
        print(f"Attempting to create directory {dir_path} without logger.")
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if log:
            log.error(f"Failed to create directory {dir_path}: {e}")
        else:
            print(f"Failed to create directory {dir_path}: {e}")
        raise

# --- Plotting Functions ---
def save_detailed_spectrograms(
    wav_path: Path, output_dir: Path, title_prefix: str, target_name: str = "TargetSpeaker",
    sample_sec: float = SPECTROGRAM_SEC, segments_to_mark: list = None, overlap_timeline = None
):
    if plt is None or librosa is None or numpy is None or log is None:
        print("ERROR: Plotting libraries (matplotlib, librosa), numpy or log not initialized in save_detailed_spectrograms.")
        return
    ensure_dir_exists(output_dir)
    safe_prefix = safe_filename(title_prefix)
    if not wav_path.exists():
        log.warning(f"Audio file {wav_path} not found for spectrogram '{safe_prefix}'. Skipping.")
        return
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True, duration=sample_sec)
    except Exception as e:
        log.error(f"Failed to load audio {wav_path} for spectrogram '{safe_prefix}': {e}")
        return
    if len(y) == 0:
        log.warning(f"Audio file {wav_path} is empty. Cannot generate spectrogram '{safe_prefix}'.")
        return

    duration = librosa.get_duration(y=y, sr=sr)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=SPEC_FIGSIZE)
    try:
        D = librosa.amplitude_to_db(numpy.abs(librosa.stft(y, n_fft=HIGH_RES_NFFT, hop_length=HIGH_RES_NFFT // 4)), ref=numpy.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', hop_length=HIGH_RES_NFFT // 4, cmap='magma', ax=ax)
        ax.set_ylim(0, FREQ_MAX)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        for low, high, label, color in SPEECH_BANDS:
            if high <= FREQ_MAX:
                ax.axhspan(low, high, color=color, alpha=0.15, ec='none')
                if duration > 0:
                    ax.text(duration * 0.02, (low + high) / 2, label, verticalalignment='center', fontsize=7, color='black', bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.2'))

        overlap_legend_added_spec = False
        if overlap_timeline:
            for segment in overlap_timeline:
                if segment.start > duration: continue
                plot_start, plot_end = max(0, segment.start), min(duration, segment.end)
                if plot_end > plot_start:
                    ax.axvspan(plot_start, plot_end, color='gray', alpha=0.4, label='Overlap Region' if not overlap_legend_added_spec else None)
                    if not overlap_legend_added_spec: overlap_legend_added_spec = True

        unique_labels_plotted_spec = set()
        if segments_to_mark:
            cmap_segments = plt.cm.get_cmap('viridis')
            for i, (start, end, label) in enumerate(segments_to_mark):
                if start >= duration or end <= 0 or start >= end: continue
                plot_start_seg, plot_end_seg = max(0, start), min(duration, end)
                if plot_end_seg <= plot_start_seg: continue

                is_target_segment_for_color = target_name.lower() in label.lower() if target_name and label else False
                color_val = 'orange' if is_target_segment_for_color else cmap_segments(i / len(segments_to_mark) if len(segments_to_mark) > 1 else 0.5)
                alpha_val = 0.5 if is_target_segment_for_color else 0.3
                label_for_legend = label if label not in unique_labels_plotted_spec else None
                ax.axvspan(plot_start_seg, plot_end_seg, color=color_val, alpha=alpha_val, label=label_for_legend)
                if label_for_legend: unique_labels_plotted_spec.add(label)
                ax.text(plot_start_seg + (plot_end_seg - plot_start_seg) / 2, FREQ_MAX * 0.95, label, horizontalalignment='center', color=color_val, fontsize=8, bbox=dict(facecolor='white', alpha=0.7, pad=0.2))

        plot_title = f"{target_name} - {title_prefix} (Sample: {duration:.1f}s, Max Freq: {FREQ_MAX/1000:.1f}kHz)"
        ax.set_title(plot_title, fontsize=12)
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Amplitude (dB)')
        if ax.has_data() and (unique_labels_plotted_spec or overlap_legend_added_spec): ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        out_path = output_dir / f"{safe_filename(target_name)}_{safe_prefix}_linear_hires.png"
        plt.savefig(out_path, dpi=SPEC_DPI)
        log.info(f"Saved detailed spectrogram: {out_path.name}")
    except Exception as e_spec:
        log.error(f"Error generating detailed spectrogram for {safe_prefix}: {e_spec}")
        if args_for_debug_plotting and args_for_debug_plotting.debug: log.exception("Traceback for detailed spectrogram error:")
    finally:
        plt.close(fig)


def create_comparison_spectrograms(
    files_to_compare: list, output_dir: Path, target_name: str,
    main_prefix: str = "Audio_Stages_Comparison", sample_sec: float = SPECTROGRAM_SEC,
    overlap_timeline_dict: dict = None
):
    if plt is None or librosa is None or numpy is None or log is None:
        print("ERROR: Plotting libraries (matplotlib, librosa), numpy or log not initialized in create_comparison_spectrograms.")
        return
    ensure_dir_exists(output_dir)
    if not files_to_compare: log.warning("No files provided for spectrogram comparison."); return

    valid_files = []
    for item in files_to_compare:
        if isinstance(item, tuple) and len(item) == 2:
            fp_candidate, title = item
            fp = Path(fp_candidate) if isinstance(fp_candidate, str) else fp_candidate
            if fp and fp.exists() and fp.stat().st_size > 0:
                valid_files.append((fp, title))
            else:
                log.warning(f"File '{fp}' for comparison spec not valid or empty. Skipping.")
        else:
            log.warning(f"Invalid item format for comparison spec: {item}. Expected (path, title). Skipping.")

    if not valid_files: log.warning("No valid files for spectrogram comparison after checking."); return

    n_files = len(valid_files)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs_flat = plt.subplots(n_files if n_files > 0 else 1, 1,
                                 figsize=(19.2, 4.0 * n_files if n_files > 0 else 4.0),
                                 sharex=True, sharey=True, layout="constrained", squeeze=False)
    axs = axs_flat.flatten()

    common_sr = None
    img_ref_for_colorbar = None

    for i, (file_path, title) in enumerate(valid_files):
        resolved_file_path_str = str(file_path.resolve())
        current_overlap_timeline = overlap_timeline_dict.get(resolved_file_path_str) if overlap_timeline_dict else None
        try:
            y, sr_current = librosa.load(file_path, sr=None, mono=True, duration=sample_sec)
            if common_sr is None: common_sr = sr_current
            elif common_sr != sr_current: y = librosa.resample(y, orig_sr=sr_current, target_sr=common_sr)

            if len(y) == 0:
                axs[i].set_title(f"{title} (Empty Audio)", fontsize=10); axs[i].text(0.5, 0.5, "Empty Audio", ha='center', va='center', transform=axs[i].transAxes); continue

            D = librosa.amplitude_to_db(numpy.abs(librosa.stft(y, n_fft=HIGH_RES_NFFT, hop_length=HIGH_RES_NFFT // 4)), ref=numpy.max)
            current_duration = librosa.get_duration(y=y, sr=common_sr)
            img = librosa.display.specshow(D, sr=common_sr, x_axis='time', y_axis='linear', hop_length=HIGH_RES_NFFT // 4, ax=axs[i], cmap='magma')
            if i == 0: img_ref_for_colorbar = img

            axs[i].set_title(f"{title} ({current_duration:.1f}s sample, {common_sr/1000:.1f}kHz)", fontsize=10)
            axs[i].set_ylabel("Frequency (Hz)"); axs[i].set_ylim(0, FREQ_MAX)
            for low, high, band_label, color in SPEECH_BANDS:
                if high <= FREQ_MAX: axs[i].axhspan(low, high, color=color, alpha=0.1)

            if current_overlap_timeline:
                overlap_legend_added_comp = False
                for segment in current_overlap_timeline:
                    if segment.start > current_duration: continue
                    plot_start_ov, plot_end_ov = max(0, segment.start), min(current_duration, segment.end)
                    if plot_end_ov > plot_start_ov:
                        axs[i].axvspan(plot_start_ov, plot_end_ov, color='dimgray', alpha=0.35, label='Overlap Region' if not overlap_legend_added_comp else None)
                        if not overlap_legend_added_comp: overlap_legend_added_comp = True
                if overlap_legend_added_comp and axs[i].has_data(): axs[i].legend(loc="upper right", fontsize=7)

        except Exception as e_comp_spec:
            log.error(f"Failed to process {file_path.name} for comparison spectrogram: {e_comp_spec}")
            axs[i].set_title(f"{title} (Error)", fontsize=10); axs[i].text(0.5, 0.5, "Error loading/processing", ha='center', va='center', transform=axs[i].transAxes, wrap=True)

    if n_files > 0: axs[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{target_name} - {main_prefix}", fontsize=14, y=1.02 if n_files > 1 else 1.05)
    if img_ref_for_colorbar and n_files > 0 :
        fig.colorbar(img_ref_for_colorbar, ax=axs.tolist(), format='%+2.0f dB', label='Amplitude (dB)', orientation='vertical', aspect=max(15, 30*n_files), pad=0.01)

    out_path = output_dir / f"{safe_filename(target_name)}_{safe_filename(main_prefix)}.png"
    try:
        plt.savefig(out_path, dpi=SPEC_DPI, bbox_inches='tight')
        log.info(f"Saved comparison spectrogram: {out_path.name}")
    except Exception as e_save_comp:
        log.error(f"Failed to save comparison spectrogram {out_path.name}: {e_save_comp}")
    finally:
        plt.close(fig)


def create_diarization_plot(
    annotation, target_speaker_label: str, target_name: str, output_dir: Path,
    plot_title_prefix: str = "Diarization_Results", overlap_timeline = None
):
    if plt is None or log is None:
        print("ERROR: Plotting library (matplotlib) or log not initialized in create_diarization_plot.")
        return
    ensure_dir_exists(output_dir)
    plt.style.use('seaborn-v0_8-darkgrid')

    num_labels_in_annotation = 0
    if annotation and hasattr(annotation, 'labels') and callable(annotation.labels):
        num_labels_in_annotation = len(annotation.labels())

    plot_height = max(4, num_labels_in_annotation * 0.8 if num_labels_in_annotation > 0 else 1 * 0.8)
    fig, ax = plt.subplots(figsize=(20, plot_height))

    speakers = list(annotation.labels()) if num_labels_in_annotation > 0 else []

    if not speakers:
        log.warning("No speaker labels found in annotation for diarization plot.")
        ax.text(0.5, 0.5, "No speaker segments found.", ha='center', va='center')
        ax.set_title(f"{target_name} - {plot_title_prefix} (No Speaker Data)", fontsize=12)
    else:
        sorted_speakers = sorted(speakers, key=lambda spk: (spk != target_speaker_label if target_speaker_label else True, spk))
        speaker_y_pos = {spk: i for i, spk in enumerate(sorted_speakers)}
        plot_colors = plt.cm.get_cmap('tab20')

        max_time = 0
        if annotation and hasattr(annotation, 'itertracks') and callable(annotation.itertracks):
            for segment_obj, _, _ in annotation.itertracks(yield_label=True):
                if segment_obj.end > max_time: max_time = segment_obj.end
        if overlap_timeline:
            for segment_obj in overlap_timeline:
                if segment_obj.end > max_time: max_time = segment_obj.end
        if max_time == 0 and annotation and hasattr(annotation, 'get_timeline') and callable(annotation.get_timeline):
             max_time = annotation.get_timeline().duration()


        unique_legend_labels_spk = set()
        for i, spk_label_from_list in enumerate(sorted_speakers):
            segments_for_this_speaker = []
            if annotation and hasattr(annotation, 'itertracks') and callable(annotation.itertracks):
                for segment, _, label_in_track in annotation.itertracks(yield_label=True):
                    if label_in_track == spk_label_from_list:
                        segments_for_this_speaker.append((segment.start, segment.end))

            is_target = (spk_label_from_list == target_speaker_label)
            bar_color = 'crimson' if is_target else plot_colors(i % plot_colors.N if plot_colors.N > 0 else 0)
            display_label_base = f"Target: {target_name} ({spk_label_from_list})" if is_target else f"Other Spk ({spk_label_from_list})"

            for seg_idx, (start, end) in enumerate(segments_for_this_speaker):
                legend_label_spk = display_label_base if display_label_base not in unique_legend_labels_spk else None
                ax.barh(y=speaker_y_pos[spk_label_from_list], width=end - start, left=start, height=0.7,
                        color=bar_color, alpha=0.8 if is_target else 0.6,
                        edgecolor='black' if is_target else bar_color, linewidth=0.5, label=legend_label_spk)
                if legend_label_spk: unique_legend_labels_spk.add(display_label_base)

        overlap_legend_added_plot = False
        if overlap_timeline:
            for seg_overlap in overlap_timeline:
                ax.axvspan(seg_overlap.start, seg_overlap.end, color='gray', alpha=0.3,
                           label='Overlapped Speech' if not overlap_legend_added_plot else None, zorder=0)
                if not overlap_legend_added_plot: overlap_legend_added_plot = True

        ax.set_yticks(list(speaker_y_pos.values()))
        ax.set_yticklabels([f"{target_name} ({spk})" if spk == target_speaker_label else f"Speaker {spk}" for spk in sorted_speakers])
        ax.set_xlabel("Time (seconds)"); ax.set_ylabel("Speaker")
        title_suffix = f"(Target Label: {target_speaker_label})" if target_speaker_label and target_speaker_label in speakers else "(Target Not Identified or Not in Plot)"
        ax.set_title(f"{target_name} - {plot_title_prefix} {title_suffix}", fontsize=12)
        ax.set_xlim(0, max_time * 1.02 if max_time > 0 else 10)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        if unique_legend_labels_spk or overlap_legend_added_plot: ax.legend(loc='upper right', fontsize=9)

    out_path = output_dir / f"{safe_filename(target_name)}_{safe_filename(plot_title_prefix)}.png"
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
        log.info(f"Saved diarization visualization: {out_path.name}")
    except Exception as e_diar_plot:
        log.error(f"Failed to save diarization plot {out_path.name}: {e_diar_plot}")
        if args_for_debug_plotting and args_for_debug_plotting.debug: log.exception("Traceback for diarization plot error:")
    finally:
        plt.close(fig)


def plot_verification_scores(
    scores_dict: dict, threshold: float, output_dir: Path, target_name: str,
    plot_title_prefix: str = "Verification_Scores"
):
    if plt is None or log is None:
        print("ERROR: Plotting library (matplotlib) or log not initialized in plot_verification_scores.")
        return 0, 0
    ensure_dir_exists(output_dir)
    if not scores_dict: log.warning("No verification scores provided to plot."); return 0, 0

    plt.style.use('seaborn-v0_8-darkgrid')
    fig_width = max(14, len(scores_dict) * 0.4 if len(scores_dict) < 50 else len(scores_dict) * 0.2)
    fig_height = 7
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sorted_scores_list = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

    segment_names = [Path(item[0]).name for item in sorted_scores_list]
    score_values = [item[1] for item in sorted_scores_list]

    bar_colors = ['mediumseagreen' if score >= threshold else 'lightcoral' for score in score_values]
    bars = ax.bar(range(len(segment_names)), score_values, color=bar_colors)
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, label=f'Verification Threshold ({threshold:.2f})')

    accepted_count = sum(1 for s_val in score_values if s_val >= threshold)
    rejected_count = len(score_values) - accepted_count

    for bar_idx, (bar, score_val) in enumerate(zip(bars, score_values)):
        yval = bar.get_height()
        text_rotation = 0
        if len(scores_dict) > 20: text_rotation = 90
        elif len(scores_dict) > 10: text_rotation = 45
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{score_val:.3f}',
                ha='center', va='bottom', fontsize=max(5, 9 - len(scores_dict)//10), rotation=text_rotation)

    ax.set_xticks(range(len(segment_names)))
    xtick_rotation = 45
    if len(scores_dict) > 15: xtick_rotation = 60
    if len(scores_dict) > 30: xtick_rotation = 90
    ax.set_xticklabels(segment_names, rotation=xtick_rotation, ha="right", fontsize=max(6, 10 - len(scores_dict)//8))

    ax.set_ylabel("Verification Score (Combined)", fontsize=10)
    ax.set_title(f"{target_name} - {plot_title_prefix} (Accepted: {accepted_count}, Rejected: {rejected_count})", fontsize=12)
    ax.set_ylim(0, 1.05); ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

    out_path = output_dir / f"{safe_filename(target_name)}_{safe_filename(plot_title_prefix)}.png"
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
        log.info(f"Saved verification scores plot: {out_path.name}")
    except Exception as e_ver_plot:
        log.error(f"Failed to save verification scores plot {out_path.name}: {e_ver_plot}")
        if args_for_debug_plotting and args_for_debug_plotting.debug: log.exception("Traceback for verification plot error:")
    finally:
        plt.close(fig)
    return accepted_count, rejected_count

args_for_debug_plotting = None
def set_args_for_debug(cli_args):
    global args_for_debug_plotting
    args_for_debug_plotting = cli_args

def download_with_progress(url, filepath):
    """Download file with progress bar"""
    logger_func = log.info if log else print
    logger_func(f"[Setup] Downloading: {Path(filepath).name}")
    
    try:
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('Content-Length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=Path(filepath).name) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        logger_func(f"[Setup] Download failed: {e}")
        raise

def detect_silence_boundaries(audio_file: Path, start_time: float, end_time: float, 
                             min_silence_duration: float = 0.1, 
                             silence_threshold: float = -40) -> list[float]:
    """
    Detect silence boundaries in an audio segment for intelligent cutting.
    
    Args:
        audio_file: Path to the audio file
        start_time: Start time of the segment to analyze
        end_time: End time of the segment to analyze
        min_silence_duration: Minimum duration of silence to consider as a boundary (in seconds)
        silence_threshold: Threshold in dB below which audio is considered silence
    
    Returns:
        List of time positions (relative to start_time) where silence occurs
    """
    if librosa is None or numpy is None:
        print("ERROR: librosa or numpy not imported yet in detect_silence_boundaries.")
        return []
    
    try:
        # Load only the segment we need to analyze
        segment_duration = end_time - start_time
        y, sr = librosa.load(audio_file, sr=16000, offset=start_time, duration=segment_duration)
        
        if len(y) == 0:
            return []
        
        # Calculate RMS energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
          # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=numpy.max(rms))
        
        # Find frames below silence threshold
        silent_frames = rms_db < silence_threshold
        
        # Convert frame indices to time
        times = librosa.frames_to_time(numpy.arange(len(rms_db)), sr=sr, hop_length=hop_length)
        
        # Find continuous silent regions
        silence_boundaries = []
        in_silence = False
        silence_start = 0
        
        for i, (time_pos, is_silent) in enumerate(zip(times, silent_frames)):
            if is_silent and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = time_pos
            elif not is_silent and in_silence:
                # End of silence
                in_silence = False
                silence_duration = time_pos - silence_start
                if silence_duration >= min_silence_duration:
                    # Use middle of silence region as boundary
                    boundary_time = silence_start + silence_duration / 2
                    silence_boundaries.append(boundary_time)
        
        # Handle case where segment ends in silence
        if in_silence:
            silence_duration = times[-1] - silence_start
            if silence_duration >= min_silence_duration:
                boundary_time = silence_start + silence_duration / 2
                silence_boundaries.append(boundary_time)
        
        return silence_boundaries
        
    except Exception as e:
        if log:
            log.warning(f"Could not detect silence boundaries in audio segment: {e}")
        return []


def calculate_intelligent_chunks(total_duration: float, silence_boundaries: list[float],
                                min_chunk_duration: float = 3.0, 
                                max_chunk_duration: float = 30.0) -> list[tuple[float, float]]:
    """
    Calculate intelligent chunk boundaries with varied lengths.
    
    Args:
        total_duration: Total duration of the segment to chunk
        silence_boundaries: List of silence boundary positions 
        min_chunk_duration: Minimum chunk duration
        max_chunk_duration: Maximum chunk duration
    
    Returns:
        List of (start_time, end_time) tuples for each chunk
    """
    if total_duration <= max_chunk_duration:
        return [(0.0, total_duration)]
    
    chunks = []
    current_start = 0.0
    
    # Add boundaries at start and end
    all_boundaries = [0.0] + silence_boundaries + [total_duration]
    all_boundaries = sorted(set(all_boundaries))  # Remove duplicates and sort
    
    while current_start < total_duration:
        remaining_duration = total_duration - current_start
        
        if remaining_duration <= max_chunk_duration:
            # Last chunk
            chunks.append((current_start, total_duration))
            break
        
        # Find potential end points within our preferred range
        target_min_end = current_start + min_chunk_duration
        target_max_end = current_start + max_chunk_duration
        
        # Find boundaries in our target range
        valid_boundaries = [b for b in all_boundaries 
                          if target_min_end <= b <= target_max_end]
        
        if valid_boundaries:
            # Choose a boundary, preferring those that create varied chunk sizes
            # Add some randomness to avoid uniform chunks
            if len(valid_boundaries) > 1:
                # Weight boundaries based on distance from edges of range
                weights = []
                for boundary in valid_boundaries:
                    # Prefer boundaries not too close to min or max
                    distance_from_min = boundary - target_min_end
                    distance_from_max = target_max_end - boundary
                    weight = min(distance_from_min, distance_from_max) + random.uniform(0.5, 1.5)
                    weights.append(weight)
                
                # Select weighted random boundary
                total_weight = sum(weights)
                r = random.uniform(0, total_weight)
                cumulative_weight = 0
                chosen_boundary = valid_boundaries[-1]
                
                for boundary, weight in zip(valid_boundaries, weights):
                    cumulative_weight += weight
                    if r <= cumulative_weight:
                        chosen_boundary = boundary
                        break
                
                chunk_end = chosen_boundary
            else:
                chunk_end = valid_boundaries[0]
        else:
            # No suitable boundary found, find closest boundary or use max duration
            future_boundaries = [b for b in all_boundaries if b > target_min_end]
            
            if future_boundaries:
                next_boundary = min(future_boundaries)
                if next_boundary - current_start <= max_chunk_duration * 1.2:  # Allow slight overage
                    chunk_end = next_boundary
                else:
                    chunk_end = current_start + max_chunk_duration
            else:
                chunk_end = current_start + max_chunk_duration
        
        chunks.append((current_start, chunk_end))
        current_start = chunk_end
    
    return chunks


def ff_slice_intelligent(src_path: Path, dst_dir: Path, segment_start_time: float, 
                        segment_end_time: float, base_name: str, target_sr: int = 16000, 
                        target_ac: int = 1, max_segment_duration: float = 30.0,
                        min_chunk_duration: float = 3.0) -> list[Path]:
    """
    Intelligently slice a long audio segment into smaller, meaningful chunks.
    
    Args:
        src_path: Source audio file
        dst_dir: Destination directory for chunks
        segment_start_time: Start time of the segment in the source file
        segment_end_time: End time of the segment in the source file
        base_name: Base name for output files
        target_sr: Target sample rate
        target_ac: Target audio channels
        max_segment_duration: Maximum duration for a single chunk
        min_chunk_duration: Minimum duration for a single chunk
    
    Returns:
        List of paths to created chunk files
    """
    if ffmpeg is None or log is None:
        print("ERROR: ffmpeg or log not initialized in ff_slice_intelligent.")
        raise RuntimeError("ffmpeg or log not initialized")
    
    segment_duration = segment_end_time - segment_start_time
    
    # If segment is not too long, use regular slicing
    if segment_duration <= max_segment_duration:
        dst_path = dst_dir / f"{base_name}.wav"
        ff_slice(src_path, dst_path, segment_start_time, segment_end_time, target_ac)
        return [dst_path]
    
    # Ensure destination directory exists
    ensure_dir_exists(dst_dir)
    
    log.info(f"Segment duration {segment_duration:.1f}s exceeds {max_segment_duration}s. "
             f"Intelligently splitting into smaller chunks...")
    
    # Detect silence boundaries for intelligent cutting
    silence_boundaries = detect_silence_boundaries(
        src_path, segment_start_time, segment_end_time,
        min_silence_duration=0.1, silence_threshold=-40
    )
    
    log.info(f"Found {len(silence_boundaries)} potential cut points based on silence detection.")
    
    # Calculate intelligent chunk boundaries
    chunk_boundaries = calculate_intelligent_chunks(
        segment_duration, silence_boundaries, min_chunk_duration, max_segment_duration
    )
    
    log.info(f"Split into {len(chunk_boundaries)} chunks with varied durations:")
    for i, (start, end) in enumerate(chunk_boundaries):
        duration = end - start
        log.info(f"  Chunk {i+1}: {duration:.1f}s")
    
    # Create the chunks
    created_files = []
    
    for i, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        # Calculate absolute times in the source file
        abs_start = segment_start_time + chunk_start
        abs_end = segment_start_time + chunk_end
        
        # Create filename for this chunk
        chunk_filename = f"{base_name}_chunk{i+1:02d}_{chunk_start:.1f}s_{chunk_end:.1f}s.wav"
        chunk_path = dst_dir / chunk_filename
        
        try:
            # Slice the chunk
            ff_slice(src_path, chunk_path, abs_start, abs_end, target_ac)
            
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                created_files.append(chunk_path)
                log.info(f"Created chunk: {chunk_filename}")
            else:
                log.warning(f"Failed to create chunk: {chunk_filename}")
                
        except Exception as e:
            log.error(f"Error creating chunk {chunk_filename}: {e}")
            continue
    
    if not created_files:
        log.warning("No chunks were successfully created. Falling back to regular slicing.")
        dst_path = dst_dir / f"{base_name}_fallback.wav"
        ff_slice(src_path, dst_path, segment_start_time, segment_end_time, target_ac)
        return [dst_path]
    
    log.info(f"Successfully created {len(created_files)} intelligent chunks.")
    return created_files

def ff_slice_smart(src_path: Path, dst_path: Path, start_time: float, end_time: float, 
                   target_sr: int = 16000, target_ac: int = 1, 
                   max_segment_duration: float = 30.0, min_chunk_duration: float = 3.0,
                   return_chunks: bool = False) -> Path | list[Path]:
    """
    Smart slice function that provides backward compatibility with ff_slice while adding intelligent chunking.
    
    For segments <= max_segment_duration: Creates a single file at dst_path (like original ff_slice)
    For segments > max_segment_duration: Creates intelligent chunks and either concatenates them or returns chunk list
    
    Args:
        src_path: Source audio file
        dst_path: Destination file path (single output file or base name for chunks)
        start_time: Start time of the segment
        end_time: End time of the segment
        target_sr: Target sample rate
        target_ac: Target audio channels
        max_segment_duration: Threshold for using intelligent chunking
        min_chunk_duration: Minimum chunk duration for intelligent chunking
        return_chunks: If True, returns list of chunk paths instead of concatenating them
    
    Returns:
        Path to the created file (if return_chunks=False) or list of chunk paths (if return_chunks=True)
    """
    if ffmpeg is None or log is None:
        print("ERROR: ffmpeg or log not initialized in ff_slice_smart.")
        raise RuntimeError("ffmpeg or log not initialized")
    
    segment_duration = end_time - start_time
    
    if segment_duration <= max_segment_duration:
        # Use regular ff_slice for short segments
        ff_slice(src_path, dst_path, start_time, end_time, target_ac)
        if return_chunks:
            return [dst_path]  # Return as list for consistency
        else:
            return dst_path
    
    # Use intelligent chunking for long segments
    log.info(f"Segment duration {segment_duration:.1f}s > {max_segment_duration}s. Using intelligent chunking.")
    
    if return_chunks:
        # Return individual chunks instead of concatenating
        temp_chunks_dir = dst_path.parent / f"temp_chunks_{dst_path.stem}"
        ensure_dir_exists(temp_chunks_dir)
        
        try:
            base_name = dst_path.stem
            chunk_files = ff_slice_intelligent(
                src_path=src_path,
                dst_dir=temp_chunks_dir,
                segment_start_time=start_time,
                segment_end_time=end_time,
                base_name=base_name,
                target_sr=target_sr,
                target_ac=target_ac,
                max_segment_duration=max_segment_duration,
                min_chunk_duration=min_chunk_duration
            )
            
            if not chunk_files:
                log.warning("No chunks created by intelligent chunking. Creating single fallback chunk.")
                ff_slice(src_path, dst_path, start_time, end_time, target_ac)
                return [dst_path]
            
            # Move chunks to final location with proper naming
            final_chunks = []
            for i, chunk_file in enumerate(chunk_files):
                final_chunk_name = f"{dst_path.stem}_chunk{i+1:02d}{dst_path.suffix}"
                final_chunk_path = dst_path.parent / final_chunk_name
                shutil.move(str(chunk_file), str(final_chunk_path))
                final_chunks.append(final_chunk_path)
            
            # Clean up temp directory
            if temp_chunks_dir.exists():
                shutil.rmtree(temp_chunks_dir, ignore_errors=True)
            
            log.info(f"Created {len(final_chunks)} chunks, returning individual chunk files for separate processing.")
            return final_chunks
            
        except Exception as e:
            log.error(f"Error in chunking mode: {e}. Falling back to single file.")
            ff_slice(src_path, dst_path, start_time, end_time, target_ac)
            return [dst_path]
    
    # Original concatenation behavior (return_chunks=False)
    temp_chunks_dir = dst_path.parent / f"temp_chunks_{dst_path.stem}"
    ensure_dir_exists(temp_chunks_dir)
    
    try:
        # Create intelligent chunks
        base_name = dst_path.stem
        chunk_files = ff_slice_intelligent(
            src_path=src_path,
            dst_dir=temp_chunks_dir,
            segment_start_time=start_time,
            segment_end_time=end_time,
            base_name=base_name,
            target_sr=target_sr,
            target_ac=target_ac,
            max_segment_duration=max_segment_duration,
            min_chunk_duration=min_chunk_duration
        )
        
        if not chunk_files:
            log.warning("No chunks created by intelligent chunking. Falling back to regular ff_slice.")
            ff_slice(src_path, dst_path, start_time, end_time, target_ac)
            return dst_path
        
        if len(chunk_files) == 1:
            # Only one chunk, just move it to the final destination
            chunk_files[0].rename(dst_path)
            log.info(f"Single chunk created, moved to {dst_path.name}")
        else:
            # Multiple chunks, concatenate them
            log.info(f"Concatenating {len(chunk_files)} chunks to create final segment.")
            
            # Create concatenation list file
            concat_list_file = temp_chunks_dir / "concat_list.txt"
            concat_lines = []
            
            for chunk_file in chunk_files:
                if chunk_file.exists() and chunk_file.stat().st_size > 0:
                    concat_lines.append(f"file '{chunk_file.resolve().as_posix()}'")
            
            if not concat_lines:
                raise RuntimeError("No valid chunks found for concatenation")
            
            concat_list_file.write_text("\n".join(concat_lines), encoding="utf-8")
            
            # Use ffmpeg to concatenate chunks
            try:
                (
                    ffmpeg.input(str(concat_list_file), format="concat", safe=0)
                    .output(
                        str(dst_path),
                        acodec="pcm_s16le",
                        ar=target_sr,
                        ac=target_ac,
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                log.info(f"Successfully concatenated chunks to {dst_path.name}")
            except ffmpeg.Error as e:
                err_msg = e.stderr.decode('utf8', errors='ignore') if e.stderr else 'ffmpeg concatenation error'
                log.error(f"ffmpeg concatenation failed: {err_msg}")
                # Fallback to regular slicing
                log.info("Falling back to regular ff_slice due to concatenation error.")
                ff_slice(src_path, dst_path, start_time, end_time, target_ac)
        
        return dst_path
        
    finally:
        # Clean up temporary chunks directory
        if temp_chunks_dir.exists():
            try:
                shutil.rmtree(temp_chunks_dir)
                log.debug(f"Cleaned up temporary chunks directory: {temp_chunks_dir}")
            except Exception as e:
                log.warning(f"Could not clean up temporary directory {temp_chunks_dir}: {e}")
