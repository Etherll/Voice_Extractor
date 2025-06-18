# Voice_Extractor

A tool for identifying, isolating, and transcribing clean solo segments of a target speaker from multi-speaker audio recordings. Processes audio to extract only non-overlapped speech of a specific person using modern AI techniques.

## Google Colab Version With a GUI:

[colab.research.google.com/github/ReisCook/Voice_Extractor_Colab/blob/main/Voice_Extractor_Colab.ipynb
](https://colab.research.google.com/github/ReisCook/Voice_Extractor_Colab/blob/main/Voice_Extractor_Colab.ipynb)


## Features
- **Speaker Diarization**: Identifies who is speaking when using PyAnnote
- **Overlap Detection**: Finds and removes segments with multiple speakers  
- **Target Identification**: Matches speakers to a reference sample using WeSpeaker Deep r-vector
- **Vocal Separation**: Optional Bandit-v2 cinematic audio source separation to isolate speech from music/effects
- **Speaker Verification**: Multi-stage verification using WeSpeaker and SpeechBrain models
- **Transcription**: OpenAI Whisper for accurate speech-to-text
- **Visualization**: Spectrograms, diarization plots, and verification score charts

## Tech Stack
- **AI Models**: Bandit-v2 (vocal separation), PyAnnote (diarization/overlap detection), WeSpeaker (speaker identification), SpeechBrain ECAPA-TDNN (verification), Silero-VAD (voice activity), OpenAI Whisper (transcription)
- **Libraries & Frameworks**: PyTorch, torchaudio, torchvision, librosa, ray, asteroid, ffmpeg-python
- **Output**: High-quality verified WAV segments, transcripts (CSV/TXT), spectrograms

## Min Specs:

NVIDIA GPU with 16GB VRAM

## Installation



Install all required dependencies:        

Python 3.10

FFmpeg

pip install -r requirements.txt

You'll need a Hugging Face access token which you can create at: https://huggingface.co/settings/tokens

Request access to the following gated repos:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/overlapped-speech-detection
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/segmentation

# Base Command (required arguments only)
python run_extractor.py \
    --input-audio "path/to/input_audio.wav" \
    --reference-audio "path/to/target_sample.wav" \
    --target-name "TargetName" \
    --token "hf_YourHuggingFaceToken"

## Optional Arguments (add as needed)
    --output-base-dir "path/to/output"        # Output directory (default: ./output_runs)
    --output-sr 44100                         # Output sample rate in Hz (default: 44100)
    --bandit-repo-path "repos/bandit-v2"     # Path to Bandit-v2 repository
    --bandit-model-path "models/checkpoint.ckpt"  # Path to Bandit checkpoint file
    --wespeaker-rvector-model "english"       # WeSpeaker model for speaker ID (english/chinese)
    --wespeaker-gemini-model "english"        # WeSpeaker model for verification
    --osd-model "pyannote/overlapped-speech-detection"  # Overlap detection model
    --whisper-model "large-v3"                # Transcription model (default: large-v3)
    --language "en"                           # Language for transcription (default: en)
    --min-duration 1.0                        # Minimum segment duration in seconds
    --merge-gap 0.25                          # Maximum gap between segments to merge
    --verification-threshold 0.7              # Speaker verification strictness (0-1)
    --concat-silence 0.25                     # Silence between segments in output
    --preload-whisper                         # Pre-load Whisper model at startup
    --dry-run                                 # Process only first minute (testing)
    --debug                                   # Enable verbose logging
    --skip-bandit                             # Skip vocal separation stage
    --disable-speechbrain                     # Disable SpeechBrain verification
    --skip-rejected-transcripts               # Don't transcribe rejected segments
    --keep-temp-files                         # Keep temporary processing files


# Issues & Bug Reports
If you encounter any problems or have suggestions for improvements:
- Open an issue on GitHub
- Email: reiscook@gmail.com
- This program contains zero telemetry - your feedback helps make it better for everyone

