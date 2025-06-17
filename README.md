# Voice_Extractor

A comprehensive AI-powered tool for identifying, isolating, and transcribing clean solo segments of a target speaker from multi-speaker audio recordings. Features advanced audio processing, speaker verification, and automatic dataset creation for machine learning applications.

## Google Colab Version With a GUI:

[colab.research.google.com/github/ReisCook/Voice_Extractor_Colab/blob/main/Voice_Extractor_Colab.ipynb
](https://colab.research.google.com/github/ReisCook/Voice_Extractor_Colab/blob/main/Voice_Extractor_Colab.ipynb)


## Features
- **Speaker Diarization**: Identifies who is speaking when using PyAnnote 3.1
- **Overlap Detection**: Finds and removes segments with multiple speakers  
- **Target Identification**: Matches speakers to a reference sample using WeSpeaker Deep r-vector
- **Vocal Separation**: Audio Separator (Mel-Roformer) for isolating speech from music/effects
- **Classify & Clean Workflow**: Optional noise classification and selective cleaning of segments
- **Speaker Verification**: Multi-stage verification using WeSpeaker and SpeechBrain models
- **Transcription**: Choose between NVIDIA NeMo ASR (default) or OpenAI Whisper for accurate speech-to-text
- **Dataset Creation**: Automatic Hugging Face dataset generation for ML training
- **Intelligent Chunking**: Handles long segments with smart chunking
- **Visualization**: Spectrograms, diarization plots, and verification score charts

## Tech Stack
- **AI Models**: Audio Separator Mel-Roformer (vocal separation), PyAnnote 3.1 (diarization/overlap detection), WeSpeaker (speaker identification), SpeechBrain ECAPA-TDNN (verification), Silero-VAD (voice activity), NVIDIA NeMo ASR & OpenAI Whisper (transcription), NoisySpeechDetection (audio quality classification)
- **Libraries & Frameworks**: PyTorch, torchaudio, torchvision, librosa, ray, asteroid, ffmpeg-python, nemo_toolkit, datasets, audio-separator
- **Output**: High-quality verified WAV segments, transcripts (TXT), ML-ready datasets, spectrograms

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

## ASR Engine Options

**NeMo ASR (Default)**
- Uses NVIDIA's Parakeet TDT model (nvidia/parakeet-tdt-0.6b-v2)
- Optimized for high-quality transcription
- Automatically downloads from Hugging Face Hub (~600MB)
- Requires mono audio input (automatically converted)

**Whisper ASR (Alternative)**
- Uses OpenAI's Whisper large-v3 model
- Multi-language support with language detection
- Use `--asr-engine whisper` to enable

Both engines save transcripts as `{segment_name}_transcript.txt`

## Processing Workflows

**Standard Workflow** (Default)
1. Audio Separator vocal separation on entire input
2. Diarization and overlap detection on separated vocals
3. Target speaker identification and verification
4. Transcription of verified segments

**Classify & Clean Workflow** (`--classify-and-clean`)
1. Diarization and overlap detection on original audio
2. Target speaker identification on original audio
3. Slice segments from original audio
4. Classify segments as clean/noisy using NoisySpeechDetection
5. Apply Audio Separator only to noisy segments
6. Verify all segments (both clean originals and cleaned noisy)
7. Transcription of verified segments

## Dataset Creation (Stage 10)

Automatically creates ML-ready Hugging Face datasets:
- **Audio files**: Verified speaker segments
- **Transcripts**: Corresponding text transcriptions  
- **Metadata**: Speaker info, verification scores, audio properties
- **Format**: Compatible with Hugging Face `datasets` library
- **Output**: Ready for TTS, ASR, or voice cloning training

Use `--skip-dataset-creation` to disable this feature.

# Base Command (required arguments only)
python run_extractor.py \
    --input-audio "path/to/input_audio.wav" \
    --reference-audio "path/to/target_sample.wav" \
    --target-name "TargetName" \
    --token "hf_YourHuggingFaceToken"

## Example Commands

**Standard Processing with NeMo ASR (Default)**
```bash
python run_extractor.py \
    --input-audio "interview.wav" \
    --reference-audio "speaker_sample.wav" \
    --target-name "JohnDoe" \
    --token "hf_your_token_here"
```

**Using Whisper ASR Instead**
```bash
python run_extractor.py \
    --input-audio "interview.wav" \
    --reference-audio "speaker_sample.wav" \
    --target-name "JohnDoe" \
    --asr-engine whisper \
    --whisper-model large-v3 \
    --token "hf_your_token_here"
```

**Classify & Clean Workflow (Recommended for Noisy Audio)**
```bash
python run_extractor.py \
    --input-audio "noisy_podcast.wav" \
    --reference-audio "speaker_sample.wav" \
    --target-name "Host" \
    --classify-and-clean \
    --noise-threshold 0.8 \
    --token "hf_your_token_here"
```

**High-Performance Setup with Model Preloading**
```bash
python run_extractor.py \
    --input-audio "long_interview.wav" \
    --reference-audio "speaker_sample.wav" \
    --target-name "Expert" \
    --preload-nemo \
    --max-segment-duration 15.0 \
    --token "hf_your_token_here"
```

**Dataset Creation for ML Training**
```bash
python run_extractor.py \
    --input-audio "training_audio.wav" \
    --reference-audio "target_voice.wav" \
    --target-name "TrainingVoice" \
    --dataset-name "my_tts_dataset" \
    --dataset-sampling-rate 22050 \
    --dataset-copy \
    --token "hf_your_token_here"
```

## Optional Arguments (add as needed)
    --output-base-dir "path/to/output"        # Output directory (default: ./output_runs)
    --output-sr 44100                         # Output sample rate in Hz (default: 44100)
    --audio-separator-model "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"  # Audio Separator model
    --wespeaker-rvector-model "english"       # WeSpeaker model for speaker ID (english/chinese)
    --wespeaker-gemini-model "english"        # WeSpeaker model for verification
    --diar-model "pyannote/speaker-diarization-3.1"  # Diarization model (default: v3.1)
    --osd-model "pyannote/overlapped-speech-detection"  # Overlap detection model
    --asr-engine "nemo"                       # ASR engine: nemo (default) or whisper
    --nemo-asr-model "nvidia/parakeet-tdt-0.6b-v2"  # NeMo ASR model (default)
    --whisper-model "large-v3"                # Whisper model if using --asr-engine whisper
    --language "en"                           # Language for Whisper transcription (default: en)
    --diar-hyperparams "{}"                   # JSON hyperparameters for diarization
    --min-duration 1.0                        # Minimum segment duration in seconds
    --merge-gap 0.25                          # Maximum gap between segments to merge
    --verification-threshold 0.7              # Speaker verification strictness (0-1)
    --noise-threshold 0.7                     # Noise classification threshold (0-1)
    --max-segment-duration 30.0               # Max segment duration before chunking
    --concat-silence 0.25                     # Silence between segments in output
    --preload-whisper                         # Pre-load Whisper model at startup
    --preload-nemo                            # Pre-load NeMo ASR model at startup
    --classify-and-clean                      # Enable classify & clean workflow
    --skip-audio-separator                    # Skip Audio Separator vocal separation
    --disable-speechbrain                     # Disable SpeechBrain verification
    --skip-rejected-transcripts               # Don't transcribe rejected segments
    --skip-dataset-creation                   # Skip Hugging Face dataset creation (Stage 10)
    --dataset-output-dir "path/to/dataset"    # Dataset output directory
    --dataset-name "my_dataset"               # Custom dataset name
    --dataset-sampling-rate 16000             # Dataset audio sampling rate (default: 16kHz)
    --dataset-copy                            # Copy audio files to dataset directory
    --dry-run                                 # Process only first minute (testing)
    --debug                                   # Enable verbose logging
    --keep-temp-files                         # Keep temporary processing files


# Issues & Bug Reports
If you encounter any problems or have suggestions for improvements:
- Open an issue on GitHub
- Email: reiscook@gmail.com
- This program contains zero telemetry - your feedback helps make it better for everyone

