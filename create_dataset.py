#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_dataset.py
Creates a dataset from verified audio segments and their transcripts.
Saves the dataset in Hugging Face format with proper audio column casting.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import shutil

try:
    from datasets import Dataset, Audio
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False
    Dataset = None
    Audio = None

from common import ensure_dir_exists

# Set up logging
import logging
try:
    from common import log
    if log is None:
        raise AttributeError("Log is None")
except (ImportError, AttributeError):
    # Fallback logging setup
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    log = logging.getLogger(__name__)


def create_audio_text_dataset(
    verified_segments_dir: Path,
    transcripts_dir: Path,
    output_dataset_dir: Path,
    dataset_name: str = "voice_extractor_dataset",
    copy_audio_files: bool = False,
) -> bool:
    """
    Creates a Hugging Face dataset from verified audio segments and their transcripts.
    
    Args:
        verified_segments_dir: Directory containing verified audio segments
        transcripts_dir: Directory containing transcript files
        output_dataset_dir: Directory to save the dataset
        dataset_name: Name for the dataset
        copy_audio_files: Whether to copy audio files to dataset directory
        
    Returns:
        True if dataset creation was successful, False otherwise
    """
    
    if not HAVE_DATASETS:
        log.error("The 'datasets' library is not installed. Please install it with: pip install datasets")
        return False
    
    log.info(f"Creating dataset '{dataset_name}' from verified segments...")
    
    # Ensure output directory exists
    ensure_dir_exists(output_dataset_dir)
    
    # Find all verified audio segments
    audio_files = list(verified_segments_dir.glob("*.wav"))
    if not audio_files:
        log.warning(f"No audio files found in {verified_segments_dir}")
        return False
    
    log.info(f"Found {len(audio_files)} verified audio segments")
    
    # Collect audio paths and transcripts
    audio_paths = []
    transcripts = []
    
    # Create audio subdirectory in dataset if copying files
    if copy_audio_files:
        dataset_audio_dir = output_dataset_dir / "audio"
        ensure_dir_exists(dataset_audio_dir)
    
    for audio_file in audio_files:        # Find corresponding transcript
        transcript_file = transcripts_dir / f"{audio_file.stem}_transcript.txt"
        
        if not transcript_file.exists():
            log.warning(f"No transcript found for {audio_file.name}, skipping...")
            continue
        
        # Read transcript
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read().strip()
        except Exception as e:
            log.error(f"Failed to read transcript {transcript_file.name}: {e}")
            continue
        
        if not transcript_text:
            log.warning(f"Empty transcript for {audio_file.name}, skipping...")
            continue
        
        # Handle audio file path
        if copy_audio_files:
            # Copy audio file to dataset directory
            copied_audio_path = dataset_audio_dir / audio_file.name
            try:
                shutil.copy2(str(audio_file), str(copied_audio_path))
                final_audio_path = str(copied_audio_path)
            except Exception as e:
                log.error(f"Failed to copy {audio_file.name}: {e}")
                continue
        else:
            # Use original path
            final_audio_path = str(audio_file.resolve())
        
        audio_paths.append(final_audio_path)
        transcripts.append(transcript_text)
        
        log.debug(f"Added: {audio_file.name} -> '{transcript_text[:50]}...'")
    
    if not audio_paths:
        log.error("No valid audio-transcript pairs found")
        return False
    
    log.info(f"Successfully collected {len(audio_paths)} audio-transcript pairs")
    
    # Create dataset
    try:
        log.info("Creating Hugging Face dataset...")
        
        # Create dataset from dictionary
        dataset_dict = {
            "audio": audio_paths,
            "text": transcripts,
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Cast audio column to Audio type with specified sampling rate
        dataset = dataset.cast_column("audio", Audio())
        
        # Save dataset
        dataset_path = output_dataset_dir / dataset_name
        log.info(f"Saving dataset to: {dataset_path}")
        dataset.save_to_disk(str(dataset_path))
        
        # Save dataset info
        dataset_info = {
            "name": dataset_name,
            "num_samples": len(audio_paths),
            "created_from": {
                "verified_segments_dir": str(verified_segments_dir),
                "transcripts_dir": str(transcripts_dir),
            },
            "columns": ["audio", "text"],
            "audio_files_copied": copy_audio_files,
        }
        
        info_file = output_dataset_dir / f"{dataset_name}_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        log.info(f"[green]✓ Dataset created successfully![/]")
        log.info(f"  - Dataset path: {dataset_path}")
        log.info(f"  - Number of samples: {len(audio_paths)}")
        log.info(f"  - Info file: {info_file}")
        
        # Test loading a sample
        log.info("Testing dataset loading...")
        loaded_dataset = Dataset.load_from_disk(str(dataset_path))
        sample = loaded_dataset[0]
        
        log.info(f"Sample audio info:")
        log.info(f"  - Path: {sample['audio']['path']}")
        log.info(f"  - Sampling rate: {sample['audio']['sampling_rate']}")
        log.info(f"  - Array shape: {sample['audio']['array'].shape}")
        log.info(f"  - Text: '{sample['text'][:100]}...'")
        
        return True
        
    except Exception as e:
        log.error(f"Failed to create dataset: {e}")
        return False


def load_dataset_from_disk(dataset_path: Path) -> Optional[Dataset]:
    """
    Loads a previously saved dataset from disk.
    
    Args:
        dataset_path: Path to the saved dataset directory
        
    Returns:
        Loaded dataset or None if failed
    """
    
    if not HAVE_DATASETS:
        log.error("The 'datasets' library is not installed. Please install it with: pip install datasets")
        return None
    
    try:
        log.info(f"Loading dataset from: {dataset_path}")
        dataset = Dataset.load_from_disk(str(dataset_path))
        log.info(f"Successfully loaded dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        return None


def create_dataset_from_run_output(
    run_output_dir: Path,
    dataset_output_dir: Path,
    dataset_name: Optional[str] = None,
    copy_audio_files: bool = False,
) -> bool:
    """
    Creates a dataset from the output of a Voice Extractor run.
    
    Args:
        run_output_dir: Directory containing Voice Extractor run output
        dataset_output_dir: Directory to save the dataset
        dataset_name: Name for the dataset (auto-generated if None)
        copy_audio_files: Whether to copy audio files to dataset directory
        
    Returns:
        True if dataset creation was successful, False otherwise
    """
    
    # Auto-generate dataset name if not provided
    if dataset_name is None:
        dataset_name = f"voice_extractor_dataset_{run_output_dir.name}"
    
    # Find verified segments and transcripts directories
    verified_segments_dir = run_output_dir / "target_segments_solo"
    transcripts_dir = run_output_dir / "transcripts_solo_verified_whisper"
    
    if not verified_segments_dir.exists():
        log.error(f"Verified segments directory not found: {verified_segments_dir}")
        return False
    
    if not transcripts_dir.exists():
        log.error(f"Transcripts directory not found: {transcripts_dir}")
        return False
    
    return create_audio_text_dataset(
        verified_segments_dir=verified_segments_dir,
        transcripts_dir=transcripts_dir,
        output_dataset_dir=dataset_output_dir,
        dataset_name=dataset_name,
        copy_audio_files=copy_audio_files,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dataset from Voice Extractor output")
    parser.add_argument("--run-output-dir", type=Path, required=True,
                       help="Directory containing Voice Extractor run output")
    parser.add_argument("--dataset-output-dir", type=Path, required=True,
                       help="Directory to save the dataset")
    parser.add_argument("--dataset-name", type=str,
                       help="Name for the dataset (auto-generated if not provided)")

    parser.add_argument("--no-copy", action="store_true",
                       help="Don't copy audio files, use original paths")
    
    args = parser.parse_args()
    
    success = create_dataset_from_run_output(
        run_output_dir=args.run_output_dir,
        dataset_output_dir=args.dataset_output_dir,
        dataset_name=args.dataset_name,
        copy_audio_files=not args.no_copy,
    )
    
    if success:
        print("Dataset creation completed successfully!")
    else:
        print("Dataset creation failed!")
        exit(1)
