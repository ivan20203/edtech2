#!/usr/bin/env python3
"""
generate_audio_from_mooncast.py
===============================
Generate audio from MoonCast semantic tokens using the exact same pipeline as MoonCast.

This script:
1. Loads semantic token files from data/train/
2. Uses MoonCast's exact working pipeline to generate audio
3. Saves audio files for listening and analysis
"""

import sys
import os
import torch
import numpy as np
import librosa
import torchaudio
import io
import base64
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
import argparse
import time
from typing import List, Tuple

# Add MoonCast to path for imports
mooncast_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "MoonCast")
sys.path.append(mooncast_path)

# Import MoonCast's exact components
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_detokenizer.audio_detokenizer import get_audio_detokenizer, detokenize_noref
from transformers import AutoModelForCausalLM, GenerationConfig


class MoonCastPipeline:
    """
    MoonCast's exact pipeline: Text → Semantic Tokens → Audio → Mel-Spectrograms
    """
    
    def __init__(self):
        """Initialize MoonCast's complete pipeline."""
        print("Initializing MoonCast Pipeline...")
        
        # Store original working directory
        original_cwd = os.getcwd()
        
        try:
            # Change to MoonCast directory for initialization (so relative paths work)
            os.chdir(mooncast_path)
            print(f"Changed working directory to: {mooncast_path}")
            
            # Initialize tokenizer (same as MoonCast)
            self.tokenizer, self.extra_tokens = get_tokenizer_and_extra_tokens()
            self.speech_token_offset = 163840
            
            # Special token IDs (same as MoonCast)
            self.assistant_ids = self.tokenizer.encode("assistant")
            self.user_ids = self.tokenizer.encode("user")
            self.audio_ids = self.tokenizer.encode("audio")
            self.spk_0_ids = self.tokenizer.encode("0")
            self.spk_1_ids = self.tokenizer.encode("1")
            
            # Extra tokens (same as MoonCast)
            self.msg_end = self.extra_tokens.msg_end
            self.user_msg_start = self.extra_tokens.user_msg_start
            self.assistant_msg_start = self.extra_tokens.assistant_msg_start
            self.name_end = self.extra_tokens.name_end
            self.media_begin = self.extra_tokens.media_begin
            self.media_content = self.extra_tokens.media_content
            self.media_end = self.extra_tokens.media_end
            
            # Load text2semantic model (same as MoonCast)
            model_path = "resources/text2semantic"
            model_path = os.path.abspath(model_path)
            
            print(f"Loading text2semantic model from: {model_path}")
            
            # Check if model path exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="cuda:0", 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True, 
                force_download=False
            ).to(torch.cuda.current_device())
            
            # Generation config (same as MoonCast)
            self.generate_config = GenerationConfig(
                max_new_tokens=200 * 50,
                do_sample=True,
                top_k=30,
                top_p=0.8,
                temperature=0.8,
                eos_token_id=self.media_end,
            )
            
            # Load detokenizer (same as MoonCast)
            print("Loading MoonCast detokenizer...")
            self.audio_detokenizer = get_audio_detokenizer()
            
            print("✅ MoonCast Pipeline initialized")
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            print(f"Restored working directory to: {original_cwd}")
    
    def semantic_tokens_to_audio(self, tokens: np.ndarray) -> torch.Tensor:
        """
        Convert semantic tokens to audio using MoonCast's exact pipeline.
        
        Args:
            tokens: Semantic tokens as numpy array (already in correct range [0, 8191])
            
        Returns:
            Audio tensor
        """
        # Convert to tensor format expected by MoonCast
        if isinstance(tokens, np.ndarray):
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.model.device)
        
        # MoonCast expects tokens in format (batch_size, sequence_length)
        # Our semantic tokens are 1D, so we need to add batch dimension
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)  # Add batch dimension
        
        # The tokens are already in the correct range for the audio detokenizer
        # (MoonCast's audio tokenizer produces tokens in range [0, 8191])
        # We don't need to add the speech_token_offset since these are direct semantic tokens
        torch_token = tokens
        
        # Generate audio using MoonCast's detokenizer (exact same as MoonCast)
        gen_speech_fm = detokenize_noref(self.audio_detokenizer, torch_token)
        gen_speech_fm = gen_speech_fm.cpu()
        gen_speech_fm = gen_speech_fm / gen_speech_fm.abs().max()
        
        return gen_speech_fm
    
    def audio_to_mel(self, audio, sample_rate=24000, n_mels=128, n_fft=1024, hop_length=256):
        """
        Convert audio tensor to mel-spectrogram.
        
        Args:
            audio: Audio tensor (1, samples)
            sample_rate: Audio sample rate
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            
        Returns:
            mel_spectrogram: Mel-spectrogram as numpy array (n_mels, time_frames)
        """
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Extract mel-spectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann'
        )
        
        # Convert to log scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return mel_spectrogram


def analyze_semantic_tokens(tokens: np.ndarray, name: str = "Semantic tokens") -> dict:
    """Analyze semantic token statistics."""
    print(f"\n{name} Analysis:")
    print(f"  Shape: {tokens.shape}")
    print(f"  Data type: {tokens.dtype}")
    print(f"  Value range: {tokens.min()} to {tokens.max()}")
    print(f"  Mean: {tokens.mean():.2f}")
    print(f"  Unique values: {len(np.unique(tokens))}")
    
    # Check for common values
    unique_vals, counts = np.unique(tokens, return_counts=True)
    print(f"  Most common values:")
    for val, count in sorted(zip(unique_vals, counts), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {val}: {count} times ({count/tokens.size*100:.1f}%)")
    
    return {
        'shape': tokens.shape,
        'min': tokens.min(),
        'max': tokens.max(),
        'mean': tokens.mean(),
        'unique_count': len(unique_vals)
    }


def generate_audio_from_tokens(pipeline: MoonCastPipeline, tokens: np.ndarray, output_path: str) -> Tuple[np.ndarray, dict]:
    """Generate audio from semantic tokens using MoonCast's exact pipeline."""
    try:
        print(f"Generating audio to {output_path}...")
        
        # Convert semantic tokens to audio using MoonCast's exact pipeline
        print("  Converting semantic tokens to audio...")
        audio_tensor = pipeline.semantic_tokens_to_audio(tokens)
        
        # Convert to numpy
        audio_np = audio_tensor.numpy()
        
        # Check audio properties
        rms = np.sqrt(np.mean(audio_np**2))
        duration = len(audio_np) / 24000  # MoonCast uses 24kHz
        peak = np.max(np.abs(audio_np))
        
        print(f"  Audio shape: {audio_np.shape}")
        print(f"  Audio range: {audio_np.min():.4f} to {audio_np.max():.4f}")
        print(f"  Audio RMS: {rms:.4f}")
        print(f"  Audio peak: {peak:.4f}")
        print(f"  Audio duration: {duration:.2f} seconds")
        
        # Save audio using torchaudio (more reliable than soundfile)
        audio_tensor_for_save = torch.tensor(audio_np, dtype=torch.float32)
        if audio_tensor_for_save.dim() == 1:
            audio_tensor_for_save = audio_tensor_for_save.unsqueeze(0)  # Add channel dimension
        
        torchaudio.save(output_path, audio_tensor_for_save, 24000)  # MoonCast uses 24kHz
        
        print(f"  ✅ Saved to {output_path}")
        
        return audio_np, {
            'rms': rms,
            'peak': peak,
            'duration': duration,
            'has_voice': rms > 0.01,
            'sample_rate': 24000
        }
        
    except Exception as e:
        print(f"  ❌ Error generating audio: {e}")
        return None, {'error': str(e)}


def process_semantic_file(mc_path: Path, output_dir: Path, pipeline: MoonCastPipeline, analyze: bool = True) -> dict:
    """Process a single semantic token file."""
    print(f"\n{'='*60}")
    print(f"Processing: {mc_path.name}")
    print(f"{'='*60}")
    
    # Load semantic tokens
    try:
        tokens = np.load(mc_path)
    except Exception as e:
        print(f"❌ Error loading {mc_path.name}: {e}")
        return {'error': str(e)}
    
    # Analyze tokens
    if analyze:
        analysis = analyze_semantic_tokens(tokens, f"Semantic tokens from {mc_path.name}")
    
    # Generate output path
    stem = mc_path.stem
    output_path = output_dir / f"{stem}_mooncast_audio.wav"
    
    # Generate audio
    audio, audio_stats = generate_audio_from_tokens(pipeline, tokens, str(output_path))
    
    if audio is not None:
        print(f"✅ Successfully processed: {audio_stats['duration']:.2f}s")
        if audio_stats.get('has_voice', False):
            print("🎤 Audio contains voice!")
        else:
            print("🔇 Audio is silent or very quiet")
    
    return {
        'file': mc_path.name,
        'token_analysis': analysis if analyze else None,
        'audio_stats': audio_stats,
        'output_path': str(output_path) if audio is not None else None
    }


def main():
    parser = argparse.ArgumentParser(description="Generate audio from MoonCast semantic tokens")
    parser.add_argument("--input-dir", default="data/train", help="Input directory with semantic token files")
    parser.add_argument("--output-dir", default="mooncast_audio_output", help="Output directory for audio files")
    parser.add_argument("--files", nargs="+", help="Specific files to process (e.g., 00004.mc.npy)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--no-analyze", action="store_true", help="Skip token analysis")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Initialize MoonCast pipeline
    try:
        pipeline = MoonCastPipeline()
    except Exception as e:
        print(f"❌ Error initializing MoonCast pipeline: {e}")
        return
    
    # Find semantic token files
    if args.files:
        mc_files = [input_dir / f for f in args.files if f.endswith('.mc.npy')]
    else:
        mc_files = sorted(input_dir.glob("*.mc.npy"))
    
    if not mc_files:
        print(f"❌ No semantic token files found in {input_dir}")
        return
    
    print(f"Found {len(mc_files)} semantic token files")
    
    # Limit number of files
    if args.max_files and len(mc_files) > args.max_files:
        print(f"Processing first {args.max_files} files (use --max-files to change)")
        mc_files = mc_files[:args.max_files]
    
    # Process files
    results = []
    
    for i, mc_file in enumerate(mc_files):
        print(f"\n[{i+1}/{len(mc_files)}] Processing {mc_file.name}")
        
        result = process_semantic_file(mc_file, output_dir, pipeline, not args.no_analyze)
        results.append(result)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    print(f"Files with errors: {sum(1 for r in results if 'error' in r)}")
    print(f"Success rate: {(len(results) - sum(1 for r in results if 'error' in r))/len(results)*100:.1f}%")
    print(f"Files saved to: {output_dir}")
    
    print(f"\n✅ Audio generation complete using MoonCast's exact pipeline!")


if __name__ == "__main__":
    main() 