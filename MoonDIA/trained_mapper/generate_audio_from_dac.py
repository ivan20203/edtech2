#!/usr/bin/env python3
"""
generate_audio_from_dac.py
==========================
Generate audio from DIA DAC files in the train directory using DIA's exact pipeline.

This script:
1. Loads DAC files from data/train/
2. Decodes them to audio using DIA's built-in DAC decoder
3. Saves audio files for listening and analysis
"""

import numpy as np
import torch
import soundfile as sf
from pathlib import Path
import argparse
import time
from typing import List, Tuple
import sys
import os

# Add DIA to path for imports
dia_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dia")
sys.path.append(dia_path)

def load_dia_model():
    """Load the DIA model for decoding DAC codes."""
    try:
        from dia.model import Dia
        print("Loading DIA model...")
        # Use DIA's from_pretrained to get the model with DAC loaded
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", load_dac=True)
        print(f"✅ DIA model loaded on {model.device}")
        return model
    except ImportError as e:
        print(f"❌ DIA library not available: {e}")
        print("Make sure you're in the DIA environment and DIA is installed")
        return None
    except Exception as e:
        print(f"❌ Error loading DIA model: {e}")
        return None

def analyze_dac_codes(codes: np.ndarray, name: str = "DAC codes") -> dict:
    """Analyze DAC code statistics."""
    print(f"\n{name} Analysis:")
    print(f"  Shape: {codes.shape}")
    print(f"  Data type: {codes.dtype}")
    print(f"  Value range: {codes.min()} to {codes.max()}")
    print(f"  Mean: {codes.mean():.2f}")
    print(f"  Unique values: {len(np.unique(codes))}")
    
    # Check for common values
    unique_vals, counts = np.unique(codes, return_counts=True)
    print(f"  Most common values:")
    for val, count in sorted(zip(unique_vals, counts), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {val}: {count} times ({count/codes.size*100:.1f}%)")
    
    # Check for potential silence patterns
    silence_candidates = [0, 1, 2, 1023, 1022, 1021]  # Common silence codes
    silence_count = sum(counts[unique_vals == val] for val in silence_candidates if val in unique_vals)
    silence_percentage = float(silence_count) / codes.size * 100
    
    print(f"  Potential silence codes: {silence_count} ({silence_percentage:.1f}%)")
    
    return {
        'shape': codes.shape,
        'min': codes.min(),
        'max': codes.max(),
        'mean': codes.mean(),
        'unique_count': len(unique_vals),
        'silence_percentage': silence_percentage
    }

def decode_dac_codes(model, codes: np.ndarray, output_path: str) -> Tuple[np.ndarray, dict]:
    """Decode DAC codes to audio using DIA's built-in DAC decoder."""
    try:
        print(f"Decoding to {output_path}...")
        
        # Ensure correct dtype and range
        codes = codes.astype("uint16")
        
        # Check for invalid codes
        invalid_mask = (codes < 0) | (codes > 1023)
        if invalid_mask.any():
            print(f"  ⚠️  Found {invalid_mask.sum()} invalid codes, replacing with 0")
            codes[invalid_mask] = 0
        
        # Convert to tensor and move to model device
        dac_tensor = torch.tensor(codes, dtype=torch.long, device=model.device)
        
        # The codes from generate_dia_data.py (with load_dac=False) are already 
        # in the correct format for DAC decoding - they've been processed through
        # delay pattern reversion in _generate_output
        with torch.no_grad():
            audio = model._decode(dac_tensor)
        
        # Check audio properties
        audio_np = audio.cpu().numpy()
        rms = np.sqrt(np.mean(audio_np**2))
        duration = len(audio_np)/44100  # DIA uses 44.1kHz
        peak = np.max(np.abs(audio_np))
        
        print(f"  Audio shape: {audio_np.shape}")
        print(f"  Audio range: {audio_np.min():.4f} to {audio_np.max():.4f}")
        print(f"  Audio RMS: {rms:.4f}")
        print(f"  Audio peak: {peak:.4f}")
        print(f"  Audio duration: {duration:.2f} seconds")
        
        # Save audio using DIA's save_audio method
        model.save_audio(output_path, audio_np)
        print(f"  ✅ Saved to {output_path}")
        
        return audio_np, {
            'rms': rms,
            'peak': peak,
            'duration': duration,
            'has_voice': rms > 0.01
        }
        
    except Exception as e:
        print(f"  ❌ Error decoding: {e}")
        return None, {'error': str(e)}

def process_dac_file(dac_path: Path, output_dir: Path, model, analyze: bool = True) -> dict:
    """Process a single DAC file."""
    print(f"\n{'='*60}")
    print(f"Processing: {dac_path.name}")
    print(f"{'='*60}")
    
    # Load DAC codes
    try:
        codes = np.load(dac_path)
    except Exception as e:
        print(f"❌ Error loading {dac_path.name}: {e}")
        return {'error': str(e)}
    
    # Analyze codes
    if analyze:
        analysis = analyze_dac_codes(codes, f"DAC codes from {dac_path.name}")
    
    # Generate output path
    stem = dac_path.stem
    output_path = output_dir / f"{stem}_dia_audio.wav"
    
    # Decode to audio
    audio, audio_stats = decode_dac_codes(model, codes, str(output_path))
    
    if audio is not None:
        print(f"✅ Successfully generated audio: {audio_stats['duration']:.2f}s, RMS: {audio_stats['rms']:.4f}")
        if audio_stats['has_voice']:
            print("🎤 Audio contains voice!")
        else:
            print("🔇 Audio appears to be silent")
    
    return {
        'file': dac_path.name,
        'dac_analysis': analysis if analyze else None,
        'audio_stats': audio_stats,
        'output_path': str(output_path) if audio is not None else None
    }

def main():
    parser = argparse.ArgumentParser(description="Generate audio from DIA DAC files using DIA's exact pipeline")
    parser.add_argument("--input-dir", default="data/train", help="Input directory with DAC files")
    parser.add_argument("--output-dir", default="dia_audio_output", help="Output directory for audio files")
    parser.add_argument("--files", nargs="+", help="Specific files to process (e.g., 00000.dac.npy)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process (default: process all files)")
    parser.add_argument("--no-analyze", action="store_true", help="Skip DAC code analysis")
    parser.add_argument("--min-rms", type=float, default=0.01, help="Minimum RMS to consider as voiced")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Load DIA model
    model = load_dia_model()
    if model is None:
        return
    
    # Find DAC files
    if args.files:
        dac_files = [input_dir / f for f in args.files if f.endswith('.dac.npy')]
    else:
        dac_files = sorted(input_dir.glob("*.dac.npy"))
    
    if not dac_files:
        print(f"❌ No DAC files found in {input_dir}")
        return
    
    print(f"Found {len(dac_files)} DAC files")
    
    # Limit number of files
    if args.max_files and len(dac_files) > args.max_files:
        print(f"Processing first {args.max_files} files (use --max-files to change)")
        dac_files = dac_files[:args.max_files]
    
    # Process files
    results = []
    voiced_count = 0
    silent_count = 0
    
    for i, dac_file in enumerate(dac_files):
        print(f"\n[{i+1}/{len(dac_files)}] Processing {dac_file.name}")
        
        result = process_dac_file(dac_file, output_dir, model, not args.no_analyze)
        results.append(result)
        
        if 'audio_stats' in result and 'has_voice' in result['audio_stats']:
            if result['audio_stats']['has_voice']:
                voiced_count += 1
            else:
                silent_count += 1
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    print(f"Files with voice (RMS > {args.min_rms}): {voiced_count}")
    print(f"Silent files: {silent_count}")
    print(f"Success rate: {voiced_count/len(results)*100:.1f}%")
    print(f"Audio files saved to: {output_dir}")
    
    # Show some examples
    if results:
        print(f"\nExample results:")
        for result in results[:3]:
            if 'audio_stats' in result and 'error' not in result['audio_stats']:
                stats = result['audio_stats']
                voice_status = "🎤 VOICED" if stats.get('has_voice', False) else "🔇 SILENT"
                print(f"  {result['file']}: {stats['duration']:.2f}s, RMS: {stats['rms']:.4f} - {voice_status}")

if __name__ == "__main__":
    main() 