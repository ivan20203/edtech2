#!/usr/bin/env python3
"""
debug_dac_codes.py
=================
Debug script to understand why DAC codes are generating silent audio.
"""

import numpy as np
import torch
import soundfile as sf
from pathlib import Path
import sys
import os

# Add DIA to path for imports
dia_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dia")
sys.path.append(dia_path)

def analyze_dac_codes_detailed(codes: np.ndarray, name: str = "DAC codes") -> dict:
    """Detailed analysis of DAC codes."""
    print(f"\n{name} Detailed Analysis:")
    print(f"  Shape: {codes.shape}")
    print(f"  Data type: {codes.dtype}")
    print(f"  Value range: {codes.min()} to {codes.max()}")
    print(f"  Mean: {codes.mean():.2f}")
    print(f"  Std: {codes.std():.2f}")
    print(f"  Unique values: {len(np.unique(codes))}")
    
    # Check for common values
    unique_vals, counts = np.unique(codes, return_counts=True)
    print(f"  Most common values:")
    for val, count in sorted(zip(unique_vals, counts), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {val}: {count} times ({count/codes.size*100:.1f}%)")
    
    # Check for potential silence patterns
    silence_candidates = [0, 1, 2, 1023, 1022, 1021]  # Common silence codes
    silence_count = sum(counts[unique_vals == val] for val in silence_candidates if val in unique_vals)
    silence_percentage = float(silence_count) / codes.size * 100
    
    print(f"  Potential silence codes: {silence_count} ({silence_percentage:.1f}%)")
    
    # Check for repeated patterns
    if codes.shape[1] > 1:  # Multiple channels
        print(f"  Channel analysis:")
        for ch in range(min(3, codes.shape[1])):  # First 3 channels
            ch_codes = codes[:, ch]
            ch_unique = len(np.unique(ch_codes))
            ch_mean = ch_codes.mean()
            print(f"    Channel {ch}: {ch_unique} unique values, mean={ch_mean:.1f}")
    
    return {
        'shape': codes.shape,
        'min': codes.min(),
        'max': codes.max(),
        'mean': codes.mean(),
        'std': codes.std(),
        'unique_count': len(unique_vals),
        'silence_percentage': silence_percentage
    }

def test_dac_decoding():
    """Test DAC decoding with different approaches."""
    try:
        from dia.model import Dia
        print("Loading DIA model...")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", load_dac=True)
        print(f"✅ DIA model loaded on {model.device}")
    except Exception as e:
        print(f"❌ Error loading DIA model: {e}")
        return
    
    # Load a sample DAC file
    dac_file = Path("data/train/00002.dac.npy")  # This one had more unique values
    if not dac_file.exists():
        print(f"❌ DAC file not found: {dac_file}")
        return
    
    codes = np.load(dac_file)
    print(f"Loaded DAC codes: {codes.shape}")
    
    # Analyze the codes
    analysis = analyze_dac_codes_detailed(codes, f"DAC codes from {dac_file.name}")
    
    # Test different decoding approaches
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT DECODING APPROACHES")
    print(f"{'='*60}")
    
    # Approach 1: Direct decoding (current approach)
    print(f"\n1. Direct decoding (current approach):")
    try:
        dac_tensor = torch.tensor(codes, dtype=torch.long, device=model.device)
        with torch.no_grad():
            audio = model._decode(dac_tensor)
        audio_np = audio.cpu().numpy()
        rms = np.sqrt(np.mean(audio_np**2))
        print(f"   ✅ Success: RMS={rms:.6f}, shape={audio_np.shape}")
        
        # Save for listening
        output_path = f"debug_direct_{dac_file.stem}.wav"
        model.save_audio(output_path, audio_np)
        print(f"   💾 Saved to {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Approach 2: Try with different data types
    print(f"\n2. Testing with different data types:")
    for dtype in ['int32', 'int64', 'float32']:
        try:
            dac_tensor = torch.tensor(codes, dtype=getattr(torch, dtype), device=model.device)
            with torch.no_grad():
                audio = model._decode(dac_tensor)
            audio_np = audio.cpu().numpy()
            rms = np.sqrt(np.mean(audio_np**2))
            print(f"   {dtype}: RMS={rms:.6f}")
        except Exception as e:
            print(f"   {dtype}: ❌ {e}")
    
    # Approach 3: Try generating fresh codes with load_dac=True
    print(f"\n3. Generating fresh codes with load_dac=True:")
    try:
        # Load model with DAC
        model_with_dac = Dia.from_pretrained("nari-labs/Dia-1.6B", load_dac=True)
        
        # Generate audio directly
        audio = model_with_dac.generate(
            text="Hello world",
            cfg_scale=2.0,
            temperature=0.8,
            top_p=0.9,
            verbose=False
        )
        
        if isinstance(audio, list):
            audio = audio[0]
        
        rms = np.sqrt(np.mean(audio**2))
        print(f"   ✅ Fresh generation: RMS={rms:.6f}, shape={audio.shape}")
        
        # Save for comparison
        output_path = "debug_fresh_generation.wav"
        model_with_dac.save_audio(output_path, audio)
        print(f"   💾 Saved to {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_dac_decoding() 