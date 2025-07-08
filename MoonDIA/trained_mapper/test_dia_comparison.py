#!/usr/bin/env python3
"""
test_dia_comparison.py
=====================
Test script to compare different DIA generation approaches.
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

def test_dia_generation():
    """Test different DIA generation approaches."""
    try:
        from dia.model import Dia
        print("Loading DIA model...")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", load_dac=True)
        print(f"✅ DIA model loaded on {model.device}")
    except Exception as e:
        print(f"❌ Error loading DIA model: {e}")
        return
    
    test_text = "Hello world, this is a test sentence."
    
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT GENERATION APPROACHES")
    print(f"{'='*60}")
    
    # Test 1: Generate audio directly (should work)
    print(f"\n1. Direct audio generation:")
    try:
        audio = model.generate(
            text=test_text,
            cfg_scale=3.0,
            temperature=1.2,
            top_p=0.95,
            verbose=False
        )
        
        if isinstance(audio, list):
            audio = audio[0]
        
        rms = np.sqrt(np.mean(audio**2))
        duration = len(audio)/44100
        print(f"   ✅ Success: RMS={rms:.6f}, duration={duration:.2f}s, shape={audio.shape}")
        
        # Save for listening
        output_path = "test_direct_audio.wav"
        model.save_audio(output_path, audio)
        print(f"   💾 Saved to {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Generate DAC codes with load_dac=False (current approach)
    print(f"\n2. DAC codes generation (load_dac=False):")
    try:
        # Load model without DAC
        model_no_dac = Dia.from_pretrained("nari-labs/Dia-1.6B", load_dac=False)
        
        dac_codes = model_no_dac.generate(
            text=test_text,
            cfg_scale=3.0,
            temperature=1.2,
            top_p=0.95,
            max_tokens=512,
            verbose=False
        )
        
        if isinstance(dac_codes, list):
            dac_codes = dac_codes[0]
        
        dac_codes = np.array(dac_codes, dtype="uint16")
        print(f"   ✅ Success: shape={dac_codes.shape}, unique={len(np.unique(dac_codes))}")
        
        # Save DAC codes
        np.save("test_dac_codes.npy", dac_codes)
        print(f"   💾 Saved DAC codes to test_dac_codes.npy")
        
        # Try to decode the DAC codes
        print(f"   🔄 Attempting to decode DAC codes...")
        dac_tensor = torch.tensor(dac_codes, dtype=torch.long, device=model.device)
        with torch.no_grad():
            decoded_audio = model._decode(dac_tensor)
        
        decoded_rms = np.sqrt(np.mean(decoded_audio.cpu().numpy()**2))
        decoded_duration = len(decoded_audio)/44100
        print(f"   ✅ Decoded: RMS={decoded_rms:.6f}, duration={decoded_duration:.2f}s")
        
        # Save decoded audio
        output_path = "test_decoded_from_dac.wav"
        model.save_audio(output_path, decoded_audio.cpu().numpy())
        print(f"   💾 Saved decoded audio to {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Compare with existing DAC file
    print(f"\n3. Testing existing DAC file (00002.dac.npy):")
    try:
        dac_file = Path("data/train/00002.dac.npy")
        if dac_file.exists():
            existing_codes = np.load(dac_file)
            print(f"   📁 Loaded: shape={existing_codes.shape}, unique={len(np.unique(existing_codes))}")
            
            # Decode existing codes
            existing_tensor = torch.tensor(existing_codes, dtype=torch.long, device=model.device)
            with torch.no_grad():
                existing_audio = model._decode(existing_tensor)
            
            existing_rms = np.sqrt(np.mean(existing_audio.cpu().numpy()**2))
            existing_duration = len(existing_audio)/44100
            print(f"   ✅ Decoded: RMS={existing_rms:.6f}, duration={existing_duration:.2f}s")
            
            # Save for comparison
            output_path = "test_existing_dac_decoded.wav"
            model.save_audio(output_path, existing_audio.cpu().numpy())
            print(f"   💾 Saved to {output_path}")
        else:
            print(f"   ❌ File not found: {dac_file}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_dia_generation() 