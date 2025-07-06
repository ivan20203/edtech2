"""
generate_dia_data.py
===================
Generates DIA DAC codes for all sentences in sentences.txt.

This script runs in the DIA environment and creates .dac.npy files.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add DIA to path (we're running in DIA environment)
dia_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dia")
sys.path.append(dia_path)

# Import DIA
from dia.model import Dia

def main():
    # Initialize DIA model
    print("Initializing DIA model...")
    tts = Dia.from_pretrained(
        "nari-labs/Dia-1.6B", 
        compute_dtype="float16",
        load_dac=False  # This will return raw DAC codes instead of audio
    )
    
    # I/O paths
    root = Path("data/train")
    root.mkdir(parents=True, exist_ok=True)
    
    lines = Path("sentences.txt").read_text(encoding="utf-8").splitlines()
    
    # Main loop
    print(f"Processing {len(lines)} sentences for DIA DAC codes...")
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        print(f"Processing {idx:05d}: {line[:50]}{'...' if len(line) > 50 else ''}")
        
        # --- DIA DAC codes ----------------------------------------------------
        try:
            # This call returns raw DAC codes (not audio) because load_dac=False
            dac_codes = tts.generate(
                text=line,
                max_tokens=1000,  # Adjust as needed
                cfg_scale=3.0,
                temperature=1.2,
                top_p=0.95,
                verbose=False
            )
            
            # Ensure we get the first (and only) result
            if isinstance(dac_codes, list):
                dac_codes = dac_codes[0]
            
            # Ensure numpy uint16:
            dac_codes = np.array(dac_codes, dtype="uint16")
            np.save(root / f"{idx:05d}.dac.npy", dac_codes)
            print(f"  → DIA: {dac_codes.shape} DAC codes")
            
        except Exception as e:
            print(f"  → DIA ERROR: {e}")
            continue
        
        if idx % 10 == 0:
            print(f"→ {idx:05d}  ok  ({dac_codes.shape[0]} frames)")
    
    print("✓ DIA DAC codes ready:", len(list(root.glob('*.dac.npy'))), "files")

if __name__ == "__main__":
    main() 