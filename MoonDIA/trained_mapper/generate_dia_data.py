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
import time
import signal

# Add DIA to path (we're running in DIA environment)
dia_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dia")
sys.path.append(dia_path)

# Import DIA
from dia.model import Dia

def main():
    # Check GPU availability
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize DIA model
    print("Initializing DIA model...")
    tts = Dia.from_pretrained(
        "nari-labs/Dia-1.6B", 
        compute_dtype="float16",
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        load_dac=False  # This will return raw DAC codes instead of audio
    )
    
    print(f"Model device: {tts.device}")
    print(f"Model dtype: {tts.compute_dtype}")
    
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
            print(f"  → Starting DIA generation...")
            start_time = time.time()
            
            print(f"    → Sentence: {len(line.split())} words")
            
            # Add timeout for generation (30 seconds)
            def timeout_handler(signum, frame):
                raise TimeoutError("Generation timed out after 30 seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                # This call returns raw DAC codes (not audio) because load_dac=False
                dac_codes = tts.generate(
                    text=line,
                    cfg_scale=2.0,   # Reduced for faster generation
                    temperature=0.8, # Reduced for more deterministic generation
                    top_p=0.9,       # Slightly reduced
                    verbose=False    # Disable verbose to reduce overhead
                )
                
                signal.alarm(0)  # Cancel the alarm
                end_time = time.time()
                generation_time = end_time - start_time
            except TimeoutError:
                print(f"  → TIMEOUT: Generation took too long for sentence {idx}")
                continue
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                raise e
            
            # Ensure we get the first (and only) result
            if isinstance(dac_codes, list):
                dac_codes = dac_codes[0]
            
            # Ensure numpy uint16:
            dac_codes = np.array(dac_codes, dtype="uint16")
            np.save(root / f"{idx:05d}.dac.npy", dac_codes)
            print(f"  → DIA: {dac_codes.shape} DAC codes (took {generation_time:.2f}s)")
            
        except Exception as e:
            print(f"  → DIA ERROR: {e}")
            continue
        
        if idx % 10 == 0:
            print(f"→ {idx:05d}  ok  ({dac_codes.shape[0]} frames, {generation_time:.2f}s)")
    
    print("✓ DIA DAC codes ready:", len(list(root.glob('*.dac.npy'))), "files")

if __name__ == "__main__":
    main() 