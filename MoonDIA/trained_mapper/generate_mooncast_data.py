"""
generate_mooncast_data.py
========================
Generates MoonCast semantic tokens for all sentences in sentences.txt.

This script runs in the mooncast environment and creates .mc.npy files.
"""

import numpy as np
from pathlib import Path
import sys
import os
import signal
import time

# Add the parent directory to path to import TextToSemantic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from TextToSemantic import TextToSemantic

def main():
    # Initialize MoonCast semantic extractor
    print("Initializing TextToSemantic...")
    extractor = TextToSemantic()
    
    # I/O paths
    root = Path("data/train")
    root.mkdir(parents=True, exist_ok=True)
    
    lines = Path("sentences.txt").read_text(encoding="utf-8").splitlines()
    
    # Main loop
    print(f"Processing {len(lines)} sentences for MoonCast semantic tokens...")
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        print(f"Processing {idx:05d}: {line[:50]}{'...' if len(line) > 50 else ''}")
        
        # --- MoonCast semantic tokens -----------------------------------------
        try:
            print(f"  → Generating semantic tokens for: {line}")
            
            # Add timeout for generation (15 seconds)
            def timeout_handler(signum, frame):
                raise TimeoutError("Generation timed out after 15 seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(75)
            
            try:
                print(f"    → Starting generation...")
                start_time = time.time()
                sem = extractor.generate_semantic_tokens_simple(line)[0].astype("int16")
                end_time = time.time()
                signal.alarm(0)  # Cancel the alarm
                print(f"    → Generation took {end_time - start_time:.2f} seconds")
                print(f"  → Generated {len(sem)} semantic tokens")
                np.save(root / f"{idx:05d}.mc.npy", sem)
                print(f"  → MoonCast: {len(sem)} semantic tokens")
            except TimeoutError:
                print(f"  → TIMEOUT: Generation took too long for sentence {idx}")
                continue
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                raise e
                
        except Exception as e:
            print(f"  → MoonCast ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if idx % 10 == 0:
            print(f"→ {idx:05d}  ok  ({len(sem)} frames)")
    
    print("✓ MoonCast semantic tokens ready:", len(list(root.glob('*.mc.npy'))), "files")

if __name__ == "__main__":
    main() 