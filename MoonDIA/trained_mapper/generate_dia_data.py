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
    
    lines = Path("ten_thousand_sentences.txt").read_text(encoding="utf-8").splitlines()
    
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
            
            # Add timeout for generation (60 seconds)
            def timeout_handler(signum, frame):
                raise TimeoutError("Generation timed out after 60 seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)
            
            try:
                # This call returns raw DAC codes (not audio) because load_dac=False
                dac_codes = tts.generate(
                    text=line,
                    cfg_scale=5,
                    temperature=1.5,
                    top_p=1,
                    cfg_filter_top_k=50,
                    max_tokens=400,  # Use model config default
                    verbose=False    # Disable verbose to reduce overhead
                )

                """                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=3072,
                    value=model.config.data.audio_length,  # Use config default if available, else fallback
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,  # Default from inference.py
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=1.0,
                    maximum=1.5,
                    value=1.3,  # Default from inference.py
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,  # Default from inference.py
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=50,
                    value=30,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=0.94,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )
"""
                
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