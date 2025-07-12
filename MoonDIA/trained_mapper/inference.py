#!/usr/bin/env python3
"""
Comprehensive semantic-to-DAC mapper tester for s2d.pt model.
Features:
    1. Load semantic tokens from file or generate from text
    2. Map to DAC codes using trained s2d.pt model
    3. Generate audio using DIA vocoder or Descript DAC
    4. Compare with training data for accuracy metrics
    5. Save all intermediate results

Usage:
    python inference.py --sem-file data/train/00033.mc.npy --ckpt s2d.pt --wav
    python inference.py --text "Hello world" --ckpt s2d.pt --wav
    python inference.py --test-training-data --ckpt s2d.pt
"""

from __future__ import annotations

import argparse, sys, os, json, textwrap
from pathlib import Path
from typing import Optional
import numpy as np
import torch

# -----------------------------------------------------------------------------
# local imports
# -----------------------------------------------------------------------------
from semantic2dac_model import Semantic2DAC, EOS, PAD
from dataset import flatten_dac

# optional MoonCast text→semantic generator
try:
    from TextToSemantic import TextToSemantic
    TEXT2SEM_OK = True
except Exception:
    TEXT2SEM_OK = False

# optional DIA vocoder
try:
    from transformers import AutoProcessor, DiaVocoder
    DIA_OK = True
except Exception:
    DIA_OK = False

# optional Descript DAC
try:
    import dac
    DAC_OK = True
except Exception:
    DAC_OK = False

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def unflatten_dac(seq: np.ndarray) -> np.ndarray:
    """Convert 1-D sequence with band offset → (T, 9) raw DAC matrix."""
    seq = seq.reshape(-1, 9)
    offset = (np.arange(9, dtype=np.int64) * 1024)[None, :]
    return seq - offset

def load_mapper(ckpt: Path, d_model: int = 384) -> Semantic2DAC:
    """Load the trained s2d.pt model."""
    print(f"Loading mapper from {ckpt} …")
    mapper = Semantic2DAC(d_model=d_model, n_heads=12, dropout=0.0).eval()
    mapper.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return mapper

def get_semantic_tokens(path: Path | None = None, text: str | None = None) -> np.ndarray:
    """Load tokens from .npy or generate from text via MoonCast."""
    if path is not None:
        arr = np.load(path)
        print(f"✔ {len(arr)} semantic tokens loaded from {path}")
        return arr

    if not TEXT2SEM_OK:
        raise RuntimeError("TextToSemantic not available; use --sem-file")

    text = text or "Hello world. This is a mapper test."
    print(f"Generating semantic tokens for:\n{text}\n")
    generator = TextToSemantic()
    tokens = generator.generate_semantic_tokens_simple(text, role="0")[0]
    print("✔ generated", len(tokens), "tokens")
    return np.asarray(tokens, dtype=np.int64)

def predict_dac(mapper: Semantic2DAC, sem: np.ndarray) -> np.ndarray:
    """Map semantic tokens to DAC codes."""
    device = next(mapper.parameters()).device
    sem_t = torch.tensor(sem, dtype=torch.long, device=device)
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        dac_mat = mapper.generate(sem_t)
    
    print(f"✅ Generated DAC codes shape: {dac_mat.shape}")
    print(f"   DAC code range: {dac_mat.min()} to {dac_mat.max()}")
    print(f"   DAC code mean: {dac_mat.mean():.2f}")
    print(f"   Unique DAC values: {len(np.unique(dac_mat))}")
    
    return dac_mat

def save_numpy(arr: np.ndarray, name: str):
    """Save numpy array to file."""
    np.save(name, arr)
    print(f"numpy → {name}")

def decode_audio_dia(dac_mat: np.ndarray, wav_out: Path):
    """Generate audio using DIA vocoder."""
    if not DIA_OK:
        print("❌ DIA vocoder unavailable")
        return None

    print("Decoding audio with DIA vocoder …")
    try:
        proc = AutoProcessor.from_pretrained("nari-labs/Dia-1.6B-0626")
        voc = DiaVocoder.from_pretrained("nari-labs/Dia-1.6B-0626").to(
            "cuda" if torch.cuda.is_available() else "cpu")

        codes = torch.tensor(dac_mat, dtype=torch.long, device=voc.device)
        codes = codes.unsqueeze(0).transpose(1, 2)  # [1,9,T]
        
        with torch.no_grad():
            audio = voc.decode(voc.quantizer.from_codes(codes)[0])[0]

        proc.save_audio([audio.cpu().numpy()], wav_out)
        print(f"✅ WAV saved → {wav_out}")
        return wav_out
    except Exception as e:
        print(f"❌ Error with DIA vocoder: {e}")
        return None

def decode_audio_dac(dac_mat: np.ndarray, wav_out: Path):
    """Generate audio using Descript DAC decoder."""
    if not DAC_OK:
        print("❌ Descript DAC unavailable")
        return None

    print("Decoding audio with Descript DAC...")
    try:
        # Convert flattened DAC codes back to raw codes (remove band offsets)
        raw_dac_codes = unflatten_dac(dac_mat)
        
        # Ensure codes are in valid range for Descript DAC (0-1023)
        raw_dac_codes = np.clip(raw_dac_codes, 0, 1023).astype("uint16")
        
        print(f"   Raw DAC range: {raw_dac_codes.min()} to {raw_dac_codes.max()}")
        
        decoder = dac.DAC.load(dac.utils.download()).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert DAC codes to proper dtype and device
        dac_tensor = torch.tensor(raw_dac_codes, dtype=torch.long, device=decoder.device)
        
        # [T,9] -> [1,9,T] format expected by DAC
        codes = dac_tensor.unsqueeze(0).transpose(1, 2)
        
        with torch.no_grad():
            audio = decoder.decode(decoder.quantizer.from_codes(codes)[0]).squeeze()
        
        import soundfile as sf
        sf.write(wav_out, audio.cpu().numpy(), 44100)
        print(f"✅ WAV saved → {wav_out}")
        return wav_out
    except Exception as e:
        print(f"❌ Error with Descript DAC: {e}")
        return None

def test_with_training_data(mapper: Semantic2DAC, sample_idx: int = 0):
    """Test the mapper with actual training data and calculate accuracy."""
    print("\n" + "="*60)
    print("TESTING WITH TRAINING DATA")
    print("="*60)
    
    try:
        # Load training sample
        base_name = f"{sample_idx:05d}"
        sem_file = f"data/train/{base_name}.mc.npy"
        dac_file = f"data/train/{base_name}.dac.npy"
        
        semantic_tokens = np.load(sem_file)
        true_dac_codes = np.load(dac_file)
        
        print(f"Loaded training sample {sample_idx}: {len(semantic_tokens)} tokens")
        print(f"True DAC shape: {true_dac_codes.shape}")
        
        # Generate predictions
        predicted_dac = predict_dac(mapper, semantic_tokens)
        
        # Compare shapes
        print(f"Predicted DAC shape: {predicted_dac.shape}")
        
        # Handle length mismatch by truncating to shorter length
        min_length = min(predicted_dac.shape[0], true_dac_codes.shape[0])
        predicted_truncated = predicted_dac[:min_length]
        true_truncated = true_dac_codes[:min_length]
        
        print(f"Comparing first {min_length} frames...")
        
        # Calculate accuracy metrics
        exact_matches = np.sum(predicted_truncated == true_truncated, axis=1)
        frame_accuracy = np.mean(exact_matches == 9)  # All 9 channels match
        band_accuracy = np.mean(predicted_truncated == true_truncated)  # Overall band-wise accuracy
        
        print(f"Frame accuracy (all 9 bands): {frame_accuracy:.2%}")
        print(f"Band-wise accuracy: {band_accuracy:.2%}")
        
        # Save comparison results
        np.save("test_true_dac.npy", true_truncated)
        np.save("test_predicted_dac.npy", predicted_truncated)
        np.save("test_semantic_tokens.npy", semantic_tokens)
        print("✅ Comparison saved to:")
        print("  - test_true_dac.npy")
        print("  - test_predicted_dac.npy") 
        print("  - test_semantic_tokens.npy")
        
        return semantic_tokens, predicted_dac, true_dac_codes, frame_accuracy, band_accuracy
        
    except Exception as e:
        print(f"❌ Error testing with training data: {e}")
        return None, None, None, 0, 0

def test_complete_pipeline(mapper: Semantic2DAC, sem_file: Path = None, text: str = None, 
                          generate_audio: bool = False, audio_method: str = "dia"):
    """Test the complete pipeline from semantic tokens to audio."""
    print("="*60)
    print("TESTING SEMANTIC-TO-DAC MAPPER PIPELINE")
    print("="*60)
    
    # Get semantic tokens
    semantic_tokens = get_semantic_tokens(sem_file, text)
    
    # Map to DAC codes
    dac_codes = predict_dac(mapper, semantic_tokens)
    
    # Save intermediate results
    out_name = f"pred_{sem_file.stem if sem_file else 'text'}"
    save_numpy(semantic_tokens, f"{out_name}.mc.npy")
    save_numpy(dac_codes, f"{out_name}.dac.npy")
    
    # Generate audio if requested
    audio_path = None
    if generate_audio:
        wav_out = Path(f"{out_name}.wav")
        if audio_method == "dia":
            audio_path = decode_audio_dia(dac_codes, wav_out)
        elif audio_method == "dac":
            audio_path = decode_audio_dac(dac_codes, wav_out)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    if text:
        print(f"Input text: '{text}'")
    elif sem_file:
        print(f"Input file: {sem_file}")
    print(f"Semantic tokens: {len(semantic_tokens)} tokens")
    print(f"DAC codes: {dac_codes.shape[0]} frames × {dac_codes.shape[1]} channels")
    if audio_path:
        print(f"Audio output: {audio_path}")
    else:
        print("Audio generation: Not requested or failed")
    
    return semantic_tokens, dac_codes, audio_path

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""
        Comprehensive semantic→DAC mapper tester for s2d.pt model.

        Examples:
          python inference.py --sem-file data/train/00033.mc.npy --ckpt s2d.pt --wav
          python inference.py --text "Hello there" --ckpt s2d.pt --wav --audio-method dac
          python inference.py --test-training-data --ckpt s2d.pt
        """))
    
    p.add_argument("--ckpt", default="s2d.pt", help="checkpoint path of trained mapper")
    p.add_argument("--d-model", type=int, default=384, help="d_model size used in training")
    
    # Input options
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sem-file", type=Path, help=".mc.npy file to convert")
    g.add_argument("--text", help="Plain text to run through TextToSemantic")
    g.add_argument("--test-training-data", action="store_true", 
                   help="Test with training data and calculate accuracy")
    
    # Audio options
    p.add_argument("--wav", action="store_true", help="generate audio output")
    p.add_argument("--audio-method", choices=["dia", "dac"], default="dia",
                   help="Audio generation method: dia (DIA vocoder) or dac (Descript DAC)")
    
    args = p.parse_args()

    # Load model
    mapper = load_mapper(Path(args.ckpt), d_model=args.d_model)

    if args.test_training_data:
        # Test with training data
        test_with_training_data(mapper)
    else:
        # Test complete pipeline
        test_complete_pipeline(
            mapper, 
            sem_file=args.sem_file, 
            text=args.text,
            generate_audio=args.wav,
            audio_method=args.audio_method
        )

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("Available audio methods:")
    print(f"  - DIA vocoder: {'✅' if DIA_OK else '❌'}")
    print(f"  - Descript DAC: {'✅' if DAC_OK else '❌'}")
    print(f"  - TextToSemantic: {'✅' if TEXT2SEM_OK else '❌'}")


if __name__ == "__main__":
    main()




