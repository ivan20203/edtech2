#!/usr/bin/env python3
"""
test_mapper.py
==============
Test the trained semantic-to-DAC mapper with a complete pipeline:
Text → MoonCast Semantic Tokens → DAC Codes → DIA Audio
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Import local TextToSemantic (optional - only needed for generation)
try:
    from TextToSemantic import TextToSemantic
    TEXT_TO_SEMANTIC_AVAILABLE = True
except ImportError:
    TEXT_TO_SEMANTIC_AVAILABLE = False
    print("⚠️  TextToSemantic not available - will use file loading only")

# Add DIA path for audio generation (optional)
try:
    # Try multiple possible paths for DIA
    possible_paths = ['../../dia', '../dia', './dia']
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.append(path)
            break
    
    from dia import Dia
    DIA_AVAILABLE = True
    print("✅ Successfully imported DIA")
except ImportError as e:
    DIA_AVAILABLE = False
    print(f"⚠️  DIA not available - audio generation will be skipped: {e}")

from semantic_to_dac_mapper import SemanticToDACMapper

def load_trained_model(model_path: str = "semantic_to_dac_mapper.pt"):
    """
    Load the trained semantic-to-DAC mapper.
    """
    print("Loading trained model...")
    
    # Initialize model
    model = SemanticToDACMapper()
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    return model

def generate_semantic_tokens(text: str, max_tokens: int = 200):
    """
    Generate MoonCast semantic tokens from text.
    """
    if not TEXT_TO_SEMANTIC_AVAILABLE:
        print("❌ TextToSemantic not available in this environment")
        print("Please use option 2 to load from file instead")
        return None
    
    print(f"Generating semantic tokens for: '{text}'")
    
    try:
        # Initialize MoonCast text-to-semantic model
        text_to_semantic = TextToSemantic()
        
        # Generate semantic tokens using the simple interface
        semantic_tokens = text_to_semantic.generate_semantic_tokens_simple(text, role="0")
        
        print(f"✅ Generated {len(semantic_tokens[0])} semantic tokens")
        return semantic_tokens[0]  # Return the actual tokens array
        
    except Exception as e:
        print(f"❌ Error generating semantic tokens: {e}")
        print("❌ Cannot proceed without TextToSemantic - aborting")
        return None

def load_semantic_tokens_from_file(file_path: str = "semantic_tokens.npy"):
    """
    Load semantic tokens from a file (for when generated in MoonCast environment).
    """
    print(f"Loading semantic tokens from {file_path}")
    
    try:
        semantic_tokens = np.load(file_path)
        print(f"✅ Loaded {len(semantic_tokens)} semantic tokens from {file_path}")
        return semantic_tokens
        
    except Exception as e:
        print(f"❌ Error loading semantic tokens: {e}")
        return None

def map_to_dac_codes(model: SemanticToDACMapper, semantic_tokens: np.ndarray):
    """
    Map semantic tokens to DAC codes using the trained mapper.
    """
    print("Mapping semantic tokens to DAC codes...")
    
    print(f"Input semantic tokens: {len(semantic_tokens)}")
    
    # Convert to tensor (no padding needed with new regression model)
    semantic_tensor = torch.tensor(semantic_tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    # Generate DAC codes
    with torch.no_grad():
        dac_codes = model.predict(semantic_tensor)
    
    # Convert back to numpy
    dac_codes = dac_codes.squeeze(0).numpy()  # Remove batch dimension
    
    print(f"✅ Generated DAC codes shape: {dac_codes.shape}")
    print(f"   DAC code range: {dac_codes.min()} to {dac_codes.max()}")
    print(f"   DAC code mean: {dac_codes.mean():.2f}")
    print(f"   Unique DAC values: {len(np.unique(dac_codes))}")
    
    return dac_codes

def generate_audio_from_dac(dac_codes: np.ndarray, output_path: str = "output_audio.wav"):
    """
    Generate audio from DAC codes using Descript DAC decoder.
    """
    print("Generating audio with Descript DAC...")
    
    try:
        import dac
        decoder = dac.DAC.load(dac.utils.download()).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert DAC codes to proper dtype and device
        dac_codes = dac_codes.astype("uint16")  # Ensure correct dtype
        dac_tensor = torch.tensor(dac_codes, dtype=torch.long, device=decoder.device)
        
        # [T,9] -> [1,9,T] format expected by DAC
        codes = dac_tensor.unsqueeze(0).transpose(1, 2)
        
        with torch.no_grad():
            audio = decoder.decode(decoder.quantizer.from_codes(codes)[0]).squeeze()
        
        import soundfile as sf
        sf.write(output_path, audio.cpu().numpy(), 44100)
        print(f"✅ Audio saved to {output_path}")
        return output_path
        
    except ImportError:
        print("❌ DAC library not available - skipping audio generation")
        return None
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        return None

def test_complete_pipeline(text: str = "Hello world. This is a test of the semantic to DAC mapper.", use_file: bool = False):
    """
    Test the complete pipeline from text to audio.
    """
    print("="*60)
    print("TESTING SEMANTIC-TO-DAC MAPPER PIPELINE")
    print("="*60)
    
    # Step 1: Load trained model
    model = load_trained_model()
    
    # Step 2: Get semantic tokens
    if use_file:
        semantic_tokens = load_semantic_tokens_from_file()
        if semantic_tokens is None:
            print("❌ Could not load semantic tokens from file")
            return None, None, None
    else:
        semantic_tokens = generate_semantic_tokens(text)
    
    # Step 3: Map to DAC codes
    dac_codes = map_to_dac_codes(model, semantic_tokens)
    
    # Step 4: Generate audio (optional - requires DIA setup)
    audio_path = generate_audio_from_dac(dac_codes)
    
    # Step 5: Save intermediate results
    print("\nSaving intermediate results...")
    np.save("test_semantic_tokens.npy", semantic_tokens)
    np.save("test_dac_codes.npy", dac_codes)
    print("✅ Intermediate results saved:")
    print("  - test_semantic_tokens.npy")
    print("  - test_dac_codes.npy")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    if not use_file:
        print(f"Input text: '{text}'")
    print(f"Semantic tokens: {len(semantic_tokens)} tokens")
    print(f"DAC codes: {dac_codes.shape[0]} frames × {dac_codes.shape[1]} channels")
    if audio_path:
        print(f"Audio output: {audio_path}")
    else:
        print("Audio generation: Requires DIA setup")
    
    return semantic_tokens, dac_codes, audio_path

def test_with_training_data():
    """
    Test the mapper with actual training data to verify it works.
    """
    print("\n" + "="*60)
    print("TESTING WITH TRAINING DATA")
    print("="*60)
    
    # Load a sample from training data
    try:
        semantic_tokens = np.load("data/train/00000.mc.npy")
        true_dac_codes = np.load("data/train/00000.dac.npy")
        
        print(f"Loaded training sample: {len(semantic_tokens)} tokens")
        
        # Load model
        model = load_trained_model()
        
        # Generate predictions
        semantic_tensor = torch.tensor(semantic_tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            predicted_dac = model.predict(semantic_tensor).squeeze(0).numpy()
        
        # Compare
        print(f"True DAC shape: {true_dac_codes.shape}")
        print(f"Predicted DAC shape: {predicted_dac.shape}")
        
        # Handle length mismatch by truncating to shorter length
        min_length = min(predicted_dac.shape[0], true_dac_codes.shape[0])
        predicted_truncated = predicted_dac[:min_length]
        true_truncated = true_dac_codes[:min_length]
        
        print(f"Comparing first {min_length} frames...")
        
        # Calculate accuracy (how many frames match exactly)
        exact_matches = np.sum(predicted_truncated == true_truncated, axis=1)
        frame_accuracy = np.mean(exact_matches == 9)  # All 9 channels match
        band_accuracy = np.mean(predicted_truncated == true_truncated)  # Overall band-wise accuracy
        print(f"Frame accuracy (all 9 bands): {frame_accuracy:.2%}")
        print(f"Band-wise accuracy: {band_accuracy:.2%}")
        
        # Save truncated comparison
        np.save("test_true_dac.npy", true_truncated)
        np.save("test_predicted_dac.npy", predicted_truncated)
        print("✅ Comparison saved to test_true_dac.npy and test_predicted_dac.npy")
        
    except Exception as e:
        print(f"❌ Error testing with training data: {e}")

def main():
    """
    Main test function.
    """
    print("Semantic-to-DAC Mapper Test")
    print("="*60)
    
    # Choose testing mode
    print("Choose testing mode:")
    print("1. Generate semantic tokens from text (requires MoonCast environment)")
    print("2. Load semantic tokens from file (semantic_tokens.npy)")
    print("3. Test with training data only")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        if not TEXT_TO_SEMANTIC_AVAILABLE:
            print("❌ TextToSemantic not available in this environment")
            print("Please use option 2 to load from semantic_tokens.npy")
            return
        
        # Test with custom text
        test_text = input("Enter test text (or press Enter for default): ").strip()
        if not test_text:
            test_text = "Hello world. This is a test of the semantic to DAC mapper."
        
        # Run complete pipeline test
        semantic_tokens, dac_codes, audio_path = test_complete_pipeline(test_text, use_file=False)
        
    elif choice == "2":
        # Test with pre-generated semantic tokens
        semantic_tokens, dac_codes, audio_path = test_complete_pipeline(use_file=True)
        
    elif choice == "3":
        # Test with training data only
        test_with_training_data()
        return
        
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Test with training data (only if we ran the pipeline)
    if semantic_tokens is not None:
        test_with_training_data()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("Check the generated files:")
    print("  - test_semantic_tokens.npy (MoonCast semantic tokens)")
    print("  - test_dac_codes.npy (Predicted DIA DAC codes)")
    print("  - test_true_dac.npy (True DAC codes from training)")
    print("  - test_predicted_dac.npy (Predicted DAC codes)")
    if audio_path:
        print(f"  - {audio_path} (Generated audio)")
    
    print("\nTo generate semantic tokens in MoonCast environment:")
    print("1. Activate MoonCast environment: conda activate mooncast")
    print("2. Run: python generate_semantic_tokens.py")
    print("3. Copy semantic_tokens.npy to this directory")
    print("4. Run this test script again with option 2")

if __name__ == "__main__":
    main() 