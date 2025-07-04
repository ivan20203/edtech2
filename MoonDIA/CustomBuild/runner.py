#!/usr/bin/env python3
"""
Runner script for TextToSemantic component.
Demonstrates how to use the TextToSemantic class to generate semantic tokens from text.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from TextToSemantic import TextToSemantic
except ImportError as e:
    print(f"Error importing TextToSemantic: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed.")
    sys.exit(1)


def check_cuda():
    """Check if CUDA is available and print device info."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        print(f"✅ CUDA available: {device_name}")
        print(f"   Device ID: {device}")
        print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("❌ CUDA not available. This will run on CPU (very slow).")
        return False


def example_single_text():
    """Example: Generate semantic tokens for a single text input."""
    print("\n" + "="*50)
    print("EXAMPLE 1: Single Text Input")
    print("="*50)
    
    # Initialize model
    print("Initializing TextToSemantic model...")
    model = TextToSemantic()
    
    # Test text
    text = "Hello, how are you today? I hope you're doing well."
    
    print(f"Input text: '{text}'")
    print("Generating semantic tokens...")
    
    # Generate semantic tokens
    semantic_tokens = model.generate_semantic_tokens_simple(text, role="0")
    
    print(f"✅ Generated semantic tokens:")
    print(f"   Shape: {semantic_tokens.shape}")
    print(f"   Length: {len(semantic_tokens[0])} tokens")
    print(f"   Data type: {semantic_tokens.dtype}")
    print(f"   Value range: {semantic_tokens.min()} to {semantic_tokens.max()}")
    
    return semantic_tokens


def example_dialogue():
    """Example: Generate semantic tokens for a dialogue."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Dialogue Input")
    print("="*50)
    
    # Initialize model
    print("Initializing TextToSemantic model...")
    model = TextToSemantic()
    
    # Test dialogue
    dialogue = [
        {"role": "0", "text": "Hello, how are you today?"},
        {"role": "1", "text": "I'm doing great, thank you! How about you?"},
        {"role": "0", "text": "I'm doing well too. Nice weather we're having."},
        {"role": "1", "text": "Yes, it's beautiful outside!"}
    ]
    
    print("Input dialogue:")
    for i, turn in enumerate(dialogue):
        print(f"   Turn {i+1} ({turn['role']}): '{turn['text']}'")
    
    print("\nGenerating semantic tokens...")
    
    # Generate semantic tokens
    semantic_tokens_list = model.generate_semantic_tokens(dialogue)
    
    print(f"✅ Generated {len(semantic_tokens_list)} semantic token sequences:")
    for i, (turn, tokens) in enumerate(zip(dialogue, semantic_tokens_list)):
        print(f"   Turn {i+1} ({turn['role']}): {len(tokens[0])} tokens")
    
    return semantic_tokens_list


def example_batch_processing():
    """Example: Process multiple texts in batch."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Batch Processing")
    print("="*50)
    
    # Initialize model
    print("Initializing TextToSemantic model...")
    model = TextToSemantic()
    
    # Test texts
    texts = [
        "Good morning!",
        "How's the weather today?",
        "I love this music.",
        "What time is the meeting?",
        "Have a great day!"
    ]
    
    print(f"Processing {len(texts)} texts...")
    
    # Process each text
    results = []
    for i, text in enumerate(texts):
        print(f"   Processing text {i+1}/{len(texts)}: '{text}'")
        semantic_tokens = model.generate_semantic_tokens_simple(text, role="0")
        results.append(semantic_tokens)
    
    print(f"✅ Batch processing complete:")
    for i, (text, tokens) in enumerate(zip(texts, results)):
        print(f"   Text {i+1}: {len(tokens[0])} tokens")
    
    return results


def save_semantic_tokens(tokens, filename):
    """Save semantic tokens to a numpy file."""
    try:
        # Handle different token shapes
        if isinstance(tokens, list):
            # For dialogue/batch results, save as a list of arrays
            np.save(filename, tokens, allow_pickle=True)
        else:
            # For single results, save as regular array
            np.save(filename, tokens)
        print(f"✅ Saved semantic tokens to {filename}")
    except Exception as e:
        print(f"❌ Error saving tokens: {e}")
        # Try alternative save method
        try:
            import pickle
            with open(filename.replace('.npy', '.pkl'), 'wb') as f:
                pickle.dump(tokens, f)
            print(f"✅ Saved semantic tokens to {filename.replace('.npy', '.pkl')} (pickle format)")
        except Exception as e2:
            print(f"❌ Error saving with pickle: {e2}")


def load_semantic_tokens(filename):
    """Load semantic tokens from a numpy file."""
    try:
        tokens = np.load(filename)
        print(f"✅ Loaded semantic tokens from {filename}")
        return tokens
    except Exception as e:
        print(f"❌ Error loading tokens: {e}")
        return None


def main():
    """Main function to run all examples."""
    print("🚀 TextToSemantic Runner")
    print("="*50)
    
    # Check CUDA
    cuda_available = check_cuda()
    
    if not cuda_available:
        print("\n⚠️  Warning: Running without CUDA will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    try:
        # Example 1: Single text
        single_tokens = example_single_text()
        
        # Example 2: Dialogue
        dialogue_tokens = example_dialogue()
        
        # Example 3: Batch processing
        batch_tokens = example_batch_processing()
        
        # Save results
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        save_semantic_tokens(single_tokens, "single_text_tokens.npy")
        save_semantic_tokens(dialogue_tokens, "dialogue_tokens.npy")
        save_semantic_tokens(batch_tokens, "batch_tokens.npy")
        
        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Use these semantic tokens with SemantiCodecMapper")
        print("2. Convert to DAC tokens")
        print("3. Generate audio with DIA")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()