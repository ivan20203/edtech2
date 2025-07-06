#!/usr/bin/env python3
"""
Generate MoonCast semantic tokens from text.
Run this in the MoonCast environment.
"""

import numpy as np
import sys
import os

try:
    from TextToSemantic import TextToSemantic
    print("✅ Successfully imported local TextToSemantic")
except ImportError as e:
    print(f"❌ Error importing TextToSemantic: {e}")
    print("Make sure TextToSemantic.py is in the current directory")
    sys.exit(1)

def generate_semantic_tokens(text: str, max_tokens: int = 200):
    """
    Generate MoonCast semantic tokens from text.
    """
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
        return None

def main():
    """
    Main function to generate semantic tokens.
    """
    print("MoonCast Semantic Token Generator")
    print("="*50)
    
    # Get text input
    test_text = input("Enter text (or press Enter for default): ").strip()
    if not test_text:
        test_text = "Hello world. This is a test of the semantic to DAC mapper."
    
    # Generate tokens
    semantic_tokens = generate_semantic_tokens(test_text)
    
    if semantic_tokens is not None:
        # Save to file
        output_file = "semantic_tokens.npy"
        np.save(output_file, semantic_tokens)
        print(f"✅ Saved {len(semantic_tokens)} semantic tokens to {output_file}")
        
        # Show some stats
        print(f"\nToken statistics:")
        print(f"  - Number of tokens: {len(semantic_tokens)}")
        print(f"  - Token range: {semantic_tokens.min()} to {semantic_tokens.max()}")
        print(f"  - Unique tokens: {len(np.unique(semantic_tokens))}")
        
        print(f"\nThe semantic tokens are ready to use with test_mapper.py")
    else:
        print("❌ Failed to generate semantic tokens")

if __name__ == "__main__":
    main() 