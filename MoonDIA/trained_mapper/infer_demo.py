"""
infer_demo.py
============
Demo script for using the trained Semantic→DAC mapper.

This script demonstrates the complete pipeline:
1. Text → MoonCast semantic tokens
2. Semantic tokens → DAC codes (via trained mapper)
3. DAC codes → Audio (via DIA DAC decoder)
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
import soundfile as sf

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from TextToSemantic import TextToSemantic
from semantic_to_dac_mapper import SemanticToDACMapper

# Set up DIA environment paths
dia_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dia", ".diaenv")
dia_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dia")

# Add DIA's site-packages to Python path for imports
dia_site_packages = os.path.join(dia_env_path, 'lib', 'python3.10', 'site-packages')
if os.path.exists(dia_site_packages):
    sys.path.insert(0, dia_site_packages)

# Add DIA module path
sys.path.append(dia_path)

# Now import DIA
from dia.model import Dia


class SemanticToDACInference:
    """
    Complete inference pipeline for Semantic→DAC mapping.
    """
    
    def __init__(
        self,
        mapper_checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            mapper_checkpoint_path: Path to trained mapper checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Initialize MoonCast semantic extractor
        print("Loading MoonCast semantic extractor...")
        self.semantic_extractor = TextToSemantic()
        
        # Load trained mapper
        print("Loading trained semantic→DAC mapper...")
        self.mapper = SemanticToDACMapper()
        checkpoint = torch.load(mapper_checkpoint_path, map_location=device)
        self.mapper.load_state_dict(checkpoint['model_state_dict'])
        self.mapper.to(device)
        self.mapper.eval()
        
        # Initialize DIA for DAC decoding
        print("Loading DIA DAC decoder...")
        self.dia_model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B",
            compute_dtype="float16",
            load_dac=True  # We need DAC for decoding
        )
        
        print("✓ Inference pipeline ready!")
    
    def text_to_semantic_tokens(self, text: str) -> np.ndarray:
        """
        Convert text to MoonCast semantic tokens.
        
        Args:
            text: Input text
            
        Returns:
            Semantic tokens as numpy array
        """
        semantic_tokens = self.semantic_extractor.generate_semantic_tokens_simple(text)
        return semantic_tokens[0]  # Return first (and only) result
    
    def semantic_tokens_to_dac_codes(self, semantic_tokens: np.ndarray) -> np.ndarray:
        """
        Convert semantic tokens to DAC codes using the trained mapper.
        
        Args:
            semantic_tokens: MoonCast semantic tokens
            
        Returns:
            DAC codes as numpy array
        """
        # Convert to tensor
        semantic_tensor = torch.tensor(semantic_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate DAC codes
        with torch.no_grad():
            dac_codes = self.mapper(semantic_tensor)
        
        # Convert back to numpy
        return dac_codes[0].cpu().numpy()
    
    def dac_codes_to_audio(self, dac_codes: np.ndarray) -> np.ndarray:
        """
        Convert DAC codes to audio using DIA's DAC decoder.
        
        Args:
            dac_codes: DAC codes from the mapper
            
        Returns:
            Audio waveform as numpy array
        """
        # Convert to tensor
        dac_tensor = torch.tensor(dac_codes, dtype=torch.long, device=self.dia_model.device)
        
        # Decode using DIA's DAC decoder
        audio = self.dia_model._decode(dac_tensor)
        
        return audio.cpu().numpy()
    
    def text_to_audio(self, text: str) -> np.ndarray:
        """
        Complete pipeline: text → semantic tokens → DAC codes → audio.
        
        Args:
            text: Input text
            
        Returns:
            Audio waveform as numpy array
        """
        print(f"Processing text: {text}")
        
        # Step 1: Text → Semantic tokens
        print("  → Converting text to semantic tokens...")
        semantic_tokens = self.text_to_semantic_tokens(text)
        print(f"    Generated {len(semantic_tokens)} semantic tokens")
        
        # Step 2: Semantic tokens → DAC codes
        print("  → Converting semantic tokens to DAC codes...")
        dac_codes = self.semantic_tokens_to_dac_codes(semantic_tokens)
        print(f"    Generated DAC codes shape: {dac_codes.shape}")
        
        # Step 3: DAC codes → Audio
        print("  → Converting DAC codes to audio...")
        audio = self.dac_codes_to_audio(dac_codes)
        print(f"    Generated audio length: {len(audio)} samples")
        
        return audio
    
    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int = 44100):
        """
        Save audio to file.
        
        Args:
            audio: Audio waveform
            output_path: Output file path
            sample_rate: Audio sample rate
        """
        sf.write(output_path, audio, sample_rate)
        print(f"✓ Audio saved to: {output_path}")


def demo_with_sample_text():
    """
    Run demo with sample text inputs.
    """
    print("Semantic→DAC Mapper Demo")
    print("========================")
    
    # Sample texts to test
    sample_texts = [
        "Hello, how are you today?",
        "The weather is beautiful outside.",
        "I love listening to music.",
        "Can you help me with this problem?",
        "Thank you for your assistance."
    ]
    
    # Initialize inference pipeline
    # Note: You'll need to train the mapper first and provide the checkpoint path
    checkpoint_path = "checkpoints/semantic_to_dac_mapper.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the mapper first using semantic_to_dac_mapper.py")
        return
    
    try:
        inference = SemanticToDACInference(checkpoint_path)
        
        # Process each sample text
        for i, text in enumerate(sample_texts):
            print(f"\n--- Sample {i+1} ---")
            
            # Generate audio
            audio = inference.text_to_audio(text)
            
            # Save audio
            output_path = f"demo_output_{i+1:02d}.wav"
            inference.save_audio(audio, output_path)
            
        print(f"\n✓ Demo completed! Generated {len(sample_texts)} audio files.")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


def interactive_demo():
    """
    Interactive demo where user can input text.
    """
    print("Interactive Semantic→DAC Mapper Demo")
    print("====================================")
    
    checkpoint_path = "checkpoints/semantic_to_dac_mapper.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the mapper first using semantic_to_dac_mapper.py")
        return
    
    try:
        inference = SemanticToDACInference(checkpoint_path)
        
        print("\nEnter text to convert to audio (or 'quit' to exit):")
        
        while True:
            text = input("\n> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            try:
                # Generate audio
                audio = inference.text_to_audio(text)
                
                # Save audio
                output_path = f"interactive_output_{len(os.listdir('.'))}.wav"
                inference.save_audio(audio, output_path)
                
            except Exception as e:
                print(f"❌ Error processing text: {e}")
        
        print("Goodbye!")
        
    except Exception as e:
        print(f"❌ Error during interactive demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic→DAC Mapper Demo")
    parser.add_argument("--mode", choices=["sample", "interactive"], default="sample",
                       help="Demo mode: sample (predefined texts) or interactive (user input)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/semantic_to_dac_mapper.pth",
                       help="Path to trained mapper checkpoint")
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        demo_with_sample_text()
    else:
        interactive_demo()
