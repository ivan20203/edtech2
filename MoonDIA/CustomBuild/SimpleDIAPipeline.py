#!/usr/bin/env python3
"""
Simple DIA Pipeline
Takes DAC tokens and uses DIA's DAC decoder to generate audio.
This is a simplified version that avoids import issues.
"""

import sys
import os
import torch
import numpy as np
import soundfile as sf
import torchaudio

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# SIMPLE DIA DAC DECODER
# ============================================================================

class SimpleDIADACDecoder:
    """Simplified DIA DAC decoder that only handles the decoding part."""
    
    def __init__(self):
        self.device = device
        self._load_dac_model()
    
    def _load_dac_model(self):
        """Loads the Descript Audio Codec (DAC) model."""
        import dac
        
        try:
            print("  Downloading and loading DAC model...")
            dac_model_path = dac.utils.download()
            self.dac_model = dac.DAC.load(dac_model_path).to(self.device)
            self.dac_model.eval()
            print("  ✅ DAC model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load DAC model: {e}")
    
    def decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Decodes the given DAC codes into an output audio waveform.
        This is the core DIA decoding logic.
        
        Args:
            audio_codes: DAC tokens tensor [T, C] where T is time frames, C is channels
            
        Returns:
            audio: Audio tensor [samples] at 44.1kHz
        """
        # Ensure codes are on correct device
        audio_codes = audio_codes.to(self.device)
        
        # Use DIA's exact decoding logic
        with torch.no_grad():
            audio_codes = audio_codes.unsqueeze(0).transpose(1, 2)  # [1, C, T]
            audio_values, _, _ = self.dac_model.quantizer.from_codes(audio_codes)
            audio_values = self.dac_model.decode(audio_values)
            audio_values: torch.Tensor
            return audio_values.squeeze()


# ============================================================================
# SIMPLE DIA PIPELINE
# ============================================================================

class SimpleDIAPipeline:
    """
    Simple pipeline that uses DIA's DAC decoder to generate audio from DAC tokens.
    This focuses on the core DIA functionality without complex model loading.
    """
    
    def __init__(self):
        """Initialize the Simple DIA Pipeline."""
        print("Initializing Simple DIA Pipeline...")
        
        # Set device
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize DAC decoder
        print("Loading DAC decoder...")
        self.dac_decoder = SimpleDIADACDecoder()
        print("✅ DAC decoder loaded")
        
        print("✅ Simple DIA Pipeline initialized")
    
    def dac_tokens_to_audio(self, dac_tokens, output_path=None):
        """
        Convert DAC tokens to audio using DIA's DAC decoder.
        
        Args:
            dac_tokens: DAC tokens tensor [T, C] where T is time frames, C is channels
            output_path: Path to save audio (optional)
            
        Returns:
            audio: Generated audio tensor
        """
        print(f"Converting DAC tokens to audio using DIA...")
        print(f"  DAC tokens shape: {dac_tokens.shape}")
        
        # Convert to tensor if needed
        if isinstance(dac_tokens, np.ndarray):
            dac_tokens = torch.tensor(dac_tokens, dtype=torch.long)
        
        # Use DIA's DAC decoder to convert tokens to audio
        audio = self.dac_decoder.decode(dac_tokens)
        print(f"  Generated audio shape: {audio.shape}")
        
        # Save if output path provided
        if output_path is None:
            output_path = "dia_generated_audio.wav"
        
        # Save audio
        sf.write(output_path, audio.cpu().numpy(), 44100)
        print(f"  Saved audio to {output_path}")
        
        return audio
    
    def process_dac_tokens_file(self, dac_tokens_path, output_path=None):
        """
        Process a saved DAC tokens file and generate audio using DIA.
        
        Args:
            dac_tokens_path: Path to the .npy file containing DAC tokens
            output_path: Path to save the audio file (optional)
            
        Returns:
            audio: Generated audio
        """
        print(f"Processing DAC tokens from {dac_tokens_path}")
        
        # Load DAC tokens
        dac_tokens = np.load(dac_tokens_path)
        print(f"Loaded DAC tokens: {dac_tokens.shape}")
        
        # Generate audio using DIA
        audio = self.dac_tokens_to_audio(dac_tokens, output_path)
        
        return audio
    
    def process_multiple_dac_tokens(self, dac_tokens_dir=".", output_dir="."):
        """
        Process all DAC tokens files in a directory using DIA.
        
        Args:
            dac_tokens_dir: Directory containing DAC tokens files
            output_dir: Directory to save audio files
        """
        print(f"Processing all DAC tokens in {dac_tokens_dir}")
        
        # Find all DAC tokens files
        dac_files = [f for f in os.listdir(dac_tokens_dir) 
                    if f.endswith('.npy') and ('dac_tokens' in f or 'generated_dac_tokens' in f)]
        
        if not dac_files:
            print("No DAC tokens files found")
            return
        
        print(f"Found {len(dac_files)} DAC tokens files")
        
        for dac_file in dac_files:
            dac_path = os.path.join(dac_tokens_dir, dac_file)
            output_file = dac_file.replace('.npy', '_dia_generated.wav')
            output_path = os.path.join(output_dir, output_file)
            
            print(f"\nProcessing {dac_file}...")
            try:
                self.process_dac_tokens_file(dac_path, output_path)
            except Exception as e:
                print(f"Error processing {dac_file}: {e}")


def main():
    """Test the Simple DIA Pipeline."""
    print("Testing Simple DIA Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = SimpleDIAPipeline()
    
    # Look for DAC tokens files
    print("\n1. Looking for DAC tokens files...")
    
    # Check for the specific file we generated
    if os.path.exists("generated_dac_tokens.npy"):
        print("Found generated_dac_tokens.npy")
        
        # Process the file with DIA
        audio = pipeline.process_dac_tokens_file(
            "generated_dac_tokens.npy", 
            "dia_generated_audio.wav"
        )
        
        print(f"✅ Successfully generated audio using DIA")
        
    else:
        print("No generated_dac_tokens.npy found")
        print("Processing all DAC tokens files in current directory...")
        pipeline.process_multiple_dac_tokens()
    
    print("\n✅ Simple DIA pipeline test completed!")


if __name__ == "__main__":
    main()