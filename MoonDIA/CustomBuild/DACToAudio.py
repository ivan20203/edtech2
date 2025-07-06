#!/usr/bin/env python3
"""
DAC Tokens to Audio Generator
Takes DAC tokens from SemantiCodecToDAC.py and generates audio using DIA's DAC decoder.
Uses the actual DIA code directly without external dependencies.
"""

import sys
import os
import torch
import numpy as np
import soundfile as sf
import time
from enum import Enum

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# DIA UTILITIES (copied from DIAAudio)
# ============================================================================

DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE_RATIO = 512


def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        if self == ComputeDtype.FLOAT32:
            return torch.float32
        elif self == ComputeDtype.FLOAT16:
            return torch.float16
        elif self == ComputeDtype.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")


# ============================================================================
# SIMPLIFIED DIA DAC DECODER
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
        This is copied directly from DIA's _decode method.
        
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
# DAC TO AUDIO GENERATOR
# ============================================================================

class DACToAudioGenerator:
    """
    Generator that converts DAC tokens to audio using DIA's DAC decoder.
    """
    
    def __init__(self):
        """
        Initialize the DAC to Audio generator.
        """
        print("Initializing DAC to Audio Generator...")
        
        # Set device
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize DAC decoder
        print("Loading DAC decoder...")
        self.dac_decoder = SimpleDIADACDecoder()
        print("✅ DAC decoder loaded")
        
        print("✅ DAC to Audio Generator initialized successfully")
    
    def dac_tokens_to_audio(self, dac_tokens):
        """
        Convert DAC tokens to audio using DIA's DAC decoder.
        
        Args:
            dac_tokens: DAC tokens tensor [T, C] where T is time frames, C is channels
            
        Returns:
            audio: Audio tensor [samples] at 44.1kHz
        """
        print(f"Converting DAC tokens to audio...")
        print(f"  DAC tokens shape: {dac_tokens.shape}")
        
        # Use DAC decoder to convert tokens to audio
        audio = self.dac_decoder.decode(dac_tokens)
        
        print(f"  Generated audio shape: {audio.shape}")
        return audio
    
    def save_audio(self, audio, output_path, sample_rate=44100):
        """
        Save audio to file.
        
        Args:
            audio: Audio tensor or numpy array
            output_path: Path to save the audio file
            sample_rate: Sample rate (default 44.1kHz for DAC)
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Save using soundfile
        sf.write(output_path, audio, sample_rate)
        print(f"✅ Audio saved to {output_path}")
    
    def process_dac_tokens_file(self, dac_tokens_path, output_path=None):
        """
        Process a saved DAC tokens file and generate audio.
        
        Args:
            dac_tokens_path: Path to the .npy file containing DAC tokens
            output_path: Path to save the audio file (optional)
            
        Returns:
            audio: Generated audio tensor
        """
        print(f"Processing DAC tokens from {dac_tokens_path}")
        
        # Load DAC tokens
        dac_tokens = np.load(dac_tokens_path)
        print(f"Loaded DAC tokens: {dac_tokens.shape}")
        
        # Convert to tensor
        dac_tokens = torch.tensor(dac_tokens, dtype=torch.long)
        
        # Generate audio
        audio = self.dac_tokens_to_audio(dac_tokens)
        
        # Save if output path provided
        if output_path is None:
            output_path = dac_tokens_path.replace('.npy', '.wav')
        
        self.save_audio(audio, output_path)
        
        return audio
    
    def process_multiple_dac_tokens(self, dac_tokens_dir=".", output_dir="."):
        """
        Process all DAC tokens files in a directory.
        
        Args:
            dac_tokens_dir: Directory containing DAC tokens files
            output_dir: Directory to save audio files
        """
        print(f"Processing all DAC tokens in {dac_tokens_dir}")
        
        # Find all DAC tokens files
        dac_files = [f for f in os.listdir(dac_tokens_dir) 
                    if f.startswith(('dac_tokens', 'generated_dac_tokens')) and f.endswith('.npy')]
        
        if not dac_files:
            print("No DAC tokens files found")
            return
        
        print(f"Found {len(dac_files)} DAC tokens files")
        
        for dac_file in dac_files:
            dac_path = os.path.join(dac_tokens_dir, dac_file)
            output_file = dac_file.replace('.npy', '.wav')
            output_path = os.path.join(output_dir, output_file)
            
            print(f"\nProcessing {dac_file}...")
            try:
                self.process_dac_tokens_file(dac_path, output_path)
            except Exception as e:
                print(f"Error processing {dac_file}: {e}")


def main():
    """Test the DAC to Audio Generator."""
    print("Testing DAC to Audio Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = DACToAudioGenerator()
    
    # Look for DAC tokens files
    print("\n1. Looking for DAC tokens files...")
    
    # Check for the specific file we generated
    if os.path.exists("generated_dac_tokens.npy"):
        print("Found generated_dac_tokens.npy")
        
        # Process the file
        audio = generator.process_dac_tokens_file(
            "generated_dac_tokens.npy", 
            "generated_audio.wav"
        )
        
        print(f"✅ Successfully generated audio: {audio.shape}")
        
    else:
        print("No generated_dac_tokens.npy found")
        print("Processing all DAC tokens files in current directory...")
        generator.process_multiple_dac_tokens()
    
    print("\n✅ DAC to Audio test completed!")


if __name__ == "__main__":
    main()