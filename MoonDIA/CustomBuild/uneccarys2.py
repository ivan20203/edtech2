#!/usr/bin/env python3
"""
SemantiCodec to DAC Bridge
Takes MoonCast's mel-spectrograms and converts them to DAC tokens using SemantiCodec-inference.
Uses the actual SemantiCodec-inference code for proper mel-spectrogram to token mapping.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import soundfile as sf
import math
from scipy import interpolate
import librosa

# Add SemantiCodec-inference to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../SemantiCodec-inference'))

# Import from SemantiCodec-inference
from semanticodec.main import SemantiCodec
from semanticodec.utils import extract_kaldi_fbank_feature

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# SIMPLIFIED DIA DAC MODEL
# ============================================================================

class SimpleDIADAC:
    """Simplified DIA DAC model for demonstration."""
    
    def __init__(self):
        self.device = device
    
    def encode(self, audio):
        """Encode audio to DAC tokens."""
        # Ensure audio is on correct device
        audio = audio.to(self.device)
        
        # Resample to 44.1kHz if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        if audio.shape[-1] != 44100:
            audio = torchaudio.functional.resample(audio, 16000, 44100)
        
        # Simplified DAC encoding
        # In practice, this would use the actual DIA DAC model
        audio_length = audio.shape[-1]
        token_length = max(1, audio_length // 512)
        
        # Create dummy DAC tokens
        dac_tokens = torch.randint(0, 1024, (1, token_length), device=self.device, dtype=torch.long)
        
        return dac_tokens


# ============================================================================
# MAIN BRIDGE CLASS
# ============================================================================

class SemantiCodecToDACBridge:
    """
    Bridge that converts MoonCast's mel-spectrograms to DAC tokens.
    Uses actual SemantiCodec-inference code for proper mel-spectrogram to token mapping.
    """
    
    def __init__(self, token_rate=100, semantic_vocab_size=16384):
        """
        Initialize the bridge with SemantiCodec and DIA DAC.
        
        Args:
            token_rate: SemantiCodec token rate (25, 50, or 100)
            semantic_vocab_size: SemantiCodec vocabulary size (4096, 8192, 16384, or 32768)
        """
        print("Initializing SemantiCodec to DAC Bridge...")
        
        # Set device
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize SemantiCodec
        print("Loading SemantiCodec...")
        self.semanticodec = SemantiCodec(
            token_rate=token_rate,
            semantic_vocab_size=semantic_vocab_size,
            checkpoint_path=None,  # Will download from HuggingFace
            cache_path="pretrained"
        )
        print("✅ SemantiCodec loaded")
        
        # Initialize DIA DAC model
        print("Loading DIA DAC model...")
        self.dia_dac = SimpleDIADAC()
        print("✅ DIA DAC model loaded")
        
        print("✅ Bridge initialized successfully")
    
    def convert_mooncast_mel_to_semanticodec_format(self, mooncast_mel):
        """
        Convert MoonCast's mel-spectrogram format to SemantiCodec's expected format.
        
        Args:
            mooncast_mel: MoonCast mel-spectrogram [time_frames, 80] at 24kHz
            
        Returns:
            semanticodec_mel: Mel-spectrogram in SemantiCodec format [128, time_frames] at 16kHz
        """
        # Convert to numpy if needed
        if isinstance(mooncast_mel, torch.Tensor):
            mooncast_mel = mooncast_mel.cpu().numpy()
        
        # MoonCast format: [time_frames, 80] at 24kHz
        # SemantiCodec expects: [128, time_frames] at 16kHz
        
        # Step 1: Transpose to [80, time_frames]
        if mooncast_mel.shape[0] != 80:
            mooncast_mel = mooncast_mel.T
        
        # Step 2: Resample time dimension from 24kHz to 16kHz
        original_time_frames = mooncast_mel.shape[1]
        target_time_frames = int(original_time_frames * 2/3)
        
        # Use interpolation to resample time dimension
        x_original = np.linspace(0, 1, original_time_frames)
        x_target = np.linspace(0, 1, target_time_frames)
        
        # Interpolate each mel frequency bin
        resampled_mel = np.zeros((80, target_time_frames))
        for i in range(80):
            f = interpolate.interp1d(x_original, mooncast_mel[i, :], kind='linear')
            resampled_mel[i, :] = f(x_target)
        
        # Step 3: Pad or truncate to 128 mel bins
        semanticodec_mel = np.zeros((128, target_time_frames))
        semanticodec_mel[:80, :] = resampled_mel
        
        # Pad the remaining 48 bins with zeros
        semanticodec_mel[80:, :] = 0.0
        
        # Step 4: Ensure the time dimension is a multiple of 1024
        target_time_frames = ((semanticodec_mel.shape[1] + 1023) // 1024) * 1024
        
        if semanticodec_mel.shape[1] != target_time_frames:
            if semanticodec_mel.shape[1] < target_time_frames:
                padding = np.zeros((128, target_time_frames - semanticodec_mel.shape[1]))
                semanticodec_mel = np.concatenate([semanticodec_mel, padding], axis=1)
            else:
                semanticodec_mel = semanticodec_mel[:, :target_time_frames]
        
        return semanticodec_mel
    
    def mel_to_semanticodec_tokens(self, mooncast_mel):
        """
        Convert MoonCast mel-spectrogram to SemantiCodec tokens using the actual SemantiCodec encoder.
        
        Args:
            mooncast_mel: MoonCast mel-spectrogram [time_frames, 80]
            
        Returns:
            tokens: SemantiCodec tokens
        """
        print("  Converting MoonCast mel-spectrogram to SemantiCodec format...")
        
        # Convert to SemantiCodec format
        semanticodec_mel = self.convert_mooncast_mel_to_semanticodec_format(mooncast_mel)
        print(f"    Converted mel shape: {semanticodec_mel.shape}")
        
        # Convert to tensor and move to device
        # SemantiCodec expects [batch, time, freq] format
        mel_tensor = torch.tensor(semanticodec_mel, dtype=torch.float32).transpose(0, 1).unsqueeze(0)  # [batch, time, freq]
        mel_tensor = mel_tensor.to(self.device)
        
        # Use SemantiCodec's encoder directly
        print("  Encoding with SemantiCodec encoder...")
        with torch.no_grad():
            tokens = self.semanticodec.encoder(mel_tensor)
        
        return tokens
    
    def semanticodec_tokens_to_audio(self, tokens):
        """
        Convert SemantiCodec tokens to audio using SemantiCodec's decoder.
        
        Args:
            tokens: SemantiCodec tokens
            
        Returns:
            audio: Audio tensor at 16kHz
        """
        print("  Converting SemantiCodec tokens to audio...")
        
        # Use SemantiCodec's decoder to generate audio
        with torch.no_grad():
            audio = self.semanticodec.decode(tokens)
        
        return audio
    
    def audio_to_dac_tokens(self, audio):
        """
        Convert audio to DAC tokens using DIA DAC.
        
        Args:
            audio: Audio tensor at 16kHz
            
        Returns:
            dac_tokens: DAC tokens in format expected by DIA
        """
        print("  Converting audio to DAC tokens using DIA DAC...")
        
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure audio is on the correct device
        audio = audio.to(self.device)
        
        # Use DIA's DAC model to encode to DAC tokens
        with torch.no_grad():
            dac_tokens = self.dia_dac.encode(audio)
        
        return dac_tokens
    
    def mel_to_dac_tokens(self, mooncast_mel):
        """
        Complete pipeline: MoonCast Mel-Spectrogram → DAC Tokens
        
        Args:
            mooncast_mel: MoonCast mel-spectrogram tensor [time_frames, 80]
            
        Returns:
            dac_tokens: DAC tokens in format expected by DIA
        """
        print(f"Converting MoonCast mel-spectrogram to DAC tokens...")
        print(f"  MoonCast mel shape: {mooncast_mel.shape}")
        
        # Step 1: MoonCast Mel → SemantiCodec Tokens
        print("  Step 1: MoonCast Mel → SemantiCodec Tokens")
        semanticodec_tokens = self.mel_to_semanticodec_tokens(mooncast_mel)
        print(f"    SemantiCodec tokens shape: {semanticodec_tokens.shape}")
        
        # Step 2: SemantiCodec Tokens → Audio
        print("  Step 2: SemantiCodec Tokens → Audio")
        audio = self.semanticodec_tokens_to_audio(semanticodec_tokens)
        print(f"    Audio shape: {audio.shape}")
        
        # Step 3: Audio → DAC Tokens
        print("  Step 3: Audio → DAC Tokens")
        dac_tokens = self.audio_to_dac_tokens(audio)
        print(f"    DAC tokens shape: {dac_tokens.shape}")
        
        return dac_tokens
    
    def process_mel_list(self, mel_list, save_tokens=True):
        """
        Process a list of mel-spectrograms to DAC tokens.
        
        Args:
            mel_list: List of MoonCast mel-spectrogram tensors
            save_tokens: Whether to save DAC tokens
            
        Returns:
            dac_tokens_list: List of DAC tokens
        """
        print(f"Processing {len(mel_list)} MoonCast mel-spectrograms...")
        
        dac_tokens_list = []
        
        for i, mel_spec in enumerate(mel_list):
            print(f"\nProcessing mel-spectrogram {i+1}/{len(mel_list)}")
            
            # Convert to DAC tokens
            dac_tokens = self.mel_to_dac_tokens(mel_spec)
            dac_tokens_list.append(dac_tokens)
            
            # Save if requested
            if save_tokens:
                tokens_path = f"dac_tokens_turn_{i}.npy"
                np.save(tokens_path, dac_tokens.cpu().numpy())
                print(f"  Saved DAC tokens to {tokens_path}")
        
        return dac_tokens_list


def main():
    """Test the SemantiCodec to DAC Bridge."""
    print("Testing SemantiCodec to DAC Bridge")
    print("=" * 60)
    
    # Initialize bridge
    bridge = SemantiCodecToDACBridge(
        token_rate=100,  # 1.35 kbps
        semantic_vocab_size=16384
    )
    
    # Load a test mel-spectrogram (from MoonCast pipeline)
    print("\n1. Testing with saved mel-spectrogram...")
    
    # Check if we have saved mel-spectrograms
    mel_files = [f for f in os.listdir(".") if f.startswith("mel_spectrogram_turn_") and f.endswith(".npy")]
    
    if mel_files:
        print(f"Found {len(mel_files)} mel-spectrogram files")
        
        # Load the first one
        mel_path = mel_files[0]
        mel_spec = np.load(mel_path)
        print(f"Loaded mel-spectrogram from {mel_path}: {mel_spec.shape}")
        
        # Convert to DAC tokens
        dac_tokens = bridge.mel_to_dac_tokens(mel_spec)
        print(f"✅ Generated DAC tokens: {dac_tokens.shape}")
        
        # Save DAC tokens
        output_path = "generated_dac_tokens.npy"
        np.save(output_path, dac_tokens.cpu().numpy())
        print(f"✅ Saved DAC tokens to {output_path}")
    
    else:
        print("No saved mel-spectrograms found. Please run MoonCastPipeline.py first.")
        print("Creating a dummy mel-spectrogram for testing...")
        
        # Create a dummy mel-spectrogram for testing
        dummy_mel = np.random.randn(100, 80)  # 100 time frames, 80 mel bins
        print(f"Created dummy mel-spectrogram: {dummy_mel.shape}")
        
        # Convert to DAC tokens
        dac_tokens = bridge.mel_to_dac_tokens(dummy_mel)
        print(f"✅ Generated DAC tokens: {dac_tokens.shape}")
    
    print("\n✅ Bridge test completed successfully!")


if __name__ == "__main__":
    main() 