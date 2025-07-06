#!/usr/bin/env python3
"""
Audio to DAC Bridge
Takes MoonCast's audio and converts it to DAC tokens using SemantiCodec-inference.
Works in the audio domain to avoid mel-spectrogram conversion issues.
"""

import sys
import os
import torch
import numpy as np
import torchaudio
import soundfile as sf
import time

# Add SemantiCodec-inference to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../SemantiCodec-inference'))

# Import from SemantiCodec-inference
from semanticodec.main import SemantiCodec

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# REAL DIA DAC MODEL
# ============================================================================

class RealDIADAC:
    """Real DIA DAC model using the actual Descript Audio Codec."""
    
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
    
    def encode(self, audio):
        """Encode audio to DAC tokens using the real DAC model.
        
        Args:
            audio: Audio tensor [samples] or [1, samples] at 44.1kHz
            
        Returns:
            dac_tokens: DAC tokens tensor [T, C] where T is time frames, C is channels
        """
        # Ensure audio is on correct device
        audio = audio.to(self.device)
        
        # Ensure audio is [1, samples] format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Audio should already be at 44.1kHz from previous steps
        # Use DAC model to encode
        with torch.no_grad():
            audio_data = self.dac_model.preprocess(audio, 44100)
            _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)
            # Return [T, C] format as expected by DIA
            return encoded_frame.squeeze(0).transpose(0, 1)


# ============================================================================
# MAIN BRIDGE CLASS
# ============================================================================

class AudioToDACBridge:
    """
    Bridge that converts MoonCast's audio to DAC tokens.
    Works in the audio domain: MoonCast audio → SemantiCodec → DAC tokens
    """
    
    def __init__(self, token_rate=100, semantic_vocab_size=16384):
        """
        Initialize the bridge with SemantiCodec and DIA DAC.
        
        Args:
            token_rate: SemantiCodec token rate (25, 50, or 100)
            semantic_vocab_size: SemantiCodec vocabulary size (4096, 8192, 16384, or 32768)
        """
        print("Initializing Audio to DAC Bridge...")
        
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
        self.dia_dac = RealDIADAC()
        print("✅ DIA DAC model loaded")
        
        print("✅ Bridge initialized successfully")
    
    def resample_audio(self, audio, orig_sr, target_sr):
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio tensor [samples] or [1, samples]
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            resampled_audio: Audio tensor at target sample rate
        """
        # Ensure audio is [1, samples] format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Resample using torchaudio
        resampled = torchaudio.functional.resample(audio, orig_sr, target_sr)
        return resampled
    
    def audio_to_semanticodec_tokens(self, mooncast_audio):
        """
        Convert MoonCast audio to SemantiCodec tokens.
        
        Args:
            mooncast_audio: MoonCast audio tensor [samples] at 24kHz
            
        Returns:
            tokens: SemantiCodec tokens
        """
        print("  Converting MoonCast audio to SemantiCodec tokens...")
        print(f"    MoonCast audio shape: {mooncast_audio.shape}")
        
        # Step 1: Resample from 24kHz to 16kHz (SemantiCodec expects 16kHz)
        print("    Resampling 24kHz → 16kHz...")
        audio_16k = self.resample_audio(mooncast_audio, 24000, 16000)
        print(f"    Audio 16kHz shape: {audio_16k.shape}")
        
        # Step 2: Save audio to temporary file and use SemantiCodec's encode method
        print("    Saving audio to temporary file...")
        temp_audio_path = "temp_audio_16k.wav"
        sf.write(temp_audio_path, audio_16k.squeeze().cpu().numpy(), 16000)
        
        # Step 3: Use SemantiCodec's encoder on the file
        print("    Encoding with SemantiCodec...")
        with torch.no_grad():
            tokens = self.semanticodec.encode(temp_audio_path)
        
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        print(f"    SemantiCodec tokens shape: {tokens.shape}")
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
        
        print(f"    SemantiCodec audio shape: {audio.shape}")
        return audio
    
    def audio_to_dac_tokens(self, audio):
        """
        Convert audio to DAC tokens using DIA DAC.
        
        Args:
            audio: Audio tensor or numpy array at 16kHz
            
        Returns:
            dac_tokens: DAC tokens in format expected by DIA
        """
        print("  Converting audio to DAC tokens using DIA DAC...")
        
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # Step 1: Resample from 16kHz to 44.1kHz (DAC expects 44.1kHz)
        print("    Resampling 16kHz → 44.1kHz...")
        audio_44k = self.resample_audio(audio, 16000, 44100)
        print(f"    Audio 44.1kHz shape: {audio_44k.shape}")
        
        # Step 2: Use DIA's DAC model to encode to DAC tokens
        with torch.no_grad():
            dac_tokens = self.dia_dac.encode(audio_44k)
        
        print(f"    DAC tokens shape: {dac_tokens.shape}")
        return dac_tokens
    
    def mooncast_audio_to_dac_tokens(self, mooncast_audio):
        """
        Complete pipeline: MoonCast Audio → DAC Tokens
        
        Args:
            mooncast_audio: MoonCast audio tensor [samples] at 24kHz
            
        Returns:
            dac_tokens: DAC tokens in format expected by DIA
        """
        print(f"Converting MoonCast audio to DAC tokens...")
        print(f"  MoonCast audio shape: {mooncast_audio.shape}")
        
        # Step 1: MoonCast Audio → SemantiCodec Tokens
        print("  Step 1: MoonCast Audio → SemantiCodec Tokens")
        semanticodec_tokens = self.audio_to_semanticodec_tokens(mooncast_audio)
        print(f"    SemantiCodec tokens shape: {semanticodec_tokens.shape}")
        
        # Step 2: SemantiCodec Tokens → Audio
        print("  Step 2: SemantiCodec Tokens → Audio")
        audio_16k = self.semanticodec_tokens_to_audio(semanticodec_tokens)
        print(f"    Audio 16kHz shape: {audio_16k.shape}")
        
        # Step 3: Audio → DAC Tokens
        print("  Step 3: Audio → DAC Tokens")
        dac_tokens = self.audio_to_dac_tokens(audio_16k)
        print(f"    DAC tokens shape: {dac_tokens.shape}")
        
        return dac_tokens
    
    def process_audio_list(self, audio_list, save_tokens=True):
        """
        Process a list of MoonCast audio to DAC tokens.
        
        Args:
            audio_list: List of MoonCast audio tensors
            save_tokens: Whether to save DAC tokens
            
        Returns:
            dac_tokens_list: List of DAC tokens
        """
        print(f"Processing {len(audio_list)} MoonCast audio clips...")
        
        dac_tokens_list = []
        
        for i, audio in enumerate(audio_list):
            print(f"\nProcessing audio clip {i+1}/{len(audio_list)}")
            
            # Convert to DAC tokens
            dac_tokens = self.mooncast_audio_to_dac_tokens(audio)
            dac_tokens_list.append(dac_tokens)
            
            # Save if requested
            if save_tokens:
                tokens_path = f"audio_dac_tokens_turn_{i}.npy"
                np.save(tokens_path, dac_tokens.cpu().numpy())
                print(f"  Saved DAC tokens to {tokens_path}")
        
        return dac_tokens_list


def main():
    """Test the Audio to DAC Bridge."""
    print("Testing Audio to DAC Bridge")
    print("=" * 60)
    
    # Initialize bridge
    bridge = AudioToDACBridge(
        token_rate=100,  # 1.35 kbps
        semantic_vocab_size=16384
    )
    
    # Load a test audio file (from MoonCast pipeline)
    print("\n1. Testing with saved audio...")
    
    # Check if we have saved audio files
    audio_files = [f for f in os.listdir(".") if f.startswith("turn_") and f.endswith("_audio.wav")]
    
    if audio_files:
        print(f"Found {len(audio_files)} audio files")
        
        # Load the first one
        audio_path = audio_files[0]
        print(f"Loading audio from {audio_path}")
        
        # Load audio using soundfile
        audio_data, sample_rate = sf.read(audio_path)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        
        print(f"Loaded audio: {audio_tensor.shape}, sample rate: {sample_rate}")
        
        # Convert to DAC tokens
        dac_tokens = bridge.mooncast_audio_to_dac_tokens(audio_tensor)
        print(f"✅ Generated DAC tokens: {dac_tokens.shape}")
        
        # Save DAC tokens
        output_path = "audio_generated_dac_tokens.npy"
        np.save(output_path, dac_tokens.cpu().numpy())
        print(f"✅ Saved DAC tokens to {output_path}")
    
    else:
        print("No saved audio files found. Please run MoonCastPipeline.py first.")
        print("Creating a dummy audio for testing...")
        
        # Create a dummy audio for testing (1 second of sine wave at 24kHz)
        t = torch.linspace(0, 1, 24000)
        dummy_audio = torch.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
        print(f"Created dummy audio: {dummy_audio.shape}")
        
        # Convert to DAC tokens
        dac_tokens = bridge.mooncast_audio_to_dac_tokens(dummy_audio)
        print(f"✅ Generated DAC tokens: {dac_tokens.shape}")
    
    print("\n✅ Bridge test completed successfully!")


if __name__ == "__main__":
    main() 