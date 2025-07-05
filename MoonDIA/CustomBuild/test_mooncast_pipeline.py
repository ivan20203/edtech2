#!/usr/bin/env python3
"""
Test script for MoonCastPipeline.
Tests the complete pipeline: Text → Audio → Mel-Spectrograms using MoonCast's exact code.
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MoonCastPipeline import MoonCastPipeline


def test_single_text():
    """Test single text input."""
    print("=" * 50)
    print("Testing Single Text Input")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MoonCastPipeline()
    
    # Test text
    text = "Hello, how are you today?"
    print(f"Input text: {text}")
    
    # Generate audio and mel-spectrogram
    audio, mel = pipeline.single_text_to_mel(
        text, 
        role="0", 
        save_audio=True, 
        save_mel=True
    )
    
    print(f"✅ Generated audio shape: {audio.shape}")
    print(f"✅ Generated mel-spectrogram shape: {mel.shape}")
    print(f"Mel-spectrogram range: [{mel.min():.3f}, {mel.max():.3f}]")
    
    return audio, mel


def test_dialogue():
    """Test dialogue input."""
    print("\n" + "=" * 50)
    print("Testing Dialogue Input")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = MoonCastPipeline()
    
    # Test dialogue
    dialogue = [
        {"role": "0", "text": "Hello, how are you?"},
        {"role": "1", "text": "I'm doing great, thank you!"},
        {"role": "0", "text": "That's wonderful to hear."}
    ]
    
    print("Input dialogue:")
    for i, turn in enumerate(dialogue):
        print(f"  Turn {i+1} ({turn['role']}): {turn['text']}")
    
    # Generate audio and mel-spectrograms
    audio_list, mel_list = pipeline.text_to_mel(
        dialogue, 
        save_audio=True, 
        save_mel=True
    )
    
    print(f"\n✅ Generated {len(audio_list)} audio sequences")
    print(f"✅ Generated {len(mel_list)} mel-spectrogram sequences")
    
    for i, (turn, audio, mel) in enumerate(zip(dialogue, audio_list, mel_list)):
        print(f"  Turn {i+1} ({turn['role']}): audio {audio.shape}, mel {mel.shape}")
    
    return audio_list, mel_list


def main():
    """Run all tests."""
    print("Testing MoonCast Pipeline")
    print("=" * 60)
    
    try:
        # Test single text
        single_audio, single_mel = test_single_text()
        
        # Test dialogue
        dialogue_audios, dialogue_mels = test_dialogue()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("=" * 60)
        
        # Summary
        print(f"✅ Single text: audio {single_audio.shape}, mel {single_mel.shape}")
        print(f"✅ Dialogue: {len(dialogue_audios)} audio sequences, {len(dialogue_mels)} mel sequences")
        
        print("\nGenerated files:")
        print("  - turn_1_0_audio.wav (single text audio)")
        print("  - turn_1_0_mel.npy (single text mel-spectrogram)")
        print("  - turn_1_0_audio.wav, turn_2_1_audio.wav, turn_3_0_audio.wav (dialogue audio)")
        print("  - turn_1_0_mel.npy, turn_2_1_mel.npy, turn_3_0_mel.npy (dialogue mel-spectrograms)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 All tests passed! The MoonCast pipeline is working correctly.")
    else:
        print("\n💥 Some tests failed. Please check the error messages above.") 