#!/usr/bin/env python3
"""
text_to_audio.py
================
A simple text-to-audio converter using MoonCast's pipeline.

This script:
1. Takes text input from user
2. Generates semantic tokens using TextToSemantic
3. Converts semantic tokens to audio using MoonCast's pipeline
4. Saves the audio file

Usage:
    python text_to_audio.py "Your text here"
    python text_to_audio.py --interactive
"""

import sys
import os
import torch
import numpy as np
import torchaudio
import argparse
import time
from pathlib import Path
from typing import Optional

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
mooncast_path = os.path.join(current_dir, "..", "..", "MoonCast")
sys.path.append(current_dir)  # For TextToSemantic
sys.path.append(mooncast_path)  # For MoonCast modules

# Import required components
from TextToSemantic import TextToSemantic
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_detokenizer.audio_detokenizer import get_audio_detokenizer, detokenize_noref
from transformers import AutoModelForCausalLM, GenerationConfig


class TextToAudioConverter:
    """
    Complete text-to-audio pipeline using MoonCast's components.
    """
    
    def __init__(self):
        """Initialize the text-to-audio pipeline."""
        print("Initializing Text-to-Audio Converter...")
        
        # Store original working directory
        original_cwd = os.getcwd()
        
        try:
            # Initialize TextToSemantic (for text → semantic tokens)
            print("  Loading TextToSemantic...")
            self.text_to_semantic = TextToSemantic()
            
            # Change to MoonCast directory for initialization
            os.chdir(mooncast_path)
            print(f"  Changed working directory to: {mooncast_path}")
            
            # Initialize MoonCast components (for semantic tokens → audio)
            print("  Loading MoonCast tokenizer...")
            self.tokenizer, self.extra_tokens = get_tokenizer_and_extra_tokens()
            self.speech_token_offset = 163840
            
            # Special token IDs
            self.assistant_ids = self.tokenizer.encode("assistant")
            self.user_ids = self.tokenizer.encode("user")
            self.audio_ids = self.tokenizer.encode("audio")
            self.spk_0_ids = self.tokenizer.encode("0")
            self.spk_1_ids = self.tokenizer.encode("1")
            
            # Extra tokens
            self.msg_end = self.extra_tokens.msg_end
            self.user_msg_start = self.extra_tokens.user_msg_start
            self.assistant_msg_start = self.extra_tokens.assistant_msg_start
            self.name_end = self.extra_tokens.name_end
            self.media_begin = self.extra_tokens.media_begin
            self.media_content = self.extra_tokens.media_content
            self.media_end = self.extra_tokens.media_end
            
            # Load text2semantic model
            model_path = "resources/text2semantic"
            model_path = os.path.abspath(model_path)
            
            print(f"  Loading text2semantic model from: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="cuda:0", 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True, 
                force_download=False
            ).to(torch.cuda.current_device())
            
            # Generation config
            self.generate_config = GenerationConfig(
                max_new_tokens=200 * 50,
                do_sample=True,
                top_k=30,
                top_p=0.8,
                temperature=0.8,
                eos_token_id=self.media_end,
            )
            
            # Load audio detokenizer
            print("  Loading MoonCast detokenizer...")
            self.audio_detokenizer = get_audio_detokenizer()
            
            print("✅ Text-to-Audio Converter initialized successfully!")
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            print(f"  Restored working directory to: {original_cwd}")
    
    def text_to_semantic_tokens(self, text: str) -> np.ndarray:
        """
        Convert text to semantic tokens using TextToSemantic.
        
        Args:
            text: Input text to convert
            
        Returns:
            Semantic tokens as numpy array
        """
        print(f"Converting text to semantic tokens: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        start_time = time.time()
        semantic_tokens = self.text_to_semantic.generate_semantic_tokens_simple(text)[0].astype("int16")
        end_time = time.time()
        
        print(f"Generated {len(semantic_tokens)} semantic tokens in {end_time - start_time:.2f} seconds")
        return semantic_tokens
    
    def semantic_tokens_to_audio(self, tokens: np.ndarray) -> torch.Tensor:
        """
        Convert semantic tokens to audio using MoonCast's pipeline.
        
        Args:
            tokens: Semantic tokens as numpy array
            
        Returns:
            Audio tensor
        """
        print("Converting semantic tokens to audio...")
        
        # Convert to tensor format expected by MoonCast
        if isinstance(tokens, np.ndarray):
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.model.device)
        
        # Add batch dimension if needed
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        # Generate audio using MoonCast's detokenizer
        start_time = time.time()
        gen_speech_fm = detokenize_noref(self.audio_detokenizer, tokens)
        gen_speech_fm = gen_speech_fm.cpu()
        gen_speech_fm = gen_speech_fm / gen_speech_fm.abs().max()
        end_time = time.time()
        
        print(f"Generated audio in {end_time - start_time:.2f} seconds")
        return gen_speech_fm
    
    def convert_text_to_audio(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Complete text-to-audio conversion pipeline.
        
        Args:
            text: Input text to convert to audio
            output_path: Optional output path for audio file
            
        Returns:
            Path to the generated audio file
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"output_audio_{timestamp}.wav"
        
        print(f"\n{'='*60}")
        print(f"Converting text to audio")
        print(f"{'='*60}")
        print(f"Input text: {text}")
        print(f"Output file: {output_path}")
        
        try:
            # Step 1: Text → Semantic tokens
            semantic_tokens = self.text_to_semantic_tokens(text)
            
            # Step 2: Semantic tokens → Audio
            audio_tensor = self.semantic_tokens_to_audio(semantic_tokens)
            
            # Step 3: Save audio
            print(f"Saving audio to {output_path}...")
            
            # Convert to numpy and prepare for saving
            audio_np = audio_tensor.numpy()
            audio_tensor_for_save = torch.tensor(audio_np, dtype=torch.float32)
            
            if audio_tensor_for_save.dim() == 1:
                audio_tensor_for_save = audio_tensor_for_save.unsqueeze(0)
            
            # Save using torchaudio
            torchaudio.save(output_path, audio_tensor_for_save, 24000)
            
            # Calculate audio statistics
            duration = len(audio_np) / 24000
            rms = np.sqrt(np.mean(audio_np**2))
            peak = np.max(np.abs(audio_np))
            
            print(f"✅ Audio generated successfully!")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  RMS: {rms:.4f}")
            print(f"  Peak: {peak:.4f}")
            print(f"  Sample rate: 24kHz")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            raise


def interactive_mode(converter: TextToAudioConverter):
    """Run the converter in interactive mode."""
    print("\n🎤 Interactive Text-to-Audio Mode")
    print("Enter text to convert to audio (or 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            # Generate output filename
            timestamp = int(time.time())
            output_path = f"interactive_audio_{timestamp}.wav"
            
            # Convert text to audio
            result_path = converter.convert_text_to_audio(text, output_path)
            print(f"\n🎵 Audio saved to: {result_path}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert text to audio using MoonCast's pipeline")
    parser.add_argument("text", nargs="?", help="Text to convert to audio")
    parser.add_argument("--output", "-o", help="Output audio file path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check if we have text or interactive mode
    if not args.text and not args.interactive:
        print("Please provide text to convert or use --interactive mode.")
        print("Example: python text_to_audio.py 'Hello, world!'")
        print("Example: python text_to_audio.py --interactive")
        return
    
    try:
        # Initialize the converter
        converter = TextToAudioConverter()
        
        if args.interactive:
            # Run interactive mode
            interactive_mode(converter)
        else:
            # Convert single text
            output_path = converter.convert_text_to_audio(args.text, args.output)
            print(f"\n🎵 Audio saved to: {output_path}")
            
    except Exception as e:
        print(f"❌ Failed to initialize converter: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main() 