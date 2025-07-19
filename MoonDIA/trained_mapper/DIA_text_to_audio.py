#!/usr/bin/env python3
"""
DIA_text_to_audio.py
====================
Advanced text-to-audio converter using DIA's pipeline with sophisticated duration calculation.

This script:
1. Takes text input from user
2. Analyzes text complexity (syllables, punctuation, sentence structure)
3. Calculates optimal tokens using multiple factors and speaking style
4. Generates audio using DIA's text-to-speech model
5. Saves the audio file

Advanced Duration Calculation:
- Word count with accurate tokenization
- Syllable counting for phonetic complexity
- Punctuation analysis for natural pauses
- Sentence complexity adjustment
- Speaking style multipliers (slow/natural/fast/very_fast)
- Formula: tokens = base_duration * style * syllables * punctuation * complexity * 100

Speaking Styles:
- slow: 30% slower (multiplier: 1.3)
- natural: Normal speed (multiplier: 1.0) - default
- fast: 20% faster (multiplier: 0.8)
- very_fast: 40% faster (multiplier: 0.6)

Example: "Hello, how are you today?" (5 words, 7 syllables, 2 punctuation marks)
- Base: 2.0 seconds
- Syllables: +0.2 seconds (more complex words)
- Punctuation: +0.1 seconds (natural pauses)
- Final: ~2.3 seconds ≈ 230 tokens

Usage:
    python DIA_text_to_audio.py "Your text here"
    python DIA_text_to_audio.py "Your text here" --wpm 120 --style fast
    python DIA_text_to_audio.py --interactive --style slow
    python DIA_text_to_audio.py "Complex text with many syllables!" --wpm 140 --style natural
"""

import sys
import os
import torch
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Optional

# Add DIA to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
dia_path = os.path.join(current_dir, "..", "..", "dia")
sys.path.append(dia_path)

# Import DIA
from dia.model import Dia


class DIATextToAudioConverter:
    """
    Complete text-to-audio pipeline using DIA's components.
    """
    
    def __init__(self):
        """Initialize the DIA text-to-audio pipeline."""
        print("Initializing DIA Text-to-Audio Converter...")
        
        # Check GPU availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        try:
            # Initialize DIA model with DAC decoder loaded
            print("  Loading DIA model...")
            self.model = Dia.from_pretrained(
                "nari-labs/Dia-1.6B", 
                compute_dtype="float16",
                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                load_dac=True  # Load DAC decoder for audio generation
            )
            
            print(f"  Model device: {self.model.device}")
            print(f"  Model dtype: {self.model.compute_dtype}")
            print("✅ DIA Text-to-Audio Converter initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing DIA model: {e}")
            raise
    
    def calculate_optimal_tokens(self, text: str, wpm: float = 150, speaking_style: str = "natural") -> int:
        """
        Calculate optimal number of tokens using advanced audio duration estimation.
        
        Args:
            text: Input text
            wpm: Words per minute (default: 150 WPM for natural speech)
            speaking_style: "slow", "natural", "fast", or "very_fast"
            
        Returns:
            Optimal number of tokens to generate
        """
        # Advanced word counting (handles punctuation, numbers, etc.)
        import re
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Count words more accurately (handles contractions, numbers, etc.)
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        # Count syllables for more accurate duration
        syllable_count = self._count_syllables(text)
        
        # Count punctuation marks that affect pacing
        punctuation_count = len(re.findall(r'[.!?;:,]', text))
        
        # Speaking style adjustments
        style_multipliers = {
            "slow": 1.3,      # 30% slower
            "natural": 1.0,   # Normal speed
            "fast": 0.8,      # 20% faster
            "very_fast": 0.6  # 40% faster
        }
        
        style_multiplier = style_multipliers.get(speaking_style, 1.0)
        
        # Base duration calculation using multiple factors
        base_duration = (word_count / wpm) * 60
        
        # Syllable-based adjustment (more syllables = longer duration)
        # Much more conservative adjustment
        syllable_factor = 1.0 + (syllable_count - word_count) * 0.02
        
        # Punctuation adjustment (more punctuation = slight pauses)
        punctuation_factor = 1.0 + (punctuation_count * 0.02)
        
        # Sentence complexity adjustment
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_sentence_length = word_count / max(sentence_count, 1)
        complexity_factor = 1.0 + (avg_sentence_length - 15) * 0.01  # Much smaller adjustment
        
        # Calculate final duration
        expected_duration_seconds = (
            base_duration * 
            style_multiplier * 
            syllable_factor * 
            punctuation_factor * 
            complexity_factor
        )
        
        # Sanity check: don't let adjustments more than double the base duration
        max_adjustment = 2.0
        if expected_duration_seconds > base_duration * max_adjustment:
            print(f"  ⚠️  Duration adjustment too large, capping at {max_adjustment}x base duration")
            expected_duration_seconds = base_duration * max_adjustment
        
        # DIA token rate: ~100 tokens per second of audio
        optimal_tokens = int(expected_duration_seconds * 100)
        
        # Apply reasonable bounds
        min_tokens = 200   # Minimum 2 seconds
        max_tokens = 2000  # Maximum 20 seconds
        
        optimal_tokens = max(min_tokens, min(optimal_tokens, max_tokens))
        
        # Detailed output
        print(f"  Word count: {word_count}")
        print(f"  Syllable count: {syllable_count}")
        print(f"  Sentence count: {sentence_count}")
        print(f"  Punctuation marks: {punctuation_count}")
        print(f"  Speaking style: {speaking_style} (multiplier: {style_multiplier:.2f})")
        print(f"  Base duration: {base_duration:.1f} seconds")
        print(f"  Adjusted duration: {expected_duration_seconds:.1f} seconds")
        print(f"  Calculated tokens: {optimal_tokens}")
        
        return optimal_tokens
    
    def _count_syllables(self, text: str) -> int:
        """
        Count syllables in text using a simple but effective algorithm.
        
        Args:
            text: Input text
            
        Returns:
            Syllable count
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Split into words
        words = text.split()
        
        total_syllables = 0
        
        for word in words:
            # Skip empty words
            if not word:
                continue
                
            # Count vowels (simple syllable approximation)
            vowels = re.findall(r'[aeiouy]', word)
            
            # Handle special cases
            if len(vowels) == 0:
                # No vowels, count as 1 syllable
                total_syllables += 1
            elif len(vowels) == 1:
                # Single vowel = 1 syllable
                total_syllables += 1
            else:
                # Multiple vowels, but need to handle diphthongs
                # Remove consecutive vowels (diphthongs)
                cleaned_vowels = []
                for i, vowel in enumerate(vowels):
                    if i == 0 or vowel != vowels[i-1]:
                        cleaned_vowels.append(vowel)
                
                # Handle silent 'e' at end
                if word.endswith('e') and len(cleaned_vowels) > 1:
                    cleaned_vowels = cleaned_vowels[:-1]
                
                total_syllables += max(1, len(cleaned_vowels))
        
        return total_syllables
    
    def clean_audio_end(self, audio: np.ndarray, sample_rate: int = 44100, method: str = "smart") -> np.ndarray:
        """
        Clean up artifacts at the end of the audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate (default: 44100 for DIA)
            method: Cleaning method - "smart", "aggressive", or "conservative"
            
        Returns:
            Cleaned audio array
        """
        if len(audio) == 0:
            return audio
        
        # Convert to float if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Method-specific parameters
        if method == "aggressive":
            lookback_time = 1.0  # Look back 1 second
            rms_threshold = 0.02  # Higher threshold
            audio_threshold = 0.01
            fade_time = 0.05  # Shorter fade
        elif method == "conservative":
            lookback_time = 0.3  # Look back 0.3 seconds
            rms_threshold = 0.005  # Lower threshold
            audio_threshold = 0.002
            fade_time = 0.2  # Longer fade
        else:  # smart (default)
            lookback_time = 0.5  # Look back 0.5 seconds
            rms_threshold = 0.01  # Medium threshold
            audio_threshold = 0.005
            fade_time = 0.1  # Medium fade
        
        # Find the actual end of speech by looking for silence
        lookback_samples = min(int(lookback_time * sample_rate), len(audio))
        end_section = audio[-lookback_samples:]
        
        # Calculate RMS of the end section
        rms = np.sqrt(np.mean(end_section**2))
        
        # If the end is very quiet (likely artifacts), trim it
        if rms < rms_threshold:
            # Find the last point where audio is above threshold
            above_threshold = np.abs(audio) > audio_threshold
            if above_threshold.any():
                last_audio_point = np.where(above_threshold)[0][-1]
                
                # Add a fade-out
                fade_samples = int(fade_time * sample_rate)
                start_fade = max(0, last_audio_point - fade_samples)
                
                # Apply fade-out
                for i in range(start_fade, last_audio_point + 1):
                    fade_factor = 1.0 - ((i - start_fade) / fade_samples)
                    audio[i] *= fade_factor
                
                # Trim everything after the fade
                original_length = len(audio)
                audio = audio[:last_audio_point + 1]
                removed_samples = original_length - len(audio)
                
                if removed_samples > 0:
                    print(f"  Cleaned audio ({method}): removed {removed_samples} samples ({removed_samples/sample_rate:.2f}s) from end")
        
        return audio
    
    def text_to_audio(self, text: str, wpm: float = 150, speaking_style: str = "natural", clean_method: str = "smart") -> np.ndarray:
        """
        Convert text directly to audio using DIA's text-to-speech model.
        
        Args:
            text: Input text to convert
            wpm: Words per minute for token calculation (default: 150)
            speaking_style: "slow", "natural", "fast", or "very_fast" (default: "natural")
            clean_method: Audio cleaning method (default: "smart")
            
        Returns:
            Audio as numpy array
        """
        print(f"Converting text to audio: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Calculate optimal token count based on word count and speaking style
        optimal_tokens = self.calculate_optimal_tokens(text, wpm, speaking_style)
        
        start_time = time.time()
        
        # Generate audio directly using DIA's model (load_dac=True means it returns audio, not DAC codes)
        audio = self.model.generate(
            text=text,
            cfg_scale=5,
            temperature=1.5,
            top_p=1,
            cfg_filter_top_k=50,
            max_tokens=optimal_tokens,  # Use calculated optimal tokens
            verbose=False    # Disable verbose to reduce overhead
        )
        
        end_time = time.time()
        
        # Ensure we get the first (and only) result
        if isinstance(audio, list):
            audio = audio[0]
        
        # Convert to numpy if it's not already
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Clean up artifacts at the end
        print("Cleaning audio artifacts...")
        audio = self.clean_audio_end(audio, method=clean_method)
        
        print(f"Generated audio in {end_time - start_time:.2f} seconds")
        return audio
    
    def convert_text_to_audio(self, text: str, output_path: Optional[str] = None, wpm: float = 150, speaking_style: str = "natural", clean_method: str = "smart") -> str:
        """
        Complete text-to-audio conversion pipeline.
        
        Args:
            text: Input text to convert to audio
            output_path: Optional output path for audio file
            wpm: Words per minute for token calculation (default: 150)
            speaking_style: "slow", "natural", "fast", or "very_fast" (default: "natural")
            clean_method: Audio cleaning method (default: "smart")
            
        Returns:
            Path to the generated audio file
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"dia_output_audio_{timestamp}.wav"
        
        print(f"\n{'='*60}")
        print(f"Converting text to audio using DIA")
        print(f"{'='*60}")
        print(f"Input text: {text}")
        print(f"Output file: {output_path}")
        print(f"Speaking rate: {wpm} WPM")
        print(f"Speaking style: {speaking_style}")
        print(f"Cleaning method: {clean_method}")
        
        try:
            # Step 1: Text → Audio (direct conversion)
            audio_np = self.text_to_audio(text, wpm, speaking_style, clean_method)
            
            # Step 2: Save audio
            print(f"Saving audio to {output_path}...")
            
            # Save audio using DIA's save_audio method
            self.model.save_audio(output_path, audio_np)
            
            # Calculate audio statistics
            duration = len(audio_np) / 44100  # DIA uses 44.1kHz
            rms = np.sqrt(np.mean(audio_np**2))
            peak = np.max(np.abs(audio_np))
            
            print(f"✅ Audio generated successfully!")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  RMS: {rms:.4f}")
            print(f"  Peak: {peak:.4f}")
            print(f"  Sample rate: 44.1kHz")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            raise


def interactive_mode(converter: DIATextToAudioConverter, wpm: float = 150, speaking_style: str = "natural", clean_method: str = "smart"):
    """Run the converter in interactive mode."""
    print("\n🎤 Interactive DIA Text-to-Audio Mode")
    print(f"Speaking rate: {wpm} WPM")
    print(f"Speaking style: {speaking_style}")
    print(f"Cleaning method: {clean_method}")
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
            output_path = f"dia_interactive_audio_{timestamp}.wav"
            
            # Convert text to audio
            result_path = converter.convert_text_to_audio(text, output_path, wpm, speaking_style, clean_method)
            print(f"\n🎵 Audio saved to: {result_path}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert text to audio using DIA's pipeline with advanced duration calculation")
    parser.add_argument("text", nargs="?", help="Text to convert to audio")
    parser.add_argument("--output", "-o", help="Output audio file path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--wpm", type=float, default=150.0, help="Words per minute for token calculation (default: 150)")
    parser.add_argument("--style", choices=["slow", "natural", "fast", "very_fast"], default="natural", 
                       help="Speaking style: slow (30%% slower), natural (default), fast (20%% faster), very_fast (40%% faster)")
    parser.add_argument("--clean", choices=["smart", "aggressive", "conservative"], default="smart",
                       help="Audio cleaning method: smart (default), aggressive (more cleaning), conservative (less cleaning)")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale for generation (default: 5.0)")
    parser.add_argument("--temperature", type=float, default=1.5, help="Temperature for generation (default: 1.5)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for generation (default: 1.0)")
    parser.add_argument("--max-tokens", type=int, default=400, help="Maximum tokens to generate (default: 400, overridden by WPM calculation)")
    
    args = parser.parse_args()
    
    # Check if we have text or interactive mode
    if not args.text and not args.interactive:
        print("Please provide text to convert or use --interactive mode.")
        print("Example: python DIA_text_to_audio.py 'Hello, world!'")
        print("Example: python DIA_text_to_audio.py --interactive")
        print("Example: python DIA_text_to_audio.py 'Hello world!' --wpm 120 --style fast")
        return
    
    try:
        # Initialize the converter
        converter = DIATextToAudioConverter()
        
        # Update generation parameters if provided (these will be overridden by WPM calculation)
        if args.cfg_scale != 5.0 or args.temperature != 1.5 or args.top_p != 1.0:
            print(f"Using custom generation parameters:")
            print(f"  CFG Scale: {args.cfg_scale}")
            print(f"  Temperature: {args.temperature}")
            print(f"  Top-p: {args.top_p}")
            
            # Override the default parameters in the converter
            converter.cfg_scale = args.cfg_scale
            converter.temperature = args.temperature
            converter.top_p = args.top_p
        
        if args.interactive:
            # Run interactive mode
            interactive_mode(converter, args.wpm, args.style, args.clean)
        else:
            # Convert single text
            output_path = converter.convert_text_to_audio(args.text, args.output, args.wpm, args.style, args.clean)
            print(f"\n🎵 Audio saved to: {output_path}")
            
    except Exception as e:
        print(f"❌ Failed to initialize converter: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main() 