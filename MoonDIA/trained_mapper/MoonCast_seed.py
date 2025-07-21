#!/usr/bin/env python3
"""
MoonCast_seed.py
=================
A podcast generation system using MoonCast without voice cloning and with seed setting for reproducible generation.

This script:
1. Takes a topic input from user
2. Calls GPT-4o-mini to generate a podcast script with multiple speakers
3. Uses MoonCast's no-reference audio generation (no voice cloning)
4. Sets random seeds for reproducible generation
5. Converts the script to audio using MoonCast's pipeline
6. Saves the audio file

Usage:
    python MoonCast_seed.py "Your podcast topic here"
    python MoonCast_seed.py --interactive
    python MoonCast_seed.py "AI topic" --seed 42
"""

import sys
import os
import torch
import numpy as np
import torchaudio
import argparse
import time
import json
import openai
import random
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# Try to load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

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

# Import MoonCast's text cleaning method
    
def _clean_text(text):
    text = text.replace("“", "")
    text = text.replace("”", "")
    text = text.replace("...", " ")
    text = text.replace("…", " ")
    text = text.replace("*", "")
    text = text.replace(":", ",")
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.strip()
    return text


def set_seed(seed: int):
    """Set random seeds for reproducible generation.
    
    Args:
        seed: Random seed value
    """
    print(f"Setting random seed to: {seed}")
    print(f"  - Python random.seed({seed})")
    print(f"  - NumPy np.random.seed({seed})")
    print(f"  - PyTorch torch.manual_seed({seed})")
    print(f"  - CUDA torch.cuda.manual_seed({seed})")
    print(f"  - CUDA torch.cuda.manual_seed_all({seed})")
    print(f"  - Setting torch.backends.cudnn.deterministic = True")
    print(f"  - Setting torch.backends.cudnn.benchmark = False")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PodcastGenerator:
    """
    Complete podcast generation pipeline using MoonCast without voice cloning.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, seed: Optional[int] = None):
        """Initialize the podcast generation pipeline.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o-mini (if None, will try to get from env)
            seed: Random seed for reproducible generation (if None, will use random seed)
        """
        print("Initializing Podcast Generator...")
        
        # Set seed if provided
        if seed is not None:
            print(f"Using provided seed: {seed}")
            set_seed(seed)
        else:
            # Use a random seed for this session
            random_seed = random.randint(1, 1000000)
            print(f"No seed provided. Generated random seed: {random_seed}")
            set_seed(random_seed)
        
        # Initialize OpenAI client
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it to the constructor.")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
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
            
            print("✅ Podcast Generator initialized successfully!")
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            print(f"  Restored working directory to: {original_cwd}")
    
    def read_input_text(self, input_file: str = "input_text.txt") -> str:
        """
        Read topic from input_text.txt file.
        
        Args:
            input_file: Path to the input file (default: input_text.txt)
            
        Returns:
            The topic text from the file
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                topic = f.read().strip()
            
            if not topic:
                raise ValueError("Input file is empty")
            
            print(f"Read topic from {input_file}: {topic[:100]}{'...' if len(topic) > 100 else ''}")
            return topic
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file '{input_file}' not found")
        except Exception as e:
            raise Exception(f"Error reading input file: {e}")

    def generate_podcast_script(self, topic: str, duration_minutes: int = 5, save_to_file: bool = True) -> List[Dict[str, str]]:
        """
        Generate a podcast script using GPT-4o-mini.
        
        Args:
            topic: The podcast topic
            duration_minutes: Target duration in minutes (default: 5)
            save_to_file: Whether to save the generated script to script.txt
            
        Returns:
            List of dialogue turns with 'role' and 'text' keys
        """
        print(f"Generating podcast script for topic: '{topic}' (target duration: {duration_minutes} minutes)")
        
        # Calculate target tokens based on duration
        # Assuming each turn is ~20 tokens and we want ~20 turns per minute
        target_tokens = max(1400, int(duration_minutes * 1000))  # 1000 tokens per minute
        target_turns = target_tokens // 20  # Each turn is approximately 20 tokens
        
        prompt = f"""You are a podcast script writer. Create an engaging podcast script about "{topic}" with two speakers having a natural conversation.

CRITICAL REQUIREMENTS:
1. Create a dialogue between two speakers (Speaker 0 and Speaker 1)
2. Target: {target_tokens} tokens total ({target_turns} dialogue turns)
3. Each turn should be approximately 20 tokens (1-2 sentences)
4. Each speaker should have distinct personality and perspective
5. Keep each speaker's turn SHORT (1-2 sentences maximum)
6. Alternate between speakers naturally (Speaker 0, then Speaker 1, then Speaker 0, etc.)
7. Use lots of exclamation marks to show excitement and enthusiasm!
8. Only use periods for punctuation. No commas or apostrophes.
9. Expand all abbreviations. Even chemical abbreviations.

TOKEN CALCULATION:
- Target: {target_tokens} tokens
- Each turn: ~20 tokens
- Expected turns: {target_turns}
- Duration: {duration_minutes} minutes

Format the response as a JSON array of objects with 'role' and 'text' fields:
- 'role': "0" for Speaker 0, "1" for Speaker 1
- 'text': The spoken text for that turn

Example format:
[
    {{"role": "0", "text": "Welcome! Today we're discussing {topic}!"}},
    {{"role": "1", "text": "Thanks! This is going to be fascinating!"}},
    ...
]

Remember: Aim for {target_tokens} tokens total across {target_turns} dialogue turns!"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional podcast script writer. You must always generate the exact number of dialogue turns requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=15000
            )
            
            script_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON from the response
            try:
                # Find JSON array in the response
                start_idx = script_text.find('[')
                end_idx = script_text.rfind(']') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = script_text[start_idx:end_idx]
                    dialogue = json.loads(json_str)
                    
                    # Clean text using MoonCast's method
                    for turn in dialogue:
                        if 'text' in turn:
                            turn['text'] = _clean_text(turn['text'])
                else:
                    raise ValueError("No JSON array found in response")
                
                # Validate dialogue format
                if not isinstance(dialogue, list):
                    raise ValueError("Response is not a list")
                
                for turn in dialogue:
                    if not isinstance(turn, dict) or 'role' not in turn or 'text' not in turn:
                        raise ValueError("Invalid dialogue turn format")
                    if turn['role'] not in ['0', '1']:
                        raise ValueError("Invalid role, must be '0' or '1'")
                
                print(f"Generated podcast script with {len(dialogue)} dialogue turns")
                
                # Save script to file if requested
                if save_to_file:
                    script_file = "script.txt"
                    try:
                        with open(script_file, 'w', encoding='utf-8') as f:
                            f.write(f"Podcast Script: {topic}\n")
                            f.write(f"Duration: {duration_minutes} minutes\n")
                            f.write(f"Total turns: {len(dialogue)}\n")
                            f.write(f"Seed: {seed}\n")
                            f.write("=" * 50 + "\n\n")
                            
                            for i, turn in enumerate(dialogue):
                                f.write(f"Turn {i+1} (Speaker {turn['role']}): {turn['text']}\n\n")
                        
                        print(f"Script saved to {script_file}")
                    except Exception as e:
                        print(f"Warning: Could not save script to {script_file}: {e}")
                
                return dialogue
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from GPT response: {e}")
                print(f"Raw response: {script_text}")
                raise
                
        except Exception as e:
            print(f"Error generating podcast script: {e}")
            raise
    
    def generate_semantic_tokens_no_reference(self, dialogue: List[Dict[str, str]]) -> List[np.ndarray]:
        """
        Generate semantic tokens from dialogue without voice cloning.
        This follows the exact same approach as MoonCast's infer_without_prompt method.
        """
        print("Generating semantic tokens without voice cloning...")
        total_start = time.time()
        
        # Build role IDs (same as MoonCast)
        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids + [self.name_end]
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]
        
        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]
        
        # Convert to tensors
        assistant_role_0_ids = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(torch.cuda.current_device())
        assistant_role_1_ids = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(torch.cuda.current_device())
        media_start = torch.LongTensor(media_start).unsqueeze(0).to(torch.cuda.current_device())
        media_end = torch.LongTensor(media_end).unsqueeze(0).to(torch.cuda.current_device())
        
        # Build initial prompt (following MoonCast's exact pattern for no-reference)
        prompt = []
        
        # Add dialogue turns (no voice prompts)
        for turn in dialogue:
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            text_bpe_ids = self.tokenizer.encode(turn["text"])
            cur_start_ids = cur_user_ids + text_bpe_ids + [self.msg_end]
            prompt = prompt + cur_start_ids
        
        prompt = torch.LongTensor(prompt).unsqueeze(0).to(torch.cuda.current_device())
        
        generation_config = self.generate_config
        
        # Generate semantic tokens for each turn (same as MoonCast)
        semantic_tokens_list = []
        
        for i, turn in enumerate(dialogue):
            start = time.time()
            print(f"  Turn {i+1}/{len(dialogue)}...")
            
            # Clear GPU memory before each turn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"    Cleared GPU cache before turn")
            
            role_id = turn["role"]
            cur_assistant_ids = assistant_role_0_ids if role_id == "0" else assistant_role_1_ids
            
            # Add assistant role and media start to prompt
            prompt = torch.cat([prompt, cur_assistant_ids, media_start], dim=-1)
            len_prompt = prompt.shape[1]
            generation_config.min_length = len_prompt + 2
            
            # Generate tokens
            outputs = self.model.generate(prompt, generation_config=generation_config)
            
            # Remove media_end if present
            if outputs[0, -1] == self.media_end:
                outputs = outputs[:, :-1]
            
            # Extract generated tokens
            output_token = outputs[:, len_prompt:]
            prompt = torch.cat([outputs, media_end], dim=-1)
            
            # Convert to semantic tokens (remove speech token offset)
            semantic_tokens = output_token - self.speech_token_offset
            
            # Add to results
            semantic_tokens_list.append(semantic_tokens.cpu().numpy())
            print(f"    Done in {time.time() - start:.1f}s")
            
            # Clear GPU memory after each turn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"    Cleared GPU cache after turn")
        
        print(f"Total time: {time.time() - total_start:.1f}s")
        return semantic_tokens_list
    
    def semantic_tokens_to_audio(self, tokens_list: List[np.ndarray], dialogue: List[Dict[str, str]]) -> torch.Tensor:
        """
        Convert semantic tokens to audio using MoonCast's no-reference pipeline.
        
        Args:
            tokens_list: List of semantic token sequences
            dialogue: List of dialogue turns with 'role' and 'text' keys
            
        Returns:
            Combined audio tensor
        """
        print("Converting semantic tokens to audio (no voice cloning)...")
        
        audio_segments = []
        
        for i, tokens in enumerate(tokens_list):
            print(f"  Processing turn {i+1}/{len(tokens_list)}...")
            
            # Clear GPU memory before processing each turn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"    Cleared GPU cache")
            
            # Convert to tensor format expected by MoonCast
            if isinstance(tokens, np.ndarray):
                tokens = torch.tensor(tokens, dtype=torch.long, device=self.model.device)
            
            # Add batch dimension if needed
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            
            # Generate audio using MoonCast's no-reference detokenizer
            start_time = time.time()
            
            # Use no-reference generation (no voice cloning)
            gen_speech_fm = detokenize_noref(self.audio_detokenizer, tokens)
            gen_speech_fm = gen_speech_fm.cpu()
            
            # Normalize audio
            if gen_speech_fm.abs().max() > 0:
                gen_speech_fm = gen_speech_fm / gen_speech_fm.abs().max()
            
            end_time = time.time()
            
            print(f"    Generated audio in {end_time - start_time:.2f} seconds")
            print(f"    Audio shape: {gen_speech_fm.shape}")
            print(f"    Audio max: {gen_speech_fm.abs().max().item():.4f}")
            
            audio_segments.append(gen_speech_fm)
            
            # Clear GPU memory after processing each turn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"    Cleared GPU cache after processing")
        
        # Combine all audio segments
        if audio_segments:
            combined_audio = torch.cat(audio_segments, dim=-1)
            print(f"Combined {len(audio_segments)} audio segments")
            print(f"Final audio shape: {combined_audio.shape}")
            print(f"Final audio max: {combined_audio.abs().max().item():.4f}")
            return combined_audio
        else:
            raise ValueError("No audio segments generated")
    
    def generate_podcast(self, topic: str, output_path: Optional[str] = None, duration_minutes: int = 1, 
                        use_input_file: bool = False, save_script: bool = True) -> str:
        """
        Complete podcast generation pipeline without voice cloning.
        
        Args:
            topic: Podcast topic (ignored if use_input_file is True)
            output_path: Optional output path for audio file
            duration_minutes: Target duration in minutes (default: 1 for testing)
            use_input_file: Whether to read topic from input_text.txt
            save_script: Whether to save generated script to script.txt
            
        Returns:
            Path to the generated audio file
        """
        # Read topic from input file if requested
        if use_input_file:
            topic = self.read_input_text()
        elif not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"podcast_{timestamp}.wav"
        
        print(f"\n{'='*60}")
        print(f"Generating Podcast (No Voice Cloning)")
        print(f"{'='*60}")
        print(f"Topic: {topic}")
        print(f"Target Duration: {duration_minutes} minutes")
        print(f"Output file: {output_path}")
        
        try:
            # Step 1: Generate podcast script using GPT-4o-mini
            dialogue = self.generate_podcast_script(topic, duration_minutes, save_script)
            
            # Step 2: Generate semantic tokens without voice prompts
            print(f"Dialogue turns: {len(dialogue)}")
            for i, turn in enumerate(dialogue):
                print(f"  Turn {i+1}: {turn['text'][:50]}...")
            
            semantic_tokens_list = self.generate_semantic_tokens_no_reference(dialogue)
            
            # Step 3: Convert semantic tokens to audio
            audio_tensor = self.semantic_tokens_to_audio(semantic_tokens_list, dialogue)
            
            # Step 4: Save audio
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
            
            print(f"✅ Podcast generated successfully!")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  RMS: {rms:.4f}")
            print(f"  Peak: {peak:.4f}")
            print(f"  Sample rate: 24kHz")
            print(f"  Dialogue turns: {len(dialogue)}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during podcast generation: {e}")
            import traceback
            traceback.print_exc()
            raise


def interactive_mode(generator: PodcastGenerator):
    """Run the generator in interactive mode."""
    print("\n🎙️ Interactive Podcast Generation Mode")
    print("Enter a topic to generate a podcast (or 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            topic = input("\nEnter podcast topic: ").strip()
            
            if topic.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not topic:
                print("Please enter a topic.")
                continue
            
            # Ask for duration
            try:
                duration_input = input("Enter duration in minutes (default: 1): ").strip()
                duration_minutes = int(duration_input) if duration_input else 1
            except ValueError:
                duration_minutes = 1
                print("Invalid duration, using 1 minute.")
            
            # Ask for seed
            try:
                seed_input = input("Enter random seed (default: random): ").strip()
                seed = int(seed_input) if seed_input else None
            except ValueError:
                seed = None
                print("Invalid seed, using random seed.")
            
            # Generate output filename
            timestamp = int(time.time())
            output_path = f"interactive_podcast_{timestamp}.wav"
            
            # Generate podcast
            result_path = generator.generate_podcast(topic, output_path, duration_minutes)
            print(f"\n🎵 Podcast saved to: {result_path}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate podcasts using MoonCast without voice cloning")
    parser.add_argument("topic", nargs="?", help="Podcast topic to generate")
    parser.add_argument("--output", "-o", help="Output audio file path")
    parser.add_argument("--duration", "-d", type=int, default=1, help="Target duration in minutes (default: 1)")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducible generation")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--input-file", action="store_true", help="Read topic from input_text.txt instead of command line")
    parser.add_argument("--no-save-script", action="store_true", help="Don't save generated script to script.txt")
    
    args = parser.parse_args()
    
    # Check if we have topic, input file, or interactive mode
    if not args.topic and not args.input_file and not args.interactive:
        print("Please provide a podcast topic, use --input-file, or use --interactive mode.")
        print("Example: python MoonCast_seed.py 'The future of artificial intelligence'")
        print("Example: python MoonCast_seed.py 'AI topic' --duration 2 --seed 42")
        print("Example: python MoonCast_seed.py --input-file")
        print("Example: python MoonCast_seed.py --interactive")
        return
    
    try:
        # Initialize the generator
        generator = PodcastGenerator(openai_api_key=args.openai_key, seed=args.seed)
        
        if args.interactive:
            # Run interactive mode
            interactive_mode(generator)
        else:
            # Generate single podcast
            output_path = generator.generate_podcast(
                topic=args.topic or "", 
                output_path=args.output, 
                duration_minutes=args.duration,
                use_input_file=args.input_file,
                save_script=not args.no_save_script
            )
            print(f"\n🎵 Podcast saved to: {output_path}")
            
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
