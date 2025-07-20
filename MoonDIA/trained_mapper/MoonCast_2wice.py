#!/usr/bin/env python3
"""
MoonCast_2wice.py
=================
A podcast generation system using MoonCast with audio prompts for voice cloning.

This script:
1. Takes a topic input from user
2. Calls GPT-4o-mini to generate a podcast script with multiple speakers
3. Uses audio prompts for voice cloning (prompt1.wav, prompt2.wav)
4. Converts the script to audio using MoonCast's pipeline
5. Saves the audio file

Usage:
    python MoonCast_2wice.py "Your podcast topic here"
    python MoonCast_2wice.py --interactive
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
from modules.audio_detokenizer.audio_detokenizer import get_audio_detokenizer, detokenize_noref, detokenize
from modules.audio_tokenizer.audio_tokenizer import get_audio_tokenizer
from transformers import AutoModelForCausalLM, GenerationConfig


class PodcastGenerator:
    """
    Complete podcast generation pipeline using MoonCast with audio prompts.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the podcast generation pipeline.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o-mini (if None, will try to get from env)
        """
        print("Initializing Podcast Generator...")
        
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
            
            # Load audio tokenizer for voice cloning
            print("  Loading MoonCast audio tokenizer...")
            self.audio_tokenizer = get_audio_tokenizer()
            
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
            
            # Load voice prompts
            print("  Loading voice prompts...")
            self.voice_prompts = self._load_voice_prompts()
            
            print("✅ Podcast Generator initialized successfully!")
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            print(f"  Restored working directory to: {original_cwd}")
    
    def _load_voice_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Load voice prompts from audio files."""
        import librosa
        
        voice_prompts = {}
        
        # Load prompt1.mp3 for speaker 0
        prompt1_path = os.path.join(current_dir, "prompt1.mp3")
        prompt1_text_path = os.path.join(current_dir, "prompt1.txt")
        if os.path.exists(prompt1_path):
            print(f"    Loading voice prompt 1: {prompt1_path}")
            waveform_24k = librosa.load(prompt1_path, sr=24000)[0]
            waveform_16k = librosa.load(prompt1_path, sr=16000)[0]
            
            waveform_24k = torch.tensor(waveform_24k).unsqueeze(0).to(torch.cuda.current_device())
            waveform_16k = torch.tensor(waveform_16k).unsqueeze(0).to(torch.cuda.current_device())
            
            semantic_tokens = self.audio_tokenizer.tokenize(waveform_16k)
            semantic_tokens = semantic_tokens.to(torch.cuda.current_device())
            prompt_ids = semantic_tokens + self.speech_token_offset
            
            # Load the actual transcript
            ref_text = "This is the first speaker's voice prompt."  # Default
            if os.path.exists(prompt1_text_path):
                with open(prompt1_text_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
            
            voice_prompts["0"] = {
                "ref_audio": prompt1_path,
                "ref_text": ref_text,
                "ref_bpe_ids": self.tokenizer.encode(ref_text),  # Add this for MoonCast compatibility
                "wav_24k": waveform_24k,
                "semantic_tokens": semantic_tokens,
                "prompt_ids": prompt_ids
            }
        else:
            print(f"    Warning: {prompt1_path} not found")
        
        # Load prompt2.mp3 for speaker 1
        prompt2_path = os.path.join(current_dir, "prompt2.mp3")
        prompt2_text_path = os.path.join(current_dir, "prompt2.txt")
        if os.path.exists(prompt2_path):
            print(f"    Loading voice prompt 2: {prompt2_path}")
            waveform_24k = librosa.load(prompt2_path, sr=24000)[0]
            waveform_16k = librosa.load(prompt2_path, sr=16000)[0]
            
            waveform_24k = torch.tensor(waveform_24k).unsqueeze(0).to(torch.cuda.current_device())
            waveform_16k = torch.tensor(waveform_16k).unsqueeze(0).to(torch.cuda.current_device())
            
            semantic_tokens = self.audio_tokenizer.tokenize(waveform_16k)
            semantic_tokens = semantic_tokens.to(torch.cuda.current_device())
            prompt_ids = semantic_tokens + self.speech_token_offset
            
            # Load the actual transcript
            ref_text = "This is the second speaker's voice prompt."  # Default
            if os.path.exists(prompt2_text_path):
                with open(prompt2_text_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
            
            voice_prompts["1"] = {
                "ref_audio": prompt2_path,
                "ref_text": ref_text,
                "ref_bpe_ids": self.tokenizer.encode(ref_text),  # Add this for MoonCast compatibility
                "wav_24k": waveform_24k,
                "semantic_tokens": semantic_tokens,
                "prompt_ids": prompt_ids
            }
        else:
            print(f"    Warning: {prompt2_path} not found")
        
        # Check if we have at least one voice prompt
        if not voice_prompts:
            print("    Warning: No voice prompts found. The script will continue without voice cloning.")
            print("    To enable voice cloning, ensure prompt1.mp3 and prompt2.mp3 files exist.")
        
        return voice_prompts
    
    def generate_podcast_script(self, topic: str, duration_minutes: int = 5) -> List[Dict[str, str]]:
        """
        Generate a podcast script using GPT-4o-mini.
        
        Args:
            topic: The podcast topic
            duration_minutes: Target duration in minutes (default: 5)
            
        Returns:
            List of dialogue turns with 'role' and 'text' keys
        """
        print(f"Generating podcast script for topic: '{topic}' (target duration: {duration_minutes} minutes)")
        
        # Calculate approximate number of turns based on duration
        # Assuming each turn takes about 15-20 seconds
        target_turns = max(3, min(20, duration_minutes * 3))  # 3-20 turns range
        
        prompt = f"""You are a podcast script writer. Create a SHORT, engaging podcast script about "{topic}" with two speakers having a natural conversation.

Requirements:
1. Create a dialogue between two speakers (Speaker 0 and Speaker 1)
2. Target duration: {duration_minutes} minutes (approximately {target_turns} dialogue turns)
3. Each speaker should have distinct personality and perspective
4. The conversation should be natural, engaging, and informative
5. Keep each speaker's turn SHORT (1-2 sentences maximum)
6. Alternate between speakers naturally
7. End with a brief conclusion
8. Keep the total script concise for testing purposes

Format the response as a JSON array of objects with 'role' and 'text' fields:
- 'role': "0" for Speaker 0, "1" for Speaker 1
- 'text': The spoken text for that turn

Example format:
[
    {{"role": "0", "text": "Welcome! Today we're discussing {topic}."}},
    {{"role": "1", "text": "Thanks! This is fascinating."}},
    ...
]

Make sure the roles alternate properly and keep the conversation brief and focused."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional podcast script writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2000
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
                return dialogue
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from GPT response: {e}")
                print(f"Raw response: {script_text}")
                raise
                
        except Exception as e:
            print(f"Error generating podcast script: {e}")
            raise
    
    def generate_semantic_tokens_with_prompts(self, dialogue: List[Dict[str, str]]) -> List[np.ndarray]:
        """
        Generate semantic tokens from dialogue using voice prompts.
        This follows the exact same approach as MoonCast's inference.py
        """
        print("Generating semantic tokens with voice prompts...")
        
        # Check if we have voice prompts
        if not self.voice_prompts:
            print("  No voice prompts available, falling back to simple text-to-semantic generation...")
            return self._generate_semantic_tokens_simple(dialogue)
        
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
        
        # Build initial prompt (following MoonCast's exact pattern)
        prompt = []
        
        # Add voice prompts for each speaker (same as MoonCast)
        prompt = prompt + user_role_0_ids + self.voice_prompts["0"]["ref_bpe_ids"] + [self.msg_end]
        prompt = prompt + user_role_1_ids + self.voice_prompts["1"]["ref_bpe_ids"] + [self.msg_end]
        
        # Add dialogue turns
        for turn in dialogue:
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            text_bpe_ids = self.tokenizer.encode(turn["text"])
            cur_start_ids = cur_user_ids + text_bpe_ids + [self.msg_end]
            prompt = prompt + cur_start_ids
        
        prompt = torch.LongTensor(prompt).unsqueeze(0).to(torch.cuda.current_device())
        
        # Add voice prompt responses (same as MoonCast)
        prompt = torch.cat([prompt, assistant_role_0_ids, media_start, self.voice_prompts["0"]["prompt_ids"], media_end], dim=-1)
        prompt = torch.cat([prompt, assistant_role_1_ids, media_start, self.voice_prompts["1"]["prompt_ids"], media_end], dim=-1)
        
        generation_config = self.generate_config
        
        # Generate semantic tokens for each turn (same as MoonCast)
        semantic_tokens_list = []
        
        for turn in dialogue:
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
        
        return semantic_tokens_list
    
    def _generate_semantic_tokens_simple(self, dialogue: List[Dict[str, str]]) -> List[np.ndarray]:
        """
        Fallback method to generate semantic tokens without voice prompts.
        
        Args:
            dialogue: List of dialogue turns with 'role' and 'text' keys
            
        Returns:
            List of semantic token sequences for each dialogue turn
        """
        print("  Using simple text-to-semantic generation...")
        
        semantic_tokens_list = []
        for turn in dialogue:
            text = turn["text"]
            role = turn["role"]
            print(f"    Generating tokens for {text[:50]}...")
            
            # Use the existing TextToSemantic method
            semantic_tokens = self.text_to_semantic.generate_semantic_tokens_simple(text, role=role)
            semantic_tokens_list.append(semantic_tokens)
        
        return semantic_tokens_list
    
    def semantic_tokens_to_audio(self, tokens_list: List[np.ndarray], dialogue: List[Dict[str, str]]) -> torch.Tensor:
        """
        Convert semantic tokens to audio using MoonCast's pipeline.
        
        Args:
            tokens_list: List of semantic token sequences
            dialogue: List of dialogue turns with 'role' and 'text' keys
            
        Returns:
            Combined audio tensor
        """
        print("Converting semantic tokens to audio...")
        
        audio_segments = []
        
        for i, tokens in enumerate(tokens_list):
            print(f"  Processing turn {i+1}/{len(tokens_list)}...")
            
            # Convert to tensor format expected by MoonCast
            if isinstance(tokens, np.ndarray):
                tokens = torch.tensor(tokens, dtype=torch.long, device=self.model.device)
            
            # Add batch dimension if needed
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            
            # Generate audio using MoonCast's detokenizer
            start_time = time.time()
            
            # Use voice cloning with reference audio (same as MoonCast)
            if self.voice_prompts and len(self.voice_prompts) > 0:
                # Get the correct role for this turn
                role_id = dialogue[i]["role"]  # Use the actual role from dialogue
                if role_id in self.voice_prompts:
                    voice_prompt = self.voice_prompts[role_id]
                    print(f"    Using voice cloning for role {role_id}")
                    gen_speech_fm = detokenize(self.audio_detokenizer, tokens, voice_prompt["wav_24k"], voice_prompt["semantic_tokens"])
                else:
                    print(f"    Warning: No voice prompt for role {role_id}, using no reference")
                    gen_speech_fm = detokenize_noref(self.audio_detokenizer, tokens)
            else:
                print("    No voice prompts available, using no reference")
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
        
        # Combine all audio segments
        if audio_segments:
            combined_audio = torch.cat(audio_segments, dim=-1)
            print(f"Combined {len(audio_segments)} audio segments")
            print(f"Final audio shape: {combined_audio.shape}")
            print(f"Final audio max: {combined_audio.abs().max().item():.4f}")
            return combined_audio
        else:
            raise ValueError("No audio segments generated")
    
    def generate_podcast(self, topic: str, output_path: Optional[str] = None, duration_minutes: int = 1, use_voice_cloning: bool = True) -> str:
        """
        Complete podcast generation pipeline.
        
        Args:
            topic: Podcast topic
            output_path: Optional output path for audio file
            duration_minutes: Target duration in minutes (default: 1 for testing)
            
        Returns:
            Path to the generated audio file
        """
        if not topic.strip():
            raise ValueError("Topic cannot be empty")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"podcast_{timestamp}.wav"
        
        print(f"\n{'='*60}")
        print(f"Generating Podcast")
        print(f"{'='*60}")
        print(f"Topic: {topic}")
        print(f"Target Duration: {duration_minutes} minutes")
        print(f"Output file: {output_path}")
        
        try:
            # Step 1: Generate podcast script using GPT-4o-mini
            dialogue = self.generate_podcast_script(topic, duration_minutes)
            
            # Step 2: Generate semantic tokens with voice prompts
            print(f"Dialogue turns: {len(dialogue)}")
            for i, turn in enumerate(dialogue):
                print(f"  Turn {i+1}: {turn['text'][:50]}...")
            
            semantic_tokens_list = self.generate_semantic_tokens_with_prompts(dialogue)
            
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
    parser = argparse.ArgumentParser(description="Generate podcasts using MoonCast with voice cloning")
    parser.add_argument("topic", nargs="?", help="Podcast topic to generate")
    parser.add_argument("--output", "-o", help="Output audio file path")
    parser.add_argument("--duration", "-d", type=int, default=1, help="Target duration in minutes (default: 1)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Check if we have topic or interactive mode
    if not args.topic and not args.interactive:
        print("Please provide a podcast topic or use --interactive mode.")
        print("Example: python MoonCast_2wice.py 'The future of artificial intelligence'")
        print("Example: python MoonCast_2wice.py 'AI topic' --duration 2")
        print("Example: python MoonCast_2wice.py --interactive")
        return
    
    try:
        # Initialize the generator
        generator = PodcastGenerator(openai_api_key=args.openai_key)
        
        if args.interactive:
            # Run interactive mode
            interactive_mode(generator)
        else:
            # Generate single podcast
            output_path = generator.generate_podcast(args.topic, args.output, args.duration)
            print(f"\n🎵 Podcast saved to: {output_path}")
            
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
