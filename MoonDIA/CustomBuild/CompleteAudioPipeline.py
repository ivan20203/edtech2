#!/usr/bin/env python3
"""
Complete Audio Pipeline: Text → MoonCast Audio → DAC Tokens → DIA Audio
Uses audio domain throughout to avoid mel-spectrogram conversion issues.
"""

import sys
import os
import torch
import numpy as np
import soundfile as sf
import time

# Add MoonCast to path
sys.path.append("../../MoonCast")

# Import MoonCast's exact components
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_detokenizer.audio_detokenizer import get_audio_detokenizer, detokenize_noref
from transformers import AutoModelForCausalLM, GenerationConfig

# Import our audio bridge
from AudioToDACBridge import AudioToDACBridge

# Import DIA decoder
from SimpleDIAPipeline import SimpleDIADACDecoder


class CompleteAudioPipeline:
    """
    Complete pipeline: Text → MoonCast Audio → DAC Tokens → DIA Audio
    """
    
    def __init__(self):
        """Initialize the complete audio pipeline."""
        print("Initializing Complete Audio Pipeline...")
        
        # Initialize MoonCast components
        print("Loading MoonCast components...")
        self._init_mooncast()
        
        # Initialize Audio to DAC Bridge
        print("Loading Audio to DAC Bridge...")
        self.audio_bridge = AudioToDACBridge(
            token_rate=100,
            semantic_vocab_size=16384
        )
        
        # Initialize DIA DAC decoder
        print("Loading DIA DAC decoder...")
        self.dia_decoder = SimpleDIADACDecoder()
        
        print("✅ Complete Audio Pipeline initialized")
    
    def _init_mooncast(self):
        """Initialize MoonCast components."""
        # Initialize tokenizer (same as MoonCast)
        self.tokenizer, self.extra_tokens = get_tokenizer_and_extra_tokens()
        self.speech_token_offset = 163840
        
        # Special token IDs (same as MoonCast)
        self.assistant_ids = self.tokenizer.encode("assistant")
        self.user_ids = self.tokenizer.encode("user")
        self.audio_ids = self.tokenizer.encode("audio")
        self.spk_0_ids = self.tokenizer.encode("0")
        self.spk_1_ids = self.tokenizer.encode("1")
        
        # Extra tokens (same as MoonCast)
        self.msg_end = self.extra_tokens.msg_end
        self.user_msg_start = self.extra_tokens.user_msg_start
        self.assistant_msg_start = self.extra_tokens.assistant_msg_start
        self.name_end = self.extra_tokens.name_end
        self.media_begin = self.extra_tokens.media_begin
        self.media_content = self.extra_tokens.media_content
        self.media_end = self.extra_tokens.media_end
        
        # Load text2semantic model (same as MoonCast)
        model_path = "resources/text2semantic"
        print(f"Loading text2semantic model from: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda:0", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            force_download=False
        ).to(torch.cuda.current_device())
        
        # Generation config (improved for longer generation)
        self.generate_config = GenerationConfig(
            max_new_tokens=200 * 200,  # Increased for longer generation
            do_sample=True,
            top_k=30,
            top_p=0.8,
            temperature=0.6,  # Lowered for more consistent generation
            eos_token_id=self.media_end,
            pad_token_id=self.media_end,  # Add pad token
            repetition_penalty=1.1,  # Add repetition penalty
        )
        
        # Load detokenizer (same as MoonCast)
        print("Loading MoonCast detokenizer...")
        self.audio_detokenizer = get_audio_detokenizer()
        
        # Set device
        self.device = torch.cuda.current_device()
    
    def _clean_text(self, text):
        """Clean input text (same as MoonCast)."""
        text = text.replace(""", "")
        text = text.replace(""", "")
        text = text.replace("...", " ")
        text = text.replace("…", " ")
        text = text.replace("*", "")
        text = text.replace(":", ",")
        text = text.replace("'", "'")
        text = text.replace("'", "'")
        text = text.strip()
        return text
    
    def _process_text(self, dialogue):
        """Process dialogue text (same as MoonCast)."""
        processed_dialogue = []
        for turn in dialogue:
            processed_turn = turn.copy()
            processed_turn["bpe_ids"] = self.tokenizer.encode(self._clean_text(turn["text"]))
            processed_dialogue.append(processed_turn)
        return processed_dialogue
    
    @torch.inference_mode()
    def text_to_semantic_tokens(self, dialogue):
        """Convert text to semantic tokens using MoonCast's exact pipeline."""
        print("Converting text to semantic tokens...")
        
        # Process text (same as MoonCast)
        processed_dialogue = self._process_text(dialogue)
        
        # Build role IDs (same as MoonCast)
        user_role_0_ids = [self.user_msg_start] + self.user_ids + self.spk_0_ids + [self.name_end]
        user_role_1_ids = [self.user_msg_start] + self.user_ids + self.spk_1_ids + [self.name_end]
        assistant_role_0_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_0_ids + [self.name_end]
        assistant_role_1_ids = [self.assistant_msg_start] + self.assistant_ids + self.spk_1_ids + [self.name_end]
        
        media_start = [self.media_begin] + self.audio_ids + [self.media_content]
        media_end = [self.media_end] + [self.msg_end]
        
        # Convert to tensors (same as MoonCast)
        assistant_role_0_ids = torch.LongTensor(assistant_role_0_ids).unsqueeze(0).to(torch.cuda.current_device())
        assistant_role_1_ids = torch.LongTensor(assistant_role_1_ids).unsqueeze(0).to(torch.cuda.current_device())
        media_start = torch.LongTensor(media_start).unsqueeze(0).to(torch.cuda.current_device())
        media_end = torch.LongTensor(media_end).unsqueeze(0).to(torch.cuda.current_device())
        
        # Build initial prompt (same as MoonCast)
        prompt = []
        for turn in processed_dialogue:
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            cur_start_ids = cur_user_ids + turn["bpe_ids"] + [self.msg_end]
            prompt = prompt + cur_start_ids
        
        prompt = torch.LongTensor(prompt).unsqueeze(0).to(torch.cuda.current_device())
        generation_config = self.generate_config
        
        # Generate semantic tokens for each turn (same as MoonCast)
        semantic_tokens_list = []
        
        for turn in processed_dialogue:
            role_id = turn["role"]
            cur_assistant_ids = assistant_role_0_ids if role_id == "0" else assistant_role_1_ids
            
            # Add assistant role and media start to prompt
            prompt = torch.cat([prompt, cur_assistant_ids, media_start], dim=-1)
            len_prompt = prompt.shape[1]
            generation_config.min_length = len_prompt + 2
            
            # Generate tokens (same as MoonCast)
            outputs = self.model.generate(prompt, generation_config=generation_config)
            
            # Remove media_end if present
            if outputs[0, -1] == self.media_end:
                outputs = outputs[:, :-1]
            
            # Extract generated tokens
            output_token = outputs[:, len_prompt:]
            prompt = torch.cat([outputs, media_end], dim=-1)
            
            # Convert to semantic tokens
            torch_token = output_token - self.speech_token_offset
            semantic_tokens_list.append(torch_token)
        
        return semantic_tokens_list
    
    @torch.inference_mode()
    def text_to_audio(self, dialogue):
        """Convert text to audio using MoonCast's exact pipeline."""
        print("Converting text to audio using MoonCast...")
        
        # Generate semantic tokens using MoonCast's exact method
        semantic_tokens_list = self.text_to_semantic_tokens(dialogue)
        
        # Convert semantic tokens to audio
        audio_list = []
        
        for i, torch_token in enumerate(semantic_tokens_list):
            print(f"Processing turn {i+1}/{len(semantic_tokens_list)}")
            
            # Use MoonCast's exact detokenize_noref function to generate audio
            audio = detokenize_noref(self.audio_detokenizer, torch_token)
            audio_list.append(audio.squeeze(0))  # Remove batch dimension
        
        return audio_list
    
    def audio_to_dac_tokens(self, audio_list):
        """Convert MoonCast audio to DAC tokens using the audio bridge."""
        print("Converting MoonCast audio to DAC tokens...")
        
        dac_tokens_list = []
        
        for i, audio in enumerate(audio_list):
            print(f"Processing audio turn {i+1}/{len(audio_list)}")
            
            # Convert to DAC tokens using the audio bridge
            dac_tokens = self.audio_bridge.mooncast_audio_to_dac_tokens(audio)
            dac_tokens_list.append(dac_tokens)
            
            # Save intermediate DAC tokens
            tokens_path = f"turn_{i}_audio_dac_tokens.npy"
            np.save(tokens_path, dac_tokens.cpu().numpy())
            print(f"  Saved DAC tokens to {tokens_path}")
        
        return dac_tokens_list
    
    def dac_tokens_to_audio(self, dac_tokens_list):
        """Convert DAC tokens to audio using DIA decoder."""
        print("Converting DAC tokens to audio using DIA...")
        
        audio_list = []
        
        for i, dac_tokens in enumerate(dac_tokens_list):
            print(f"Processing DAC tokens turn {i+1}/{len(dac_tokens_list)}")
            
            # Use DIA decoder to convert tokens to audio
            audio = self.dia_decoder.decode(dac_tokens)
            audio_list.append(audio)
            
            # Save intermediate audio
            audio_path = f"turn_{i}_dia_generated_audio.wav"
            sf.write(audio_path, audio.cpu().numpy(), 44100)
            print(f"  Saved audio to {audio_path}")
        
        return audio_list
    
    def text_to_dia_audio(self, dialogue, save_intermediate=True):
        """
        Complete pipeline: Text → MoonCast Audio → DAC Tokens → DIA Audio
        
        Args:
            dialogue: List of dialogue turns, each with 'role' and 'text' keys
            save_intermediate: Whether to save intermediate files
            
        Returns:
            final_audio_list: List of final audio tensors
        """
        print("Running complete audio pipeline: Text → DIA Audio")
        print("=" * 60)
        
        # Step 1: Text → MoonCast Audio
        print("\nStep 1: Text → MoonCast Audio")
        mooncast_audio_list = self.text_to_audio(dialogue)
        print(f"Generated {len(mooncast_audio_list)} audio clips")
        
        if save_intermediate:
            for i, audio in enumerate(mooncast_audio_list):
                audio_path = f"turn_{i}_mooncast_audio.wav"
                sf.write(audio_path, audio.cpu().numpy(), 24000)
                print(f"  Saved MoonCast audio to {audio_path}")
        
        # Step 2: MoonCast Audio → DAC Tokens
        print("\nStep 2: MoonCast Audio → DAC Tokens")
        dac_tokens_list = self.audio_to_dac_tokens(mooncast_audio_list)
        print(f"Generated {len(dac_tokens_list)} DAC token sets")
        
        # Step 3: DAC Tokens → DIA Audio
        print("\nStep 3: DAC Tokens → DIA Audio")
        final_audio_list = self.dac_tokens_to_audio(dac_tokens_list)
        print(f"Generated {len(final_audio_list)} final audio clips")
        
        # Save concatenated final audio
        if len(final_audio_list) > 1:
            concatenated_audio = torch.cat(final_audio_list, dim=0)
            sf.write("concatenated_dia_audio.wav", concatenated_audio.cpu().numpy(), 44100)
            print("  Saved concatenated final audio to concatenated_dia_audio.wav")
        
        print("\n✅ Complete audio pipeline finished successfully!")
        return final_audio_list
    
    def single_text_to_dia_audio(self, text, role="0", save_intermediate=True):
        """
        Convert single text to DIA audio.
        
        Args:
            text: Input text string
            role: Speaker role ("0" or "1")
            save_intermediate: Whether to save intermediate files
            
        Returns:
            final_audio: Final audio tensor
        """
        # Create dialogue format
        dialogue = [{"role": role, "text": text}]
        
        # Run complete pipeline
        final_audio_list = self.text_to_dia_audio(dialogue, save_intermediate)
        
        return final_audio_list[0] if final_audio_list else None


def main():
    """Test the complete audio pipeline."""
    print("Testing Complete Audio Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = CompleteAudioPipeline()
    
    # Test data
    text = "Hello, this is a test of the complete audio pipeline. We are now generating longer audio using improved parameters."
    
    print("\n1. Testing single text:")
    final_audio = pipeline.single_text_to_dia_audio(text, role="0", save_intermediate=True)
    print(f"Generated final audio shape: {final_audio.shape}")
    
    print("\n2. Testing dialogue:")
    dialogue = [
        {"role": "0", "text": "Hello, how are you today? I hope you're having a wonderful day."},
        {"role": "1", "text": "I'm doing great, thank you for asking! The weather is beautiful and I'm feeling very positive."},
        {"role": "0", "text": "That's wonderful to hear. I'm glad everything is going well for you."}
    ]
    
    final_audio_list = pipeline.text_to_dia_audio(dialogue, save_intermediate=True)
    print(f"Generated {len(final_audio_list)} final audio clips")
    for i, audio in enumerate(final_audio_list):
        print(f"  Turn {i+1}: {audio.shape}")
    
    print("\n✅ Complete audio pipeline test completed successfully!")


if __name__ == "__main__":
    main() 