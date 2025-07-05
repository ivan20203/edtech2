#!/usr/bin/env python3
"""
MoonCast Pipeline: Text → Mel-Spectrograms
Copies MoonCast's exact working pipeline to generate mel-spectrograms from text.
"""

import sys
import os
import torch
import numpy as np
import librosa
import torchaudio
import io
import base64
from tqdm import tqdm

# Add MoonCast to path
sys.path.append("../../MoonCast")

# Import MoonCast's exact components
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens
from modules.audio_detokenizer.audio_detokenizer import get_audio_detokenizer, detokenize_noref
from transformers import AutoModelForCausalLM, GenerationConfig


class MoonCastPipeline:
    """
    MoonCast's exact pipeline: Text → Semantic Tokens → Audio → Mel-Spectrograms
    """
    
    def __init__(self):
        """Initialize MoonCast's complete pipeline."""
        print("Initializing MoonCast Pipeline...")
        
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
        
        # Generation config (same as MoonCast)
        self.generate_config = GenerationConfig(
            max_new_tokens=200 * 50,
            do_sample=True,
            top_k=30,
            top_p=0.8,
            temperature=0.8,
            eos_token_id=self.media_end,
        )
        
        # Load detokenizer (same as MoonCast)
        print("Loading MoonCast detokenizer...")
        self.audio_detokenizer = get_audio_detokenizer()
        
        # Set device
        self.device = torch.cuda.current_device()
        
        print("✅ MoonCast Pipeline initialized")
    
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
        """
        Convert text to semantic tokens using MoonCast's exact pipeline.
        This is the missing method that was causing the error.
        """
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
    def text_to_mel_direct(self, dialogue):
        """
        Convert text to mel-spectrograms using MoonCast's exact pipeline.
        This uses their exact approach: Text → Semantic Tokens → Audio → Mel-Spectrograms
        """
        print("Converting text to mel-spectrograms using MoonCast's exact pipeline...")
        
        # Generate semantic tokens using MoonCast's exact method
        semantic_tokens_list = self.text_to_semantic_tokens(dialogue)
        
        # Convert semantic tokens to audio, then extract mel-spectrograms
        mel_list = []
        
        for i, torch_token in enumerate(semantic_tokens_list):
            print(f"Processing turn {i+1}/{len(semantic_tokens_list)}")
            
            # Use MoonCast's exact detokenize_noref function to generate audio
            audio = detokenize_noref(self.audio_detokenizer, torch_token)
            
            # Extract mel-spectrogram from audio using the vocoder
            mel_spec = self.audio_detokenizer.vocoder.extract_mel_from_wav(wav_data=audio.squeeze(0))
            mel_list.append(mel_spec)
        
        return mel_list
    
    def audio_to_mel(self, audio, sample_rate=24000, n_mels=128, n_fft=1024, hop_length=256):
        """
        Convert audio tensor to mel-spectrogram.
        
        Args:
            audio: Audio tensor (1, samples)
            sample_rate: Audio sample rate
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            
        Returns:
            mel_spectrogram: Mel-spectrogram as numpy array (n_mels, time_frames)
        """
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Extract mel-spectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann'
        )
        
        # Convert to log scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return mel_spectrogram
    
    def text_to_mel(self, dialogue, save_mel=True):
        """
        Complete pipeline: Text → Mel-Spectrograms (Direct)
        
        Args:
            dialogue: List of dialogue turns, each with 'role' and 'text' keys
            save_mel: Whether to save mel-spectrograms
            
        Returns:
            mel_list: List of mel-spectrogram tensors
        """
        print("Running MoonCast pipeline: Text → Mel-Spectrograms (Direct)")
        mel_list = self.text_to_mel_direct(dialogue)
        
        if save_mel:
            # Save mel-spectrograms
            for i, mel in enumerate(mel_list):
                mel_path = f"mel_spectrogram_turn_{i}.npy"
                np.save(mel_path, mel.cpu().numpy())
                print(f"Saved mel-spectrogram to {mel_path}")
        
        return mel_list
    
    def single_text_to_mel(self, text, role="0", save_mel=True):
        """
        Convert single text to mel-spectrogram.
        
        Args:
            text: Input text string
            role: Speaker role ("0" or "1")
            save_mel: Whether to save mel-spectrogram
            
        Returns:
            mel: Mel-spectrogram tensor
        """
        # Create dialogue format
        dialogue = [{"role": role, "text": text}]
        
        # Run pipeline
        mel_list = self.text_to_mel(dialogue, save_mel)
        
        return mel_list[0] if mel_list else None


def main():
    """Test the MoonCast pipeline."""
    print("Testing MoonCast Pipeline (Direct Mel-Spectrogram Generation)")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = MoonCastPipeline()
    
    # Test data
    text = "Hello, this is a test of the MoonCast pipeline for generating mel-spectrograms from text."
    
    print("\n1. Testing single text:")
    mel = pipeline.single_text_to_mel(text, role="0", save_mel=True)
    print(f"Generated mel-spectrogram shape: {mel.shape}")
    
    print("\n2. Testing dialogue:")
    dialogue = [
        {"role": "0", "text": "Hello, how are you today?"},
        {"role": "1", "text": "I'm doing great, thank you for asking!"},
        {"role": "0", "text": "That's wonderful to hear."}
    ]
    
    mel_list = pipeline.text_to_mel(dialogue, save_mel=True)
    print(f"Generated {len(mel_list)} mel-spectrograms")
    for i, mel in enumerate(mel_list):
        print(f"  Turn {i+1}: {mel.shape}")
    
    print("\n✅ MoonCast pipeline test completed successfully!")


if __name__ == "__main__":
    main() 