import sys
import os
import torch
from transformers import AutoModelForCausalLM, GenerationConfig

# Add MoonCastSemantic to path for imports
sys.path.append("../MoonCastSemantic")
from tokenizer.tokenizer import get_tokenizer_and_extra_tokens


class TextToSemantic:
    """
    Simplified text-to-semantic token generator from MoonCast.
    Only generates semantic tokens, no audio generation.
    """
    
    def __init__(self, model_path=None):
        """Initialize the text-to-semantic model.
        
        Args:
            model_path: Path to the pretrained text2semantic model (default: MoonCast resources)
        """
        # Set default model path to local resources in CustomBuild
        if model_path is None:
            # Get absolute path to local resources in CustomBuild
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "resources", "text2semantic")
            model_path = os.path.abspath(model_path)
        
        # Initialize tokenizer
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
        
        # Load the text2semantic model from existing MoonCast installation
        print(f"Loading model from: {model_path}")
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda:0", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            force_download=False  # Don't download, use existing
        ).to(torch.cuda.current_device())
        
        # Generation config
        self.generate_config = GenerationConfig(
            max_new_tokens=200 * 50,  # no more than 200s per turn
            do_sample=True,
            top_k=30,
            top_p=0.8,
            temperature=0.8,
            eos_token_id=self.media_end,
        )
    
    def _clean_text(self, text):
        """Clean input text by removing special characters."""
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
        """Process dialogue text into tokenized format.
        
        Args:
            dialogue: List of dialogue turns with 'role' and 'text' keys
            
        Returns:
            List of processed dialogue turns with 'bpe_ids' added
        """
        processed_dialogue = []
        for turn in dialogue:
            processed_turn = turn.copy()
            processed_turn["bpe_ids"] = self.tokenizer.encode(self._clean_text(turn["text"]))
            processed_dialogue.append(processed_turn)
        return processed_dialogue
    
    @torch.inference_mode()
    def generate_semantic_tokens(self, dialogue, streaming=False):
        """
        Generate semantic tokens from dialogue text.
        
        Args:
            dialogue: List of dialogue turns, each with 'role' and 'text' keys
                     Example: [{"role": "0", "text": "Hello"}, {"role": "1", "text": "Hi"}]
            streaming: Whether to use streaming generation
            
        Returns:
            List of semantic token sequences for each dialogue turn
        """
        # Process text
        processed_dialogue = self._process_text(dialogue)
        
        # Build role IDs
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
        
        # Build initial prompt from user messages
        prompt = []
        for turn in processed_dialogue:
            role_id = turn["role"]
            cur_user_ids = user_role_0_ids if role_id == "0" else user_role_1_ids
            cur_start_ids = cur_user_ids + turn["bpe_ids"] + [self.msg_end]
            prompt = prompt + cur_start_ids
        
        prompt = torch.LongTensor(prompt).unsqueeze(0).to(torch.cuda.current_device())
        generation_config = self.generate_config
        
        # Generate semantic tokens for each turn
        semantic_tokens_list = []
        
        for turn in processed_dialogue:
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
    
    def generate_semantic_tokens_simple(self, text, role="0"):
        """
        Simple interface to generate semantic tokens for a single text input.
        
        Args:
            text: Input text string
            role: Speaker role ("0" or "1")
            
        Returns:
            numpy array of semantic tokens
        """
        dialogue = [{"role": role, "text": text}]
        semantic_tokens_list = self.generate_semantic_tokens(dialogue)
        return semantic_tokens_list[0]  # Return first (and only) result


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = TextToSemantic()
    
    # Example 1: Single text input
    text = "Hello, how are you today?"
    semantic_tokens = model.generate_semantic_tokens_simple(text, role="0")
    print(f"Generated {len(semantic_tokens[0])} semantic tokens for: {text}")
    
    # Example 2: Dialogue
    dialogue = [
        {"role": "0", "text": "Hello, how are you?"},
        {"role": "1", "text": "I'm doing great, thank you!"},
        {"role": "0", "text": "That's wonderful to hear."}
    ]
    
    semantic_tokens_list = model.generate_semantic_tokens(dialogue)
    print(f"Generated {len(semantic_tokens_list)} semantic token sequences")
    
    for i, (turn, tokens) in enumerate(zip(dialogue, semantic_tokens_list)):
        print(f"Turn {i+1} ({turn['role']}): {len(tokens[0])} tokens for '{turn['text']}'")
