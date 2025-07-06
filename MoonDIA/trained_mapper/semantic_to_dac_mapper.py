"""
semantic_to_dac_mapper.py
========================
A neural network model that maps MoonCast semantic tokens to DIA DAC codes.

This model learns to convert semantic token sequences from MoonCast to the 
corresponding DAC code sequences that DIA would generate for the same text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class SemanticToDACMapper(nn.Module):
    """
    Neural network that maps MoonCast semantic tokens to DIA DAC codes.
    
    Architecture:
    - Input: MoonCast semantic tokens (variable length)
    - Output: DIA DAC codes (9-band, variable length)
    """
    
    def __init__(
        self,
        semantic_vocab_size: int = 16384,  # MoonCast semantic vocabulary size
        dac_vocab_size: int = 1024,        # DAC vocabulary size (typically 1024)
        dac_channels: int = 9,             # DAC has 9 bands
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        self.semantic_vocab_size = semantic_vocab_size
        self.dac_vocab_size = dac_vocab_size
        self.dac_channels = dac_channels
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Semantic token embedding
        self.semantic_embedding = nn.Embedding(semantic_vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Transformer encoder for processing semantic tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection to DAC codes
        self.output_projection = nn.Linear(hidden_dim, dac_channels * dac_vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        semantic_tokens: torch.Tensor,
        semantic_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: convert semantic tokens to DAC codes.
        
        Args:
            semantic_tokens: Input semantic tokens [batch_size, seq_len]
            semantic_lengths: Length of each sequence in the batch [batch_size]
            
        Returns:
            DAC codes [batch_size, seq_len, dac_channels]
        """
        batch_size, seq_len = semantic_tokens.shape
        
        # Embed semantic tokens
        x = self.semantic_embedding(semantic_tokens)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Create attention mask if lengths provided
        attention_mask = None
        if semantic_lengths is not None:
            attention_mask = torch.arange(seq_len, device=semantic_tokens.device)[None, :] >= semantic_lengths[:, None]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)
        
        # Project to DAC output space
        logits = self.output_projection(x)  # [batch_size, seq_len, dac_channels * dac_vocab_size]
        
        # Reshape to separate channels
        logits = logits.view(batch_size, seq_len, self.dac_channels, self.dac_vocab_size)
        
        # Sample DAC codes from logits
        dac_codes = torch.argmax(logits, dim=-1)  # [batch_size, seq_len, dac_channels]
        
        return dac_codes
    
    def generate(
        self,
        semantic_tokens: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Generate DAC codes from semantic tokens with sampling.
        
        Args:
            semantic_tokens: Input semantic tokens [batch_size, seq_len]
            max_length: Maximum generation length (defaults to input length)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated DAC codes [batch_size, seq_len, dac_channels]
        """
        if max_length is None:
            max_length = semantic_tokens.shape[1]
        
        batch_size, input_len = semantic_tokens.shape
        device = semantic_tokens.device
        
        # Initialize output with zeros
        output_codes = torch.zeros(batch_size, max_length, self.dac_channels, 
                                 dtype=torch.long, device=device)
        
        # Copy input tokens
        output_codes[:, :input_len, :] = semantic_tokens.unsqueeze(-1).expand(-1, -1, self.dac_channels)
        
        # Generate autoregressively
        for i in range(input_len, max_length):
            # Get current sequence
            current_tokens = output_codes[:, :i, 0]  # Use first channel as semantic tokens
            
            # Forward pass
            with torch.no_grad():
                logits = self.forward(current_tokens)[:, -1, :, :]  # [batch_size, dac_channels, vocab_size]
            
            # Apply sampling
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            if top_p < 1.0:
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from logits
            probs = F.softmax(logits, dim=-1)
            next_codes = torch.multinomial(probs.view(-1, self.dac_vocab_size), 1)
            next_codes = next_codes.view(batch_size, self.dac_channels)
            
            # Add to output
            output_codes[:, i, :] = next_codes
        
        return output_codes


def load_training_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from the generated dataset.
    
    Args:
        data_dir: Path to data/train directory
        
    Returns:
        Tuple of (semantic_tokens, dac_codes) arrays
    """
    data_path = Path(data_dir)
    
    # Find all pairs
    mc_files = sorted(data_path.glob("*.mc.npy"))
    dac_files = sorted(data_path.glob("*.dac.npy"))
    
    if len(mc_files) != len(dac_files):
        raise ValueError(f"Mismatch in file counts: {len(mc_files)} MC files vs {len(dac_files)} DAC files")
    
    semantic_tokens = []
    dac_codes = []
    
    for mc_file, dac_file in zip(mc_files, dac_files):
        # Load semantic tokens
        sem = np.load(mc_file)
        semantic_tokens.append(sem)
        
        # Load DAC codes
        dac = np.load(dac_file)
        dac_codes.append(dac)
    
    return np.array(semantic_tokens), np.array(dac_codes)


def train_mapper(
    model: SemanticToDACMapper,
    semantic_tokens: np.ndarray,
    dac_codes: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> SemanticToDACMapper:
    """
    Train the semantic to DAC mapper.
    
    Args:
        model: The mapper model to train
        semantic_tokens: Training semantic tokens
        dac_codes: Training DAC codes
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.long, device=device)
    dac_codes = torch.tensor(dac_codes, dtype=torch.long, device=device)
    
    num_samples = len(semantic_tokens)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_sem = semantic_tokens[batch_indices]
            batch_dac = dac_codes[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get logits from model
            logits = model.forward(batch_sem)  # [batch_size, seq_len, dac_channels, vocab_size]
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, model.dac_vocab_size)
            targets_flat = batch_dac.view(-1)
            
            # Calculate loss
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (num_samples // batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Semantic to DAC Mapper")
    print("=====================")
    
    # Initialize model
    model = SemanticToDACMapper()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    batch_size, seq_len = 2, 100
    semantic_tokens = torch.randint(0, 16384, (batch_size, seq_len))
    
    with torch.no_grad():
        dac_codes = model(semantic_tokens)
        print(f"Input shape: {semantic_tokens.shape}")
        print(f"Output shape: {dac_codes.shape}")
    
    print("Model ready for training!")
