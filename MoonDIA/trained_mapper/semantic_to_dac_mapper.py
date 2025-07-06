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
        hidden_dim: int = 256,             # Reduced for small dataset
        num_layers: int = 4,               # Reduced for small dataset
        num_heads: int = 8,
        dropout: float = 0.2,              # Increased dropout for regularization
        max_seq_len: int = 512             # Reduced since max length is ~250
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
            DAC logits [batch_size, seq_len, dac_channels, dac_vocab_size]
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
        
        return logits
    
    def predict(self, semantic_tokens: torch.Tensor, semantic_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict DAC codes from semantic tokens (for inference).
        
        Args:
            semantic_tokens: Input semantic tokens [batch_size, seq_len]
            semantic_lengths: Length of each sequence in the batch [batch_size]
            
        Returns:
            DAC codes [batch_size, seq_len, dac_channels]
        """
        with torch.no_grad():
            logits = self.forward(semantic_tokens, semantic_lengths)
            dac_codes = torch.argmax(logits, dim=-1)  # [batch_size, seq_len, dac_channels]
            return dac_codes
    



def load_training_data(data_dir: str) -> Tuple[list, list, list]:
    """
    Load training data from the generated dataset.
    
    Args:
        data_dir: Path to data/train directory
        
    Returns:
        Tuple of (semantic_tokens, dac_codes, lengths) lists
    """
    data_path = Path(data_dir)
    
    # Find all pairs
    mc_files = sorted(data_path.glob("*.mc.npy"))
    dac_files = sorted(data_path.glob("*.dac.npy"))
    
    if len(mc_files) != len(dac_files):
        raise ValueError(f"Mismatch in file counts: {len(mc_files)} MC files vs {len(dac_files)} DAC files")
    
    semantic_tokens = []
    dac_codes = []
    lengths = []
    
    for mc_file, dac_file in zip(mc_files, dac_files):
        # Load semantic tokens
        sem = np.load(mc_file)
        semantic_tokens.append(sem)
        
        # Load DAC codes
        dac = np.load(dac_file)
        dac_codes.append(dac)
        
        # Store length
        lengths.append(len(sem))
    
    return semantic_tokens, dac_codes, lengths


def pad_sequences(semantic_tokens: list, dac_codes: list, lengths: list, pad_token: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad sequences to the same length for batch processing.
    
    Args:
        semantic_tokens: List of semantic token arrays
        dac_codes: List of DAC code arrays
        lengths: List of sequence lengths
        pad_token: Token to use for padding
        
    Returns:
        Tuple of (padded_semantic, padded_dac, length_tensor)
    """
    max_len = max(lengths)
    batch_size = len(semantic_tokens)
    
    # Pad semantic tokens
    padded_sem = torch.full((batch_size, max_len), pad_token, dtype=torch.long)
    for i, (sem, length) in enumerate(zip(semantic_tokens, lengths)):
        padded_sem[i, :length] = torch.tensor(sem, dtype=torch.long)
    
    # Pad DAC codes
    padded_dac = torch.full((batch_size, max_len, 9), pad_token, dtype=torch.long)
    for i, (dac, length) in enumerate(zip(dac_codes, lengths)):
        padded_dac[i, :length, :] = torch.tensor(dac, dtype=torch.long)
    
    # Create length tensor
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return padded_sem, padded_dac, length_tensor


def train_mapper(
    model: SemanticToDACMapper,
    semantic_tokens: list,
    dac_codes: list,
    lengths: list,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_split: float = 0.2
) -> SemanticToDACMapper:
    """
    Train the semantic to DAC mapper.
    
    Args:
        model: The mapper model to train
        semantic_tokens: Training semantic tokens (list of arrays)
        dac_codes: Training DAC codes (list of arrays)
        lengths: Sequence lengths
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device to train on
        val_split: Validation split ratio
        
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    # Split into train/val
    num_samples = len(semantic_tokens)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size
    
    # Shuffle indices
    indices = torch.randperm(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle training indices
        train_indices_shuffled = train_indices[torch.randperm(len(train_indices))]
        
        for i in range(0, train_size, batch_size):
            batch_indices = train_indices_shuffled[i:i+batch_size]
            
            # Get batch data
            batch_sem = [semantic_tokens[idx] for idx in batch_indices]
            batch_dac = [dac_codes[idx] for idx in batch_indices]
            batch_lengths = [lengths[idx] for idx in batch_indices]
            
            # Pad sequences
            padded_sem, padded_dac, length_tensor = pad_sequences(
                batch_sem, batch_dac, batch_lengths
            )
            
            # Move to device
            padded_sem = padded_sem.to(device)
            padded_dac = padded_dac.to(device)
            length_tensor = length_tensor.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get logits from model
            logits = model.forward(padded_sem, length_tensor)  # [batch_size, seq_len, dac_channels, vocab_size]
            
            # Create mask for non-padded positions
            mask = torch.arange(logits.size(1), device=device)[None, :] < length_tensor[:, None]
            mask = mask.unsqueeze(-1).expand(-1, -1, logits.size(-2))  # [batch_size, seq_len, dac_channels]
            
            # Reshape for loss calculation
            logits_flat = logits.reshape(-1, model.dac_vocab_size)
            targets_flat = padded_dac.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            # Calculate loss only on non-padded positions
            loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, val_size, batch_size):
                batch_indices = val_indices[i:i+batch_size]
                
                # Get batch data
                batch_sem = [semantic_tokens[idx] for idx in batch_indices]
                batch_dac = [dac_codes[idx] for idx in batch_indices]
                batch_lengths = [lengths[idx] for idx in batch_indices]
                
                # Pad sequences
                padded_sem, padded_dac, length_tensor = pad_sequences(
                    batch_sem, batch_dac, batch_lengths
                )
                
                # Move to device
                padded_sem = padded_sem.to(device)
                padded_dac = padded_dac.to(device)
                length_tensor = length_tensor.to(device)
                
                # Forward pass
                logits = model.forward(padded_sem, length_tensor)
                
                # Create mask for non-padded positions
                mask = torch.arange(logits.size(1), device=device)[None, :] < length_tensor[:, None]
                mask = mask.unsqueeze(-1).expand(-1, -1, logits.size(-2))
                
                # Reshape for loss calculation
                logits_flat = logits.reshape(-1, model.dac_vocab_size)
                targets_flat = padded_dac.reshape(-1)
                mask_flat = mask.reshape(-1)
                
                # Calculate loss only on non-padded positions
                loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model


if __name__ == "__main__":
    # Example usage and training
    print("Semantic to DAC Mapper")
    print("=====================")
    
    # Initialize model
    model = SemanticToDACMapper()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    batch_size, seq_len = 2, 100
    semantic_tokens = torch.randint(0, 16384, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(semantic_tokens)
        dac_codes = model.predict(semantic_tokens)
        print(f"Input shape: {semantic_tokens.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Output shape: {dac_codes.shape}")
    
    # Load and train on actual data
    print("\nLoading training data...")
    try:
        # Try aligned dataset first, fall back to original
        try:
            semantic_tokens, dac_codes, lengths = load_training_data("data/train_aligned")
            print(f"Loaded {len(semantic_tokens)} aligned training pairs")
        except:
            semantic_tokens, dac_codes, lengths = load_training_data("data/train")
            print(f"Loaded {len(semantic_tokens)} original training pairs")
            print("⚠️  Consider running align_temporal_lengths.py for better results")
        
        # Train the model
        print("\nStarting training...")
        trained_model = train_mapper(
            model=model,
            semantic_tokens=semantic_tokens,
            dac_codes=dac_codes,
            lengths=lengths,
            num_epochs=200,  # More epochs for small dataset
            batch_size=4,    # Smaller batch size for small dataset
            learning_rate=5e-5,  # Lower learning rate for stability
            val_split=0.3    # Larger validation split for small dataset
        )
        
        # Save the trained model
        torch.save(trained_model.state_dict(), "semantic_to_dac_mapper.pt")
        print("\nModel saved to semantic_to_dac_mapper.pt")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure you have training data in data/train/")
    
    print("Done!")
