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
import json
from pathlib import Path
from typing import Optional, Tuple, List


class SemanticToDACMapper(nn.Module):
    """
    Neural network that maps MoonCast semantic tokens to DIA DAC codes.
    
    Architecture:
    - Input: MoonCast semantic tokens (variable length)
    - Output: Discrete DAC codes (9 bands, 1024 possible values each)
    - Classification approach for better audio quality
    """
    
    def __init__(
        self,
        semantic_vocab_size: int = 16384,  # MoonCast semantic vocabulary size
        dac_channels: int = 9,             # DAC has 9 bands
        dac_vocab_size: int = 1024,        # DAC codes range from 0-1023
        hidden_dim: int = 256,             # Smaller to prevent overfitting
        num_layers: int = 3,               # Fewer layers
        num_heads: int = 8,
        dropout: float = 0.3,              # More dropout
        max_seq_len: int = 1024
    ):
        super().__init__()
        
        self.semantic_vocab_size = semantic_vocab_size
        self.dac_channels = dac_channels
        self.dac_vocab_size = dac_vocab_size
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
            dim_feedforward=hidden_dim * 2,  # Smaller FF
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Length predictor (predicts output sequence length)
        self.length_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Ensure positive, smooth length
        )
        
        # Output projection to DAC logits (classification)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dac_channels * dac_vocab_size)  # [B, T, 9*1024]
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        semantic_tokens: torch.Tensor,
        semantic_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: convert semantic tokens to DAC logits.
        
        Args:
            semantic_tokens: Input semantic tokens [batch_size, seq_len]
            semantic_lengths: Length of each sequence in the batch [batch_size]
            target_lengths: Target DAC lengths [batch_size] (for training)
            
        Returns:
            Tuple of (DAC logits, predicted lengths)
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
        
        # Predict output length
        if semantic_lengths is not None:
            # Masked mean pooling
            mask = torch.arange(seq_len, device=semantic_tokens.device)[None, :] < semantic_lengths[:, None]
            masked_x = x * mask.unsqueeze(-1)
            pooled = masked_x.sum(dim=1) / semantic_lengths.unsqueeze(-1).float()
        else:
            pooled = x.mean(dim=1)
        
        predicted_lengths = self.length_predictor(pooled).squeeze(-1)  # [batch_size]
        
        # Use target lengths during training, predicted lengths during inference
        if target_lengths is not None:
            use_lengths = target_lengths
        else:
            use_lengths = predicted_lengths
        
        # Project to DAC logits
        dac_logits = self.output_projection(x)  # [batch_size, seq_len, dac_channels * dac_vocab_size]
        dac_logits = dac_logits.view(batch_size, seq_len, self.dac_channels, self.dac_vocab_size)
        
        return dac_logits, predicted_lengths
    
    def predict(self, semantic_tokens: torch.Tensor, semantic_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict discrete DAC codes from semantic tokens using classification.
        
        Args:
            semantic_tokens: Input semantic tokens [batch_size, seq_len]
            semantic_lengths: Length of each sequence in the batch [batch_size]
            
        Returns:
            DAC codes [batch_size, predicted_seq_len, dac_channels]
        """
        with torch.no_grad():
            dac_logits, predicted_lengths = self.forward(semantic_tokens, semantic_lengths)  # [B, T, 9, 1024]
            
            # Get class predictions: [B, T, 9]
            class_preds = dac_logits.argmax(dim=-1)
            
            # Estimate predicted output length
            pred_len = int(predicted_lengths.clamp(min=semantic_tokens.size(1)).mean().item())
            
            # Interpolate to match predicted length
            # [B, T, 9] → [B, 9, T]
            class_preds = class_preds.permute(0, 2, 1)
            
            # Use nearest neighbor interpolation on discrete class predictions
            class_preds_interp = F.interpolate(
                class_preds.float(), size=pred_len, mode='nearest'
            ).long()
            
            # [B, 9, pred_len] → [B, pred_len, 9]
            dac_codes = class_preds_interp.permute(0, 2, 1)
            
            return dac_codes


def load_audio_assessment_results(assessment_file: str = "audio_assessment_results.json", min_similarity: float = 1.0) -> List[int]:
    """
    Load the indices of satisfactory pairs from audio assessment results.
    """
    with open(assessment_file, 'r') as f:
        results = json.load(f)
    
    # Extract indices of satisfactory pairs with perfect similarity
    satisfactory_indices = []
    for pair in results.get("satisfactory_pairs", []):
        if pair.get("both_satisfactory", False) and pair.get("similarity", 0) >= min_similarity:
            satisfactory_indices.append(pair["index"])
    
    print(f"Loaded {len(satisfactory_indices)} satisfactory pairs with similarity >= {min_similarity} from audio assessment")
    return sorted(satisfactory_indices)


def load_training_data(data_dir: str, assessment_file: str = "audio_assessment_results.json", min_similarity: float = 1.0) -> Tuple[list, list, list]:
    """
    Load training data from the generated dataset, filtering to only satisfactory pairs.
    """
    data_path = Path(data_dir)
    
    # Load satisfactory indices
    satisfactory_indices = load_audio_assessment_results(assessment_file, min_similarity)
    
    semantic_tokens = []
    dac_codes = []
    lengths = []
    
    for idx in satisfactory_indices:
        # Construct filenames with zero-padding
        mc_file = data_path / f"{idx:05d}.mc.npy"
        dac_file = data_path / f"{idx:05d}.dac.npy"
        
        if not mc_file.exists() or not dac_file.exists():
            print(f"Warning: Missing files for index {idx}")
            continue
        
        # Load semantic tokens
        sem = np.load(mc_file)
        semantic_tokens.append(sem)
        
        # Load DAC codes and convert to int for classification
        dac = np.load(dac_file).astype(np.int64)
        dac_codes.append(dac)
        
        # Store DAC code length
        lengths.append(dac.shape[0])
    
    print(f"Successfully loaded {len(semantic_tokens)} satisfactory training pairs")
    return semantic_tokens, dac_codes, lengths


def pad_sequences(semantic_tokens: list, dac_codes: list, lengths: list, pad_token: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad sequences to the same length for batch processing.
    """
    if len(semantic_tokens) != len(dac_codes) or len(semantic_tokens) != len(lengths):
        raise ValueError(f"Mismatch in list lengths: semantic_tokens={len(semantic_tokens)}, dac_codes={len(dac_codes)}, lengths={len(lengths)}")
    
    max_len = max(lengths)
    batch_size = len(semantic_tokens)
    
    # Pad semantic tokens to match DAC code lengths using interpolation
    padded_sem = torch.full((batch_size, max_len), pad_token, dtype=torch.long)
    for i, (sem, dac, length) in enumerate(zip(semantic_tokens, dac_codes, lengths)):
        if len(sem) < length:
            # Use linear interpolation with nearest neighbor for discrete tokens
            sem_indices = np.linspace(0, len(sem) - 1, length)
            sem_indices = np.round(sem_indices).astype(int)
            sem_indices = np.clip(sem_indices, 0, len(sem) - 1)
            interpolated_sem = sem[sem_indices]
            padded_sem[i, :length] = torch.tensor(interpolated_sem, dtype=torch.long)
        else:
            # Truncate if semantic tokens are longer
            padded_sem[i, :length] = torch.tensor(sem[:length], dtype=torch.long)
    
    # Pad DAC codes (keep as long for classification)
    padded_dac = torch.zeros((batch_size, max_len, 9), dtype=torch.long)
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
    num_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_split: float = 0.2,
    patience: int = 10  # Early stopping patience
) -> SemanticToDACMapper:
    """
    Train the semantic to DAC mapper.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )
    
    # CrossEntropy loss for classification
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Split into train/val
    num_samples = len(semantic_tokens)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size
    
    # Shuffle indices
    indices = torch.randperm(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
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
            
            # Get continuous DAC values from model
            dac_logits, predicted_lengths = model.forward(padded_sem, length_tensor, target_lengths=length_tensor)
            
            # Create mask for non-padded positions
            mask = torch.arange(dac_logits.size(1), device=device)[None, :] < length_tensor[:, None]
            
            # Calculate CrossEntropy loss for each DAC channel
            dac_loss = 0
            for channel in range(dac_logits.size(2)):  # 9 channels
                channel_logits = dac_logits[:, :, channel, :]  # [B, T, 1024]
                channel_targets = padded_dac[:, :, channel]    # [B, T]
                
                # Apply mask and calculate loss
                valid_mask = mask[:, :, channel] if mask.dim() > 2 else mask
                if valid_mask.any():
                    channel_loss = ce_criterion(
                        channel_logits[valid_mask], 
                        channel_targets[valid_mask]
                    )
                    dac_loss += channel_loss
            
            dac_loss = dac_loss / dac_logits.size(2)  # Average over channels
            
            # Calculate length prediction loss
            length_loss = F.mse_loss(predicted_lengths, length_tensor.float())
            
            # Combined loss
            loss = dac_loss + 0.1 * length_loss
            
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
                dac_logits, predicted_lengths = model.forward(padded_sem, length_tensor, target_lengths=length_tensor)
                
                # Create mask for non-padded positions
                mask = torch.arange(dac_logits.size(1), device=device)[None, :] < length_tensor[:, None]
                
                # Calculate CrossEntropy loss for each DAC channel
                dac_loss = 0
                for channel in range(dac_logits.size(2)):  # 9 channels
                    channel_logits = dac_logits[:, :, channel, :]  # [B, T, 1024]
                    channel_targets = padded_dac[:, :, channel]    # [B, T]
                    
                    # Apply mask and calculate loss
                    valid_mask = mask[:, :, channel] if mask.dim() > 2 else mask
                    if valid_mask.any():
                        channel_loss = ce_criterion(
                            channel_logits[valid_mask], 
                            channel_targets[valid_mask]
                        )
                        dac_loss += channel_loss
                
                dac_loss = dac_loss / dac_logits.size(2)  # Average over channels
                
                # Calculate length prediction loss
                length_loss = F.mse_loss(predicted_lengths, length_tensor.float())
                
                # Combined loss
                loss = dac_loss + 0.1 * length_loss
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✅ New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"  🛑 Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
    
    # Load best model if we didn't early stop
    if best_model_state is not None and patience_counter < patience:
        model.load_state_dict(best_model_state)
        print(f"  ✅ Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model


if __name__ == "__main__":
    # Example usage and training
    print("Semantic to DAC Mapper (Classification)")
    print("=======================================")
    
    # Initialize model
    model = SemanticToDACMapper()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    batch_size, seq_len = 2, 100
    semantic_tokens = torch.randint(0, 16384, (batch_size, seq_len))
    
    with torch.no_grad():
        dac_logits, predicted_lengths = model(semantic_tokens)
        dac_codes = model.predict(semantic_tokens)
        print(f"Input shape: {semantic_tokens.shape}")
        print(f"DAC logits shape: {dac_logits.shape}")
        print(f"Predicted DAC codes shape: {dac_codes.shape}")
        print(f"DAC codes range: {dac_codes.min().item()}-{dac_codes.max().item()}")
    
    # Load and train on actual data
    print("\nLoading training data...")
    try:
        # Load only satisfactory pairs from audio assessment
        semantic_tokens, dac_codes, lengths = load_training_data("data/train", "audio_assessment_results.json")
        
        # Train the model
        print("\nStarting training...")
        trained_model = train_mapper(
            model=model,
            semantic_tokens=semantic_tokens,
            dac_codes=dac_codes,
            lengths=lengths,
            num_epochs=50,  # Fewer epochs to prevent overfitting
            batch_size=8,    # Larger batch size for stability
            learning_rate=5e-5,  # Lower learning rate
            val_split=0.3    # More validation data
        )
        
        # Save the trained model
        torch.save(trained_model.state_dict(), "semantic_to_dac_mapper.pt")
        print("\nModel saved to semantic_to_dac_mapper.pt")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure you have training data in data/train/ and audio_assessment_results.json")
    
    print("Done!")
