"""
Neural Network Models Module
Implements transformer-based architecture for the chatbot
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = query.size(0)
        
        # Linear projections
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        output = self.output_dropout(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class ChatbotModel(nn.Module):
    """Main chatbot transformer model"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 4,
                 num_heads: int = 8, d_ff: int = 1024, max_seq_length: int = 512,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.pad_idx = pad_idx
    
    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask"""
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        mask = self.create_mask(src)
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output projection
        output = self.output_layer(x)
        
        return output
    
    def generate(self, src: torch.Tensor, max_length: int = 100,
                 temperature: float = 0.7, top_k: int = 50) -> torch.Tensor:
        """Generate text autoregressively"""
        self.eval()
        
        with torch.no_grad():
            generated = src.clone()
            
            for _ in range(max_length):
                output = self.forward(generated)
                next_token_logits = output[:, -1, :] / temperature
                
                # Top-k sampling
                filtered_logits = self._top_k_filtering(next_token_logits, top_k)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == 3:  # EOS token
                    break
            
            return generated
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits to top-k candidates"""
        actual_k = min(top_k, logits.size(-1))
        if actual_k >= logits.size(-1):
            return logits
        indices_to_remove = logits < torch.topk(logits, actual_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def save_model(self, path: str) -> None:
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.output_layer.out_features,
            'd_model': self.d_model,
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model weights"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])


def create_model(vocab_size: int, config: Dict[str, Any]) -> ChatbotModel:
    """Factory function to create model from config"""
    return ChatbotModel(
        vocab_size=vocab_size,
        d_model=config.get('embedding_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_heads=8,
        d_ff=config.get('hidden_dim', 512),
        dropout=config.get('dropout', 0.1),
        max_seq_length=config.get('max_seq_length', 512),
    )
