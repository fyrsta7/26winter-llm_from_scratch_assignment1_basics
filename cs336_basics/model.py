"""
Transformer Language Model components.
Implements the building blocks for a GPT-style decoder-only Transformer.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -3.0,
    b: float = 3.0,
) -> Tensor:
    """
    Truncated normal initialization.
    Fills the input Tensor with values drawn from a truncated normal distribution.
    Values are truncated to [a*std + mean, b*std + mean].
    
    Args:
        tensor: Tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        a: Lower truncation bound (in units of std)
        b: Upper truncation bound (in units of std)
    
    Returns:
        The initialized tensor
    """
    with torch.no_grad():
        # Use PyTorch's native truncated normal
        tensor.normal_(mean, std)
        tensor.clamp_(mean + a * std, mean + b * std)
    return tensor


class Linear(nn.Module):
    """
    Linear transformation: y = xW (without bias).
    
    Args:
        in_features: Size of input dimension
        out_features: Size of output dimension
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal: N(0, 2/(d_in + d_out))"""
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0, b=3.0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (..., in_features)
        
        Returns:
            Tensor of shape (..., out_features)
        """
        # y = xW^T (since W is stored as (out_features, in_features))
        return torch.matmul(x, self.weight.t())


class Embedding(nn.Module):
    """
    Embedding layer: lookup table mapping token IDs to vectors.
    
    Args:
        vocab_size: Number of embeddings
        d_model: Dimension of each embedding vector
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Weight shape: (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with truncated normal: N(0, 1)"""
        trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: Integer tensor of shape (...)
        
        Returns:
            Embeddings of shape (..., d_model)
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Normalizes input by RMS and applies learnable affine transform.
    Computation is done in float32 for numerical stability.
    
    Args:
        d_model: Dimension of input
        eps: Small constant for numerical stability
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # Learnable scale parameter (gain), initialized to 1
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of shape (..., d_model)
        """
        # Save original dtype
        original_dtype = x.dtype
        
        # Convert to float32 for computation
        x_float = x.float()
        
        # Compute RMS: sqrt(mean(x^2) + eps)
        # Formula: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_normalized = (x_float / rms) * self.weight
        
        # Convert back to original dtype
        return x_normalized.to(original_dtype)


def silu(x: Tensor) -> Tensor:
    """
    SiLU (Sigmoid Linear Unit) activation function.
    Also known as Swish: SiLU(x) = x * sigmoid(x)
    
    Args:
        x: Input tensor of any shape
    
    Returns:
        Output tensor of same shape as input
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU feedforward network.
    
    SwiGLU(x) = (SiLU(xW1) ⊙ xW3) W2
    where ⊙ denotes element-wise multiplication.
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension (typically ~8/3 * d_model, rounded to multiple of 64)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Three linear transformations
        self.w1 = Linear(d_model, d_ff)  # Gate projection
        self.w2 = Linear(d_ff, d_model)  # Down projection
        self.w3 = Linear(d_model, d_ff)  # Up projection
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (..., d_model)
        
        Returns:
            Tensor of shape (..., d_model)
        """
        # SwiGLU(x) = (SiLU(xW1) ⊙ xW3) W2
        gate = silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotary position embeddings to queries or keys.
    Pre-computes cos/sin values for efficiency.
    
    Args:
        d_k: Dimension of each attention head
        theta: Base for computing frequencies (default 10000)
        max_seq_len: Maximum sequence length to pre-compute
    """
    
    def __init__(self, d_k: int, theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        
        # Pre-compute frequency bands
        # freq_i = 1 / (theta ^ (2i / d_k)) for i in [0, d_k/2)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        
        # Pre-compute position indices
        positions = torch.arange(max_seq_len).float()
        
        # Compute outer product: (max_seq_len, d_k/2)
        angles = torch.outer(positions, freqs)
        
        # Pre-compute cos and sin values
        # Shape: (max_seq_len, d_k/2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Register as buffers (not parameters, not saved in state_dict)
        # persistent=False means they won't be saved in state_dict
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Tensor of shape (..., seq_len, d_k)
            positions: Position indices of shape (..., seq_len)
        
        Returns:
            Tensor of shape (..., seq_len, d_k) with RoPE applied
        """
        # Get cos/sin for the given positions
        # positions: (..., seq_len) -> need to index into (max_seq_len, d_k/2)
        cos = self.cos[positions]  # (..., seq_len, d_k/2)
        sin = self.sin[positions]  # (..., seq_len, d_k/2)
        
        # Split x into even and odd indices
        # x: (..., seq_len, d_k)
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]   # (..., seq_len, d_k/2)
        
        # Apply rotation:
        # x_even' = x_even * cos - x_odd * sin
        # x_odd' = x_even * sin + x_odd * cos
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        
        # Interleave back together
        x_out = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_out = x_out.flatten(-2, -1)  # (..., seq_len, d_k)
        
        return x_out


def softmax(x: Tensor, dim: int) -> Tensor:
    """
    Numerically stable softmax.
    
    Subtracts max before exp to prevent overflow.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension to apply softmax over
    
    Returns:
        Softmax probabilities of same shape as input
    """
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Compute softmax
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., values, d_v)
        mask: Optional boolean mask of shape (..., queries, keys)
              True indicates positions to mask (set to -inf before softmax)
    
    Returns:
        Attention output of shape (..., queries, d_v)
    """
    # Get dimension for scaling
    d_k = Q.shape[-1]
    
    # Compute attention scores: QK^T / sqrt(d_k)
    # Q: (..., queries, d_k), K: (..., keys, d_k)
    # scores: (..., queries, keys)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Set masked positions to -inf (they'll become 0 after softmax)
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Apply softmax
    attn_weights = softmax(scores, dim=-1)
    
    # Apply attention weights to values
    # attn_weights: (..., queries, keys), V: (..., values, d_v)
    # output: (..., queries, d_v)
    output = torch.matmul(attn_weights, V)
    
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (optionally causal, optionally with RoPE).
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        use_rope: Whether to use RoPE
        causal: Whether to use causal masking
        theta: RoPE theta parameter
        max_seq_len: Maximum sequence length
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        causal: bool = False,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.use_rope = use_rope
        self.causal = causal
        
        # Q, K, V projections
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = Linear(d_model, d_model)
        
        # RoPE for positional encoding (if needed)
        if use_rope:
            self.rope = RoPE(self.d_k, theta=theta, max_seq_len=max_seq_len)
    
    def forward(
        self, 
        x: Tensor,
        token_positions: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position indices of shape (..., seq_len)
        
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # Get shape information
        *batch_dims, seq_len, d_model = x.shape
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        
        # Flatten batch dimensions
        x_flat = x.view(batch_size, seq_len, d_model)
        
        # Project to Q, K, V
        Q = self.q_proj(x_flat)  # (batch, seq_len, d_model)
        K = self.k_proj(x_flat)  # (batch, seq_len, d_model)
        V = self.v_proj(x_flat)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE to Q and K if enabled
        if self.use_rope:
            if token_positions is None:
                # Default to sequential positions
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            else:
                # Use provided token_positions
                # It should have shape (..., seq_len) which broadcasts with (batch_size, seq_len)
                positions = token_positions
            
            # Flatten positions to match flattened batch dimensions
            # positions: (..., seq_len) -> (batch_size, seq_len)
            positions_flat = positions.reshape(-1, seq_len)
            if positions_flat.shape[0] == 1:
                # Broadcasting case: expand to match batch_size
                positions_flat = positions_flat.expand(batch_size, -1)
            
            # Apply RoPE: (batch, num_heads, seq_len, d_k)
            # Expand positions for all heads: (batch, seq_len) -> (batch, num_heads, seq_len)
            positions_expanded = positions_flat.unsqueeze(1).expand(-1, self.num_heads, -1)
            Q = self.rope(Q, positions_expanded)
            K = self.rope(K, positions_expanded)
        
        # Create causal mask if needed
        mask = None
        if self.causal:
            # mask[i, j] = True if j > i (future positions)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1
            )
            # Expand for batch and heads: (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Apply scaled dot-product attention
        # Q, K, V: (batch, num_heads, seq_len, d_k)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # Reshape back: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Apply output projection
        output = self.output_proj(attn_output)
        
        # Reshape back to original batch dimensions
        output = output.view(*batch_dims, seq_len, d_model)
        
        return output


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        theta: RoPE theta parameter
        max_seq_len: Maximum sequence length
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Q, K, V projections
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = Linear(d_model, d_model)
        
        # RoPE for positional encoding
        self.rope = RoPE(self.d_k, theta=theta, max_seq_len=max_seq_len)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)  # (batch, seq_len, d_model)
        V = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE to Q and K
        # Create position indices: (batch, seq_len)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Apply RoPE: (batch, num_heads, seq_len, d_k)
        # RoPE expects (..., seq_len, d_k), so we apply it correctly
        Q = self.rope(Q, positions.unsqueeze(1).expand(-1, self.num_heads, -1))
        K = self.rope(K, positions.unsqueeze(1).expand(-1, self.num_heads, -1))
        
        # Create causal mask: lower triangular matrix
        # mask[i, j] = True if j > i (future positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        )
        # Expand for batch and heads: (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply scaled dot-product attention
        # Q, K, V: (batch, num_heads, seq_len, d_k)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        
        # Reshape back: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Apply output projection
        output = self.output_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.
    
    Architecture:
        y = x + MHA(RMSNorm(x))
        z = y + FFN(RMSNorm(y))
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension
        theta: RoPE theta parameter
        max_seq_len: Maximum sequence length
        eps: RMSNorm epsilon
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        # Layer norms
        self.ln1 = RMSNorm(d_model, eps=eps)
        self.ln2 = RMSNorm(d_model, eps=eps)
        
        # Multi-head self-attention
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, theta=theta, max_seq_len=max_seq_len
        )
        
        # Feedforward network (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm attention with residual
        y = x + self.attn(self.ln1(x))
        
        # Pre-norm FFN with residual
        z = y + self.ffn(self.ln2(y))
        
        return z


class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    Architecture:
        embedding -> num_layers × TransformerBlock -> RMSNorm -> lm_head
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension
        theta: RoPE theta parameter
        max_seq_len: Maximum sequence length
        eps: RMSNorm epsilon
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embedding
        self.embedding = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff,
                theta=theta, max_seq_len=max_seq_len, eps=eps
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(d_model, eps=eps)
        
        # Language model head (projects to vocabulary)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: Token IDs of shape (batch, seq_len)
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embedding(token_ids)  # (batch, seq_len, d_model)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return logits
