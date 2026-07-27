"""
gpt_model.py
============
Decoder-only GPT-style Transformer — the architecture behind GPT-2/J/3, Llama, Falcon.

Difference from our Encoder-Decoder model:
  - NO encoder, NO cross-attention
  - Single stack of decoder layers with ONLY masked self-attention
  - Autoregressive language modeling: predict next token given all previous tokens
  - One unified sequence: source and target are the same (no separate src/tgt)

This is the architecture that scales to 6B+ parameters in production LLMs.

Two variants implemented:
  - GPTModel:   Standard (LayerNorm, ReLU, sinusoidal PE) — consistent with our codebase
  - LlamaModel: Modern  (RMSNorm, SwiGLU, RoPE)          — what Llama/Mistral use

Why decoder-only at scale?
  - Simpler: one set of weights, one pass
  - Better scaling behavior (empirically)
  - Encoder-decoder shines for tasks with distinct input/output (translation)
  - Decoder-only trained on raw text generalizes to almost any task via prompting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .layers import AddAndNorm
from .embeddings import TokenEmbedding, PositionalEncoding


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD GPT MODEL (consistent with our existing codebase)
# ─────────────────────────────────────────────────────────────────────────────

class GPTDecoderLayer(nn.Module):
    """
    Single GPT decoder layer: masked self-attention + FFN.

    Unlike the Transformer decoder, there is NO cross-attention here —
    this is a self-contained language model layer.

    Diagram:
      x ──────────────────────────────────────────────► +
      │                                                  │
      │   ┌──────────────────────────────────────────┐  │
      └──►│  Masked Multi-Head Self-Attention        │──┘
          └──────────────────────────────────────────┘
          ↓  Add & Norm
      x ──────────────────────────────────────────────► +
      │                                                  │
      │   ┌──────────────────────────────────────────┐  │
      └──►│  Position-wise FFN                       │──┘
          └──────────────────────────────────────────┘
          ↓  Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm_1 = AddAndNorm(d_model, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.add_norm_2 = AddAndNorm(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:           (batch, seq, d_model)
            causal_mask: (batch, 1, seq, seq) — lower-triangular causal mask
        Returns:
            (batch, seq, d_model)
        """
        attn_out = self.self_attention(x, x, x, mask=causal_mask)
        x = self.add_norm_1(x, attn_out)
        ff_out = self.feed_forward(x)
        x = self.add_norm_2(x, ff_out)
        return x


class GPTModel(nn.Module):
    """
    Standard GPT-style language model.

    Architecture (same as GPT-2):
      Token Embedding + Positional Encoding
      → N × GPTDecoderLayer (masked self-attn + FFN)
      → LayerNorm
      → Linear projection → vocab logits

    For 6B parameters:
      vocab_size = 32000, d_model = 4096, num_heads = 32,
      num_layers = 32, d_ff = 16384
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        num_heads: int = 32,
        num_layers: int = 32,
        d_ff: int = 16384,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Decoder stack
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output projection — weight-tied with token embedding
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular causal mask: position i can only attend to 0..i."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq) — token IDs
        Returns:
            logits: (batch, seq, vocab_size)
        """
        seq_len = input_ids.size(1)
        causal_mask = self.make_causal_mask(seq_len, input_ids.device)

        x = self.pos_encoding(self.token_embedding(input_ids))  # (batch, seq, d_model)

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.norm(x)
        return self.lm_head(x)  # (batch, seq, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with temperature + top-k sampling.

        Args:
            prompt_ids:     (1, seq) — prompt token IDs
            max_new_tokens: How many new tokens to generate
            temperature:    > 1 = more random, < 1 = more focused, 1 = as-is
            top_k:          Keep only top-k tokens at each step (0 = no filter)
        Returns:
            (1, seq + max_new_tokens) — prompt + generated tokens
        """
        self.eval()
        generated = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Only feed the last max_seq_len tokens if context overflows
                context = generated[:, -2048:]
                logits = self.forward(context)          # (1, seq, vocab)
                next_logits = logits[:, -1, :] / temperature  # (1, vocab)

                # Top-k filtering: zero out all but top-k logits
                if top_k > 0:
                    topk_vals, _ = next_logits.topk(top_k)
                    threshold = topk_vals[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # sample
                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# MODERN LLAMA-STYLE MODEL
# (RMSNorm + SwiGLU + Rotary Positional Embeddings)
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Simpler than LayerNorm — no mean subtraction, just RMS scaling:
        RMSNorm(x) = x / RMS(x) × γ
        RMS(x) = sqrt(mean(x²) + ε)

    Why? Removing the mean-centering step saves ~30% compute vs LayerNorm
    with no quality loss. Used in Llama, Mistral, Falcon.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # learned scale γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS across feature dimension
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE, Su et al. 2021).

    Instead of adding a positional vector, RoPE *rotates* the Q and K vectors
    in 2D subspaces by angles proportional to position:
        q_rotated[2i:2i+2] = rotate(q[2i:2i+2], pos × θᵢ)
        θᵢ = 1 / 10000^(2i / d_k)

    Key advantage: relative position information is naturally captured because:
        dot(q_rotated_at_pos_m, k_rotated_at_pos_n) depends only on (m - n)

    Used in: Llama, Mistral, Falcon, PaLM, GPT-NeoX.
    """

    def __init__(self, d_k: int, max_seq_len: int = 4096):
        super().__init__()
        # Compute inverse frequencies for each dimension pair
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)            # (seq_len, d_k/2)
        emb = torch.cat([freqs, freqs], dim=-1)          # (seq_len, d_k)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate: [x1, x2, x3, x4] → [-x3, -x4, x1, x2]"""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary embedding to Q or K tensor."""
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq,d_k)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


class SwiGLU(nn.Module):
    """
    SwiGLU activation (Shazeer, 2020): used in Llama, PaLM, GPT-4.

    FFN(x) = (xW₁ ⊙ SiLU(xW₂)) · W₃

    Split the FFN into two parallel projections — one acts as a gate:
        gate   = SiLU(x · W_gate)   — controls information flow
        hidden = x · W_up           — the actual transformation
        output = (gate ⊙ hidden) · W_down

    Why better than ReLU?
        - Smooth gradient everywhere (SiLU = x × sigmoid(x))
        - The gate mechanism adds expressivity without extra depth
        - ~10% lower perplexity than ReLU-FFN at same param count

    Note: To keep params equal to a standard FFN with d_ff neurons,
    SwiGLU uses 2/3 × d_ff for each of the two projections.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # Two parallel up-projections + one down-projection
        hidden = int(d_ff * 2 / 3)  # 2/3 × d_ff to keep params equal
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up   = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))   # SiLU = x × sigmoid(x)
        hidden = self.w_up(x)
        return self.dropout(self.w_down(gate * hidden))


class LlamaAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Positional Embeddings (RoPE).
    Applied to Q and K before computing attention scores.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_k)
        self.attn_weights = None

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = x.shape
        Q = self.split_heads(self.W_q(x))  # (b, h, s, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Apply RoPE to Q and K
        Q = self.rope.apply_rope(Q, s)
        K = self.rope.apply_rope(K, s)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        self.attn_weights = F.softmax(scores, dim=-1)
        out = self.dropout(self.attn_weights) @ V

        # Combine heads
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.W_o(out)


class LlamaDecoderLayer(nn.Module):
    """
    Modern Llama-style decoder layer:
      Pre-norm (RMSNorm BEFORE sublayer, not after)
      RoPE attention + SwiGLU FFN

    Pre-norm vs Post-norm:
      - Post-norm (original paper): LayerNorm(x + Sublayer(x))
        → Deeper models can have unstable gradients
      - Pre-norm (modern):  x + Sublayer(RMSNorm(x))
        → Gradients flow cleanly through the residual path
        → Allows training models with 100+ layers stably
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm_1 = RMSNorm(d_model)
        self.attention = LlamaAttention(d_model, num_heads, dropout)
        self.norm_2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm: normalize BEFORE sublayer, add residual AFTER
        x = x + self.attention(self.norm_1(x), mask)
        x = x + self.ffn(self.norm_2(x))
        return x


class LlamaModel(nn.Module):
    """
    Modern Llama-style language model.

    Improvements over GPT-2:
      RMSNorm    → faster normalization, same quality
      RoPE       → better relative position encoding, no fixed max length
      SwiGLU     → better activation function (~10% better perplexity)
      Pre-norm   → stable training at any depth
      No bias    → cleaner weight matrices (empirically works better at scale)

    6B config (matches Llama-7B architecture closely):
      vocab_size=32000, d_model=4096, heads=32, layers=32, d_ff=16384
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        num_heads: int = 32,
        num_layers: int = 32,
        d_ff: int = 16384,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool() \
                    .unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)  # (batch, seq, d_model)
        mask = self.make_causal_mask(x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.norm(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        self.eval()
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            context = generated[:, -2048:]
            logits = self.forward(context)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
