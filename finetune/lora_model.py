"""
lora_model.py
=============
LoRA (Low-Rank Adaptation) implemented from scratch.

This module shows EXACTLY what HuggingFace PEFT does under the hood,
written from first principles so you understand every step.

Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al. 2021)
https://arxiv.org/abs/2106.09685

───────────────────────────────────────────────────────────────────────────────
THE CORE IDEA
───────────────────────────────────────────────────────────────────────────────

A pretrained weight matrix W ∈ ℝ^(d × k) encodes all the model's knowledge.
Full fine-tuning updates every element: W ← W + ΔW  (billions of parameters).

LoRA observes that the update ΔW has LOW INTRINSIC RANK —
it lies in a much smaller subspace than the full matrix space.

So instead of storing ΔW ∈ ℝ^(d × k) directly, we factor it:

    ΔW = B × A      where B ∈ ℝ^(d × r), A ∈ ℝ^(r × k), r << min(d,k)

The adapted forward pass becomes:

    h = x·Wᵀ + (x·Aᵀ)·Bᵀ × (α/r)
        ──────   ────────────────────
        frozen     LoRA adapter (tiny)

Parameters saved:
    Full fine-tune: d × k  (e.g. 4096 × 4096 = 16.8M per weight)
    LoRA (r=16):   d×r + r×k = 4096×16 + 16×4096 = 131K per weight (99.2% fewer!)

───────────────────────────────────────────────────────────────────────────────
INITIALIZATION
───────────────────────────────────────────────────────────────────────────────
- A is initialized with Kaiming uniform (random Gaussian-like)
- B is initialized to ZERO

So at the start of training: ΔW = B × A = 0 × A = 0
→ The model starts as the unmodified pretrained model. Perfect.

───────────────────────────────────────────────────────────────────────────────
SCALING FACTOR α/r
───────────────────────────────────────────────────────────────────────────────
α is a hyperparameter (usually = r or 2r). The α/r ratio controls the
effective learning rate of the adapter relative to the base model.
Setting α = 2r makes the adapter 2× the scale you'd get without it.
This is separate from the optimizer LR and allows stable training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a LoRA adapter.

    Forward pass:
        h = x · Wᵀ + (x · Aᵀ) · Bᵀ × (α / r)
            ────────   ─────────────────────────
            base (frozen)       LoRA (trainable)

    Args:
        in_features:  Input dimension (k)
        out_features: Output dimension (d)
        r:            LoRA rank. Controls adapter expressivity.
                      r=4  → very small, fast, good for simple tasks
                      r=16 → standard, good balance
                      r=64 → high expressivity, risk of overfitting
        lora_alpha:   Scaling factor α. Effective scale = α/r.
                      Set to r for 1× scale, 2r for 2× scale.
        lora_dropout: Dropout on LoRA path (prevents overfitting on small datasets)
        bias:         Whether base linear has bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32,
        lora_dropout: float = 0.05,
        bias: bool = False,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r  # α/r scale factor

        # ── Base weight (will be frozen after loading pretrained weights) ──
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # ── LoRA adapter: two low-rank matrices ──────────────────────────
        # A: (r, in_features)   — projects input DOWN to rank-r space
        # B: (out_features, r)  — projects UP from rank-r to output space
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # ── Initialize ────────────────────────────────────────────────────
        # A: random (Kaiming), B: zeros → ΔW = B×A = 0 at start
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # ── Freeze base weight ────────────────────────────────────────────
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path: frozen
        base_out = self.linear(x)

        # LoRA path: trainable
        # x → dropout → A → B → scale
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA adapter back into the base weight.

        After training, we can merge ΔW = B×A × scaling into W:
            W_merged = W + B×A×(α/r)

        This produces a standard nn.Linear with NO runtime overhead —
        inference is exactly as fast as the original model.
        """
        merged = nn.Linear(
            self.linear.in_features, self.linear.out_features,
            bias=self.linear.bias is not None
        )
        # W_merged = W + B×A×scale
        delta_W = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        merged.weight = nn.Parameter(self.linear.weight + delta_W)
        if self.linear.bias is not None:
            merged.bias = nn.Parameter(self.linear.bias.clone())
        return merged

    def extra_repr(self) -> str:
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"r={self.r}, alpha={self.lora_alpha}, scale={self.scaling:.3f}")


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def print_lora_summary(model: nn.Module, model_name: str = "Model"):
    """Print a summary of LoRA parameter efficiency."""
    trainable, total = count_trainable_params(model)
    print(f"\n{model_name} LoRA Summary:")
    print(f"  Total parameters    : {total:>15,}")
    print(f"  Trainable (LoRA)    : {trainable:>15,}  ({100*trainable/total:.4f}%)")
    print(f"  Frozen (base)       : {total-trainable:>15,}  ({100*(total-trainable)/total:.4f}%)")
    print(f"  Memory for LoRA     : {trainable*2/1e6:.1f} MB (bf16)")
    print(f"  Memory for Adam     : {trainable*8/1e6:.1f} MB (fp32 m+v states)")


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration: inject LoRA into a toy model
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  LoRA DEMONSTRATION")
    print("=" * 60)

    # Simulate a small transformer attention layer
    d_model = 512
    r = 16
    alpha = 32

    # Standard linear (pretrained)
    base_linear = nn.Linear(d_model, d_model, bias=False)
    base_params = sum(p.numel() for p in base_linear.parameters())

    # LoRA-augmented linear
    lora_linear = LoRALinear(d_model, d_model, r=r, lora_alpha=alpha)
    lora_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_linear.parameters())

    print(f"\nBase linear: {d_model}×{d_model} = {base_params:,} params")
    print(f"LoRA linear: {total_params:,} total params")
    print(f"  Frozen  : {total_params - lora_params:,}  (base weight)")
    print(f"  Trainable: {lora_params:,}  (A: {d_model*r:,} + B: {r*d_model:,})")
    print(f"  Savings : {100*(1 - lora_params/base_params):.1f}% fewer trainable params")
    print(f"  Scale   : α/r = {alpha}/{r} = {alpha/r}")

    # Forward pass
    x = torch.randn(2, 32, d_model)  # (batch, seq, d_model)
    out = lora_linear(x)
    print(f"\nForward pass: {tuple(x.shape)} → {tuple(out.shape)}")

    # Verify ΔW = 0 at initialization
    delta = (lora_linear.lora_B.weight @ lora_linear.lora_A.weight) * lora_linear.scaling
    print(f"ΔW norm at init: {delta.norm().item():.6f} (should be ~0)")

    # Merge
    merged = lora_linear.merge_weights()
    out_merged = merged(x.reshape(-1, d_model)).reshape(2, 32, d_model)
    print(f"Merged output matches: {torch.allclose(out, out_merged, atol=1e-5)}")

    print("\n" + "=" * 60)
    print("For a 3.8B model (Phi-3.5-mini):")
    num_attn_layers = 32
    num_lora_targets = 7  # q,k,v,o,gate,up,down projections
    d_phi = 3072
    total_lora = num_attn_layers * num_lora_targets * 2 * d_phi * r
    print(f"  LoRA targets: {num_attn_layers} layers × {num_lora_targets} projections")
    print(f"  Trainable   : {total_lora:,} params ({total_lora/1e6:.1f}M)")
    print(f"  vs Base     : 3,800,000,000 params")
    print(f"  Ratio       : {100*total_lora/3.8e9:.3f}% of base model")
    print("=" * 60)
