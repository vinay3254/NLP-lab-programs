"""
train.py
========
Training loop for the Transformer on the EN→FR translation task.

Two scheduler strategies are kept here for educational comparison:

1. NOAM SCHEDULE (original paper, great for large-scale training):
   lrate = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
   - Linear warm-up then inverse-sqrt decay
   - Works best when training for many thousands of steps on large data
   - Minimum label-smoothing loss ≈ H(smooth_dist) ≈ 0.87 — never reaches 0
   - ❌ Does NOT work well for tiny memorization tasks (145 examples)

2. ADAM + COSINE ANNEALING (used here, great for small-scale training):
   - Adam handles per-parameter adaptive learning rates automatically
   - CosineAnnealingLR smoothly decays LR from lr_max → lr_min over T epochs
   - Standard CrossEntropyLoss targets → loss can reach 0 → confident predictions
   - ✅ Works well for tiny datasets; model CAN overfit and produce real translations

Why does label smoothing hurt here?
   The entropy of the smooth label distribution is H ≈ 0.87 bits/token.
   This is the minimum achievable loss — the optimizer never drives the model
   to be *confident* (P(correct) → 1), so greedy decoding stays unreliable.
   For real large-scale training (millions of examples), label smoothing helps
   generalization. For 145 examples, we deliberately want to overfit.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import build_dataset_and_vocabs, EN_FR_PAIRS
from .transformer import Transformer


# ── Noam LR Scheduler (kept for educational reference) ───────────────────────

class NoamLRScheduler:
    """
    Learning rate scheduler from "Attention is All You Need".

    lrate = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))

    Great for large-scale training; not ideal for tiny datasets.
    Provided here purely as a reference implementation.
    """

    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _compute_lr(self) -> float:
        step = self.step_num
        warmup = self.warmup_steps
        return (self.d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


# ── Label Smoothing Loss (kept for educational reference) ─────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing (ε=0.1, from original paper).

    Instead of hard targets (one-hot), we use:
        P_smooth(y) = (1-ε) if y == correct else ε / (V - 1)

    Prevents overconfidence on large datasets. For tiny datasets,
    the entropy floor (≈ 0.87/token) prevents convergence to confident
    predictions — use standard CrossEntropyLoss instead.

    Kept here as a reference; NOT used in the default training loop below.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth_val = self.smoothing / (self.vocab_size - 2)
        with torch.no_grad():
            soft_targets = torch.full_like(logits, smooth_val)
            soft_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
            soft_targets[:, self.pad_idx] = 0.0
        pad_mask = (targets != self.pad_idx).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1)
        return (loss * pad_mask).sum() / pad_mask.sum().clamp(min=1)


# ── Greedy Decode (used for qualitative monitoring during training) ────────────

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_vocab,
    tgt_vocab,
    max_len: int = 30,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Greedy left-to-right decoding for a single source sentence.
    At each step, picks the single most-probable next token.
    Stops when <EOS> is generated (after at least one real token).
    """
    model.eval()
    if src.dim() == 1:
        src = src.unsqueeze(0)
    src = src.to(device)

    src_mask = model.make_src_mask(src)
    encoder_output = model.encode(src, src_mask)

    bos_idx = tgt_vocab.word2idx["<BOS>"]
    eos_idx = tgt_vocab.word2idx["<EOS>"]
    pad_idx = 0

    tgt_ids = [bos_idx]
    for step in range(max_len):
        tgt = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt)
        decoder_out = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = model.output_projection(decoder_out)  # (1, step+1, vocab)

        # Suppress <PAD> and <BOS> — they should never be generated
        logits[0, -1, pad_idx] = float("-inf")
        logits[0, -1, bos_idx] = float("-inf")

        next_token = logits[0, -1, :].argmax().item()
        tgt_ids.append(next_token)

        # Only end if we've generated at least 1 real token before EOS
        if next_token == eos_idx and step >= 1:
            break

    # Decode, skip the leading BOS (tgt_ids[0]) and any trailing EOS
    return tgt_vocab.decode(tgt_ids[1:])


# ── Main Training Function ────────────────────────────────────────────────────

def train(
    num_epochs: int = 1000,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 512,
    batch_size: int = 16,
    lr: float = 5e-4,
    dropout: float = 0.0,
    grad_clip: float = 1.0,
    early_stop_loss: float = 0.15,
    save_path: str = None,
    device: str = "auto",
):
    """
    Train the Transformer on the tiny EN→FR dataset.

    Optimizer Strategy:
        Adam (adaptive per-parameter LR) with CosineAnnealingLR.
        Cosine decay smoothly reduces LR from `lr` to `lr/100` over all epochs.
        This is the standard recipe for small-scale memorization tasks.

    Loss:
        Standard CrossEntropyLoss (ignore_index=0 for PAD).
        Loss CAN reach 0 → model CAN be confident → greedy decode works.

    Args:
        num_epochs:      Number of training epochs
        d_model:         Embedding dimension
        num_heads:       Number of attention heads
        num_layers:      Number of encoder/decoder layers
        d_ff:            Feed-forward inner dimension
        batch_size:      Batch size
        lr:              Peak learning rate for Adam
        dropout:         Dropout (0.0 = off; overfit is OK for 145 examples)
        grad_clip:       Max gradient norm
        early_stop_loss: Stop early when avg loss drops below this value
        save_path:       Checkpoint path
        device:          "auto", "cpu", or "cuda"
    """
    # ── Device ──
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # ── Data ──
    dataloader, src_vocab, tgt_vocab = build_dataset_and_vocabs(
        pairs=EN_FR_PAIRS, batch_size=batch_size
    )
    print(f"Source vocab: {len(src_vocab)} tokens | Target vocab: {len(tgt_vocab)} tokens")
    print(f"Dataset: {len(dataloader.dataset)} sentence pairs | {len(dataloader)} batches/epoch")

    # ── Model ──
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=100,
        dropout=dropout,
        pad_idx=0,
    ).to(device)

    print(f"\nModel: {model.count_parameters():,} parameters")
    print(f"  d_model={d_model}, heads={num_heads}, layers={num_layers}, d_ff={d_ff}")

    # ── Optimizer: Adam with cosine LR decay ──────────────────────────────────
    #
    # Why Adam (not SGD)?
    #   Adam maintains per-parameter moving averages of gradients (m) and
    #   squared gradients (v), computing adaptive step sizes:
    #       θ ← θ - lr × m̂ / (√v̂ + ε)
    #   This makes it robust to sparse gradients and varying loss scales —
    #   crucial for attention weights which can be highly non-uniform.
    #
    # Why CosineAnnealingLR?
    #   LR follows a half-cosine from lr_max → lr_min:
    #       lr_t = lr_min + 0.5(lr_max - lr_min)(1 + cos(π·t/T))
    #   Avoids abrupt LR drops; allows finding sharp minima at the end.
    #
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr / 100
    )

    # ── Loss: Standard cross-entropy, ignoring PAD tokens ────────────────────
    #
    # CrossEntropyLoss(ignore_index=0):
    #   - Computes -log P(correct_token) at each non-PAD position
    #   - Loss CAN reach 0 when model assigns probability 1.0 to correct tokens
    #   - No entropy floor → optimizer CAN make model confident → greedy decode works
    #
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # ── Checkpoint path ──
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "..", "best_model.pt")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_loss = float("inf")
    history = {"train_loss": [], "lr": []}
    log_interval = max(1, num_epochs // 10)  # log ~10 times total

    print(f"\n{'='*65}")
    print(f"Training: {num_epochs} epochs | lr={lr} | Adam + CosineAnnealingLR")
    print(f"Loss: CrossEntropyLoss (no label smoothing — overfit is the goal)")
    print(f"Early stop when loss < {early_stop_loss}")
    print(f"{'='*65}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # ── Teacher forcing ──────────────────────────────────────────────
            # At training time, we always feed the GROUND TRUTH target tokens
            # as decoder input, even if previous predictions were wrong.
            # This stabilizes early training by providing a clean learning signal.
            #
            # tgt_input  = [<BOS>, word1, word2, ..., wordN]   (decoder input)
            # tgt_target = [word1, word2, ..., wordN, <EOS>]   (what to predict)
            #
            # Causal mask ensures position i can only attend to positions 0..i,
            # simulating the autoregressive (one-token-at-a-time) inference mode.
            tgt_input  = tgt_batch[:, :-1]  # (batch, tgt_seq - 1)
            tgt_target = tgt_batch[:, 1:]   # (batch, tgt_seq - 1)

            logits = model(src_batch, tgt_input)  # (batch, tgt_seq-1, vocab)

            # Flatten for CrossEntropyLoss: (batch × seq, vocab) and (batch × seq,)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_target.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: caps the gradient norm to prevent large
            # parameter updates that destabilize attention weights early on
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()  # decay LR after each epoch (not each step)

        avg_loss = epoch_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - epoch_start
        history["train_loss"].append(avg_loss)
        history["lr"].append(current_lr)

        # ── Checkpoint ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "config": {
                        "d_model": d_model,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                        "d_ff": d_ff,
                    },
                },
                save_path,
            )

        # ── Logging ──
        if epoch % log_interval == 0 or epoch == 1:
            sample_en, sample_fr = EN_FR_PAIRS[epoch % len(EN_FR_PAIRS)]
            src_ids = torch.tensor(src_vocab.encode(sample_en), dtype=torch.long)
            pred_fr = greedy_decode(model, src_ids, src_vocab, tgt_vocab, device=device)
            model.train()  # restore train mode after eval in greedy_decode

            print(f"Epoch {epoch:4d}/{num_epochs} | Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.2e} | {elapsed:.1f}s")
            print(f"  EN: '{sample_en}'  GT: '{sample_fr}'  PR: '{pred_fr}'")
            print()

        # ── Early stopping ──
        if avg_loss < early_stop_loss:
            print(f"\nEarly stop at epoch {epoch} — loss {avg_loss:.4f} < {early_stop_loss}")
            break

    print(f"{'='*65}")
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {save_path}")
    print(f"{'='*65}")

    return model, src_vocab, tgt_vocab, history


if __name__ == "__main__":
    model, src_vocab, tgt_vocab, history = train()
