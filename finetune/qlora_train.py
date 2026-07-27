"""
qlora_train.py
==============
QLoRA fine-tuning of Phi-3.5-mini-instruct on your custom dataset.

────────────────────────────────────────────────────────────────────
WHAT IS QLORA?
────────────────────────────────────────────────────────────────────
QLoRA = Quantized + Low-Rank Adaptation  (Dettmers et al. 2023)
https://arxiv.org/abs/2305.14314

It combines three techniques to fit a 4B fine-tuning job in 6 GB VRAM:

1. NF4 (Normal Float 4) Quantization
   ─────────────────────────────────
   Standard INT4 quantization divides the value range into 16 equal buckets.
   But transformer weights follow a roughly normal (Gaussian) distribution —
   most weights cluster near 0, very few are large.

   NF4 places quantization levels at the QUANTILES of a normal distribution:
       bucket boundaries at: N(0,1).ppf([1/16, 2/16, ..., 15/16])
   This means equal numbers of weights fall into each bucket → minimal error.
   NF4 achieves the information-theoretic optimum for normally distributed data.

   Result: 16-bit weights → 4-bit storage. 4× compression, <1% quality loss.

2. Double Quantization
   ────────────────────
   Each NF4 quantization block needs a "scale" constant (fp32, 32 bits).
   With block_size=64, that's 32/64 = 0.5 extra bits per weight.
   Double quantization quantizes THESE constants too → saves 0.37 bits/param.
   Combined: effective storage ≈ 4.37 bits/param vs 16 bits → 3.7× savings.

3. Paged Adam Optimizer
   ─────────────────────
   Adam needs 2 fp32 tensors per trainable parameter (momentum m, variance v).
   For 3.8B params, that's 3.8B × 8 bytes = 30 GB — impossible on 6 GB GPU.
   Solution: only LoRA parameters are trainable (~3M params → 24 MB).
   "Paged" = bitsandbytes uses unified memory: if VRAM fills up, optimizer
   states spill to CPU RAM automatically (like virtual memory for GPU).

Memory breakdown for Phi-3.5-mini (3.8B) with QLoRA, r=16:
    Base model (NF4, 4-bit):          ~2.1 GB
    LoRA adapter weights (bf16):       ~0.1 GB
    LoRA optimizer states (Adam fp32): ~0.3 GB
    Activations (batch=2, seq=512):    ~1.5 GB
    KV cache + overhead:               ~0.5 GB
    ────────────────────────────────────────────
    TOTAL:                             ~4.5 GB  ✅ fits in RTX 4050 (6 GB)

────────────────────────────────────────────────────────────────────
WHY THESE TARGET MODULES?
────────────────────────────────────────────────────────────────────
We apply LoRA to: q_proj, k_proj, v_proj, o_proj  (attention)
              and: gate_proj, up_proj, down_proj   (SwiGLU FFN)

Why attention projections?
  Q, K, V, O are where the model's "understanding" of relationships lives.
  Fine-tuning these teaches the model what to attend to for your task.

Why FFN projections?
  The SwiGLU FFN stores factual knowledge and transformation patterns.
  Adapting gate/up/down lets the model produce different outputs for
  the same attended context — critical for instruction following.

Why NOT embedding layers?
  The vocabulary embeddings are tied between input and output.
  Fine-tuning them is expensive and rarely improves task performance.
"""

import argparse
import os
import sys

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (only load if actually running)
# ─────────────────────────────────────────────────────────────────────────────

def _imports():
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from trl import SFTTrainer, SFTConfig
    return (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            TrainingArguments, LoraConfig, TaskType, get_peft_model,
            prepare_model_for_kbit_training, SFTTrainer, SFTConfig)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Quantization config
# ─────────────────────────────────────────────────────────────────────────────

def make_bnb_config():
    """
    BitsAndBytes NF4 quantization configuration.

    load_in_4bit:             Load all Linear layers in 4-bit NF4
    bnb_4bit_quant_type:      'nf4' = Normal Float 4 (optimal for Gaussian weights)
    bnb_4bit_use_double_quant: Quantize the quantization scales too (saves 0.37 bits/param)
    bnb_4bit_compute_dtype:   Upcast to bfloat16 for forward pass math
                              (4-bit storage, bf16 compute — best of both worlds)
    """
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load base model in 4-bit
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(model_name: str = "microsoft/Phi-3.5-mini-instruct"):
    """
    Load Phi-3.5-mini-instruct quantized to 4-bit NF4.

    After loading, the model occupies ~2.1 GB VRAM instead of 7.6 GB (fp32)
    or 3.8 GB (bf16). The weights are FROZEN — we only train LoRA adapters.

    prepare_model_for_kbit_training:
        1. Casts LayerNorm layers back to fp32 for stability
           (LayerNorm in 4-bit causes numerical instability)
        2. Enables gradient checkpointing (recompute activations on backward
           pass instead of caching them → trades compute for VRAM)
        3. Wraps the model for mixed 4-bit / bf16 computation
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import prepare_model_for_kbit_training

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"  # Important: pad right for causal LM

    print(f"Loading model in 4-bit NF4...")
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1e9
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=make_bnb_config(),
        device_map="auto",          # automatically place layers on GPU/CPU
        trust_remote_code=True,
        attn_implementation="eager", # disable flash attention for compatibility
    )

    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM used by model: {vram_after - vram_before:.2f} GB")
        print(f"VRAM total used:    {vram_after:.2f} GB of "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Prepare for k-bit training (gradient checkpointing + LayerNorm in fp32)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.config.use_cache = False  # Disable KV cache during training

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 3. LoRA configuration
# ─────────────────────────────────────────────────────────────────────────────

def create_lora_config(r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    Configure LoRA adapter injection.

    Args:
        r:            Rank of the low-rank decomposition.
                      r=4:  Very parameter-efficient, good for simple style changes
                      r=16: Standard — good balance for instruction tuning
                      r=64: High expressivity, risk of overfitting on small datasets
                      r=128: For large datasets or very domain-specific tasks

        lora_alpha:   Scaling factor α. Effective scale = α/r.
                      Standard: set alpha = 2*r (gives 2× scaling).
                      With r=16, alpha=32 → scale=2.0

        lora_dropout: Dropout on the LoRA path.
                      0.05 = 5% dropout, prevents overfitting on small datasets.
                      Set to 0.0 for large datasets (>10K examples).

        target_modules: Which weight matrices to inject LoRA into.
            q_proj, k_proj, v_proj, o_proj = attention projections
            gate_proj, up_proj, down_proj  = SwiGLU FFN projections
            All 7 targets × 32 layers × 2 matrices (A, B) each =
            7 × 32 × 2 × 3072 × 16 = ~21.5M trainable parameters

        bias="none": Don't train bias terms — they're tiny and rarely help.

        task_type=CAUSAL_LM: Phi-3.5 is a causal language model
                             (predicts next token, autoregressive).
    """
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
            "gate_proj", "up_proj", "down_proj",        # SwiGLU FFN
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    dataset_path: str,
    output_dir: str = "./phi35_finetuned",
    model_name: str = "microsoft/Phi-3.5-mini-instruct",
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_seq_len: int = 512,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    save_steps: int = 50,
):
    """
    Full QLoRA fine-tuning pipeline for Phi-3.5-mini-instruct.

    Effective batch size = batch_size × gradient_accumulation_steps
                         = 2 × 8 = 16  (standard for instruction tuning)

    Why gradient accumulation?
        GPU VRAM limits us to batch_size=2 per step. But small batches have
        high gradient variance → noisy training. Accumulation simulates a
        larger batch by summing gradients over N steps before updating weights.
        Mathematically equivalent to batch_size × N = 16.

    Learning rate 2e-4:
        LoRA adapters start from zero (ΔW=0) so they need a higher LR than
        full fine-tuning (typically 1e-5 to 5e-5). 2e-4 is standard for LoRA.
        The cosine schedule decays this to ~0 by the end of training.

    Warmup ratio 0.03:
        First 3% of steps linearly ramp LR from 0 → 2e-4. Prevents large
        updates early when the LoRA matrices are random and unstable.
    """
    from peft import get_peft_model
    from trl import SFTConfig, SFTTrainer
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from finetune.dataset_prep import load_dataset_from_json, prepare_dataset, split_dataset

    print("\n" + "="*65)
    print("  QLoRA FINE-TUNING: Phi-3.5-mini-instruct")
    print("="*65)

    # ── Step 1: Load model in 4-bit ──────────────────────────────────────────
    model, tokenizer = load_base_model(model_name)

    # ── Step 2: Inject LoRA adapters ─────────────────────────────────────────
    lora_config = create_lora_config(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    model = get_peft_model(model, lora_config)

    # Print trainable parameter summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA Adapter Summary:")
    print(f"  Trainable params : {trainable:>12,}  ({100*trainable/total:.4f}%)")
    print(f"  Frozen params    : {total-trainable:>12,}  ({100*(total-trainable)/total:.4f}%)")
    print(f"  Total params     : {total:>12,}")
    print(f"  LoRA memory      : {trainable*2/1e6:.1f} MB (bf16 adapters)")
    print(f"  Adam memory      : {trainable*8/1e6:.1f} MB (fp32 m+v states)")

    # ── Step 3: Prepare dataset ───────────────────────────────────────────────
    print(f"\nLoading dataset from: {dataset_path}")
    raw_data = load_dataset_from_json(dataset_path)
    dataset = prepare_dataset(raw_data, tokenizer=tokenizer, max_seq_len=max_seq_len)
    train_ds, val_ds = split_dataset(dataset, val_fraction=0.1)

    # ── Step 4: Training configuration ───────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # ── Optimizer ──
        optim="paged_adamw_32bit",   # Paged Adam: spills to CPU RAM if VRAM is full
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",  # Smooth decay from lr → ~0
        warmup_ratio=warmup_ratio,
        # ── Precision ──
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=False,                  # bf16 is better on Ampere/Ada (RTX 30/40xx)
        # ── Sequence ──
        max_length=max_seq_len,
        dataset_text_field="text",   # Which column in the dataset to use
        # ── Logging ──
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_steps=save_steps,
        save_total_limit=2,          # Keep only the 2 best checkpoints
        load_best_model_at_end=True,
        report_to="none",            # Disable wandb/tensorboard
        # ── Memory ──
        gradient_checkpointing=True, # Recompute activations on backward (saves VRAM)
        dataloader_pin_memory=False,
    )

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print(f"\n{'='*65}")
    print(f"  Starting training...")
    print(f"  Epochs : {num_epochs}")
    print(f"  Batch  : {batch_size} × {gradient_accumulation_steps} steps = {batch_size*gradient_accumulation_steps} effective")
    print(f"  LR     : {learning_rate} (cosine decay, {warmup_ratio*100:.0f}% warmup)")
    print(f"  LoRA r : {lora_r}, alpha={lora_alpha}")
    print(f"{'='*65}\n")

    trainer.train()

    # ── Step 6: Save LoRA adapters ────────────────────────────────────────────
    # We save ONLY the LoRA adapter weights — not the 4-bit base model.
    # The adapter is tiny (~50 MB for r=16) and merges with the base at inference.
    adapter_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapters saved to: {adapter_path}")
    print(f"Adapter size: ~{sum(os.path.getsize(os.path.join(adapter_path, f)) for f in os.listdir(adapter_path) if os.path.isfile(os.path.join(adapter_path, f)))/1e6:.1f} MB")

    # ── Step 7: Final VRAM report ─────────────────────────────────────────────
    if torch.cuda.is_available():
        used = torch.cuda.max_memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nPeak VRAM used during training: {used:.2f} GB / {total_vram:.1f} GB")

    print(f"\n{'='*65}")
    print(f"  Training complete!")
    print(f"  Run inference: python finetune/inference_ft.py --adapter_path {adapter_path}")
    print(f"{'='*65}\n")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Phi-3.5-mini-instruct")
    parser.add_argument("--dataset",      type=str,   required=True,              help="Path to JSON dataset file")
    parser.add_argument("--output",       type=str,   default="./phi35_finetuned",help="Output directory for checkpoints")
    parser.add_argument("--model",        type=str,   default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--epochs",       type=int,   default=3,                  help="Number of training epochs")
    parser.add_argument("--batch_size",   type=int,   default=2,                  help="Per-device batch size")
    parser.add_argument("--grad_accum",   type=int,   default=8,                  help="Gradient accumulation steps")
    parser.add_argument("--max_seq_len",  type=int,   default=512,                help="Maximum sequence length")
    parser.add_argument("--lr",           type=float, default=2e-4,               help="Learning rate")
    parser.add_argument("--lora_r",       type=int,   default=16,                 help="LoRA rank")
    parser.add_argument("--lora_alpha",   type=int,   default=32,                 help="LoRA alpha scaling")
    parser.add_argument("--save_steps",   type=int,   default=50,                 help="Save checkpoint every N steps")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        save_steps=args.save_steps,
    )
