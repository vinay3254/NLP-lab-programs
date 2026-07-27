"""
scale_6b.py
===========
6B parameter analysis: instantiate, count, profile memory, benchmark GPU vs CPU.

What this script does:
  1. Shows GPU info and confirms CUDA is available
  2. Instantiates GPT-Standard and Llama-style models at multiple scales
     using torch.device('meta') — zero RAM, just traces tensor shapes
  3. Shows full parameter breakdown by component
  4. Computes exact memory requirements at float32 / bfloat16 / int8 / int4
  5. Runs a REAL forward pass of our EN->FR model on GPU to benchmark speedup
  6. Runs a REAL forward pass of a medium GPT on GPU to show it works

Run: python scale_6b.py
"""

import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, ".")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_params(n: int) -> str:
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.1f}M"
    if n >= 1e3:  return f"{n/1e3:.0f}K"
    return str(n)

def fmt_mem(bytes_: int) -> str:
    if bytes_ >= 1e9:  return f"{bytes_/1e9:.2f} GB"
    if bytes_ >= 1e6:  return f"{bytes_/1e6:.1f} MB"
    return f"{bytes_/1e3:.0f} KB"

def param_memory(n_params: int, dtype_bytes: int) -> str:
    return fmt_mem(n_params * dtype_bytes)

def count_params_meta(model_cls, **kwargs) -> int:
    """Instantiate on meta device (no RAM) and count parameters."""
    with torch.device("meta"):
        m = model_cls(**kwargs)
    return sum(p.numel() for p in m.parameters())

def breakdown_by_component(model: nn.Module):
    """Print parameter count per named top-level module."""
    rows = []
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        rows.append((name, type(module).__name__, n))
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  {'Component':<30} {'Type':<25} {'Params':>12} {'Share':>7}")
    print(f"  {'-'*30} {'-'*25} {'-'*12} {'-'*7}")
    for name, typ, n in rows:
        print(f"  {name:<30} {typ:<25} {fmt_params(n):>12} {100*n/total:>6.1f}%")
    print(f"  {'TOTAL':<30} {'':<25} {fmt_params(total):>12} {'100.0%':>7}")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 1. GPU Info
# ─────────────────────────────────────────────────────────────────────────────

def print_gpu_info():
    print("\n" + "="*65)
    print("  GPU / CUDA INFO")
    print("="*65)
    if not torch.cuda.is_available():
        print("  CUDA not available. Run on CPU only.")
        return False

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        vram = p.total_memory / 1024**3
        print(f"  GPU {i}: {p.name}")
        print(f"    VRAM       : {vram:.1f} GB ({p.total_memory/1e6:.0f} MB)")
        print(f"    CUDA cap   : {p.major}.{p.minor}")
        print(f"    SM count   : {p.multi_processor_count}")
        print(f"    torch build: {torch.version.cuda}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parameter Scaling Table (meta device — zero RAM)
# ─────────────────────────────────────────────────────────────────────────────

def print_scaling_table():
    from src.gpt_model import GPTModel, LlamaModel

    configs = [
        # name,                  vocab,  d,    h,   N,   d_ff,  model_cls
        ("Ours (EN->FR, tiny)",  241,    128,  4,   2,   512,   GPTModel),
        ("GPT-2 Small",          50257,  768,  12,  12,  3072,  GPTModel),
        ("GPT-2 XL",             50257,  1600, 25,  48,  6400,  GPTModel),
        ("GPT-J 6B (standard)",  50400,  4096, 16,  28,  16384, GPTModel),
        ("Llama-7B (standard)",  32000,  4096, 32,  32,  16384, GPTModel),
        ("Llama-7B (modern)",    32000,  4096, 32,  32,  16384, LlamaModel),
        ("Llama-13B",            32000,  5120, 40,  40,  13824, LlamaModel),
        ("Llama-70B",            32000,  8192, 64,  80,  28672, LlamaModel),
    ]

    print("\n" + "="*105)
    print("  PARAMETER SCALING TABLE (meta device — no RAM used)")
    print("="*105)
    print(f"  {'Config':<26} {'d':>6} {'h':>4} {'N':>4} {'Params':>10} {'fp32':>10} {'bf16':>10} {'int8':>10} {'int4':>8}")
    print(f"  {'-'*26} {'-'*6} {'-'*4} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for name, vocab, d, h, n_layers, dff, cls in configs:
        try:
            total = count_params_meta(cls, vocab_size=vocab, d_model=d,
                                      num_heads=h, num_layers=n_layers, d_ff=dff)
            fp32 = param_memory(total, 4)
            bf16 = param_memory(total, 2)
            int8 = param_memory(total, 1)
            int4 = param_memory(total, 1) + " (est)"  # roughly half
            print(f"  {name:<26} {d:>6} {h:>4} {n_layers:>4} {fmt_params(total):>10} {fp32:>10} {bf16:>10} {int8:>10} {int4:>8}")
        except Exception as e:
            print(f"  {name:<26} ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training Memory Estimate
# ─────────────────────────────────────────────────────────────────────────────

def print_training_memory():
    configs = [
        ("Our tiny model",   982_000),
        ("GPT-2 Small",      117_000_000),
        ("6B (Llama-7B)",    6_738_415_616),
        ("13B",              13_000_000_000),
        ("70B",              70_000_000_000),
    ]
    print("\n" + "="*75)
    print("  TRAINING MEMORY REQUIREMENTS (Adam optimizer, bf16 mixed precision)")
    print("="*75)
    print(f"  {'Model':<22} {'Params':>10} {'Weights':>10} {'Adam states':>12} {'Total':>10} {'GPUs needed':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")
    # Adam: 2 fp32 copies (m + v) + fp16 weights + fp16 grads = 4+4+2+2 = 12 bytes/param
    for name, p in configs:
        weights_gb = p * 2 / 1e9          # bf16 weights
        adam_gb    = p * 8 / 1e9          # fp32 m + v states
        grads_gb   = p * 2 / 1e9          # bf16 gradients
        total_gb   = weights_gb + adam_gb + grads_gb
        gpus_80gb  = max(1, int(total_gb / 70))   # 80GB A100, ~70GB usable
        print(f"  {name:<22} {fmt_params(p):>10} {weights_gb:>9.1f}G {adam_gb:>11.1f}G {total_gb:>9.1f}G {gpus_80gb:>8}x A100")

    print("\n  Training cost estimate for 6B (Chinchilla optimal: 120B tokens):")
    flops_per_token = 6 * 6.74e9           # 6 × params per forward+backward
    total_flops = flops_per_token * 120e9  # 120B tokens
    a100_tflops = 312e12                   # A100 bf16 peak
    gpu_seconds = total_flops / a100_tflops
    print(f"    FLOPs needed : {total_flops:.2e}")
    print(f"    1× A100 time : {gpu_seconds/3600:.0f} hours  ({gpu_seconds/3600/24:.0f} days)")
    print(f"    8× A100 time : {gpu_seconds/3600/8:.0f} hours  ({gpu_seconds/3600/8/24:.1f} days)")
    print(f"    64× A100 time: {gpu_seconds/3600/64:.0f} hours  (~${gpu_seconds/3600/64 * 2:.0f} on cloud at $2/GPU-hr)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Real GPU benchmark: our EN->FR model CPU vs GPU
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_cpu_vs_gpu():
    import os
    from src.transformer import Transformer

    print("\n" + "="*65)
    print("  BENCHMARK: Our EN->FR Transformer — CPU vs GPU")
    print("="*65)

    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cpu = Transformer(218, 241, d_model=128, num_heads=4, num_layers=2, d_ff=512, pad_idx=0)
    model_gpu = Transformer(218, 241, d_model=128, num_heads=4, num_layers=2, d_ff=512, pad_idx=0).to(device_gpu)

    # Copy weights to make it fair
    model_gpu.load_state_dict(model_cpu.state_dict())
    model_cpu.eval(); model_gpu.eval()

    batch_sizes = [1, 8, 32, 128]
    seq_len = 20

    print(f"\n  Seq len={seq_len} | 100 forward passes per measurement")
    print(f"  {'Batch':>6} {'CPU (ms/batch)':>16} {'GPU (ms/batch)':>16} {'Speedup':>10}")
    print(f"  {'-'*6} {'-'*16} {'-'*16} {'-'*10}")

    for bs in batch_sizes:
        src = torch.randint(1, 218, (bs, seq_len))
        tgt = torch.randint(1, 241, (bs, seq_len))

        # CPU warmup + measure
        with torch.no_grad():
            for _ in range(5): model_cpu(src, tgt)
            t0 = time.perf_counter()
            for _ in range(100): model_cpu(src, tgt)
            cpu_ms = (time.perf_counter() - t0) * 10  # ms per batch

        # GPU warmup + measure
        src_g = src.to(device_gpu); tgt_g = tgt.to(device_gpu)
        with torch.no_grad():
            for _ in range(5): model_gpu(src_g, tgt_g)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100): model_gpu(src_g, tgt_g)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) * 10

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        gpu_label = "GPU" if torch.cuda.is_available() else "CPU(same)"
        print(f"  {bs:>6} {cpu_ms:>14.2f}ms {gpu_ms:>14.2f}ms {speedup:>8.1f}x")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Real forward pass of a medium GPT on GPU
# ─────────────────────────────────────────────────────────────────────────────

def run_medium_gpt_on_gpu():
    from src.gpt_model import GPTModel, LlamaModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*65)
    print(f"  REAL FORWARD PASS: Medium GPT on {str(device).upper()}")
    print("="*65)

    # Medium config: ~120M params — fits in 6GB VRAM easily
    cfg = dict(vocab_size=32000, d_model=1024, num_heads=16, num_layers=12, d_ff=4096, dropout=0.0)
    print(f"\n  Config: d={cfg['d_model']}, heads={cfg['num_heads']}, layers={cfg['num_layers']}")

    print("  Instantiating GPT (standard)...", end=" ", flush=True)
    t0 = time.perf_counter()
    model = GPTModel(**cfg).to(device)
    print(f"{fmt_params(model.count_parameters())} params | {time.perf_counter()-t0:.2f}s")

    print("  Instantiating Llama (modern)... ", end=" ", flush=True)
    t0 = time.perf_counter()
    llama = LlamaModel(**cfg).to(device)
    print(f"{fmt_params(llama.count_parameters())} params | {time.perf_counter()-t0:.2f}s")

    # Forward pass
    input_ids = torch.randint(0, 32000, (4, 512), device=device)  # batch=4, seq=512
    print(f"\n  Forward pass: batch=4, seq=512")

    model.eval(); llama.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(input_ids)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        gpt_ms = (time.perf_counter() - t0) * 1000
        print(f"  GPT   forward: {gpt_ms:.1f}ms | output: {tuple(logits.shape)}")

        t0 = time.perf_counter()
        logits = llama(input_ids)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        llama_ms = (time.perf_counter() - t0) * 1000
        print(f"  Llama forward: {llama_ms:.1f}ms | output: {tuple(logits.shape)}")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n  VRAM used: {used:.2f} GB | reserved: {reserved:.2f} GB")
        remaining = torch.cuda.get_device_properties(0).total_memory / 1e9 - reserved
        print(f"  VRAM free: {remaining:.2f} GB of {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 6B instantiation via meta device
# ─────────────────────────────────────────────────────────────────────────────

def instantiate_6b_meta():
    from src.gpt_model import LlamaModel

    print("\n" + "="*65)
    print("  6B MODEL — INSTANTIATION VIA meta DEVICE (zero RAM)")
    print("="*65)
    print("  Config: vocab=32000, d=4096, heads=32, layers=32, d_ff=16384")
    print("  This traces all tensor shapes without allocating any memory.\n")

    with torch.device("meta"):
        model = LlamaModel(
            vocab_size=32000, d_model=4096, num_heads=32,
            num_layers=32, d_ff=16384, dropout=0.0
        )

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {fmt_params(total)}  ({total:,})")
    print(f"  Memory (float32) : {param_memory(total, 4)}")
    print(f"  Memory (bfloat16): {param_memory(total, 2)}")
    print(f"  Memory (int8)    : {param_memory(total, 1)}")
    print(f"  Memory (int4)    : ~{param_memory(total//2, 1)} (quantized)")

    print(f"\n  Parameter breakdown:")
    breakdown_by_component(model)

    print(f"\n  RTX 4050 Laptop (6GB) can:")
    vram_gb = 6.1
    model_bf16 = total * 2 / 1e9
    print(f"    Fit in VRAM?   {'YES' if model_bf16 < vram_gb else 'NO'} "
          f"(model={model_bf16:.1f}GB, VRAM={vram_gb}GB)")
    print(f"    Max batch for inference (seq=512, bf16): ~1 (barely)")
    print(f"    Training: NOT feasible (needs ~80GB for Adam states)")
    print(f"\n  What FITS in 6GB VRAM:")
    for name, p in [("Our tiny model", 982_000), ("GPT-2 Small 117M", 117e6),
                    ("GPT-2 Medium 345M", 345e6), ("GPT-2 Large 762M", 762e6),
                    ("GPT-2 XL 1.5B", 1.5e9), ("~2B model", 2e9)]:
        mem = p * 2 / 1e9
        fits = "YES" if mem < 5.0 else ("maybe (tight)" if mem < 6.1 else "NO")
        print(f"    {name:<28}: {mem:.2f} GB — {fits}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    has_gpu = print_gpu_info()
    print_scaling_table()
    print_training_memory()
    instantiate_6b_meta()
    benchmark_cpu_vs_gpu()
    run_medium_gpt_on_gpu()

    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    if has_gpu:
        print("  GPU detected — EN->FR model now trains on CUDA.")
        print("  Run: python run_training.py  (will auto-use GPU)")
        print()
        print("  For 6B: your RTX 4050 (6GB) can run inference on")
        print("  quantized 4-bit models (e.g., llama.cpp, GGUF format).")
        print("  Training 6B needs multi-GPU cloud (A100s).")
    print()
