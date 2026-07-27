"""
inference_ft.py
===============
Compare base Phi-3.5-mini-instruct vs your QLoRA fine-tuned model side-by-side.

Usage:
    # Side-by-side comparison on test prompts
    python finetune/inference_ft.py --adapter_path ./phi35_finetuned/final_adapter

    # Interactive chat with your fine-tuned model
    python finetune/inference_ft.py --adapter_path ./phi35_finetuned/final_adapter --interactive
"""

import argparse
import os
import sys
import torch

# Reconfigure stdout/stderr to UTF-8 to prevent UnicodeEncodeError on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from finetune.dataset_prep import format_phi35_prompt

DEFAULT_TEST_PROMPTS = [
    "What is the difference between a list and a tuple in Python?",
    "How do I reverse a string in Python without using slicing?",
    "Explain what a decorator does in Python with a simple example.",
    "What is the time complexity of binary search and why?",
    "How does Python's garbage collector work?",
]


def load_model(model_name: str, adapter_path: str = None):
    """
    Load Phi-3.5-mini in 4-bit. Optionally merge LoRA adapters.

    When adapter_path is provided, we load the PeftModel on top of the
    base — the LoRA deltas (B×A) are added to every target weight.
    This happens at bf16 precision so the adapter is exact.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path else base_name,
        trust_remote_code=False
    )
    tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="eager",
    )

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapters from: {adapter_path}")
        vram_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        model = PeftModel.from_pretrained(model, adapter_path)
        vram_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"Adapter VRAM overhead: {vram_after - vram_before:.2f} GB")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    system: str = "You are Mimir, an expert Python programming assistant. Provide clear, accurate, and educational answers.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response for the given instruction prompt.

    Sampling strategy:
        temperature: Divide logits by T before softmax.
                     T<1 = sharper (more focused), T>1 = flatter (more random)
        top_k:       Keep only top-k highest probability tokens at each step.
                     Removes the long tail of nonsense tokens.
        top_p:       Nucleus sampling: keep smallest set of tokens whose
                     cumulative probability >= p. More adaptive than top_k.

    We use both top_k AND top_p together — whichever is more restrictive wins.
    """
    formatted = format_phi35_prompt(
        instruction=prompt,
        output="",
        system=system,
        training=False,         # Don't include output — let model generate it
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
    )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def compare_base_vs_finetuned(adapter_path: str, test_prompts: list[str]):
    """Load base and fine-tuned model, compare responses on test prompts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Load base model ──────────────────────────────────────────────────────
    print("\n--- Loading BASE model (no adapters) ---")
    base_model, tokenizer = load_model(model_name="microsoft/Phi-3.5-mini-instruct")

    print("\n" + "="*70)
    print("  BASE MODEL RESPONSES")
    print("="*70)
    base_responses = []
    for prompt in test_prompts:
        response = generate(base_model, tokenizer, prompt)
        base_responses.append(response)
        print(f"\n  Q: {prompt}")
        print(f"  A: {response}")

    # ── Free base model, load fine-tuned ────────────────────────────────────
    del base_model
    torch.cuda.empty_cache()

    print(f"\n--- Loading FINE-TUNED model (with LoRA adapters) ---")
    ft_model, tokenizer = load_model(
        model_name="microsoft/Phi-3.5-mini-instruct",
        adapter_path=adapter_path,
    )

    print("\n" + "="*70)
    print("  FINE-TUNED MODEL RESPONSES")
    print("="*70)
    ft_responses = []
    for prompt in test_prompts:
        response = generate(ft_model, tokenizer, prompt)
        ft_responses.append(response)
        print(f"\n  Q: {prompt}")
        print(f"  A: {response}")

    # ── Side-by-side diff ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SIDE-BY-SIDE COMPARISON: BASE vs MIMIR")
    print("="*70)
    for i, (prompt, base_r, ft_r) in enumerate(zip(test_prompts, base_responses, ft_responses)):
        print(f"\n[{i+1}] {prompt}")
        print(f"  BASE : {base_r[:200]}{'...' if len(base_r)>200 else ''}")
        print(f"  MIMIR: {ft_r[:200]}{'...' if len(ft_r)>200 else ''}")

    return ft_model, tokenizer


def interactive_chat(model, tokenizer, system: str = None):
    """Interactive REPL for chatting with the fine-tuned model."""
    system = system or "You are Mimir, an expert Python programming assistant. Provide clear, accurate, and educational answers."
    print("\n" + "="*70)
    print("  INTERACTIVE CHAT WITH MIMIR (Phi-3.5-mini-instruct)")
    print("  Type 'quit' or 'exit' to stop")
    print("="*70)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not user_input:
            continue

        response = generate(model, tokenizer, user_input, system=system)
        print(f"\n  Mimir: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned Phi-3.5-mini")
    parser.add_argument("--adapter_path",  type=str, required=True,
                        help="Path to saved LoRA adapter directory")
    parser.add_argument("--interactive",   action="store_true",
                        help="Launch interactive chat after comparison")
    parser.add_argument("--system",        type=str, default=None,
                        help="System prompt for the fine-tuned model")
    parser.add_argument("--max_new_tokens",type=int, default=256,
                        help="Maximum tokens to generate per response")
    parser.add_argument("--temperature",   type=float, default=0.7)
    args = parser.parse_args()

    test_prompts = DEFAULT_TEST_PROMPTS

    ft_model, tokenizer = compare_base_vs_finetuned(args.adapter_path, test_prompts)

    if args.interactive:
        interactive_chat(ft_model, tokenizer, system=args.system)
