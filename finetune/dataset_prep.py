"""
dataset_prep.py
===============
Prepares instruction-following datasets for QLoRA fine-tuning of Phi-3.5-mini-instruct.

────────────────────────────────────────────────────────────────────
CHATML FORMAT  (Phi-3.5-mini-instruct's native prompt template)
────────────────────────────────────────────────────────────────────
Phi-3.5 was trained using Microsoft's ChatML format. You MUST use
the same format at fine-tune time — otherwise the model interprets
your tokens as regular text instead of turn boundaries.

Template:
    <|system|>
    {system_prompt}<|end|>
    <|user|>
    {instruction}<|end|>
    <|assistant|>
    {output}<|end|>

Special tokens:
    <|system|>    — begins system role
    <|user|>      — begins user turn
    <|assistant|> — begins assistant turn
    <|end|>       — ends any turn (Phi-3.5's EOS-equivalent for turns)

During training we compute loss ONLY on the assistant tokens
(everything after <|assistant|>\\n up to and including <|end|>).
The instruction and system tokens are masked (loss = 0) because
we are teaching the model WHAT TO SAY, not how to repeat prompts.
"""

import json
import os
from typing import Optional

from datasets import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM = "You are a helpful, respectful, and knowledgeable AI assistant."

def format_phi35_prompt(
    instruction: str,
    output: str,
    system: Optional[str] = None,
    training: bool = True,
) -> str:
    """
    Format a single instruction/output pair into Phi-3.5 ChatML format.

    Args:
        instruction: The user's question or task
        output:      The desired model response
        system:      Optional system prompt (persona/constraints)
        training:    If True, include the output (for training loss).
                     If False, stop before assistant response (for inference).
    Returns:
        Formatted string ready for tokenization.
    """
    system = system or DEFAULT_SYSTEM
    prompt = (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{instruction}<|end|>\n"
        f"<|assistant|>\n"
    )
    if training:
        prompt += f"{output}<|end|>"
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_from_json(path: str) -> list[dict]:
    """
    Load instruction/output pairs from a JSON file.

    Expected format:
        [
          {
            "system": "optional system prompt",   ← optional
            "instruction": "user question/task",  ← required
            "output": "desired model response"    ← required
          },
          ...
        ]

    Args:
        path: Path to the .json file
    Returns:
        List of dicts with keys: instruction, output, system (optional)
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required keys are missing
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list of objects.")

    for i, item in enumerate(data):
        if "instruction" not in item:
            raise ValueError(f"Item {i} missing 'instruction' key.")
        if "output" not in item:
            raise ValueError(f"Item {i} missing 'output' key.")

    print(f"Loaded {len(data)} examples from {path}")
    return data


def prepare_dataset(
    data: list[dict],
    tokenizer=None,
    max_seq_len: int = 512,
) -> Dataset:
    """
    Format and tokenize the dataset for SFTTrainer.

    SFTTrainer accepts a HuggingFace Dataset with a 'text' column
    containing the fully formatted prompt strings. It handles
    tokenization and loss masking internally.

    Args:
        data:        List of instruction/output dicts
        tokenizer:   HuggingFace tokenizer (used to check lengths)
        max_seq_len: Maximum sequence length — examples longer than
                     this will be truncated. Phi-3.5 supports up to 128K
                     tokens but 512-2048 is typical for fine-tuning.
    Returns:
        HuggingFace Dataset with 'text' column
    """
    formatted = []
    skipped = 0

    for item in data:
        text = format_phi35_prompt(
            instruction=item["instruction"],
            output=item["output"],
            system=item.get("system"),
            training=True,
        )

        # Optionally check length if tokenizer is available
        if tokenizer is not None:
            tokens = tokenizer(text, return_length=True)["length"][0]
            if tokens > max_seq_len:
                skipped += 1
                continue

        formatted.append({"text": text})

    if skipped > 0:
        print(f"Skipped {skipped} examples exceeding {max_seq_len} tokens.")

    print(f"Prepared {len(formatted)} examples for training.")
    return Dataset.from_list(formatted)


def split_dataset(
    dataset: Dataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset:      Full dataset
        val_fraction: Fraction of data to use for validation (default 10%)
        seed:         Random seed for reproducibility
    Returns:
        (train_dataset, val_dataset)
    """
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    train = split["train"]
    val = split["test"]
    print(f"Train: {len(train)} | Val: {len(val)}")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    examples = [
        {
            "system": "You are an expert Python programmer.",
            "instruction": "What is a list comprehension?",
            "output": "A list comprehension is a concise way to create lists in Python. "
                      "Syntax: [expr for item in iterable if condition]. "
                      "For example: [x**2 for x in range(10) if x % 2 == 0] "
                      "produces [0, 4, 16, 36, 64]. They are faster than equivalent "
                      "for-loops because they are optimized at the C level.",
        },
        {
            "instruction": "Explain what a generator is in Python.",
            "output": "A generator is a function that yields values one at a time "
                      "using the 'yield' keyword instead of returning all at once. "
                      "It is memory-efficient because it computes values lazily — "
                      "only when requested. Use generators for large sequences "
                      "you do not need all in memory simultaneously.",
        },
    ]

    print("=" * 65)
    print("  FORMATTED PROMPT EXAMPLES (Phi-3.5 ChatML format)")
    print("=" * 65)
    for i, ex in enumerate(examples):
        text = format_phi35_prompt(
            instruction=ex["instruction"],
            output=ex["output"],
            system=ex.get("system"),
            training=True,
        )
        print(f"\n--- Example {i+1} ---")
        print(text)

    ds = prepare_dataset(examples)
    print(f"\nDataset created: {ds}")
    train, val = split_dataset(ds, val_fraction=0.5)
    print(f"Train: {len(train)} | Val: {len(val)}")
