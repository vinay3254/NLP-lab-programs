"""
inference.py
============
Greedy and beam search decoding for the trained Transformer.

Theory:
-------
At inference time, we generate target tokens ONE BY ONE (autoregressive),
feeding each predicted token back as input for the next step.

Two strategies:

1. GREEDY DECODING:
   At each step, pick the single most-probable next token:
     token_t = argmax P(token | token_{1..t-1}, source)
   
   Fast but can get stuck in suboptimal sequences (locally good, globally bad).

2. BEAM SEARCH:
   Maintain the top-k hypotheses (beams) at each step:
     - Expand each beam by all vocab tokens
     - Score each expanded hypothesis: sum of log-probabilities
     - Keep top-k scoring complete hypotheses
   
   More expensive (O(beam_width × vocab × seq) per step) but produces
   significantly better translations.
   
   Log-probabilities are used instead of probabilities to avoid numerical
   underflow and to turn the product into a sum (numerically stable).
"""

import os
import torch
from .transformer import Transformer
from .dataset import Vocabulary


def load_model(checkpoint_path: str, device: torch.device = torch.device("cpu")):
    """
    Load a saved model checkpoint.

    Returns:
        model:     Loaded Transformer (eval mode)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
    """
    # weights_only=False needed because checkpoint includes Vocabulary objects
    # (Python class instances), not just tensors. Safe for our own local checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    src_vocab = checkpoint["src_vocab"]
    tgt_vocab = checkpoint["tgt_vocab"]

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        pad_idx=0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (loss={checkpoint['loss']:.4f})")
    return model, src_vocab, tgt_vocab


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    sentence: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    max_len: int = 50,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Greedy decoding: always pick the highest-probability next token.

    Args:
        model:     Trained Transformer model
        sentence:  Source sentence (English string)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        max_len:   Maximum number of tokens to generate
        device:    torch device

    Returns:
        Translated sentence (French string)
    """
    model.eval()

    # Encode source
    src_ids = torch.tensor(src_vocab.encode(sentence), dtype=torch.long).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_ids)
    encoder_output = model.encode(src_ids, src_mask)

    # Initialize decoder input with <BOS>
    bos_idx = tgt_vocab.word2idx["<BOS>"]
    eos_idx = tgt_vocab.word2idx["<EOS>"]
    tgt_ids = [bos_idx]

    for step in range(max_len):
        tgt = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt)

        decoder_out = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = model.output_projection(decoder_out)  # (1, seq, vocab)

        # Suppress <PAD> and <BOS> from being predicted
        logits[0, -1, 0] = float("-inf")         # PAD
        logits[0, -1, bos_idx] = float("-inf")   # BOS

        # Pick the most likely next token (greedy)
        next_token = logits[0, -1, :].argmax().item()
        tgt_ids.append(next_token)

        if next_token == eos_idx and step >= 1:
            break

    return tgt_vocab.decode(tgt_ids[1:])  # skip leading BOS


@torch.no_grad()
def beam_search(
    model: Transformer,
    sentence: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    beam_width: int = 4,
    max_len: int = 50,
    device: torch.device = torch.device("cpu"),
    length_penalty: float = 0.6,
) -> list[tuple[float, str]]:
    """
    Beam search decoding: explore top-k hypotheses simultaneously.

    Args:
        model:          Trained Transformer
        sentence:       Source sentence
        src_vocab:      Source vocabulary
        tgt_vocab:      Target vocabulary
        beam_width:     Number of beams to maintain (k)
        max_len:        Max output length
        device:         torch device
        length_penalty: Alpha for length normalization. score / len^alpha.
                        Prevents beam search from favoring short sequences.

    Returns:
        List of (score, translation) tuples sorted by score (best first)
    """
    model.eval()
    bos_idx = tgt_vocab.word2idx["<BOS>"]
    eos_idx = tgt_vocab.word2idx["<EOS>"]

    # Encode source
    src_ids = torch.tensor(src_vocab.encode(sentence), dtype=torch.long).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_ids)
    encoder_output = model.encode(src_ids, src_mask)

    # Repeat encoder output for each beam
    encoder_output = encoder_output.repeat(beam_width, 1, 1)  # (beam, src_seq, d_model)
    src_mask = src_mask.repeat(beam_width, 1, 1, 1)

    # Each beam: (cumulative_log_prob, [token_ids])
    beams = [(0.0, [bos_idx])]
    completed = []

    for step in range(max_len):
        new_beams = []

        for score, token_ids in beams:
            if token_ids[-1] == eos_idx:
                # This beam already completed
                completed.append((score, token_ids))
                continue

            # Decode current sequence
            tgt = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
            # Expand for all beams (simplification: decode one beam at a time)
            tgt_mask = model.make_tgt_mask(tgt)

            dec_out = model.decode(tgt, encoder_output[:1], src_mask[:1], tgt_mask)
            logits = model.output_projection(dec_out)     # (1, seq, vocab)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)  # (vocab,)

            # Get top-k next tokens
            topk_log_probs, topk_tokens = log_probs.topk(beam_width)

            for log_prob, token in zip(topk_log_probs.tolist(), topk_tokens.tolist()):
                new_score = score + log_prob
                new_beams.append((new_score, token_ids + [token]))

        if not new_beams:
            break

        # Keep top beam_width beams by score
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

        # Early stopping: all beams ended with <EOS>
        if all(ids[-1] == eos_idx for _, ids in beams):
            completed.extend(beams)
            break

    # Add remaining beams to completed
    completed.extend(beams)

    # Length-normalize scores and decode
    def normalized_score(score: float, ids: list[int]) -> float:
        length = max(1, len(ids) - 1)  # -1 for <BOS>
        return score / (length ** length_penalty)

    results = []
    for score, ids in completed:
        norm_score = normalized_score(score, ids)
        translation = tgt_vocab.decode(ids)
        results.append((norm_score, translation))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def interactive_translate(
    checkpoint_path: str = None,
    device: str = "auto",
):
    """
    Interactive translation REPL.
    Load the model and translate sentences one by one.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "best_model.pt")

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Run training first.")
        return

    model, src_vocab, tgt_vocab = load_model(checkpoint_path, device)

    print("\n" + "=" * 60)
    print("  Transformer EN→FR Translation (type 'quit' to exit)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("English: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # Greedy
        greedy_result = greedy_decode(model, user_input, src_vocab, tgt_vocab, device=device)
        print(f"French (greedy): {greedy_result}")

        # Beam search (top-3)
        beam_results = beam_search(model, user_input, src_vocab, tgt_vocab, beam_width=4, device=device)
        if beam_results:
            print(f"French (beam-1): {beam_results[0][1]}")
            if len(beam_results) > 1:
                print(f"French (beam-2): {beam_results[1][1]}")
        print()


if __name__ == "__main__":
    interactive_translate()
