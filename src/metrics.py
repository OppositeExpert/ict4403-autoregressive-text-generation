"""Token-weighted natural-log cross entropy and transparent text diagnostics."""
import math
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F


def summarize_predictions(nll, correct, targets):
    nll, correct, targets = np.asarray(nll), np.asarray(correct), np.asarray(targets)
    known = targets != 0
    return {"loss": float(nll.mean()), "perplexity": math.exp(float(nll.mean())),
            "accuracy": float(correct.mean()),
            "known_word_accuracy": float(correct[known].mean()) if known.any() else None,
            "unk_target_fraction": float((~known).mean()), "examples": len(targets)}


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    nll, correct, targets = [], [], []
    for x, y in loader:
        logits = model(x)
        nll.extend(F.cross_entropy(logits, y, reduction="none").cpu().tolist())
        correct.extend((logits.argmax(1) == y).cpu().tolist())
        targets.extend(y.cpu().tolist())
    arrays = {"nll": np.asarray(nll), "correct": np.asarray(correct), "targets": np.asarray(targets)}
    return summarize_predictions(**arrays), arrays


def text_metrics(words):
    def distinct(n):
        grams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        return len(set(grams))/len(grams) if grams else 0.0
    counts = Counter(words)
    probabilities = np.asarray(list(counts.values()), dtype=float)/len(words)
    return {"distinct_1": distinct(1), "distinct_2": distinct(2),
            "repeated_bigram_fraction": 1-distinct(2),
            "word_entropy_bits": float(-(probabilities*np.log2(probabilities)).sum()),
            "unk_fraction": words.count("<unk>")/len(words)}


@torch.inference_mode()
def generate(model, vocab, prompt, context_length, temperature, seed, words):
    from .data import tokenize
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    reverse = {i: w for w, i in vocab.items()}
    context = [vocab.get(w, 0) for w in tokenize(prompt)]
    if not context:
        raise ValueError("prompt must contain at least one word")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    continuation = []
    for _ in range(words):
        x = torch.tensor([context[-context_length:]], dtype=torch.long)
        # Full categorical sampling: no top-k, top-p, UNK mask, or repetition penalty.
        probabilities = torch.softmax(model(x)[0]/temperature, dim=-1)
        next_id = torch.multinomial(probabilities, 1, generator=rng).item()
        context.append(next_id)
        continuation.append(reverse[next_id])
    return continuation
