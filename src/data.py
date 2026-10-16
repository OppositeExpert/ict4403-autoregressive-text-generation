"""Train-only vocabulary and chapter-disjoint next-word examples."""
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset

SOURCE_URL = "https://www.gutenberg.org/ebooks/11"
DOWNLOAD_URL = "https://www.gutenberg.org/ebooks/11.txt.utf-8"
EXPECTED_SHA256 = "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"
SPLITS = {"train": list(range(1, 9)), "validation": [9, 10], "test": [11, 12]}


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text)


def read_chapters(path):
    raw = Path(path).read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise ValueError(f"Dataset checksum changed: {checksum}. Use the archived raw file.")
    text = raw.decode("utf-8-sig")
    body = text.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[1]
    body = body.split("***", 1)[1].split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    headings = list(re.finditer(r"(?m)^CHAPTER [IVX]+\.\s*\n[^\n]+\n", body))
    if len(headings) != 12:
        raise ValueError(f"Expected 12 chapter headings, found {len(headings)}")
    chapters = {}
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        chapters[i + 1] = tokenize(body[heading.end():end])
    return chapters, checksum


def make_vocab(chapters, min_frequency):
    counts = Counter(w for i in SPLITS["train"] for w in chapters[i])
    words = ["<unk>"] + sorted(w for w, count in counts.items() if count >= min_frequency)
    return {w: i for i, w in enumerate(words)}, counts


def build_examples(chapters, chapter_ids, vocab, context_length):
    xs, ys, origins = [], [], []
    for chapter_id in chapter_ids:
        ids = [vocab.get(w, 0) for w in chapters[chapter_id]]
        for target_position in range(context_length, len(ids)):
            xs.append(ids[target_position - context_length:target_position])
            ys.append(ids[target_position])
            origins.append((chapter_id, target_position - context_length, target_position))
    return (TensorDataset(torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)),
            np.asarray(origins, dtype=np.int64))


def prepare(path, config):
    chapters, checksum = read_chapters(path)
    vocab, counts = make_vocab(chapters, config["min_frequency"])
    datasets, origins, stats = {}, {}, {}
    for split, chapter_ids in SPLITS.items():
        datasets[split], origins[split] = build_examples(
            chapters, chapter_ids, vocab, config["context_length"])
        tokens = [w for i in chapter_ids for w in chapters[i]]
        targets = datasets[split].tensors[1]
        stats[split] = {
            "chapters": chapter_ids, "word_tokens": len(tokens), "examples": len(targets),
            "unique_words": len(set(tokens)),
            "oov_word_fraction": sum(w not in vocab for w in tokens) / len(tokens),
            "unk_target_fraction": (targets == 0).float().mean().item()}
    meta = {"source": SOURCE_URL, "download": DOWNLOAD_URL, "sha256": checksum,
            "vocabulary_size_including_unk": len(vocab), "min_frequency": config["min_frequency"],
            "total_word_tokens": sum(len(x) for x in chapters.values()),
            "train_raw_unique_words": len(counts), "splits": stats,
            "context_length": config["context_length"],
            "tokenization": "NFKC; lowercase; curly apostrophe to ASCII; [a-z]+(?:'[a-z]+)*; no punctuation or digits",
            "prompts": [" ".join(chapters[i][:config["context_length"]]) for i in [1, 2]]}
    return datasets, vocab, origins, meta
