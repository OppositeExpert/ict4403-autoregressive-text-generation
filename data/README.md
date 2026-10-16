# Dataset provenance

Lewis Carroll, *Alice's Adventures in Wonderland* (1865), Project Gutenberg eBook 11, English plain text. A compact narrative with dialogue, recurring characters, unusual words, and 12 chapters makes training feasible and supports a chapter-disjoint generalization test.

- Catalogue and source: https://www.gutenberg.org/ebooks/11
- Download endpoint: https://www.gutenberg.org/ebooks/11.txt.utf-8
- Retrieved: 16 October 2026
- SHA-256 of the archived bytes: `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`
- The source identifies the work as public domain in the USA. The complete downloaded file, including its Project Gutenberg license and jurisdiction notice, is retained unchanged in `raw/alice_pg11.txt`. No account, paid subscription, or scraped private data is used.
- Release/update metadata belongs to this digital edition; 1865 refers to the original literary work.

The code excludes the Gutenberg header/footer, table of contents, and chapter headings from modelling. Training chapters are 1-8, validation chapters 9-10, and test chapters 11-12. The vocabulary is built from training text only. Frozen dataset statistics are in `results/dataset.json` and chapter/position sequence provenance is in `results/sequence_origins.npz`.

The two bracketed later-edition verse additions and their brief editorial labels inside chapter 10 are retained as part of this edition's body text. Bracket punctuation is removed by word tokenization. This is a small editorial component of the validation corpus; no passages were selectively removed after evaluation.

The archived text is the reproducibility authority. A live re-download may change; `python -m src.download_data` checks the exact checksum and refuses a different edition.
