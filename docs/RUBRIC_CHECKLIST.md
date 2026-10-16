# ICT4403 Assignment 2 rubric audit

Authority: the supplied two-page `ICT 4403 Take Home Assignment - Generative AI.pdf`, recovered from the referenced Build AI Assignment chat. The assignment title is Foundations of Generative AI (ICT4403), Assignment 2. This checklist maps requirements to delivered evidence; it does not substitute for the original rubric.

| Rubric item | Evidence | Status |
|---|---|---|
| Suitable textual dataset and characteristics | `data/README.md`, `results/dataset.json`, README | Complete |
| Preprocess and construct input-target sequences | `src/data.py`, `docs/METHODS.md`, `results/sequence_origins.npz` | Complete |
| Implement and train Simple RNN, Bidirectional RNN, LSTM, GRU | `src/models.py`, `src/run.py`, 12 saved checkpoints and `results/training.log` | Complete |
| Comparable experimental conditions | One fixed configuration; 3 seeds and 18 epochs per model; shared epoch ordering | Complete |
| Autoregressive next-word prediction and sequential generation | Many-to-one target outside input window; `src/generate.py`; observed-window-only bidirectionality | Complete |
| Quantitative loss, perplexity, accuracy, training time | `results/runs.csv`, `results/summary.csv`, `results/baseline.json` | Complete |
| Qualitative coherence, diversity, contextual consistency, repetition | `docs/QUALITATIVE_ANALYSIS.md`, all 72 samples, surface diagnostics | Complete |
| Same seed sequences and at least 3 temperatures | 2 identical prompts × T 0.5, 1.0, 1.5 × 3 identical sampling seeds for every model | Complete |
| Temperature influence and justified model choice | README, qualitative analysis; LSTM predictive winner with semantic limits | Complete |
| Correct execution and authentic results | 8 unit tests; 12 checkpoint test metrics and arrays exactly reproduced; source hashes | Complete |
| Python scripts or notebook, supporting files, results, plots, samples | Organized `src/`, `configs/`, `tests/`, `data/`, `results/`, `docs/` | Complete |
| Correct dataset link | https://www.gutenberg.org/ebooks/11 | Complete |
| README covers objective, dataset, models, dependencies, execution, results, comparison, temperature, conclusion | Root README.md | Complete |
| Relevant GitHub repository and accessible files | Public repository at https://github.com/OppositeExpert/ict4403-autoregressive-text-generation; uploaded commit and integrity verification recorded in local delivery receipt | Complete |
| Word document named using name and registration number, containing six required fields | Local `submission/` document with the student's name and registration number in its filename; literal template-name copy also supplied; all six fields and hyperlinks verified | Complete |
| Convert Word document to PDF | Seven-page PDF exported from the final DOCX with LibreOffice; every rendered page inspected; DOCX integrity and required-field checks pass | Complete |
| Upload final PDF to the LMS assignment section before the deadline | User action required; LMS destination and deadline were not supplied in the rubric or current chat | Outstanding |

The report and registration number are distributed in the local delivery package; the public code repository contains the reproducible project and aggregate evidence. The rubric requires uploading code/results to GitHub and sharing its link in the submission document; it does not require publishing personal registration details.
