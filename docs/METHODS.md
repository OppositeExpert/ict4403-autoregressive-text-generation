# Experimental method

## Objective and protocol

Estimate the next-word distribution from a 16-word observed context, compare Simple RNN, Bidirectional Simple RNN, LSTM and GRU under one fixed training protocol, and examine free-running generation at temperatures 0.5, 1.0 and 1.5. `configs/experiment.json` was fixed before the recorded experiment. Three training seeds (17, 29, 43), 18 epochs each, produce 12 complete runs. No architecture-specific tuning, external pretrained embeddings, or held-out-test tuning is used.

## Source and preprocessing

The source is Lewis Carroll's *Alice's Adventures in Wonderland*, Project Gutenberg eBook 11, https://www.gutenberg.org/ebooks/11. See `data/README.md` for source integrity and licensing. Strip the Gutenberg wrapper, contents and chapter headings; apply Unicode NFKC normalization, lowercase, normalize curly apostrophes, and retain only ASCII words with optional internal apostrophes. Digits and punctuation are omitted. Hyphenated terms become separate words. This makes the prediction unit an actual word but removes sentence cues and prevents generated punctuation. Unknown words are visibly represented as `<unk>`.

Chapter 10 contains two bracketed later-edition verse additions with short editorial labels; these remain in the body corpus. Brackets disappear under tokenization. Thus the validation split includes a small editorial component alongside narrative/dialogue and verse. The complete edition is preserved and no held-out passages were selectively removed after results were observed.

Split entire chapters BEFORE constructing any windows: train 1-8, validation 9-10, test 11-12. Each chapter is tokenized independently. For chapter tokens w[0:n], create x = w[t-16:t] and y = w[t] for t = 16,...,n-1. All windows have stride 1, remain within one chapter, and exclude their target. The first 16 words of each chapter are context only. There are no padded targets. Overlap within a split is intentional; random splitting of overlapping windows would leak near-identical context and is forbidden here. Chapter IDs and input-start/target positions are archived. Chapter-disjoint source positions are verified; natural repeated phrases across chapters are not artificially removed.

Fit a sorted vocabulary using only training words with frequency at least 2. Reserve ID 0 for `<unk>` and map rare training words and unseen validation/test words there. This fixes a closed evaluation vocabulary and its unknown-word burden; perplexity is for this mapped word space, not raw unrestricted English. Report overall accuracy, known-target accuracy and unknown-target fraction. Known-word accuracy is conditional on the target being in-vocabulary and is not an alternate test set.

## Models and autoregressive validity

All four models use a learned 64-dimensional embedding, one recurrent layer, a 96-dimensional output representation, dropout 0.15 on that representation, and a linear vocabulary classifier. Simple RNN uses tanh with 96 hidden units. BiRNN uses tanh with 48 units in each direction and concatenates the two final hidden states. LSTM and GRU each use 96 hidden units. Parameter counts are measured, not assumed equal. Equal input/output representations keep the embedding and classifier comparable; gating increases the recurrent parameter budget. This is an equal-protocol comparison, not an equal-parameter or equal-compute ablation.

The factorization is p(w[t] | w[max(0,t-16):t]). For BiRNN the forward encoder reads observed words from left to right, and the reverse encoder reads those SAME observed words from right to left. Neither encoder sees w[t] or later words. This is a bidirectional encoder of a past-only window; it is valid for sequential next-word generation. It is not a tokenwise bidirectional language model predicting words inside a full future-containing sequence. Both final hidden states are concatenated, rather than taking only the reverse output at the last input position, which would see too little reversed context. Hidden states reset for each example and each recomputed generation window.

## Optimization and selection

CPU execution with four intra-operation threads, one inter-operation thread, deterministic PyTorch algorithms and fixed random seeds. AdamW with learning rate 0.002, weight decay 0.0001, batch size 128 and global gradient norm clipping at 1.0. Train on every example for exactly 18 epochs. Each architecture receives identical epoch permutations for a given seed via a separate order generator. Dropout and initialization RNG draws differ by architecture, as expected; identical numeric seeds do not imply identical parameters. No scheduler or early stopping is used. Save the epoch with lowest validation cross entropy within each run; evaluate that checkpoint on train, validation and test. Pick the preferred architecture by mean validation loss across the three runs. Test results describe generalization and do not drive tuning or model selection.

## Metrics and timing

Loss = -(1/N) sum log p(y[i] | x[i]), in nats, using natural logs and one target per example. Perplexity = exp(loss), computed AFTER token-weighted aggregation rather than averaging batch perplexities. Top-1 accuracy = correct predictions / N, including mapped unknown targets. Also report known-target accuracy. All checkpoint evaluation uses evaluation mode and disables dropout. Training curves show online training loss while weights change and dropout is active; validation curves use evaluation mode, so these are not exactly like-for-like train/test estimates. The final train metrics do use the selected checkpoint in evaluation mode.

Training wall time sums epoch training loops (data loading, forward, backward, optimizer and scalar accounting); excludes validation, checkpoint writes and final evaluation. Validation time, separate evaluation times, and total per-run wall time are also saved. CPU work is synchronous and measured with `time.perf_counter`. Runs execute sequentially in fixed architecture order; timings are illustrative on this host, can include thermal/background-process effects, and are not randomized performance benchmarks. The run manifest records actual environment and elapsed execution time. Runtime clock timestamps are diagnostic; the submission date follows the user's local date.

Tables show arithmetic mean and sample SD across the three training seeds. Mean perplexity is the mean of per-run perplexities and is not necessarily exp(mean loss). Seed SD is training variability, not a confidence interval or corpus-level uncertainty; overlapping windows and only two test chapters preclude independent-token inference. No claim of statistical superiority is made from three seeds alone.

## Generation protocol

Use the validation-selected checkpoint from the FIRST configured training seed (17) for every model, chosen without test or sample inspection. Prompts are the first 16 training words of chapters 1 and 2. For every model and prompt generate 60 new words at temperatures 0.5, 1.0, 1.5, using sampling seeds 101, 202, 303: 4 × 2 × 3 × 3 = 72 continuations. Reset the same CPU sampling generator for every condition. Sampling draws therefore use the same random-seed schedule; differing probability distributions need not produce corresponding words. Recompute the last 16-word context after each sampled token. No teacher forcing occurs during generation.

Sample from softmax(logits/T) over the full vocabulary, including `<unk>`. Do not filter, suppress repetition, cherry-pick, or repair text. T below 1 sharpens probabilities and T above 1 flattens them. All samples are archived. Distinct-1 and distinct-2 are unique n-grams / total n-grams of the continuation only. Repeated-bigram fraction = 1 - distinct-2; word entropy uses empirical word counts in bits. Unknown fraction is shown explicitly. These are surface diagnostics, not measurements of coherence or semantic correctness. Temperature summaries contain 6 samples per model/temperature (2 prompts × 3 sampling seeds); their SD is sample variability conditional on one trained checkpoint, not training uncertainty. Qualitative commentary inspects grammar, topic/character consistency and repetition in the unedited continuations.

## Limitations

One small literary work and a fixed 16-word window cannot establish general language-model quality or long-range story consistency. The sequential chapter split intentionally creates lexical/topic shift; large unknown fractions reduce linguistic resolution and can make mapped perplexity appear easier. `<unk>` can be generated and counts in metrics. Punctuation removal limits fluency. Different recurrent parameter counts, shared hyperparameters and a short fixed budget may favor particular architectures. The simple unigram baseline helps reveal whether context improves prediction, but is not an equally optimized neural model. More books, subword modelling, parameter-matched comparisons and more seeds are future work, not completed experiments.

## Technical references

1. Project Gutenberg. Lewis Carroll, *Alice's Adventures in Wonderland*, eBook 11. https://www.gutenberg.org/ebooks/11 (accessed 16 October 2026).
2. PyTorch 2.9 RNN documentation. https://docs.pytorch.org/docs/2.9/generated/torch.nn.RNN.html
3. PyTorch 2.9 LSTM documentation. https://docs.pytorch.org/docs/2.9/generated/torch.nn.LSTM.html
4. PyTorch 2.9 GRU documentation. https://docs.pytorch.org/docs/2.9/generated/torch.nn.GRU.html
5. PyTorch 2.9 CrossEntropyLoss documentation. https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html
