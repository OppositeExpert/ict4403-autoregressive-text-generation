# Generated text and temperature analysis

## Inspection protocol

All 72 raw, unedited 60-word continuations are in `results/samples.md`, with machine-readable equivalents in JSON and CSV. The quantitative surface diagnostics summarize all 72. The qualitative observations below inspect the fixed sampling-seed-101 subset: every architecture, both prompts, and all three temperatures (24 continuations). This subset uses the first configured sampling seed rather than choosing the most attractive text. Training seed 17 was fixed for generation; no favorable checkpoint or prompt was selected after inspecting samples. The diagnostics exclude the prompt.

The prompts are the first 16 words of chapters 1 and 2:

1. `alice was beginning to get very tired of sitting by her sister on the bank and`
2. `curiouser and curiouser cried alice she was so much surprised that for the moment she quite`

Both prompts are training text. Reproducing familiar phrases after these prompts is not evidence of novel composition or test generalization. The same categorical RNG seeds are reset across model/temperature conditions; vocabulary order is fixed. Shared random draws can produce similar words when predictive distributions are diffuse, so cross-model sample similarity alone is not proof of identical model behavior.

## Coherence and contextual consistency

All four architectures learn recognizable story vocabulary and short local constructions, but the inspected samples fail to maintain grammatical or narrative coherence for 60 words. Characters such as the Hatter, Queen, Rabbit and March Hare appear, yet references jump between scenes without a stable event. Topic vocabulary is therefore only weak contextual consistency. Lower temperature does not repair this limitation: it often produces common phrase fragments and repeated `<unk>`.

The Simple RNN at T=0.5, prompt 1, begins `she said the hatter of the wood and she began to herself in a story tone`. It mixes familiar entities with broken syntax, and later repeats `i don't think i don't`. BiRNN at T=0.5, prompt 1, begins `she said the best of <unk> and ran <unk> her <unk>`. The reverse pass over observed context does not yield a qualitative advantage here, consistent with its weaker validation/test prediction. This does not imply all bidirectional encoders are inferior; it describes this window, dataset and protocol.

LSTM at T=0.5, prompt 2, begins `<unk> out and the march hare she ran out of the <unk>` and later degenerates into repeated unknown tokens. At T=1.0 both prompts produce the same continuation for sampling seed 101, showing prompt information can wash out rapidly under this sampled trajectory. Its best held-out prediction metrics therefore do not establish the best semantic coherence. GRU at T=0.5, prompt 2, opens `forgetting in the air of the little door she was a very difficult of the words`; its local phrase fragments are recognizable but inconsistent. At T=1.0 it has the strongest surface diversity/repetition balance across the six diagnostics samples, without coherent storytelling.

## Temperature effects

The following values are means across 2 prompts × 3 sampling seeds, conditional on training seed 17. Distinct-1 is unique words / 60. Repeated-bigram fraction counts duplicate bigram occurrences beyond the first. These are descriptive diagnostics, not semantic quality scores or significance tests. `<unk>` counts as a word and its fraction is reported separately.

| Model | T | Distinct-1 | Repeated bigrams | Unknown words |
|---|---:|---:|---:|---:|
| Simple RNN | 0.5 | 0.503 | 14.69% | 14.44% |
| Simple RNN | 1.0 | 0.769 | 1.69% | 5.56% |
| Simple RNN | 1.5 | 0.917 | 0.00% | 2.78% |
| BiRNN | 0.5 | 0.456 | 21.19% | 20.83% |
| BiRNN | 1.0 | 0.786 | 1.13% | 5.28% |
| BiRNN | 1.5 | 0.914 | 0.00% | 2.50% |
| LSTM | 0.5 | 0.444 | 29.66% | 30.56% |
| LSTM | 1.0 | 0.769 | 2.54% | 6.94% |
| LSTM | 1.5 | 0.911 | 0.00% | 1.94% |
| GRU | 0.5 | 0.433 | 28.53% | 25.56% |
| GRU | 1.0 | 0.814 | 0.28% | 3.89% |
| GRU | 1.5 | 0.925 | 0.00% | 2.22% |

At **T=0.5**, a sharper distribution favors high-probability function words, short learned constructions and `<unk>`. Across models, distinct-1 is 0.433-0.503 and repeated bigrams occupy 14.69-29.66% of positions. The inspected continuations have more familiar local fragments but strong loops and missing lexical content. This illustrates why “lower temperature means better coherence” is not universally true.

At **T=1.0**, repetition drops substantially and vocabulary variety increases. For LSTM, repeated bigrams decrease from 29.66% to 2.54% and unknown words from 30.56% to 6.94%. GRU has 0.28% repeated bigrams and 0.814 distinct-1, the best surface balance at this temperature. However, the LSTM fragment `as she had asked rules <unk> to herself by her arms off with a story` still lacks grammatical and semantic continuity. T=1.0 is a reasonable exploratory compromise among the tested values, not a validated fluent-writing setting.

At **T=1.5**, all models reach zero repeated bigrams in these samples and distinct-1 above 0.91, with fewer unknown tokens. The flatter distribution samples more low-probability words. The inspected text becomes loosely connected word sequences, such as the GRU fragment `that's say you asked rules it's a bit while messages now i'm had a story`. Zero repeated bigrams therefore reflects surface diversity, not coherent generation. There is no monotonic semantic improvement with temperature.

## Model choice

**LSTM is the preferred predictive model for this experiment:** mean validation loss 4.488 is lowest, and the held-out result independently shows the lowest perplexity (89.67 ± 0.55) and highest top-1 accuracy (19.07 ± 0.47%). Use T=1.0 as a defensible initial generation setting to avoid the severe low-temperature unknown loops. This recommendation prioritizes the prespecified next-word criterion while acknowledging weak free-running quality. GRU is a useful alternative when modestly lower training time and the observed T=1.0 surface diversity are priorities. Simple RNN is fastest; BiRNN provides no observed benefit here. **No architecture demonstrates reliable coherent story generation in this study.** A single most-effective semantic generator cannot be established from these outputs.
