# Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents (SyncLLM)

| Field | Value |
|---|---|
| Org | Meta AI + University of Washington (p.1) |
| Date / arXiv | Not printed in the provided PDF (no arXiv stamp). Webpage syncllm.cs.washington.edu (p.1) |
| Category | full-duplex S2S (two-speaker dialogue continuation, plus simulated two-agent interaction) |
| Backbone | Llama3-8B (text-only), fully fine-tuned. Vocab extended with 501 HuBERT units plus the speaker/sync tags [S0], [S1]. Context 8,192 (p.5, p.11) |
| Audio input | HuBERT units: 25 Hz (one token per 40 ms), vocab 501, using SpiRit-LM's tokenization parameters (p.4). The same tokenizer is used for both speakers |
| Audio output | The same HuBERT units: a single codebook, deduplicated, interleaved per time chunk with sync tags (p.4–5) |
| Waveform decoder | A simple HiFi-GAN vocoder from units (p.9). Training details not reported |
| Duplex mechanism | Time-division multiplexing of both speakers' 160–240 ms chunks in one sequence; the model predicts the user's chunk before its own (p.3–4) |
| Open weights / code | Not stated in the paper |
| Relevance to us | **high**: the closest published recipe to ours, with stage-by-stage data and hyperparameters |

## TL;DR

- Llama3-8B becomes a full-duplex model by interleaving fixed-duration chunks of both speakers' deduplicated HuBERT tokens. A periodic tag ([S0]) anchors the sequence to wall-clock time (p.3–5).
- **Three stages** (p.5–6, Tab.1):
  1. ≈193k h of TTS'd text SFT dialogues, turn-based, with sentence-level text/speech interleaving.
  2. ≈20k h of synthetic two-channel dialogue, with the other speaker silent.
  3. Fisher, 1,927 h of real two-channel dialogue.
- **Meaningfulness MOS** 3.74 vs 1.55 for dGSLM, at similar turn-taking naturalness (3.90 vs 3.95). The resynthesized-ground-truth topline is 3.87 / 4.03 (p.8, Tab.3).
- **Latency:** two models talk to each other with one-chunk delay. Performance holds at 160–200 ms and degrades at 240 ms (App. C, p.13).
- **Not in the paper:** no data-scaling experiment (no sweep over synthetic or real hours) and no ablation that removes Stage 1 or Stage 2.

## Architecture

**Token format** (p.4–5, Fig.2):
- Each speaker channel is tokenized separately at 25 Hz.
- Chunk size is 160, 200 or 240 ms, i.e. 4, 5 or 6 raw tokens per stream per chunk (derived).
- Inside a chunk: [S0] + speaker 0's *new* tokens, then [S1] + speaker 1's new tokens.
  - Repeats of the previous token are dropped.
  - If speaker 1 has no new token, [S1] is dropped too.
  - [S0] is always present, so each chunk boundary (and therefore time) is unambiguous.

**Deduplication.** The raw stream is ≈58 tokens/s. With dedup the distribution centres at ≈26–27 tokens/s (both read from Fig.3, p.5, measured on 15 h of Fisher). The authors cite ~50% lower semantic content per token without dedup (p.4).

**Interpolation for synthesis.** Each chunk's token budget is known, so deduplicated tokens are repeated evenly to refill the chunk before vocoding. The error is bounded by the chunk size; the authors found its effect imperceptible even at 240 ms (p.4–5).

**Run-time loop** (p.3–4, Fig.1):
- At chunk N, the context holds the model's chunks up to N and the user's chunks up to N−1.
- The model generates an *estimate* of the user's chunk N, then its own chunk N+1.
- When the real user chunk N arrives, it replaces the estimate.
- Real-time constraint: both must be generated within one chunk duration (Fig.1 caption, p.2).

**Context budget:** ≈5 min of dialogue (derived: 8,192 ÷ ≈27 tokens/s). The authors say this limit also rules out higher-rate multi-codebook tokenizers such as EnCodec (p.9).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
The HuBERT tokenizer reuses SpiRit-LM's parameters (p.4). The vocoder is HiFi-GAN (p.9). No training details are given.

### Stage 1: turn-based spoken dialogue, text–speech interleaved (synthetic)
- **Data:** text SFT dialogue datasets (not named), rendered with Bark TTS using its 10 speaker presets. ≈193k h of speech-equivalent audio (p.5, Tab.1).
- **Format:** `[S1]<sent0>[S0]<sent0><sent1>[S1]...`. Each sentence is randomly kept as text or replaced by deduplicated speech tokens (p.5).
- **Speech-ratio curriculum.** The per-sample speech share is drawn from a truncated normal whose mean rises across epochs (read from Fig.4, p.5):
  - epoch 0: ≈20%
  - epoch 1: ≈50%
  - epoch ≥2: ≈85%

  The legend implies at least 3 epochs.
- **Stability finding:** training on all-speech sequences, or raising the speech share in discrete steps, was unstable (p.5–6).
- **Hyperparameters** (p.11):
  - 128×A100 at per-GPU batch 1 × 8,192 tokens, i.e. about 1M tokens per step.
  - LR 3e-5, 500 warmup steps, 40k iterations.
  - Tokens seen ≈42B (derived). The whole model is trained; nothing is reported as frozen.

### Stage 2: full-duplex format, no overlaps (synthetic)
- **Data:** text dialogues with shorter turns (not named), ≈20k h (p.6).
- **Construction:** for each turn, speech for the current speaker and *silence of equal duration* for the other. The two channels are then tokenized into the chunk format (p.6).
- **What it teaches:** format and timing within utterances, but not backchannels or overlaps (p.6).
- **Hyperparameters:** 512k-token batch, LR 2.2e-5, 200 warmup steps, 6,000 iterations; ≈3.1B tokens (derived) (p.11).

### Stage 3: real duplex dialogue (Fisher)
- **Data:** Fisher, 2,000 h with one channel per speaker, split 98:1:1. Tab.1 lists 1,927 h (p.5–6).
- **What it teaches:** turn-taking, the distribution of pauses, and backchanneling (p.6).
- **Hyperparameters:** 256k-token batch, LR 1.5e-5, 100 warmup steps, 2,000 iterations; ≈0.5B tokens (derived) (p.11).
- **Variant:** SyncLLM-C uses the CANDOR training split instead (p.8). CANDOR's size is not reported.

## Data pipeline

- **Synthetic audio:** text SFT dialogues → Bark TTS (10 presets), with no overlap phenomena; those come only from Fisher (p.6).
- **Mixture shares** (derived):
  - Real duplex is ≈0.9% of hours (1,927 / ≈214.9k).
  - Real duplex is ≈1.1% of tokens (≈0.52B / ≈45.6B).
  - Stage 1 alone is ≈92% of tokens.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| 1 | ≈193k h synthetic; ≈42B tokens (derived) | 128×A100; wall time not reported | LR 3e-5, 40k iterations, 1M tokens/batch |
| 2 | ≈20k h synthetic; ≈3.1B tokens (derived) | not reported | LR 2.2e-5, 6k iterations, 512k tokens/batch |
| 3 | 1,927 h Fisher; ≈0.5B tokens (derived) | not reported | LR 1.5e-5, 2k iterations, 256k tokens/batch |

## Preserving text intelligence / knowledge

- **Argument.** The authors argue the speech-only dGSLM cannot exploit textual knowledge. Keeping the standard LLM architecture lets SyncLLM inherit text pretraining (p.3, p.7).
- **Mechanism.** Sentence-level text/speech interleaving, with the speech share rising gradually.
- **Not measured.** There is no text benchmark or spoken-QA evaluation after training. "Semantics" means ASR-transcript perplexity under a text dialogue model, plus human Meaningfulness MOS.

## Key results

- **Semantics** (Figs.5–6, p.6–7; 10 s prompts): median transcript perplexity is about 70 points worse than ground truth for dGSLM and about 15 worse for SyncLLM. Read from Fig.6: dGSLM ≈80–134; SyncLLM-F ≈27–37 in-distribution, ≈30–34 out-of-distribution.
- **Turn-taking naturalness** (Tab.2, p.7). Pearson correlation of IPU, pause and floor-transfer-offset durations with ground truth; 30 s prompt → 90 s generation; average shown:

| | dGSLM | SyncLLM 160 ms | 200 ms | 240 ms | Prompt | Resynth-GT |
|---|---|---|---|---|---|---|
| Fisher (in-distribution) | 0.33 | 0.43 | 0.43 | 0.41 | 0.52 | 0.79 |
| CANDOR (OOD) | 0.14 | 0.23 | 0.29 | 0.31 | 0.32 | 0.71 |

- **Human evaluation** (Tab.3, p.8; 32 raters, 180 items, 160 ms chunks):

| Model | Meaningfulness MOS | Naturalness MOS |
|---|---|---|
| dGSLM | 1.55 | 3.95 |
| SyncLLM-C | 3.40 | 3.96 |
| SyncLLM-F | 3.74 | 3.90 |
| Re-synth | 3.87 | 4.03 |
| GT | 4.96 | 4.96 |

  - Drop when moving out of distribution: dGSLM −0.24 / −0.51; SyncLLM-F −0.15 / −0.16 (Meaningfulness / Naturalness).
- **Interaction** (Tab.4, p.9; two models, one-chunk latency): Meaningfulness / Naturalness MOS
  - SyncLLM-F-F: 3.47 / 3.72.
  - SyncLLM-F-C: 3.39 / 3.78.

  Below continuation mode (3.74 / 3.90), far above dGSLM.

## Ablations & lessons

- **Sentence-level beats turn-level interleaving** in Stage 1 (Tab.5, p.12; zero-shot):

| Metric | Sentence-level | Turn-level |
|---|---|---|
| sWUGGY | 70.3 | 63.0 |
| sBLIMP | 56.3 | 56.0 |
| Topic-StoryCloze | 83.0 | 76.5 |
| StoryCloze | 61.8 | 55.1 |

- **A speech-ratio curriculum is required for stability** when starting from a text-only LLM (p.5–6). Qualitative only, no curves.
- **Latency tolerance** (App. C, p.13):
  - Interaction-mode perplexity is ≈28–31 at 160/200 ms vs ≈35–36.5 at 240 ms (read from Fig.8).
  - Turn-taking correlation (Tab.6) at 160/200/240 ms: F-F 0.32 / 0.31 / 0.28; F-C 0.36 / 0.35 / 0.32.
  - The authors conclude it is robust up to 200 ms. This conflicts with the intro, which claims resilience up to 240 ms (p.2).
- **Chunk size in continuation mode** barely matters in-distribution (0.43 / 0.43 / 0.41). Larger chunks improved out-of-distribution turn-taking (0.23 → 0.29 → 0.31) (Tab.2).
- **Generalization.**
  - Fisher-trained SyncLLM degrades less out of distribution than dGSLM, partly credited to general-purpose rather than Fisher-finetuned HuBERT units (p.8).
  - Agents trained on different data (Fisher vs CANDOR) still converse coherently (Fig.7, p.8–9).
- **The tokenizer/vocoder is the main ceiling.** Resynthesized ground truth scores 3.87 Meaningfulness vs 4.96 for real audio; the LM loses only a further 0.13 (Tab.3).
- **Missing evidence.**
  - No ablation of dedup vs no-dedup; its benefit is argued from Fig.3 and prior work.
  - No data-scaling curve and no stage-removal ablation. The claim that the synthetic stages let Stage 3 "converge much faster" (p.7) has no reported experiment behind it.

## Implications for our setup

- **We skip the stage they spend most of their compute on.** ≈92% of SyncLLM's tokens are text-anchored spoken dialogue built from text SFT data; real duplex is ≈1%. Weak semantics after duplex-only mid-training is consistent with that.
  - Direct test: insert a stage of sentence-level text/speech-interleaved dialogue (TTS'd text SFT data) with a rising speech ratio (≈20% → 50% → 85%).
  - Then a cheap synthetic two-channel stage (turns plus silence).
  - Then our 10K h of duplex data.
- **Interleave at sentence granularity, not turn granularity** (+6–7 points on StoryCloze-type tests).
- **Check how much of our loss is silence** (our inference). Each channel of duplex data is silent much of the time, and SyncLLM's dedup shrinks silence to roughly one tag per chunk. We cannot dedup a multi-codebook codec, but we can measure the silence share of our tokens and consider down-weighting it.
- **Measure the codec topline before blaming the LM.** The tokenizer+vocoder gap (4.96 → 3.87) dwarfs the LM gap (3.87 → 3.74). Resynthesize our conversational audio through our tokenizer and vocoder and score it first.
- **If we use chunked interleaving,** keep chunks at or below 200 ms. Predicting the user's next chunk absorbs one chunk of delay.
- **LR schedule.** Decreasing LR and batch across stages for an 8B model (3e-5 → 2.2e-5 → 1.5e-5; 1M → 512k → 256k tokens) is a reasonable default.

## Open questions / unclear points

- Which text SFT datasets were used; how Stage-2 inter-turn silence was set; how Bark presets were assigned to roles.
- The abstract says 212k h of synthetic data, while Tab.1 gives 193k + 20k = 213k h (rounding).
- Fisher is described as 2,000 h but Tab.1 lists 1,927 h.
- The ASR model and the text dialogue model used for perplexity are not named.
- Whether the loss covers the user-channel tokens. This is implied, since continuation mode generates both sides.
