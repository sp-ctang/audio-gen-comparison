# Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs

| Field | Value |
|---|---|
| Org | Max Planck Institute for Intelligent Systems / Tübingen AI Center / ETH Zurich / Univ. Hospital Tübingen / Univ. Tübingen / ELLIS Institute Tübingen (p.1) |
| Date / arXiv | arXiv 2605.12460v1, 12 May 2026 (p.1) |
| Category | method (text-only, multi-stream instruction tuning; no audio) |
| Backbone | Efficiency experiments: Qwen3-1.7B and Qwen3-4B (thinking mode). Security: Qwen2.5-7B-base and Qwen3-4B-base. Monitorability: Qwen3-8B and Qwen3.5-27B. All parameters are shared across streams (p.7–9, p.32) |
| Audio input | n/a |
| Audio output | n/a |
| Waveform decoder | n/a |
| Duplex mechanism | H parallel token streams, one per role (user input, system, model output, thinking, audit…). Each forward pass ("row") emits one token per stream, with "-" marking an idle slot. A block-causal mask lets every stream see all streams at earlier rows. Each stream has its own RoPE counter plus a learnable stream embedding (p.5–7) |
| Open weights / code | Code, model, demo and data links are given (p.1) |
| Relevance to us | **medium**: no speech. But it is a clean, partly ablated text analogue of Moshi-style multi-stream duplex modeling, covering cross-stream position encoding, idle tokens, loss masking and causally valid synthetic data |

## TL;DR

- The paper replaces sequential chat messages with H **parallel streams**. Every forward pass reads all input streams and writes one token to each output stream. The model is a standard decoder-only Transformer with three changes:
  1. **Per-stream RoPE indexing**: each stream's counter starts at 0, so tokens in the same row share a position.
  2. A **learnable stream embedding**.
  3. A **stream-causal mask**: a key is visible if it sits in an earlier row (p.6).
- The format is cheap to learn: 3 epochs of SFT on roughly 8k–48k synthetic samples on 4×B200, LR 1e-5 (Tab.5, p.24–27).
- **Solving while reading** cuts TNFT (the number of tokens generated before the first answer token) to 0 and first-answer delay by 17–47% (derived). Accuracy stays within about ±3.4 points of single-stream SFT on the same data (Tab.1).
- A **concurrent audit stream** beats sequential reflection on accuracy and cuts delay by about 41% (derived) (Tab.2).
- **Ablation: position encoding matters.** The shared-time per-stream RoPE beats 2D RoPE, offsets and NoPE on loose IFEval metrics (Tab.7).
- Side results: lower prompt-injection success rates and better monitorability (Tab.3–4). These are tangential to us.

## Architecture

**Factorization (Eq.1, p.5).** Each stream is autoregressive over its own past and over all streams' earlier rows. In Multiverse-style parallel reasoning, by contrast, branches cannot see each other.

**Positions (p.6).** RoPE is counted per stream, so time aligns across streams. A learnable stream embedding marks identity. These address "attention contention" and RoPE "positional conflicts".

**Packing (p.6, Fig.3).** Interleaved (row-major) packing gives a near lower-triangular mask and reuses FlashAttention's causal path. The cost is "benign" leakage: tokens in the same row on lower-indexed streams become visible, which is the visibility rule the causal-verification rubric uses (p.34).

**Loss.**
- Cross-entropy is averaged within each stream (Eq.3), and the user column is masked out (p.31).
- An optional stream-contrastive reweighting upweights tokens that rely most on other streams: weight = min(exp(LPS), γ), with LPS = log p(y | all streams) − log p(y | own stream). It targets cross-stream loss imbalance (p.27–28). **No results are reported for it.**

**Inference (p.7).** Rows are decoded synchronously. Idle "-" tokens are masked and get **no KV-cache entries**.

**Hybrid backbones (p.31–32).** With Gated DeltaNet layers (Qwen3.5), each stream keeps its own recurrent state, and only the attention layers mix streams.

**Relation to Moshi (p.3, p.20).** Moshi *sums* the streams' embeddings into one input per timestep. Here each stream keeps its own token and prediction in every row.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
n/a (text only).

### Stage 1: Multi-stream SFT (three separate instantiations)

**(a) Efficiency.**
- Models: Qwen3-1.7B and Qwen3-4B, thinking mode.
- Data, converted to stream format: MetaMathQA (about 8k samples), LogicNLI (about 8k), ProofWriter (about 25k), PubMedQA (about 26k), SQuAD (about 15k) (p.23–25).
- Optimization: 3 epochs, global batch 64, LR 1e-5 constant with 10% warmup, weight decay 1e-4, bf16, AdamW, 4×B200 (Tab.5, p.27).
- The "Vanilla" baseline trains on the same data in single-stream form.

**(b) Security.**
- Models: Qwen2.5-7B-base and Qwen3-4B-base.
- Data: about 48k Alpaca-cleaned samples rebuilt into system / user / assistant streams, with Qwen3-Next-80B used for both construction and verification.
- 3 epochs and no adversarial training. Vanilla is the same data collapsed into a single stream (p.8, p.25, p.29–30).

**(c) Monitorability.**
- Model: Qwen3.5-27B with 10 streams (input, output, and 8 thinking streams with assigned roles).
- Data: **3,864 synthetic conversations** of about 100 rows each, written directly as markdown tables by Claude Opus 4.5–4.7.
- Optimization: 2 epochs, 4×B200, LR 2e-5, weight decay 1e-3, attention dropout 0.2. Random concatenation of 7 samples into sequences of up to 1,024 rows.
- The same pipeline trains a Qwen3-8B counterpart on 4×A100 (p.9, p.31).
- Wall-clock time is not reported for any run.

## Data pipeline

**Wait-k stream data (p.5, p.20).**
- An LLM rewrites existing corpora into multi-stream samples.
- The assistant starts after only k user tokens, opening with a **bridging utterance** ("Let me start helping you…").
- Each later chunk t_i may use only the user prefix s_1…s_i.
- k varies across samples, through different bridges and onset timings.

**Direct stream tables (p.6).** Frontier models write all streams row by row in a table. Writing row by row stops the generator from using other streams' future content, so it is preferred over writing the streams one at a time.

**Causal verification (p.6, p.21).**
- An LLM judge sees the readable user prefix available at each assistant chunk, plus the previous assistant tokens.
- It flags any content that could only come from future user tokens, including across multiple assistant streams.
- Failing samples are discarded or regenerated.

**Quality filtering (p.6, p.21).**
- Per stream: malformed tokens, fluency, repetition, completeness.
- Across streams: each stream fulfils its role, and audits are substantive.
- Samples are scored by an LLM judge and thresholded.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Efficiency SFT (1.7B / 4B) | about 8k–26k samples per task; tokens not reported | 4×B200; time not reported | 3 epochs, batch 64, LR 1e-5 |
| Security SFT (7B / 4B) | about 48k Alpaca samples | 4×B200 (Tab.5 "all experiments"); time not reported | 3 epochs |
| Monitorability SFT (27B / 8B) | 3,864 conversations × about 100 rows × 10 streams | 4×B200 (27B), 4×A100 (8B); time not reported | 2 epochs, LR 2e-5 |

## Preserving text intelligence / knowledge

The format change is learned with little or no accuracy loss compared with single-stream SFT on the same data (Tab.1):

| Model | GSM8K (stream / vanilla) | MATH500 | LogicNLI | SQuAD |
|---|---|---|---|---|
| Qwen3-1.7B | 89.51 / 90.60 | 51.60 / 48.20 | 61.25 / 61.55 | 53.50 / 51.70 |
| Qwen3-4B | 88.85 / 89.36 | 64.00 / 60.80 | 63.55 / 62.00 | 74.74 / 75.10 |

IFEval comparison (Tab.9):

| Model | Inst-L (stream / vanilla) | Prompt-L | Inst-S | Prompt-S |
|---|---|---|---|---|
| Qwen3-4B | 60.19 / 51.79 | 49.72 / 39.56 | 47.72 / 48.20 | 35.86 / 35.67 |
| Qwen2.5-7B | 54.43 / 54.92 | 43.62 / 44.36 | 44.36 / 44.60 | 32.90 / 31.98 |

On Qwen2.5-7B the two are roughly at parity. **No broad knowledge benchmark (such as MMLU) is evaluated**, and the SFT sets are tiny.

## Key results

**Delay, stream vs vanilla (Tab.1).**

| Model | GSM8K | MATH500 | LogicNLI | SQuAD |
|---|---|---|---|---|
| Qwen3-1.7B | 14.93 → 11.29 | 43.14 → 22.94 | 45.95 → 38.18 | 7.79 → 4.62 |
| Qwen3-4B | 20.17 → 14.53 | 45.82 → 26.51 | 58.31 → 47.39 | 9.68 → 5.48 |

TNFT falls from about 93–358 tokens to 0 in every case. Delay units are not stated.

**Audit while solving vs Vanilla + Reflection (LogicNLI, Tab.2).**

| Model | Accuracy | Delay | Max stream length |
|---|---|---|---|
| Qwen3-1.7B | 65.65 vs 64.95 | 59.70 vs 102.08 | 2,453.57 vs 4,206.94 |
| Qwen3-4B | 65.55 vs 65.15 | 78.86 vs 132.85 | 2,454.89 vs 4,251.33 |

Throughput is 43.05 vs 26.49 tokens/s, i.e. 1.63× (Fig.10).

**Side results (tangential to us).**
- Security (Tab.3): indirect-injection success (StruQ-ID) falls from 76.00 to 42.23 (Qwen2.5-7B) and from 80.69 to 41.91 (Qwen3-4B). Gandalf barely moves (97.6→96.2).
- Monitorability (Tab.4): concern sub-vocalization rises from 0.07 to 0.42 (8B) and from 0.06 to 0.44 (27B).

## Ablations & lessons

- **(b)/(c) Cross-stream position encoding is a first-order design choice.** IFEval on Qwen3-4B (Tab.7, p.29), as Inst-L / Inst-S / Prompt-L / Prompt-S:

| Scheme | Inst-L | Inst-S | Prompt-L | Prompt-S |
|---|---|---|---|---|
| Ours (per-stream counter + stream embedding) | **60.19** | 47.72 | **49.72** | 35.86 |
| Rotate | 54.56 | **49.40** | 42.14 | **36.41** |
| Offset, d=128 | 54.84 | 45.33 | 41.82 | 30.65 |
| Offset, d=0 | 46.49 | 39.24 | 34.29 | 25.90 |
| 2D RoPE, Axial α=0.25 | 52.52 | 43.17 | 41.22 | 31.24 |
| 2D RoPE, Mixed α=1.0 | 47.60 | 38.25 | 36.23 | 26.80 |
| NoPE | 51.44 | 42.21 | 40.67 | 30.13 |

  - Ours wins the loose metrics by 5–15 points.
  - **Rotate wins both strict metrics.** This contradicts the text's claim that Ours "consistently outperforms all alternatives across all four metrics".
- **(c) Parallel input/output streams remove read-then-respond blocking** at little or no accuracy cost (Tab.1). The latency gain grows with parallelism (Tab.2).
- **(c) A parallel audit or thinking stream helps less on deep reasoning.** On MATH500, the authors attribute most of the audit-while-solving gain to fixing answer format rather than to real reflection (p.28).
- **(d) A small amount of data suffices to learn the format** (≤48k samples, 3 epochs; 3,864 conversations for 27B). The authors flag that the scale is "tiny" relative to modern post-training (p.10).
- Overclaim: the text says stream models are more monitorable on all four Table 4 metrics, but alignment-faking sub-vocalization at 27B drops (0.78 → 0.74).

## Implications for our setup

- **Position IDs.** Raon and MiniCPM interleave user audio, assistant text and assistant audio into one sequence, so positions grow with the interleaved token count.
  - This paper's evidence favours **shared time-indexed positions (one per 80 ms frame) per stream, plus stream embeddings.** In its text experiments this beat offset and 2D schemes by 5+ IFEval points.
  - **Hypothesis (untested for audio):** keeping each stream's RoPE distances "text-like" may reduce how much the backbone's knowledge degrades.
- **Per-stream tokens vs Moshi's summed embeddings.** Give user audio, assistant text/"thinking" and assistant audio their own tokens. Idle "-" slots cost no KV, so silence is cheap.
  - A thinking stream that plans while the user is still talking extends Raon's 1-frame lookahead and Moshi's inner monologue, cutting TNFT to 0.
- **Loss.** Average the CE per stream so sparse text/state tokens are not swamped by dense audio. Mask the user stream: predicting user audio is a deliberate choice, and Raon and MiniCPM never do it.
- **Order within a row.** Same-row leakage means input streams (user audio) should come before output streams. Decide explicitly whether the assistant may see the concurrent user frame.
- **Synthetic data hygiene.** Add an LLM-judge causal check (assistant content at time t uses only user speech heard before t). Generate dialogues as row-wise tables. Wait-k bridging utterances are the text analogue of fillers and backchannels.
- **Diagnosis (inference).** The duplex *format* is cheap to learn and need not hurt the LLM (Tab.1, Tab.9). So our knowledge loss more likely comes from the speech-token modality gap and the data mixture than from the stream structure itself.
- **Hybrid backbones.** With hybrid backbones (Gated DeltaNet), keep per-stream recurrent states.

## Open questions / unclear points

- No speech or audio experiments. Rows are text tokens. Multi-codebook audio streams at a fixed frame rate, and streams with different token rates, are not addressed.
- No knowledge benchmarks. The models are small and the SFT sets tiny.
- Delay units are unstated, and TNFT is measured in tokens rather than seconds.
- The stream-contrastive loss is defined but has no reported results.
- Table inconsistency: for Qwen2.5-7B, Table 3's "Inst-L" column (31.98 / 32.90) matches Table 9's Prompt-S, not its Inst-L (54.92 / 54.43).
- Section 4 says "two settings" but lists three: Base, Vanilla, Ours (p.7).
