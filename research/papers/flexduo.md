# FlexDuo: A Pluggable System for Enabling Full-Duplex Capabilities in Speech Dialogue Systems

| Field | Value |
|---|---|
| Org | Kuaishou Technology (p.1) |
| Date / arXiv | arXiv 2502.13472v2, 29 May 2025 (p.1) |
| Category | method: a pluggable full-duplex controller for half-duplex spoken dialogue systems |
| Backbone | Controller: Qwen2-Audio-7B-Instruct, with the audio encoder frozen and the LLM fine-tuned (p.6–7). Dialogue model: any half-duplex system; GLM-4-Voice and MiniCPM-o were tested, both unchanged (p.8) |
| Audio input | Qwen2-Audio encoder over audio chunks, evaluated every 120 ms on a sliding window (p.4–5). Frame rate not reported |
| Audio output | n/a. Speech is produced by the plugged-in half-duplex model |
| Waveform decoder | n/a |
| Duplex mechanism | External state controller: every 120 ms, 1 of 7 actions over Speak / Listen / Idle; Idle filters noise, backchannels and third-party speech from the context (p.4–5) |
| Open weights / code | Not stated |
| Relevance to us | medium-low: not native duplex, but the Fisher labelling pipeline and Idle-state evidence are reusable |

## TL;DR

- Decouples duplex *control* from dialogue *generation*. The 7B audio-LLM controller emits one action token per 120 ms. The half-duplex S2S model only sees filtered audio plus control signals (p.3–4).
- **The new Idle state** absorbs noise, user backchannels and third-party speech. Without it, these would trigger replies or pollute the context (p.2, p.4).
- **Training data:** Fisher English 671 h + Chinese 263 h, auto-labelled with VAD (IPUs, overlap-based backchannel detection) and GPT-4o filtering of third-party speech (p.5).
- **English Fisher, combined scores:**
  - FlexDuo: turn-taking F1 0.79, false-interruption rate 0.30.
  - VAD-controlled half-duplex: 0.32 / 0.53.
  - Freeze-Omni: 0.71 / 0.61.

  (p.7, Tab.2)
- **Dialogue quality improves:** conditional perplexity 28.94 vs 64.32 for a VAD-gated GLM-4-Voice (p.9, Tab.5).

## Architecture

**Components** (p.3–5, Fig.2):
- **Context manager:** stores the dialogue history as text, as shown in Fig.2. It is updated when the state leaves Listen (the user's turn ended) and when it leaves Speak (the assistant's turn ended) (p.4–5).
- **State manager:** predicts π_t = F(C, S_{t−1}, W_t) from the history C, the previous state, and the sliding window of audio (p.4, Eq.1).

**The 7 actions** (p.4):

| Action | Meaning |
|---|---|
| K.S (keep speaking) | assistant keeps speaking |
| K.L (keep listening) | user is still speaking |
| K.I (keep idling) | noise, user backchannel, or third-party speech |
| S2L (speak → listen) | user interrupts |
| S2I (speak → idle) | assistant finishes normally |
| L2S (listen → speak) | assistant responds or interrupts |
| I2L (idle → listen) | user starts speaking |

**Finite-state machine edges** (Fig.2): I→L, L↔S, S→I, plus self-loops on each state.

**Control semantics** (p.4):
- Speak: the half-duplex model generates a reply.
- Listen: user audio is passed to the half-duplex model for prefilling.
- Idle / Listen: the model's ongoing output is interrupted.
- Idle: incoming user audio is dropped from the context.

**Sliding window** (p.5, Eq.2):
- In Listen it keeps growing.
- Otherwise it is a fixed window of w=4 chunks. Derived: 480 ms, if one chunk equals the 120 ms decision interval.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
n/a. The audio encoder is Qwen2-Audio's own, kept frozen.

### Stage 1: controller fine-tuning
- **Model:** Qwen2-Audio-7B-Instruct. The audio encoder is frozen and the LLM is trained (p.6). Special action tokens are added to the vocabulary.
- **Input / output:** the input is the history, the current window's user audio, and the current state. The output is one action token (p.6).
- **Loss:** cross-entropy, masked on the history and the audio input (p.7).
- **Data:**
  - Fisher 831 h English + 389 h Chinese, split 10:1:1 (p.7).
  - The "final dataset" is 671 h English + 263 h Chinese, with 4,792,219 and 3,259,022 action labels respectively (p.5–6, Tab.1).
- **Optimizer:** AdamW, LR 1e-5, batch 3 per GPU (global 24, derived), 40,000 steps, 500 warmup steps.
- **Hardware:** 8×H800 with DeepSpeed ZeRO-3 (p.7).

## Data pipeline

**Labelling Fisher automatically** (p.5–6, Fig.3):
- **Segmenting:** VAD on each speaker channel gives inter-pausal units (IPUs). IPUs with a gap under 160 ms are merged.
- **Backchannels:** an IPU *completely overlapped* by the other speaker's speech is labelled a backchannel (no reply, no interruption). Every other IPU is a valid turn.
- **Third-party speech:** each turn's IPUs are transcribed, and GPT-4o judges, given the preceding context, which segments are irrelevant (prompt in App. A.1, p.12). Human check on 100 samples: 84% acceptance.
- **Roles:** one speaker is randomly made "user", the other "assistant". Turns are ordered by VAD timestamps.

**Labelling policy** (p.5–6):
- The assistant takes the turn strictly *after* the user's turn ends.
- The user's real interruption timing is kept as recorded.
- The "listening phase" starts 500 ms *after* the user begins speaking. This gives the controller time to judge semantically whether the speech is a backchannel or an interruption.

**Label distribution** (Tab.1, p.6):

| Label | English | Chinese |
|---|---|---|
| K.L (keep listening) | 59.37% | 52.26% |
| L2S (listen → speak) | 15.83% | 18.38% |
| I2L (idle → listen) | 10.29% | 11.98% |
| K.S (keep speaking) | 9.07% | 10.78% |
| K.I (keep idling) | 4.98% | 6.02% |
| S2L (speak → listen) | 0.47% | 0.57% |

S2I does not appear in the table.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Controller fine-tuning | 671 h EN + 263 h ZH Fisher; 8.05M labels | 8×H800, 40k steps; wall time not reported | LR 1e-5, batch 3/GPU, ZeRO-3 |

## Preserving text intelligence / knowledge

- **By construction:** the half-duplex dialogue model is never modified, so its knowledge is untouched (p.2–3, p.9).
- **Cleaner context helps content:** the Idle state keeps irrelevant audio out of the context, which measurably improves response quality (Tab.5).
- **Limitation:** the half-duplex model cannot produce backchannels (p.14).

## Key results

**Interaction on English Fisher** (Tab.2, p.7). Turn-taking is F1@1, false interruption rate (FIR) is lower-is-better; values are combined (assistant / user):

| System | Turn-taking F1@1 | FIR |
|---|---|---|
| FlexDuo + GLM-4-Voice / MiniCPM-o | 0.79 (0.68 / 0.89) | 0.30 (0.35 / 0.25) |
| Same models, VAD-controlled | 0.32 (0.0 / 0.64) | 0.53 (0.58 / 0.49) |
| Freeze-Omni | 0.71 | 0.61 (0.72 / 0.49) |
| MinMo | 0.54 | not reported |
| Moshi | 0.27 | assistant FIR 0.37 only |

**Dialogue quality** (Tab.5, p.9):
- Setup: GLM-4-Voice; ASR transcripts scored by Qwen2.5-1.5B-Instruct, given 20 turns of history.
- Conditional perplexity: English 28.94 (FlexDuo) vs 64.32 (VAD); Chinese 44.45 vs 63.51.
- The "35.3%" and "19%" reductions quoted in the text are *absolute* perplexity differences (derived check).

**Latency** (Tab.6, p.9): 343.47 ms vs a 500 ms VAD threshold, i.e. 156 ms faster.

## Ablations & lessons

- **An explicit Idle state matters** (Tab.3, p.8–9). Removing it:

| | With Idle | Without Idle |
|---|---|---|
| Combined turn-taking F1 | 0.79 | 0.63 |
| Combined FIR | 0.30 | 0.44 |
| User F1@1 | 0.89 | 0.64 |
| User FIR | 0.25 | 0.49 |

  The biggest effect is on the user side.
- **Window size** (Tab.4, p.8):
  - w=4 has the best combined F1 (0.79); w=8 gets 0.68.
  - w=8 has the lowest user FIR (0.16) but the worst assistant FIR (0.42).
  - w=2's combined score is printed as 0.34, which is inconsistent with its F1s (0.60 / 0.87, mean ≈0.74). Likely a typo.
  - Takeaway: short windows are faster but understand less; long windows understand more but add latency (p.9).
- **There is an inherent trade-off.** An aggressive assistant scores better on assistant turn-taking but makes more false interruptions; a conservative one does the reverse (p.7).
- **Semantic endpointing beats a fixed-silence VAD** on both latency (343 vs 500 ms) and context quality, even though the training labels were themselves derived from VAD. The labels are computed offline with global context (p.8).

## Implications for our setup

- **Reuse the labelling pipeline** on our 10K h duplex corpus: IPU merge under 160 ms, fully-overlapped IPU = backchannel, LLM filter for third-party speech. Uses:
  - per-frame turn-state labels as auxiliary targets or as diagnostics;
  - finding and removing third-party speech and noise segments, which teach a model to respond to irrelevant audio.
- **Context pollution measurably hurts content**: conditional perplexity 64 → 29 when filtered. In a native duplex model, user backchannels and noise also sit in the context. Worth checking whether our weak semantics is worse on backchannel-heavy or noisy conversations.
- **Their policy choices can be imposed when curating assistant-channel data:** the assistant speaks only after the user's turn ends, and the "listening" decision is delayed 500 ms after user onset.
- **Fallback architecture:** a pluggable controller on top of a strong half-duplex Qwen3 speech-to-speech model keeps knowledge intact. The price is no backchannels or overlap from the assistant (p.14).

## Open questions / unclear points

- The Fisher hours do not match: 831 / 389 h with a 10:1:1 split (p.7) vs 671 / 263 h "final" (p.5).
- S2I is missing from Tab.1, and the number of decisions per training sample is not given. 40k steps × 24 ≈ 0.96M samples vs 8.05M labels (derived).
- How user turns are transcribed into the text context at inference.
- No human evaluation, and no test beyond telephone-domain Fisher.
