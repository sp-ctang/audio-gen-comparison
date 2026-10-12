# VoiceChat-TTS: low-latency continuous speech synthesis for interactive agents

| Field | Value |
|---|---|
| Org | NVIDIA (equal-contribution first authors E. Casanova and J. Kim; p.1) |
| Date / arXiv | arXiv:2608.13831v1, 13 Aug 2026 (p.1) |
| Category | TTS: a streaming, always-on, duplex-aware speech decoder |
| Backbone | 778M Gemma 3-based decoder-only streaming TTS module plus a 199M codec, 977M in total (p.2). It is built directly on the Audio Flamingo 3-Chat streaming speech decoder (p.2). Whether weights were initialized from AF3-Chat or from Gemma 3 is **not stated**. |
| Audio input | Text, not audio: Nemotron Nano 2 subword tokens from the upstream LLM plus BOS and interruption tokens, right-padded with PAD onto the 12.5 Hz acoustic timeline. A character-aware subword encoder embeds them. A 3-s reference audio prompt is prefilled as acoustic tokens (p.2). |
| Audio output | Codec tokens: 31-level RVQ at 12.5 Hz (80 ms per frame), 22 kHz audio. The model generates one frame per step. A Mixture-of-Gaussians head (MoGH) fills the RVQ levels by iterative unmasking, using 4–8 iterations (8 in the evaluations) instead of 31 sequential steps (p.2, p.4). |
| Waveform decoder | 199M fully causal convolutional codec. It caches inputs only within the receptive field, which allows persistent streaming (p.2). |
| Duplex mechanism | The model runs over the whole conversation timeline. PAD means generate silence, BOS starts an assistant turn, and the interruption token means stop and go silent. An optional forced silence frame can be injected. A mid-utterance interruption does not reset the KV cache. There is no user-audio input: turn-taking is decided upstream (p.1–2, p.5). |
| Open weights / code | Code in NVIDIA NeMo Speech. The checkpoint ships inside nvidia/NVIDIA-NemotronLabs-VoiceChat-11B on Hugging Face (p.2). |
| Relevance to us | **high**. It shows how to bolt a duplex-aware streaming speech decoder onto a text-emitting LLM, and how to train it. It measures directly against both Qwen3-TTS-12Hz-1.7B and the joint text+audio PersonaPlex. |

## TL;DR

- This is the speech decoder of NemotronLabs VoiceChat. The VoiceChat paper names it as its speech-generation component (VoiceChat p.4), and this paper confirms it is the decoder in VoiceChat-11B (p.6–7).
- It extends the AF3-Chat streaming decoder with BOS/PAD/interruption control tokens, silence generation with a continuous KV cache, a 3-s speaker prompt, a character-aware subword encoder for raw LLM tokens, and gated fusion for numerical stability (p.2).
- Training has two stages:
  1. 1.4M steps on about 70k hours of single-turn TTS data (8×A100).
  2. 200k steps on 32×H100 with a mixture: 40% single-turn TTS, 15% word/proper-name lists, 10% real conversation, 35% synthetic duplex. The synthetic duplex set is about 2.5k hours, half of it with overlap and barge-in (p.3–4).
- Results:
  - LibriTTS unseen-speaker WER 2.00%, versus 4.51% for AF3-Chat and 1.01% for Qwen3-TTS-12Hz.
  - Re-voicing PersonaPlex's own text cuts WER from 5.00% to 2.42%.
  - Interruptions obeyed within 320 ms in 96.8% of cases.
  - 9.62 ms per 80-ms frame (Tab.1–4).
- **No component ablations**; the authors say this outright (p.7).

## Architecture

**Codec (p.2).** A fully causal convolutional autoencoder like AF3-Chat's, retrained here for 22 kHz audio at 12.5 Hz with 31-codebook RVQ. The paper notes that this frame rate matches duplex S2S models such as SALM-Duplex and Moshi. The streaming optimizations are listed in the latency section (p.6).

**Text interface (p.2, Fig.1 p.3).**
- **Tokenizer:** the upstream LLM's own tokenizer (Nemotron Nano 2) plus BOS and interruption tokens. On LibriTTS test-clean it produces 4.16 text tokens per second of speech, roughly one token per three acoustic frames (p.2).
- **Padding:** each turn's text is right-padded with PAD to the length of its acoustic span.
- **Look-ahead:** text leads audio by one token, so the model has seen at least two subwords before it speaks the first.
- **Character-aware Subword Encoder (Fig.2):** each subword is split into characters, run through a shallow Transformer and average-pooled. This generalizes to LLM subwords that are rare or absent in TTS corpora.
- **Continuation embedding:** marks a subword that continues the current word, which helps pronounce words fragmented across LLM tokens.

**Acoustic generation (p.2).**
- A 778M Gemma-3-based decoder predicts one RVQ frame per step.
- The MoG head predicts continuous embeddings for the masked RVQ levels and quantizes them progressively.

**Speaker conditioning (p.2).** AF3-Chat inferred the speaker from preceding audio, which fails at conversation start and while the user talks (context is mostly silence). VoiceChat-TTS instead prefills a 3-s prompt with its loss masked.

**Boundary embeddings and gated fusion (p.2).**
- BOS and interruption get learnable text embeddings.
- A gated fusion of text and acoustic embeddings fixes mixed-precision overflow caused by large-magnitude RVQ embeddings in the base decoder.

**Interruption (p.5–6).** The model learns to go silent at an interruption token. Inference can also inject a fixed silence frame: the most frequent 31-token RVQ pattern from encoding a long stretch of silence.

**Relation to VoiceChat.** VoiceChat's 9B duplex STT LLM emits the text/BOS/PAD stream and this model speaks it. The two are trained separately with no shared gradients (VoiceChat p.5).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training

- The codec is trained for 22 kHz / 12.5 Hz / 31-level RVQ (p.2). Its training data, steps and compute are **not reported**.
- The text tokenizer is reused from Nemotron Nano 2 (p.2).

### Stage 1: single-turn TTS pretraining

- **Objective:** autoregressive acoustic-token prediction through the MoG head, with the prompt region loss-masked (p.2). The loss form and per-codebook weighting are not detailed.
- **Data:** single-turn TTS only. About 70,159 hours of English (p.3):
  - LibriTTS train-clean-360 and train-clean-100;
  - HiFiTTS;
  - a 36.7k-hour subset of HiFiTTS-2;
  - a proprietary 62-hour two-speaker set.
  - Listed separately as an addition: a 32k-hour internal set derived from Creative Commons YouTube.

  36.7k + 32k is already about 68.7k (derived), which suggests the 70,159 h figure includes the YouTube set, but the paper does not say so.
- **Silence augmentation** (the paper does not say which stage applies it): 50% of single-turn data gets 0.5–5.0 s of silence prepended, with text alignments shifted to match. The purpose is that the model cannot easily tell plain TTS apart from duplex contexts that start in a listening state (p.3).
- **Optimization:** 1.4M steps on 8 NVIDIA A100; AdamW, LR 4e-5 (p.4).
- **Not reported:** batch size for this stage, wall-clock time, LR schedule.

### Stage 2: duplex / multi-turn fine-tuning

- **Data mixture (p.4):**

  | Share | Data |
  |---|---|
  | 40% | single-turn TTS |
  | 15% | word and proper-name lists |
  | 10% | real conversational speech (Fisher + small internal set) |
  | 35% | synthetic duplex dialogues (≈2.5k h) |

  Single-turn and multi-turn samples are formatted identically: text right-padded to each turn's length.
- **Batching:** Lhotse dynamic bucketing, about 240 s of audio per batch. Whether that is per GPU or global is not stated.
  - If global: 200k × 240 s ≈ 13.3k hours of audio seen.
  - If per GPU (×32): ≈ 427k hours.
  - Both figures are derived.
- **Optimization:** 200k steps on 32 NVIDIA H100; AdamW, LR 4e-5 (p.4).
- No RL or preference stage. (Koel-TTS, one of the engines used to synthesize the training data, uses preference alignment per its reference title, but that is a different model.)

## Data pipeline

**Synthetic multi-turn set, about 2.5k hours (p.3–4).**
- **Scripts:** millions of dialogue scripts written by open-source LLMs with NeMo Data Designer, prompts tuned for topic and situation coverage. Up to 15 turns each, with LLM-generated paralinguistic annotations.
- **Rendering:** a Chatterbox + Koel-TTS pipeline, with speakers drawn from the LibriTTS training sets.
- **Duplex events:** 50% of the set simulates overlapping speech and user barge-in, which gives aligned interruption control tokens.
- **Hard-word dialogues:** an extra multi-turn subset where each turn is four distinct difficult words or proper names, for phonetic coverage.

**Real conversation (p.4).**
- Unscripted Fisher plus a small internal conversational set. Hours are **not reported**.
- These supply natural transitions, spontaneous backchannels and turn-taking, and include untranscribed laughter, sighs and backchannels that the model must handle robustly.

**Evaluation sets (p.4–5).** LibriTTS test-clean (unseen speakers) and 5 training speakers (seen). Multi-turn tests group same-speaker utterances. Interruption tests use the 200 FDB User-Interruption items with controlled assistant text.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Codec | not reported | not reported | 199M, 22 kHz, 12.5 Hz, 31 RVQ |
| Single-turn pretraining | ≈70.2k h | 8× A100, 1.4M steps; wall-clock not reported | LR 4e-5 |
| Duplex fine-tuning | ≈240 s audio per batch; mixture 40/15/10/35 | 32× H100, 200k steps | audio seen 13k–427k h (derived range) |
| Inference | — | RTX A6000: 9.62 ms per 80-ms frame at concurrency 1, 15.24 ms at concurrency 4 | vLLM-Omni |

## Preserving text intelligence / knowledge

Not a goal here, since the TTS holds no knowledge. Its role is architectural: it lets VoiceChat's LLM emit only text, consuming the LLM's native subwords with no re-tokenization (p.2). The paper's motivation is that end-to-end duplex models produce worse speech because one network must jointly optimize ASR, interruption handling and synthesis (p.1).

## Key results

**TTS quality, LibriTTS** (Tab.1, p.5). Mean ± 95% CI over 10 runs. Top-p 0.95, CFG 0.2, MoG noise 0.001, 8 MoGH iterations (p.4).

| System | CER% | WER% | SECS | SQuIM-MOS |
|---|---|---|---|---|
| Ground truth | 0.48 | 1.40 | 0.830 | 4.457 |
| Chatterbox-TTS (offline) | 0.45 | 1.24 | 0.887 | 4.27 |
| Qwen3-TTS-12Hz-1.7B (streaming) | 0.34 | 1.01 | 0.827 | 4.45 |
| AF3-Chat decoder (streaming) | 2.85 | 4.51 | 0.761 | 3.60 |
| VoiceChat-TTS, turn 1 (unseen) | 1.00 | 2.00 | 0.757 | 4.380 |
| VoiceChat-TTS, turn 4 (unseen) | 1.20 | 2.20 | 0.685 | 4.376 |
| VoiceChat-TTS, turn 1 / 4 (seen) | 1.20 / 0.80 | 2.40 / 1.80 | 0.785 / 0.778 | 4.365 / 4.359 |

**Same text, two decoders** (Tab.2, p.5). Full-Duplex-Bench Smooth Turn-Taking subset, PersonaPlex's own text and timing, scored on the full untrimmed output, so speech during user intervals counts as insertions.

| Decoder | CER | WER | SQuIM-MOS |
|---|---|---|---|
| PersonaPlex audio | 4.06 | 5.00 | 4.094 |
| VoiceChat-TTS | 2.05 | 2.42 | 4.292 |

**Interruption** (Tab.3, p.6). 200 FDB-timed examples.

| Setting | IOR@320ms | Mean stop latency | Leakage@1s | After-interruption CER |
|---|---|---|---|---|
| Learned behaviour only | 96.8% | 228.3 ms | 169.1 ms | 0.255% |
| Forced silence | 100% | 89.9 ms | 55.8 ms | 0.095% |

**Latency on RTX A6000** (Tab.4, p.6). Next-frame total = acoustic inter-token latency + codec.

| Concurrency | VoiceChat-TTS (977M) | Qwen3-TTS-12Hz (1.7B) |
|---|---|---|
| 1 | 7.16 + 2.46 = 9.62 ms | 15.44 + 4.90 = 20.34 ms |
| 4 | 15.24 ms | 22.08 ms |

Both are far below the 80-ms real-time budget.

## Ablations & lessons

The authors state there are no controlled component-wise ablations. They say only that preliminary experiments suggested the gains were cumulative (p.7). Every lesson below is therefore a system-level comparison or a stated design rationale.

- **A dedicated decoder beats joint text+audio generation on intelligibility.**
  - With PersonaPlex's text held fixed, WER drops from 5.00 to 2.42 and SQuIM-MOS rises from 4.094 to 4.292 (Tab.2).
  - Because the output is untrimmed, this also shows little speech leaking into intervals that should be silent.
  - This is the cleanest evidence in the batch that a Moshi-style model's audio stream, not its text, limits intelligibility.
- **The duplex extensions improve quality over AF3-Chat rather than hurting it.** WER 4.51 → 2.00 and MOS 3.60 → 4.38 at similar SECS (0.761 vs 0.757) (Tab.1). This is confounded with the new data, the codec and longer training.
- **Duplex capability still costs single-turn quality.** Qwen3-TTS-12Hz-1.7B beats VoiceChat-TTS on every isolated-response metric (WER 1.01 vs 2.00, SECS 0.827 vs 0.757) (Tab.1). The authors argue those models lack silence, interruption and continuous-state behavior.
- **Content stays stable over turns, but a zero-shot voice drifts.** Over 4 turns WER stays at 1.90–2.20 and MOS at about 4.38, but unseen-speaker SECS falls 0.757 → 0.710 → 0.696 → 0.685. Seen speakers stay at 0.778–0.789 (Tab.1). The authors link the drift to alternating speech and silence (p.4).
- **Learned interruption is reliable; a deterministic silence frame makes it strict.** IOR@320ms 96.8 → 100%, stop latency 228 → 90 ms, leakage 169 → 56 ms, with no harm to recovery: after-interruption CER 0.255 → 0.095% (Tab.3).
- **Silence prepending** (see Stage 1) is justified by rationale only, with no numbers (p.3).
- **Mixing 40% single-turn TTS into duplex fine-tuning**, formatted identically, is how the authors keep acoustic quality (p.4). Again a rationale only.
- **Mixed-precision warning:** gated fusion was needed because large RVQ embeddings overflowed (p.2). This matters for any model feeding summed RVQ embeddings into a bf16 LLM.
- **Latency:** at a similar frame rate, the smaller 977M model with MoGH and optimized kernels is about 2.1× faster than Qwen3-TTS-12Hz (Tab.4).

## Implications for our setup

Our setup: Qwen3 LLM backbone, Qwen3 audio tokenizer and its vocoder/decoder, mid-trained on about 10K hours of duplex conversational speech. Problems: weak audio continuation and weak semantic knowledge in the generated speech.

- **If our duplex LM's audio continuation is the weak link, decouple it.**
  - Have the Qwen3 LM emit text plus BOS/PAD/interrupt control tokens on the frame grid.
  - Train a duplex streaming TTS on the Qwen3 tokenizer to speak them.
  - Qwen3-TTS-12Hz is a strong starting point (12.5 Hz, WER 1.01). What it lacks is exactly what this paper adds: silence under PAD, BOS/interruption tokens, continuous operation without cache resets, and a speaker prompt.
  - The adaptation budget here is modest: 200k steps on 32 H100s, with about 2.5k hours of synthetic duplex data plus real conversation.
- **If we keep audio generation inside the LM, borrow the data tricks:**
  - Anchor acoustic quality with clean single-turn TTS data (40% of their mix).
  - Prepend random silence to single-turn data.
  - Include hard-word and proper-name drills (15%).
  - Use an explicit speaker prompt rather than relying on context, which goes silent while the user speaks.
- **Measure the decoupling gap on our model**, as in their Table 2: resynthesize our model's own text channel with a strong TTS and compare WER and MOS against our native audio. If the gap is large, as with PersonaPlex (5.00 vs 2.42), the bottleneck is acoustic modelling capacity or data, not knowledge.
- **Multi-codebook prediction:** the MoGH (4–8 iterative unmasking/refinement steps covering 31 RVQ levels) is an alternative to a depth transformer or 31 sequential predictions when per-frame latency matters.
- **Eval protocol to copy:** LibriTTS WER/CER (Parakeet-TDT), SECS (TitaNet-Large) and SQuIM-MOS; multi-turn SECS drift; FDB-timed interruption metrics (IOR@320ms, stop latency, Leakage@1s, after-interruption CER); per-frame latency against the 80-ms budget.
- **Trade-off to remember:** this decoder ignores user audio (limitation, p.7), so prosody cannot adapt to the user. An integrated model could in principle do better if its audio quality were fixed.

## Open questions / unclear points

- **Initialization:** were the decoder weights initialized from AF3-Chat, from Gemma 3, or from scratch? The paper only says it is built on AF3-Chat's decoder.
- **Codec training:** data and compute are not reported.
- **Data accounting:** is the 32k-hour YouTube set inside the 70,159-hour total? How many hours do Fisher and the internal real set contribute?
- **Batching:** is the 240 s batch per GPU or global? What is the Stage-1 batch size?
- **Training-time details:** CFG is used at inference (scale 0.2), but conditioning dropout during training is not described. The loss weighting across RVQ levels is also not described.
- **Overlap rendering:** how are overlaps placed in the synthetic duplex set (offset distribution, timing of the interruption token relative to user onset)?
- **Interface to VoiceChat:** how does VoiceChat's EOS map to this model's interruption token? Neither paper states it.
- **Evaluation scope:** quality is measured only with SQuIM-MOS (no human MOS) and on read-speech LibriTTS, not on conversational output from the full VoiceChat system.
