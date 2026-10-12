# Raon-Speech & Raon-SpeechChat (KRAFTON technical report)

| Field | Value |
|---|---|
| Org | KRAFTON |
| Date / arXiv | Report dated 2026-05-26; arXiv 2605.23912v1 (p.1) |
| Category | half-duplex S2S (Raon-Speech) + full-duplex S2S (Raon-SpeechChat) |
| Backbone | 8.2B LLM (36 layers, hidden 4096) initialized from **Qwen3-VL-8B-Instruct**. Full model is 9.1B (Speech) / 9.8B (SpeechChat) (p.3, Tab.7–8) |
| Audio input | Speech: AuT encoder (from Qwen3-ASR; non-causal, 24 layers, 319M) at 12.5 Hz, then a 2-layer GELU MLP adaptor + RMSNorm (weight init 0.02). SpeechChat: causal Voxtral-Mini-4B-Realtime-2602 encoder (32 layers, sliding-window attention with a 15 s window, 4-frame stacking, 970M) (p.3–4, Tab.8) |
| Audio output | Mimi RVQ codes at 12.5 Hz, using the first 16 of 32 codebooks (size 2048): 1 semantic + 15 acoustic codes per frame. A 4-layer "speech generation expert" (hidden 2048) reads LLM hidden states and predicts the semantic code. A 5-layer residual code predictor (RCP, i.e. a depth transformer, hidden 1024, initialized from Qwen3-Omni-30B-A3B-Instruct) predicts the 15 acoustic codes (p.3–4, Fig.2, Tab.8) |
| Waveform decoder | Mimi decoder (96M, frozen; causal sliding-window attention with a 10 s window) (Tab.7–8) |
| Duplex mechanism | A single autoregressive sequence. Per 80 ms frame it interleaves: user-speech embedding, an assistant text-channel token (SIL / BOW / word / PAD / BC), and the assistant speech embedding. Text and speech are aligned per word, with PAD filling the gap; 1-frame text lookahead. Explicitly *not* Moshi-style parallel streams (p.4–5, p.8) |
| Open weights / code | Yes: checkpoints, training and inference pipeline, demo (p.1); github.com/krafton-ai/Raon-Speech |
| Relevance to us | **high**: Qwen3-family 8B backbone, a Qwen3-Omni-initialized code predictor, Qwen3-TTS synthetic data. Explicit open recipe for keeping knowledge and then adding duplex by continual training, with hyperparameters for every stage |

## TL;DR

- The authors first build a half-duplex SpeechLM on 1.01M h / 69.98B tokens of pretraining data. The duplex model then comes from continual training on 119.5K h of time-aligned dialogue (derived: 105.69K + 13.85K) (Tab.3).
- The abstract says the model "successfully transforms a pre-trained LLM into a SpeechLM … while preserving strong text capabilities" (p.1). The recipe behind this has three parts:
  1. Frozen-LLM alignment, done separately for the input adaptor and for a dedicated speech-generation stack.
  2. End-to-end pretraining with CE plus **on-policy KL from two teachers**: the model given the transcript (for audio inputs), and the original backbone (for text inputs). Backbone-generated TextQA targets are also used.
  3. 800 steps of SFT + SimPO (p.6–7).
- Text retention: MMLU-Pro 64.05 and MMLU-Redux 78.87, the best of the 9 audio models compared (Tab.4). The backbone's own scores are not reported.
- Duplex:
  - SIL/BOW/PAD/BC state tokens separate *when to speak* from *what to say*.
  - 10% of the SpeechLM data is replayed during duplex training.
  - 89% of the duplex hours are synthetic (derived), generated as Qwen3-LLM dialogues voiced by Qwen3-TTS (p.5, p.8, p.10).
- Results:
  - Best on FDB v1.0 turn-taking and interruption metrics (internal evaluator).
  - Behind MiniCPM-o 4.5 and PersonaPlex on FDB v2.0 multi-turn content.
  - Lowest interruption-response relevance (Tab.6).
- There are **no ablation tables**. The "what matters" evidence is qualitative.

## Architecture

**Understanding (p.3).** Encoder (12.5 Hz) → randomly initialized 2-layer MLP adaptor → RMSNorm with weight 0.02. The RMSNorm makes speech embedding norms match text embedding norms at initialization. The result is fed through the LLM's normal input interface.

**Generation (p.3–4, Tab.8).**
- The LLM keeps its text head.
- A projector (MLP 4096→6144→2048) feeds its hidden states to the **speech generation expert**, a 4-layer decoder-only Transformer (205M including its head). The expert predicts the semantic code.
- The 15-depth RCP (146M) then predicts the acoustic codes, one residual level at a time.
- The dequantized codes are summed into a codec embedding. This embedding goes both to the Mimi decoder and back through the output adaptor (MLP 512→4096 + RMSNorm 0.02) as the next LLM input.
- Speaker identity comes from a frozen ECAPA-TDNN embedding (192→4096 projection) of a random 2–8 s chunk of target speech (10 s for SpeechChat), inserted into the sequence.
- In Raon-Speech, speech output appears only as TTS (text in the prompt → `<audio>`). The QA tasks output text (Tab.1). No speech-to-speech answer mode is described.

**Full-duplex (p.4–5, Fig.3).** Three changes:
1. A causal encoder.
2. One interleaved sequence, with assistant text and speech aligned per word; PAD fills frames where text runs out before the speech.
3. Explicit states:
   - **SIL** = silent listening. A speech-onset token switches the model to speaking; in Fig.3 this onset is BOW. Predicting SIL returns it to listening.
   - **BOW** precedes every assistant word. It has no lexical content and separates "decide to speak" from "what to say".
   - **BC** marks backchannels, which can be disabled or rate-controlled at inference.

**Text lookahead.** Text is emitted ahead of speech.

**Interleaving layout.** In Fig.3, each frame has a user embedding followed by an assistant-speech embedding. BOW or word tokens are inserted between them only when present; SIL and PAD appear only as outputs. My reading is that the text channel takes a sequence slot only when it carries content; the text does not state this. 4,096 tokens correspond to about 2 minutes (p.8).

## Training pipeline

All stages use AdamW. Token counts are upper bounds (steps × batch × length, assuming full packing).

### Stage 0: tokenizer / codec / decoder training
- None. Mimi is pretrained and frozen throughout, with 16 of its 32 RVQ levels kept.
- The AuT and Voxtral encoders are pretrained; ECAPA-TDNN is frozen.
- The RCP is warm-started from Qwen3-Omni-30B-A3B-Instruct to speed up convergence (p.4, p.6–7).

### Stage 1: Speech understanding alignment
- **Objective:** CE.
- **Trainable:** the input adaptor only (p.6).
- **Data:** STT, SpeechQA and SpokenQA in English and Korean, in 8 s chunks. Hours not reported.
- **Augmentation:** on-the-fly, Moshi-style: noise (including DNS-Challenge) at SNR −30 to 6 dB, reverb, channel filtering, bandwidth degradation.
- **Optimization:** LR 1.5e-4 → 1.5e-5 (cosine), 950 warmup, 13.5k steps, batch 128, length 24,576 (Tab.2).
- **Tokens:** ≤42.5B (derived).

### Stage 2: Speech generation alignment
- **Trainable:** output adaptor, expert and RCP. LLM and codec are frozen. Speaker-embedding dropout 0.2 (p.6).
- **Data:** English and Korean TTS. Hours not reported.
- **Optimization:** LR 1e-4 → 1e-5, 6,550 warmup, **65.5k steps**, batch 64, length 8,192.
- **Tokens:** ≤34.3B (derived).

### Stage 3: End-to-end pretraining with knowledge distillation
- **Objective:** CE on all five tasks (STT, TTS, SpeechQA, SpokenQA, TextQA), plus an equally weighted KL on SpeechQA, SpokenQA and TextQA (p.7). Teachers:
  - Audio inputs: **the same model conditioned on the transcript** (self-distillation).
  - Text inputs: **the backbone before pretraining**.
  - The KL is computed **on-policy**, on student-generated trajectories.
- **TextQA targets:** generated by the backbone, so that they match its original output distribution.
- **Trainable:** all modules except the encoder, codec and speaker encoder. Freezing these is reported to train better and more stably than tuning them.
- **Data:** 1.01M h / 69.98B tokens / 404.76M samples (Tab.3). **Text-only share not reported.**
- **Optimization:** LR 1.2e-5 → 1e-7, 4,000 warmup, 60k steps, batch 112, length 8,192.
- **Tokens:** ≤55.1B (derived), about 0.79 epoch.

### Stage 4: Post-training (SFT + SimPO)
- **Objective:** SFT on all tasks. SimPO (reference-free) on STT, SpeechQA, SpokenQA and TextQA (p.7).
- **Preference pairs:**
  - Offline: curated pairs against repetition.
  - Online: the chosen answer is the SFT ground truth; the rejected answer is the model's own output scored lower by a task reward (a verifier, or an LLM judge otherwise).
  - A pair is dropped if the judge prefers the model's output.
- **Data:** 104.07K h / 11.19B tokens / 23.36M samples (Tab.3).
- **Optimization:** constant LR 1e-6, 112 warmup, **800 steps**, batch 96, length 8,192.
- **Tokens:** ≤0.63B (derived), i.e. ≤6% of the listed data.
- **Outcome:** less repetition and higher task reward scores across tasks; no numbers.

### Stage 5: Causal adaptation (SpeechChat)
- **Setup:** AuT is swapped for Voxtral; CE only. Encoder and codec use native sliding windows (chunk sizes 30 s / 12 s) (p.7).
- **(a) Alignment:** LR 1.5e-4 → 1.5e-5, 700 warmup, 10k steps, batch 128, length 12,288, 30 s audio cap. ≤15.7B tokens (derived).
- **(b) Full training:** LR 2e-6, 10k steps, batch 160, length 8,192, with "expanded" data. ≤13.1B tokens (derived).
- **Data:** the Raon-Speech tasks.
- **Trainable modules:** given only as the same two-stage scheme as Raon-Speech.

### Stage 6: Full-duplex pretraining
- **Objective:** CE with loss weights **PAD 0.75, SIL 0.5** (p.8).
- **Data:**
  - Duplex data plus **10% of the Raon-Speech pretraining data**, to limit forgetting.
  - Duplex corpus: 105.69K h / 10.85B tokens / 4.38M samples (Tab.3).
  - Encoders run on full streams.
  - One-frame text lookahead: semantic codes are conditioned on text generated one frame earlier.
- **Optimization:** LR 5e-6 → 5e-7, 1,250 warmup (5%), 25k steps, batch 128, length 4,096 (about 2 min).
- **Tokens:** ≤13.1B (derived), about 1.2 epochs (about 107K h derived from 3.2M two-minute sequences).
- **Trainable modules:** not reported.

### Stage 7: Full-duplex fine-tuning I and II
- **Stage I:** high-quality conversations, for turn-taking, timing and fluency (p.8).
- **Stage II:** synthetic scenario data covering personas, safety, self-repair, interruptions and backchannel variants:
  - Persona and context are given as a system prompt (PersonaPlex-style).
  - The **assistant speaker is fixed** during synthesis.
  - BOW→**BC** for backchannels; BC is initialized from BOW, and its CE is weighted **×50**.
- **Data (both stages):** 13.85K h / 1.43B tokens / 595.57K samples (Tab.3).
- **Optimization (each stage):** 5k steps, LR 3e-6 → 3e-7, 5% warmup, batch 64, length 4,096, SIL weight 0.25. ≤1.31B tokens per stage (derived).

## Data pipeline

**SpeechLM data (p.9).** The raw pool is 1.38M h / 124.36B tokens (Tab.3). Sources:
- **Audio-text** (public and in-house): formatted as STT and as TTS. An LLM generates SpeechQA from transcripts and metadata.
- **Audio-only:** pseudo-labelled with Whisper.
- **Text-only:** reading comprehension, commonsense and instruction corpora, used as TextQA or turned into SpeechQA with Qwen3-TTS.

Processing:
- Normalization (neural punctuation and capitalization).
- Filtering (STT error rate, forced-alignment checks, perceptual quality, degenerate LLM outputs).
- Relabelling (Whisper; LLM-rewritten answers).
- Rebalancing (by domain, task and QA format).

**Full-duplex data (p.10).**
- **Real:** 13.21K h of public and in-house conversation, with natural turn-taking, fillers and overlap.
- **Synthetic:** 106.33K h, comprising:
  1. **SpeechLM samples reconstructed into duplex form, to keep the model's prior knowledge**.
  2. Curated scenarios: task-oriented, open-domain, safety and general. These include context-free multi-turn dialogues and returns to earlier topics, for robustness to topic shifts.

Processing:
- Diarization, because most real data is single-channel; target speech is extracted from overlapped regions.
- Forced-aligner word timestamps.
- Filtering: misaligned or non-overlapping backchannels, failed TTS, audio too long for its transcript, and clustered duplicates.

**Synthetic generator (App. D, p.21–22).**
1. **Dialogues.** A Qwen3-based LLM writes dialogues from 15 task scenarios × 3 levels of prompt specificity, plus open-domain chat and 7 speech games (some of which induce overlap). It mixes direct and clarification flows. Utterances are tagged as speech, backchannel, interrupt or simultaneous.
2. **Timelines.** Dual-channel timelines are built at 24 kHz with sample-accurate timestamps and voiced by speaker-conditioned **Qwen3-TTS**.
3. **Timing.** An audio-based backchannel predictor re-times backchannels, drops low-confidence ones and inserts new ones. Overlap and barge-in timing is rule-based.
4. **Barge-in truncation.** The assistant audio is cut at a random point, and the transcript is cut at the nearest forced-aligned word boundary.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Understanding alignment | ≤42.5B tok (derived) | not reported | adaptor only |
| Generation alignment | ≤34.3B tok (derived) | not reported | LLM frozen, TTS |
| Pretraining + KD | ≤55.1B tok (derived) of a 69.98B-tok / 1.01M-h corpus | not reported | CE + on-policy KL |
| Post-training | ≤0.63B tok (derived) of 11.19B / 104.07K h | not reported | SFT + SimPO |
| Causal alignment + full | ≤15.7B + ≤13.1B tok (derived) | not reported | encoder swap |
| FD pretraining | ≤13.1B tok (derived) of 10.85B / 105.69K h | not reported | +10% replay |
| FD fine-tuning I + II | ≤1.31B + ≤1.31B tok (derived) of 1.43B / 13.85K h | not reported | BC ×50 |
| **Totals** | about 132.5B (Speech) + about 44.6B (Chat) tok (derived upper bounds) | GPU type, count and hours not reported | the duplex-specific part is ≈15.7B tok |

## Preserving text intelligence / knowledge

Here is what the paper credits; none of it is ablated:
1. **On-policy KL to the frozen backbone on text inputs, plus backbone-generated TextQA targets.** Together these keep the output distribution unchanged; the paper credits the backbone teacher with mitigating catastrophic forgetting (p.7).
2. **Self-distillation from the transcript-conditioned model on audio inputs.** This transfers text-input behaviour to speech input (p.7).
3. **A frozen LLM in both alignment stages** (p.6), then a low LR of 1.2e-5 with the encoder side frozen (p.7).
4. **Architectural separation.** The LLM head emits text; the expert + RCP emit codes (p.3–4).
5. **In the duplex stage:** 10% SpeechLM replay and duplex reconstructions of SpeechLM data (p.8, p.10).

Evidence:
- MMLU-Pro 64.05 vs the next best, Fun-Audio-Chat at 61.12 (MiniCPM-o 4.5: 55.20).
- MMLU-Redux 78.87 vs 74.70 / 72.53 (Tab.4).
- KMMLU-Pro 46.85 / KMMLU-Redux 51.80 are the best (Tab.5).

**Caveats:**
- The backbone's scores are not reported, so the actual drop is unknown.
- **No text or SpokenQA results are reported for Raon-SpeechChat.**
- SpokenQA is speech in → *text* out (Tab.1), so knowledge *in generated speech* is never measured.

Factual recall through speech input lags reasoning (Tab.9, p.22):
- TriviaQA 62.00 vs 68.70 for MiniCPM-o 4.5 and 67.30 for Fun-Audio-Chat; WebQ 60.10 vs 65.90.
- By contrast BBH is 85.10 (next best 67.70) and IFEval 78.06.

## Key results

- **ASR, WER (Tab.4):** LibriSpeech-clean 1.44, LibriSpeech-other 2.89, FLEURS-en 3.59.
- **ASR, Korean CER (Tab.5):** KSponSpeech-clean 6.56, FLEURS-ko 1.81 (best).
- **TTS (Tab.4):** Seed WER **1.93** (best) / UTMOS 3.20; LibriSpeech-clean WER 2.01 / UTMOS 3.26. UTMOS is below Step-Audio 2 mini (3.85) and Qwen2.5-Omni (3.56).
- **QA and understanding (Tab.4):** VoiceBench **76.79** (MiniCPM-o 4.5: 76.06); OpenAudioBench 70.21 (MiniCPM-o 4.5: 74.82); MMAU-speech 78.68; MMAU-Pro-speech 64.65.
- **Full-duplex (Tab.6, internal evaluator).** Columns are Raon / MiniCPM-o 4.5 / PersonaPlex / Moshi:

| Benchmark | Metric | Raon | MiniCPM-o 4.5 | PersonaPlex | Moshi |
|---|---|---|---|---|---|
| v1.0 pause handling | Candor TOR ↓ | 0.213 | 0.343 | 0.204 | 0.370 |
| v1.0 backchannel | TOR ↓ | **0.091** | 0.418 | 0.236 | 0.309 |
| v1.0 backchannel | Freq ↑ | **0.081** | 0.008 | 0.046 | 0.050 |
| v1.0 smooth turn-taking | TOR ↑ | 0.832 | 0.891 | 0.782 | 0.437 |
| v1.0 smooth turn-taking | latency (s) | 1.034 | 1.000 | 1.101 | 0.726 |
| v1.0 interruption | TOR ↑ | **0.980** | 0.845 | 0.880 | 0.705 |
| v1.0 interruption | latency (s) | 1.219 | 1.233 | 0.980 | 2.027 |
| v1.0 interruption | Judge ↑ | **2.790 (lowest)** | 3.408 | 2.943 | 2.908 |
| v1.5 user backchannel | resume ↑ | 0.398 | 0.520 | 0.418 | 0.092 |
| v1.5 interruption | respond ↑ | 0.725 | 0.660 | 0.710 | 0.560 |
| v2.0 | TT fluency / IF / Task ↑ | 3.552 / 3.042 / 2.944 | **3.984 / 3.534 / 3.241** | 3.706 / 3.162 / 3.111 | 3.274 / 2.533 / 2.259 |

## Ablations & lessons

No component ablations are reported. The lessons below come from the authors' statements, design choices and cross-model numbers.

- **(b) Encoder side.** Freezing the encoder, codec and speaker encoder during end-to-end training gives better and more stable optimization than tuning them (qualitative, p.7).
- **(b) On-policy KL.** Computing the KL on student trajectories is reported to help both acquiring speech skills and limiting forgetting (qualitative, p.7). Its effect is not separated from that of the backbone-generated TextQA targets.
- **(a)/(b) Post-training.** 800 steps of SimPO on the model's own rejected outputs reduces repetition and raises rewards (qualitative, p.7).
- **(a) Audio head pre-aligned.** The audio head was pre-aligned in a large frozen-LLM TTS stage (65.5k steps), and the RCP was warm-started from Qwen3-Omni. The result: best intelligibility (Seed WER 1.93), but **lower naturalness** (UTMOS 3.20 vs 3.85). The authors concede the naturalness gap (p.12).
- **(c) Text lookahead.** One frame of lookahead reduces drift between the text plan and the generated speech; the authors call it especially important for full duplex (p.5). No number is given.
- **(c) Dedicated state tokens.** SIL (rather than Moshi's single PAD) plus BOW separate timing from content. A dedicated BC token with ×50 CE fixes label imbalance (p.5, p.8). The result is the best backchannel TOR/frequency (Tab.6). There is no no-BC ablation.
- **(c) Loss down-weighting.** PAD is weighted 0.75 and SIL 0.5, then 0.25 in fine-tuning (p.8). No sweep is reported.
- **(c) Timing ≠ content.** The best interruption TOR (0.980) comes with the *worst* interruption-response relevance (2.790), and FDB v2.0 IF is 3.042 vs 3.534 for MiniCPM-o 4.5 (Tab.6). The authors name multi-turn instruction following as future work (p.15).
- **(d) Scale.** Duplex is a thin layer: about 15.7B duplex-specific tokens on top of about 132.5B SpeechLM tokens (derived). Real duplex data is only 13.21K h, about 11% of the duplex hours (derived).
- **(d) Mixture.** 10% replay plus knowledge-bearing duplex reconstructions (p.8, p.10). Not ablated.
- **(e) Compute.** GPU-hours are not reported. Token budgets can be derived from the fully specified step counts, batch sizes and lengths.
- **Evaluator caveat.** Raon's FDB v1.0/v1.5 setup uses a 1.5 s / 5-word takeover threshold (official: 1.0 s / 3 words), ASR-refined anchors and a GPT-5.2 judge (p.23–24). Its numbers are not comparable with other papers' FDB scores.

## Implications for our setup

- **10K h of duplex audio alone is unlikely to teach knowledge transfer.**
  - Raon's real duplex data (13.21K h) is about our size. But it sits on a SpeechLM pretrained on 1.01M h with explicit speech→text knowledge tasks; our weak semantics match the missing foundation stage.
  - Add ASR/TTS/SpokenQA/TextQA training before or alongside duplex training.
  - Then replay about 10% of it during duplex training.
- **Cheap KD.** Use the frozen Qwen3 text model as teacher on transcripts of our dialogues, with on-policy KL on the assistant text channel and equal CE:KL weights. Regenerate text targets with the backbone.
- **Keep the LLM in text space.** If our Qwen3 backbone predicts audio tokens through its own head, consider Raon's split:
  - A text/state channel from the LLM.
  - A small expert for the semantic codebook.
  - A depth transformer for the residual codebooks, warm-started from a Qwen3-family code predictor where shapes allow.
  - Feedback of the summed codec embedding.
- **Train the audio head first with the LLM frozen** (Raon: 65.5k TTS steps). Continuation quality should not depend on learning acoustics from scratch during duplex training.
- **Duplex format.** Use SIL/BOW/PAD/BC tokens, 1-frame text lookahead, down-weighted PAD and SIL losses, and an up-weighted BC loss.
- **Synthesize knowledge-bearing duplex data.**
  - Qwen3-written dialogues built from QA and instruction data, voiced by Qwen3-TTS in a fixed assistant voice.
  - Re-timed backchannels, and barge-ins truncated at word boundaries.
  - Raon's duplex data is 89% synthetic.
- **Low duplex-stage LR** (Raon: 5e-6, then 3e-6) once the model has a strong initialization.
- **Measure what Raon did not.** Run spoken-output VoiceBench/OpenAudioBench, FDB v2.0 IF/Task and the FDB v1.0 interruption Judge after the duplex stages, using the official FDB scripts.

## Open questions / unclear points

- What is the text-only share or task mixture per stage? How many hours went into the alignment stages?
- Which modules are trainable in the causal and duplex stages? Is the encoder frozen?
- Post-training data is listed as 11.19B tokens, but 800 × 96 × 8,192 ≤ 0.63B (derived). Was the data subsampled?
- How was a Qwen3-Omni code predictor mapped onto Mimi's 2048-entry codebooks?
- Exact per-frame layout: are SIL and PAD ever inputs? What position IDs are used?
- How much do KD, replay and lookahead each contribute? What knowledge and naturalness remain after the duplex stages?
- What are the GPU-hours for any stage?
