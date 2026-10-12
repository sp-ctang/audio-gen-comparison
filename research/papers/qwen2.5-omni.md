# Qwen2.5-Omni Technical Report (Qwen2.5-Omni-7B)

| Field | Value |
|---|---|
| Org | Qwen Team, Alibaba (p.1) |
| Date / arXiv | 2025-03-27 (title page); arXiv:2503.20215v1, 26 Mar 2025 (p.1) |
| Category | omni; half-duplex streaming S2S |
| Backbone | Thinker: Transformer decoder initialized from Qwen2.5 (p.6), evaluated as the 7B model (Tab.1). Talker: dual-track autoregressive Transformer decoder, inspired by Mini-Omni (p.3); its size is not reported |
| Audio input | Qwen2-Audio encoder, initialized from Whisper-large-v3 (p.6). 16 kHz, 128-channel mel, 25 ms window / 10 ms hop. About 40 ms per frame, i.e. 25 Hz (p.4; rate derived). Attention restricted to 2-s blocks for streaming prefill (p.5) |
| Audio output | Discrete codes from "qwen-tts-tokenizer". Frame rate and codebook size are not reported. Qwen3-Omni later describes this design as single-track (Qwen3-Omni p.2). The Talker conditions on Thinker hidden states plus embeddings of the text tokens the Thinker samples (p.5) |
| Waveform decoder | Flow-matching DiT (code → mel) with sliding-window block attention, then a modified BigVGAN (mel → wave); both decoded chunk by chunk. The DiT's receptive field is 4 blocks: 2 back, current, 1 ahead (p.5, Fig.4) |
| Duplex mechanism | n/a — turn-based streaming |
| Open weights / code | GitHub QwenLM/Qwen2.5-Omni and HF (p.1); license not stated in the paper |
| Relevance to us | Medium. This is the predecessor to the Qwen3 stack. It gives evidence on text degradation, on how the Talker should be conditioned, and on DPO for speech stability |

## TL;DR

- **Introduces Thinker–Talker.** The Thinker (initialized from Qwen2.5) generates text. The Talker generates speech tokens from two inputs: the Thinker's hidden representations and embeddings of the text tokens the Thinker samples. The whole model is trained end to end (p.3, p.5).
- **Pretraining in three stages** (p.6):
  - S1: encoder alignment with the LLM frozen.
  - S2: adds 800B image/video, 300B audio and 100B video-with-audio tokens. The amount of text is not reported.
  - S3: 32K-token context.
- **Talker training** (p.7): first in-context "context continuation", then DPO with rewards based on WER and punctuation-pause errors, then speaker fine-tuning. DPO cut SEED test-hard WER from 7.97 to 6.54 (Tab.9).
- **Text ability degraded.** The 7B Omni model lands between Qwen2-7B and Qwen2.5-7B, e.g. MMLU-Pro 47.0 vs 56.3 for Qwen2.5-7B (Tab.1). Qwen3-Omni's "early mixing" recipe was designed to fix this.
- **No duplex** capability, and the codec itself is not described.

## Architecture

- **Thinker.**
  - Qwen2.5 LLM with a 151,643-token BPE vocabulary (p.4).
  - TMRoPE positions: one temporal ID per 40 ms. Video with audio is interleaved in 2-s chunks, video first, then audio (p.4).
  - Vision encoder from Qwen2.5-VL, ~675M params (p.4).
- **Talker conditioning** (p.5; shares the Thinker's full context, p.3):
  - **Thinker hidden states** carry tone and attitude before the full text exists.
  - **Sampled text-token embeddings** are also needed, because hidden states encode semantic rather than phonetic similarity.
- **Talker output** (p.5): audio and text tokens generated autoregressively. No word- or timestamp-level text–speech alignment is required.
- **Codec.** The paper says only that qwen-tts-tokenizer efficiently represents key speech information and can be decoded in streaming by a causal decoder (p.5). Frame rate, codebook size and training are not given. Qwen3-TTS's 25 Hz tokenizer uses the same block-DiT + BigVGAN decoder design and cites this paper, but neither paper says the tokens are the same.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **Audio encoder:** from Qwen2-Audio, initialized from Whisper-large-v3, then changed to 2-s block attention (p.5–6).
- **Vision encoder:** from Qwen2.5-VL (p.4).
- **qwen-tts-tokenizer, DiT and BigVGAN:** training not described.

### Stage 1: Encoder alignment
- LLM frozen (Qwen2.5 init).
- Vision and audio encoders are trained separately: adapters first, then the encoders (p.6).
- Data: audio-text and image-text pairs, size not reported. Context 8,192 (p.6).
- Optimizer, LR, batch size, compute: not reported.

### Stage 2: General multimodal
- All parameters unfrozen (p.6).
- New data (p.6):
  - 800B tokens of image/video data;
  - 300B tokens of audio data;
  - 100B tokens of video-with-audio data;
  - 1.2T multimodal tokens in total (derived).
- Pure text is described as essential for keeping language proficiency, but its amount is not reported. The text share therefore cannot be computed.
- Task tags are replaced with natural-language prompts (p.6).

### Stage 3: Long context
- Context extended to 32,768 tokens, with long audio and long video added. "Significant improvement" is claimed without numbers (p.6).

### Stage 4: Thinker SFT
- ChatML instruction tuning on pure text, visual, audio and mixed-modality conversations (p.6). Size not reported.

### Stage 5: Talker (p.7)
1. **In-context learning (ICL) / context continuation.** Text supervision, as for the Thinker, plus next-token speech continuation on a large dataset of dialogues with multimodal context and spoken responses. This teaches a monotonic mapping from semantics to speech, plus context-appropriate prosody, emotion and accent. Timbre disentanglement keeps the model from tying voices to rare text patterns.
2. **DPO for stability.** Each example is a triplet (input, good speech, bad speech). Samples are ranked by reward scores based on WER and punctuation-pause error rate. The goal is to fix hallucinations caused by label noise and mispronunciations in the pretraining data.
3. **Multi-speaker / speaker fine-tuning** for naturalness and controllability.

Data hours, steps and frozen modules: not reported.

## Data pipeline

- Data types: image-text, video-text, video-audio, audio-text and text (p.6).
- The pretraining data contains label noise and pronunciation errors; this is the motivation for DPO (p.7).
- Not described: collection, filtering, synthesis, and any duplex features.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| S1 | not reported | not reported | LLM frozen; 8,192 ctx |
| S2 | +1.2T multimodal tokens (300B audio) + text (amount not reported) (p.6) | not reported | all params |
| S3 | not reported | not reported | 32,768 ctx |
| Thinker SFT | not reported | not reported | ChatML |
| Talker | not reported | not reported | ICL → DPO → speaker FT |

## Preserving text intelligence / knowledge

- **Mechanisms:**
  - Text-only data is kept in S2 (p.6).
  - The Thinker generates the content as text; the Talker reads the sampled text tokens.
- **Text-to-text, 7B Omni vs Qwen2.5-7B** (Tab.1). It is lower on all 10 benchmarks, by 2.9–9.3 points (derived):

  | Benchmark | Qwen2.5-Omni-7B | Qwen2.5-7B |
  |---|---|---|
  | MMLU-Pro | 47.0 | 56.3 |
  | MMLU-redux | 71.0 | 75.4 |
  | GPQA | 30.8 | 36.4 |
  | MATH | 71.5 | 75.5 |
  | GSM8K | 88.7 | 91.6 |
  | HumanEval | 78.7 | 84.8 |

- **Spoken vs text instructions** (Tab.4; ~90% of items usable as speech). Omni with speech input vs Qwen2-7B with text input: MMLU 65.6 vs 69.3, GSM8K 85.4 vs 82.3, CEval 61.1 vs 78.4, IFEval 41.7 vs 53.3. The paper's "comparable to text input" claim (p.1) therefore rests on an older text model.

## Key results

- **Understanding** (Tab.2–3, Tab.8):
  - VoiceBench average 74.12; MMAU average 65.60; OmniBench 56.13%.
  - LibriSpeech test-clean / test-other WER 1.8 / 3.4.
- **SEED zero-shot TTS** (test-zh / test-en / test-hard; Tab.9):

  | Metric | ICL (before DPO) | After DPO ("RL") |
  |---|---|---|
  | WER | 1.70 / 2.72 / 7.97 | 1.42 / 2.33 / 6.54 |
  | Speaker similarity | 0.752 / 0.632 / 0.747 | 0.754 / 0.641 / 0.752 |

- **Speaker-fine-tuned voices** (Tab.10): naturalness MOS 4.46–4.51 (zh) and 4.51–4.62 (en); human recordings score 4.51 (zh).

## Ablations & lessons

- **(a) Preference optimization on WER-ranked pairs works.** WER dropped 16.5% on test-zh, 14.3% on test-en and 17.9% on test-hard (relative, derived). Speaker similarity was unchanged or slightly up (Tab.9). The paper attributes the gain to fewer attention misalignments, mispronunciations and wrong pauses (p.12).
- **(a) Speaker fine-tuning keeps stability and adds prosodic style.** WER on test-zh is 1.28–1.37, versus 1.30 for the base model as listed in Tab.10 (Tab.10, p.12).
- **(b) Adding multimodal data after text pretraining degrades text.** S2 adds 1.2T multimodal tokens on top of a finished Qwen2.5, and the result trails Qwen2.5-7B on every text benchmark (Tab.1). This is the configuration Qwen3-Omni later moved away from.
- **(b) Hidden states alone are not enough for the Talker.** It needs explicit text tokens because hidden states lack phonetic identity (p.5). This is an argument, not an ablation. Qwen3-Omni later dropped the text hidden states entirely and kept only the token embeddings (Qwen3-Omni p.3).
- **(c) Duplex:** no evidence.
- **(d), (e) Data scale and compute:** only token counts are reported.

## Implications for our setup

1. **Expect text loss when adding speech to a finished LLM.** Even with text data kept in S2, the 7B Omni model lost 3–9 points to Qwen2.5-7B. Mid-training a finished Qwen3 on 10K h of duplex speech is the same pattern, only more extreme, so text replay and text evaluations are mandatory.
2. **Give the speech generator explicit text tokens.** Qwen's argument (hidden states encode meaning, not phonetics) explains why a speech LM without a text channel produces fluent-sounding but wrong content. Both Qwen generations route content through text tokens.
3. **DPO on WER-ranked samples is cheap and effective.** It gave a 14–18% relative WER reduction here and needs only ASR-based rewards. Apply it to our generated agent turns to stabilize continuation.
4. **Timbre disentanglement matters with few speakers.** Duplex corpora often have few speakers. The paper warns that voices can become tied to particular content, and its fix is timbre disentanglement during Talker pretraining (p.7).
5. **Avoid this decoder for duplex.** The single-codebook + block-DiT/BigVGAN decoder needs future blocks before it can emit audio. Qwen3-Omni replaced it with a frame-causal ConvNet decoder precisely to cut first-packet latency (Qwen3-Omni p.6). Use the 12.5 Hz causal codec instead.

## Open questions / unclear points

- **Codec:** the frame rate, codebook size and training of qwen-tts-tokenizer are not given, nor whether it equals Qwen3-TTS's 25 Hz tokenizer.
- **Talker:** size, initialization, training hours, and which modules are frozen during Talker training are not reported.
- **Inconsistent "RL" figures:** Tab.10 lists the RL model's test-zh WER as 1.30, while Tab.9 says 1.42.
- **Inconsistent VoiceBench average:** Qwen3-Omni's Tab.7 lists 73.6 for this model, while this paper says 74.12.
- **Text share** in S2 is not reported, so the cause of the text degradation (too little text, or late mixing) cannot be separated.
