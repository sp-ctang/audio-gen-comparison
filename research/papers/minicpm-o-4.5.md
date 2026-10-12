# MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction

| Field | Value |
|---|---|
| Org | MiniCPM-o Team, OpenBMB (p.1) |
| Date / arXiv | arXiv 2604.27393v1, 30 Apr 2026 (p.1) |
| Category | omni (vision + audio in → text + speech out) with full duplex |
| Backbone | Qwen3-8B (8,189.2M; 36 layers, hidden 4096, 32 query / 8 KV heads, context 40,960), initialized **from the MiniCPM-V 4.5 pretraining checkpoint**. 9.34B learnable parameters in total (p.4, p.8, p.21, Tab.13) |
| Audio input | Whisper-Medium encoder (307.2M), streaming in chunks at 50 Hz. A 2-layer MLP projector (1024→4096→4096, 21.0M) compresses time 5×, giving **10 audio tokens/s** to the LLM (p.4, Tab.13) |
| Audio output | CosyVoice **S3 tokens at 25 Hz, 1 codebook** (6,562 entries), produced by a separate Llama-style **speech token decoder of about 0.3B** (20 layers, hidden 768; 188.8M transformer + 116.8M text embedding). Its input for each text token is the decoder's text embedding plus an MLP projection of the LLM hidden state (4096→768→768, 10.5M). Text and speech tokens are interleaved: fixed ratio in turn-based mode, TAIL in duplex mode (p.4, p.6–7, Tab.13) |
| Waveform decoder | Streaming flow-matching decoder (CosyVoice2 / Step-Audio-2 style), conditioned on reference audio from the multimodal system prompt. Size and training not reported; it is not counted in the 9.34B (derived: the component sum in Tab.13 is 9,340.2M) (p.4) |
| Duplex mechanism | **Time-slicing ("Omni-Flow", time-division multiplexing)** with **1.0 s chunks**. Each chunk is g_k = [visual tokens; audio tokens; output tokens]. The output is `[listen]`, or a *speak* control token followed by text tokens (the "LS" formulation). Explicit chunk-boundary tokens are used. Time-Aligned Interleaving (TAIL) keeps speech playback in step with chunk time (p.5–7) |
| Open weights / code | Yes: model, code and demo (p.1). Also the llama.cpp-omni inference framework (p.14) |
| Relevance to us | **high**: Qwen3-8B backbone. The clearest ablation showing that duplex format choices (chunk size, control token, boundaries) move knowledge benchmarks. The thinker/talker split keeps the LLM in text space. Weak on data and compute detail |

## TL;DR

- A 9B omni model that sees, listens and speaks at the same time in a time-division-multiplexed loop of 1.0 s chunks. The LLM emits **only text** (3–4 tokens/s). A 0.3B decoder turns LLM hidden states into 25 Hz S3 speech tokens, and a flow-matching decoder produces the waveform (p.4–5).
- Format ablations (Tab.1, p.5):
  - **1.0 s chunks are far better than 0.2 s or 0.1 s** (MMLU 0.65 vs 0.45 / 0.32).
  - **Explicit boundary tokens beat implicit ones.**
  - **A separate listen/speak control token (LS) beats a shared listen-or-text output (LT)** (MMLU 0.65 vs 0.56).
- TAIL is chunked, history-dependent text/speech interleaving built from token timestamps. It stops speech from lagging behind the model's state, at an intelligibility cost: Seed-EN WER 3.93 vs 2.38 with a fixed text/speech ratio (Tab.10, p.13).
- Text capability is roughly kept: average 82.1 vs 81.6 for Qwen3-8B. But MMLU drops 4.7 and MATH500 drops 7.0 (Tab.6, p.12).
- Training has four stages: speech pretraining (frozen backbone) → joint pretraining (all parameters unfrozen; the mixture includes full-duplex data) → two-phase SFT → RL (GRPO with a smooth length reward, then RLAIF-V). **Data volumes, hyperparameters and compute are not reported.**
- The paper reports no speech full-duplex benchmark. Raon's independent evaluation finds it strongest on FDB v2.0 multi-turn content, but it almost never backchannels.

## Architecture

The model has three parts, all differentiably connected through token-level hidden states (p.3):
1. Streaming encoders.
2. The LLM.
3. The speech decoders.

**Vision (p.3).**
- Images are split into slices with LLaVA-UHD.
- SigLIP ViT (417.8M) encodes each slice into 1024 tokens, and a resampler (88.9M) compresses them to **64 tokens**.
- Maximum resolution is 448×448 in full-duplex streaming mode and 2240×2240 otherwise.

**Audio in (p.4).**
- Whisper-Medium runs in chunks (WeNet-style streaming) at 50 Hz.
- An MLP projector compresses this 5× to 10 Hz.
- User speech is a **continuous input only**: it is never tokenized or predicted.

**Text out (p.4).**
- The LLM generates only text tokens, which needs only 3–4 decoding steps per second.
- The authors argue that having the backbone emit about 25 speech tokens/s hurts efficiency and tends to degrade core language ability, citing Hsiao et al. 2025 and Qwen2.5-Omni.
- They therefore delegate speech tokens to a lightweight decoder.

**Speech out (p.4, Fig.4).**
- For each text token, the speech decoder receives the sum of its own text-token embedding and the projected LLM hidden state. The idea is that the LLM has already encoded the prosodic decisions, so the small decoder can focus on speech modeling.
- The decoder then generates S3 tokens autoregressively.
- The flow-matching decoder renders the waveform in the voice of the reference audio, which also enables voice cloning (p.3).

**Omni-Flow (p.5).** Three time-aligned streams:
- *env-visual*;
- *env-audio*, which carries the acoustic scene including user speech (user requests are treated as part of the continuously observed world, not as a privileged role);
- *out-stream*, the assistant's text and speech.

**Serialization (p.5).**
- Each chunk of duration t is serialized as g_k = [v_k; a_k; o_k], and the chunks are concatenated into one causal sequence.
- Within a chunk the model first reads the new perception tokens, then generates output, so every output is conditioned on the latest observation.
- When there is nothing to say, o_k = `[listen]`.
- This reduces dependence on an external voice-activity detector (VAD), and proactive speech can emerge naturally.
- Fig.4 shows that every speaking chunk starts with a *speak* token followed by text tokens; each text token's hidden state is merged into the speech decoder.

**TAIL (p.6–7).**
- At chunk k, the model generates as much text as it can vocalize so that playback approaches the boundary kt.
- If playback is already lagging, it generates fewer text tokens.
- A **bounded look-ahead** defers the speech of the last few text tokens of chunk k to chunk k+1. Example from the paper: "the" is pronounced differently in "the apple" and "the car".

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- Not described.
- The S3 tokenizer and the streaming flow-matching decoder are taken from prior work (CosyVoice / CosyVoice2 / Step-Audio 2). Whether they were fine-tuned is not reported.
- Whisper-Medium is pretrained. SigLIP and the resampler come with the MiniCPM-V 4.5 checkpoint (p.4, p.8).

### Stage 1: Speech pretraining
- **Initialization:** Whisper encoder + MiniCPM-V 4.5 pretraining checkpoint, plus randomly initialized audio projector, LLM-to-speech projector and speech decoder (p.8).
- **Trainable:** **pretrained components are frozen; only the new modules are trained**, to protect the backbone's visual and language abilities.
- **Purpose:** align Whisper features with the LLM hidden space, and train the decoder to map LLM hidden states to speech tokens grounded in both semantics and prosody.
- **Data:** presumably the ASR / TTS sets from §4.1. The stage's data mix and hours are not reported.
- **Optimization and compute:** LR, batch, steps and compute are not reported.

### Stage 2: Joint pretraining
- **Trainable:** **all parameters unfrozen**.
- **Objective:** a unified next-token prediction objective (p.8).
- **Data:** a balanced mixture of vision-language, speech and omni-modal data that includes turn-based samples **plus proactive and full-duplex interaction data**, with text tokens aligned to speech and visual signals on a shared timeline.
- **Stability trick:** each modality combination is assigned to different data-parallel ranks, which **fixes the data ratio at every training step**.
- **Not reported:** mixture ratios, text share, tokens, hours, hyperparameters, compute.

### Stage 3: Joint supervised fine-tuning
- **Two phases (p.8):**
  1. Large-scale instruction tuning, for broad capability.
  2. High-quality human-annotated tuning, for fine-grained behaviour.
- **Augmentation:** omni-modal samples are augmented with maximum frame resolution 0.2–0.4 MP and 1–5 FPS.
- **Not reported:** sizes and hyperparameters.

### Stage 4: Reinforcement learning
- **GRPO** for reasoning and instruction following (p.9). Rewards:
  - Answer accuracy, from rule-based verification plus the CompassVerifier judge.
  - A format reward.
  - A **smooth length reward** (Eq.1, adapted from Kimi-K1.5). It rewards shorter correct responses, never rewards short wrong ones, and is scaled down when length differences are small. It is disabled for the first 480 steps.
  - A **general reward model**, to raise answer quality and suppress accidental language mixing.
- **RLAIF-V** to reduce visual hallucination. The authors report that this carries over to full-duplex streaming, reducing hallucination there too (qualitative).
- No speech-specific or duplex-specific RL or preference stage is described.

## Data pipeline

**Large-scale natural speech (p.7).**
- Millions of hours of unlabelled speech (no exact count given) are processed with Silero VAD, Whisper, Paraformer, self-supervised diarization and Demucs-style source separation.
- The output is training sets for zero-shot TTS, ASR and **multi-turn multi-speaker dialogue**.
- Exact hours are not reported.

**Spoken dialogue (p.7).**
- A text LLM writes colloquial, instruction-following dialogues from diverse seed queries.
- **A subset is re-recorded by professional voice actors** in a studio. They speak conversationally and improvise, not reading verbatim, and vary emotion, speaking rate and emphasis **under a consistent vocal identity**.
- The result covers instruction-following TTS, QA and multi-turn dialogue.

**Vision-language data (p.7).**
- CapsFusion captions with a better generator.
- Document / OCR data with relevance-aware masking.
- Real-world queries with short answers rewritten into chain-of-thought, filtered by a reward model.
- Dense video captions.

**Text-only data (p.7).** **MiniCPM 4.1 post-training instruction data**, included to keep language ability robust. The share is not reported.

**Omni-modal full-duplex data (p.8).**
- *Web audio-video*:
  - Segments dominated by a single speaker, or with weak audio-visual relevance, are filtered out.
  - OCR-based subtitle removal, talking-head detection, and filtering on ASR transcripts.
- *Full-duplex task data*: scenarios built by hand with annotated instruction-following data, for continuous scene description and proactive reminders.
- Every element (visual input, audio input, output text, output speech) carries a time index.

**TAIL labels (p.6–7).**
- Start and end times of each text token come from the full-duplex streaming data.
- Each token, together with its speech tokens, is assigned to the chunk whose interval [(k−1)t, kt) contains the token's start time.

**Not described:**
- How user and assistant roles are assigned in web audio-video.
- Any synthetic overlap, backchannel or interruption generation.
- Silence ratios.
- Duplex hours.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Speech pretraining | not reported | not reported | new modules only |
| Joint pretraining | not reported (millions of hours of raw speech overall, p.7) | not reported | fixed per-step ratio via DP ranks |
| Joint SFT (2 phases) | not reported | not reported | |
| RL (GRPO, RLAIF-V) | not reported (length reward off for the first 480 steps) | not reported | |
| Inference | – | One RTX 4090: BF16 154.3 tok/s, 0.59 s first-token latency, 19 GB; INT4 212.3 tok/s, 0.58 s, 11 GB (Tab.11). llama.cpp-omni RTF: 0.27 (FP16) / 0.21 (INT4) on 4090; 0.46 / 0.20 on DGX Spark (Tab.12) | runs in under 12 GB RAM (p.2) |

## Preserving text intelligence / knowledge

**Mechanisms named (none ablated):**
- The LLM decodes text only; a separate talker produces speech tokens (p.4).
- The backbone is frozen during speech pretraining (p.8).
- Text-only instruction data is in the mixture, and the authors credit a deliberate balance of text and multimodal data (p.11–12).
- Data ratios are fixed at every step (p.8).
- The claim that direct speech-token generation by the backbone degrades language ability is **cited, not tested** here.

**Text retention (Tab.6, p.12), Qwen3-8B-Instruct → MiniCPM-o 4.5:**

| Benchmark | Qwen3-8B | MiniCPM-o 4.5 | Δ |
|---|---|---|---|
| IFEval | 83.0 | 84.7 | +1.7 |
| BBH | 69.4 | 81.1 | +11.7 |
| CMMLU | 78.7 | 79.6 | +0.9 |
| MMLU | 81.7 | 77.0 | **−4.7** |
| HumanEval | 86.6 | 86.6 | 0 |
| MBPP | 75.9 | 76.7 | +0.8 |
| MATH500 | 84.0 | 77.0 | **−7.0** |
| GSM8K | 93.4 | 94.5 | +1.1 |
| **Average** | 81.6 | 82.1 | +0.5 |

Deltas are derived. Knowledge recall (MMLU) and competition maths regress, while reasoning and instruction following improve.

**Speech-input knowledge QA (Tab.4).** Output modality is not stated.

| Benchmark | MiniCPM-o 4.5 | Qwen3-Omni-30B-A3B | Kimi-Audio |
|---|---|---|---|
| Speech TriviaQA | **75.5** | 62.9 | 41.9 |
| Speech WebQuestions | 70.2 | 74.9 | 46.4 |
| Speech CMMU | 59.2 | 47.8 | 67.0 |
| VoiceBench AlpacaEval (1–5) | 4.81 | 4.74 | 4.46 |

Raon's independent evaluation measures MiniCPM-o 4.5 at MMLU-Pro 55.20, MMLU-Redux 72.53 and VoiceBench 76.06 (Raon Tab.4, a different protocol).

**In duplex mode, knowledge depends on the format.** MMLU ranges from 0.32 to 0.65 across the Tab.1 configurations.

## Key results

**Speech generation (Tab.5).** Turn-based, fixed-ratio interleaving mode.

| Model | SeedTTS ZH CER ↓ | SeedTTS ZH SIM | SeedTTS EN WER ↓ | SeedTTS EN SIM | LongTTS EN WER ↓ | LongTTS ZH CER ↓ | Expresso | ESD |
|---|---|---|---|---|---|---|---|---|
| MiniCPM-o 4.5 | **0.86** | 74.5 | **2.38** | 64.9 | **3.37** | 6.58 | 29.8 | 82.1 |
| CosyVoice2 | 1.45 | 74.8 | 2.57 | 65.2 | 14.80 | 5.27 | 17.9 | 53.4 |
| Qwen3-Omni | 1.41 | N/A | 3.39 | N/A | 17.33 | 18.99 | N/A | N/A |

N/A means "not supported" in Tab.5. (Raon's protocol gives MiniCPM-o 4.5 a Seed WER of 4.72 and UTMOS 3.06; Raon Tab.4.)

**ASR and audio understanding (Tab.4).**

| Benchmark | Score |
|---|---|
| LibriSpeech clean / other (WER) | 1.4 / 2.8 |
| GigaSpeech (WER) | 8.5 (best) |
| VoxPopuli (WER) | 6.2 (best) |
| AISHELL-1 / AISHELL-2 | 0.9 / 2.5 |
| MMAU | 76.9 |
| MELD | 60.2 |
| CoVoST 2 en→zh / zh→en | 49.9 / 26.4 |

**Omni understanding (Tab.7).** Best on 5 of 7 benchmarks: Daily-Omni 80.2, WorldSense 55.7, Video-Holmes 64.3, JointAVBench 60.0, AVUT-Human 78.6.

**Full duplex, own paper.** Only LiveSports-3K-CC, which is vision-only and has no audio: win rate **54.4** vs 41.5 for LiveCC and 45.6 for StreamingVLM (Tab.8). **No Full-Duplex-Bench results are reported.**

**Full duplex, from Raon's report (Raon Tab.6, internal evaluator):**

| Benchmark | Metric | MiniCPM-o 4.5 |
|---|---|---|
| FDB v1.0 smooth turn-taking | TOR (↑) | 0.891 (best) |
| FDB v1.0 smooth turn-taking | latency | 1.000 s |
| FDB v1.0 interruption | TOR / Judge | 0.845 / **3.408** (best) |
| FDB v1.0 backchannel | freq / TOR | **0.008 / 0.418** (worst-but-one) |
| FDB v1.0 pause handling | synthetic TOR | 0.182 (best) |
| FDB v1.0 pause handling | Candor TOR | 0.343 |
| FDB v2.0 | TT fluency / IF / Task | **3.984 / 3.534 / 3.241** (all best) |

## Ablations & lessons

- **(c)+(b) Chunk size is the main latency-versus-capacity knob (Tab.1, p.5–6).** Explicit boundaries and LS control in all rows:

| Chunk | AdvBench | AlpacaEval | IFEval | SD-QA | MMLU |
|---|---|---|---|---|---|
| 1.0 s | 0.98 | 3.56 | 0.29 | 0.36 | 0.65 |
| 0.2 s | 0.81 | 1.22 | 0.10 | 0.09 | 0.45 |
| 0.1 s | 0.67 | 2.40 | 0.10 | 0.13 | 0.32 |

  - Too-short chunks give the model too little information per window to make stable decisions and produce coherent output (p.6).
  - **Finer time slicing directly erodes knowledge and instruction following.**
  - AlpacaEval is non-monotonic (0.1 s scores above 0.2 s), and no variance is reported.
- **(c)+(b) Explicit chunk boundaries help** (LT control, 1.0 s chunks), with MMLU the largest gain:

| Boundary | AdvBench | AlpacaEval | IFEval | SD-QA | MMLU |
|---|---|---|---|---|---|
| Explicit | 0.92 | 3.60 | 0.24 | 0.35 | 0.56 |
| Implicit | 0.96 | 3.31 | 0.22 | 0.28 | 0.45 |

  - The authors' reading: telling newly observed inputs apart from newly generated outputs is hard for the model, and marking the boundary explicitly lightens that load.
- **(c)+(b) Decouple *whether* to speak from *what* to say.** LS vs LT, explicit boundaries, 1.0 s chunks:

| Control | AdvBench | AlpacaEval | IFEval | SD-QA | MMLU |
|---|---|---|---|---|---|
| LS | 0.98 | 3.56 | 0.29 | 0.36 | 0.65 |
| LT | 0.92 | 3.60 | 0.24 | 0.35 | 0.56 |

  - This is the same conclusion as Raon's BOW and SIL tokens.
- **Caveats on Tab.1.**
  - The protocol is not described: input modality, whether streaming, subset sizes.
  - The column names match VoiceBench subsets; that is my inference.
  - The absolute values (for example IFEval 0.29) are far below the turn-based scores, so these may be small-scale runs.
- **(a) Interleaving mode versus speech quality (Tab.10, Seed TTS):**

| Mode | ZH CER | EN WER | SIM (ZH / EN) |
|---|---|---|---|
| Text first (no interleave) | 1.44 | 2.70 | 74.1 / 64.9 |
| Fixed-ratio interleave | **0.86** | **2.38** | 74.5 / 64.9 |
| TAIL | 1.04 | 3.93 | 74.1 / 65.1 |

  - Chunked interleaving *improves* pronunciation over text-first generation.
  - Time-aligned (duplex) interleaving costs about 1.5 WER points in English, while speaker similarity is unchanged.
  - **The headline speech numbers (Tab.5) are fixed-ratio mode, not duplex mode.**
  - The authors admit streaming-mode speech "can occasionally be unstable, including mispronunciation or unintended mixing between English and Chinese" (p.15).
- **(b) Thinker/talker separation plus text-only data roughly preserves text** (average +0.5), but MMLU −4.7 and MATH500 −7.0 (Tab.6). There is no ablation against a backbone that emits audio tokens.
- **RL length reward (Tab.9, lightweight vision-language RL).**

| Reward | Thinking avg | Instruct avg | Length cut (thinking / instruct) |
|---|---|---|---|
| None | 73.5 | 70.9 | – |
| Kimi-K1.5-style | 73.0 | 70.1 | −50.7% / −20.2% |
| Smooth (theirs) | **74.3** | **70.9** | −35.3% / −20.5% |

  - With the K1.5-style reward, training accuracy stalls late in training (Fig.6).
  - This is not speech-specific, but it is relevant if we later RL the text channel.
- **(b)/(c) Post-training transfers across modes.** RLAIF-V on image-text reduces hallucination in streaming duplex mode too (qualitative, p.9).
- **(c) Cross-paper view (Raon Tab.6).** A coarse 1 s time-division format with LS control gives the best FDB v2.0 content scores and interruption-response relevance. Backchannels are almost absent (frequency 0.008), and latency is about 1 s. My inference: this format trades micro-timing for content quality.
- **(d)/(e)** No evidence on data scale or compute is reported.

## Implications for our setup

- **Keep the Qwen3 LLM generating text, and give audio to a talker.**
  - MiniCPM's LLM decodes at 3–4 tokens/s and never predicts audio tokens. A 0.3B decoder conditioned on its hidden states and text tokens produces the speech.
  - If our backbone predicts Qwen3 codec tokens directly, both the knowledge loss and the weak continuation could come from that design.
  - A talker conditioned on hidden states, and warm-started from a pretrained TTS-style model for our tokenizer, is the standard fix.
- **If we use time slicing:**
  - Start with chunks of about 1 s, explicit boundary tokens, and an explicit listen/speak control token. Their ablation shows that sub-second chunks halve MMLU and IFEval.
  - Expect slow responses (about 1 s) and almost no backchannels; Raon's frame-level design does better on these.
  - If we keep frame-level (80 ms) Moshi/Raon-style streaming, the LS lesson still applies.
- **Build TAIL-style labels from our 10K h.** Forced-align the assistant words, assign tokens to chunks by start time, and defer the speech for the last few tokens of each chunk.
  - Evaluate speech in duplex mode separately: expect about +1.5 WER versus turn-based mode.
- **Data and mixture:**
  - Mix in text-only instruction data.
  - Fix the per-step modality ratio by assigning sources to DP ranks.
  - Use freeze-then-unfreeze: train only the new modules first, then train everything jointly with duplex data already in the pretraining mixture.
- **Output voice.** A consistent, high-quality assistant voice from voice actors, plus a reference-audio-conditioned vocoder, sidesteps open-ended voice continuation.
  - Neither MiniCPM nor Raon trains the model to *continue* arbitrary speaker audio. Both condition on a fixed target voice.
- **Post-training.** Text-channel RL (GRPO with verifiable rewards and a smooth length reward; RLAIF-style preference data) transferred to streaming mode without any duplex-specific RL. This is a cheap way to lift semantic quality after duplex SFT.

## Open questions / unclear points

- Hours and tokens per stage; mixture ratios and text share; amount of duplex data; how web audio-video is turned into user/assistant roles; whether overlap, backchannels or interruptions are synthesized.
- All optimization hyperparameters and compute.
- The evaluation protocol behind Tab.1. Were the ablation models trained at full scale?
- Whether the S3 tokenizer and flow-matching decoder were fine-tuned. Is reference audio always required?
- How barge-in is handled: does the model simply emit `[listen]` in the next chunk? How does TAIL interact with interruption and truncation?
- Whether the speech QA scores in Tab.4 are measured on text or speech output.
- The paper reports no FDB results.
