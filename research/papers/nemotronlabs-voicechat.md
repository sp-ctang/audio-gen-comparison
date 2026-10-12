# NemotronLabs VoiceChat (NVIDIA-NemotronLabs-VoiceChat-11B)

| Field | Value |
|---|---|
| Org | NVIDIA (NemotronLabs) |
| Date / arXiv | 2 Oct 2026; arXiv:2609.21967v2 (1 Oct 2026) (p.1) |
| Category | full-duplex S2S (modular: a full-duplex speech-to-text LLM plus a separately trained streaming TTS) |
| Backbone | NVIDIA Nemotron-Nano-9B-v2-Base, a 9B hybrid Mamba-Transformer, initialized from the *base* checkpoint, not the instruct one (p.3; Mamba states mentioned on p.17). The release is named "NVIDIA-NemotronLabs-VoiceChat-11B" (VoiceChat-TTS paper, p.2). The listed components add up to about 10.6B (derived: 9B + 0.6B encoder + 0.977B TTS). |
| Audio input | 16 kHz audio → 128-bin log-Mel (25 ms Hann window, 10 ms stride) → 600M cache-aware streaming FastConformer (24 layers, d=1024, 8× causal subsampling → one state per 80 ms = 12.5 Hz; 70-frame left context, no right context) → identity adapter + projection into the LLM. The LLM sees continuous embeddings, not audio tokens (p.3). |
| Audio output | The LLM emits **no audio tokens**: the audio-loss weight is 0 (p.5). It emits agent-text subwords on an 80-ms frame grid plus a parallel function-call channel. Speech comes from VoiceChat-TTS: a 22 kHz codec at 12.5 Hz with 31 RVQ levels. Each step generates one frame, and a Mixture-of-Gaussians head fills in all 31 levels in 4–8 refinement iterations (p.4). |
| Waveform decoder | 199M fully causal convolutional codec decoder, output at 22.05 kHz (p.4, p.15) |
| Duplex mechanism | Time-aligned streams on one 80-ms timeline. The per-frame input is a weighted sum of the user-audio embedding (weight 1), the previous agent-text token (1) and the previous function token (2). Frame-level BOS/EOS/PAD targets teach the model when to start talking, when to yield and when to stay silent. The TTS runs continuously and follows BOS, PAD and interruption control tokens. As a fallback, RNN-T-based endpointing heuristics can force BOS/EOS (p.3, p.6). |
| Open weights / code | Checkpoint on Hugging Face (p.2 fn.2). TTS code is in NVIDIA NeMo Speech (VoiceChat-TTS paper, p.2). Whether the STT training code is released is not stated. |
| Relevance to us | **high**. This is the most detailed open recipe for turning a text LLM into a full-duplex agent that keeps its knowledge, with exact mixture and loss weights. Its central choice, that the LLM never generates audio, bears directly on our weak audio continuation. |

## TL;DR

- The core is a duplex **speech-to-text** model: a 600M streaming encoder plus the 9B Nemotron-Nano-v2 base LLM. On parallel 80-ms channels it writes agent text (with BOS/EOS/PAD turn-taking targets) and tool calls. A separately trained 977M streaming TTS (VoiceChat-TTS) speaks the text. An RNN-T head, trained on the frozen encoder afterwards, provides live user transcripts (p.3–5).
- The LLM is trained in two stages, both on time-aligned two-stream data (p.5):
  - **CPT** uses TTS-rendered pseudo-dialogues built from text pretraining corpora (weight 0.95) plus single-turn QA (0.05).
  - **SFT** samples a 10-bucket mixture: about 47% retention (including 8.5% text-only with a text-to-text loss), 24% conversational, 26% tool calling and 3% safety (p.14–15).
  - Nearly all speech is TTS-rendered from text corpora (p.14).
- Both stages run on 64 GPUs with AdamW at LR 5e-5 (p.6). Steps, batch size, tokens seen and GPU-hours are **not reported**. There is no RL or preference stage.
- Results:
  - Best open-weight pause handling on FDB 1.0.
  - 93% Resume on FDB 1.5 user backchannels.
  - VoiceBench 55.1, about Freeze-Omni's level and roughly double Moshi and PersonaPlex.
  - Good tool routing but weak arguments on FDB-v3 (Pass@1 33.0).
- **The paper reports no ablations.** All evidence about what matters is indirect: comparisons with other systems, plus the authors' stated reasons for design choices.

## Architecture

**Perception and STT core (p.3).**
- The encoder cites nemotron-speech-streaming-en-0.6b [19]. That model's reference entry describes a 600M cache-aware FastConformer-RNNT trained on about 530k hours of audio (p.12). Initializing the encoder from it is implied but never stated.
- The encoder runs once and feeds two consumers:
  - an identity modality adapter plus projection into the LLM;
  - an RNN-T branch: a 2-layer, 640-d prediction network and a 640-d joint network over 1,024 BPE units plus blank, decoded incrementally.
- The RNN-T transcript is **not** fed back into the LLM. The authors do this to keep the response path conditioned directly on speech (p.3).

**Output streams and timeline (p.3).**
- The agent response sits on the encoder's 80-ms grid. Every example starts as all-PAD.
- At each agent turn onset the model must emit BOS, then the response subwords in consecutive frames, one token per 80 ms. The text therefore runs ahead of the speech; the TTS paper measures about 4.16 text tokens per second of speech (VoiceChat-TTS p.2).
- EOS is the target shortly after the next user turn begins, so the two briefly overlap. The last agent turn of an example has no EOS. All other frames are PAD.
- So BOS means "start talking", EOS means "yield the floor once the user has started", and PAD means "stay silent".
- The per-frame input has no agent-audio stream: only user audio, agent text and function tokens (p.3). The model therefore never hears its own voice. Our reading is that agent audio in the training data only fixes the turn timing.

**Function (tool) channel (p.3–4, Fig.2).**
- It is a separate autoregressive channel with its own head.
- It emits PAD when no tool is needed. The authors call this negative supervision important for avoiding spurious calls (p.3).
- Call format: `<SOTC>` `<TOOLCALL>[{name, arguments},…]</TOOLCALL>` `<EOTC>`, then the injected `<TOOL_RESPONSE>…</TOOL_RESPONSE>`, then `<EOTR>`. The three boundary tokens are reserved vocabulary tokens `<SPECIAL_20/21/22>`.
- A JSON list allows parallel calls. After `<EOTR>` the model can call again, so multi-step chains are possible.
- During a tool span, the agent-text channel is supervised as PAD, user audio is replaced by silence, and tool-response tokens are context only (no loss).
- The system prompt with tool schemas sits at the start of the system/tool channel (Fig.2). Its Jinja template is adapted from Nano v2 (p.18–19).

**Speech output: VoiceChat-TTS (p.4–5).** The 977M decoder is detailed in voicechat-tts.md.
- It consumes the LLM's incremental text stream (same tokenizer) plus BOS, PAD and interruption control symbols.
- It does not decide turn-taking from user audio itself; it follows the upstream model's timeline (p.4).
- Other features: audio lags text by one step; character-aware subword embeddings; a 3-s speaker prompt; on interruption it can inject a fixed silence frame (p.5).

**Runtime (p.15–17).**
- Serving: a custom vLLM fork serves the LLM and the TTS, coordinated by Triton, with a WebSocket facing the client. Audio moves in chunks that are multiples of 80 ms.
- Tool calls:
  1. On `<SOTC>`, the runtime decodes the full call asynchronously ("fast-decode").
  2. While the tool executes, the TTS speaks a per-tool `ack_message` (p.6, p.18).
  3. The result is force-inserted into the function channel and context ("fast-inject").
- During execution, user audio is transcribed but ignored, so **there is no barge-in during tool calls** (p.15).

**How the NVIDIA papers in this batch relate.**
- **VoiceChat-TTS (arXiv 2608.13831) is VoiceChat's speech decoder.** VoiceChat cites it as an anonymous manuscript ("V-Model-TTS", ref [20]; §2.3, p.4). The TTS paper says its checkpoint ships inside NVIDIA-NemotronLabs-VoiceChat-11B. The two are trained independently, and no gradients flow between them (p.5).
- **Lineage of the STT core.** The paper says it resembles the architecture of NVIDIA's earlier duplex models: SALM-Duplex [16] and the ASRU-2025 open full-duplex voice-agent paper [17] (p.3). The TTS paper describes [17] as a dedicated speech decoder attached to the backbone through latent representations (VoiceChat-TTS p.1). VoiceChat explicitly disables direct audio-codec prediction (audio-loss weight 0.0, p.5).
- **PersonaPlex** comes from a different NVIDIA author group (Roy, Catanzaro et al.) and is a Moshi fine-tune. Here it is only a baseline (Tab.1, Tab.3).
- **FDB-v3** is a joint NTU–NVIDIA benchmark. Its NVIDIA authors (Chen Chen, Zhehuai Chen) are VoiceChat contributors.

## Training pipeline

The STT backbone, the RNN-T branch and the TTS are trained separately and combined only at inference (p.5).

### Stage 0: tokenizer / codec / decoder training

- **Text tokenizer:** Nemotron Nano 2 subword tokenizer, shared by the LLM and the TTS (p.4).
- **Speech encoder:** the 600M encoder cites [19] (about 530k hours according to that reference, p.12). Reusing its weights is implied, not stated. Its training data and compute are not reported in this paper.
- **TTS and codec:** trained separately: 1.4M single-turn steps on 8×A100, then 200k duplex fine-tuning steps on 32×H100 (VoiceChat-TTS p.4; see voicechat-tts.md).
- **LLM:** initialized from Nemotron-Nano-9B-v2-Base (p.3).

### Stage 1: Continued pretraining (CPT)

- **Objective:** next-token prediction on agent-side text only, conditioned on user speech and earlier agent text (p.5).
  - Loss: L_CPT = 3·L_text + L_FC (λ_T2T = 0) (p.6).
  - Agent-text token weights: BOS 10, EOS 10, text 1.0, PAD 0.5 (p.6).
  - CPT contains no tool data. The function channel is still trained to output PAD, which discourages spurious tool calls (p.6).
- **Trainable modules:** the "full-duplex STT backbone". The audio head is off (audio-loss weight 0.0). Whether the encoder is frozen is **not stated**. The RNN-T head has to be retrained afterwards, which hints that the encoder changed, but that is our inference (p.5).
- **Data:**
  - Speech-text pretraining bucket, weight 0.95, with scale "≈530k". **No unit is given**, and the number matches the encoder's 530k-hour figure in [19].
  - Single-turn QA, about 12k hours, weight 0.05 (p.15, Tab.5).
  - Both are almost entirely TTS-rendered (p.14). The only augmentation is mild additive noise (p=0.1) (p.14).
- **Sequence length:** audio context of at most about 2 minutes (p.9), i.e. at most 1,500 frames (derived).
- **Optimization (p.6):** AdamW (β1=0.9, β2=0.98), weight decay 0, LR 5e-5 with inverse-square-root schedule, 2,500 warmup steps, minimum LR 5e-6, gradient clipping 2.0, bf16, 64 GPUs (8 nodes × 8) with full data parallelism.
- **Not reported:** GPU type, batch size, steps, tokens seen, wall-clock time.

### Stage 2: Supervised fine-tuning (SFT)

Purpose: teach instruction following, turn-taking, barge-in recovery, tolerance of user backchannels, and tool calling (p.5). It resumes from the CPT checkpoint.

- **Objective:** L = 0.5·(L_text + L_FC) + 0.5·L_T2T, where L_T2T is plain next-token loss on text-only data (p.6). Token weights (p.6):

  | Channel | BOS / `<SOTC>`,`<EOTC>` | EOS / `<EOTR>` | Content | PAD |
  |---|---|---|---|---|
  | Agent text | BOS 12.5 | EOS 7.5 | 5.0 | 1.0 |
  | Function | 6.0 each | 3.0 | 64.0 (`<TOOLCALL>` content) | 0.3 |

  Tool-response tokens are masked out of the loss.
- **Data:** weighted randomized round-robin over the buckets in Table 5 below (p.5, p.15).
- **Augmentations:** online duplex augmentations (see Data pipeline).
- **Optimization:** same as CPT except gradient clipping 5.0 (p.6).
- **Not reported:** batch size, steps, tokens seen, wall-clock time.

### Stage 3: RNN-T user-transcription head

- Trains only the RNN-T prediction and joint networks, with the standard transducer loss.
- Frozen: encoder, LLM, agent-text head, function head and TTS (p.5).
- Data and compute: not reported.

### Inference-time additions (not trained)

- Per-tool filler (`ack_message`) messages while a tool runs.
- An endpointing fallback for when the model does not handle turn-taking or barge-in natively: heuristics on RNN-T speech/silence activity plus the current generation state force-inject BOS/EOS (p.6).

## Data pipeline

**CPT pseudo-dialogues (p.5, p.14).**
1. Take plain-text passages from LLM pretraining corpora. The paper cites Nemotron-CC [25].
2. Split them into sentences and assign the sentences alternately to user and agent.
3. A turn ends after one sentence with 80% probability; extra sentences are added with decaying probability. A role switch is forced once a turn passes 200 words.
4. Synthesize each turn with a voice-cloning TTS using two fixed speaker prompts, one per role. The TTS engine is **not named**.
5. Place the turns on a shared timeline in dialogue order and concatenate each role's audio into its own channel, giving two-stream audio.

The training targets are only the agent-side text.

**SFT corpora (p.14–15).**
- Also drawn from text corpora of the text-only Nemotron backbone, not from natively recorded conversations. They are TTS-rendered onto a two-channel timeline with realistic gaps between turns; the timing model is not described.
- One bucket stays as text: the extract-knowledge shards.
- In Table 5, Scale is the approximate size of the shards each bucket indexes, not the amount actually consumed.

| Bucket | Scale | CPT weight | SFT raw weight (normalized share, derived) |
|---|---|---|---|
| Speech-text pretraining (a different collection in each stage) | ≈530k (no unit) | 0.95 | 0.40 (34.0%) |
| Text-only extract-knowledge | text shards | – | 0.10 (8.5%) |
| Spoken MCQ | ≈24k h | – | 0.03 (2.6%) |
| Single-turn QA | ≈12k h | 0.05 | 0.02 (1.7%) |
| Duplex chat | ≈72k h / 4.3M dialogues | – | 0.15 (12.8%) |
| Natural voice conversations | ≈4.2k h / 45k conversations | – | 0.08 (6.8%) |
| VoiceBench-targeted | ≈3.8k h / 37k conversations | – | 0.04 (3.4%) |
| Voice-agent instruction following | IF shards | – | 0.01 (0.9%) |
| Tool calling (all) | ≈7.0k h / 268k conversations | – | 0.305 (26.0%) |
| Spoken safety alignment | ≈1.1k h / 48k conversations | – | 0.04 (3.4%) |

Notes on the table:
- The group shares are given in the paper: retention 46.8%, conversational 23.8%, tool calling 26.0%, safety 3.4% (p.14).
- The speech buckets other than the pretraining bucket index about 124k hours in total (derived).
- Average dialogue lengths (derived): duplex chat about 60 s; tool calling about 94 s; natural and VoiceBench-targeted conversations about 336 s and 370 s. The last two are longer than the roughly 2-minute training window, so they must be cropped somehow; the paper does not say how.

**Tool-calling data (p.14).**
- Rendering text tool transcripts directly does not work. They contain URLs, markdown and code that have no spoken form, and models trained on them do poorly on tool relevance.
- Instead, a multi-agent pipeline generates scenarios, turn plans, executable tool backends and simulated conversations.
- A "voice adaptation" step follows: filtering, normalization, TTS with diverse reference voices, ASR round-trip verification and dialogue assembly.
- The mixture covers tool use, abstention, open-domain chat, greetings and user interruptions.

**Duplex augmentations (SFT only, applied online; p.14–15).**
- **Early interruption (p=0.1):** an agent turn is cut mid-utterance and continues 8 frames (640 ms) into the next user turn before EOS. Shards that already contain rendered interruptions opt out.
- **Backchannel injection:** p=0.05 per sample and 0.5 per agent turn. Recorded "uh-huh"-type clips are loudness-matched into the user channel during agent speech, so the model learns not to treat them as interruptions.
- **Text-channel delay:** agent-text targets are shifted 2 frames (160 ms) later, so the model hears more user audio before committing to each token. The function channel is **not** shifted.
- **Acoustic augmentation:**
  - DNS5 + DEMAND noise at p=0.5 (p=0.1 in CPT), SNR −30 to 60 dB.
  - Room impulse responses p=0.8, microphone impulse responses p=0.6, codec augmentation p=0.1. All three are off in CPT.

**Real versus synthetic.**
- Explicitly real: recorded backchannel clips, noise and impulse-response corpora.
- Not stated: whether the "Natural voice conversations" bucket is real recordings.
- Everything else is described as TTS-rendered.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Speech encoder (inherited) | ~530k h (ref [19]) | not reported | nemotron-speech-streaming-en-0.6b |
| VoiceChat-TTS pretraining | ≈70.2k h single-turn TTS | 8×A100 for 1.4M steps; wall-clock not reported | VoiceChat-TTS p.3–4 |
| VoiceChat-TTS fine-tuning | mixture incl. ≈2.5k h synthetic duplex | 32×H100 for 200k steps | VoiceChat-TTS p.4 |
| CPT | "≈530k" (unit not given) + ≈12k h QA | 64 GPUs (type not reported); steps not reported | bf16, LR 5e-5 |
| SFT | ≈124k h indexed speech (derived) + text shards | 64 GPUs; steps not reported | gradient clipping 5.0 |
| RNN-T head | not reported | not reported | everything else frozen |
| Inference | — | 1× H100 PCIe 80GB: 4 concurrent streams, p95 118 ms per 160-ms chunk (1.36× real time) | p.17 |

## Preserving text intelligence / knowledge

- **The LLM never models acoustics.** The audio loss is 0, so the 9B backbone predicts only text and control tokens. Our reading is that none of its capacity goes to producing speech.
- **Knowledge-bearing CPT.**
  - The "speech" task is speech-conditioned *next-sentence prediction* over pretraining text: hear user sentence(s), then write the next agent sentence.
  - The supervision is in the text domain, where the LLM's knowledge lives.
- **Retention data in SFT.**
  - About 47% of SFT sampling is retention data (p.14).
  - It includes text-only extract-knowledge shards with λ_T2T = 0.5, which the paper credits with keeping text-domain competence through an otherwise speech-dominated stage (p.14).
  - It also includes spoken MCQ (≈24k h) and single-turn QA (≈12k h).
- **Speech path preserved.** The RNN-T transcript is kept out of the LLM input (p.3).
- **Residual gap, as admitted.** The authors acknowledge that knowledge, instruction following, reasoning and safety may be weaker than in the Nemotron-Nano-9B-v2 backbone (p.9). **No text-benchmark comparison against the backbone is given.**
- **VoiceBench profile (Tab.3, p.8).**
  - Strong on multiple-choice subsets: OBQA 61.3, MMSU 46.1, BBH 51.7.
  - Weak on SD-QA (32.6), open-ended quality (CommonEval 3.0, AlpacaEval 3.4, WildVoice 2.8) and IFEval (19.3).
  - Freeze-Omni, whose LLM is frozen per its reference title, does better on SD-QA (53.5), open-ended quality and IFEval (23.4). So fine-tuning the LLM gained MCQ accuracy but did not protect open-ended QA.

## Key results

**FDB 1.0** (Tab.1, p.7). TOR is in %, latency in seconds, quality is GPT-4o on a 0–5 scale. The backchannel track is **not reported**.

| Model | Pause TOR synth↓ | Pause TOR CANDOR↓ | Smooth-turn TOR↑ | Smooth-turn latency↓ | Interruption TOR↑ | Interruption quality↑ | Interruption latency↓ |
|---|---|---|---|---|---|---|---|
| VoiceChat | 15.3 | 25.5 | 81.5 | 0.448 | 100.0 | 4.33 | 0.480 |
| PersonaPlex (released) | 35.8 | 43.1 | 90.8 | 0.170 | 95.0 | 4.29 | 0.240 |
| Moshi | 98.5 | 98.0 | 94.1 | 0.265 | 100.0 | 0.77 | 0.257 |
| Freeze-Omni | 64.2 | 48.1 | 33.6 | 0.953 | 86.7 | 3.62 | 1.409 |
| Gemini Live 2.0 | 25.5 | 31.0 | 65.5 | 1.301 | 89.1 | 3.38 | 1.183 |
| GPT-Realtime | 1.0 | 12.0 | 100.0 | 1.470 | 97.0 | 3.85 | 1.500 |

**FDB 1.5, user backchannel** (Tab.2, p.8): Respond / Resume / Uncertain / Unknown = 1 / **93** / 2 / 4%. Gemini Live 2.0 has the identical split. Freeze-Omni Resumes 80%, GPT-4o Realtime 70%, Moshi 6%.

**VoiceBench average** (Tab.3, p.8):

| System | Average |
|---|---|
| VoiceChat | 55.1 |
| Freeze-Omni | 55.2 |
| PersonaPlex | 30.6 |
| Moshi | 29.5 |
| DuplexCascade | 65.4 |
| MiniCPM-o 4.5 (different judge) | 76.1 |

AdvBench refusal: VoiceChat 100.0, Moshi 44.2, PersonaPlex 8.1.

**FDB-v3 tool calling** (Tab.4, p.9):

| System | Tool-selection F1 | Argument accuracy | Pass@1 |
|---|---|---|---|
| VoiceChat | 82.5 | 42.2 | 33.0 |
| Gemini Live 2.5 | 78.6 | 59.3 | 49.0 |
| Gemini Live 3.1 | 81.7 | 58.8 | 54.0 |

The paper leaves out GPT-Realtime from the same benchmark source. It scores 87.6 / 68.0 / 60.0 (FDB-v3 paper, Tab.2).

**Other evaluations:**
- **ASR (RNN-T branch), OpenASR leaderboard average WER:** 9.02% with 80-ms chunks, 8.28% with 160-ms chunks (Tab.7, p.17).
- **TTS on LibriTTS, unseen speakers:** WER 2.00% at turn 1 and 2.20% at turn 4. SECS falls from 0.757 to 0.685 (Tab.6, p.16).

## Ablations & lessons

None of these are controlled ablations. Each bullet gives the strongest evidence the paper offers.

- **A text-only duplex LLM plus a dedicated TTS reaches far better knowledge than Moshi-style joint models.**
  - VoiceBench: 55.1 versus 29.5 (Moshi) and 30.6 (PersonaPlex) (Tab.3).
  - This is confounded by backbone and data.
  - The companion TTS paper adds direct evidence: on identical text, a dedicated decoder halves WER compared with PersonaPlex's own audio (5.00 → 2.42; VoiceChat-TTS Tab.2).
- **The mixture shapes the capability profile.**
  - The large MCQ-style buckets (24k h spoken MCQ, a VoiceBench-targeted bucket) line up with strong OBQA and MMSU.
  - Instruction following, at only 0.9% of SFT sampling, lines up with IFEval 19.3, versus 43.4 for the cascade (Tab.3, Tab.5).
  - This is our inference; no ablation tests it.
- **Negative supervision on idle channels is load-bearing, by the authors' account.**
  - The function channel predicts PAD in CPT and SFT (PAD weight 0.3 in SFT), and the authors call it important against spurious calls (p.3, p.6).
  - No numbers are given.
- **Naively rendered text tool data fails.** Models trained on text tool-calling transcripts do poorly on tool relevance (p.14, qualitative). The fix is voice-native tool dialogues with ASR round-trip verification.
- **The turn-taking recipe works but trades responsiveness for restraint.**
  - It combines heavy BOS/EOS weights, interruption and backchannel augmentation, and a 160-ms text delay.
  - Wins: the lowest open-weight pause takeover, 100% interruption takeover, 93% backchannel Resume.
  - Losses against PersonaPlex: smooth-turn takeover 81.5 vs 90.8, latency 448 vs 170 ms (Tab.1).
  - The 160-ms text delay plus the TTS's one-step audio lag plausibly add latency (our inference).
  - The model still needs an RNN-T endpointing fallback in deployment (p.6). The paper does not say whether that fallback was active during the benchmark runs.
- **Tool routing is learnable from synthetic data; argument grounding is not yet.**
  - Tool-selection F1 is 82.5, but argument accuracy is 42.2 and Pass@1 33.0 (Tab.4).
  - Against the FDB-v3 paper's own table, that Pass@1 is below every system there; the lowest is Ultravox at 41.0 (cross-paper comparison).
  - Practical limits stated in the paper (p.9): use at most 5 tools, don't rely on simultaneous multi-tool calls, and no barge-in during tool execution.
- **Dedicated spoken safety data works.** A 3.4% safety bucket (≈1.1k h) gives 100% AdvBench refusal, versus 8.1 for PersonaPlex, whose paper reports no safety data (Tab.3, Tab.5).
- **Context length is a hard limit.** Training used at most about 2 minutes of audio, and information beyond that window may not be retained reliably (p.9).
- **Robustness remains weak** under strong noise, reverberation and competing speech, despite heavy augmentation (p.9).
- **ASR chunk size:** going from 80 to 160 ms lowers average WER from 9.02 to 8.28% (Tab.7).
- **Zero-shot speaker identity drifts across turns:** SECS 0.757 → 0.685 over 4 turns (Tab.6).

## Implications for our setup

Our setup: Qwen3 LLM backbone, Qwen3 audio tokenizer and its vocoder/decoder, mid-trained on about 10K hours of duplex conversational speech. Problems: weak audio continuation and weak semantic knowledge in the generated speech.

1. **Find out whether the problem is semantics or acoustics.**
   - If our model has a text (inner-monologue) channel, score that text on VoiceBench-style QA separately from the audio.
   - If the text is good and the audio is poor, VoiceChat's answer is to take audio generation away from the LLM: audio loss 0, plus a streaming duplex TTS fed by the LLM's text and control tokens.
   - On our stack, that could mean adapting Qwen3-TTS-12Hz (same 12.5 Hz regime; WER 1.01 single-turn per VoiceChat-TTS Tab.1) to VoiceChat-TTS's duplex conventions.
   - The cost is that output prosody no longer reacts to the user's audio.
2. **Don't expect 10K hours of duplex speech to supply knowledge. VoiceChat doesn't get knowledge from real dialogue either.**
   - Its knowledge comes from TTS-rendering text pretraining corpora into pseudo-dialogues, at a scale reported as "≈530k" (unit not given) at 95% of CPT.
   - We can build the same thing cheaply: alternate sentences between roles, 80% one-sentence turns, a forced switch at 200 words, two cloned voices, loss on agent text only.
   - Our real 10K hours then become a behavior/naturalness bucket. VoiceChat's "natural voice conversations" bucket is only 4.2k h at a 6.8% share.
3. **Keep the text LLM alive during speech training.**
   - About 47% retention data in SFT, including 8.5% text-only shards with λ_T2T = 0.5.
   - LR 5e-5 with inverse-sqrt decay.
   - These are cheap to copy and are the only explicit retention mechanisms the paper reports.
4. **Turn-taking supervision to port directly**, assuming our tokenizer also runs at 12.5 Hz:
   - Frame-level BOS/EOS/PAD targets with boundary-token weights of about 10× (SFT: 12.5 / 7.5 / 5 / 1).
   - A 160-ms text delay.
   - Early-interruption augmentation (p=0.1, 640-ms overlap) and backchannel injection (p=0.05 per sample, 0.5 per agent turn).
   - Noise/RIR/microphone/codec augmentation.
   - Our real duplex data already contains overlaps. These tools show how to scale duplex behavior onto synthetic, turn-based knowledge data.
5. **Compute.** Both stages fit on 64 GPUs. The paper gives no step counts, so we cannot budget from it directly.
6. **Context length.** Plan for training sequences of at least 2 minutes (1,500 frames at 80 ms). VoiceChat's 2-minute cap is a stated limitation.
7. **Evaluation suite worth adopting:**
   - FDB 1.0 (pause, smooth turn-taking, interruption) and FDB 1.5 (backchannel).
   - VoiceBench, read per subset: MCQ versus open-ended.
   - FDB-v3, only if tools are in scope.
   - TTS WER / SECS / SQuIM-MOS across multiple turns.

## Open questions / unclear points

- **Data scale and units.**
  - The "≈530k" scale of the speech-text pretraining bucket has no unit, and it equals the hour count of the encoder's training data in [19]. Is it hours, samples, or a copy-over?
  - The Table 5 caption says CPT and SFT draw that bucket from *different* collections, but only one scale is given.
- **Training details not reported:** GPU type, global batch, steps per stage, total tokens or hours consumed, epochs, and which TTS rendered the CPT/SFT data and how inter-turn timing was sampled.
- **Is the encoder trained or frozen during CPT/SFT?** Is it initialized from [19]? Is the LLM fully fine-tuned? The paper implies full fine-tuning but never states it.
- **Is the "Natural voice conversations" bucket real recorded speech?** How are the ~5–6-minute average conversations fit into the 2-minute window?
- **Does EOS from the LLM map to the TTS interruption token?** And was the RNN-T endpointing fallback enabled in the FDB runs?
- **FDB coverage gaps.**
  - The FDB 1.0 backchannel track (does the agent itself produce backchannels?) is not reported. The training data is TTS-rendered turn-based text, so the agent probably does not backchannel.
  - The FDB-v3 latency, turn-take and interruption metrics are not reported, nor whether the FDB-v3 harness (LiveKit) was used.
- **Possible benchmark targeting.** A "VoiceBench-targeted" bucket exists (≈3.8k h), but no contamination or overlap analysis is given.
- **No text-domain evaluation** to quantify how much backbone knowledge was lost.
