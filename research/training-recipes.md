# Training recipes compared: architecture, data and compute by stage

This is the comparison reference for `synthesis.md`. Every number comes from the per-paper notes in `papers/` or the web notes in `web/`; those notes carry page and table citations. In the tables:

- **n.r.** = not reported.
- **(d)** = derived by us from reported numbers.
- **≈** = approximate as printed in the source.

Rows are ordered roughly by relevance to our setup: full-duplex systems first, then speech-text foundation models, then generation-side models.

---

## 1. Architecture

### 1a. Full-duplex and real-time systems

| System (org, date) | Backbone (init) | Audio in | Audio out | Who generates audio | Text channel | Duplex mechanism |
|---|---|---|---|---|---|---|
| **StepAudio 3 Realtime** (StepFun, Sep 2026) | 196B MoE / 11B active (Step 3.7 Flash) | Qwen3-Omni AuT, continuous; encodes *both* user and own audio | n.r. (separate streaming "generator") | Separate generator; LLM emits one state or text token per 320 ms block | Yes (text/state tokens) | Two audio streams time-interleaved in 320 ms blocks, state tokens (`listening: no_voice`, `user_voice`, `speaking transition`) |
| **NemotronLabs VoiceChat** (NVIDIA, Oct 2026) | Nemotron-Nano-9B-v2-**Base** (hybrid Mamba-Transformer) | 600M streaming FastConformer, continuous 12.5 Hz | VoiceChat-TTS: 22 kHz codec, 12.5 Hz × 31 RVQ, MoG head | **Separate 977M TTS**; LLM audio loss = 0 | Text only, plus a parallel function-call channel | Time-aligned 80 ms streams; per-frame BOS/EOS/PAD targets; 160 ms text delay |
| **Raon-SpeechChat** (KRAFTON, May 2026) | Qwen3-VL-8B-Instruct (8.2B LLM; 9.8B total) | Voxtral-Realtime causal encoder, 12.5 Hz | Mimi 12.5 Hz, 16 of 32 RVQ | 4-layer "speech-gen expert" on LLM hidden → semantic code; 5-layer depth transformer (init from Qwen3-Omni) → 15 acoustic | Yes, 1 frame ahead | One AR sequence per 80 ms frame: [user emb, text/state token (SIL/BOW/word/PAD/BC), agent speech emb] |
| **MiniCPM-o 4.5** (OpenBMB, Apr 2026) | Qwen3-8B via MiniCPM-V 4.5 (9.34B total) | Whisper-Medium streaming → 10 Hz | CosyVoice S3 25 Hz × 1 cb (6,562) | Separate 0.3B speech-token decoder fed LLM hidden + text | LLM emits only text (3–4 tok/s) | Time-slicing, 1.0 s chunks; explicit listen/speak control token |
| **Covo-Audio-Chat-FD** (Tencent, Feb 2026) | Qwen2.5-7B-Base | Whisper-v3 → 6.25 Hz adapter | WavLM-VQ 25 Hz × 1 cb (16,384) | Same LLM (interleaved) | 1 text token : 4 speech tokens per chunk | Chunk-interleaved user/model 1:4 (0.16 s); THINK / SHIFT / BREAK state tokens |
| **PersonaPlex** (NVIDIA, ICASSP 2026) | Moshi 7B | Mimi tokens | Mimi 12.5 Hz | Same LM + depth transformer | Inner monologue | Moshi dual-stream + hybrid system prompt (voice prompt on agent audio, text prompt on agent text) |
| **Moshi** (Kyutai, Oct 2024) | Helium 7B (own text LLM, 2.1T tokens) | Mimi tokens | Mimi 12.5 Hz × 8 (1 semantic + 7 acoustic) | Same LM (temporal) + 6-layer depth transformer | Inner monologue, time-aligned, text first in each frame | Two parallel streams, 17 tokens per 80 ms step |
| **Chronological Thinking** (NTU + StepFun, Oct 2025) | Qwen2.5-1.5B-Instruct | Streaming encoder 12.5 Hz | NanoCodec FSQ 12.5 Hz | Separate AR decoder from LLM hidden + text | LLM predicts text only | SALM/Moshi-style dual stream; thinking chain replaces trailing `<SIL>` while listening |
| **SyncLLM** (Meta, 2024) | Llama-3-8B | HuBERT 25 Hz units (dedup) | HuBERT units → HiFi-GAN | Same LM | Sentence-level interleave in pretraining only | Time-division multiplexing, 160–240 ms chunks; predicts the user's chunk too |
| **MinMo** (Alibaba, Jan 2025) | Qwen2.5-7B-Instruct (frozen + LoRA) | SenseVoice encoder | CosyVoice 2 tokens via separate 370M LM | Separate | Text-first (5 LLM vectors : 15 speech tokens) | External 18M duplex predictor on frozen LLM states |
| **Freeze-Omni** (Tencent, Nov 2024) | Qwen2-7B-Instruct (fully frozen) | 350M streaming encoder 12.5 Hz | TiCodec 40 Hz × 1 cb | Separate 120M decoder | Text | VAD + 3-state head |
| **TML-Interaction-Small** (Thinking Machines, May 2026) | 276B MoE / 12B active, **from scratch** | dMel (encoder-free) | Flow head | Same model | Yes | 200 ms micro-turns + async "background model" |
| **GPT-Live-1** (OpenAI, Sep 2026) | undisclosed | undisclosed | undisclosed | "frontend" S2S model | — | Full-duplex frontend + delegation to a text backend (results injected as ≤500-token text) |
| **Gemini 3.8 Live** (Google, Sep 2026) | "based on Gemini 3 Pro" | ~25 audio tok/s (Live) | undisclosed | — | Thinking (Extended Thinking variant) | Server VAD (~800 ms) + barge-in truncation + "proactive audio" gate; no stated listen-while-speaking |

### 1b. Half-duplex speech-text foundation models (where continuation and knowledge come from)

| System | Backbone (init) | Audio in | Audio out | Multi-codebook / generation head | Decoder |
|---|---|---|---|---|---|
| **Qwen3-Omni** (Sep 2025) | Thinker 30B-A3B MoE (Qwen3); Talker 3B-A0.3B | AuT continuous 12.5 Hz | **12.5 Hz × 16 × 2048** | Talker predicts cb0; 80M MTP (5 layers) predicts cb1–15 autoregressively; input = sum of 16 cb embeddings | Code2Wav 200M, causal |
| **StepAudio 3 Gen** (Sep 2026) | unnamed pretrained text LLM | **12.5 Hz × 16 × 2048** RVQ (SSL + acoustic fused) | same | LLM predicts cb0 in the text softmax; **4-layer depth transformer** predicts cb1–15; zero-init RVQ adaptor on summed cb embeddings | Causal Vocos-style + ISTFT |
| **MiMo-Audio** (Xiaomi, ~Aug 2025) | MiMo-7B-Base | 25 Hz × 8 RVQ, patched ×4 → 6.25 Hz | same | 16-layer patch decoder, delay pattern | Causal transformer + Vocos |
| **Kimi-Audio** (Apr 2025) | Qwen2.5-7B | 12.5 Hz semantic tokens + Whisper features | 12.5 Hz semantic tokens | Parallel text and audio heads, audio delayed 6 tokens | Chunked flow matching + BigVGAN |
| **Step-Audio 2** (Jul 2025) | unnamed (mini: Qwen2.5-7B) | 25 Hz → 12.5 Hz continuous | CosyVoice 2 tokens | Interleaved with text | Flow matching + HiFi-GAN |
| **GLM-4-Voice** (Dec 2024) | GLM-4-9B-Base | 12.5 Hz single-cb supervised semantic tokens | same | Text-first: 13 text : 26 speech | Flow matching + HiFi-GAN |
| **Audex** (NVIDIA, Jun 2026) | Nemotron-Cascade-2 30B-A3B | AF-Whisper 25 Hz | X-Codec2 50 Hz × 1 (FSQ 65,536) | Flat in extended vocab | Streaming ConvNeXt |
| **CALM** (Kyutai, Sep 2025) | Helium-1 2B | continuous VAE 12.5 Hz (32-dim) | same | **10M MLP consistency head** (continuous) vs depth-RVQ baseline | VAE decoder |

### 1c. Generation-side models (what makes good audio from an LLM)

| System | Backbone | Tokenizer | Generation head | Data |
|---|---|---|---|---|
| **Qwen3-TTS** (Jan 2026) | Qwen3 0.6B / 1.7B | **Qwen-TTS-Tokenizer-12Hz** (12.5 Hz × 16 × 2048) or 25 Hz × 1 × 32,768 (unreleased) | MTP 5 layers | >5M h, 10 languages |
| **Breeze-TTS-2** (BreezeBlue, 2026) | Qwen3-1.7B-shaped backbone + T5Gemma-2 text encoder | **Qwen3-TTS-Tokenizer-12Hz** | 12-layer depth decoder | n.r. (#1 open-weight on the AA Speech Arena) |
| **Sesame CSM** (2025) | Llama-architecture 1B/3B/8B (not text-LLM init) | Mimi | Small decoder; trained on 1/16 of frames | ~1M h × 5 epochs |
| **MOSS-TTS** (Mar 2026) | 8B (delay pattern) / 1.7B (local transformer) | 12.5 Hz × 32 RVQ | **1.7B local transformer beat the 8B delay model** | "millions of hours" |
| **Voxtral TTS** (Mar 2026) | Ministral 3B (text embeddings frozen) | 12.5 Hz: 1 semantic VQ + 36 FSQ dims | AR semantic + **flow-matching acoustic head** (beat depth transformer and MaskGIT) | n.r. |
| **VoxCPM2** (Jun 2026) | MiniCPM-4-1B | continuous VAE 25 Hz, patched → 6.25 Hz | residual acoustic LM + local DiT | >2M h |
| **VibeVoice** (Aug 2025) | Qwen2.5 1.5B / 7B | σ-VAE 7.5 Hz continuous | 4-layer diffusion head | n.r. |
| **VoiceChat-TTS** (NVIDIA, Aug 2026) | 778M Gemma-3-based | 12.5 Hz × 31 RVQ | MoG head | 70.2k h + 2.5k h synthetic duplex |

---

## 2. Data and compute by stage

"Duplex data" means two-channel or time-aligned interaction data. Speech-text pretraining covers ASR, TTS, interleaved, continuation and audio-only tasks.

| System | Speech-text pretraining / mid-training | Text share | Duplex-specific data | SFT | RL / preference | Compute |
|---|---|---|---|---|---|---|
| **StepAudio 3 Realtime** | 1.2T tokens at 32K, 3 stages, then mid-training at 128K | "increased" (n.r.) | **>10K h synthetic** in mid-training, plus streaming ASR/VAD/completeness targets; text mixed in | 100K curated audio-QA (beat 2M random); self-play dialogues; agent trajectories; 4 teachers merged 3:1:1:1 | none described | n.r. |
| **NemotronLabs VoiceChat** | CPT on TTS-rendered pseudo-dialogues from text pretraining corpora ("≈530k", **unit not printed**) + ~12k h QA; loss on agent text only | 0 in CPT; 8.5% text-only (T2T loss 0.5) in SFT | ~72k h duplex chat (4.3M dialogues, synthetic); online interruption/backchannel augmentation | 10-bucket round robin: 47% retention, 24% conversational, 26% tools (7k h), 3.4% safety (1.1k h) | none | 64 GPUs per stage, LR 5e-5; steps n.r. |
| **Raon-SpeechChat** | 1.01M h / 69.98B tokens (STT, TTS, SpeechQA, SpokenQA, TextQA); ~132.5B tokens seen (d) | n.r. | **119.5k h = 13.2k real + 106.3k synthetic** (Qwen3 dialogues voiced by Qwen3-TTS); duplex pretraining 10.85B tokens + 10% replay; duplex FT 1.43B tokens | 104k h / 11.2B tokens, 800 steps | SimPO + SFT on S2T/QA (no duplex RL) | n.r.; LR 1.2e-5 end-to-end |
| **MiniCPM-o 4.5** | "millions of hours", fixed-ratio joint omni pretraining incl. FD data | n.r. (text instruction data included) | n.r. | 2-phase SFT, sizes n.r. | GRPO + RLAIF-V (text side) | n.r. |
| **Covo-Audio** | ~8M h, ~2T tokens: 800B text, 540B interleaved, 240B audio-only, 160B TTS, 160B continuation, 80B ASR, **5B full-duplex** | **40%** | 5B tokens (0.25%) in pretraining (half-duplex TTS dialogues → dual channel with barge-ins and backchannels), then SFT | 50K steps; 40% general intelligence with top-20-logit KL from a stronger text LLM | GRPO (audio understanding only) | n.r. (~500K steps × ~4M tokens (d)) |
| **PersonaPlex** | inherits Moshi | — | 2,250 h synthetic (+1,217 h Fisher in release) | — | none | **48 A100-hours** (24.6k steps × batch 32 × 2,048) |
| **Moshi** | **7M h**, 1M steps × 16 h batches (~8M h processed); then 100k steps of diarization-simulated two-stream | **50%** of steps text-only (separate optimizer); 10% in two-stream stage | Fisher 2k h (10k steps); **>20k h synthetic instruct dialogues** (30k steps) | (the synthetic dialogues *are* the SFT) | none (RL added later by the alignment paper) | H100 FSDP; GPU-hours n.r. |
| **Moshi + interactivity RL** (Kyutai/Gradium, Jun 2026) | — | — | ≤2,000 VAD-mined segments per behavior from Fisher or Seamless | — | **GRPO**, 16 rollouts, LR 2e-7, VAD rewards + LLM-judge content reward | **32 H100** |
| **SyncLLM** | ~193k h Bark-TTS'd text dialogues, sentence-level interleave, speech share 20→85% (~42B tokens) | 80% → 15% over the stage | ~20k h synthetic two-channel (3.1B tokens) + Fisher 1.9k h (0.5B tokens) | — | none | 128 A100; 40k × 1M-token steps |
| **Chronological Thinking** | none | 0 | 15.2k h synthetic knowledge-dense dialogues (Qwen2.5-72B scripts, Step-Audio-TTS) | (same) | none | 8× L40S |
| **MinMo** | ~1.26M h S2T alignment (LLM frozen); ~1.44M h total | n.r. | 4k h (~3k real + ~1k simulated) for the 18M predictor | ~1.3M samples; voice decoder 170k h TTS + 10k h dialogue | none | n.r. |
| **Qwen3-Omni** | S2 ~2T tokens: text 0.57T, audio 0.77T, image 0.82T; S3 8K→32K | **~25%** | none | Thinker SFT + strong-to-weak distillation; Talker "hundreds of millions" of samples | GSPO (Thinker), DPO (Talker) | n.r. |
| **MiMo-Audio** | **>100M h**: Stage 1 2.6T tokens (text-loss only), Stage 2 5T tokens | **46–52%**; text loss weight 100 vs ≤12 per codebook | none | 100B tokens | none | n.r. (batch 16.8M tokens) |
| **Kimi-Audio** | ~13M h real audio; 585B audio + 585B text tokens; 3 interleaving variants | **50%** | none | ~300k h | none (on the LLM) | n.r. |
| **Step-Audio 2** | 1.356T tokens / 21 days; "680B text + 8M h audio" | **50%** after alignment | none | 4B tokens | PPO → PPO+RM → GRPO | 21 days (GPUs n.r.) |
| **StepAudio 3 Gen** | ~2.7T tokens, 4 stages; tokenizer on ~700k h | **50% from Stage 2** | none | ~5k h (incl. 1.5k h natural conversational) | GRPO, ASR-gated reward | n.r. |
| **GLM-4-Voice** | 1T tokens: 455B synthetic-interleaved speech + 279B text (from text corpora via text-to-token) + 700k h real (31B) | ~30% text-only | none | n.r.; speech-output loss trained 20 epochs vs 4 for text | none | n.r. |
| **Audex** | no audio pretraining; 1.09M h / 157.4B audio tokens SFT | **44% → 59% → 69%** | none | (all stages are SFT) | text-only Cascade RL | 512 H100 |
| **CALM** (speech continuation) | ~1.8M h-equivalent seen (d) | n.r. | — | — | — | 48 H100 × 150k steps |
| **DSM** | 2.5M h pseudo-labelled (whisper-timestamped) | 0 | — | 28k h GT ASR FT | — | 48 H100 (ASR), 32 H100 (TTS) |

**Duplex-specific data in context.** Where both numbers are reported, the duplex data is a thin layer over the speech-text stage before it: Covo 0.25% of tokens; Raon ~12% of tokens (15.7B on ~132.5B (d)); Moshi ~0.3% of hours. StepAudio 3 Realtime's duplex data (>10k h) is about the size of our corpus, but it sits on 1.2T tokens of speech-text pretraining.

| System | Duplex data | Foundation underneath |
|---|---|---|
| Moshi | 2k h real + 20k h synthetic | 7M h |
| SyncLLM | 2k h real + 20k h synthetic | 193k h |
| PersonaPlex | 2.25k h (+1.2k h Fisher) | Moshi |
| MinMo | 4k h | 1.4M h |
| Covo | 5B tokens (hours n.r.; layout-dependent) | 2T tokens |
| **StepAudio 3 RT** | >10k h synthetic | 1.2T tokens |
| Chrono | 15k h synthetic | none (1.5B model, Llama Q ~31) |
| VoiceChat | ~72k h synthetic | CPT on rendered text corpora |
| **Raon** | 119.5k h (11% real) | 1.01M h |
| **Ours** | **10k h real** | **none** |

---

## 3. Preserving text intelligence

| System | Mechanisms | Measured cost to text / knowledge |
|---|---|---|
| Moshi | 50% text-only steps (separate optimizer); new modules at higher LR | MMLU 54.3 → 49.7, almost all lost during audio pretraining. Removing text replay: Audio TriviaQA 22.8 → 18.3 |
| MiMo-Audio | ~50% text tokens; text loss weight 100; LLM LR 3e-5 vs 2e-4 for new modules; understanding-only stage first | SpeechMMLU T2T 72.5 vs S2S 69.1 (smallest gap among compared models) |
| StepAudio 3 Gen | frozen-backbone alignment stage; 50% text in every later stage | beat a 3-stage baseline on all 8 text benchmarks (MMLU +4.2; confounded) |
| StepAudio 3 Realtime | raised text share; text in duplex mid-training; **merge with a general-text teacher** | text macro 57.2 (best dialogue teacher) → **76.5** merged |
| Covo-Audio | 40% text; **top-20-logit KL** from a stronger text LLM in SFT | T2T topic-StoryCloze 99.4 vs 98.6 base, but spoken general knowledge 49.95 vs 66.86 (Qwen3-Omni) |
| Raon | **on-policy KL to frozen backbone** (text inputs) and to itself given the transcript (audio inputs); TextQA targets regenerated by the backbone; LLM frozen during alignment; LR 1.2e-5 | no text-benchmark delta reported |
| Audex | warm up new audio embeddings with LLM and text embeddings frozen; 44–69% text; LLM LR 2e-5 | unfreezing embeddings: −7.7 MMLU-Pro, −11.5 GPQA, −21.4 AIME25; one-shot mixing needed 88% text and still broke long context |
| Qwen3-Omni | ~25% text early; strong-to-weak distillation | still trails text-only Qwen3-2507 (ZebraLogic −14, AIME25 −11 for Thinking) |
| Qwen2.5-Omni | (text amount n.r.) | trails Qwen2.5-7B on all 10 text benchmarks by 2.9–9.3 |
| MiniCPM-o 4.5 | LLM emits only text | text average 82.1 vs 81.6 (Qwen3-8B), but MMLU −4.7, MATH500 −7.0 |
| VoiceChat | LLM never models audio; 47% retention in SFT incl. 8.5% text-only | no text-benchmark comparison given |
| Freeze-Omni / MinMo | frozen LLM | Freeze-Omni spoken QA ≈ text QA (gap only on the input side); no duplex backchannels |
| Voxtral TTS | text embeddings frozen | — |

---

## 4. Knowledge in speech: the speech-to-text vs speech-to-speech gap

| System | Measure | Text-ish | Speech output |
|---|---|---|---|
| Moshi | Audio TriviaQA | Helium text 56.4 | 22.8 (7.3 without inner monologue) |
| GLM-4-Voice | TriviaQA S→T vs S→S | 39.1 | 26.5 |
| Kimi-Audio (base, MiMo's eval) | SpeechMMLU S2T vs T2S / S2S | 67.9 | **0.0 / 11.8** |
| MiMo-Audio (base) | SpeechMMLU T2T vs S2S | 72.5 | 69.1 |
| MiMo-Audio (after SFT) | Big Bench Audio S2T − S2S | — | 12.7-point gap |
| MinMo | spoken QA S2T − S2S | — | 11–15 points lower |
| Chrono (1.5B) | Llama Q, joint text+audio vs text-only LLM + AR decoder | — | 15.0 vs **30.4** |
| PersonaPlex / Moshi | Artificial Analysis Big Bench Audio | — | 19% / 4% |
| VoiceChat vs Moshi / PersonaPlex | VoiceBench | — | 55.1 vs 29.5 / 30.6 |
| Gemini 3.1 Flash Live | Big Bench Audio, minimal vs high thinking | — | 70.5 → 95.9 |
| GPT-Live | τ-Voice, single-model voice vs delegated backend | — | 30–46% vs 74.5–86.2% |
| TML-Interaction-Small | Big Bench Audio alone vs with background model | — | 75.7 vs 96.5 |
| MoshiRAG (Kyutai) | TriviaQA with async retrieval | — | 22.8 → 78.2 |

---

## 5. Duplex behavior

FDB numbers from different papers are **not directly comparable**: thresholds, judges and harnesses differ. Raon uses a 1.5 s / 5-word threshold and a GPT-5.2 judge. AA uses its own weighted subset.

| System | Source | Key numbers |
|---|---|---|
| StepAudio 3 Realtime | AA FD subset (self-reported) | Overall 98.9 (pause 98.9, turn 100, interruption 99.0, backchannel 98.0) |
| Grok Voice Think Fast 2.0 | AA | FDB 95.1 (v1.0 was 77.8: the gain came from post-training); TTFA 0.70 s |
| Gemini 3.8 Live | AA Conversational Dynamics | 96.1% (3.1 Live: 72–74%) |
| GPT-Live | OpenAI | FDB v1.5 interactivity 80.1%; FDB v1 turn-taking latency 0.798 s |
| TML-Interaction-Small | TML (self-reported) | FDB v1.5 77.8; latency 0.40 s |
| VoiceChat | FDB 1.0 | pause TOR 15.3 / 25.5 (best open); smooth-turn latency 0.448 s; backchannel Resume 93% |
| PersonaPlex | FDB 1.0 | smooth-turn latency 0.170 s; pause TOR 35.8 |
| Moshi | FDB 1.0 | pause TOR 98.5 (talks over pauses) |
| Moshi + GRPO | Kyutai alignment | pause takeover 0.445 → 0.226; interruption latency 1.38 → 0.41 s |
| Covo-FD | own eval | turn-taking 99.7%, pause handling 97.6%; URO-Bench −2–4 vs half-duplex |
| MiniCPM-o 4.5 | Raon's FDB v2.0 | instruction following 3.53 (Raon 3.04); ~1 s latency, few backchannels |

---

## 6. Evidence on the generation head (multi-codebook / continuous)

| Comparison | Result | Source |
|---|---|---|
| Continuous consistency head vs discrete RVQ + depth (same Helium backbone + inner monologue) | MOS 3.45 vs 2.75; meaningfulness Elo 2023 vs 1870 | CALM |
| 1.7B local transformer vs 8B delay pattern | local transformer wins SIM (+2.4 EN) and WER in all 9 languages | MOSS-TTS |
| Flow matching vs depth transformer vs MaskGIT | flow > depth > MaskGIT in human evals (expressivity) | Voxtral TTS |
| Depth transformer vs independent heads (Moshi delay [0,2,…]) | perplexity 36.8 vs 135.4 | Moshi |
| Semantic-codebook loss weight 1 → 100 | NLL 4.09 → 3.65 | Moshi |
| LM on semantic layer + light NAR acoustics vs flat | WER 6.5 vs 7.9; SIM 0.84 vs 0.75 | SpeechTokenizer |
| Zero-init RVQ adaptor on summed cb embeddings (12.5 Hz × 16 × 2048) | SpeechMMLU spoken answers 12.13 → 58.76; WER 6.00 → 3.41 | StepAudio 3 Gen |
| LLM text-only + separate AR audio decoder vs joint | Llama Q 15.0 → 30.4; latency 0.92 → 0.45 s | Chronological Thinking |
| Dedicated duplex TTS vs PersonaPlex's own audio (same text) | WER 5.00 → 2.42 | VoiceChat-TTS |
| Depth decoder trained on 1/16 of frames | "no perceivable difference" | Sesame CSM |
| Sampling temperature 1.0 → 0.8 (RVQ model) | transcript perplexity 52.4 → 26.8 | CALM |
| Silence-frame loss down-weighting (VAD-based) | used for stability | Voxtral TTS, Raon (SIL 0.5 → 0.25) |
