# Frontier full-duplex / real-time S2S and omni models: Chinese labs and Asian academia (2025–2026)

Scope: systems not covered by other notes. Out of scope here (they appear only as initializations or baselines): Step-Audio 2, StepAudio 3 Realtime/Gen, Chronological Thinking, Qwen2.5-Omni, Qwen3-Omni, Qwen3-TTS, MiMo-Audio, Kimi-Audio, GLM-4-Voice, Covo-Audio, Raon-Speech, MiniCPM-o 4.5, MinMo, MOSS-Audio/TTS, VoxCPM2, Freeze-Omni, Mini-Omni2, FlexDuo, LSLM, SpeechTokenizer, and J-Moshi (in the western note). Accessed 2026-10-12.

Labels: **[stated]** is claimed by the developer or paper. **[measured]** is a third-party measurement: the Artificial Analysis (AA) S2S leaderboard as fetched on 2026-10-12, or one paper measuring another team's model. **[inferred]** is my own reading. "n.d." means not disclosed. BBA = Big Bench Audio; CD = AA "Conversational Dynamics" (a weighted FDB v1/v1.5 score); FDB = Full-Duplex-Bench.

Coverage caveat: the session's web-search quota ran out partway. iFlytek, Huawei, MiniMax realtime, Moonshot and Korean industrial systems therefore got only shallow checks (see "Not found / gaps").

## Summary

- **Three tiers.**
  - *Industrial omni Thinker–Talker models.* They have huge speech pretraining and partly disclosed recipes, but turn-taking is undisclosed or turn-based: Qwen3.5-Omni, LongCat-Flash-Omni, the Ming family, JoyAI-Talker.
  - *Closed native full-duplex products with no paper.* ByteDance Seeduplex and SeedRealtime, and Qwen Audio 3.0 Realtime (AA only).
  - *A 2026 wave of "duplex conversions" of strong half-duplex checkpoints.* All use 10^5–10^6 synthetic dialogues (~2k–60k h):
    - DuplexOmni and AdaptDuplex, from Qwen3-Omni;
    - Realtime-Venus, from MiniCPM-o 4.5;
    - BayLing-Duplex, from GLM-4-Voice;
    - Lychee-FD, from StepAudio-2-mini;
    - MultiTalk, from Moshi.
- **Scale gap.**
  - Strong systems pretrain on 10^5–10^8 h of speech:
    - Qwen3.5-Omni: ~2T audio tokens;
    - LongCat-Flash-Omni: 5.1T text-speech tokens;
    - MOSS-Speech: ~4.9M h;
    - FLM-Audio: ~1M h;
    - Baichuan-Audio: 887k h;
    - DuplexPO: 530k h of synthetic continuation.
  - Duplex stages use 10^3–10^4 h. Our 10K h is duplex-stage scale, not speech-LM scale [inferred].
- **Text preservation, with numbers.**
  - Modality-separated deep layers. In Lychee-FD, removing them drops S2S LlamaQ from 65.3 to 36.0.
  - Freeze-then-unfreeze:
    - MOSS-Speech: sStoryCloze 63.1 vs 56.6 and MMLU 66.5 vs 62.1, against no freezing.
    - Baichuan-Audio: sStoryCloze 79.6 vs 77.5.
  - Heavy text share or text-loss weight; a 5–6.25 Hz LLM-side audio rate; model merging.
  - On-policy distillation toward text-conditioned answers; TTS-voiced knowledge corpora.
- **Post-training.**
  - DPO is everywhere, including timing-perturbed negatives and per-codebook audio DPO.
  - GRPO/GSPO with rule-based timing rewards is emerging. DuplexPO reports GRPO > DPO, with semantics unharmed.
  - Seed LiveInterpret 2.0 documents reward hacking.
- **Measured knowledge gap [measured, AA].**
  - Native open duplex models: Moshi 4%, FLM-Audio 16%, Freeze-Omni 33% BBA.
  - Qwen/StepFun realtime systems: 96–99%.

## Compact table

Cells are [stated] unless marked. "inh." = inherited from the init checkpoint [inferred].

| Model | Org | Date | Backbone / size | Audio in → out | Duplex mechanism | Pretrain audio scale | Duplex / post-train data | Link |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-Omni | Alibaba Qwen | 2026-03/04 | Hybrid-attn MoE (GDN) Thinker+Talker, "hundreds of B" | AuT 6.25 Hz → RVQ + MTP, causal ConvNet | "semantic interruption"; mechanism n.d. | ~4T-token stage (1.99T audio); Talker >20M h | OPD, interaction RL; Talker DPO+GSPO | [2604.15804](https://arxiv.org/abs/2604.15804) |
| Qwen Audio 3.0 Realtime | Alibaba Cloud | 2026 | n.d. | n.d. | full-duplex API | n.d. | n.d. | [AA](https://artificialanalysis.ai/speech-to-speech) |
| Fun-Audio-Chat(-Duplex) | Alibaba Tongyi | 2025-12 | Qwen3-30B-A3B / Qwen3-VL-8B | Whisper → S3 25 Hz via refined head (LLM 5 Hz), CosyVoice3 FM | parallel speech-text input stream | "millions of hours" | Core-Cocktail merge; multi-task DPO; synthetic duplex | [2512.20156](https://arxiv.org/abs/2512.20156) |
| LongCat-Flash-Omni | Meituan | 2025-10 | 560B / 27B-active MoE | streaming FSMN (continuous) → 4-cb codec 16.67 Hz, GAN decoder | chunked streaming, turn-based [inferred] | 5.1T text-speech tokens (2:1); >10M h | TTS-dialogue SFT; text+audio DPO | [2511.00279](https://arxiv.org/abs/2511.00279) |
| LongCat-Next | Meituan | 2026-03 | 68.5B / 3B-active, from scratch | 12.5 Hz 8-RVQ in/out | delayed parallel text/audio | >2T tokens total | n.d. | [2603.27538](https://arxiv.org/abs/2603.27538) |
| DuplexOmni | XJTU / PKU / Meituan | 2026-06 | Qwen3-Omni | Qwen3-Omni RVQ stack | 480 ms slices + control tokens; async thinker | inh. | 3.02M synthetic convs; 2-stage SFT | [2606.09186](https://arxiv.org/abs/2606.09186) |
| Seed LiveInterpret 2.0 | ByteDance Seed | 2025-07 | Seed LLM, n.d. | n.d. | duplex chunked read/write | ~100B-token CT | human SFT; PPO | [2507.17527](https://arxiv.org/abs/2507.17527) |
| Seeduplex / SeedRealtime | ByteDance Seed | 2026-04 / 08 | n.d. | n.d. | native full-duplex (SeedRealtime: AV) | n.d. | n.d. | [blog](https://seed.bytedance.com/en/blog/introducing-seed-full-duplex-speech-llm-attentive-listening-robust-interference-suppression-enabling-more-natural-interaction) |
| Ming-flash-omni (Prev./2.0) | Ant inclusionAI | 2025-10 / 2026-02 | Ling-flash-2.0, 100B / 6.1B-active | Whisper → continuous VAE latents (0.5B head + FM) | half-duplex | n.d. | D-GRPO, U-DPO | [2510.24821](https://arxiv.org/abs/2510.24821) |
| Ming-UniAudio | Ant | 2025-11 | MoE 16.8B / 2.8B-active | continuous 50 Hz VAE in/out, diffusion head | half-duplex | 400k h und. + 800k h gen. | anneal 370k:370k h | [2511.05516](https://arxiv.org/abs/2511.05516) |
| Realtime-Venus | Ant | 2026-09 | 9B, from MiniCPM-o 4.5 (Qwen3-8B) | Whisper-M 1 s chunks → S3 25 Hz + FM | serialized user / assistant / background streams + listen/speak tokens | inh. | 2.8M SFT samples; no RL | [2609.13814](https://arxiv.org/abs/2609.13814) |
| JoyAI-Talker | JD | 2026-08 | 48.9B / 3.28B-active MoE | 12.5 Hz encoder → AR + causal-DiT talker | separate 1.7B duplex model, 160 ms, state tokens | n.d. | joint SFT + speech-only DPO | [2608.01119](https://arxiv.org/abs/2608.01119) |
| AdaptDuplex | Zuoyebang | 2026-09 | Qwen3-Omni-30B-A3B | 12.5 Hz frames | dynamic 0.48–0.96 s windows + action tokens | inh. | 765k synthetic dialogues (17.3k h); GRPO | [2609.29217](https://arxiv.org/abs/2609.29217) |
| Step-Audio-R1 / R1.1 | StepFun | 2025-11 / 2026 | Qwen2.5-32B | 12.5 Hz → text (R1); R1.1 speaks | R1.1 "dual-brain" | 4B audio tokens (cold start) | MGRD, RLVR, DPO | [2511.15848](https://arxiv.org/abs/2511.15848) |
| VITA-Audio | Tencent Youtu | 2025-05 | Qwen2.5-7B | GLM-4-Voice 12.5 Hz; 10 tokens/fwd | half-duplex | 100k h ASR + 100k h TTS | 1.4M QA SFT | [2505.03739](https://arxiv.org/abs/2505.03739) |
| NTPP | Tencent + academia | 2025-06 | 7–9B LLMs | 40 Hz VQ/RVQ, both channels | next-token-pair | ~140k h single-channel | 2.2k h Fisher | [2506.00975](https://arxiv.org/abs/2506.00975) |
| Baichuan-Audio | Baichuan | 2025-02 | Qwen2.5-7B | 12.5 Hz 8-RVQ; depth head; FM | half-duplex | 887k h | 242k SFT pairs | [2502.17239](https://arxiv.org/abs/2502.17239) |
| InteractiveOmni | SenseTime | 2025-10 | Qwen3-4B/8B | Whisper → CosyVoice2-style | half-duplex | 240k h und. + 202k h TTS | 22k h S2S; SFT+DPO+merge | [2510.13747](https://arxiv.org/abs/2510.13747) |
| SALMONN-omni | Tsinghua / ByteDance | 2025-05 | Llama-3-8B + LoRA | Mamba 25 Hz → embeddings to CosyVoice2 | env + echo streams; think/shift/listen/speak | n.d. (~1.8M samples) | DPO (barge-in) | [2505.17060](https://arxiv.org/abs/2505.17060) |
| BayLing-Duplex | ICT/CAS | 2026-06 | GLM-4-Voice 9B | 12.5 Hz tokens | single-AR 0.8 s blocks + 4 state tokens | inh. | 400K synthetic; timing DPO | [2606.14528](https://arxiv.org/abs/2606.14528) |
| Lychee-FD | HIT-SZ (+Tencent PCG) | 2026-07 | StepAudio-2-mini ~10B | Whisper → CosyVoice2 25 Hz | text / acoustic / control channels, split deep heads | inh. | 140K synthetic; 8×H20 ~16 h | [2607.06540](https://arxiv.org/abs/2607.06540) |
| MultiTalk | CUHK MMLab | 2026-09 | Moshi-7B | Mimi 12.5 Hz ×8 | Moshi dual-stream | 54.4k h synthetic | 3.2k h FT | [2609.36903](https://arxiv.org/abs/2609.36903) |
| FLM-Audio | BAAI | 2025-09 | 7B (Qwen2.5-VL LM) | Mimi 12.5 Hz ×8 | listen/speak/monologue; sentence-level | ~1M h | 200K synthetic dialogs | [2509.02521](https://arxiv.org/abs/2509.02521) |
| DuplexPO | NTU-SG/NTU-TW/HKUST/NVIDIA | 2026-07 | Qwen2.5-7B | Parakeet 12.5 Hz → CosyVoice2 25 Hz | dual-stream | 530k h synthetic continuation + 70k h QA | GRPO on 67.7K Fisher/Seamless | [2607.07148](https://arxiv.org/abs/2607.07148) |
| MOSS-Speech | SII OpenMOSS / Fudan | 2025-10 | Qwen3-8B, last 4/36 layers split | 12.5 Hz single-cb in/out | half-duplex, no text guidance | ~4.9M h | 1.5M QA SFT | [2510.00499](https://arxiv.org/abs/2510.00499) |
| OmniFlatten | Alibaba Tongyi | 2024-10 | Qwen2-0.5B | CosyVoice tokens | flattened chunk interleave | ~100k h | 2k h synthetic duplex | [2410.17799](https://arxiv.org/abs/2410.17799) |
| SoulX-Duplug | SJTU / Soul / NWPU | 2026-03 | Qwen3-0.6B plug-in | GLM-4-Voice 12.5 Hz | streaming ASR + 5 states, 160 ms | 78k h ASR | ~11k h Fisher-format | [2603.14877](https://arxiv.org/abs/2603.14877) |
| FireRedChat | Xiaohongshu | 2025-09 | Qwen2.5 + FireRedTTS-2 | pVAD + EoT; cascaded / semi-cascaded | modular controller | TTS 1.1M h + 300k h dialog | 250k TTS dialogues | [2509.06502](https://arxiv.org/abs/2509.06502) |

## Systems

### Alibaba

**Qwen3.5-Omni.** Sources: [arXiv 2604.15804](https://arxiv.org/abs/2604.15804) and the [realtime doc](https://www.alibabacloud.com/help/en/model-studio/qwen3-5-omni-flash-realtime). Launched 2026-03-30; snapshot dated 2026-03-15.

- **Architecture [stated].**
  - Plus, Flash and Light variants. Thinker and Talker are both hybrid-attention MoE with Gated DeltaNet; sizes n.d.
  - The AuT encoder was trained from scratch on 40M h of audio-text pairs; it runs at 6.25 Hz.
  - The Talker emits RVQ tokens, predicting the residual codebooks with MTP, and a causal ConvNet decodes them.
  - **ARIA** interleaves text and speech. The cumulative speech:text ratio of any prefix may not exceed the global ratio. Its stated purpose is to stop skipped words and misread numbers.
- **Pretraining [stated].**
  - Encoder alignment with the LLM frozen.
  - A ~4T-token general stage: text 0.92T, audio 1.99T, image 0.95T, video 0.14T, video-audio 0.29T.
  - Context extension to 256k.
  - ">100M h of audio-visual content".
- **Post-training [stated].**
  - Thinker:
    - specialist distillation;
    - **on-policy distillation**, which trains the audio-conditioned query toward the response produced for the same query given as text;
    - interaction-aligned multi-turn RL.
  - Talker:
    - a general stage on >20M h of speech;
    - long-context CPT on curated subsets;
    - DPO plus rule-based rewards with GSPO;
    - speaker fine-tuning.
- **Results [stated].**
  - VoiceBench 93.1 (Plus) vs Gemini-3.1 Pro 88.9; MMAU 82.2; seed-tts WER zh 0.99 / en 1.26.
  - First-packet latency 235 / 435 ms.
  - Text on par with Qwen3.5-Plus.
  - No duplex mechanism or benchmark.
- **Measured.**
  - AA: Plus Realtime BBA 99% (TTFA 2.64 s); Flash Realtime 59% (0.79 s).
  - The DuplexOmni authors scored Flash Realtime 26.4 on their FDB aggregate.
- **Qwen Audio 3.0 Realtime** appears only on AA [measured]:
  - Plus: BBA 99%, CD 98.4, TTFA 1.54 s. Flash: 96%, 96.9, 1.55 s.
  - No Alibaba document was found for it.
  - The related Qwen-Audio-3.0-TTS ([2607.23938](https://arxiv.org/abs/2607.23938)) conditions flow matching on LM hidden states and uses a 12.5 Hz tokenizer [stated].

**Fun-Audio-Chat / -Duplex** (MinMo successor; [2512.20156](https://arxiv.org/abs/2512.20156), Dec 2025) [stated]

- **Architecture.**
  - Initialized from Qwen3-30B-A3B or Qwen3-VL-8B, with Whisper-L-v3 input.
  - **Dual-resolution representations.** Groups of five 25 Hz S3 tokens are projected to 5 Hz for the LLM. A "Speech Refined Head" regenerates the 25 Hz tokens, and a frozen CosyVoice 3 FM decodes them.
  - They claim ~50% fewer GPU-hours.
- **Training.**
  - Pre-alignment with the LLM frozen.
  - **Core-Cocktail:**
    - full fine-tuning at LR 1e-4→1e-5 on speech that CosyVoice 3 synthesized from "billions of text tokens", filtered by WER;
    - merge 0.5·new + 0.5·base LLM;
    - refine at 1e-5→1e-6.
  - Then multi-task DPO on robustness, instruction-following, audio understanding and empathy.
  - Data: "millions of hours". Per-stage sizes are n.d., and there is no merging ablation.
- **Duplex.** A parallel speech-text input stream. Training data is half-duplex dialogue with simulated duplex behaviour, OmniFlatten-style; size n.d.
- **Results.**
  - 30B-A3B: VoiceBench 85.63, OpenAudioBench 80.59.
  - Duplex-30B on UltraEval-Audio: S2M-T 54.89 and S2M-S 49.28, vs Moshi 33.17 / 29.86. Turn-taking success 100%.

**OmniFlatten** ([2410.17799](https://arxiv.org/abs/2410.17799)) and **OmniDRCA** ([2506.09349](https://arxiv.org/abs/2506.09349)), the Tongyi precursors [stated]

- **OmniFlatten.**
  - Qwen2-0.5B. User and assistant chunks (2 text and 10 speech tokens) are flattened into one sequence, and training moves 4 → 3 → 2 streams.
  - Data: ~100k h for alignment, plus 2k h of duplex dialogue synthesized from 390K text sessions.
  - Going straight to 3-stream training scores 2.99, vs 4.88 with alignment plus a half-duplex stage first. Dropping the assistant text stream hurts.
- **OmniDRCA.**
  - Qwen2.5-7B with 25 Hz tokens grouped ×5 to 5 Hz, parallel speech and text embeddings, and a contrastive alignment loss.
  - Data: 210k h + 17.8k h. S2S LlamaQ 65.0 vs Moshi 43.7.
  - A TDM duplex variant costs 18.2% on standard benchmarks.
- **WildTurn** ([2608.28630](https://arxiv.org/abs/2608.28630), Tongyi authors): 2,981 h of real stereo multi-turn conversations, labelled with turn-taking and backchannel styles.

### Meituan

**LongCat-Flash-Omni** ([2511.00279](https://arxiv.org/abs/2511.00279), 2025-10-31; [GitHub](https://github.com/meituan-longcat/LongCat-Flash-Omni)) [stated]

- **Architecture.**
  - 560B / 27B-active MoE.
  - Input: a ~600M streaming FSMN encoder on continuous fbank.
  - Output: LongCat-Audio-Codec, 4 codebooks at 16.67 Hz (1 semantic + 3 acoustic). Semantic tokens are predicted one step ahead of the acoustic ones. A GAN decoder with 3-frame look-ahead produces the waveform.
  - Interaction: chunked audio-visual streaming. No duplex or barge-in is described, so it is turn-based [inferred].
- **Stages.**
  - Stage 0: 16T text tokens.
  - Stage 1: 5.1T text-speech tokens at a 2:1 text:audio ratio. Loss weights: text 1.75, audio 0.25, audio-text 1.5, first-audio 0.1.
  - Stage 2: >3T multimodal tokens (2:1:1).
  - Stage 3: 0.33T annealing.
  - Stage 4: context to 128K.
  - Stage 5: encoder alignment with the LLM frozen.
- **Data.**
  - "Tens of millions of hours" of raw audio; >10M h of processed zh/en speech.
  - Interleaving splits samples at punctuation and randomly masks the audio or the text of each fragment.
- **Post-training.**
  - SFT with the encoder frozen, on spoken dialogues voiced by a TTS fine-tuned on voice actors.
  - DPO with a text term plus a per-codebook audio term, over 6 rollouts per prompt.
- **Lessons.**
  - Discrete *input* tokens "hinder fine-grained acoustic detail", so they switched to the continuous encoder.
  - ASR data adds little to alignment.
- **Results** (README): LibriSpeech 1.57 / 4.01; MMAU 75.9. The Fun-Audio-Chat authors scored it VoiceBench 88.72 [measured].

**LongCat-Next** ([2603.27538](https://arxiv.org/abs/2603.27538), Mar 2026) [stated]

- A discrete native AR model: 68.5B / 3B-active, trained from scratch on >2T tokens.
- The audio tokenizer is a Whisper encoder with 4× downsampling and 8-RVQ at 12.5 Hz, decoded through coarse mel → FM → vocoder.
- Text and audio segments are aligned with *random delays*. One model can therefore do parallel or serial text-guided speech, and "supports full-duplex modeling".
- Results: seed-tts WER 1.90 / 1.89; LlamaQ 79.67.

**DuplexOmni** (XJTU / PKU / Meituan; [2606.09186](https://arxiv.org/abs/2606.09186), Jun 2026) [stated]

- **Mechanism.**
  - Initialized from Qwen3-Omni. Interaction runs in 480 ms slices.
  - Each slice emits a thinking-control signal, a semantic reading of the user, and text plus 480 ms of speech. Control tokens include [THINK], [CUT] and [WAIT].
  - An asynchronous thinking layer (default Gemini-3.1-Flash-Lite) handles reasoning.
- **Data.** ~620k seeds expanded into ~3.02M conversations (70% zh), voiced by Qwen3-TTS and mixed with real recordings.
- **Training.** SFT alternating Thinker and Talker, with the Thinker frozen during Talker training. 128×H20.
- **Results.**
  - FDB aggregate 72.6. The authors' measurements of rivals [measured]: MiniCPM-o 4.5 36.3, Doubao 27.8, Gemini-3.1-Flash-Live 24.1.
  - BBA 77.2 with the thinker, 22.2 without.

### ByteDance Seed

**Seed LiveInterpret 2.0** ([2507.17527](https://arxiv.org/abs/2507.17527), Jul 2025) [stated]

- A duplex zh↔en speech-to-speech interpretation model with voice cloning. Backbone size and token rates are n.d.
- Training:
  - ~100B-token continual pretraining; the [blog](https://seed.bytedance.com/en/blog/seed-liveinterpret-2-0-released-an-end-to-end-simultaneous-interpretation-model-featuring-ultra-high-accuracy-close-to-human-interpreters-low-latency-of-3-seconds-and-real-time-voice-cloning) says "hundreds of thousands of hours";
  - human-annotated SFT;
  - PPO, first with single-turn rewards (detection, initiative, quality, time compliance, format), then multi-turn rewards (lag, sequence quality).
- RL vs SFT, S2S: quality 66.6 → 67.8; first-letter latency 4.26 → 2.71 s.
- **Reward hacking:** the time-compliance reward alone cut audio duration by ~35%.

**Seeduplex** (2026-04-09; [blog](https://seed.bytedance.com/en/blog/introducing-seed-full-duplex-speech-llm-attentive-listening-robust-interference-suppression-enabling-more-natural-interaction)) and **SeedRealtime** (2026-08-05; [blog](https://seed.bytedance.com/en/blog/seedrealtime-audio-visual-full-duplex-llm-released-toward-omni-modal-natural-interaction))

- **Seeduplex [stated].**
  - A native listen-while-speaking model that replaces Doubao's half-duplex model, with multi-task post-training.
  - Claims: false responses and interruptions halved; endpoint latency −250 ms; satisfaction +8.34 pts.
- **SeedRealtime [stated].**
  - Audio-video full-duplex.
  - The only number given: "pacing issues halved" vs cascaded systems.
- Neither has a paper, weights or data.

### Ant Group

**Ming family** [stated]

- **Ming-flash-omni** ([2510.24821](https://arxiv.org/abs/2510.24821), Oct 2025):
  - Ling-flash-2.0, 100B / 6.1B-active, with a Whisper encoder.
  - Output replaces discrete tokens with continuous VAE-GAN latents. A *fixed* 0.5B Qwen2.5-based audio head reads the LLM's text tokens and conditions a flow-matching head. This is effectively a text-to-latent Talker [inferred].
  - VoiceBench 83.50.
- **Ming-UniAudio** ([2511.05516](https://arxiv.org/abs/2511.05516)):
  - A 1.35B causal VAE tokenizer at 50 Hz, trained on ~390k h, serves both understanding and generation.
  - The LLM is a 16.8B / 2.8B-active MoE with a diffusion head.
  - Data: 400k h understanding + 800k h generation, then annealing at 370k:370k h.
  - seed-tts WER zh 0.95.
- **Ming-flash-omni 2.0** ([HF](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0), 2026-02-11) adds unified speech, sound and music generation. Nothing about duplex.

**Realtime-Venus** ([2609.13814](https://arxiv.org/abs/2609.13814), Sep 2026) [stated]

- **Architecture.**
  - Two 9B models adapted from MiniCPM-o 4.5 (Qwen3-8B). Only the Thinker is trained.
  - Three serialized streams: user, assistant (text + S3 speech) and *background* (delegation results). `<|listen|>`, `<|speak|>` and `<|turn_eos|>` tokens control the turn.
- **Data.** 2.8M SFT samples (37% proactive duplex, 6% delegation), synthesized via scenario plan → paralinguistic realization → alignment on a shared clock. No RL.
- **Results.**
  - FDB v1.5: interruption response 75% (Gemini 3.1 Live 77%), backchannel continuation 97%, background speech 86%.
  - LlamaQ 83.8; MMAU 78.0.

**VocalNet-M2** (SJTU + Ant; [2511.10232](https://arxiv.org/abs/2511.10232)) [stated]

- A Qwen3-8B Thinker; the Talker emits 8-codebook tokens through MTP layers, with no flow matching.
- Against VocalNet-8B: first-chunk latency 556 → 349 ms, but WER 3.64 → 6.07.
- Lesson: multi-codebook output "requires more training data and higher-quality supervision".

### Other companies

**JoyAI-Talker** (JD; [2608.01119](https://arxiv.org/abs/2608.01119), Aug 2026) [stated]

- **Architecture.**
  - Thinker: a 48.9B / 3.28B-active MoE with a 12.5 Hz encoder.
  - Talker: AR Transformer + causal DiT.
  - Duplex is a *separate* 1.7B model. It reads 160 ms chunks and emits ASR text plus partial / complete / backchannel / accept / reject tokens, which "shields the semantic core".
- **Training.**
  - Joint speech-text mid-training at 8K context: adapter warm-up, then full-parameter.
  - Context extension to 64K, then SFT.
  - DPO with preference pairs built only within the speech-conditioned pool. Loss on assistant tokens only.
  - Data scale n.d.
- **Results.**
  - Text preserved: MATH 94.62, GPQA 64.51.
  - FDB v1.5: interruption respond 0.88; background-speech false respond 0.10. The authors measured Gemini 3.1 Live at 0.77 / 0.28 [measured].

**AdaptDuplex + DuplexDrama** (Zuoyebang; [2609.29217](https://arxiv.org/abs/2609.29217), [2609.12872](https://arxiv.org/abs/2609.12872)) [stated]

- **Mechanism.**
  - Full fine-tuning of Qwen3-Omni-30B-A3B.
  - At each window end the model predicts the next window length (0.48 / 0.64 / 0.96 s). Action tokens handle memory and asynchronous requests.
  - "Bounded text-ahead" lets at most 2 text tokens lead the speech.
- **Training.**
  - Curriculum on 765k synthetic dialogues (17.3k h): turn-based (64k) → basic duplex (191k) → full duplex (509k).
  - The Talker is trained with the Thinker frozen.
  - GRPO with rhythm, quality (UTMOSv2 + judge) and action rewards.
- **Ablations.**
  - Curriculum vs mixed SFT: overlap response latency 0.37 vs 1.65 s.
  - Random text-ahead vs aligned: UTMOS 3.65 vs 2.90.
- **Results.** Beats DuplexOmni on 18/21 FDB metrics.
- **DuplexDrama:** >2,000 h of synthetic dialogue, but only 3.8% of turns contain interruptions or backchannels.

**StepFun (brief)** [stated]

- **Step-Audio-R1** ([2511.15848](https://arxiv.org/abs/2511.15848)):
  - Qwen2.5-32B with text output.
  - Training: a cold start on 1B text + 4B audio tokens; reasoning distillation grounded in the audio modality; RLVR with reward 0.8·acc + 0.2·reasoning-present, which prevents reasoning collapse.
  - DPO cut "I cannot hear" errors from 6.76% to 0.02%.
- **R1.1 Realtime** uses "dual-brain" Mind-Paced Speaking ([2510.09592](https://arxiv.org/abs/2510.09592)). AA: BBA 98%, TTFA 1.53 s [measured].
- **EditX** ([2511.03601](https://arxiv.org/abs/2511.03601)) uses large-margin synthetic data for expressive editing.

**Tencent** [stated]

- **VITA-1.5** ([2501.01957](https://arxiv.org/abs/2501.01957)): Qwen2-7B. Alignment on 11k h ASR, then NAR+AR speech decoders trained on 3k h TTS with the LLM frozen. Half-duplex.
- **VITA-Audio** ([2505.03739](https://arxiv.org/abs/2505.03739)):
  - Qwen2.5-7B on GLM-4-Voice tokens. MCTP modules emit 10 audio tokens per forward pass, cutting first-audio latency from 236 to 53 ms.
  - Data: 100k h ASR + 100k h TTS, all open-source.
- **NTPP** ([2506.00975](https://arxiv.org/abs/2506.00975), ICML 2025): predicts both speakers' tokens per step with one KV cache. Trained on ~140k h single-channel, then 2.2k h Fisher. **Dropping the single-channel stage "degraded performance significantly".**

**Baichuan-Audio / Omni-1.5** ([2502.17239](https://arxiv.org/abs/2502.17239), [2501.15368](https://arxiv.org/abs/2501.15368)) [stated]

- Qwen2.5-7B with a 12.5 Hz 8-RVQ tokenizer (Whisper semantics + reconstruction) and a depth-transformer head.
- 887k h of pretraining, including 393k h interleaved and 142k h interleaved-TTS.
- Two stages: LLM frozen, then nearly all parameters. sStoryCloze 79.6 vs 77.5 for one stage.
- No loss on interleaved audio segments helped.
- Omni-1.5 keeps **40% pure text**.

**SenseTime** [stated]

- **InteractiveOmni** ([2510.13747](https://arxiv.org/abs/2510.13747)):
  - Qwen3-4B/8B with a CosyVoice2-style decoder.
  - Data: 240k h understanding, 202k h TTS, 22k h S2S dialogue.
  - Post-training: SFT, DPO, merging.
- **Routing study** (CUHK / SenseTime / Tsinghua; [2605.10199](https://arxiv.org/abs/2605.10199)): fusing the user stream into the LLM vs cross-attention to it:
  - WebQ 28.0 vs 18.4;
  - FDB v1.5 interruption respond 0.72 vs 0.32;
  - but fusion corrupts the context when it misses an interruption.

**Soul AI and Xiaohongshu** [stated]

- **SoulX-Duplug** ([2603.14877](https://arxiv.org/abs/2603.14877)):
  - A plug-in Qwen3-0.6B state predictor: streaming ASR plus 5 user states in 160 ms chunks.
  - Data: Fisher plus ~10k h of in-house Fisher-format Mandarin.
  - Latency 205 ms vs FlexDuo 295 ms. Without ASR pretraining, accuracy drops 84.3 → 80.5.
- **SoulX-Podcast** ([2510.23541](https://arxiv.org/abs/2510.23541)):
  - Qwen3-1.7B dialogue TTS on 1.0M h monologue + 0.3M h dialogue, using a diarization/ASR/DNSMOS pipeline.
  - A natural duplex-data synthesizer [inferred].
- **FireRedChat** ([2509.06502](https://arxiv.org/abs/2509.06502)):
  - Personalized VAD + a BERT end-of-turn detector around cascaded or semi-cascaded pipelines.
  - FireRedTTS-2 is trained on 1.1M h + 300k h of dialogue.
  - False barge-in 10.2% vs LiveKit 33.4% [measured by authors].

**Others** [stated]

- **Baidu ERNIE 5.0** ([2602.04705](https://arxiv.org/abs/2602.04705)): a 12.5 Hz RVQ tokenizer with a Whisper-distilled first code, generated by depth-wise "next-codec prediction". Audio data and duplex are n.d.
- **Xiaomi** ([updates](https://mimo.mi.com/docs/en-US/updates/model)): MiMo-V2-Omni and MiMo-V2-TTS (">100M hours", multi-codebook). No duplex model.
- **Zhipu:** GLM-TTS ([2512.14291](https://arxiv.org/abs/2512.14291)): 100k h + RL.
- **MiniMax:** MiniMax-Speech ([2505.07916](https://arxiv.org/abs/2505.07916)), a TTS.

### Academic full-duplex and S2S models

**SALMONN-omni** (Tsinghua / ByteDance; [2505.17060](https://arxiv.org/abs/2505.17060), NeurIPS 2025) [stated]

- Llama-3-8B + LoRA with a 25 Hz Mamba streaming encoder.
- Output: LLM *embeddings* drive CosyVoice2, so no codec tokens enter the LLM.
- Two input streams: environment and the model's own echo. State tokens think / shift / listen / speak, in 80 ms blocks.
- Data: ~1.8M samples, including 80k synthesized dialogues and barge-in/backchannel sets. 32×A100.
- DPO lifts barge-in F1 0.86 → 0.90. S2T LlamaQ 80.0.
- No evidence is given that avoiding codecs preserves knowledge.
- Follow-up: **SALMONN-duo** ([2609.34247](https://arxiv.org/abs/2609.34247)) adds an asynchronous slow agent.

**BayLing-Duplex** (ICT/CAS; [2606.14528](https://arxiv.org/abs/2606.14528)) [stated]

- **Mechanism.** A single AR GLM-4-Voice-9B. 0.8 s blocks hold 10 user tokens, 5 text tokens and 10 assistant tokens. [SILENCE], [ASSISTANT], [PAD] and [EPAD] tokens carry the timing.
- **Data.** 400K samples, built from 200K Alpaca/UltraChat dialogues rewritten by an LLM and voiced by CosyVoice.
  - User gaps: U(0.5, 3) s.
  - Interruption reactions: U(0.8, 2) s.
- **Training.** Weighted CE (role tokens ×10, silence ×0.1). Then DPO against *timing-perturbed* negatives, plus a 0.5 SFT loss.
- **Results.**
  - Uniform weights collapse into silence: 60.3% turn-taking success.
  - Turn-taking 92.0% vs Moshi 71.9%; LlamaQ 46.0 vs 21.0.
  - Duplex quality ≈ turn-based (44.3 vs 45.3).
- Same group: **LLaMA-Omni 2** ([2505.02625](https://arxiv.org/abs/2505.02625)): 200K synthetic dialogues; 7B S2S QA 60.7 vs GLM-4-Voice 50.7; gains plateau at 200K.

**Lychee-FD** (HIT-SZ + Tencent PCG; [2607.06540](https://arxiv.org/abs/2607.06540)) [stated]

- **Diagnosis.**
  - Text and speech gradients agree in shallow layers and conflict in deep layers. The same holds in Moshi (layers 23–31).
  - Padding ~3 Hz text to 25 Hz dilutes the semantic gradient.
- **Fix.** 24 shared layers, then semantic (4), acoustic (4) and control (2) heads, plus a continuous monologue channel.
- **Training.** 140K synthetic dialogues; 1 epoch on 8×H20.
- **Results.**
  - S2S LlamaQ / WebQ / TriviaQA 65.3 / 33.9 / 39.4. The init model scores 62.0 / 30.8 / 29.8.
  - **Without the separation: 36.0 / 22.5 / 24.2.**
  - FDB v1.5 interruption response 78%.

**MultiTalk** (CUHK MMLab; [2609.36903](https://arxiv.org/abs/2609.36903)) [stated]

- Moshi-7B continued on 57.6k h of fully synthetic data: Claude-written scripts voiced by IndexTTS2, with 0.2–0.6 s overlaps.
- ~593 H800-hours.
- Multi-party benchmark: 13.15 vs Qwen3-Omni-30B 4.94 and humans 68.74 [measured by authors].

**FLM-Audio** (BAAI; [2509.02521](https://arxiv.org/abs/2509.02521)) [stated]

- 7B, initialized from the Qwen2.5-VL LM. Mimi ×8 tokens, Moshi-like.
- **"Natural monologue":** sentence-level text padded with `<wait>`, trained "dual" as text-leads and text-follows.
- Data: ~1M h of auto-transcribed audio + 200K Fish-TTS dialogs.
- That is <15% of Moshi's audio.
- AA: BBA 16%, CD 62.0 [measured].

**DuplexPO** (NTU-SG / NTU-TW / HKUST / NVIDIA; [2607.07148](https://arxiv.org/abs/2607.07148)) [stated]

- Qwen2.5-7B, pretrained on 530k h of *synthetic pseudo-dialogue continuation*: consecutive sentences of text corpora alternate between the user and agent streams. Plus 70k h of spoken QA.
- **RL:** GRPO with a rule-based timing reward (onset, backchannel, yield) from word timings on 67.7K Fisher/Seamless samples, with KL to SFT. 64×A800.
- **Results.**
  - Voiced-interrupt rate 7.33% → 0.24%.
  - LlamaQ 72.0 → 75.3.
  - GRPO > DPO; the rules beat a neural reward.

**MOSS-Speech** (SII OpenMOSS / Fudan; [2510.00499](https://arxiv.org/abs/2510.00499)) [stated]

- Qwen3-8B generating speech *without text guidance*. The last 4 of 36 blocks are modality-specific; tokens are 12.5 Hz single-codebook.
- Data: ~4.9M h, including 4M h of real web speech, interleaved data and FineWeb-Edu rendered by TTS.
- Training: the backbone is frozen first.
- Frozen vs not:
  - sStoryCloze 63.12 vs 56.60;
  - MMLU 66.5 vs 62.1, against 76.6 for the base model.
- S2S LlamaQ 63.67 vs GLM-4-Voice 65.67. TriviaQA trails: 28.8 vs 43.2.

**Brief** [stated]

- **EchoX** ([2509.09174](https://arxiv.org/abs/2509.09174)): speech targets are derived from semantic hidden states, trained on 6.2k h. 8B S2S LlamaQ 63.3. An interleaved baseline without it averages 12.8.
- **OpenS2S** ([2507.05177](https://arxiv.org/abs/2507.05177), CASIA): fully synthetic empathetic SFT.
- **DuplexMamba** ([2502.11123](https://arxiv.org/abs/2502.11123)): Mamba-2.8B, text-only output, state tokens.

### Turn-taking modules, data and benchmarks (brief)

- **Modules:**
  - Easy Turn ([2509.23938](https://arxiv.org/abs/2509.23938); NWPU + Huawei): a 0.5B LLM with 4 states, on a 1,145 h set.
  - HiThink Turn ([2609.34096](https://arxiv.org/abs/2609.34096)): interruption success 89 → 98%.
  - TurnFSM ([2609.04240](https://arxiv.org/abs/2609.04240)).
  - MThreads ([2405.19487](https://arxiv.org/abs/2405.19487), 2024).
- **Data:** DuplexChat ([2607.04941](https://arxiv.org/abs/2607.04941), UTokyo): 282,634 h English + 132,723 h Japanese of speaker-separated podcast dialogue. This is the largest open real duplex corpus found.
- **Benchmarks:**
  - FDB-v2 ([2510.07838](https://arxiv.org/abs/2510.07838)) evaluated only GPT-Realtime, Moshi and Freeze-Omni.
  - DualTalk ([2505.18096](https://arxiv.org/abs/2505.18096)) is 3D talking heads, not S2S.

## Cross-cutting patterns

1. **Architecture families.**
   - (a) A Thinker–Talker omni model plus *time slices and control tokens*: DuplexOmni, AdaptDuplex, Realtime-Venus.
   - (b) Thinker–Talker plus a *separate small duplex model*: JoyAI, SoulX-Duplug, FireRedChat.
   - (c) A single AR LLM with block interleaving: BayLing-Duplex, OmniFlatten, SALMONN-omni.
   - (d) Moshi-style dual stream + monologue: FLM-Audio, MultiTalk.
   - (e) Channel-division with separated deep heads: Lychee-FD.

   The best 2026 FDB numbers come from (a), (b) and (e) [inferred].
2. **Representations.**
   - LLM input is mostly continuous, at 5–12.5 Hz.
   - Output is either:
     - 25 Hz S3 or 12.5 Hz single-codebook tokens + FM;
     - RVQ + MTP or depth heads (Qwen3.5-Omni, LongCat, ERNIE 5.0);
     - continuous latents + FM/DiT (Ming).
3. **Data.**
   - Speech-LM stages: 10^5–10^8 h.
   - Duplex stages: 10^3–6×10^4 h, almost all synthetic: an LLM rewrites text dialogues, zero-shot TTS voices them, and timing comes from rule-sampled gaps.
   - Real two-channel data is used mainly for timing, RL or state prediction: Fisher, Seamless, Fisher-format in-house sets, podcast separation.
4. **Stages.**
   - Align with the LLM frozen → interleaved continuation with a large text share → SFT → DPO/RL.
   - For duplex, a turn-based stage first beats going straight to duplex (OmniFlatten, AdaptDuplex).
5. **Text preservation.**
   - Freezing; modality-split deep layers; text share and loss weight.
   - Low LLM frame rate; merging.
   - Text-guided output with a bounded or random lead.
   - OPD; KL/SFT anchors in RL.

## Lessons for our setup

Our setup is Qwen3 LLM + Qwen3 tokenizer/decoder with 10K h of duplex. All lessons below are [inferred] from the evidence above.

1. **Our 10K h is a duplex-stage budget.** Models with good continuation and spoken knowledge saw ≥10^5 h of *single-stream* speech first, or inherited it. NTPP shows that dropping that stage hurts.
   - Add a large interleaved stage.
   - Use TTS-rendered text corpora: DuplexPO used 530k h of pseudo-dialogues, MOSS-Speech used FineWeb-Edu, Fun-Audio-Chat used WER-filtered synthesis.
   - Use LongCat-style random audio/text masking.
2. **Or start from a speech-pretrained Qwen3-family checkpoint.** Qwen3-Omni and MiniCPM-o 4.5 became strong duplex models with ≤20k h or a few million synthetic SFT samples.
3. **Protect the backbone.**
   - Freeze, then unfreeze.
   - Keep ≥30–40% text and up-weight the text loss.
   - Feed audio at ~5–6.25 Hz.
   - Split the last ~4 layers or heads by modality. This is the cheapest high-evidence fix (Lychee-FD).
4. **Keep a text channel without word alignment.** Use sentence-level or bounded-lead text. Pure speech output (MOSS-Speech) still trails on knowledge QA.
5. **Duplex SFT.**
   - Down-weight silence tokens.
   - Run a turn-based → duplex curriculum.
   - Sample gaps and delays.
   - Add real two-channel data for timing.
   - Check the backchannel/interruption share of our data.
6. **Post-training.**
   - On-policy distillation toward the model's *text-conditioned* answer (Qwen3.5-Omni).
   - DPO with an SFT auxiliary loss and per-codebook audio terms.
   - GRPO with timing rules plus KL.
   - Guard against duration hacking.
7. **Speech quality.** Multi-codebook direct generation needs more and cleaner data (VocalNet-M2). Predict the semantic codebook first, then the residual codebooks via MTP.
8. **Evaluate:**
   - S2T vs S2S on LlamaQ / WebQ / TriviaQA;
   - s/tStoryCloze;
   - BBA;
   - FDB v1 / v1.5 (+v3);
   - UTMOS and ASR-WER.

## Not found / gaps

- Not found:
  - GLM-4.5-Voice.
  - A Kimi-Audio follow-up.
  - A MiniMax or Xiaomi duplex chat model.
  - iFlytek or Huawei duplex S2S papers. Huawei appears only as an Easy Turn co-author.
- Search coverage was limited for these.
- Qwen Audio 3.0 Realtime has AA measurements only.
- Seeduplex and SeedRealtime have no technical disclosure.
- LongCat-Flash-Omni's speech-to-speech tables could not be extracted.
