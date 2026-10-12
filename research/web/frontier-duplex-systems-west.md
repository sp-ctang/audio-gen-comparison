# Frontier real-time / full-duplex voice systems: US/European industry, startups and academia (2025–2026)

Scope: systems not covered by other notes. Out of scope here: Kyutai/Gradium, NVIDIA, OpenAI, Google, xAI, Sesame, Nari, the VibeVoice TTS paper, Voxtral TTS, SyncLLM, LSLM, and the per-paper notes (Raon-Speech, Covo-Audio, Multi-Stream LLMs, etc.). Accessed 2026-10-12.

Labels: **[stated]** is claimed by the developer or paper. **[measured]** is a third-party measurement, mostly the Artificial Analysis (AA) S2S leaderboard as fetched on 2026-10-12. **[inferred]** is my own reading. "n.d." means not disclosed.

## Summary

- **TML-Interaction-Small is the one new frontier-class native duplex model here with real disclosure.** Thinking Machines Lab released it on 2026-05-11 [stated].
  - A 276B MoE with 12B active parameters, trained from scratch.
  - Encoder-free: audio goes in as dMel and comes out through a flow head.
  - 200 ms micro-turns, plus an asynchronous background model for reasoning and tools.
  - FD-bench v1.5 average of 77.8, against 39–54 for GPT-realtime, Gemini Live and Qwen realtime.
  - Training data, stages, RL and compute: n.d.
- **Most western production S2S systems are not frame-synchronous dual-stream models.**
  - Amazon Nova (2) Sonic: speech encoder → core LLM that writes a text response → speech renderer. End-of-turn detection and barge-in are learned inside the model. Trained with pretraining, SFT, RL and preference tuning [stated].
  - Hume EVI pairs a speech-language model with an external LLM.
  - Higgs Realtime and Deepslate Opal expose VAD-style turn detection.
  - Ultravox outputs text only.
- **Measured: open duplex models have the timing but not the knowledge [measured, AA].** BBA is Big Bench Audio. CD is AA's "Conversational Dynamics", a weighted FDB v1/v1.5 score.

  | System | BBA | CD | TTFA |
  |---|---|---|---|
  | Nova 2 Sonic | 88% | n/a | 1.14 s |
  | Deepslate Opal | 85% | 85.7 | 0.44 s |
  | Higgs Realtime | 69% | 92.8 | 1.47 s |
  | PersonaPlex | 19% | 91.0 | n/a |
  | Moshi | 4% | 61.0 | n/a |
- **Training recipes with numbers come mostly from smaller or open efforts.**
  - LFM2-Audio: ~472k h per epoch.
  - Hume TADA: 905k h.
  - hertz-dev: 20M h, audio-only.
  - Llama-Mimi: 240k h.
  - J-Moshi: 60k h pretraining, 344 h real stereo, 602 h synthetic stereo.
  - Human-1: 26k h of real Hindi conversation on top of Moshiko.
  - F-Actor: 2k h synthetic.
- **Evidence on our two problems.**
  - *Audio-only scale does not buy semantics.* Apple's textless 16B continuous-diffusion SLM was trained on "tens of millions of hours" and still lacks long-form coherence.
  - *Text anchoring does buy semantics.*
    - Interleaved SLMs "latently work in text", and this correlates with spoken factual recall.
    - TADA's speech-free guidance raises tStoryCloze from 84.2 to 94.7.
    - F-Actor's text stream lowers perplexity from 25.2 to 21.5.
  - *More codebooks trade spoken content for acoustic fidelity* (Llama-Mimi).
- **Post-training is RL with shaped timing rewards:**
  - HiPLEX trains timing and content as separate policies.
  - Meta multiplies a WER reward by a delay reward.
  - TiCo uses RL for duration control.
  - Nova and Hume use RL in production.
  - TML, Nova 2 Sonic and EVI 3 also hand heavy reasoning off asynchronously.

## Compact table

| Model | Org | Date | Backbone / size | Audio in → out | Duplex mechanism | Pretrain audio scale | Duplex / post-train data | Link |
|---|---|---|---|---|---|---|---|---|
| TML-Interaction-Small | Thinking Machines Lab (US) | 2026-05-11 | 276B MoE / 12B active, trained from scratch | dMel + light embedding → flow head (+text unembed); video as 40×40 patches via hMLP | 200 ms micro-turns that interleave input and output; async background model | n.d. | n.d.; safety refusal data from automated red-teaming, rendered with TTS | [blog](https://thinkingmachines.ai/blog/interaction-models/) |
| Nova Sonic / Nova 2 Sonic | Amazon AGI (US) | 2025-04-08 (rev. 06-12) / 2025-12-02 | core multimodal transformer, size n.d. | speech encoder → text response → speech renderer conditioned on the input speech | streaming bidirectional; in-model end-of-turn and barge-in; sensitivity setting; async tools | n.d. | pretrain + SFT + RL(HF) + preference tuning | [v1](https://cdn.amazon.science/86/bb/4316d28940bd9a719abb28f45aaf/amazon-nova-sonic-technical-report-and-model-card-6-12.pdf), [v2](https://cdn.amazon.science/c5/3d/84514a224666b5be6de4b43ef4aa/nova-2-0-technical-report2.pdf) |
| EVI 3 / EVI 4-mini; TADA | Hume AI (US) | 2025-05-29 / 2025-10-01 / 2026-02-26 | SLM size n.d.; TADA = Llama 3.2 1B/3B | single AR model over "text + voice tokens"; TADA: one continuous acoustic vector per text token | turn-based with interruption; external LLM | TADA 905k h | RL on voice qualities (details n.d.) | [EVI 3](https://www.hume.ai/blog/introducing-evi-3), [TADA](https://arxiv.org/abs/2602.23068) |
| LFM2-Audio / LFM2.5-Audio 1.5B | Liquid AI (US) | 2025-10-01 / 2025-11-28 | LFM2-1.2B hybrid conv+attention | FastConformer, continuous, 80 ms → Mimi 8-codebook RVQ at 12.5 Hz via RQ-Transformer | interleaved text/audio generation; turn-based | ~472k h per epoch | heavy synthetic multi-turn | [blog](https://www.liquid.ai/blog/lfm2-audio-an-end-to-end-audio-foundation-model), [card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) |
| Ultravox v0.5/v0.6 | Ultravox/Fixie (US); acquired by Inworld 2026-09-30 | 2025-02-11 (v0.5) | frozen Llama 3.1 8B / 3.3 70B (v0.6 adds Gemma 3 27B, Qwen3 32B) | Whisper-large-v3-turbo + adapter → text (external TTS) | platform turn-taking | v0.4.1: >10k h | KD onto the text LLM's logits | [card](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_3-70b) |
| Higgs Realtime | Boson AI (US) | ~2026-07 | n.d. | n.d. | "full-duplex" session; server_vad / semantic_vad | n.d. (Higgs Audio v2 TTS: >10M h) | n.d. | [docs](https://docs.boson.ai/models/higgs-realtime/overview) |
| Opal | Deepslate (Berlin) | 2026 (AA test Sep 2026) | n.d. | native S2S, 27 languages | server-side VAD; interruption handling | n.d. | n.d. | [site](https://deepslate.eu/) |
| hertz-dev | Standard Intelligence (US) | 2024-11-03 | 8.5B; text-init and from-scratch variants | 8 Hz latents, 1 kbps, 15-bit FSQ "phonetic" bits (both directions) | two-channel generation | 20M h | none (base model) | [post](https://si.inc/posts/hertz-dev/) |
| Muse Voice Transcribe | Meta MSL (US) | 2026-09-01 | Muse Spark family, size n.d. | 80 ms chunk → 1 soft token → text | per-chunk listen/emit token; endpoint tokens | n.d. (70+ languages) | RL with WER × delay reward | [blog](https://research.meta.ai/blog/introducing-muse-voice-transcribe) |
| VibeVoice-Realtime-0.5B; Phi-4-MM | Microsoft (US) | card n.d.; 2025-02 | Qwen2.5-0.5B + diffusion head; Phi-4-mini + speech LoRA | σ-VAE 7.5 Hz → diffusion; Phi: speech in, text out | streaming-text TTS; no duplex | Phi-4-MM ~2.3M h | Phi: SFT + DPO + RLHF | [VV-RT](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B), [Phi](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) |
| CD-SLM scaling | Apple (US) | 2026-04-27 | MM-DiT, 0.6M–11.5B (+16B) | 80-dim log-mel at 80 Hz, continuous diffusion, textless | none (continuation) | ~7M h; 16B on "tens of millions of h" | none | [arXiv](https://arxiv.org/abs/2604.24416) |
| Voxtral Realtime (Mini 4B) | Mistral (FR) | 2026-02-11 | Ministral 3B + 970M causal encoder trained from scratch | log-mel → 12.5 Hz → text | DSM, 80–2400 ms delay | n.d. (13 languages) | 2 stages | [arXiv](https://arxiv.org/abs/2602.11298) |
| J-Moshi | Nagoya U (JP) | 2025-06-03 | Moshi (Japanese SentencePiece) | Mimi | Moshi dual-stream | 60k h, diarized to stereo | 344 h real + 602 h TTS stereo | [arXiv](https://arxiv.org/abs/2506.02979) |
| Human-1 | JoshTalks (IN) | 2026 | Moshiko (Hindi tokenizer) | Mimi | Moshi dual-stream | 26k h real Hindi conversation | 990 h curated | [card](https://huggingface.co/JoshTalksAI/Human-1) |
| F-Actor | KIT + Edinburgh (EU) | 2026-01-16 | Llama 3.2-1B-Instruct | NeMo nano codec, 12.5 fps, 4 FSQ codebooks, parallel heads | dual stream + text; 2-token audio delay | none | 2k h synthetic (Behavior-SD) | [arXiv](https://arxiv.org/abs/2601.11329) |
| Llama-Mimi | Kyoto U / NII / Nagoya (JP) | 2025-09-18 | Llama 3.2-1B (also 8B) | Mimi, 4 quantizers flattened (50 tok/s) | none | 240k h English | none | [arXiv](https://arxiv.org/abs/2509.14882) |
| DuplexCascade | SB Intuitions + U Tokyo (JP) | 2026-03-10 | Qwen2-7B-Instruct + DSM-ASR/TTS | text cascade | 0.6 s micro-turns, 7 control tokens, no VAD | none | 50k UltraChat dialogues, LoRA | [arXiv](https://arxiv.org/abs/2603.09180) |
| HiPLEX | Qualcomm AI Research + KAIST (US/KR) | 2026-10-06 | Moshi / PersonaPlex | Mimi | dual-stream; RL that factorizes timing from content | — | 73,561 Seamless Interaction windows; LLM-judge reward | [arXiv](https://arxiv.org/abs/2610.07727) |

## Systems

### 1. Thinking Machines Lab: TML-Interaction-Small

Source: blog post "Interaction Models: A Scalable Approach to Human-AI Collaboration", 2026-05-11, with no named authors. https://thinkingmachines.ai/blog/interaction-models/

- **Status [stated].** Limited research preview, with wider release later in 2026. TML's "larger pretrained models" are currently too slow to serve this way.
- **Architecture [stated].**
  - 276B MoE, 12B active, "trained from scratch".
  - Encoder-free early fusion:
    - Audio comes in as **dMel** through a "light-weighted embedding layer". dMel is Bai et al. 2024: mel channels discretized into intensity bins, with no learned tokenizer. Its authors overlap with Apple's SLM papers [inferred].
    - Video is cut into 40×40 patches and encoded by an hMLP.
    - Audio goes out through a **flow head** (flow matching). Text goes out through a normal unembedding.
  - "All components are co-trained from scratch together with the transformer", so there is no Whisper and no external TTS.
- **Duplex [stated].**
  - "micro-turns continuously interleaving the processing of 200ms worth of input and generation of 200ms worth of output".
  - No turn boundaries and no VAD.
  - Not disclosed: tokens per chunk, and whether a text inner monologue runs alongside the audio.
- **Background model [stated].** The interaction model delegates by sending "a rich context package — not a standalone query, but the full conversation". Results stream back and are woven in "at a moment appropriate to what the user is currently doing". The background model's identity is n.d.
- **Training.**
  - Data, mixture, stages, compute and ablations: n.d.
  - Safety: an automated red-teaming harness generates multi-turn refusal data, which is rendered with TTS [stated].
  - RL is not mentioned. TML does say "bitwise trainer-sampler alignment" (batch-invariant kernels, <5% overhead) is "useful for training stability as well as debugging" [stated]. That alignment is the usual prerequisite for on-policy RL [inferred].
- **Inference [stated].**
  - Each 200 ms chunk is a separate request, appended to a GPU-resident sequence. This feature has been upstreamed to SGLang.
  - MoE uses gather+GEMV kernels instead of grouped GEMM.
- **Benchmarks [stated, self-reported].** Comparators: GPT-realtime-2 (minimal / xhigh), GPT-realtime-1.5, Gemini-3.1-flash-live (minimal / high), Qwen3.5-Omni-Plus-RT. No independent check was found, and AA does not list TML.

  | Benchmark | TML | GPT-rt-2 min / xhigh | GPT-rt-1.5 | Gemini min / high | Qwen3.5 |
  |---|---|---|---|---|---|
  | FD-bench v1 latency | **0.40 s** | 1.18 / 1.63 s | 0.59 s | 0.57 / 0.94 s | 2.14 s |
  | FD-bench v1.5 avg | **77.8** | 46.8 / 47.8 | 48.3 | 54.3 / 45.5 | 39.0 |
  | BBA, standalone | **75.7** | 71.8 / 96.6 | 81.4 | 71.3 / 96.6 | 73.0 |
  | BBA, with background agent | 96.5 | | | | |

  Baseline BBA numbers are AA's.

  - FD-bench v3 quality / pass@1 with the background agent: 82.8 / 68.0, against 81.0 / 58.0 for GPT-rt-2 xhigh.
  - Audio MultiChallenge APR: 43.4, against 48.5 for GPT-rt-2 xhigh.
  - TimeSpeak 64.7 and CueSpeak 81.7, against 4.3 and 2.9 for GPT-rt-2 minimal.
- **Reading [inferred].**
  - Interaction timing is treated as a pretraining capability.
  - Knowledge is offloaded: on BBA, standalone TML is below GPT-realtime-1.5.
  - dMel is proven only in a from-scratch, frontier-scale setting.

### 2. Amazon: Nova Sonic (Apr 2025) and Nova 2 Sonic (Dec 2025)

Sources: the two technical reports in the table, plus the Nova 2 Sonic AI Service Card (current as of 2025-12-02): https://docs.aws.amazon.com/ai/responsible-ai/nova-2-sonic/overview.html

- **Architecture [stated].**
  - "Speech encoder and speech renderer modules with a core multimodal LLM."
  - The service card spells out the flow:
    - Speech is mapped into "a joint speech/text embedding space that correspond[s] to transcriptions".
    - The model uses "these transcriptions to reason and generate a text response".
    - It then synthesizes speech "while tending to the initial vectors for an adaptive voice response".
  - Sizes, codecs and frame rates: n.d.
- **Duplex [stated].**
  - The model detects end of turn "contextually", listens while it plays back, and yields to interjections.
  - Nova 2 adds a turn-taking sensitivity setting, asynchronous tool calls (the user can keep talking while tools run) and switching between voice and text.
- **Training [stated].**
  - Data: "licensed data, proprietary data, and publicly available data".
  - Stages: "pre-training, supervised fine-tuning, reinforcement learning and preference tuning". The report cites DPO, PPO and Align-SLM, an RLAIF method for SLMs with Amazon co-authors.
  - Hours and compute: n.d.
- **Results [stated].**
  - *Nova Sonic:*
    - WER averaged over 5 languages: FLEURS 3.5 and MLS 4.7, against 3.7 and 6.6 for gpt-4o-transcribe.
    - Win rates against the GPT-4o Realtime API in human evaluation: 56.9% and 52.5%.
  - *Nova 2 Sonic:*
    - BBA 87.0, against 83.0 for GPT-Realtime and 71.0 for Gemini 2.5 Flash Live.
    - ComplexFunction 65.2, against 28.7 and 3.5.
    - BFCL 74.5, against 80.4 for GPT-Realtime.
    - Median TTFA 1.39 s, against 0.98 s and 0.63 s.
- **AA today [measured].** BBA 88%, TTFA 1.14 s.
- **Reading [inferred].** This is the strongest-knowledge western S2S outside OpenAI and Google. Its recipe is a text-centric pipeline inside one model, plus RL. Its "duplex" means endpointing and barge-in, not overlapping speech.

### 3. Hume AI: EVI 3, EVI 4-mini, Octave 2, TADA

- **EVI 3 (2025-05-29) [stated].**
  - "a single autoregressive model" over text and voice tokens. Prompts mix both kinds of token to set content and voice style.
  - It "communicates with reasoning models and web search systems as it speaks".
  - Trained with a "reinforcement learning approach" to "hone the preferred qualities of any human voice".
  - Response latency measured by Hume: ~1.2 s average, against ~2.6 s for GPT-4o RT.
  - Data and size: n.d.
- **EVI 4-mini and Octave 2 (2025-10-01) [stated].** 11 languages. EVI 4-mini *requires* an external LLM.
- **TADA (arXiv 2602.23068, 2026-02-26; open 1B and 3B) [stated].**
  - Tokenizer: forced alignment produces **one continuous acoustic vector per text token**, roughly 2–3 frames/s.
  - Backbone: continued training of Llama 3.2 1B/3B in two ~200k-step stages.
  - Data: **905k h**, of which 270k h is English.
  - **Speech-free guidance.** Audio is randomly dropped during training. At inference the model blends text-only and text+speech logits (λ=0.5), which lifts tStoryCloze from 84.2 to **94.7**. Text-only scores 94.2.
  - Reports zero content hallucinations in TTS.
- **Status [stated].** "Hume is sunsetting the TTS and EVI APIs. Access ends November 13, 2026."

### 4. Liquid AI: LFM2-Audio-1.5B (2025-10-01) and LFM2.5-Audio (2025-11-28)

Source: LFM2 technical report, arXiv 2511.23404.

- **Architecture [stated].**
  - Backbone: LFM2-1.2B.
  - Input: log-mel → 17-layer FastConformer → continuous embeddings at 12.5 Hz.
  - Output: 8-codebook Mimi RVQ (1 semantic + 7 acoustic) from a small RQ-Transformer.
  - Two generation modes: "interleaved" (a fixed text/audio pattern, for S2S) and "sequential" (for ASR and TTS). Turn-based.
- **Data [stated].** ~472k h of audio per epoch (~21.5B audio embeddings), with a "large focus on synthetic conversation data".
- **Results [stated].** VoiceBench 56.78 in the blog, against 29.51 for Moshi and 63.57 for Qwen2.5-Omni-3B. The LFM2.5 card instead lists 54.92, and 52.77 for LFM2-Audio. Latency is under 100 ms.

### 5. Ultravox (Fixie), now part of Inworld

- **Design [stated].** Whisper encoder + adapter → **frozen** LLM. v0.5 (2025-02-11) uses Llama 3.1 8B and 3.3 70B. v0.6 adds Gemma 3 27B and Qwen3 32B.
- **Training [stated].**
  - Objective: a "knowledge-distillation loss where Ultravox is trying to match the logits of the text-based Llama backbone".
  - Data: ASR corpora plus continuations written by an LLM. Version scale: v0.3 2.5k h, v0.4 ~5k h, v0.4.1 >10k h.
  - Hardware: 8×H100.
- **Results [stated].** BBA 82.70 for v0.5 70B. Text output only.
- **Acquisition [stated].** Inworld acquired Ultravox on 2026-09-30 to "accelerate its speech-to-speech research".

### 6. Boson AI: Higgs Realtime

- **Product [stated].**
  - An OpenAI-Realtime-compatible WebSocket API: "one persistent session streams audio and text in both directions; the model listens and speaks natively".
  - Turn handling: `server_vad` / `semantic_vad`.
  - ~0.7 s latency and 100+ languages. Architecture and data: n.d.
- **Higgs Audio v2, Boson's TTS [stated].** Pretrained on >10M h of auto-annotated "AudioVerse", with a semantic+acoustic tokenizer and a "DualFFN" acoustic pathway. Whether Realtime shares these components: n.d.
- **AA [measured].** BBA 69%, CD 92.8, TTFA 1.47 s.

### 7. Deepslate: Opal (Berlin)

- **Product [stated].** A native S2S model ("one realtime model replaces your STT, LLM and TTS chain"). 27 languages, hosted in the EU, server-side VAD. Deepslate raised a €7.7M seed round (press). Architecture and data: n.d.
- **AA, Sep 2026 [measured].** TTFA 0.44 s, BBA 85%, CD 85.7.

### 8. Standard Intelligence: hertz-dev (Nov 2024; no audio follow-up found)

- **Model [stated].** An 8.5B "full-duplex, audio-only base model" trained on **20M h**.
- **Codec [stated].** 8 Hz latents at 1 kbps, with a 15-bit FSQ code "initially trained to contain phonetics".
- **AR model [stated].** 40 layers; two-speaker generation uses two projection heads.
- **Text init [stated].** Variants were trained both from a 2T-token text LM and from scratch, and the recipe "effectively learns linguistics both with and without text model initialization".
- **Not done.** No SFT, RL or knowledge benchmarks.

### 9. Meta: no disclosed duplex S2S model

- **Meta AI app (2025-04-29) [stated].** Shipped a full-duplex voice *demo* that is "generating voice directly", "trained on conversational dialogue". The model is not named.
- **Seamless Interaction dataset (arXiv 2506.22554) [stated].** 4,000+ h of in-person dyadic conversation from 4,000+ participants, open.
- **Muse Voice Transcribe (Meta Superintelligence Labs, 2026-09-01) [stated].**
  - Each 80 ms chunk becomes one soft token. After each chunk the model emits text or `<|next_audio|>`.
  - Endpointing and diarization are done with special tokens.
  - "Adaptive delay" is trained with RL where "WER reward and a delay reward are combined multiplicatively".
  - Streaming WER 3.1%.
- **Reading [inferred].** This is a production listen-or-emit policy learned with RL. It is directly reusable for a duplex model's listening side.

### 10. Microsoft

- **VibeVoice-Realtime-0.5B [stated].**
  - Qwen2.5-0.5B with a σ-VAE acoustic tokenizer at **7.5 Hz** and a ~40M diffusion head.
  - No semantic tokenizer.
  - An "interleaved, windowed" design consumes streaming text input.
  - ~300 ms to first audio. LibriSpeech WER 2.00.
  - This is TTS, not duplex.
- **MAI-Voice-1 (2025-08-28) and MAI-Voice-2 (2026-06-02) [stated].** TTS models. Architecture n.d.
- **Phi-4-multimodal (Feb 2025) [stated].** A frozen Phi-4-mini with mixture-of-LoRAs. ~2.3M speech hours, SFT + DPO + RLHF, speech input only.
- **No native duplex S2S model found.**

### 11. Apple

- **Scaling Properties of Continuous Diffusion SLMs (arXiv 2604.24416) [stated].**
  - A textless MM-DiT on 80 Hz log-mel, trained on ~7M h of conversational "SpeechCrawl".
  - Scaling runs cover 10¹⁸–10²¹ FLOPs.
  - The optimal tokens per parameter *falls* with compute, to 245 at 10²¹. For AR SLMs on SSL units it rises.
  - Acoustic quality saturates early, but "long-form linguistic coherence remains elusive". This holds even at 16B on "tens of millions of hours".
  - The authors conclude: "further scaling of SLMs is impractical unless a new speech representation or modeling paradigm emerge".
- **Other work [stated].**
  - Talking Turns (CMU + Apple, ICLR 2025): a turn-taking judge model trained on human conversation.
  - AFM 3 Core Advanced (2026-06-08): "natively multimodal", powering Siri's expressive voices. No duplex disclosed.

### 12. Mistral: Voxtral Realtime (arXiv 2602.11298, 2026-02-11)

- **Architecture [stated].** Streaming ASR on Kyutai's DSM.
  - A 970M causal encoder trained from scratch, downsampled to 12.5 Hz.
  - A Ministral 3B decoder that takes one step per 80 ms.
  - The delay (80–2400 ms) is set through Ada RMS-Norm conditioning.
- **Training [stated].** An encoder warm-up with the decoder frozen (5%), then end-to-end training (95%). Batch of 370 h.
- **Result [stated].** FLEURS WER 8.72 at 480 ms, against 8.23 for offline Whisper.

### 13. Voice-agent vendors with no duplex S2S disclosed

- **Deepgram Flux (2025-10-01) [stated].** Fuses STT and end-of-turn detection.
  - An `EagerEndOfTurn` event fires at 0.3–0.5 confidence for speculative LLM calls, at a cost of 50–70% more calls.
  - About 30% fewer false interruptions.
- **Cartesia [stated].** Sonic-3.6 TTS (<90 ms) and Ink STT.
- **ElevenLabs [stated].** Eleven v3 Conversational TTS (~280 ms; GA 2026-10-07) and v4. Its "speech-to-speech" is a voice changer.
- **Sarvam [stated].** Has "no dedicated speech-to-speech model" and chains ASR, LLM and TTS instead.
- **Hugging Face [stated].** `speech-to-speech` is a cascade: Silero VAD → Parakeet → LLM → Qwen3-TTS, with Smart Turn endpointing.
- **Ai2.** No speech models in its 2026 blog index.

### 14. Academic and regional open systems

- **J-Moshi** (Nagoya U, arXiv 2506.02979) [stated]
  - Setup: Moshi with Mimi frozen. A Japanese SentencePiece replaces the tokenizer, and the text embeddings are re-initialized.
  - Pretraining: **60k h** of monaural J-CHAT. Diarization turns it into stereo, with one speaker as "Moshi" and the rest as the user.
  - Fine-tuning: **344 h** of real stereo dialogue, plus **602 h** of stereo synthesized with multi-stream TTS.
  - Compute: 128 V100 × 36 h for pretraining, 16 V100 × 2 h for fine-tuning.
  - Beats dGSLM on human ratings: 2.67 vs 2.44 for naturalness, 2.19 vs 1.76 for meaningfulness.
- **Human-1** (JoshTalks, India) [stated]
  - Moshiko with a Hindi SentencePiece.
  - **26k h** of real spontaneous Hindi conversation from 14,695 speakers, ~10k steps on 8×H100 in 13 h.
  - Fine-tuning: 990 h of curated data.
  - Naturalness 4.10/5, against 4.55 for humans.
  - The closest public analogue to our data scale [inferred].
- **F-Actor** (KIT + Edinburgh, arXiv 2601.11329) [stated]
  - Llama 3.2-1B-Instruct with a frozen 12.5 fps FSQ codec and parallel codebook heads.
  - 2k h of synthetic data, trained on 4×A100 for ~2 days.
  - The text stream lowers perplexity from 25.23 to 21.45.
  - A **2-token audio delay** works best.
  - Interruptions are limited by data (correlation 0.25).
- **Llama-Mimi** (Kyoto U / NII / Nagoya, arXiv 2509.14882) [stated]
  - Llama 3.2-1B on flattened Mimi tokens, with 240k h of English, on 32 H200 for ~48 h.
  - "increasing the number of quantizers consistently improves audio quality and speaker similarity, but … at the cost of degraded spoken content quality."
- **DuplexCascade** (SB Intuitions + U Tokyo, arXiv 2603.09180) [stated]
  - DSM-ASR, then a LoRA-tuned Qwen2-7B on **0.6 s micro-turns** with 7 control tokens, then DSM-TTS.
  - Training: 50k dialogues on 8×H100 for 5 h.
  - FDB turn-taking accuracy 0.858, against 0.778 for Gemini Live and 0.395 for Moshi. VoiceBench 65.41.
- **HiPLEX** (Qualcomm AI Research + KAIST, arXiv 2610.07727) [stated]
  - Method: RL on Moshi and PersonaPlex that splits a *control* policy (`pad` / `epad` / `cont`) from a *content* policy.
    - Timing rewards use event-causal credit masking.
    - An LLM-judge reward applies only to content.
  - Data: 73,561 windows from Seamless Interaction.
  - Results on Moshi:
    - CANDOR pause takeover 0.579 → 0.306.
    - Backchannel takeover 0.473 → 0.127.
    - Interruption judge 3.77 → 4.08.
    - Interruption latency 0.85 → 0.44 s.
    - Plain GRPO does worse.
  - Timing-only reward degrades quality. On PersonaPlex, turn takeover falls from 0.88 to 0.71.
- **Game-Time and TiCo** (NTU + MIT) [stated]
  - Game-Time: "nearly all models degrade substantially under temporal constraints".
  - TiCo inserts Spoken Time Markers (e.g., `<10.6 seconds>`) and applies RL post-training. Duration error falls 2.7×.
- **Benchmarks.** Full-Duplex-Bench comes in three versions:
  - v1 (arXiv 2503.04721): pause handling, backchannel, turn-taking, interruption.
  - v1.5 (arXiv 2507.23159): overlap from interruptions, backchannels, side talk and noise.
  - v2 (arXiv 2510.07838): multi-turn, with a gpt-realtime examiner.
- **"Interleaved Speech Language Models Latently Work In Text"** (arXiv 2606.22473) [stated]
  - The text token for each spoken word becomes decodable in intermediate layers.
  - This is strongest with text-LM init plus interleaving.
  - It correlates with spoken factual-knowledge retrieval.
- **HF community blog: a 600M Moshi-style model** (2026-09-22) [stated]
  - Setup: **Qwen3-0.6B-Base** with frozen 8-codebook Mimi and a depth transformer. Text leads audio by 2 frames.
  - Data: ~100 h of synthetic dialogue plus 423 PersonaPlex teacher rollouts.
  - Findings:
    - Text converged, but codebooks 1–7 did not learn even while overfitting a single dialogue.
    - "A pretrained text LM plus the inner monologue gives you conversation for close to free."
    - 300 clean hours beat 1,500 noisy hours.

## Cross-cutting patterns

1. **Text stays in the loop wherever knowledge is strong.**
   - Nova renders speech from an internal text response.
   - EVI uses text+voice tokens and an external LLM.
   - LFM2 interleaves text and audio.
   - Ultravox and DuplexCascade keep a text LLM intact.
   - TADA and TASTE2 (arXiv 2609.08956) give one audio vector per text token.
   - Textless systems (hertz-dev, Apple CD-SLM) have no demonstrated knowledge.
   - The open duplex-native models are AA's low-BBA outliers [measured].
2. **Light inputs, heavier output heads.**
   - Inputs are near-raw or continuous: dMel, FastConformer features, causal log-mel encoders, or one soft token per 80 ms.
   - Outputs come in two families:
     - RVQ plus a depth/RQ-Transformer (Moshi family, LFM2).
     - Continuous latents plus a flow or diffusion head (TML, VibeVoice-RT, TADA, Apple).
   - The frontier is moving toward the second [inferred].
3. **Step rates are 80–200 ms.**
   - 12.5 Hz: Mimi, LFM2, Voxtral RT, Muse, F-Actor.
   - 7.5–8 Hz: VibeVoice, hertz.
   - TML uses a 5 Hz micro-turn cadence; its internal token rate is n.d.
4. **There are three duplex designs:**
   - (a) Frame-synchronous dual streams: Moshi descendants, hertz-dev.
   - (b) Micro-turn interleaving: TML, DuplexCascade.
   - (c) Half-duplex with learned endpointing and barge-in: Nova, EVI, Higgs, Opal, Flux.
   - The FD-bench v1.5 leader uses (b).
5. **Heavy work is delegated asynchronously.** TML, Nova 2 and EVI keep the real-time model small and fast and send reasoning and tools to a background process.
6. **Data shape.**
   - Single-stream audio pretraining: 0.24M–20M h.
   - Real stereo duplex data is small: hundreds of hours, or 4k h for Seamless Interaction.
   - Synthetic multi-stream TTS fills the gap.
   - Monaural audio becomes pseudo-stereo through diarization.
7. **Post-training is RL wherever it is disclosed.**
   - Production: Nova, EVI 3, Phi-4-MM, Muse.
   - Research: HiPLEX, TiCo.
   - Design rules: route timing and content rewards to separate decisions, and combine accuracy and latency multiplicatively.

## Lessons for our setup

Our setup: Qwen3 LLM, Qwen3 tokenizer and vocoder, ~10K h of duplex speech. Symptoms: weak continuation and weak semantics.

1. **Treat the 10K h of duplex data as a fine-tuning stage, not pretraining.**
   - Comparable successes all started audio-pretrained: J-Moshi (60k h on top of Moshi) and Human-1 (26k h on top of Moshiko).
   - A cold start fails in exactly our way: the 600M Qwen3 run's fine codebooks never learned.
   - Add a single-stream continuation stage of about 0.2–1M h, as in Llama-Mimi, LFM2 and TADA. Part of it can come from diarized pseudo-stereo, as J-Moshi did [inferred].
2. **Fix semantics with text anchoring, not more audio.**
   - Keep a time-aligned text stream that leads the audio by ~2 tokens (F-Actor, HF 600M).
   - Add speech–text interleaved data to mid-training.
   - Replay text-only data.
   - Consider KD onto the frozen Qwen3's text logits (Ultravox).
   - Consider TADA-style text-only guidance at inference.
   - Apple's result says audio scale alone will not fix knowledge.
3. **Budget the codebooks.** More codebooks hurt content (Llama-Mimi). Prioritize the semantic codebook through a delay or curriculum on the fine codebooks. Alternatively, test a continuous latent with a flow head, if the vocoder allows [inferred].
4. **Duplex data.**
   - Use real stereo where licensed (Seamless Interaction).
   - Use multi-stream TTS for backchannels and interruptions, which are data-limited (F-Actor).
   - Filter aggressively.
5. **Post-training.**
   - Run GRPO-style RL with timing and content rewards factorized (HiPLEX).
   - Use a multiplicative accuracy × latency reward for listening (Muse).
   - Add time markers if timed behaviors matter (TiCo).
   - Teacher rollouts from a stronger duplex model are cheap timing supervision, if the license allows.
6. **Plan to delegate reasoning and tools to a background text LLM.** Even TML trails GPT-realtime-1.5 on BBA without it.
7. **Evaluate both axes.**
   - Interaction: FD-bench v1, v1.5 and v2, Game-Time.
   - Knowledge: BBA, VoiceBench, Audio MultiChallenge.
   - Continuation: sStoryCloze/tStoryCloze, SALMon.

## Not found / gaps

- No disclosed duplex S2S model from Meta, Microsoft, Apple, Cartesia, ElevenLabs, Deepgram, Sarvam, HF or Ai2.
- Meta's reported voice-startup acquisitions were not verified.
- No system here reports τ-voice or URO-Bench.
- The search budget ran out mid-review. Character.AI, Speechmatics, Zyphra, Canopy Labs and other Korean or Japanese labs were not checked.

## Sources (accessed 2026-10-12; in addition to the table links)

**Frontier labs and industry**
- TML: https://thinkingmachines.ai/blog/interaction-models/
- Amazon:
  - https://docs.aws.amazon.com/ai/responsible-ai/nova-2-sonic/overview.html
  - Align-SLM: https://arxiv.org/abs/2411.01834
- Hume:
  - https://www.hume.ai/blog/octave-2-launch
  - https://dev.hume.ai/voice/intro (sunset notice)
  - https://dev.hume.ai/docs/speech-to-speech-evi/configuration/evi-version
  - https://arxiv.org/html/2602.23068
  - RW-Voice-EQ: https://arxiv.org/abs/2607.14846
- Liquid AI: https://arxiv.org/abs/2511.23404
- Ultravox / Inworld:
  - https://github.com/fixie-ai/ultravox/releases
  - https://inworld.ai/blog/inworld-acquires-ultravox
- Boson:
  - https://github.com/boson-ai/higgs-audio/blob/main/README_V2.md
  - https://fortune.com/2026/07/27/ex-aws-scientist-alex-smola-boson-ai-speech-to-speech-voice-model/
- Standard Intelligence: https://huggingface.co/si-pbc/hertz-dev
- Meta:
  - https://about.fb.com/news/2025/04/introducing-meta-ai-app-new-way-access-ai-assistant/
  - https://ai.meta.com/research/seamless-interaction/
- Microsoft:
  - https://microsoft.ai/news/two-new-in-house-models/
  - https://ai.azure.com/catalog/models/MAI-Voice-2
- Apple:
  - https://arxiv.org/html/2604.24416
  - dMel: https://arxiv.org/abs/2407.15835
  - Talking Turns: https://arxiv.org/abs/2503.01174
  - https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models
- Mistral: https://arxiv.org/html/2602.11298v1

**Vendors**
- https://deepgram.com/flux
- https://www.cartesia.ai/sonic
- https://elevenlabs.io/docs/overview/models
- https://docs.sarvam.ai/api-reference-docs/getting-started/models
- https://github.com/huggingface/speech-to-speech
- https://allenai.org/blog

**Academic and community**
- https://arxiv.org/html/2506.02979v1
- https://arxiv.org/html/2601.11329
- https://arxiv.org/html/2509.14882v2
- https://arxiv.org/html/2603.09180
- https://arxiv.org/html/2610.07727
- Game-Time: https://arxiv.org/abs/2509.26388
- TiCo: https://arxiv.org/abs/2603.22267
- Full-Duplex-Bench:
  - v1: https://arxiv.org/abs/2503.04721
  - v1.5: https://arxiv.org/abs/2507.23159
  - v2: https://arxiv.org/abs/2510.07838
- Interleaved SLMs latently work in text: https://arxiv.org/abs/2606.22473
- TASTE2: https://arxiv.org/abs/2609.08956
- HF 600M Moshi-style blog: https://huggingface.co/blog/edwixx/shrinking-full-duplex-speech-training-a-600m-moshi
- AA leaderboard: https://artificialanalysis.ai/models/speech-to-speech
