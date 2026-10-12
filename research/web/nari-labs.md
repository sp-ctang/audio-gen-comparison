# Nari Labs: serving tool-using full-duplex S2S agents, plus Dia / Dia2 and Qwen3-TTS serving

Labels: **[stated]** comes from an official source (Nari Labs, or NVIDIA for the model it serves). **[measured]** is a third-party evaluation. **[inferred]** is an inference, and the tag says whose.

## Summary

- **The linked post is about inference engineering, not model training.** In "Nemotron-VoiceChat: 64 Tool-Using S2S Agents on 1 GPU" (blog index date 30 Sep 2026), Nari describes a custom serving engine for NVIDIA's open full-duplex model NemotronLabs VoiceChat 11B. The headline result [stated]:
  - 64 concurrent conversations on one H100 SXM, 16 of them making tool calls.
  - "no missed 80 ms processing deadlines or dropped frames".
  - About **$0.067 GPU-cost per conversation-hour**: $4.29/h ÷ 64.
- **Baselines on the same GPU [stated].** vLLM-Omni manages 1 session, NVIDIA NIM 4 and NVIDIA's C++ engine 6.
- **The model being served** comes from NVIDIA, Aug–Oct 2026, and is detailed in `../papers/nemotronlabs-voicechat.md` [stated: NVIDIA paper and card]:
  - A 9B Nemotron-Nano-v2 *base* LLM, a 600M streaming FastConformer encoder and a separate 977M streaming TTS.
  - The LLM emits only agent text plus a function-call channel. Its audio-loss weight is 0.
  - Continued pre-training (CPT) used TTS-rendered pseudo-dialogues from text corpora. The bucket's scale is printed as "≈530k" with no unit (VoiceChat Tab.5); it may be hours, but the paper never says. SFT added about 72k h of duplex chat and about 7k h of synthetic tool-calling data.
  - On FDB 3.0, tool-selection F1 is 82.5%, but argument accuracy is only 42.2%.
- **Who Nari Labs is.** A YC-backed SF/Seoul team that made the **Dia** (1.6B) and **Dia2** (1B/2B, streaming over Mimi) dialogue TTS models, then became an inference company for open speech models in 2026. It now serves Qwen3-TTS 1.7B, Qwen3-ASR 1.7B, PersonaPlex 7B (80 sessions per H100) and Nemotron VoiceChat [stated].
- **Why it matters for us** [inferred: me]:
  - Nari's Qwen3-TTS posts describe the Qwen3-TTS 12 Hz generation stack: a talker for codebook 0, a 15-step code predictor and a stateful streaming codec. This is likely the same tokenizer and decoder family as ours.
  - In Coval's run, two hosts of the presumably identical open Qwen3-TTS weights differed: Baseten 6.0% WER, Nari 3.8% [measured: Coval, as reported by Nari]. Alibaba's official Qwen3-TTS-Flash API scored 8.8%, but it may be a different model.
  - So the serving and decoding implementation alone shifts measured quality. We should audit our decoding stack before concluding that continuation quality is a training problem.

## Sources (accessed 2026-10-12)

**Nari Labs**
- The post: https://narilabs.com/blog/making-tool-using-voice-agents-more-affordable/ (the page itself shows no date; the date comes from the blog index https://narilabs.com/blog/)
- https://narilabs.com/blog/personaplex-7b-80-sessions-one-h100/ (15 Sep 2026)
- https://narilabs.com/blog/qwen3-tts-speed-cost-frontier/ (19 Aug 2026)
- https://narilabs.com/blog/introducing-nari-qwen3-tts/ (10 Sep 2026)
- https://narilabs.com/blog/input-streaming-for-nari-qwen3-tts/ (24 Sep 2026)
- https://narilabs.com/blog/nari-labs-leads-coval-voice-ai-benchmarks/ (14 Sep 2026)
- https://narilabs.com/blog/nari-model-apis-general-availability/ (17 Sep 2026)
- https://narilabs.com/pricing/
- https://narilabs.com/about/
- https://narilabs.com/product/training/

**Nari open models**
- https://github.com/nari-labs/dia (and raw README)
- https://huggingface.co/nari-labs/Dia-1.6B-0626
- https://github.com/nari-labs/dia2 (and raw README)
- https://huggingface.co/nari-labs/Dia2-2B, including `config.json`

**NVIDIA model being served**
- Paper: https://arxiv.org/html/2609.21967
- Model card: https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
- Related tool-call architecture paper: https://arxiv.org/abs/2609.19334
- PersonaPlex paper: https://arxiv.org/abs/2602.06053
- PersonaPlex model card: https://huggingface.co/nvidia/personaplex-7b-v1

**Third-party**
- Coval results as summarized by kompozy (search snippet): https://kompozy.io/news/nari-qwen3-tts-asr-coval-benchmarks

**Could not find**
- A Nari post giving a parameter count or architecture diagram for its engine.
- Dia/Dia2 training-data sizes. No Dia technical report exists.

## What it is

The post covers **only serving**. The engine design, as Nari states it [stated]:

- **Kernel.** A persistent GPU kernel built for the model, which cuts launch overhead between operations.
- **Clock.** The model runs on an 80 ms clock, and the client keeps a 160 ms playback buffer.
- **Scheduler.** It reserves GPU time for live conversations and fits tool work "into the time left before the next audio deadline".
  - Sessions that are generating tool calls are batched together.
  - Tool results are ingested with chunked prefill.
  - While one session handles a tool call or result, its conversational model pauses. The other sessions keep their cadence.
- **Precision and context.** Mixed precision based on BF16, with a "1,500-position context". At 12.5 Hz that is 120 s, which matches the model's roughly 2-minute training context [inferred: me].

**Microbenchmarks, single session [stated]**

| Engine | Max sessions per H100 | GPU $/session-h | Tool call (32 tok) | Tool result (128 tok) |
|---|---|---|---|---|
| vLLM-Omni | 1 | 4.290 | 2.508 s | 10.189 s |
| NVIDIA NIM | 4 | 1.073 | 0.474 s | 1.919 s |
| NVIDIA C++ engine | 6 | 0.715 | 0.243 s | 0.953 s |
| Nari | **64** | **0.067** | 0.351 s | 0.611 s |

**Under 64-session load [stated].** Tool-call generation took 0.877 s. From tool-result arrival to the resumption of live audio processing took 0.781 s.

**Demo [stated].** An unedited call books a dental appointment through tools.

**Missing.** The post reports no speech-quality, WER or task-success metrics under load.

**Sibling post on PersonaPlex 7B (15 Sep 2026) [stated].** PersonaPlex 7B is NVIDIA's model, fine-tuned from Moshi.
- Throughput: 80 concurrent sessions per H100, at "below 10 cents per hour" per user. vLLM-Omni manages 1 session in unified mode and 28 standalone.
- The tricks:
  - the depformer loop runs only "the 8 needed" steps instead of 16 (the agent stream's codebooks only [inferred: me])
  - CUDA Graphs
  - FP8 temporal KV cache
  - BF16 weights
  - fixed-shape batches on a synchronized 80 ms clock

## Architecture

**NemotronLabs VoiceChat 11B (NVIDIA) [stated: arXiv 2609.21967 and HF card].** Full detail is in `../papers/nemotronlabs-voicechat.md`. In brief:
- **Parts:** a 600M FastConformer encoder (80 ms frames), the Nemotron-Nano-9B-v2-Base hybrid Mamba/Transformer, channels for agent text, function calls and an RNN-T user transcript, and a 977M streaming TTS (31-level RVQ codec at 12.5 Hz).
- **No audio from the LLM:** "Direct audio-codec prediction is disabled… (audio-loss weight 0.0)". The TTS consumes the incremental assistant text stream.
- **Timing:** agent-text targets are shifted 160 ms later than the user audio; the function channel is not shifted.

**Dia (1.6B) [stated].** "Heavily inspired by SoundStorm, Parakeet, and Descript Audio Codec" (86 tokens per second of audio).
- English only, with `[S1]`/`[S2]` speaker tags, 21 nonverbal tags and voice cloning from a 5–10 s prompt.
- Speed: about ×2.1 real-time on an RTX 4090 in bf16 with compilation, using about 4.4 GB of VRAM.

**Dia2 (1B/2B; released 19 Nov 2025 per the README news line) [stated].** A streaming dialogue TTS over Kyutai Mimi at about 12.5 Hz that "can start generating as the first few words are given as input".
- Speaker consistency comes from prefix audio, with Whisper transcribing the prefixes.
- Output is capped at 2 minutes (1,500 steps), English only.
- The README credits Kyutai TTS and Sesame as inspiration.

**Dia2-2B `config.json` [stated config; reading inferred: me].**
- Decoder: 28 layers at d=2048 (16 query / 8 KV heads). Depformer: 4 layers at d=1024.
- 34 channels, which I read as 32 Mimi codebooks plus text and action streams.
- A per-codebook delay pattern of about 16–18 steps, consistent with Kyutai-style delayed streams in which text leads audio.

**Qwen3-TTS 12 Hz, as described by Nari [stated: Nari Qwen3-TTS posts].** The model is Qwen3-TTS-12Hz-1.7B-CustomVoice.
- **Talker:** predicts the first codebook token for each frame.
- **Code Predictor:** fills the other "15 codebook tokens", always in "15 steps per frame".
- **Codec:** decodes with Transformers plus CNNs. Nari made it *state-cached*, carrying the Transformer context and convolution state across chunks.
- **Silence:** about 80 ms of leading silence was trimmed.

## Training data & compute

**Nari's own work.** The post involves no training. Nari offers fine-tuning and distillation services but publishes no recipes [stated]. Dia and Dia2 were trained on TPU Research Cloud, with no hours or data sources disclosed [stated].

**NemotronLabs VoiceChat (NVIDIA) [stated: arXiv 2609.21967; full mixture in the paper note].**
- **CPT:** TTS-rendered pseudo-dialogues, at a scale printed as "≈530k" with no unit (possibly hours), made by splitting text passages into alternating user/agent sentences.
- **SFT mixture:** about 47% retention (including 8.5% text-only with a text-to-text loss), about 24% conversational (including about 72k h of duplex chat), about 26% synthetic tool calling (about 7k h) and about 3% safety.
- **Augmentation:** interruptions p=0.1, backchannels p=0.05, noise p=0.5, room impulse responses p=0.8.
- **Compute:** 64 GPUs, LR 5e-5. GPU-hours are not reported.
- **Order:** the LLM is trained, not frozen; the TTS is trained separately, with no gradient flow between them.

## Post-training / alignment

Nari reports no post-training stage. NemotronLabs VoiceChat has no RL or preference stage either; its only alignment is about 1.1k h of spoken-safety SFT data, and its AdvBench score is 100.0 [stated].

## Evals & numbers

**NemotronLabs VoiceChat [stated: NVIDIA paper; more in the paper note].**
- **VoiceBench:** 55.1. Moshi scores 29.5, PersonaPlex 30.6 and Freeze-Omni 55.2.
- **FDB 1.0 smooth turn-taking:** takeover rate (TOR) 81.5% at 0.448 s latency.
- **FDB 3.0 (tool use):** tool-selection F1 82.5% (Gemini Live 2.5: 78.6%), argument accuracy 42.2%, pass@1 33.0%. NVIDIA recommends at most about 5 tools per session.
- **Speed:** p95 of 118 ms per 160 ms chunk on one stream.

**Related NVIDIA alternative to the native function channel [stated: arXiv 2609.19334 abstract; NVIDIA affiliation inferred: me, from the author list].** A duplex frontend emits a *delegation token* and forwards streaming transcripts to a text backend LLM (e.g., Qwen3-235B-A22B), which makes the tool call. It reports 92–97% tool-call recall and 81.2% correct rejection of irrelevant calls.

**Coval benchmark of Nari's serving (14 Sep 2026) [measured: Coval, as reported by Nari; consistent with the kompozy summary].**
- Qwen3-TTS Fast: time to first audio (TTFA) p50 63 ms, WER 3.8%.
- Baseten Qwen3-TTS: 101 ms, 6.0%.
- Official Qwen3 TTS Flash (Alibaba's API, possibly not the open 1.7B model): 692 ms, 8.8%.
- Qwen3-ASR Fast: time to final segment (TTFS) p50 44 ms, WER 3.6%. AssemblyAI Universal 3.5 Pro scored 3.5%.

**Nari's own Qwen3-TTS load test [stated].**
- One H100 SXM sustains 10 requests per second with p95 TTFA under 50 ms, and stays under 100 ms at 20 RPS.
- Baselines at 1 RPS (p95 TTFA): vLLM-Omni 56.8 ms, VoxServe 49.3 ms, SGLang-Omni 120.9 ms.

**Prices [stated].**
- Qwen3-TTS: $5 per 1M characters (Standard), $10 (Fast).
- Qwen3-ASR: $0.06/h (Standard), $0.12/h (Fast).
- S2S hosting: no public price.

## Lessons for our setup

1. **Consider making the LLM emit text and leaving audio to a streaming TTS.** NemotronLabs VoiceChat sets the backbone's audio loss to zero and hands speech to a separately trained streaming TTS. This sidesteps both of our problems: continuation quality is owned by a dedicated TTS, and knowledge stays in a text-emitting LLM. If we keep the Qwen3 backbone predicting audio tokens, we carry both burdens; a thinker–talker split is the main alternative to evaluate [inferred: me].
2. **Recipe for injecting knowledge.** Render text pretraining corpora into speech as pseudo-dialogues. VoiceChat's bucket is "≈530k" with no unit; if that means hours, it is about 50× our duplex set. Then keep about 47% retention data and 8.5% text-only data, with a text-to-text loss, during SFT. This is the most concrete public recipe for "speech that knows things". Our 10K h of real duplex speech cannot supply world knowledge by itself [inferred: me, from NVIDIA numbers].
3. **Time alignment and augmentation.** Shift agent text targets 160 ms after user audio, so the model hears more before committing. Inject interruptions and backchannels and add noise, room response and microphone augmentation. All are cheap additions to our mid-training data pipeline [inferred: me].
4. **Audit the decoding and serving stack first.** For a Qwen3 12 Hz stack, check:
   - the code predictor runs exactly 15 fixed steps per frame
   - the codec decoder carries Transformer and convolution state across chunks
   - leading silence is handled
   - sampling settings match the reference
   Coval measured hosted Qwen3-TTS at 3.8% WER (Nari) against 6.0% (Baseten), presumably on the same open weights. Weak "continuation quality" may be partly an inference artifact [inferred: me].
5. **Design for serving now.** An 80 ms per-frame clock makes throughput a scheduling problem. Several choices let one H100 hold 64–80 sessions instead of 1–6 [stated numbers; inference me]:
   - fixed-shape per-frame compute
   - a bounded context (1,500 positions ≈ 2 min)
   - no wasted depth steps (PersonaPlex 16→8)
   - FP8 KV cache
   - tool calls handled out of band
6. **Tool use.** Picking the right tool is easy (F1 82.5%), but generating arguments in a duplex stream is hard (42.2%). Delegating to a text backend LLM (2609.19334) is a pragmatic option if agentic use matters [inferred: me].
7. **Dia2 as a reference for streaming TTS.** Dia2 is an Apache-2.0 streaming TTS over Mimi with prefix conditioning. It is a candidate "talker", or a source of synthetic dialogue for data generation; the licence permits this [inferred: me].

## Open questions

- Is Nari's engine available as open source? The post gives no code link.
- How do speech quality and turn-taking hold up at 64 concurrent sessions compared with one?
- What data trained Dia and Dia2 (hours, sources)? What exactly are the 34 channels and the "action" tokens in Dia2?
- Does NVIDIA's zero audio-loss design cost naturalness or contextual prosody compared with joint audio prediction? The paper has no ablation.
- Will Nari publish fine-tuning results for duplex models? It advertises a training service.
