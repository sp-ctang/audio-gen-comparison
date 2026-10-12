# Google Gemini Live (native audio): 2.0 Flash Live → 3.8 Live

Labels: **[stated]** = official Google source (docs, model cards, eval PDFs, Google blogs, Google staff); **[measured]** = third-party eval or leaderboard; **[inferred: who]** = an inference, with whose it is ("me" = this note's author). Keys like [S1] refer to Sources.

## Summary

- **Five generations in 21 months.** Google went from 2.0 Flash (exp) to 2.0 Flash Live (half-cascade: native audio in, TTS out), then 2.5 Flash native audio (May/Sep/Dec 2025), 3.1 Flash Live (Mar 2026) and 3.8 Live + 3.8 Live Extended Thinking (GA 2026-09-15). [stated S9]
- **Training is barely disclosed.** The 3.8 Audio card says only "based on Gemini 3 Pro". Gemini 3 Pro is a sparse MoE with native audio input and speech in pretraining; post-training used instruction tuning, RL and human preferences. The one audio-specific architecture statement is Gemini 2.5's "causal audio representations" for streaming audio in *and out*, plus "improved post-training recipes" for thinking, affect and tool use. No hours, stages or compute. [stated S14,S18,S26]
- **Turn-taking is half-duplex with fast barge-in, run by the server.** Server-side VAD (~800 ms default end silence) ends turns. A barge-in cancels generation and truncates history to the audio already delivered. "Proactive audio" lets the model decline to respond. Google claims no simultaneous listen-and-speak modelling. [stated S4,S8; inferred: me]
- **The 2026 step is to separate reasoning from speaking.** 3.8 Live Extended Thinking reasons and calls async tools "in the background" while speaking fillers. It scores AA S2S Index 82.6 and τ-Voice 68.6% at 1.35 s TTFA. Plain 3.8 Live scores τ-Voice 30.1% at 1.18 s. [stated S19; measured T1]
- **Spoken answers depend on thinking.** On Big Bench Audio, 3.1 Flash Live scores 70.5% at minimal thinking and 95.9% at high; 2.5 native audio scores 69% and 90.7%. Weak semantic knowledge in speech is expected without a text reasoning channel. [stated S17; inferred: me]

## Sources (all accessed 2026-10-12)

**Google API docs**
- [S1] https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live
- [S2] https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live-extended-thinking
- [S3] https://ai.google.dev/gemini-api/docs/live-api/thinking (`.md.txt` read)
- [S4] https://ai.google.dev/gemini-api/docs/live-api/capabilities
- [S5] https://ai.google.dev/gemini-api/docs/live-api/best-practices
- [S6] https://ai.google.dev/gemini-api/docs/live-api/session-management
- [S7] https://ai.google.dev/gemini-api/docs/live-api/tools (stale: lists only 3.1/2.5)
- [S8] https://ai.google.dev/api/live (WebSocket reference)
- [S9] https://ai.google.dev/gemini-api/docs/changelog
- [S10] https://ai.google.dev/gemini-api/docs/pricing
- [S11] https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview
- [S12] https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-native-audio-preview-12-2025
- [S13] https://ai.google.dev/gemini-api/docs/audio

**Model cards and eval reports**
- [S14] 3.8 Audio card: https://deepmind.google/models/model-cards/gemini-3-8-audio/
- [S15] 3.8 Live eval PDF: https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_live_model_evaluation.pdf
- [S16] 3.1 Flash Audio card: https://deepmind.google/models/model-cards/gemini-3-1-flash-live/
- [S17] 3.1 Flash Live eval PDF: https://storage.googleapis.com/deepmind-media/gemini/gemini-3-1-flash-live-model-evaluation.pdf
- [S18] Gemini 3 Pro card: https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf

**Google blogs**
- [S19] 3.8 launch (Ouyang & Jaganathan): https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-8-live-gemini-3-8-live-extended-thinking/
- [S20] Live Avatar GA (09-24): https://cloud.google.com/blog/products/ai-machine-learning/gemini-3-8-live-with-live-avatar-is-now-generally-available
- [S21] https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-flash-live/
- [S22] https://blog.google/innovation-and-ai/technology/developers-tools/build-with-gemini-3-1-flash-live/
- [S23] https://blog.google/products/gemini/gemini-audio-model-updates/ (Dec 2025)
- [S24] https://blog.google/technology/google-deepmind/gemini-2-5-native-audio/ (2025-06-03)
- [S25] https://developers.googleblog.com/en/the-next-chapter-of-the-gemini-era-for-developers/ (Dec 2024)
- [S35] https://deepmind.google/models/gemini-audio/
- [S36] https://ai.google.dev/api/multimodal-live (half-cascade vs native-audio text seen only as a search snippet; the current page has dropped it)

**Google research**
- [S26] Gemini 2.5 report §2.6–2.7: https://arxiv.org/abs/2507.06261
- [S27] Gemini 1.0 report: https://arxiv.org/abs/2312.11805
- [S28] https://deepmind.google/discover/blog/pushing-the-frontiers-of-audio-generation/ (Oct 2024)
- [S29] https://research.google/blog/real-time-speech-to-speech-translation/ (2025-11-19)
- [S30] SpeechSSM: https://arxiv.org/abs/2412.18603
- [S31] AudioPaLM: https://arxiv.org/abs/2306.12925
- [S32] SpectroStream: https://arxiv.org/abs/2508.05207
- [S33] Gemma 4: https://arxiv.org/abs/2607.02770
- [S34] SequenceLayers: https://arxiv.org/abs/2507.23292

**Third-party evals**
- [T1] https://artificialanalysis.ai/speech-to-speech
- [T2] https://artificialanalysis.ai/methodology/speech-to-speech-benchmarking
- [T3] https://artificialanalysis.ai/articles/announcing-the-speech-agent-arena
- [T4] https://artificialanalysis.ai/articles/announcing-the-artificial-analysis-speech-to-speech-index
- [T5] τ-Voice: https://arxiv.org/abs/2603.13686
- [T6] FDB v1: https://arxiv.org/abs/2503.04721
- [T7] FDB v1.5: https://arxiv.org/abs/2507.23159
- [T8] FDB v3: https://arxiv.org/abs/2604.04847
- [T9] https://arxiv.org/abs/2610.12802
- [T10] APEX-Voice: https://arxiv.org/abs/2609.34973
- [T11] https://the-decoder.com/googles-updated-gemini-2-5-flash-native-audio-handles-complex-voice-tasks-better/
- [T12] https://www.datacamp.com/blog/gemini-3-8-live
- [T13] https://finance.biggo.com/news/3b56bb28bd8e5502 (Thor Schaeff talk)

**Not accessible**
- TechRepublic: HTTP 403.
- tryfriday.ai: title only.
- pasqualepillitteri.it: connection error.
- Vertex / Gemini Enterprise model pages: navigation only.
- web.archive.org: blocked.
- No primary EVA-Bench leaderboard found; search budget exhausted before searching for talks/podcasts.
- aimodeling.com's unsourced "40-layer Transformer" claim was ignored.

## Model lineage & timeline

| Date | Model / event | Notes |
|---|---|---|
| 2024-12-11 | 2.0 Flash exp + Multimodal Live API | Streaming A/V in; "interruptions and voice activity detection". Native audio out (8 voices) for early testers only. [stated S25] |
| 2025-04-09 | `gemini-2.0-flash-live-001` | Session mgmt, context compression, configurable VAD [stated S9]. Half-cascade = "native audio input and text-to-speech output" [stated S36] |
| 2025-05-20 | `gemini-2.5-flash-preview-native-audio-dialog`, `…-exp-native-audio-thinking-dialog`, `gemini-live-2.5-flash-preview` (half-cascade) | Native audio out, affective dialog, proactive audio, thinking; async function calls. 24+ languages. [stated S9,S24] |
| 2025-09-23 | `…native-audio-preview-09-2025` | "Improved function calling and speech cut off handling"; CFB-Audio 66.0% [stated S9,S17] |
| 2025-12-09/12 | Half-cascade shut down; `…native-audio-preview-12-2025` | CFB-Audio 71.5% (gpt-realtime 66.5%); instruction adherence 84→90%; into Gemini Live and Search Live [stated S23; T11] |
| 2026-03-26 | `gemini-3.1-flash-live-preview` ("audio-to-audio (A2A)") | `thinkingLevel` minimal…high (default minimal); sync-only tools; no affective/proactive; 90+ languages; follows conversation thread "twice as long" [stated S9,S11,S21] |
| 2026-08-26 | `gemini-3.5-transcribe(-live)` | Separate STT models [stated S9] |
| 2026-09-15 | `gemini-3.8-live`, `gemini-3.8-live-extended-thinking` (GA) | "Based on Gemini 3 Pro"; knowledge cutoff Jan 2025 [stated S1,S2,S14] |
| 2026-09-22/24 | 3.8 Flash(-Lite) TTS; Live Avatar GA on Gemini Enterprise | ET in Enterprise private preview [stated S9,S20] |

## Product & API facts (3.8 Live in detail)

- **Model and limits.** 131,072 input / 65,536 output tokens. Input: text, image, audio or video; output: audio, with transcripts via `output_audio_transcription`. [stated S1,S4]
- **Supported.** Search grounding; function calling (async `NON_BLOCKING` default, `BLOCKING` opt-in); "interleaved reasoning" with no `thinking_level`. Not supported: caching, code execution, structured outputs, URL context, Maps, Batch. [stated S1,S3]
- **Migration from 3.1.** [stated S1]
  - Proactive audio is permanently on; false returns an error.
  - `enable_affective_dialog` is removed.
  - Turn coverage defaults to `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO`.
  - `send_client_content` is allowed all session, with `user`/`model` roles.
- **Extended Thinking (ET).** [stated S2,S3]
  - low/medium/high thinking (no minimal) and `includeThoughts` summaries.
  - Async-only tools; blocking errors.
  - `turnComplete` ends an *utterance*; `interaction_status` (IN_PROGRESS/IDLE) ends the *task*.
- **Audio.** Raw 16-bit LE PCM, 16 kHz in (server resamples) and 24 kHz out. Send 20–40 ms chunks; "approximately 25 tokens per second". [stated S4,S5]
- **Sessions.** [stated S5,S6,S8]
  - Without compression: 15 min audio-only, 2 min audio+video.
  - Connection ~10 min; resumption handles valid 2 h.
  - Sliding-window compression triggers at 80% of context and compresses to ÷2.
- **Languages and voices.** 97 languages with mid-conversation switching (blog); docs list 99. Native audio uses the TTS voice set. [stated S19,S4]
- **Pricing** (shared with 3.1 Flash Live). [stated S10,S5]

  | Direction | Text | Audio |
  |---|---|---|
  | Input | $0.75/M | $3/M ($0.005/min) |
  | Output (incl. thinking) | $4.50/M | $12/M ($0.018/min) |

  With proactive audio, input is billed "the entire time the Live API is listening".
- **Deployment.** [stated S19,S20]
  - 3.8 Live: Search Live. ET: Gemini app, Workspace, Gmail/Keep.
  - Live Avatar; SynthID on all audio.
  - Equal AI: ">1 million live calls daily".

## Architecture (stated vs inferred)

**Stated**
- 3.8 Audio is "based on Gemini 3 Pro", a "sparse mixture-of-experts… transformer… with native multimodal support for text, vision, and audio inputs". Every 3 Pro-family model, 3.1 Flash Live included, is "based on Gemini 3 Pro". [S14,S18]
- Gemini 1.0 ingests 16 kHz audio "from Universal Speech Model (USM) features". [S27]
- Gemini 2.5 "incorporated causal audio representations that also allow streaming audio into and out of Gemini 2.5". It "was trained to perform audio generation tasks such as text-to-speech or native audio-visual to audio out dialog". [S26]
- Audio understanding uses 32 tokens/s [S13]; Live uses about 25 tokens/s [S5].
- DevRel talk (secondary report): "The model itself does not go through text." [T13]

**Inferred: me**
- **Input encoder.** Plausibly a USM-family conformer emitting continuous 40 ms (25 Hz) frames. Gemma 4 uses a USM-based 305M encoder, 40 ms chunks, "no vector quantization" [S33]. SpeechSSM used 25 Hz USM-v2 units with a 32k vocabulary [S30].
- **Output.** Google's speech stack uses hierarchical RVQ codecs: SoundStream→SpectroStream [S32]; a 2024 codec at "as low as 600 bits per second" [S28]; "16 tokens… per 100 ms chunk" in S2ST [S29].
  - So the backbone plausibly emits coarse ~25 Hz tokens, and a separate decoder renders 24 kHz audio.
  - Pricing fits: $0.018/min ÷ $12/M = 1,500 tokens/min = 25 tokens/s.
  - But AA's per-hour prices for 3.1 ($0.35 in / $1.38 out) imply about 32 tokens/s [T1].
- **Backbone class.** Likely Flash-class. Its predecessor was "3.1 *Flash* Live", the price matches, and the card's frontier-safety comparison is against "Gemini 3.7 Flash" [S14].
- **Reasoning.** "Interleaved reasoning with fixed latency profile" suggests short, bounded thought segments between audio segments. ET's several utterances per task imply a parallel reasoning/tool stream; one model or two is undisclosed.

## Duplex / turn-taking behavior

**Stated**
- **VAD.** "By default, the model automatically performs VAD on a continuous audio input stream." [S4]
  - Configurable: start/end sensitivity, `prefixPaddingMs`, `silenceDurationMs` (server default ~800 ms).
  - Client alternative: `activityStart`/`activityEnd` with ≥500 ms silence threshold.
- **Barge-in.** `START_OF_ACTIVITY_INTERRUPTS` cuts the response off "in the moment of the interruption"; `NO_INTERRUPTION` is the alternative. [S8]
- **After an interruption.** "The ongoing generation is canceled and discarded. Only the information already sent to the client is retained in the session history." The client must flush its playback buffer. [S4,S5]
- **Proactivity.** The model "can reject responding… ignore out of context speech or… stay silent if the user did not make a request, yet"; there is a `waitingForInput` flag [S8]. 2.5 was "trained to discern and disregard background speech" [S24].
- **Turn coverage.** A turn holds speech activity plus all video, excluding silence; 2.5 used activity only. [S8,S11]

**Measured**

| Benchmark | Model | Result |
|---|---|---|
| FDB v1 [T6] | 2.0 Flash Live (30 ms chunks, server VAD) | Pause TOR 0.255/0.310 (best tested); backchannel TOR 0.091; turn-taking TOR 0.655 at 1.30 s; interruption TOR 0.891 at 1.18 s |
| FDB v1.5 [T7] | 2.0 Flash Live | Real interruptions: resumes own turn 55%, stops after 2.20 s. Ignores backchannels (0.93) and side talk (0.99); answers background speech 70% |
| τ-Voice [T5] | `gemini-live-2.5-flash-native-audio` | Latency 1.14 s; responsiveness 69%; interrupt rate 21%; selectivity 54%; accents cost only 1 pp |
| FDB v3 [T8] | 2.5 → 3.1, tool use + disfluency | Pass@1 0.49→0.54; latency 7.26→4.25 s; turn-take 92%→78%; 3.1 a "silent worker" in 22 scenarios (tools run, no speech) |
| AA Conversational Dynamics [T1] | FDB subset | 3.1 72–74% → 3.8 Live 96.1% (ET 91.9%; Moshi 61%) |
| People Making Conversation [T9] | Hosted APIs incl. 3.1 | "Respond at almost every pause"; ~1–4 s with silence endpointers, ~8 s with semantic turn detection |

**Inferred: me**
- The model is half-duplex over a full-duplex transport. User speech during model output either cancels it on the server or is buffered into the next turn. No backchannel generation is claimed.
- 3.8's Conversational Dynamics jump probably comes from trained respond/ignore decisions and better endpointing.
- The ~800 ms default silence sets a latency floor, which fits the measured ~1.0–1.2 s TTFA.

## Tool use & thinking

**Stated**
- **Tools.** [S7,S1,S2,S3]
  - Async calls (`NON_BLOCKING`, May 2025) take `FunctionResponse` scheduling `INTERRUPT`, `WHEN_IDLE` or `SILENT`.
  - 3.1 was sync-only: the "model will not start responding until you've sent the tool response". 3.8 Live defaults to async; ET requires it.
  - The client executes tools.
  - 3.8 Live "waits for tool execution before speaking". ET "streams intermediate conversational fillers while processing", e.g. "Checking flights to Seattle…".
- **Thinking controls.** [S11,S3,S4]

  | Model | Control |
  |---|---|
  | 2.5 native audio | `thinkingBudget` |
  | 3.1 Flash Live | `thinkingLevel` (default minimal) |
  | 3.8 Live | Fixed interleaved reasoning |
  | 3.8 Live ET | low/med/high + thought summaries |

**Measured**
- Synchronous thinking costs latency: 3.1 TTFA is 0.96 s at minimal and 2.99 s at high; ET at high is 1.35 s. [T1]
- Preference follows latency: 3.1 at minimal led AA's Speech Agent Arena (Elo 1,046 at launch) despite 74.6% task success. [T3]

**Inferred: me.** Background reasoning plus fillers hides reasoning latency. It is the most transferable idea here.

## Training (stated vs inferred, incl. relevant Google research)

**Stated**
- **Pretraining (3 Pro).** Web documents, code, images, video, "audio (including speech and other audio types)". Also licensed data, "user interactions with the model" (per ToS) and synthetic data. [S18]
- **Post-training (3 Pro).** Instruction tuning, RL and human-preference data; "reinforcement learning from human and critic feedback". TPUs, JAX + Pathways. [S18]
- **3.8 Audio card.** Defers to 3 Pro for data and training. [S14]
- **Gemini 2.5.** Pretraining data "spanning over 200 languages". "Improved post-training recipes" brought "thinking, affective dialog, contextual awareness and tool use into Gemini's native audio models". General post-training: "increased compute allocated to RL", "model-based generative rewards". [S26]
- **2024 speech stack.** [S28]
  - Pretrained on "hundreds of thousands of hours of speech".
  - Fine-tuned on "a much smaller dataset" of unscripted voice-actor dialogue with disfluencies.
  - 2 min of dialogue in under 3 s on one TPU v5e; helps power Gemini Live and Astra.
- **Streaming S2ST (Nov 2025).** [S29]
  - AudioLM-style streaming encoder over the last 10 s, plus an AR decoder over SpectroStream RVQ tokens.
  - Synthetic time-synchronized data: ASR + forced alignment → MT → voice-preserving TTS → alignment masks.
  - 60 s training chunks; 2 s lookahead.

**Lineage.** No Google paper on Gemini Live or full-duplex training found.
- SoundStream → AudioLM → SoundStorm.
- AudioPaLM: text-LLM init "improves speech processing" [S31].
- SpeechSSM (ICML 2025) [S30]:
  - USM-v2 25 Hz units, 32k vocabulary; initialized from RecurrentGemma 2B/9B; 60k h LibriLight.
  - 16 TPU v5p × 100k steps × 768k tokens ≈ 77B tokens ≈ 14 epochs (my arithmetic).
  - Without LM init, N-MOS drops from 4.12 to 3.76.
- SpectroStream; SequenceLayers (a JAX library for streamable models) [S34].

**Inferred: me (low confidence)**
1. Multimodal pretraining with speech in, and since 2.5 speech out.
2. Dialogue SFT on voice-actor, TTS-rendered and product conversations, with targeted negatives such as background speech.
3. RL against model-based rewards for instruction following and function calling.
4. Thinking distilled from text Gemini.

## Evals & leaderboard numbers

**Google-published** [stated S15,S17,S19]. FNA = 2.5 Flash Native Audio, FL = 3.1 Flash Live.

| Benchmark | FNA 09-25 | FNA 12-25 (min/high) | FL min | FL high | 3.8 Live | 3.8 ET |
|---|---|---|---|---|---|---|
| CFB-Audio (Google-run, prompts via TTS) | 66.0 | –/71.5 | – | 90.8 | – | – |
| Big Bench Audio (run by AA) | – | 69/90.7 | 70.5 | 95.9 | – | 97.7 |
| Audio MultiChallenge (Scale) | – | 13.9/21.5 | 26.8 | 36.1 | – | – |
| τ³-Banking (Sierra) | – | – | – | 11.3 | – | 35.1 |

- References: Audio MultiChallenge has GPT-Realtime 1.5 at 34.7 and Qwen3-Omni-30B-A3B at 24.3. τ³-Banking has GPT-Live-1 Astra at 32.0.
- EVA-Bench (ServiceNow), read off Google's chart (±2 pp):

  | System | Accuracy (EVA-A) | Experience (EVA-X) |
  |---|---|---|
  | 3.8 Live | ~46 | ~88 |
  | ET minimal | ~50 | ~79 |
  | ET high | ~55 | ~74 |
  | GPT Realtime 2.1 | ~59 | ~74 |
  | 3.1 Flash Live | ~29 | ~59 |

**Artificial Analysis, fetched 2026-10-12** [measured T1,T2]. Index v2 (Aug 2026) is 25% each of Big Bench Audio (BBA), τ-Voice, arena preference and arena task success. τ-Voice uses a GPT-4.1 simulator with ElevenLabs voices; TTFA is measured on the BBA set.

| Model | Index | BBA | Conv. dyn. | τ-Voice | TTFA (s) | Arena Elo / task % | $/h in-audio |
|---|---|---|---|---|---|---|---|
| 3.8 Live ET (High) | 82.6 | 98 | 91.9 | 68.6 | 1.35 | 990 / 89.1 | 3.50 |
| 3.8 Live | 76.0 | 92 | 96.1 | 30.1 | 1.18 | 1083 / 93.2 | 0.84 |
| 3.1 FL High | 72.0 | 97 | 74.3 | 37.7 | 2.99 | 1063 / 71.8 | 1.75 |
| 3.1 FL Minimal | 64.2 | 71 | 72.3 | 26.2 | 0.96 | 1096 / 74.6 | 1.50 |
| GPT-Live-1 Astra (med) | 83.2 | 90 | 94.9 | 74.5 | 1.34 | – | 5.83 |
| Grok Voice Think Fast 2.0 | 81.3 | 97 | 95.1 | 56.5 | 0.70 | – | 4.80 |
| Qwen Audio 3.0 RT Plus | 69.5 | 99 | 98.4 | 54.6 | 1.54 | 775 / 77.8 | 4.42 |
| Qwen3 Omni Flash / Realtime | – | 59 / 57 | 72.7 / – | – | 4.82 / 0.88 | – | – |

- Google's September chart showed Astra at 81.5, so AA appears to have re-run it.
- 3.8 Live costs less than 3.1 at the same token prices, so it must use fewer output and thinking tokens. [inferred: me]
- Other third-party results [measured]:
  - τ-Voice (Sierra, 2.5): pass@1 45/28/20% clean and 30/30/18% realistic for retail/airline/telecom; text GPT-5 scores 85% [T5].
  - APEX-Voice: 3.8 Live and four others all score ≤25% Pass@1 on professional workflows [T10].

## Lessons for our setup (Qwen3 tokenizer + LLM + vocoder, 10K h duplex)

1. **Scale continuation data first.** Google pretrained its dialogue speech generator on 100K+ hours and only *fine-tuned* on dialogue; SpeechSSM spent ~77B tokens on 60k h [stated]. Our 10K h is ~0.9B tokens at 25 Hz, which is post-training scale. Mid-train on 100K–1M h of mixed speech (podcasts, audiobooks, conversation), then use the duplex 10K h for interaction behaviour. [inferred: me]
2. **Route knowledge through text.** Text-LLM init is critical in AudioPaLM and SpeechSSM, and even Gemini needs thinking to reason in speech: BBA 69–71% at minimal vs 91–96% at high [stated]. Add a time-aligned text "inner monologue" that leads the audio tokens, plus interleaved text–speech and text-only replay to keep Qwen3's knowledge. [inferred: me]
3. **Split talking from thinking.** ET pairs a fast speaker with background reasoning and async tools and gets 98% BBA at 1.35 s; synchronous high thinking costs 2.99 s. Build in async tool/reasoning tokens, filler/progress speech and an `IN_PROGRESS`/`IDLE` state. Train post-tool verbalization explicitly to avoid FDB v3's "silent worker". [measured T1,T8; inferred: me]
4. **Ship Google-style turn-taking first.** VAD or learned endpointing + barge-in cancel + a learned respond/stay-silent gate scores 96% on AA Conversational Dynamics. Train the gate with negatives: background speech, side talk, backchannels. [measured T1; inferred: me]
5. **Truncate on interruption.** Keep only the audio the user actually heard, as Google does [stated S4]. Log playback position in training-data construction and at inference.
6. **Plan for context limits.** Google caps sessions at 15 min of audio and compresses with a sliding window; train with truncated or summarized histories. [stated; inferred: me]
7. **Copy Google's eval suite; it is cheap.**
   - CFB-Audio (render the text prompts with TTS), BBA, Audio MultiChallenge, FDB v1/v1.5/v3, τ-Voice.
   - Track TTFA and selectivity alongside.
   - Qwen3-Omni-30B-A3B scores 24.3% on Audio MultiChallenge vs 26.8–36.1% for 3.1 Flash Live [stated S17].

## Open questions

- What are the actual audio representations? Is input continuous or discrete? What output codec, frame rate and codebook count, and does the LLM predict all levels?
- Is 3.8 Live derived from 3.7/3.8 Flash? "Based on Gemini 3 Pro" is a family-level phrase.
- How are interleaved reasoning and ET's background reasoning implemented: text thoughts interleaved with audio, or a second process?
- Is the VAD a separate model? Does the model attend to user audio while speaking? Does it ever backchannel?
- How many hours and how much compute per stage, which audio reward models, and what role for product interaction data?
- Is billing 25 tokens/s (the docs) or ~32 tokens/s (AA's prices)?
- Did the removed affective-dialog and proactive-audio toggles become default behaviour?
