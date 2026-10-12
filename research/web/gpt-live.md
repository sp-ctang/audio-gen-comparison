# OpenAI GPT-Live (GPT-Live-1 / GPT-Live-1 mini)

Labels: **[stated]** = official OpenAI source (docs, posts, system card, OpenAI staff on X); **[measured]** = third-party eval or leaderboard; **[inferred: who]** = an inference, with whose it is. "Desh" = Desh Raj, *Dissecting GPT-Live* (2026-09-11). "me" = this note's author.

## Summary

- **Two-tier design.** A full-duplex speech-to-speech "frontend" (gpt-live-1) handles listening, speaking, backchannels, barge-in and the decision to delegate. A text "backend" (GPT-5.5 in ChatGPT; in the API, any Responses model or your own agent) does reasoning, search and tools. Results come back as text that the frontend speaks. [stated]
- **Delegation is what makes it smart on agentic tasks.** On τ-Voice with standard user simulators, OpenAI's single-model voice systems score 30–46%. GPT-Live with a GPT-6 Astra backend scores 74.5–86.2%, depending on the harness and user simulator. GPT-5 in text mode scores 85%. [measured/stated]
- **Frontend size mostly affects naturalness, not knowledge.** With the same GPT-5.5 Instant backend, the mini frontend nearly matches the full one on GPQA (74.9% vs 76.5%) and τ³-Voice Telecom (39.5% vs 37.3%). Human-rated flow drops from 4.96 to 4.33 (on 1–7). [stated; interpretation inferred: me]
- **Delegation has costs.** τ³-Voice latency is about 2.5 s for GPT-Live vs 1.15–2.0 s for most single-model systems. On AA's Big Bench Audio, GPT-Live scores 89–90% vs 96.6% for GPT-Realtime-2 (High). [measured]
- **Training: almost nothing is disclosed.** OpenAI states only that it did "specific training" for controllability and instruction following, used the "same infrastructure" as for its flagship models for safety training, and that the model needed "research advances". There are no data, compute, codec or frame-rate figures. [stated]

## Sources (all accessed 2026-10-12)

**OpenAI posts**
- API launch, 2026-09-10: https://openai.com/index/introducing-gpt-live-1-in-the-api/. WebFetch got HTTP 403, so I read it in a browser. Chart values come from the page's embedded Vega specs.
- ChatGPT launch, 2026-07-08: https://openai.com/index/introducing-gpt-live/
- Engineering post by Uberti & Malkani, 2026-08-03: https://openai.com/index/continuous-voice-interaction-with-gpt-live/
- System card, corrected 2026-08-04: https://deploymentsafety.openai.com/gpt-live

**OpenAI docs**
- https://developers.openai.com/api/docs/guides/live-delegation (both modes read)
- https://developers.openai.com/api/docs/guides/live
- https://developers.openai.com/api/docs/guides/live-prompting (including the appendix)
- https://developers.openai.com/api/docs/models/gpt-live-1
- Under the same `…/guides/` path: `live-conversations`, `live-migration`, `voice-websockets?api=live`, `voice-server-controls?api=live`, `voice-latency-cost?api=live`
- WebSocket reference: https://developers.openai.com/api/reference/resources/live/primary-websocket

**Evals and predecessors**
- Cookbook: https://developers.openai.com/cookbook/examples/audio/voice_agent_evaluation
- Harness: https://github.com/openai/openai-cookbook/tree/main/examples/audio/duplex_voice_agent_evaluation (README only; code not read)
- Predecessors: https://openai.com/index/introducing-gpt-realtime/ and https://developers.openai.com/api/docs/models/gpt-realtime-2

**OpenAI staff on X (read in a browser)**
- https://x.com/kundan2510/status/2098149959073935711
- https://x.com/kundan2510/status/2091713860528984451
- https://x.com/juberti/status/2098105712987791624
- https://x.com/daidaijiayu/status/2098561342387179748

**Third-party evals**
- https://artificialanalysis.ai/speech-to-speech (values read from page data)
- https://taubench.com/leaderboard?benchmark=voice (Standard, Custom and Banking views)
- τ-Voice paper: https://arxiv.org/abs/2603.13686
- https://github.com/DanielLin94144/Full-Duplex-Bench

**Analyses**
- https://desh2608.github.io/2026-09-11-dissecting-gpt-live/ (5 margin guesses extracted from the HTML)
- https://arxiv.org/abs/2510.07497
- https://blog.bytebytego.com/p/how-openai-built-gpt-live
- https://www.infoq.com/news/2026/09/openai-gpt-live/
- https://www.deeplearning.ai/the-batch/one-model-talks-another-one-thinks
- https://docs.pipecat.ai/api-reference/server/services/s2s/openai-live

**Not accessed or not found**
- Desh's Google Drive notes and Kwindla Kramer's posts. Kwindla is quoted via Desh only.
- Desh quotes the prompting guide's line "the live model has a small context window". It is gone from the current docs, which say 128K, so the docs were probably revised.
- The web-search budget ran out near the end.

## Product & API facts [stated]

**Releases**
- ChatGPT launch, 2026-07-08:
  - GPT-Live-1 is the default for paid ChatGPT users; GPT-Live-1 mini is the default for free users.
  - Backend at launch: GPT-5.5. Instant uses GPT-5.5 Instant; Medium and High use GPT-5.5 Thinking.
  - SynthID audio watermarking was added 2026-07-31.
- API launch, 2026-09-10: `gpt-live-1` is the only snapshot.
  - Knowledge cutoff: **2025-07-31**.
  - Audio and text in and out; no images.
  - Function calling: yes. Structured outputs and fine-tuning: no.

**Pricing and limits**
- **$0.05/min**, billed per second, including silence and time while the backend works.
- Backend usage is billed separately.
- Limits are on concurrent sessions: 50 / 300 / 500 for the Build / Launch / Grow tiers.

**Connections and audio**
- Transports: WebRTC, WebSocket or SIP, plus an optional "sideband" WebSocket for server control.
- Audio formats: PCM16 at 24 kHz or 16 kHz, or G.711 at 8 kHz.

**Context limits**
- Instructions: up to 16,384 tokens.
- Startup history: up to 128 messages and 8,192 tokens.
- Each append: up to 500 tokens.
- Context window: **128K tokens, "including … audio tokens that don't appear in the transcript"**.
- Above 90% usage, it "starts a replacement voice engine" seeded with the instructions and up to 8,192 tokens of history or summary.
- `session.usage.updated` reports `usage.seconds` and `context_window.usage_ratio`.

**Recordings and forking**
- Stored sessions can be forked for evals.
- Recordings download as stereo WAV, with the user on the left channel and the assistant on the right.

**Voices**
- 12 new voices, each tagged "Natural" or "Generated".
- Realtime voices are also accepted: the default is `marin`, and the cookbook uses `cedar`.
- ChatGPT's nine voices were "remastered".

**Partner claim**
- Speak: about 80% fewer interruptions during thinking pauses.

## Architecture

**Stated**
- **One model in a continuous loop.** It is "a single model that reasons over incoming and outgoing audio together". It "continuously processes input while generating output", making "interaction decisions many times per second: whether to speak, continue listening, pause, interrupt, or invoke a tool". It is OpenAI's "third-generation voice system", which "removes the turn detector from the audio path".
- **Native text and turn detection.** It "natively provides ASR transcripts and response text", has "strong alphanumeric understanding" and supports "keyword biasing". It also "natively supports turn detection", yet the API has **no** turn, VAD or end-of-response events, only transcript deltas with `start_ms`/`end_ms`.
- **Output audio.** Output deltas have no timestamps. "The server can finish generating while the player still has a second of audio queued."
- **Serving.**
  - Stateful inference with a per-session KV cache.
  - Delegation and tools sit behind an async RPC boundary, off the live media path.
  - Compaction is a "managed transition" to a replacement instance prefilled in parallel.
  - InfoQ reports, paraphrasing, that each session reserves capacity on a dedicated instance.
- **Backend inputs are text only.** "Delegation does not automatically send audio to your backend." Images go to a vision-capable backend; the frontend "does not accept images directly".

**Inferred**
- **Streams.** [inferred: Desh] The frontend models one interleaved sequence: user audio, streaming user ASR text, "thinking", agent text before agent audio, turn tokens, and delegation or injected blocks. Desh notes that predicting user and agent text is what let his Moshi variant "match S2S intelligence with T2T".
  - [inferred: me] Nothing states that the frontend generates its own chain of thought. "Thinking" may only be an injected-context slot.
- **Chunking.** [inferred: Desh, margin guess] Likely 80 ms tokens, with user audio prefilled in chunks of 2–5 frames (160–400 ms). This would explain the 0.8 s Full-Duplex-Bench v1 latency, vs Moshi 0.25 s and PersonaPlex under 0.2 s.
  - [inferred: me] "Many times per second" implies a step of 0.5 s or less.
  - Appends are acknowledged "when the session timeline reaches the estimated end of the added context". So injected text appears to be scheduled onto the model's timeline and to use up time there.
- **Vocoder.** [inferred: Desh] The vocoder was retrained because the voice set is "disjoint" from Realtime's.
  - Correction [stated docs]: Realtime voices *are* supported, so the sets are not disjoint. The "remastered" voices still suggest re-rendering.
- **Backbone.** [inferred: me] GPT-Live's 2025-07-31 cutoff matches no candidate backbone:

  | Model | Knowledge cutoff |
  |---|---|
  | GPT-Realtime-2 | 2024-09-30 |
  | GPT-Live-1 | 2025-07-31 |
  | GPT-5.4 | 2025-08-31 |
  | GPT-5.5 | 2025-12-01 |
  | GPT-5.6 Terra | 2026-02-16 |
  | GPT-6 Sol | 2026-04-20 |
  | GPT-6 Astra | 2026-04-30 |

  So it is a distinct snapshot, newer than Realtime-2. This is weak evidence against Desh's "Terra backbone" guess.

**Size**
- [stated] The system card says both GPT-Live models are "less capable than GPT-5.5 Thinking across several intelligence evaluations". This is in its Preparedness section, which assesses the models without delegation.
- [inferred: Desh] Assuming 12.5 Hz frames, 5-frame chunks and 20% text overhead, the frontend handles about 1,800 tokens/min, so $0.05 ≈ $27.8 per M tokens. At list prices, that token mix would cost:

  | Model | $/min |
  |---|---|
  | GPT-6 Astra | 0.060 |
  | GPT-5.5 | 0.035 |
  | GPT-5.4 | 0.018 |
  | Terra | 0.014 |

  Real-time serving overhead (no prefix caching, tiny prefills, no queueing) led him to guess a Terra- or GPT-5.4-sized model.
- [inferred: me] Desh's "GPT Realtime 2 (audio) $0.028" bar equals Realtime-2's *text* prices ($4/$24 per M). At its *audio* prices ($32/$64 per M), the same mix costs about $0.091/min. Because pricing is per reserved session, not per FLOP, size estimates are loose.

## Full-duplex behavior [stated]

- **What it does.** It backchannels ("mhmm", "got it"), waits through thinking pauses, and stays quiet when asked. It is "better at focusing on your voice" amid traffic or side conversations.
- **Prompt template.** The docs say to keep three headings:
  - "Backchannel policy". "Moderate" is "a prompting instruction, not a numerical frequency setting". A rule forbidding all overlap "can suppress them".
  - "Interruption policy": "Stop speaking when the user interrupts. Listen…"
  - "Delegation policy".
- **Optional prompt controls:** ignoring noise and side speech, a "selected requests only" listening mode, response length, IPA pronunciation, and an interpreter mode.
- **Interrupting is not cancelling.** "Stop talking" means yield the floor. "Cancel my booking" goes to the backend as a task. Interrupting leaves backend work running.
- **Playback is the app's job.** "Discarded speech remains in the model's conversation context." An appended instruction can interrupt speech, and moderation can cut audio.

## Delegation

**Mechanics [stated]**
1. **The frontend decides.** It uses the prompt's "Backend tools" capability list, "to decide which requests to hand off". It is told: "Delegate before giving an answer that depends on backend work. Do not guess the result while waiting."
2. **The event.** It emits `session.delegation.created` with `{id, target, offset_ms}`. In client mode this is "metadata, not task text". The app rebuilds the task from transcript deltas, which "may contain mistakes".
3. **Responses mode.** OpenAI runs the configured backend (start with `gpt-6-luna`, try `gpt-6-sol` for complex tasks) over persistent connections with prepared configuration. The engineering post adds that in ChatGPT the backend is prefilled at session start and uses session affinity and prompt caching.
4. **Results come back via three appends:**
   - `commentary.append`: spoken. The model "is trained to paraphrase the text".
   - `thinking.append`: silent facts.
   - `instructions.append`: steering; can interrupt speech.
5. **Concurrency.** Multiple and updated delegations are allowed, stale results are discarded, and apps can start speculative work from transcript fragments.

**Observed timing [stated: cookbook; one run, "not a performance benchmark"]**
| Time (s) | Event |
|---|---|
| 8.4 | Delegation |
| 9.8 | Filler: "let me check that for you" |
| 19.6 | Second delegation |
| 20.9 | Filler: "Checking that now." |
| 24.2 | Result spoken, about 4.6 s after the second delegation |

- Backend: gpt-5.6-terra, 2,374 input / 105 output tokens.
- Response latency was 1.51 s, measured through a 400 ms playback buffer.

**How it might have been trained [inferred]**
- **Desh.** SFT with sandboxed backends and scripted responses, then RL with real or mocked backends, "which is expensive". His key open question: how were async latency and bad backend responses simulated at RL scale?
- **Me: templated prompts.** The fixed policy headings and "trained to paraphrase" suggest post-training on templated prompts, with injected blocks placed at random points on the timeline.
- **Me: simulators and rewards.**
  - OpenAI has full-duplex user simulators. The RUN harness pairs a GPT-Live caller with a gpt-5.6-luna backend. OpenAI's τ³ runs used "a customized user model, powered by our latest reasoning models".
  - The cookbook metrics read like a reward spec: task completion, delegation accuracy, silence during delegation, latency, interruption rate.
- **Open analogs:**
  - StepAudio 3 Realtime: async delegation trained on targeted voice-agent dialogues plus real agent trajectories plus negatives. See `../papers/stepaudio3-realtime.md`.
  - MoshiRAG: async retrieval.
  - NemotronLabs VoiceChat. See `../papers/nemotronlabs-voicechat.md`.

## Training: stated vs inferred

**Stated**
- **System card.**
  - Data: "diverse datasets", including public web, partner data, and "information that our users or human trainers and researchers provide".
  - "We trained these new models to respond safely, using the same infrastructure we rely on for training our flagship models."
  - "Age-appropriate behavior [was] trained directly into the model."
  - The models were "enabled by research advances".
- **Kundan Kumar (OpenAI GPT-Live research lead):**
  - "we did specific training to further improve the controllability and instruction following of the model."
  - The full-duplex bet "seemed deeply improbable".
- **Lineage.**
  - gpt-realtime (Aug 2025) was "trained … in close collaboration with customers". It scored 82.8% on Big Bench Audio, vs 65.6% for its predecessor.
  - GPT-Realtime-2 (May 2026) added reasoning-effort levels.
- **Not disclosed:** data hours, tokens, compute, stages, codec, frame rate, RL method.

**Inferred**
- [ByteByteGo, author's explanation, not OpenAI] Silence is just another token, and timing is learned implicitly from data.
- [me] OpenAI first made the single S2S model smart, reaching 96.6% on Big Bench Audio (AA). τ-Voice still stayed at 30–46%; only delegation produced 74–86%. GPT-Live's post-training thus targets interaction and delegation policy, not knowledge in the voice model.

## Evals & leaderboard numbers

**OpenAI, API launch post** [stated]
| Benchmark | GPT-Live-1 | RT-2.1 | RT-2 |
|---|---|---|---|
| τ³-Voice (Astra-medium backend; Sierra run) | 86.2% | 45.7% | 42.4% |
| τ³ Banking-knowledge (97 tasks) | 32.0% | 12.4% | 10.3% |
| AA conversational dynamics | 97.3% | 95.7% | 95.3% |
| FDB v1.5 interactivity | 80.1% | 45.4% | 47.8% |
| FDB v1 turn-taking latency | 0.798 s | 1.41 s | 1.63 s |
| FDB v3 tool calling (Terra-low backend) | 87.0% | 60.0% | 58.0% |
| FDB v3 response quality | 90.0% | 88.0% | 81.0% |

**OpenAI, ChatGPT launch post** (vs Advanced Voice Mode) [stated]
- Pairwise preference over AVM: GPT-Live-1 75.7%; mini 69.2%.
- 1–7 human ratings:

  | Rating | GPT-Live-1 | mini | AVM |
  |---|---|---|---|
  | Flow | 4.96 | 4.33 | 3.80 |
  | Pleasantness | 5.19 | 4.47 | 3.82 |

- Benchmarks by setting:

  | Benchmark | AVM | mini | Instant | Medium | High |
  |---|---|---|---|---|---|
  | GPQA | 45.3% | 74.9% | 76.5% | 81.7% | 84.2% |
  | BrowseComp | 0.7% | 31.6% | 35.1% | 60.6% | 75.2% |
  | τ³-Voice Telecom (internal; P50 duration) | 29.5% (386 s) | 39.5% (291 s) | 37.3% (231 s) | 59.6% (295 s) | 63.4% (386 s) |

**Third-party** [measured]
- **τ³-Voice (Sierra):**
  - gpt-live-1 (Astra medium): **81.7%** with the default user simulator (#1), and 86.2% with a gpt-5.5-xhigh simulator. Responsiveness 95.5%, latency 2.54 s, interrupts 22.0%.
  - Others: Pine Voice 75.4%; Grok Think Fast 1.0 67.3%; gpt-realtime-2 xhigh 42.4%; cascaded baseline 31.2% at 4.24 s.
  - Banking: Astra high 37.1%; Astra medium 32.0%; gpt-realtime-2 10.3%.
- **Artificial Analysis:**

  | Metric | Astra-medium | Sol-low |
  |---|---|---|
  | S2S Index | 83.2 (#1) | 80.1 |
  | τ-Voice (1 trial) | 74.5% (airline 76.7, retail 72.5, telecom 74.3) | 59.3% |
  | Big Bench Audio | 90.1% | 89.0% |
  | Conversational dynamics | 94.9% | 97.3% |
  | Time to first audio | 1.34 s | 1.24 s |

  - Desh's 2026-09-11 snapshot showed 67.9% on τ-Voice. OpenAI said AA's harness "has some bugs on handling bidi full duplex".
  - Comparisons on Big Bench Audio: GPT-Realtime-2 High 96.6%, Grok Think Fast 2.0 97.2%.
  - Grok's time to first audio is 0.70 s.
  - Conversational dynamics: Moshi 61.0%, PersonaPlex 91.0%.
- **τ-Voice paper:** 278 tasks. GPT-5 in text scores 85%; voice agents score 31–51% on clean audio.

## Lessons for our setup (Qwen3 tokenizer, Qwen3 LLM, Qwen3 vocoder; ~10K h duplex)

1. **Delegation fixes weak semantic knowledge, not weak audio continuation.** A small duplex frontend plus a strong text backend gets near-backend task scores; mini ≈ full on GPQA and τ³. Naturalness and voice quality remain the frontend's job, and that is where mini loses. [inferred: me]
2. **The frontend still needs its own semantics** for direct replies, deciding when to delegate, and accurate transcripts, since the backend sees only text.
   - Add time-aligned user-ASR and agent-text streams.
   - In arXiv 2510.07497, streaming ASR plus chain-of-thought on Moshi's text channel gave 2.4× spoken-reasoning accuracy (ARC-E 30.2% → 77.7%), using about 690K synthetic TTS samples.
   - Measure your own S2S-vs-T2T gap against the Qwen3 backbone. [paper stated; applicability inferred: me]
3. **Speaking injected text is text-conditioned speech continuation**, essentially TTS within the conversation. It is also a useful auxiliary task for the audio-continuation weakness. [inferred: me]
4. **SFT recipe implied by the API.** [inferred: me]
   - Add delegate tokens, plus `<instructions>`, `<thinking>` and `<commentary>` blocks of 500 tokens or fewer at random offsets.
   - Simulate backend latency of about 1–10 s.
   - Train fillers while waiting, then paraphrased readout of the result.
   - Include negatives: no delegation for greetings or clarifications, no guessed results, and re-delegation on corrections.
   - Use a fixed prompt schema for cheap controllability.
5. **Evals and RL environment.** [inferred: me]
   - Clone CRAWL/WALK/RUN and the cookbook metrics. Build the user simulator from your duplex model plus a small LLM.
   - Track FDB v1.5 interactivity rather than the saturated v1, and add τ-Voice and FDB v3.
6. **Engineering.** [stated → applicable]
   - Pre-warm and prefill the backend.
   - Keep session affinity.
   - Keep a resident KV cache.
   - Compact context by swapping to a background replica.
   - Separate "stop talking" from "cancel task".
7. **Budget for latency.** Answer easy questions in the frontend and delegate hard ones; GPT-Live pays about 2.5 s on τ³. [inferred: me]

## Open questions

- **Frame rate and chunk size.** Anyone with API access can measure the slope of `context_window.usage_ratio` × 128K over a silent minute, which gives the token rate. [inferred: me]
- **Does the frontend think?** Is the "thinking" stream the model's own reasoning, or only injected context?
- **Delegation training.** SFT or RL? How were latency and failures simulated? How much duplex data? What base has the 2025-07-31 cutoff?
- **Audio stack.** Did the encoder, codec or vocoder change? The voices were "remastered".
- **Big Bench Audio drop.** Why is it lower than Realtime-2? Candidates: under-delegation, ASR errors in the delegated transcript, or a harness artifact.
- **Turn detection.** OpenAI claims it is native, but the API exposes no turn events. Is there an internal turn token?
- **Model sizes**, and what share of $0.05/min pays for reserved real-time capacity.
