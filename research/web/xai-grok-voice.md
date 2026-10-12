# xAI Grok Voice: Voice Agent API and Grok Voice Think Fast 1.0 / 2.0

Labels: **[stated]** comes from an official xAI source. **[measured]** is a third-party evaluation or leaderboard. **[inferred]** is an inference, and the tag says whose.

## Summary

- **Product.** xAI sells a realtime speech-to-speech (S2S) WebSocket API at `wss://api.x.ai/v1/realtime`. It also sells TTS, STT and custom voices.
- **Model line [stated]:**
  - Grok Voice Agent API (17 Dec 2025)
  - `grok-voice-think-fast-1.0` (23 Apr 2026)
  - `grok-voice-think-fast-2.0` (29 Jul 2026). It became the default `grok-voice-latest` alias on 5 Aug 2026.
- **Little technical disclosure.** There is no paper, parameter count, data size or compute figure. xAI states only three things [stated]:
  - it trained its own VAD, tokenizer and audio models "from scratch"
  - the model reasons "in parallel with speech"
  - TF2.0 used "extensive reinforcement learning" to push the model towards patterns of real human conversation
- **Launch scores for TF2.0 (High) [measured: Artificial Analysis].**
  - AA S2S Index 82.9: Big Bench Audio 97.2, Full Duplex Bench 95.1, τ-Voice 56.5.
  - Time to first audio (TTFA) 0.70 s.
  - About 79% of the gain over TF1.0 came from conversational dynamics: FDB rose from 77.8 to 95.1 [inferred: digitalapplied's decomposition of AA numbers].
- **Price [stated].** $0.05/min at launch. TF2.0 costs $0.08/min ($4.80/h) plus $0.004 per text input.
- **For us.**
  - Frontier S2S systems now score 97–99% on spoken reasoning (BBA) [measured: AA]. So weak semantic knowledge is not an inherent cost of the S2S approach [inferred: me].
  - xAI's largest recent quality jump came from post-training turn-taking and style behavior, not from a new base model [inferred: me].

## Sources (accessed 2026-10-12)

**xAI docs**
- https://docs.x.ai/developers/model-capabilities/audio/voice
- https://docs.x.ai/developers/model-capabilities/audio/voice-agent
- https://docs.x.ai/developers/model-capabilities/audio/custom-voices
- https://docs.x.ai/developers/models
- https://docs.x.ai/developers/pricing
- https://docs.x.ai/release-notes

**xAI posts**
- https://x.ai/news/grok-voice-agent-api (17 Dec 2025)
- https://x.ai/news/grok-voice-think-fast-1 (23 Apr 2026)
- https://x.ai/news/grok-custom-voices (30 Apr 2026)
- https://x.ai/news/grok-voice-think-fast-2 (29 Jul 2026)
- https://x.ai/news/grok-voice-agent-builder (1 Jul 2026, seen via search results only)

**Artificial Analysis (AA)**
- https://artificialanalysis.ai/articles/announcing-the-artificial-analysis-speech-to-speech-index (23 Jun 2026)
- https://artificialanalysis.ai/speech-to-speech
- AA's post on X: https://x.com/ArtificialAnlys/status/2082528987272957960

**τ-Voice**
- Paper: https://arxiv.org/abs/2603.13686
- Leaderboard: https://taubench.com/

**Third-party analysis**
- https://www.digitalapplied.com/blog/grok-voice-think-fast-2-speech-to-speech-agents-2026
- https://www.eesel.ai/blog/grok-voice-agent-builder
- https://www.testingcatalog.com/xai-launches-grok-voice-think-fast-1-0-for-voice-agents/
- https://www.analyticsvidhya.com/blog/2026/05/grok-voice-think-fast-1-0/
- https://breezeblue.ai/benchmarks/tts-latency
- Search snippets only:
  - https://aigazine.net/llms/xais-grok-voice-agent-scores-923-on-big-bench-audio-rankings--s
  - https://www.orcarouter.ai/blog/gemini-3-8-live-vs-gpt-live-1
  - https://www.marktechpost.com/2026/04/25/xai-launches-grok-voice-think-fast-1-0-topping-%CF%84-voice-bench-at-67-3-outperforming-gemini-gpt-realtime-and-more/

**Could not access**
- The MarkTechPost page returned HTTP 403.
- The AA live leaderboard is rendered by JavaScript. Its values reached me through a summarizer and partly conflict with each other (see Evals).
- The release-notes page summary had garbled years. I used the dates in the xAI posts instead.

## What it is

**Endpoints and session [stated]**
- Realtime S2S runs at `wss://api.x.ai/v1/realtime`. It is "broadly compatible" with the OpenAI Realtime API, and adds `force_message`, `resumption` (30-min reconnect cache) and `replace` (pronunciation fixes).
- Session options: `server_vad` turn detection, `reasoning.effort` (`high` default, or `none`), PCM at 8–48 kHz (24 kHz default), μ-law/A-law, Opus, and up to 100 `keyterms`.
- Tools: `web_search`, `x_search`, `file_search`, remote `mcp` and custom `function`.
- Languages: the docs say "20+", the TF1.0 post "25+", and the launch post "dozens".
- STT is now served by `grok-voice-transcribe-2.0` (18 Sep 2026).

**Voices [stated]**
- Built-in voices include Ara, Eve and Leo. Inline tags such as `[laugh]` and `[whisper]` work. Third-party reports (eesel, testingcatalog) say the Voice Agent Builder lists 80+ library voices.
- Custom voices (30 Apr 2026) need about 1 min of speech (docs recommend 90–120 s). Verification uses a live passphrase plus a speaker-embedding match. Limit 30 per team; US only, excluding Illinois.

**Deployments [stated]**
- Grok apps and Tesla vehicles.
- Starlink sales and support: 28 tools, "70% resolution rate", "20% conversion rate".
- Voice Agent Builder, a no-code product (1 Jul 2026): $0.05/min plus $0.01/min for telephony, per third-party reports.
- API audio is "never stored or used for training".

## Architecture

**What xAI states**
- The voice stack was built in-house: "training our own voice activity detection (VAD), tokenizer, and audio models from scratch" (Dec 2025 post) [stated].
- It runs on the same stack as the Grok app, Tesla and Starlink deployments [stated].
- TF1.0 has "real-time reasoning with zero added latency". The post describes a background reasoning thread that runs in parallel with the conversation, with full-duplex and native-audio claims [stated].
- TF2.0 is described this way [stated]:
  - "Reasoning in parallel with speech makes the model substantially smarter … with no impact on latency."
  - It uses 0.4× the reasoning tokens per response of TF1.0 (P50).
  - "Tool calls … usually executing before the end of the agent's first sentence."
- The phrase "listens, reasons and speaks inside a single network" appears only in digitalapplied's summary of xAI. I could not find it on an x.ai page [inferred: digitalapplied paraphrase].

**What the API shape suggests**
- The API is shaped like a turn-based loop:
  - turn ends come from server VAD
  - `response.create` re-triggers a response after tool calls
  - xAI advises clients to "complete audio playback before sending response.create"
- My reading: "full-duplex" here means streaming in both directions plus barge-in. Nothing public shows a Moshi-style model that listens and speaks at every frame [inferred: me].
- My guess is that "reasoning while speaking" is a thinker–talker split: a fast speech head starts talking while a text reasoning thread continues and feeds tool results back. This cannot be verified [inferred: me].

**Not disclosed:** model size, tokenizer frame rate and codebooks, whether the reasoning thread is the same network, and the sample rate of the internal codec.

## Training data & compute

- **Nothing quantitative** is disclosed: no hours, tokens, parameters or GPU counts [stated by omission].
- **TF1.0 [stated].** Built in "tight collaboration with partners like Starlink". The post says it was "battle-tested" on telephony audio, background noise, heavy accents and frequent interruptions.
- **"Trained on telephonic data".** One blog paraphrases the TF1.0 model as "trained using actual telephonic data". It gives no primary link [inferred: analyticsvidhya; unverified].
- **TF2.0 [stated].** It was trained to:
  - speak in shorter sentences
  - ask one question at a time
  - use fewer filler words
  - spend fewer reasoning tokens

## Post-training / alignment

**TF2.0 RL.** xAI says it used "extensive reinforcement learning to push the model towards patterns we see in real human conversation" [stated]. A/B tests on Starlink showed a "significant increase in sales conversion rate and support containment rate". No numbers are given, and the post does not say whether the A/B results fed back into training [stated].

**Where the gain came from.** Measured by AA, nearly all of TF2.0's gain is in conversational dynamics. That is consistent with RL that targets turn-taking, brevity and interruption handling [inferred: me, from AA sub-scores].

## Evals & numbers

**AA S2S Index [measured: AA].** The index weights three benchmarks equally:
- **Big Bench Audio (BBA):** 1,000 spoken Big-Bench-Hard reasoning questions.
- **Full Duplex Bench (FDB):** pause handling, turn taking, interruption and backchannels.
- **τ-Voice:** airline, retail and telecom tasks.

| Model (date) | Index | BBA | FDB | τ-Voice | TTFA | $/h input audio |
|---|---|---|---|---|---|---|
| TF1.0 (index launch, 23 Jun 2026) | 75.7 | 97.1 | 77.8 | 52.1 | 1.25 s | $3.00 |
| TF2.0 High (29 Jul 2026) | 82.9 (#2) | 97.2 | 95.1 | 56.5 (#1) | 0.70 s | $4.80 |
| Qwen Audio 3.0 Realtime Plus (same post) | 84.1 (#1) | 99.2 | – | – | ~4 s (digitalapplied) | – |
| GPT-Realtime-2.1 High (same post) | 79.1 | – | – | – | 1.21 s | – |

**Comparison points from the AA index launch [measured: AA]**
- GPT-Realtime-2 (High): 77.2.
- Gemini 3.1 Flash Live (High): 69.5, with TTFA 2.98 s.
- Deepslate Opal had the fastest TTFA at 0.44 s.

**Later rankings.** The ranking has moved since TF2.0 launched [measured: AA, via summarizer and secondary reports]:
- By mid-September 2026, Gemini 3.8 Live Extended Thinking reached 82.6 and GPT-Live-1 (Astra) 81.5.
- TF2.0 High now shows 81.3, so the index appears to have been re-scaled.
- The live page fetched on 2026-10-12 showed GPT-Live-1 (Astra, medium) at 83.2.
- On that page, some rows did not match their own sub-scores (e.g., Qwen Audio 3.0 Realtime Plus at 69.5). Treat this snapshot as approximate.

**τ-Voice (Sierra).**
- **Paper [measured: τ-Voice paper].** 278 tasks. Voice agents score 31–51% pass@1 on clean audio and 26–38% with noise and accents, against 85% for GPT-5 text. That means voice agents keep only 30–45% of text capability, and 79–90% of failures come from agent behavior.
- **xAI's TF1.0 claim [stated by xAI; numbers via MarkTechPost snippet].** TF1.0 scored 67.3% and ranked #1 at launch, ahead of Gemini 3.1 Flash Live (43.8%) and GPT Realtime 1.5 (35.3%).
- **taubench.com τ³-voice leaderboard today [measured].** gpt-live-1 81.7%, Pine Voice Preview 75.4%, grok-voice-think-fast-1.0 67.3%. No TF2.0 entry is listed.
- **Harness gap [inferred: me].** AA's own τ-Voice runs give lower numbers: 52.1 for TF1.0 and 56.5 for TF2.0. The two harnesses are not comparable.

**Big Bench Audio at the Dec 2025 launch.** The model ranked #1, and a third party reports 92.3% [measured: AA via aigazine snippet].

**Transcription (TF2.0) [stated].**
- WER is 1.5–2.0× better than Deepgram Nova 3 and ElevenLabs Scribe v2, 1.4× better than TF1.0, and about 10× better in noise.
- The test set was "thousands of short phrases in 24 languages".
- No absolute WER is published. digitalapplied notes the claim is unfalsifiable without independent replication [inferred: digitalapplied].

**Latency, third-party.**
- LiveKit's testing found responses under 700 ms [measured: LiveKit via eesel, unverified].
- BreezeBlue, a competitor, measured xAI TTS (not S2S) at TTFA p50 289.7 ms and p95 351.3 ms over a persistent WebSocket [measured: BreezeBlue].

**Human preference [stated].** Blind evaluations preferred Grok over the OpenAI Realtime API on pronunciation, accent and prosody. No numbers are given.

**Prices [stated, docs as of 2026-10-12]**
- S2S: $0.08/min.
- STT: $0.10/h (REST), $0.20/h (streaming).
- TTS: $15.00 per 1M characters. April 2026 launch coverage reported $4.20 per 1M, so the price appears to have risen.
- Tools: web search $5 per 1k calls, Collections search $2.50 per 1k calls.

## Lessons for our setup

1. **Benchmark against the triad.** Use BBA, FDB and τ-Voice, or local proxies for each. Frontier models reach BBA ≈ 97–99%, so speech-domain reasoning can reach text level. A spoken BBA/VoiceBench check is a cheap diagnostic for our weak-knowledge problem [inferred: me].
2. **Turn-taking post-training is high-leverage.** TF1.0 and TF2.0 have nearly the same BBA, yet FDB rose 17 points after "extensive RL" toward human conversational patterns. Our post-training plan should budget for an RL or preference stage on duplex behavior. Targets are timing, brevity, single questions and barge-in, run after mid-training fixes continuation quality [inferred: me].
3. **Decouple "thinking" from "talking".** Reasoning in parallel with speech, and calling tools before the first sentence ends, both point to a text reasoning channel running beside the speech stream. For a Qwen3-based duplex model with weak spoken knowledge, an inner-monologue or text channel is the main architectural lever. Nemotron VoiceChat and Moshi's inner monologue are open examples [inferred: me].
4. **Optimize for telephony robustness.** Production value came from noise, accents, 8 kHz μ-law audio and interruptions. τ-Voice drops 31–51% to 26–38% under realistic audio. Our 10K h should be augmented with codec, noise and accent variation [inferred: me].
5. **Style targets for SFT and RL.** xAI explicitly trained for shorter sentences, one question per turn and fewer fillers [stated]. These are measurable targets for our dialogue data curation [inferred: me].
6. **Cost envelope.** Competitive API pricing is $3–4.80 per hour of audio [stated: xAI pricing]. That bounds the serving cost we can afford. The Nari Labs note shows open-model serving at about $0.07 GPU-$ per hour [stated: Nari; comparison inferred: me].

## Open questions

- Is Grok Voice one end-to-end network or a thinker–talker pair? What are the tokenizer's frame rate and codebook design, and the model size?
- Is the system truly full-duplex (backchannels while listening), or VAD-gated with barge-in? The API's `server_vad` semantics suggest the latter.
- What did the TF2.0 RL reward: human preference, task success (e.g., Starlink containment) or rule-based brevity and timing?
- How many hours of audio, and how much telephony data, went into training? Was Starlink call data used? (API data is said to be excluded.)
- Why does AA's τ-Voice (52–57%) differ from taubench's (67.3%) for the same TF1.0 model? Is it user-simulator settings, noise conditions or the task set?
