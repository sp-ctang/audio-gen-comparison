# StepAudio 3 Realtime Technical Report

| Field | Value |
|---|---|
| Org | StepFun (StepFun-Audio Team) |
| Date / arXiv | 12 Sep 2026 · arXiv:2609.14005v1 (p.1) |
| Category | full-duplex S2S, with streaming reasoning and a tool-using voice agent. A sibling ASR model (StepAudio 3 ASR Max) shares pretraining and midtraining (p.5). |
| Backbone | MoE with ~196B total and ~11B active parameters per token. The language backbone is "based on Step 3.7 Flash" (p.3); the exact init checkpoint is not stated. |
| Audio input | Qwen3-Omni AuT encoder + adapter; text enters through a separate path (pp.3–4). **Both** the user stream and the model's own output-audio stream go through the encoder (p.3; Fig.5A p.8). The frame rate used is not stated; AuT natively emits 12.5 Hz (80 ms) continuous frames (see `qwen3-omni.md`). |
| Audio output | A separate streaming "generator" whose audio "returns to the model audio stream" (p.4). The duplex LLM emits one state **or text** token per 320 ms block (p.8). Fig.5A labels the LLM input "Serialized S2T" and draws a separate "Step-Audio model, streaming speech output". Token type, rate and codebooks: **not reported**. |
| Waveform decoder | Not reported. The generator gives incremental speech "with context-appropriate tone and rhythm", including pauses and hesitation (p.4). |
| Duplex mechanism | One model with two audio streams (user and model), time-interleaved in 320 ms blocks. Each block is followed by a state or text token, e.g. `listening: no_voice`, `listening: user_voice`, `speaking transition` (Fig.5B, p.8). |
| Open weights / code | Project page linked (p.1); weight and code release not stated. |
| Relevance to us | **High.** The best duplex system in this batch, and it states where duplex data sits: >10K h of synthetic duplex data in midtraining, after 1.2T pretraining tokens. It gives almost no hyperparameters or compute. |

## TL;DR

- **Architecture.** A 196B/11B-active MoE (Step 3.7 Flash) with the Qwen3-Omni AuT encoder. User and model audio are interleaved in 320 ms blocks, each followed by one state or text token. Speech comes from a separate generator, and that audio is re-encoded as the model stream (pp.3–4, 8).
- **Foundation.**
  - Pretraining (alignment → mixed → cooldown): fixed **32K** sequence length, **1.2T** tokens, with an *increased* pure-text share (p.4).
  - Midtraining: **128K** context, a larger share of audio-understanding and agent data, **>10,000 h of synthetic full-duplex data**, supervision for streaming ASR, VAD and utterance completeness, and text data (pp.5, 9).
- **Post-training:**
  - curated SFT: 100K quality-controlled examples beat 2M random ones (p.8);
  - Adaptive-Thinking supervision (p.13);
  - voice-agent dialogues and real agent trajectories (p.18);
  - a **3:1:1:1 weight merge of 4 specialised teachers** (p.16).

  **No RL or preference stage is described.**
- **Think-While-Speaking:** two concurrent calls run the same model. A Formulation Brain writes private reasoning; an Articulation Brain speaks short segments conditioned on the reasoning so far, and speaks first by default. MTP3 with Medusa-style acceptance makes private reasoning 2.05× faster (pp.13–16).
- **Results:**
  - AA Full-Duplex Bench **98.9** (turn-taking 100.0, interruption 99.0);
  - StepAudioChat 73.0 (reasoning mode) / 70.4 (realtime);
  - τ-Voice 56.0; MMSU 90.6; HMMT-Feb-2026 86.8.
- **Weak spots:** multi-turn constraint following (AudioMultiChallenge 49.3; realtime instruction following 54.1) and retail tool use (37.7) (pp.8, 18, 21).

**Lineage (Step-Audio 2 → StepAudio 3).**

*Step-Audio 2* (see `step-audio-2.md`) was half-duplex:
- an unnamed text-LLM init, smaller than the 130B Step-Audio;
- an in-house frozen 25 Hz encoder plus a 2× adaptor;
- CosyVoice 2 audio tokens interleaved with text, decoded by flow matching + HiFi-GAN;
- external VAD;
- 1.356T pretraining tokens at ~50% text, then SFT and PPO/GRPO.

Step-Audio-R1/R1.5 and StepAudio 2.5 sit in between ([14–16], not in this batch); the data-curation pipeline comes from 2.5 (p.4). Here is what changes in StepAudio 3 Realtime, with the reason where one is stated:

| Area | Step-Audio 2 | StepAudio 3 Realtime | Stated reason |
|---|---|---|---|
| Backbone | unnamed text LLM | 196B MoE on Step 3.7 Flash | — |
| Encoder | in-house | Qwen3-Omni AuT | — |
| Turn-taking | external VAD (SA2 p.7) | native dual-stream state tokens | pauses, backchannels, interruptions and background speech need acoustic + dialogue context (p.8) |
| Reasoning | PPO-trained to keep traces short "for real-time audio engagement" (SA2 pp.8–9) | Think-While-Speaking in parallel with speech, plus Adaptive Thinking and MTP | — |
| Tools | single calls trained from ~1K synthetic scripts per tool | asynchronous backend tasks trained with real multi-step trajectories (p.18) | — |
| Text retention | ~50% text | an "increased" text share, text in duplex midtraining, and merging with a general-text teacher | — |
| Context | 8K → 16K | 32K, then 128K | — |
| Post-training | SFT + RL | teachers + merging, no RL described | — |

*StepAudio 3 Gen*, released the same week, uses a different output stack from SA2's CosyVoice 2 + flow matching: a 12.5 Hz 16×2048 RVQ tokenizer, an LLM-predicted codebook 0, a depth predictor and a causal Vocos decoder. This report never says whether its "generator" is that model.

## Architecture

- **The loop.** The paper describes the system as listen → converse → think → act. The four functions share one conversational context: acoustic and lexical evidence, history, current turn, reasoning progress and tool status (p.2, Fig.2).
  - *Deep Perception*: ASR and audio understanding.
  - *Seamless Duplex*: listen / speak / yield.
  - *Think-While-Speaking*: Adaptive Thinking + Medusa MTP.
  - *Streaming Action*: intent → tool call → feedback.

  These run concurrently, and new speech or tool results can change ongoing decisions.
- **Dual streams ("dual-stream recurrent architecture").** A streaming encoder takes the user audio and the model's own output audio (p.3, Fig.5A). The model therefore perceives its own speech acoustically, not as its own output tokens (p.4). The schematic shows 2 user + 2 model frames per block feeding "Audio LLM and full-duplex interaction states", with one state token out per block.
- **Interleaving.** 320 ms audio blocks, each followed by a state or text token (p.8). Fig.5B draws each block as `S a a a a E` plus a state token. The decision cadence is 3.125 tokens/s (derived: 1/0.32 s). The full state vocabulary is not given.
- **Floor control** (p.9). The model decides whether to keep listening, start responding, keep speaking or yield:
  - *pause vs turn end*: acoustic timing plus semantic completeness;
  - *backchannel vs interruption*: judged relative to the ongoing response ("right" as acknowledgment vs as a preface to a correction);
  - *background speech*: rejected using dialogue history;
  - *overlap*: interpreted using its own ongoing speech, i.e. "what the user is currently hearing".
- **Think-While-Speaking** (pp.13–14, Fig.7). It builds on Mind-Paced Speaking [17] and runs two concurrent calls of the *same* model:
  - **Formulation Brain**: user context → private think stream.
  - **Articulation Brain**: previous think, current think and previous response → short response segments.

  How they coordinate:
  - Playback-aware scheduling releases segments as streamed audio plays.
  - Once reasoning completes, the rest of the reply uses the full reasoning state, and a final continuation can correct an early start.
  - **Speak-First** (no reasoning prefix) is the default; **Think-First** waits for a short prefix.
- **Adaptive Thinking** routes each turn to a direct path (empty think block) or a deliberate one (Fig.7A).
- **MTP3** has three draft heads. Spoken output uses strict speculative verification. Private reasoning uses Medusa-style typical acceptance (entropy-adaptive threshold) with repetition penalty 1.05 (p.15).
- **Voice agent** (pp.17–18). Requests are routed three ways: answered directly, sent to lightweight tools (weather, web search), or handed to an asynchronous backend for private or multi-step work. The model clarifies or confirms before acting. Users can keep talking during execution, and results feed back into context ReAct-style.

## Training pipeline

Optimizer, LR, batch size, steps, trainable-module schedule and hardware are **not reported for any stage**.

### Stage 0: tokenizer / codec / decoder training
Not reported. AuT is reused from Qwen3-Omni (p.3), and generator training is not described. The encoder is stated as frozen only in the ASR-Max SFT (p.5).

### Stages 1–3: pretraining (modality alignment → multimodal mixed training → cooldown)
- **Objectives:** build the acoustic-to-LM interface; then joint audio-text modeling "at scale"; then a cooldown with "greater weight on high-quality data" (p.4).
- **Data.** The StepAudio 2.5 curation pipeline [16], extended for language coverage and realtime dialogue (p.4). It:
  - filters with SED and VAD, and resegments into semantically complete samples;
  - tags quality, synthetic-speech likelihood and speaker count;
  - transcribes and language-IDs with several recognizers and cross-checks them;
  - grades samples for quality-aware sampling.

  Hours and task mix: not reported.
- **Mixture.** The pure-text share is increased "to preserve the general capabilities of the base language model and support subsequent reasoning and agent training" (p.4). The ratio is not reported.
- **Length / tokens.** Fixed **32K** sequence length, **1.2T tokens** across the three stages (p.4). Per-stage split not reported.

### Stage 4: midtraining for realtime interaction
- **Context:** 128K, for long histories, earlier requirements and tool results (p.5).
- **Data:** perception, synthetic conversational and voice-agent data. The audio-understanding and agent shares are "substantially" increased (p.5). Broadly useful audio-QA examples from the SFT pipeline can also enter (p.7).
- **Duplex adaptation** (p.9). Midtraining adapts the model to the interleaved format and supervises streaming ASR, VAD and streaming utterance-completeness prediction. It includes **>10,000 h of synthetic full-duplex interaction data**, plus text data "to help retain general language and reasoning capabilities."
- Token count and mixture shares: not reported.

### Stage 5a: SFT, ASR branch (StepAudio 3 ASR Max)
- **Setup** (p.5): packed to 32K; SpecAugment-style masking; **encoder frozen**, adapter + LLM trained.
- **Context:** optional history, scenario or terminology, with targets grounded in the waveform (p.5).
- **Data** (pp.5–6): short labeled utterances, ROVER-fused long pseudo-labeled sessions, and TTS-synthesized long-tail terminology that is kept only if pronunciation-verified.
- Sizes: not reported.

### Stage 5b: post-training, realtime branch
1. **Audio-understanding SFT** (pp.6–8, Fig.4).
   - *Taxonomy:* lexical, paralinguistic, events, speaker/temporal, music and reasoning.
   - *Pipeline:* sampling quotas and dedup → caption → capability tags → clip-specific queries → independent labels from several models.
   - *Filtering:* text-only LLM judges score quality and "case value", and cross-model answer consistency stands in for grounding (the judges never hear the audio). Disagreements are relabeled.
   - *Size:* final set **~100K** (p.8).
2. **Multi-turn dialogue + reasoning SFT** (pp.11–12, Fig.6).
   - *Generation* is factorized over topics/discussion points, personas, turn depth and target capabilities, with stratified rotation.
   - *Construction:* two models self-play N−1 turns; then a final-turn query (optionally with injected capability requirements) is answered and labeled *with reasoning*.
   - *QC:* three independent stages (context-query, capability, response quality / persona / spoken naturalness) plus deterministic checks; only examples scoring high in all three are kept.
   - Joint dialogue and reasoning training targets deliberation on hard turns and brevity on routine ones (p.11).
   - *Size:* not reported.
3. **Duplex post-training** (p.9): "high-quality interaction data" on turn taking, backchannels, interruptions and background-speech rejection. Size and source: not reported.
4. **Adaptive-Thinking supervision** (p.13). Labels are built per assistant turn:
   - A fixed probe model, trained without this transform, regenerates the answer with an empty think block.
   - A blind judge compares the original and no-think answers against the target and labels whether reasoning changed quality. Judgment consistency, reasoning-answer coherence and a task-type prior set how conservative to be.
   - "Reasoning-unnecessary" turns get an empty think block, up to a **per-domain no-think budget**, stratified by topic, with drop caps per capability.
   - The result is trained as a separate model (Tab.5 caption).
5. **Voice-agent data** (p.18).
   - *Targeted dialogues:* routing, clarification, private-context retrieval, confirmation, execution updates, progress queries and result reporting.
   - *Negatives* against unnecessary tool calls and unsupported claims of success.
   - *Real multi-step agent trajectories*, filtered for call structure, argument consistency, evidence grounding and spoken suitability.
   - Voice-agent data also appears in midtraining (p.5).
6. **Teachers + merge** (p.16).
   - Several teachers are trained from a common base, each on a different mixture: multi-turn dialogue, audio understanding, general text reasoning/knowledge, or mixed-domain.
   - They are merged as θ = Σαᵢθᵢ with **4 teachers at 3:1:1:1**, coefficients tuned on held-out dialogue, audio and text evaluations.
   - Which teacher gets weight 3 is not stated.

### Stage 6: RL / preference
**Not described.** The report contains no RL, reward-model or preference-optimization terms.

## Data pipeline

- Pretraining, ASR, audio-QA, dialogue and agent data construction are summarized per stage above (pp.4–8, 11–12, 18).
- **Duplex data** is only characterized as ">10,000 hours of synthetic full-duplex interaction data" (p.9). The report does **not** describe:
  - the TTS or speakers used;
  - how overlaps, backchannels, interruptions and silences were synthesized;
  - any real-recording share.

  The duplex section cites Moshi [10], Chronological Thinking [11] and DuplexSLA [12].
- **StepAudioChat (eval).** It is closed and **text-based** (p.10).
  - De-identified real utterances shape the phrasing only.
  - Each item has independently checkable criteria and a valid reference response plus a deliberately flawed one.
  - Difficulty is calibrated with two reference systems (pp.10–11).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Pretraining (3 stages) | 1.2T tokens, 32K sequence (p.4) | not reported | "increased" text share; split not reported |
| Midtraining | not reported; includes >10,000 h synthetic duplex data (p.9) | not reported | 128K context (p.5) |
| SFT, ASR Max | not reported; 32K packing (p.5) | not reported | encoder frozen |
| Realtime SFT / teachers | ~100K curated audio-QA (p.8); rest not reported | not reported | 4 teacher runs (p.16) |
| Merge | — | parameter averaging | 3:1:1:1 |
| Inference: MTP | 1.801 accepted drafts/step (MTP3, Medusa acceptance; Tab.6) | 2.05× wall-clock | private reasoning only |

## Preserving text intelligence / knowledge

- **Mechanisms:**
  - knowledge is inherited from a 196B MoE text base (p.3);
  - the pretraining text share is raised (p.4);
  - text is mixed into duplex midtraining (p.9);
  - a general-text teacher is merged in (p.16);
  - the LLM's output channel stays text/state, with the generator separate (p.8; our reading of Fig.5A).
- **Evidence.**
  - Merging lifts the general-text macro to **76.5**, from 57.2 for Teacher 1, the strongest dialogue teacher. HMMT goes from 44.0 to 86.8 (Tab.8).
  - Thinking adds +5.94 on Knowledge (Tab.5).
  - Realtime mode costs 2.7 on Knowledge and 12.2 on instruction following (derived: Tab.4 vs Tab.10).
- **Gaps in the evidence.**
  - No Step 3.7 Flash text-only scores are given, so retention cannot be computed.
  - No metric measures knowledge *in generated speech*, because StepAudioChat is text-based (p.10).

## Key results

- **ASR, StepAudio 3 ASR Max (Tab.1):** LibriSpeech clean / other 1.18 / 2.28; AISHELL-1 0.49; WenetSpeech meeting 4.35.
- **Audio understanding (Tab.2).** Macro **81.3**, vs Gemini 3.1 Pro 81.8, Doubao 2.0 Lite 77.7 and Gemini 3 Flash 77.1.

  | Benchmark | Score | Note |
  |---|---|---|
  | MMSU | 90.6 | |
  | MMAR | 86.5 | |
  | MMAU | 79.0 | |
  | WildSpeech | 77.1 | |
  | Step-Caption | 78.2 | |
  | MTalk-Bench | 91.7 | |
  | Big Bench Audio | 98.1 | |
  | AudioMultiChallenge | 49.3 | Gemini 3.1 Pro: 67.0 |
- **Full duplex: AA subset of FD-Bench v1/v1.5 (Tab.3).**

  | System | Pause | Turn-taking | Interruption | Backchannel | Overall |
  |---|---|---|---|---|---|
  | StepAudio 3 Realtime | 98.9 | 100.0 | 99.0 | 98.0 | **98.9** |
  | GPT-realtime-2 (High) | 99.3 | 100.0 | 95.0 | 86.7 | 95.3 |
  | Qwen Audio 3.0 Realtime Plus | 98.0 | 98.0 | 98.0 | 100.0 | 98.4 |
  | Grok Voice Think Fast 2.0 | 98.0 | 91.0 | 97.0 | 95.0 | 95.1 |

  **No latency in ms is reported anywhere.**
- **StepAudioChat (Tab.4, Tab.10).** Reasoning mode 73.0, vs Doubao 2.0 Lite 70.5, DeepSeek-V4-Flash 71.4 and Kimi K3 77.1; realtime mode 70.4.
- **τ-Voice (Tab.9).** 56.0 overall: Airline 60.0, Retail 37.7, Telecom 70.2. Comparisons: Grok 56.5, Qwen 54.6, GPT-Realtime-2.1 45.7.
- **Text (Tab.10).** HMMT-Feb-2026 86.8, GPQA-Diamond 83.0, MultiChallenge 59.7 (Gemini 3 Flash: 85.9 / 90.3 / 68.1).

## Ablations & lessons

- **SFT quality beats volume** (p.8). Changing only the SFT data, ~100K curated examples beat ~2M random ones at about 1/20 of the size:

  | Metric | ~2M random | ~100K curated |
  |---|---|---|
  | MMSU | 78.78 | **89.70** |
  | MMAR | 74.70 | **84.50** |
  | WildSpeech | 74.20 | 77.11 |
  | MTalk-Bench subsets | 88.83 | 90.84 |

- **Where thinking pays off** (Tab.5). Full thinking vs forced no-think, same weights:
  - Reasoning +11.37 (71.89 vs 60.52);
  - Persona +8.37;
  - Knowledge +5.94;
  - Faithfulness only +1.72; Pragmatics +1.18.
- **Adaptive Thinking recovers about 2/3 of the thinking gain at about a 2/3 think rate.** Macro scores (derived: unweighted means of Tab.5's 8 rows):
  - forced no-think 66.47;
  - full think 71.27;
  - Adaptive 69.66, at a mean think rate of 66.05%.

  The selector is therefore close to proportional, not targeted, and the paper concedes this. Reasoning gets only a 59.5% think rate despite the largest benefit, while Faithfulness gets 79.2% for +1.72 (p.15).
- **Realtime mode mostly costs instruction following.** Moving from reasoning mode to realtime Think-While-Speaking changes scores as follows (derived: Tab.4 → Tab.10):
  - Instruction Following −12.2;
  - Safety −3.9;
  - Knowledge −2.7;
  - Reasoning +0.6.

  Speaking before reasoning finishes hurts constraint tracking more than reasoning accuracy.
- **MTP** (Tabs.6–7).

  | Config | Accepted drafts/step | Wall-clock speedup |
  |---|---|---|
  | MTP3 strict | 1.231 | 1.76× |
  | MTP3 Medusa | 1.801 | **2.05×** |
  | MTP5 strict | 1.353 | 1.49× |
  | MTP5 Medusa | 2.153 | 1.72× |

  - Heads 4–5 add little: strict marginal acceptance is 10.7% and 5.7%.
  - The authors warn the wall-clock ratios are "not a controlled comparison of draft depth" (p.16).
- **Weight merging** (Tab.8). It balances capabilities rather than dominating every teacher:

  | Domain | Merged | Best teacher |
  |---|---|---|
  | Audio macro | 81.3 | 81.3 (tie) |
  | Text macro | **76.5** | 73.2 |
  | Dialogue macro | 73.0 | 74.2 |

  It lets each teacher's data mixture be developed independently and recombined without retraining on the union (p.16).
- **Duplex from synthetic data.** >10K h of synthetic duplex data plus curated post-training gives the top AA FD score (98.9). There is **no ablation** of duplex data size, synthetic vs real, or the ASR, VAD and completeness auxiliary targets (p.9).

## Implications for our setup

1. **Our 10K h should be the duplex-adaptation layer, not the whole mid-training.** SA3-RT's duplex data (>10K h synthetic) is about the size of our corpus. But it is added in midtraining *after* 1.2T speech-text tokens with a raised text share (pp.4, 9). Weak continuation and weak knowledge are what you would expect if 10K h is the only audio the model sees. Add a broad speech-text stage (ASR, TTS and interleaved data) with substantial text replay before or alongside it; Step-Audio 2 and StepAudio 3 Gen both used ~50% text.
2. **Keep the LLM's output textual.** The SA3-RT duplex LLM emits state or text tokens, and a separate generator renders the audio (pp.4, 8). If our Qwen3 LLM predicts codec tokens directly, it carries the acoustic loss. Prefer a text-first model channel, with acoustic tokens from a depth/AR head. Chrono-Thinking and SA3-Gen show this helps knowledge.
3. **Encode both channels with one perception encoder.** SA3-RT feeds both through Qwen3-Omni AuT (p.3). AuT produces continuous 12.5 Hz features at the same frame rate as the Qwen3 codec. Using AuT for at least the user channel stays within the Qwen3 family and gives the LLM perception-grade input instead of codec codes.
4. **Add cheap auxiliary targets.** Derive per-channel streaming ASR, VAD state and utterance-completeness labels from our 10K h and supervise them (p.9). Replace per-frame silence with coarse state tokens.
5. **Keep text in duplex training** (p.9), and track text benchmarks after every stage.
6. **Curate SFT** (100K > 2M; MMSU +10.9, derived). Consider teachers merged in weight space (duplex/dialogue, audio, text) to restore text/knowledge cheaply (Tab.8: text macro 57.2 for the strongest dialogue teacher → 76.5 merged).
7. **Reasoning while speaking is heavy infrastructure:** two concurrent decoding calls plus playback-aware scheduling. At our scale, Chronological Thinking (think while listening) is the cheaper first step. If we add reasoning, watch instruction following (−12.2 in realtime mode).
8. **Evaluation.** Use the AA FD-Bench subset categories, and add a *speech-output* knowledge test, which SA3-RT lacks.

## Open questions / unclear points

- **The generator** (tokens, rate, codebooks, architecture, data) is not reported. Is it StepAudio 3 Gen?
- **320 ms cadence vs response text.** One state *or* text token per 320 ms block is 3.125 tokens/s. Is response text emitted at that rate inside the duplex LLM, or produced in bursts by the Articulation Brain ("short response segments", p.13)? How the duplex state machine and the two-call reasoning system share one model is not explained.
- **Encoder frame rate** is not stated, and the 2+2 and 4-frame block drawings are schematic. At AuT's native 12.5 Hz, a 320 ms block holds 4 frames per stream, which matches Fig.5B's four `a` slots if those are per stream (inference).
- **Missing training details:** per-stage tokens, text ratios, task mixtures, LRs, batches, frozen-module schedule and hardware; midtraining token count; how the duplex synthetic data was built; size of the post-training interaction set.
- **Merging:** which teacher gets weight 3, the teacher mixtures, and whether the Adaptive-Thinking model or later fine-tuning is part of the deployed checkpoint.
- **MTP scores move under strict verification.** Strict verification should preserve the distribution, yet MTP3 strict shifts scores vs baseline (IF −2.94, Reasoning +3.54; derived from Tab.6). Possibly MTP-head training altered the backbone. Tab.6's baseline also differs from Tab.5's Direct SFT on some rows (Reasoning 70.76 vs 71.89), so evaluation settings differ.
- **RL:** was any RL used? Step-Audio 2 used PPO/GRPO; this report is silent.
- **Latency and comparability:** no latency numbers. Most benchmarks are 2026, AA-hosted or closed, which limits independent comparison.
