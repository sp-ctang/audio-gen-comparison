# Full-Duplex-Bench-v3 (FDB-v3): tool use for full-duplex voice agents under real-world disfluency

| Field | Value |
|---|---|
| Org | National Taiwan University (Guan-Ting Lin, Hung-yi Lee) + NVIDIA (Chen Chen, Zhehuai Chen) (p.1) |
| Date / arXiv | arXiv:2604.04847v1, 6 Apr 2026 (p.1) |
| Category | benchmark |
| Backbone | n/a |
| Audio input | n/a. The benchmark input is real human recordings, streamed to each system. |
| Audio output | n/a |
| Waveform decoder | n/a |
| Duplex mechanism | n/a. Every system under test runs through the LiveKit Realtime Voice Agent framework, with streaming audio and real-time tool calls (p.4). |
| Open weights / code | A demo page is linked (p.1). The text calls the benchmark "open, reproducible" (p.3) but gives no explicit data or code release link. |
| Relevance to us | **medium**. It is the only benchmark here that tests tool use with real, disfluent speech, and NVIDIA used it for VoiceChat. It only matters if tool calling is in scope, and its sample is small (n=100). |

## TL;DR

- 100 real recordings from 12 speakers, all in uncontrolled environments. Each is annotated for five disfluency types: fillers, pauses, hesitations, false starts and self-corrections (p.2–4).
- Each scenario requires one or more calls to deterministic local mock APIs across four domains (11 functions in Tab.1). Difficulty tiers are Easy, Medium and Hard. 21 scenarios test self-correction, where the user changes intent mid-utterance and the agent must roll back state (p.2–4).
- Metrics:
  - tool-selection F1, argument accuracy (GPT-4o judge), response quality (GPT-4o judge);
  - Pass@1, which requires exactly the right tools with perfect arguments;
  - turn-take rate, interruption rate and filler rate;
  - a three-way latency breakdown: first word, first tool call, task completion (p.4–5).
- GPT-Realtime leads on every accuracy metric (Pass@1 0.600) and has the lowest interruption rate (13.5%). Gemini Live 3.1 is fastest (4.25 s) but stays silent in 22% of scenarios. The cascaded Whisper→GPT-4o→TTS baseline always responds but is slowest (10.12 s) and worst on self-correction (0.176) (Tab.2–3).
- Self-correction is the hardest category for every system; the best is 0.588.
- NemotronLabs VoiceChat reports tool-selection F1 82.5, argument accuracy 42.2 and Pass@1 33.0 on this benchmark (VoiceChat Tab.4). Those numbers are not in this paper.

## Architecture

There is no model. This section describes the **benchmark design**.

**Domains and mock APIs (Tab.1, p.4).** The APIs run locally with deterministic, zero-latency responses, so measured latency is model overhead only.

| Domain | Functions |
|---|---|
| Travel & Identity | search_flights, book_ticket, update_travel_profile |
| Finance & Billing | query_card_benefits, calculate_currency_exchange, modify_autopay_source |
| Housing & Location | search_apartments, update_search_filter |
| E-Commerce Support | check_order_status, cancel_pending_action, process_exchange |

**Difficulty tiers (p.3).**
- Easy: a single step.
- Medium: two steps with moderate ambiguity.
- Hard: multiple steps with conflicting constraints.

The per-tier Pass@1 fractions are consistent with about 36 / 34 / 30 scenarios, and the per-domain fractions with about 29 e-commerce / 25 finance / 26 housing / 20 travel. Both splits are derived from the score fractions; the paper does not state them.

**Disfluency taxonomy (p.3).** Each type targets a different failure mode:
- false starts: does the model drop the abandoned intent without hallucinating a call?
- self-corrections: does it update the parameter (state rollback)?
- fillers: do redundant tokens hurt accuracy or add latency?
- pauses and hesitations: do they break end-of-turn detection?

**Metrics (p.4–5).**
- **Tool-selection F1:** over expected versus actual calls.
- **Argument accuracy:** GPT-4o judge. Dynamic references to earlier results are allowed, and formats, aliases and ±5% numeric tolerance are accepted (App. A.1).
- **Response quality:** GPT-4o judge on the spoken transcript. Partial completion of a multi-step task scores 0 (App. A.2).
- **Pass@1:** exactly the expected tools, with perfect argument accuracy on every call.
- **Turn-take rate:** fraction of turns with a "natural-timing" response.
- **Interruption:** the agent starts speaking before the user stops (Δt < 0).
- **Filler rate:** share of responses that open with a content-free sentence such as "let me check".
- **Latency:** first word, first tool call, and task completion. GPT-4o separates the filler from the key-information sentence in ASR chunks (App. A.3).
- **Pre-emptive tool call:** a call issued before the user finishes speaking (p.6).

## Training pipeline

n/a. This is a benchmark paper; no model is trained.

### Stage 0: tokenizer / codec / decoder training

n/a.

## Data pipeline

- **Speakers:** 12, including native and non-native English speakers (Korean and Russian backgrounds) with varying accent strength (p.3–4).
- **Recording:** speakers got detailed scenario contexts and performed the prompts naturally. 11 of 12 setups used everyday built-in microphones, in rooms ranging from quiet to mildly noisy (p.4).
- **Trailing silence:** 30 s of each speaker's real ambient room sound is appended, not digital silence, so the background stays consistent (p.4).
- **Assignment:** each speaker was given 10 scenarios across all four domains, with disfluencies represented proportionally. All recordings were quality-reviewed (p.4).
  - 12 × 10 = 120, but the dataset has 100 recordings. The paper does not explain the gap (derived).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| — | n/a (benchmark) | — | All cloud models were called from one fixed server region over high-bandwidth links (p.8) |

## Preserving text intelligence / knowledge

Not a training paper. The relevant evidence is the cascade versus end-to-end comparison.
- The cascade (Whisper → GPT-4o → OpenAI TTS) uses the same LLM as GPT-Realtime, yet its Pass@1 is 0.450 versus 0.600.
- On self-corrections it scores 0.176 versus 0.588. Whisper finalizes the uncorrected transcript before the correction arrives, so the LLM never sees it (p.7).
- The authors' reading is that ASR errors propagate downstream, and a finalized transcript leaves no room for rollback (p.7). For us, this favors conditioning the LLM on speech rather than on an intermediate transcript, as VoiceChat does by keeping its RNN-T transcript out of the LLM.

## Key results

**Overall** (Tab.2, p.5):

| System | Tool-sel F1 | Arg acc | Resp qual | Pass@1 | Turn-take | Latency (s) | Interrupt | Filler |
|---|---|---|---|---|---|---|---|---|
| GPT-Realtime | 0.876 | 0.680 | 0.792 | 0.600 | 96.0% | 6.89 | 13.5% | 16.9% |
| Gemini Live 2.5 | 0.786 | 0.593 | 0.554 | 0.490 | 92.0% | 7.26 | 14.1% | 8.9% |
| Gemini Live 3.1 | 0.817 | 0.588 | 0.718 | 0.540 | 78.0% | 4.25 | 19.2% | 31.7% |
| Grok | 0.797 | 0.542 | 0.617 | 0.430 | 94.0% | 6.65 | 25.5% | 44.3% |
| Ultravox v0.7 | 0.794 | 0.513 | 0.510 | 0.410 | 96.0% | 8.40 | 47.9% | 88.0% |
| Cascaded | 0.803 | 0.562 | 0.600 | 0.450 | 100.0% | 10.12 | 33.0% | 26.9% |
| *NemotronLabs VoiceChat (VoiceChat Tab.4)* | *0.825* | *0.422* | — | *0.330* | — | — | — | — |

The VoiceChat row comes from the VoiceChat paper, and its evaluation harness is not described. If its numbers are comparable, it ranks second on tool selection and last on arguments and Pass@1.

**Pass@1 by difficulty, Easy / Medium / Hard** (Tab.4, p.6):
- GPT-Realtime: 0.750 / 0.588 / 0.433.
- Gemini 3.1: 0.694 / 0.588 / 0.300.
- Cascaded: 0.639 / 0.441 / 0.233.
- Grok: lowest on Hard, 0.200.

**Pass@1 by disfluency type, Filler / Pause / Hesitation / False start / Self-correction** (Tab.3, p.6):
- GPT-Realtime: 0.621 / 0.556 / 0.700 / 0.667 / 0.588.
- Cascaded: self-correction only 0.176.
- Pauses are the weakest category for Grok and Ultravox (both 0.333).

**Pass@1 by domain** (Tab.5, p.7):
- Finance is easiest (GPT-Realtime 0.960) and Housing hardest (GPT-Realtime 0.308, Grok 0.115).

**Latency breakdown, first word / first tool call / task completion in seconds** (Tab.6, p.7):

| System | First word | First tool call | Task completion |
|---|---|---|---|
| GPT-Realtime | 6.36 | 3.89 | 6.89 |
| Gemini 3.1 | 3.95 | 2.21 | 4.25 |
| Grok | 5.97 | 0.63 | 6.65 |
| Ultravox | 3.88 | 6.01 | 8.40 |
| Cascaded | 8.78 | 3.15 | 10.12 |

## Ablations & lessons

These are findings from comparing systems; nothing is ablated.

- **Committing early locks in stale intent.**
  - Pre-emptive tool-call rates: Grok 41.6%, Ultravox 23.2%, Gemini 2.5 17.9%, Gemini 3.1 16.9%, GPT-Realtime 10.8% (p.6).
  - In the double self-correction case (travel_19), Gemini 3.1 called the API 2.27 s before the user finished and kept the uncorrected "Rome". Only GPT-Realtime applied both corrections (Tab.8, p.8).
  - In the clean three-tool case (finance_18), the same eagerness made Gemini 3.1 2.3× faster than GPT-Realtime (3.92 s vs 9.20 s) (Tab.7).
  - The open design question is when to commit to tool arguments.
- **Pre-emptive tool calls and interruptions are different failures.**
  - Grok calls tools early but lets the user finish (25.5% interruptions).
  - Ultravox speaks early with fillers: 88% filler rate, 47.9% interruptions, and its tool calls come late (6.01 s) (p.6–7).
  - Filler speech lowers perceived latency but causes barge-ins on users who are still talking.
- **"Silent worker" failure.**
  - Gemini 3.1 gave no spoken response in 22 scenarios: 0% of Easy, 23.5% of Medium, 46.7% of Hard. In 86% of those silent cases it still executed tool calls (p.7).
  - Reasoning and speech generation can come apart. In contrast, all 4 of GPT-Realtime's silent cases had zero tool calls.
- **Self-correction is unsolved.** Best 0.588; every other system is at 0.471 or below (Tab.3).
- **Difficulty compounds.** Pass@1 falls for every system from Easy to Hard. The cascade drops from 0.639 to 0.233, which the authors attribute to ASR errors compounding with task complexity (p.5).

## Implications for our setup

Our setup: Qwen3 LLM backbone, Qwen3 audio tokenizer and its vocoder/decoder, mid-trained on about 10K hours of duplex conversational speech. Problems: weak audio continuation and weak semantic knowledge in the generated speech.

- **Not a first-priority eval for our current problems.** Audio continuation and knowledge are better measured with VoiceBench-style QA, FDB 1.0/1.5 and TTS-style WER/MOS. Add FDB-v3 once tool calling is on the roadmap.
- **Statistical power is low.** n=100 gives a standard error of about ±5 points on Pass@1 (derived, binomial at p≈0.5). Per-category subsets are smaller still (17 self-correction items implied by the Tab.3 fractions, derived), so a gap of one or two items moves a category score by 6–12 points. Treat differences under about 10 points as noise.
- **Harness cost.** Results depend on the LiveKit integration, GPT-4o as judge, and network latency for cloud models. To compare our model with the published rows, we need the same harness. Check whether NVIDIA's VoiceChat numbers used it; their paper does not say.
- **Design lessons for a duplex agent with tools:**
  - Delay argument commitment until there is end-of-turn evidence. Note that VoiceChat shifts its agent-text channel 160 ms later but deliberately does *not* shift its function channel (VoiceChat p.14), so its tool calls get no such buffer.
  - Train on self-correction data: mid-utterance changes where the target tool call uses the corrected value.
  - Measure interruptions caused by filler speech; a VoiceChat-style `ack_message` adds exactly this risk.
- **Disfluency coverage for training data.** The five categories make a useful checklist for augmenting our duplex SFT data with realistic user speech, beyond the clean TTS speech most synthetic pipelines produce.

## Open questions / unclear points

- **Category counts:**
  - Per-tier, per-domain and per-category counts are not given; ours are inferred.
  - Tab.3's self-correction fractions imply 17 items, while the text says 21 scenarios test self-correction.
- **Recording count:** 12 speakers × 10 scenarios = 120, but there are 100 recordings.
- **Metric definitions:**
  - Pass@1 is said to require "three conditions" but only two are listed.
  - The turn-take rate's "natural-timing" threshold is not defined.
  - The text says "four dimensions" but lists five metric groups.
- **Release:** whether the data and evaluation code are released is not stated explicitly in the paper text.
- **Measurement noise:** latency for proprietary systems includes network and server-load noise (p.8). The zero-latency mock APIs don't test timeouts or malformed responses.
