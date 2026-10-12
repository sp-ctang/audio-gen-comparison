# Synthesis: how full-duplex S2S models are trained (state of the art, Oct 2026)

**Scope.** 35 papers + the GPT-Live post in `../papers/`, the links in `links.md`, web sweeps of frontier systems, and a methodology sweep of ~40 further papers (`web/speech-lm-training-methodology.md`). Per-source notes are in `papers/` and `web/`, cross-model tables in `training-recipes.md`, and the draft plan in `plan.md`.

**Our setup.** Qwen3 LLM + Qwen3 audio tokenizer + its decoder, mid-trained on ~10K h of real duplex speech. **Symptoms:** weak audio continuation and weak semantic knowledge in the generated speech.

**Labels.** "[web]" marks numbers from web sources (blogs, docs, leaderboards, arXiv abstracts) rather than from papers we read in full. Self-reported frontier numbers are flagged where relevant.

---

## TL;DR: ten findings that matter for us

1. **10K h of duplex speech is a post-training-scale dataset, not a mid-training dataset.**
   - Every system with good continuation and knowledge first ran a broad speech–text stage of roughly 10⁵–10⁸ hours, or 0.1–5T tokens:
     - Moshi: 7M h
     - Raon: 1.01M h
     - Covo: ~8M h / 2T tokens
     - Step-Audio 2: 1.36T tokens
     - StepAudio 3 Realtime: 1.2T tokens
     - Kimi-Audio: 13M h
     - MiMo-Audio: >100M h
   - Duplex data then comes on top as a thin layer: Moshi 2K h real + 20K h synthetic; StepAudio 3 Realtime >10K h synthetic; Covo 0.25% of tokens; Raon ~12% of tokens.
   - Our symptoms are what you'd expect when the duplex layer is trained **without the foundation underneath**.
   - Speech-only training scales poorly. Textless speech LMs gain linguistic ability per unit of compute up to ~10³× more slowly than text LMs (Cuervo & Marxer, arXiv 2404.00685). Text-LM initialization is the big lever: TWIST at ~15K h nearly matches ~150K h.
2. **Knowledge reaches speech through text, and at our data scale a text stream beats more data.** Every system with good spoken knowledge has an explicit text path: an inner monologue, text-first output, a Thinker, or a text-only LLM plus a renderer.
   - **The most directly relevant controlled result** is Chou, Zhou & Livescu (SLT 2026, arXiv 2609.14743, Tab. I/II; verified), on 10K h of Emilia. Continuation GenPPL (lower is better):

     | Model | 10K h | 45K h |
     |---|---|---|
     | Speech-only | 183.3 | 149.3 |
     | Speech + text stream (2-frame lead) | **52.9** | 51.6 |
     | Text-only reference | ~30 | — |

     Adding a text stream helps far more than 4.5× more speech.
   - Moshi's inner monologue tripled spoken QA (Audio TriviaQA 7.3 → 22.8) and continuation length (602 → 1,920 characters).
   - A text-only LLM plus a separate audio decoder doubled Llama Q over joint text+audio prediction (Chrono: 15.0 → 30.4).
   - VoiceChat, whose LLM never emits audio, scores VoiceBench 55.1 vs ~30 for Moshi and PersonaPlex.
3. **Real conversational audio does not teach knowledge; speech synthesized from text corpora does.**
   - Kimi-Audio, interleaved on 13M h of *real* speech, still has base-model SpeechMMLU T2S 0.0 / S2S 11.8 vs S2T 67.9.
   - PersonaPlex gains no knowledge from thousands of duplex hours (VoiceBench 30.6 vs Moshi 29.5).
   - The knowledge carriers in the strong recipes are all synthetic, built from text:
     - GLM-4-Voice: 455B synthetic interleaved speech tokens. In its scaling study (Zeng et al., arXiv 2411.17607), S→S Llama Q is 2.3 with no interleaving and 37.0 / 44.0 / 50.7 at 100B / 200B / 600B interleaved tokens.
     - The best interleaving mix found was ~⅓ of tokens interleaved, with 20–40% of words rendered as speech (Sternberg et al., arXiv 2606.22473) [web].
     - VoiceChat: CPT on TTS-rendered pseudo-dialogues from text pretraining corpora.
     - Moshi: >20K h synthetic instruction dialogues grounded in Wikipedia and StackExchange.
     - Raon: 106K h synthetic duplex.
     - Chrono: 15K h knowledge-dense dialogues.
4. **Keep 30–50% text in every speech-text stage and train the backbone gently.**
   - Text share, typically: ~50% for Moshi, MiMo, Kimi, Step-Audio 2 and StepAudio 3 Gen; 40% Covo; 44–69% Audex; ~25% Qwen3-Omni; ~30% GLM.
   - Backbone LR ~1e-5–5e-5, new modules ~1e-4–2e-4.
   - Even so, losses remain: Moshi MMLU 54.3 → 49.7, mostly during audio pretraining; Qwen2.5-Omni −2.9 to −9.3 on 10 text benchmarks.
   - Removing replay costs spoken knowledge directly (Moshi Audio TriviaQA 22.8 → 18.3).
5. **Take acoustics off the backbone.** For a 12.5 Hz × 16-codebook tokenizer, the proven layout is: LLM predicts text + codebook 0; a small depth/MTP head predicts codebooks 1–15.
   - This is Qwen3-Omni's Talker + MTP, StepAudio 3 Gen's 4-layer depth head, Sesame CSM, and Breeze-TTS-2, which uses *our* tokenizer and is #1 open-weight on the AA Speech Arena [web].
   - Controlled evidence:
     - Continuous/flow head over discrete depth: CALM MOS 3.45 vs 2.75.
     - Local head over a 4.7× larger delay model: MOSS-TTS.
     - Depth transformer over independent heads: Moshi perplexity 36.8 vs 135.4.
     - Zero-init RVQ adaptor on input embeddings: StepAudio 3 Gen SpeechMMLU spoken answers 12.1 → 58.8.
6. **Freeze first, unfreeze later; distill or merge back if text regresses.**
   - Freeze-then-unfreeze:
     - MOSS-Speech (Qwen3-8B, ~4.95M h; Tab.7): MMLU 66.5 vs 62.1 and sStoryCloze 63.1 vs 56.6 without freezing. Base Qwen3-8B scores 76.6, so even the frozen recipe lost ~10 MMLU [web, verified].
     - Audex: unfreezing embeddings early cost −7.7 MMLU-Pro.
   - Distillation back to the text model:
     - on-policy KL to the frozen backbone (Raon);
     - top-20-logit KL from a stronger text LLM (Covo);
     - strong-to-weak distillation (Qwen3-Omni);
     - spoken answers distilled toward the model's own text-conditioned answers (Qwen3.5-Omni [web]).
   - **On-policy distillation beats SFT for closing the speech-vs-text gap.** X-OPD (Interspeech 2026, arXiv 2603.24596, Tab. 1; verified):
     - Setup: Qwen3-Omni-30B-A3B student, Qwen3-30B-A3B text teacher, only 27.8K prompts.
     - The speech-input gap fell from 11.3% to 3.4%, and Big Bench Audio rose from 85.7 to 93.4.
     - SFT on the same prompts *widened* the gap to 22.8%.
   - Replay works even at tiny ratios, and LoRA does not protect [web]:
     - Hsiao et al. (arXiv 2505.17496): without replay, text Llama Q fell 70.0 → 14.3; replay at 0.5% of each stage restored 66.3.
     - Yano et al. (arXiv 2604.00489): on speech-only data, LoRA lost as much text ability as full fine-tuning.
   - Merging with a text-heavy teacher: StepAudio 3 Realtime text macro 57.2 → 76.5.
   - Modality-split deep layers: Lychee-FD (StepAudio-2-mini, 140K synthetic duplex dialogues, 8×H20 ~16 h) S2S Llama Q 65.3 vs 36.0 without them (Tab.1) [web, verified].
7. **Duplex behavior is cheap; content is expensive.**
   - PersonaPlex reaches most of its turn-taking quality with ~25% of 2.25K h.
   - MinMo's 18M-parameter head on 4K h gives turn-taking control.
   - Kyutai's GRPO (32 H100, ≤2,000 segments per behavior) fixes Moshi's timing: pause takeover 0.445 → 0.226, interruption latency 1.38 → 0.41 s.
   - Grok Voice gained FDB 77.8 → 95.1 between versions, mostly from post-training [web].
   - Open duplex models already have timing but not knowledge (AA Big Bench Audio: Moshi 4%, PersonaPlex 19%, FLM-Audio 16%) [web].
   - Human–human conversation alone damages assistant semantics (Li & Chu, EMNLP 2026, arXiv 2609.03321) [web]:
     - Fisher + Switchboard only: turn-taking F1 0.618, VoiceBench 18.3.
     - Mixed with assistant-style data: F1 0.654, VoiceBench 64.8 (base LLM 64.9).
8. **Frontier systems split *interaction* from *reasoning*.**
   - GPT-Live: full-duplex frontend + delegated text backend; τ-Voice 30–46% single-model → 74.5–86.2% delegated [web].
   - Gemini 3.8 Live Extended Thinking: τ-Voice 30.1% → 68.6%; Big Bench Audio 70.5 → 95.9 with thinking on 3.1 [web].
   - Thinking Machines interaction + background model: Big Bench Audio 75.7 → 96.5, self-reported [web].
   - StepAudio 3 Realtime: Think-While-Speaking.
   - MoshiRAG: TriviaQA 22.8 → 78.2 [web].
   - KAME feeds a backend LLM's text into the duplex model as an extra "oracle" stream: MT-Bench 2.05 → 6.43 at 0 s median latency, vs a cascade at 7.70 and 2.1 s (arXiv 2510.02327) [web].
   - A ~8B open frontend will not hold frontier-level knowledge by itself, so plan an async text path.
9. **Post-training is now standard.** Curated SFT beats volume: StepAudio 3 Realtime 100K curated > 2M random, MMSU 78.8 → 89.7. Then preference/RL in two flavors:
   - **Speech-quality DPO** on WER/UTMOS/SIM-ranked pairs: Qwen2.5-Omni WER −14–18%; Voxtral 2.47 → 1.84.
   - **Timing RL** with VAD/rule rewards plus an LLM-judge content reward: Kyutai, DuplexPO, HiPLEX, AdaptDuplex.
   - **Always pair timing rewards with a semantic reward.** In HiPLEX, removing the semantic reward dropped the judge score 4.08 → 2.33 [web]. In Seed LiveInterpret, a latency reward alone cut output audio ~35% [web].
   - **Text-stream-only DPO** improved Moshi QA 36.1 → 39.2; putting audio tokens in the DPO loss hurt (Wu et al., ICML 2025, arXiv 2506.21463) [web].
   - **A staged duplex curriculum beats mixed SFT** (turn-based → basic duplex → full duplex): interruption handling 0.62 → 0.82, latency 1.65 → 0.33 s (AdaptDuplex) [web].
10. **The fastest route with our components may be to start from a speech-pretrained Qwen3-family checkpoint.**
    - The 2026 "duplex conversion" wave (AdaptDuplex and DuplexOmni from Qwen3-Omni; Realtime-Venus from MiniCPM-o 4.5; Lychee-FD from StepAudio-2-mini) builds duplex models from 10⁴–10⁶ synthetic dialogues on top of an already speech-pretrained model [web].
    - Qwen3-Omni has a 12.5 Hz × 16 × 2048 output codec, AuT input at the same 80 ms clock, and Talker + MTP + Code2Wav. Qwen3-TTS (>5M h) offers a pretrained talker/MTP.

---

## 1. The canonical pipeline, and where our 10K h sits

```
 Stage 0  tokenizer/codec + decoder            (reuse: Qwen-TTS-Tokenizer-12Hz, 12.5 Hz × 16 × 2048)
 Stage 1  modality alignment                   adapters/heads only, LLM frozen (ASR, TTS)
 Stage 2  speech–text mid-training  ◄── MISSING in our run
          10^5–10^8 h, 0.1–5T tokens; ASR + TTS + interleaved + continuation + audio-only;
          30–50% text replay; synthetic knowledge speech
 Stage 3  duplex adaptation          ◄── what our 10K h is
          two-stream format, real + synthetic dialogues, turn-state targets, ~10% replay
 Stage 4  SFT                        curated assistant dialogues, persona/voice, safety, tools
 Stage 5  preference / RL            timing (VAD rewards + LLM judge), speech quality (DPO)
 (opt.)   merge/distill back to text; async reasoning/tool backend
```

| System | Foundation stage (Stage 2) | Duplex stage (Stage 3) |
|---|---|---|
| Moshi | 7M h (≈8M h processed), 50% text steps | Fisher 2K h + >20K h synthetic |
| StepAudio 3 Realtime | 1.2T tokens, increased text share | >10K h synthetic + text |
| Raon-SpeechChat | 1.01M h / 70B tokens | 119.5K h (13.2K real) + 10% replay |
| Covo-Audio | ~2T tokens (8M h), 40% text | 5B tokens (0.25%) + SFT |
| SyncLLM | 193K h synthetic, interleaved | 20K h synthetic + 1.9K h Fisher |
| VoiceChat | CPT on rendered text corpora ("≈530k", unit not printed) | ~72K h synthetic duplex chat |
| PersonaPlex | (Moshi's) | 2.25K h synthetic + 1.2K h Fisher |
| **Ours** | **none** | **10K h real** |

Compute is not the constraint; data is.

- At 12.5 Hz, one hour of dialogue is 45K backbone positions.
- One epoch over 1M h on a Qwen3-8B backbone, with 50% text replay and a small depth head, is roughly 3–4K H100-hours at 40% MFU (derived; see `plan.md` §4).
- Synthesizing speech in token space with Qwen3-TTS is also cheap (rough estimate: tens of H100-hours per 100K h).

---

## 2. Weak audio continuation: causes and levers

| Likely cause | Evidence | Lever |
|---|---|---|
| **Too little speech** for the LM to learn the codec "language" | Moshi learned continuation from 7M h; CALM ~1.8M h-equivalent; Sesame ~1M h × 5 epochs. A community Moshi-style replication on Qwen3-0.6B: fine codebooks never learned at ~100 h [web]. MiMo few-shot/continuation abilities near zero until ~0.7T tokens, then a jump. | Stage 2 at 10⁵–10⁶ h (public corpora + diarized podcasts + synthetic). Ablate scale (10K → 100K → 1M h) to find our threshold. |
| **Backbone capacity spent on acoustic codebooks** | CALM: same backbone, continuous head → MOS 3.45 vs 2.75 *and* better meaningfulness (Elo 2023 vs 1870); authors attribute it to freed backbone capacity. MOSS-TTS: a 1.7B local transformer beats an 8B delay model. SpeechTokenizer: semantic LM + light acoustic NAR beats flat (WER 6.5 vs 7.9). Llama-Mimi: more codebooks → better acoustics, worse content [web]. | Backbone predicts text + cb0; depth/MTP head (init from a Qwen3 MTP) predicts cb1–15. Up-weight cb0 (Moshi ×100: NLL 4.09 → 3.65); per-codebook depth weights; 1–2 frame acoustic delay. Train the head first with the LLM frozen (Raon: 65.5K TTS steps; StepAudio 3 Gen: detached head stage). |
| **No text anchor** | At a matched 10K h, adding a 2-frame-lead text stream cut continuation GenPPL 183 → 53, while 45K h speech-only reached only 149 (Chou et al., arXiv 2609.14743). Moshi IM: transcript NLL 3.65 → 2.77, continuation length ×3.2 ("weak models collapse to silence"). F-Actor: text stream ppl 25.2 → 21.5 [web]. DSM: text–audio lead is the main quality knob; removing a 2-word lookahead doubled TTS WER. TADA speech-free guidance: tStoryCloze 84.2 → 94.7 [web]. | Time-aligned agent text stream (DSM-style: WORD token at word onset, PAD elsewhere), text 1–2 frames ahead. Mask text ~30% during pretraining (Moshi). |
| **Dirty agent channel** | The Qwen 12Hz decoder has **no speaker conditioning**, so the LM must model timbre, room and noise in cb1–15 (93.75% of bits). FlexDuo: polluted context ppl 64.3 vs 28.9 filtered. 300 clean h > 1,500 noisy h [web]. MOSS-TTS: DNSMOS > 2.8 filtering, single-speaker segments. | Audit and clean (D7). Use a fixed assistant voice or few voices for the agent side (Moshi: 98.7% voice consistency from a fixed-voice instruction set). Speaker-consistent agent channel. |
| **Silence dominates duplex frames** | Voxtral down-weights silent frames (zero for long silences); Raon weights SIL 0.5 → 0.25, PAD 0.75; VoiceChat PAD 0.5–1.0 vs boundary ×10. | Down-weight silence/PAD losses; up-weight boundary and backchannel tokens. |
| **Decoding settings** | CALM: temperature 1.0 → 0.8 halved transcript perplexity (52.4 → 26.8). Pocket TTS: temp 1.0 → 0.3 improved WER and UTMOS; CFG 1.5 removed >¼ of word errors [web]. Moshi: temperature trades content for duplex dynamics. Nari vs Baseten on the same Qwen3-TTS weights: WER 3.8% vs 6.0% [web]. | Sweep temperature, top-k per codebook and CFG before retraining (D6). |
| **Codec ceiling** | SyncLLM: resynthesized GT 3.87 vs real 4.96 MOS. Qwen 12Hz recon on clean speech is strong (UTMOS 4.16, SIM 0.95), and Breeze-TTS-2 reaches top open-weight naturalness with it [web]. | Measure the resynthesis topline on *our* data (D1). The tokenizer is unlikely to be the main ceiling. |

---

## 3. Weak semantic knowledge: causes and levers

### 3.1 The S2T → S2S gap is universal; text paths shrink it

| System | Text / S2T | Speech output |
|---|---|---|
| Moshi Audio TriviaQA | Helium text 56.4 | 22.8 (7.3 without inner monologue) |
| GLM-4-Voice TriviaQA | S→T 39.1 | S→S 26.5 |
| Kimi-Audio base SpeechMMLU | S2T 67.9 | T2S 0.0, S2S 11.8 |
| MiMo-Audio base SpeechMMLU | T2T 72.5 | S2S 69.1 (best-in-class gap) |
| MinMo spoken QA | — | S2S 11–15 pts below S2T |

Systems that keep the LLM in text close most of the gap: Freeze-Omni (frozen LLM) has spoken QA ≈ text QA on the output side, and VoiceChat and MiniCPM-o have the LLM emit text only.

### 3.2 Where knowledge-bearing speech data comes from

| Recipe | How | Scale |
|---|---|---|
| Synthetic interleaved (GLM-4-Voice) | text corpora → text-to-token model → speech tokens, interleaved with text | 455B speech + 279B text tokens |
| Pseudo-dialogue CPT (VoiceChat) | split text passages into alternating user/agent sentences (80% one-sentence turns, switch at 200 words); TTS each role in two channels; loss on agent text | 95% of CPT |
| Oral instruction dialogues (Moshi) | LLM writes oral-style dialogues from Wikipedia/StackExchange plus voice-style requests, false premises, math/trivia, refusals; multi-stream TTS; fixed Moshi voice; user channel noise and reverb | >20K h |
| Duplex QA rewrite (Raon) | Qwen3 dialogues + Qwen3-TTS; SpeechLM QA rewritten into duplex form; backchannel timing predicted by a model | 106K h |
| Knowledge-dense dialogues (Chrono) | Qwen2.5-72B scripts seeded from Wikipedia/QA; voice-cloned with 50K+ prompts | 15.2K h → Llama Q ~31 at 1.5B |
| Interleaving granularity (SyncLLM) | sentence-level beats turn-level | StoryCloze 61.8 vs 55.1 |

**For us:** Qwen3-TTS emits *exactly our tokenizer's tokens*. That means we can synthesize knowledge-bearing speech in token space at scale without waveforms. Check whether TTS-generated codes match encoder-produced codes statistically; decode→re-encode a slice if not.

### 3.3 Keeping Qwen3's knowledge (see `training-recipes.md` §3)

Levers, roughly in order of cost:

1. Text replay at 30–50% (Moshi's 50% used a separate optimizer state for text batches). Even ~0.5% replay of earlier stages recovers most forgetting (Hsiao et al.). LoRA alone does **not** protect.
2. Low backbone LR.
3. A frozen-LLM alignment stage (Audex, StepAudio 3 Gen, Step-Audio 2, Covo, MOSS-Speech).
4. Text-only loss on a retention bucket during SFT (VoiceChat: 8.5% text, λ = 0.5).
5. Distillation to the frozen text model:
   - **on-policy distillation** from the text teacher (X-OPD: speech gap 11.3% → 3.4% with 27.8K prompts; SFT widened it to 22.8%);
   - on-policy KL (Raon);
   - top-20-logit KL (Covo);
   - TextQA targets regenerated by the backbone (Raon, Freeze-Omni).
6. Weight merging with a text-heavy teacher (StepAudio 3 Realtime; Fun-Audio-Chat [web]).
7. Modality-split top layers (MOSS-Speech, Lychee-FD [web]).

Track text benchmarks after **every** stage: Moshi lost almost all its MMLU in the first audio stage.

### 3.4 Raising the ceiling beyond what the frontend knows

| System | Mechanism | Effect |
|---|---|---|
| Chrono | think while *listening* (replace trailing silence with a typed reasoning chain) | +8.75% reasoning GPT score; ~0 factual gain; +0.23 s latency |
| StepAudio 3 Realtime | think while *speaking* (two concurrent calls) | Knowledge +5.94 with thinking; realtime mode costs instruction following −12.2 |
| GPT-Live, Gemini Extended Thinking, Thinking Machines | async text backend | τ-Voice and Big Bench Audio jumps listed in TL;DR 8 [web] |
| MoshiRAG | async retrieval | TriviaQA 22.8 → 78.2 [web] |
| "Can Speech LLMs Think while Listening?" (arXiv 2510.07497) | streaming ASR + CoT on the text channel of Moshi | ARC-Easy 30.2 → 77.7 [web] |

---

## 4. Full-duplex: mechanisms, data, post-training

### 4.1 Mechanism families

| Family | Examples | Latency | Notes |
|---|---|---|---|
| **Parallel streams per frame** (+ inner monologue + depth head) | Moshi, PersonaPlex, J-Moshi, F-Actor, MultiTalk, FLM-Audio | lowest (PersonaPlex 170 ms smooth-turn) | Native overlap and backchannels; needs the big foundation stage for content |
| **Single AR sequence, interleaved per frame/block, with state tokens** | Raon (per 80 ms), StepAudio 3 Realtime (320 ms blocks), Covo (0.16 s, THINK/SHIFT/BREAK), BayLing (0.8 s), DuplexOmni (480 ms), AdaptDuplex (0.48–0.96 s adaptive) | grows with block size | Explicit decisions; easy to bias at runtime (AdaptDuplex logits bias) |
| **Text-only duplex LLM + streaming TTS** | VoiceChat (+ VoiceChat-TTS), MiniCPM-o 4.5 (1 s chunks), Chrono | 448 ms (VoiceChat) to ~1 s (MiniCPM) | Best knowledge retention; prosody not user-reactive; MiniCPM chunk size moves MMLU (1 s 0.65 vs 0.1 s 0.32) |
| **Half-duplex model + controller / VAD** | MinMo, Freeze-Omni, FlexDuo, SoulX-Duplug, Gemini Live (server VAD, ~800 ms default) | ~600–800 ms+ | Knowledge-safe; weak backchannels |

### 4.2 Duplex data recipes

- **Real two-channel corpora:**
  - Fisher (~2K h; Moshi, SyncLLM, FlexDuo, MinMo, PersonaPlex);
  - Seamless Interaction (1,300 h improvised + 2,700 h naturalistic; Kyutai RL, HiPLEX, DuplexPO);
  - CANDOR;
  - AliMeeting (MinMo).
- **Pseudo-duplex from mono:** diarize single-channel audio and put each speaker on a channel (Moshi). This gives no real overlap.
- **Synthetic duplex:**
  - turn-based TTS dialogues placed on a two-channel timeline with sampled gaps (VoiceChat, Raon, Chrono);
  - stitching with negative silence to simulate barge-in, or Dia joint rendering for natural timing (PersonaPlex);
  - backchannel timing from a prediction model (Raon);
  - half-duplex dialogues converted to dual channel with inserted barge-ins and backchannels (Covo).
- **Online augmentation** (VoiceChat):
  - early interruption (p = 0.1, 640 ms overlap);
  - backchannel clip injection (p = 0.05 per sample, 0.5 per agent turn);
  - 160 ms text delay;
  - noise / RIR / microphone / codec augmentation on the user channel.
- **Labels from real data:**
  - merge VAD gaps < 160 ms; segments fully overlapped by the other speaker count as backchannels; an LLM filters third-party speech (FlexDuo);
  - turn end = assistant start; barge-in gap ~N(0.6, 0.4²) (MinMo).
- **Real vs synthetic is never ablated.** Raon's duplex data is 89% synthetic; StepAudio 3 Realtime's is all synthetic. Real-data RL (Seamless) beat Fisher for multi-turn behavior (Kyutai).

### 4.3 RL and preference for duplex

| Method | Recipe | Result / caveat |
|---|---|---|
| **Kyutai multi-faceted alignment** | GRPO, 16 rollouts, LR 2e-7, KL 0.01; rule rewards from VAD on rollout audio (pause, latency, backchannel F1) + LLM-judge content reward (ASR → Qwen3-235B) | Removing the judge costs the most (interruption relevance 3.58 → 3.05) |
| **DuplexPO** [web] | separate "when to speak" policy; factorized timing reward; GRPO on 67.7K Fisher/Seamless windows | GRPO > DPO; semantics preserved (Llama Q 72.0 → 75.3) |
| **HiPLEX** [web] | separate timing and content rewards on Moshi/PersonaPlex | — |
| **AdaptDuplex** [web] | GRPO after a staged SFT curriculum on Qwen3-Omni | beats DuplexOmni on 18/21 FDB v1/v1.5 metrics |
| **BayLing** [web] | DPO with timing-perturbed negatives | — |
| **Seed LiveInterpret 2.0** [web] | latency reward | reward hacking: shortened output ~35% |

### 4.4 Evaluation caveats

- FDB numbers are not comparable across papers. Raon uses a 1.5 s / 5-word threshold and a GPT-5.2 judge; AA reports a weighted subset ("Conversational Dynamics"). Run the official scripts ourselves.
- The behavior metrics (pause, turn-take, interruption, backchannel) are saturating among top systems (StepAudio 3 RT 98.9; Gemini 3.8 Live 96.1; Grok 95.1). The differentiator is content under duplex: FDB v2 instruction following, τ-Voice, Big Bench Audio, and knowledge in *speech output*.

---

## 5. Frontier proprietary systems: what is public

| System | Stated architecture | Reasoning | Public numbers [web] |
|---|---|---|---|
| **GPT-Live-1** (OpenAI, Sep 2026) | full-duplex S2S frontend; "interaction decisions many times per second"; delegation to a text backend (no audio sent; ≤500-token text results, paraphrased); 128K context | backend delegation | FDB v1.5 interactivity 80.1%; FDB v1 latency 0.798 s; τ-Voice 74.5–86.2% delegated; Big Bench Audio 90.1%; τ³ latency 2.54 s. Training: essentially undisclosed. |
| **Gemini 3.8 Live** (Google, Sep 2026) | "based on Gemini 3 Pro"; server VAD (~800 ms) + barge-in truncation + "proactive audio" gate; no stated listen-while-speaking | Extended Thinking variant with async tools and fillers | AA Conversational Dynamics 96.1%; Extended Thinking τ-Voice 68.6%, S2S index 82.6, TTFA 1.35 s; plain τ-Voice 30.1% |
| **Grok Voice Think Fast 2.0** (xAI, Jul 2026) | own VAD, tokenizer and audio models "from scratch"; reasons "in parallel with speech"; "extensive RL" | parallel reasoning | AA index 82.9; Big Bench Audio 97.2; FDB 95.1 (v1.0: 77.8); τ-Voice 56.5; TTFA 0.70 s |
| **TML-Interaction-Small** (Thinking Machines, May 2026) | 276B MoE / 12B active from scratch; dMel in, flow head out; 200 ms micro-turns | async background model | self-reported: FDB v1.5 77.8; latency 0.40 s; Big Bench Audio 75.7 → 96.5 with background |
| **Amazon Nova 2 Sonic** | encoder → internal text → speech renderer; pretrain + SFT + RL + preference | — | AA Big Bench Audio 88%, TTFA 1.14 s |
| **StepAudio 3 Realtime** (open report) | 196B/11B MoE; AuT on both streams; 320 ms blocks + state tokens; separate generator | think-while-speaking; async agent backend | AA FD subset 98.9; τ-Voice 56.0; MMSU 90.6 |

**Common pattern.**
- A fast interaction model owns timing, prosody and floor control.
- Reasoning and tools run asynchronously, in text, and their results are injected back.
- Heavy post-training drives turn-taking quality.
- Nobody discloses data scale or compute for the frontier systems.

---

## 6. What this means for the Qwen3 stack specifically

- **Tokenizer** (assuming Qwen-TTS-Tokenizer-12Hz; see `papers/qwen3-tts.md`):
  - 12.5 Hz (80 ms), 16 codebooks × 2048.
  - cb0 is WavLM-guided semantic; cb1–15 are acoustic RVQ.
  - Fully causal, no lookahead, decode 4–5 ms per 320 ms packet, so it is fine for duplex.
  - **The decoder takes no speaker vector**, so the LM owns timbre.
  - The 25 Hz variant is not released and its decoder adds ~450 ms of lookahead, so it is unsuitable.
- **The same token shape is used by** Qwen3-Omni's output codec (config-identical in codebook and upsampling params; interchangeability unverified), StepAudio 3 Gen, and Breeze-TTS-2. Those recipes transfer almost directly:
  - backbone predicts cb0 in the same softmax as text;
  - a 4–12-layer depth/MTP head predicts cb1–15;
  - input is the sum of codebook embeddings plus a zero-init adaptor;
  - 1 text token per 2 frames (StepAudio 3 Gen) or a time-aligned text stream.
- **Pretrained pieces we can reuse:**
  - Qwen3-TTS 0.6B/1.7B: talker + 5-layer MTP, >5M h. Usable as a depth-head init, a data synthesizer, or a variant-B renderer.
  - Qwen3-Omni-30B-A3B: AuT encoder at 12.5 Hz for the user stream (StepAudio 3 Realtime does this); Talker + MTP + Code2Wav; a speech-pretrained Thinker.
  - Raon warm-started its depth transformer from Qwen3-Omni.
- **Strategic fork:**

  | Path | What it means | Precedents |
  |---|---|---|
  | (i) Keep the bare Qwen3 LLM | Rebuild Stages 1–2 ourselves (10⁵–10⁶ h, cheap compute, data-bound) | — |
  | (ii) Convert Qwen3-Omni to duplex | It already carries 0.77T audio tokens of pretraining and a trained Talker | AdaptDuplex, DuplexOmni [web] |

  Path (ii) skips the most expensive missing stage, but commits us to a 30B-A3B MoE and Qwen3-Omni's design.

---

## 7. Where the evidence is thin

- **No controlled data-scaling study for duplex models** (hours vs continuation or knowledge). MiMo's ~0.7T-token emergence is for few-shot / in-context ability. Moshi, CALM and Raon report one scale each.
- **Synthetic vs real duplex data ratio** is never ablated.
- **Compute** is almost never reported (exceptions: PersonaPlex 48 A100-h; Kyutai RL 32 H100; Audex 512 H100; DSM/CALM GPU counts).
- **Several frontier claims are self-reported** (Thinking Machines, StepAudio 3 Realtime on its own benchmarks). Artificial Analysis is the main independent source.
- **Key implementation numbers are missing** in some priority papers. VoiceChat's CPT bucket size has no unit; StepAudio 3 Realtime doesn't describe its generator or duplex data synthesis.
- **No head-to-head study of word-level vs sentence-level interleaving in pretraining** was found. SyncLLM reports sentence > turn level only.

---

## 8. Recommendations

See `plan.md` for the staged plan, data buckets, compute estimates, evaluation suite and ranked experiment backlog. In one paragraph:

1. **Diagnose first** (codec topline, per-codebook loss, T2T/S2T/T2S/S2S matrix, forgetting, decoding sweep, data audit).
2. **Fix the model.** Add or repair a time-aligned text stream. Restructure the output so the backbone predicts text + cb0 and a Qwen3-initialized depth/MTP head predicts cb1–15.
3. **Build the missing speech–text stage** at 10⁵–10⁶ h with 30–50% text replay, much of it synthetic knowledge speech rendered by Qwen3-TTS in token space. Alternatively, start from Qwen3-Omni.
4. **Use our 10K h as the duplex-behavior layer**, mixed with synthetic knowledge-dense duplex dialogues.
5. **Post-train:** curated SFT, then speech-quality DPO and timing+content GRPO.
6. **Design an async text/tool path** from the start.
