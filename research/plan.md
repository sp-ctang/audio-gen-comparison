# Training plan v0 (draft)

**Status.** Draft, built from the literature review only. Revise it once the current stack is documented (§0). The evidence behind each item is in `synthesis.md` and `training-recipes.md`.

---

## 0. Intake: what we need to know about the current stack

The plan branches on these answers.

1. **Components.**
   - Which Qwen3 LLM: size, base or instruct, dense or MoE?
   - Which tokenizer: Qwen-TTS-Tokenizer-12Hz (12.5 Hz × 16 × 2048), the Qwen3-Omni codec, or another?
   - Which decoder: the tokenizer's own decoder or Qwen3-Omni Code2Wav?
2. **Sequence layout.**
   - How are the two channels represented: parallel streams with summed embeddings per 80 ms frame (Moshi-style), or time-division interleaving?
   - How many backbone positions per second?
   - How are the 16 codebooks predicted: all from the backbone, a delay pattern, a depth transformer, or an MTP head?
3. **Text channel.** Is there a time-aligned agent text stream (inner monologue)? If so:
   - How is it aligned: word timestamps, or padding?
   - What is its delay relative to audio?
   - What is the loss weight on text vs cb0 vs cb1–15?
4. **User input path.** Is the user channel fed as codec tokens or through a continuous encoder (AuT / Whisper)?
5. **The 10K h.**
   - Sources, languages and licences.
   - Real or synthetic.
   - How channels were separated (true stereo vs diarized mono).
   - Transcripts and word timestamps.
   - Which side is the "assistant", and is it one voice or many?
   - Noise and crosstalk levels.
   - Topical content: chit-chat, or information-bearing?
6. **Training config.**
   - Full fine-tune or LoRA.
   - LR, schedule, batch (hours per step), steps and epochs, sequence length.
   - Text replay?
   - Which modules were frozen?
7. **Symptoms and how they were measured.**
   - What does "weak continuation" look like: babble, collapse to silence, robotic prosody, speaker drift, noise?
   - What does "weak knowledge" look like: wrong facts, generic answers, incoherence?
   - Which evals were used?
8. **Budgets and targets.**
   - GPUs and time.
   - Can we acquire or synthesize 100K–1M h? Are there licensing constraints (commercial use)?
   - Languages, latency target, voices/persona, tools/RAG, on-device vs server.

---

## 1. Diagnostics on the current checkpoint (≈1–2 weeks, cheap)

Run these before changing anything; each one points to a different fix.

| # | Diagnostic | What it tells us | Literature anchor |
|---|---|---|---|
| D1 | **Codec topline.** Encode→decode a held-out slice of the 10K h (both channels) and measure WER, UTMOS and speaker similarity vs the original. | Upper bound on audio quality; whether the agent channel is clean enough to learn from. The Qwen 12Hz decoder takes no speaker prompt, so the LM must also model noise and room. | SyncLLM: resynthesized ground truth 3.87 vs real 4.96 MOS. Qwen 12Hz recon: UTMOS 4.16, SIM 0.95 on clean LibriSpeech. |
| D2 | **Per-codebook teacher-forced loss** (cb0 vs cb1–15, text) and its trend over training. | Where model capacity and loss go; whether cb0 (semantic) is actually being learned. | Moshi: semantic loss weight ×100; per-codebook depth weights. |
| D3 | **Knowledge matrix: T2T / S2T / T2S / S2S** on the same items (e.g. a SpeechMMLU-style set and spoken TriviaQA / Llama Q / Web Q). Also score the text channel vs ASR of the speech output, if a text channel exists. | Is knowledge lost in the backbone (T2T↓), in perception (S2T↓), or in speech rendering (S2S ≪ S2T)? | MiMo T2T 72.5 vs S2S 69.1; Kimi base T2S 0.0; MinMo S2T−S2S 11–15 pts. |
| D4 | **Backbone forgetting.** Prompt the mid-trained checkpoint text-only on MMLU, GSM8K and IFEval vs the original Qwen3. | Whether 10K h of audio-only training damaged the LLM. | Moshi MMLU 54.3 → 49.7, mostly during audio pretraining. |
| D5 | **Continuation probe.** 3 s prompt → 20–30 s continuation. Measure ASR transcript NLL under a text LLM, length (collapse to silence?) and speaker similarity drift. | A direct, cheap continuation metric. | Moshi Tab.6 protocol. |
| D6 | **Decoding sweep:** temperature 0.6–1.0, per-codebook top-k, CFG if available. | Decode settings can rival retraining. | CALM: temperature 1.0 → 0.8 halved transcript perplexity. Pocket TTS: guidance removed >¼ of word errors. |
| D7 | **Data audit** on a sample of the 10K h: crosstalk and bleed between channels, agent-channel speaker consistency, silence ratio, noise (DNSMOS), topic distribution. | Polluted context and multi-speaker agent channels hurt both content and acoustics. | FlexDuo: filtering noise and backchannels cut conditional perplexity 64.3 → 28.9. MOSS-TTS / community replication: 300 clean h > 1,500 noisy h. |

---

## 2. Architecture decision

Pick one variant after D1–D4. All three can share the same data plan.

| Variant | Description | Pros | Cons | Closest precedents |
|---|---|---|---|---|
| **A. Moshi-style joint model, with acoustics moved off the backbone** | Parallel user and agent streams per 80 ms frame. The backbone predicts the **agent text token (inner monologue) + cb0**. A small depth transformer or MTP predicts cb1–15. Inputs: summed codebook embeddings plus a zero-init RVQ adaptor. The user stream optionally comes through AuT continuous features. | Single model; reactive prosody; native backchannels and overlap; lowest latency (PersonaPlex 170 ms) | Needs the large speech-text stage to sound good; knowledge still bounded by the S2S gap | Moshi, PersonaPlex, Raon, Qwen3-Omni Talker+MTP, StepAudio 3 Gen, Sesame |
| **B. Duplex LLM outputs text and state only; a separate streaming duplex TTS speaks** | The LLM sees user audio (AuT/encoder) and its own text. It emits text with BOS/EOS/PAD (+ BC) per frame. A Qwen3-TTS-12Hz-derived streaming decoder, made duplex-aware, renders speech. | Best knowledge retention (VoiceChat VoiceBench 55 vs ~30); continuation quality inherited from a >5M h TTS; fastest path | Output prosody does not hear the user; extra latency (VoiceChat 448 ms); the TTS needs duplex adaptation (silence-prepend, interrupt token) | NemotronLabs VoiceChat (+ VoiceChat-TTS), MiniCPM-o 4.5, Chrono-Thinking, StepAudio 3 Realtime (separate generator), Nova Sonic |
| **C. A or B + async reasoning/tool backend** | The duplex frontend can delegate to a stronger text LLM; results are injected as text and paraphrased. | Raises the knowledge ceiling regardless of frontend size (τ-Voice 30–46% → 74–86%) | Latency on delegated turns; more infrastructure | GPT-Live, Gemini 3.8 Live Extended Thinking, TML background model, MoshiRAG |
| **D. Convert Qwen3-Omni to duplex instead of starting from the bare LLM** | Start from Qwen3-Omni-30B-A3B: Thinker, plus Talker + MTP + Code2Wav on a 12.5 Hz × 16 × 2048 codec, plus AuT input. Add a duplex token protocol and train on synthetic + real duplex dialogues. | Skips the missing speech–text foundation stage (Qwen3-Omni already saw ~0.77T audio tokens; its Talker saw "hundreds of millions" of samples) | Commits us to a 30B-A3B MoE and to Qwen3-Omni's Thinker–Talker design; its codec's interchangeability with Qwen-TTS-Tokenizer-12Hz is unverified | AdaptDuplex (Qwen3-Omni + 3-stage Thinker curriculum + Talker SFT + GRPO), DuplexOmni (480 ms slices, async thinker) [web] |

**Current recommendation (to revisit after intake): A with B's discipline.**

- Make the backbone's job text + cb0.
- Give acoustics to a small head initialized from a Qwen3 MTP (Qwen3-TTS or Qwen3-Omni: both 5 layers, hidden 1024; verify shapes).
- Keep B as a fallback. The B route is also a strong baseline to build first: it ships quickly, and it doubles as a teacher for A's text channel and as a source of synthetic data.
- Design the text-injection interface for C from the start.
- **Seriously consider D** if a 30B-A3B MoE is acceptable. It removes the most expensive missing stage, which is exactly the one our symptoms point to.
- Decide between "A on bare Qwen3" and "D" after intake: model-size and latency targets, and whether the current tokenizer is the Qwen3-Omni codec or Qwen-TTS-Tokenizer-12Hz.

---

## 3. Data plan

| Bucket | Purpose | Target scale | How to build | Notes |
|---|---|---|---|---|
| **S1. Single-stream speech + word timestamps** | Continuation quality; speech–text alignment | 100K–1M h | Public corpora plus podcast/YouTube-style conversational speech. ASR with word timestamps (WhisperX / whisper-timestamped / NeMo aligners). Quality filters (DNSMOS, single-speaker segments for the agent side) and dedup. | Moshi 7M h; Raon 1.01M h; MiMo emergence near 0.7T tokens. DSM: pseudo-labels are fine. Dedup removes regurgitation (Moshi). |
| **S2. Diarization-simulated two-stream** | Cheap pseudo-duplex from S1 | subset of S1 | Diarize multi-speaker audio; put one speaker per channel (Moshi). | No real overlap, and the inactive channel is silent, so real duplex data is still needed after. |
| **S3. Synthetic knowledge-rich speech** | Knowledge transfer into speech | 10⁴–10⁵ h-equivalent (more is cheap) | Render text corpora with Qwen3-TTS (12Hz) **directly in token space**; optionally decode→re-encode a subset to match encoder statistics. Three formats: (a) interleaved text/speech at sentence level; (b) VoiceChat-style pseudo-dialogues (sentences alternate user/agent, 80% one-sentence turns, role switch at 200 words); (c) Moshi/Raon-style oral dialogues (LLM-written from Wikipedia/QA/instruction data, fixed assistant voice, false premises, clarifications, refusals). | GLM-4-Voice: 455B synthetic speech tokens was the main knowledge lever. VoiceChat CPT is ~all rendered text. Generation cost is small (order tens of H100-h per 100K h in token space; estimate, to be measured). **Check licences** of the TTS weights and text sources. |
| **S4. Real duplex (our 10K h)** | Natural timing, overlap, backchannels, prosody | 10K h | Channel-clean, transcribed with word timestamps, labelled with turn states (FlexDuo-style rules: merge gaps <160 ms; fully overlapped short segments = backchannels). | Comparable to Moshi's Fisher (2K h) + PersonaPlex/Raon real (1.2–13K h). Treat it as the behavior bucket. |
| **S5. Synthetic duplex dialogues** | Turn-taking with knowledge-bearing content | 10K–100K h | Script with an LLM, render each side with Qwen3-TTS (random user voices, fixed agent voice), place on a two-channel timeline. Augment: early interruption (p≈0.1, 640 ms overlap), backchannel injection, ± silence gaps, noise/RIR on the user channel only. | StepAudio 3 RT >10K h synthetic; Raon 106K h synthetic; VoiceChat 72K h; Chrono 15K h. Use an LLM-judge causality check (Multi-Stream LLMs). |
| **T. Text replay** | Preserve Qwen3 knowledge | 30–50% of tokens in speech-text stages; ~10% in duplex stages | General pretraining-style text plus instruction data; targets regenerated by the Qwen3 backbone where applicable (Raon, Freeze-Omni). | Moshi, MiMo, Kimi, Step-Audio 2, StepAudio 3 Gen ~50%; Covo 40%; Qwen3-Omni ~25%; Audex 44–69%. |

---

## 4. Stage plan (v0)

Token counts assume one backbone position per 80 ms frame (45k positions per hour of audio). Compute is a rough one-epoch estimate for a Qwen3-8B backbone at 40% MFU on H100: audio + 50% text replay + ~15% for the depth/MTP head. Expect real efficiency to be 1.5–3× worse.

| Stage | Goal | Trainable | Data mix | Scale | ~Compute (8B) |
|---|---|---|---|---|---|
| **0. Head warm-up** | Teach cb1–15 prediction and the input adaptor without touching the LLM | Depth/MTP head + adaptor + new embeddings; **LLM frozen** | S1 (TTS-style: text → audio), S3 | 10–50K h | tens of H100-h |
| **1. Speech–text mid-training** | Continuation and knowledge transfer | All; backbone LR ~1–3e-5, new modules ~1e-4–2e-4 | S1 (ASR, TTS, interleaved, continuation, audio-only) + S3 (a, b) + **T at 40–50%** | 100K → 1M h | ~360 → ~3,600 H100-h |
| **2. Duplex adaptation** | Two-stream format, turn-taking, overlap | All, low LR | Curriculum: turn-based → basic duplex → full duplex (AdaptDuplex). S2 + S4 + S5 + 10–20% replay of Stage 1 + T ~10%. Never S4 alone: human–human data alone dropped VoiceBench to 18 vs 65 mixed (Li & Chu). | 20–100K h | ~70–360 H100-h |
| **3. SFT** | Assistant behavior, persona/voice, safety, (tools) | All, very low LR | Curated (quality > volume), knowledge-dense duplex dialogues, retention buckets (~40–50% of sampling), safety | ~5–20K h equivalent | small |
| **4. Preference / RL** | Timing and content polish; speech quality | All, LR ~1e-7–1e-6, KL to SFT | (a) GRPO on VAD-based timing rewards **always paired with** an LLM-judge content reward, on segments mined from S4 (Kyutai, DuplexPO, HiPLEX); (b) DPO on speech pairs ranked by WER/UTMOS/SIM (Qwen2.5-Omni, Voxtral); (c) content DPO on the **text stream only** (Wu et al.: audio tokens in the loss hurt) | thousands of segments | 32 H100 × days |
| **5. Recovery / gap closing** | Restore text ability; close the S2T→S2S gap | All, low LR | **On-policy distillation from the Qwen3 text teacher** (X-OPD: 27.8K prompts closed the speech gap 11.3% → 3.4%; SFT widened it); on-policy KL on the text channel (Raon); weight-merge with a text-heavy teacher (StepAudio 3) | ~10⁴–10⁵ prompts | small |

Loss-design defaults to start from:

- **Codebook weights.** Up-weight cb0 relative to cb1–15 (Moshi used ×100 on the semantic token). Use per-codebook weights in the depth head. MOSS-TTS up-weights coarse levels 3,3,3,2,2,2.
- **Delays.** 1–2 frame acoustic delay. Text leads audio by 1–2 frames (DSM, CALM, Raon, F-Actor); sweep it.
- **Silence and PAD.** Down-weight them (PAD 0.5–0.75, SIL 0.25–0.5; Raon, Voxtral). Up-weight boundary and backchannel tokens: BOS/EOS ~10× (VoiceChat), BC ~×50 (Raon).
- **Text masking.** Mask the text stream ~30% of the time and use random ±0.6 s text–audio delay in pretraining (Moshi), so the model learns to speak without perfect text.

---

## 5. Evaluation suite

| Axis | Metrics |
|---|---|
| Codec / audio | Resynthesis topline (D1); WER (Whisper / Parakeet), UTMOS, speaker similarity across turns |
| Continuation | Moshi-style transcript NLL + length from 3 s prompts; sStoryCloze / tStoryCloze; speaker drift |
| Knowledge | T2T / S2T / T2S / S2S matrix; spoken TriviaQA, Llama Q, Web Q; VoiceBench (per subset); Big Bench Audio |
| Backbone retention | MMLU, GSM8K, IFEval via text-only prompts after every stage |
| Duplex | FDB v1.0 / v1.5 with the **official** scripts (pause, smooth turn, interruption, backchannel, latency); FDB v2 instruction following; optionally FDB v3 / τ-Voice if tools are in scope |
| Robustness | Noisy and accented user audio; background speech |

---

## 6. Experiment backlog (ranked by expected value per cost)

1. **D1–D7 diagnostics.** These decide everything downstream.
2. **Decoding sweep (D6)** on the current checkpoint: free.
3. **Add or repair the inner-monologue text stream** with DSM-style word-onset alignment; ablate text lead 0/1/2 frames. This is the single highest-value change at our data scale. At a matched 10K h, a text stream cut continuation GenPPL 183 → 53, vs 149 for 45K h speech-only (Chou et al.); Moshi: spoken QA ×3, continuation length ×3.2.
4. **Restructure the output head:** backbone predicts text + cb0; depth/MTP head (init from a Qwen3 MTP) predicts cb1–15; cb0 up-weighted; zero-init RVQ adaptor. Run a small Stage 0 + Stage 1 at 10–50K h to compare against the current setup.
5. **Build S3 synthetic knowledge speech with Qwen3-TTS in token space.** Ablate 0 vs 10K vs 100K h-equivalent for its effect on S2S knowledge.
6. **Text replay ratio** 0 / 25 / 50% at fixed audio tokens, tracking MMLU and S2S QA.
7. **Scale S1** from 10K → 100K → (1M) h and plot continuation and knowledge vs hours, to find our own threshold. MiMo saw a jump near 0.7T tokens.
8. **Variant B baseline:** text-only duplex LLM + Qwen3-TTS-12Hz renderer. A cheap head-to-head and a fallback.
8b. **Variant D spike:** wrap Qwen3-Omni in a block-interleaved duplex protocol (AdaptDuplex/DuplexOmni-style) and fine-tune on our 10K h + synthetic duplex. Compare continuation and knowledge against the bare-Qwen3 route at equal duplex data.
9. **Duplex stage mix:** real-only vs real + synthetic (S5) vs + diarized (S2).
10. **GRPO timing + content rewards** after SFT.
10b. **On-policy distillation (X-OPD-style)** from the Qwen3 text teacher, on the agent text channel. It is cheap (~10⁴ prompts) and can be tried on the *current* checkpoint as soon as a text channel exists.
11. **Async backend (variant C)** prototype once the frontend is stable.
