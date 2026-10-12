# Chronological Thinking in Full-Duplex Spoken Dialogue Language Models (CT-Duplex)

| Field | Value |
|---|---|
| Org | Nanyang Technological University, StepFun and Mila (p.1) |
| Date / arXiv | arXiv:2510.05150v2, 8 Oct 2025 (p.1). StepAudio 3 Realtime cites it as SIGDIAL 2026 (SA3-RT ref [11]). |
| Category | full-duplex S2S (method: thinking while listening) |
| Backbone | Qwen2.5-1.5B-Instruct (p.7) |
| Audio input | A streaming speech encoder at 12.5 Hz gives continuous embeddings. A modality adapter projects them, and they are **summed** with the agent-text-token embedding at each step (pp.3–4). The encoder "follows" Hu et al. 2025 (SALM-Duplex) and is not named here (p.7). The LLM does not receive the agent's speech tokens (eq.1, Fig.2). |
| Audio output | NanoCodec (FSQ) speech tokens at 12.5 Hz (p.4). Codebook count not stated; Fig.2 schematically shows several audio channels. They are predicted by a separate **autoregressive Transformer decoder** conditioned on the LLM's last hidden state, the agent text tokens and previous speech tokens (eq.2, p.4). **The LLM predicts only agent text.** |
| Waveform decoder | NanoCodec decoder (implied, not described) |
| Duplex mechanism | Single-model dual stream in the Moshi / SALM-Duplex style. At every 12.5 Hz step the LLM sees user-speech plus agent-text embeddings and predicts the agent text channel: `<SIL>` while listening, then `<BOS>`, response, `<PAD>`… `<EOS>`. CT replaces the *trailing* `<SIL>`s of each user turn with `<BOC>` + thinking chain + `<EOC>` (pp.4–6). |
| Open weights / code | Not stated; built with NVIDIA NeMo speechlm2 (p.7). |
| Relevance to us | **Medium-high.** A small duplex setup (1.5B model, 15.2K h synthetic data) at a data scale like ours. It shows that LLM-predicts-text-only with a separate audio decoder **doubles** knowledge-QA accuracy over joint prediction. It also shows that listening windows can carry useful "thinking" tokens at zero extra decoding steps. |

## TL;DR

- **Problem.** Duplex models usually predict `<SIL>` repeatedly while the user speaks. This wastes the listening window and repetitive prediction can bias decoding (p.2).
- **Method: Chronological Thinking (CT).** The idea comes from the ACT-R cognitive architecture. CT is a compact chain of typed nodes, each written as `{TYPE} attributes`:
  - ENTITY, INTENT, ACTION, KNOWLEDGE, LOGIC;
  - with `@ctx:ID` entity binding across turns.

  Chains are generated offline by Qwen2.5-72B-Instruct from the dialogue data. At training time they replace the *last* M+2 silence tokens before the agent's turn, and chains that are too long are truncated at node boundaries (pp.5–6, Tab.1).
- **Architecture choice (the bigger lesson).** The LLM predicts only text, and a separate AR Transformer decoder predicts speech tokens from hidden states. The same architecture with joint text+audio prediction (SALM-Duplex) does much worse (Tabs.3–5):

  | Metric | SALM-Duplex | CT-Duplex w/o thinking |
  |---|---|---|
  | Llama Questions | 15.0 | **30.4** |
  | Web Questions | 6.7 | **13.2** |
  | SpokenWOZ GPT score | 2.11 | **2.40** |
  | Turn-taking latency (s) | 0.92 | **0.45** |

- **Effect of CT** (Tabs.3–5, Fig.3):
  - SpokenWOZ GPT score 2.40 → 2.61 (+8.75%);
  - MtBenchEval 2.39 → 2.44 (+2.09%);
  - factual QA roughly unchanged (Llama Q 30.4 → 31.4, Web Q 13.2 → 13.3);
  - barge-in success 88.63% → 94.05%;
  - turn-taking latency 0.45 → 0.68 s;
  - A/B tests: content preferred 55% vs 27.5%, audio fidelity preferred 63.75% vs 17.5%.
- **Data:** 15.2K h of synthetic two-speaker dialogue: GenConv 10.5K, SpokenWOZ-G 2.0K, Llamaq-G 2.7K. It is synthesized with Step-Audio-TTS-3B from a library of 50K+ single-speaker prompt clips, with barge-ins injected in 50% of turns. **Compute:** 8× L40S (pp.6–7).

## Architecture

- **LLM input and objective** (eq.1). User speech goes through the streaming encoder at 12.5 Hz, then the adapter, and is *summed* with the embedding of the agent's previous text token. The LLM predicts the next agent text token Yᵗˣᵗₜ given ⟨Yᵗˣᵗ₁:ₜ₋₁, X₁:ₜ₋₁⟩ (pp.3–4).
- **Speech decoder** (eq.2). An AR Transformer predicts speech token Yˢᵖᶜₜ from the previous speech tokens, the text tokens and the LLM hidden state hₜ (p.4).
  - The stated motivation is "to reduce the prediction burden on the LLM", in contrast to Hu et al. 2025 (SALM-Duplex), where the LLM predicts both (p.4).
  - The encoder, LLM and decoder are fine-tuned jointly with a multi-channel next-token loss (p.4).
- **Text-channel alignment** (eq.3, p.4).
  - While listening, the channel is `<SIL>…<SIL>`.
  - The agent turn is `<BOS> R₁…R_T <PAD>… <EOS>`; `<PAD>` fills the gap between text and speech tokens.
  - On interruption, `<EOS>` is emitted and the model returns to `<SIL>`.
- **CT insertion** (eqs.4–5, p.6). Let M be the chain length and S the number of silence slots in the user turn.
  - **If M ≤ S−2:** the last M+2 `<SIL>`s become `<BOC> C₁…C_M <EOC>`, placing thinking "as late as possible".
  - **Otherwise:** only the first n nodes that fit are kept. In practice the chain fits in 94.7–98.9% of turns across the three datasets (Tab.6, p.14).
  - Bracketed transcript spans like `[order a restaurant]` are used *only* during data generation to anchor nodes; they never appear in the model's stream (p.5).
- **Example** (Fig.4, p.14). Over three turns the chain tracks intent, the entity "Trattoria Bella" via `@ctx`, and simple arithmetic ("me + 3 friends" → `{LOGIC} 4 people`).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
Not done here. The speech encoder, text tokenizer and speech codec "follow the ones in (Hu et al., 2025)"; the codec is NanoCodec with FSQ at 12.5 Hz (pp.4, 7).

### Stage 1: joint duplex fine-tuning (the only stage)
- **Objective:** multi-channel next-token prediction over agent text (with CT tokens) and agent speech tokens. Encoder, LLM and AR speech decoder are all trainable (p.4).
- **Init:** Qwen2.5-1.5B-Instruct (p.7).
- **Data:** 15.2K h of synthetic dialogue (Tab.2); no text-only data is mentioned.
- **Optimisation:** AdamW with inverse-square-root annealing; LR 3×10⁻⁴ with a 2,500-step warm-up (p.7).
- **Hardware:** NeMo toolkit on **8× L40S (48 GB)** (p.7).
- **Not reported:** steps, batch size, epochs, sequence length, wall-clock time.
- **No speech-text pretraining stage and no RL.** The "w/o thk" model uses the same pipeline without CT tokens.

## Data pipeline

- **Text dialogues** come from Qwen2.5-72B-Instruct (p.6). The authors argue that real conversational corpora such as Fisher are casual and "insufficient to train the SDLM to respond to diverse human inquiries" (p.6).

  | Set | Seeds | Hours |
  |---|---|---|
  | **GenConv** | Wikipedia-style topics (general knowledge, common sense, current events) | 10.5K |
  | **SpokenWOZ-G** | SpokenWOZ train set, generated in its format; outputs with >90% fuzzy similarity to the seeds are discarded | 2.0K |
  | **Llamaq-G** | the LLaMA Questions format | 2.7K |

- **Thinking chains** are generated by Qwen2.5-72B-Instruct from each dialogue (p.5).
- **Speech:**
  - Step-Audio-TTS-3B voice cloning;
  - a speaker library of over 50,000 cleaned single-speaker clips from short videos, streaming media and podcasts;
  - two distinct speakers per dialogue (pp.6–7).
- **Duplex events.** Following Hu et al. 2025, each turn has a **50% chance of a barge-in** cutting off the agent, and the agent stops speaking **0.64 s** after the barge-in starts (p.7).
  - No backchannels, overlaps beyond barge-in, or pause modeling are described.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Joint duplex fine-tune | 15.2K h synthetic (Tab.2) | 8× L40S 48 GB; time not reported | LR 3e-4, warm-up 2.5K steps; steps and batch not reported |
| Chain and dialogue generation | — | not reported | Qwen2.5-72B-Instruct |
| TTS synthesis | 15.2K h | not reported | Step-Audio-TTS-3B |

## Preserving text intelligence / knowledge

- **Mechanism: keep the LLM on text.** The LLM never predicts audio and does not read its own audio; acoustic modeling is moved to the Transformer decoder (p.4). This roughly doubles factual QA over SALM-Duplex, which differs mainly in having the LLM predict text *and* audio (Tab.4; numbers in the TL;DR).
- **Knowledge-dense data.** Knowledge-oriented synthetic data (Wikipedia-seeded GenConv; QA-format Llamaq-G) replaces casual conversation (p.6).
- **No text replay** and no text-benchmark evaluation of the fine-tuned LLM.
- **Retention vs a cascade upper bound (GT-LM).** GT-LM feeds ground-truth user text to the text LLM:
  - On MtBenchEval, GT-LM scores 3.15 vs 2.44 for CT-Duplex. The duplex S2S model keeps ≈77% of the text model's everyday-dialogue score (derived: 2.44/3.15).
  - On reasoning-heavy SpokenWOZ, CT-Duplex beats GT-LM (2.61 vs 2.48), because GT-LM does no CoT (p.8).
- **Calibration.** Factual QA stays far below the half-duplex 9B GLM-4-Voice (Llama Q 31.4 vs 50.7; Web Q 13.3 vs 15.9). The authors attribute this to model size and the absence of real-time/causal constraints (p.8).

## Key results

- **Response quality (Tab.3).** GPT score (0–5, judged by gpt-4o-mini), BLEU and Sentence-BERT similarity:

  | Model | SpokenWOZ GPT / BLEU / SBERT | MtBenchEval GPT / BLEU / SBERT |
  |---|---|---|
  | GT-LM (cascade) | 2.48 / 7.60 / 0.55 | **3.15** / **10.18** / **0.73** |
  | SALM-Duplex* | 2.11 / 8.73 / 0.34 | 2.25 / 5.31 / 0.47 |
  | CT-Duplex w/o thk | 2.40 / 12.92 / 0.52 | 2.39 / 7.00 / 0.64 |
  | CT-Duplex w/ thk | **2.61** / **16.30** / **0.59** | 2.44 / 7.34 / 0.67 |

- **Factual QA, accuracy % (Tab.4).** Baselines other than SALM-Duplex are copied from GLM-4-Voice's paper.

  | Model | Size | Llama Q | Web Q |
  |---|---|---|---|
  | Moshi | 7B | 21.0 | 9.2 |
  | GLM-4-Voice (half duplex) | 9B | 50.7 | 15.9 |
  | SALM-Duplex* | 1.5B | 15.0 | 6.7 |
  | CT-Duplex w/o thk | 1.5B | 30.4 | 13.2 |
  | CT-Duplex w/ thk | 1.5B | 31.4 | 13.3 |

- **Interaction, "impatient" set (Tab.5).** Interruptions come about every 2 s. Barge-in success = the agent stops within 1.5 s. Baselines other than SALM-Duplex are copied from Chen et al.

  | Model | Turn-taking latency (s) | Barge-in latency (s) | Barge-in success |
  |---|---|---|---|
  | CT-Duplex w/ thk | 0.68 | 0.54 | 94.05% |
  | CT-Duplex w/o thk | 0.45 | 0.53 | 88.63% |
  | SALM-Duplex* | 0.92 | 0.69 | 87.50% |
  | ORISE | 0.43 | 0.61 | 96.80% |
  | Moshi | n.a. | 0.81 | 55.10% |
  | Freeze-Omni | 1.17 | 1.20 | 79.50% |

- **Human A/B, CT vs no-CT** (Fig.3, p.9). 10 fluent English raters on SpokenWOZ items.

  | Criterion | CT wins | Tie | CT loses |
  |---|---|---|---|
  | Content quality | 0.5500 | 0.1750 | 0.2750 |
  | Audio fidelity | 0.6375 | 0.1875 | 0.1750 |

## Ablations & lessons

- **Decoupling audio prediction from the LLM is the largest effect in the paper.** CT-Duplex w/o thk vs the authors' SALM-Duplex reproduction (TL;DR table) roughly doubles factual QA and halves turn-taking latency, and MtBenchEval also improves (2.25 → 2.39). The paper does not state whether the reproduction used identical data.
- **Thinking while listening helps reasoning, not recall.**
  - +8.75% GPT score on reasoning-heavy SpokenWOZ and +3.38 BLEU (derived);
  - +2.09% on everyday dialogue;
  - +1.0 / +0.1 on factual QA, because "factual knowledge-based QA tasks require minimal reasoning" (p.8).
- **Structured nodes beat a streaming-ASR transcript as thinking content.**
  - A streaming-ASR transcript is upper-bounded by GT-LM, which loses to CT on SpokenWOZ (2.48 vs 2.61).
  - Streaming ASR also inserts padding tokens between words, which "disrupts semantic consistency and degrades SDLM's performance" (Appendix A.1, p.13).
- **Compactness matters.** Typed nodes keep chains shorter than the silence window in 94.7–98.9% of turns (Tab.6), so truncation is rare and thinking stays preemptible.
- **Thinking also improved perceived audio fidelity** (63.75% vs 17.5% A/B). The authors' explanation: lower text loss lets the model "learn more from the audio prediction loss". No loss curves support this (p.9).
- **Turn-taking latency rose 0.23 s** (derived: 0.68 − 0.45) even though the paper claims "no additional latency". Barge-in latency was unchanged and barge-in success went up by 5.4 points (derived).

## Implications for our setup

1. **Change the output factorization first.** At 1.5B and 15.2K h, close to our data scale, having the LLM predict only text, with a separate AR decoder for audio tokens conditioned on hidden states, doubled factual QA and halved turn-taking latency compared with joint prediction. If our Qwen3 LLM is predicting Qwen3 audio tokens directly, this is the best-evidenced fix for "weak semantic knowledge in generated speech". It is consistent with SA3-Gen (LLM predicts only codebook 0) and SA3-RT (text/state LLM plus a separate generator). It is also the same principle as Qwen3-Omni's Thinker–Talker split (`qwen3-omni.md`), where a Talker initialized from Qwen3-TTS-12Hz could play the role of CT-Duplex's AR speech decoder. In the duplex setting it would condition on the LLM's hidden states and text channel.
2. **Real casual conversation is not enough.** Mix in knowledge-dense synthetic dialogues: Wikipedia-seeded and QA-format, generated by an LLM and voice-cloned with Qwen3-TTS or similar using two distinct speakers. Inject barge-ins in about 50% of turns, with a 0.64 s stop delay.
3. **Put the listening window to work.** Replace `<SIL>` runs with compact, typed thinking tokens; it is nearly free at inference. Watch two things:
   - causal placement (see the open questions below);
   - turn-taking latency (+0.23 s here).

   Expect gains on reasoning or task dialogue, not on factual recall.
4. **Measure knowledge loss against a cascade upper bound.** Run the text Qwen3 on ground-truth transcripts (GT-LM); CT-Duplex keeps ~77% of it on MtBenchEval. Report Llama Q and Web Q on speech output, using Whisper ASR and an LLM judge, as here.
5. **It is cheap to replicate.** 8× L40S suffices, so this is a feasible ablation on our stack before committing to larger changes.

## Open questions / unclear points

- **Is the placement causal?** Chains are packed into the *last* M+2 slots before `<BOS>`, at 80 ms per token. A 30-token chain therefore spans 2.56 s before the response (derived: 32 × 0.08 s). Nodes about the user's final phrase may be trained to appear *before* that phrase is heard. The paper only says it is "striving to ensure" causality (p.6), so training targets may leak future content.
- **Latency claim.** "No additional latency" conflicts with the 0.45 → 0.68 s turn-taking result. The text even reports the gap as "0.20s" when it is 0.23 s (p.9).
- **Fairness of the baseline.** The SALM-Duplex reproduction's data and training budget are not stated, which matters for the headline decoupling result.
- **Missing training details:** steps, batch size, epochs, sequence length, training time, and the encoder and NanoCodec configuration (codebook count).
- **Is there catastrophic forgetting?** No text-only data and no text benchmarks after fine-tuning.
- **Weak evaluation.** Results rest on Whisper transcripts with gpt-4o-mini scoring, BLEU against synthetic targets, and a 10-rater A/B test. Baseline interaction and QA numbers are copied from other papers, under different setups.
- **Relation to StepAudio 3 Realtime.** SA3-RT cites this paper in its duplex section. Its own reasoning mechanism, Think-While-Speaking, instead builds on Mind-Paced Speaking by the same first author (SA3-RT p.13). Whether SA3-RT uses listening-time thinking is not stated.
