# StepAudio 3 Gen Technical Report

| Field | Value |
|---|---|
| Org | StepFun (StepFun-Audio Team) |
| Date / arXiv | 11 Sep 2026 · arXiv:2609.12945v1 (p.1) |
| Category | Unified audio generation and TTS (zero-shot TTS, voice design, singing, sound effects, music, "vibe speech", mixtures), plus a codec (StepAudio Tokenizer). |
| Backbone | "a pretrained text LLM"; name and size **not reported** (p.8). Decoder-only; the vocabulary is extended by 2,048 audio tokens, which are codebook 0 (p.4). |
| Audio input | StepAudio Tokenizer codes at 12.5 Hz. A frame's 16 codebook embeddings are **summed** and passed through the **RVQ Adaptor** (zero-init, token-wise residual SwiGLU blocks). The result is added to the codebook-0 (cb0) token embedding at audio positions only (p.5). |
| Audio output | 12.5 Hz RVQ, **16 codebooks × 2,048 entries**. cb0 comes from the shared LM head along the time axis, in the same softmax as text. cb1–15 come from a **4-layer causal "RVQ Code Predictor"** along the depth axis, conditioned on the projected backbone hidden state + cb0 (pp.2, 4–5). This is a time–depth layout, not a delay pattern. |
| Waveform decoder | Fully causal Vocos-style Transformer (RoPE, 25-frame sliding-window attention) with an ISTFT head, producing 24 kHz. No diffusion or flow-matching renderer (p.4). |
| Duplex mechanism | n/a |
| Open weights / code | Not stated; a samples page is linked (p.1). |
| Relevance to us | **High.** The most controlled evidence in this batch on two things: how to lay out multi-codebook generation around an LLM, and how to add audio generation without eroding the LLM's text ability (Tabs.2–3). Its token space has the same 12.5 Hz × 16 × 2048 shape as the Qwen3 12 Hz tokenizer and codec we build on (see `qwen3-tts.md`, `qwen3-omni.md`). |

## TL;DR

- **Model.** One discrete autoregressive model for speech, singing, music, sound and mixtures, over a shared 12.5 Hz 16×2048 RVQ space (pp.1–5).
  - The LLM predicts cb0, interleaved with text.
  - A 4-layer depth Transformer fills cb1–15.
  - A causal Vocos decoder outputs 24 kHz. There is no flow or diffusion renderer.
- **Pretraining (~2.7T tokens), staged to avoid interference:**
  1. frozen-backbone alignment with a zero-init adaptor, so text drift is zero;
  2. understanding mixed 1:1 with text;
  3. generation with the code predictor **detached** from the backbone (text:TTS:dialogue = 3:1:2);
  4. a joint 32K cool-down with λ 1.0 → 0.1.

  Text makes up **50% of every step from Stage 2 on** (pp.8–9, Tab.1).
- **Ablations.**
  - **RVQ Adaptor:** SpeechMMLU (T2S, text-to-speech; presumably text question, spoken answer) 12.13 → 58.76, and LibriSpeech WER 6.00 → 3.41 (Tab.2).
  - **Staged recipe vs a 3-stage baseline:** better on all 8 text benchmarks, e.g. MMLU 64.99 → 69.20 (Tab.3).
- **Post-training.** SFT on ~5,000 h, including 1,500 h of natural conversational speech. Then **GRPO** (G=16) with reward = LLM-judged caption–instruction consistency × exp(−3·ASR error), zeroed if the error exceeds 0.5 (pp.9–10).
- **Results:**
  - Chinese human-likeness Elo **1755.3**, 82.0% head-to-head wins over 5 commercial TTS systems;
  - InstructTTSEval **77.7 EN / 85.2 ZH**;
  - voice-design Elo **1668.5** (Figs.2–3, Tab.4).

## Architecture

- **StepAudio Tokenizer** (pp.3–4).
  - Follows the X-Codec line: a *frozen* SSL encoder supplies semantic features, and a convolutional encoder with SnakeBeta activations extracts acoustic features from raw waveform at 50 Hz.
  - The two are fused on the channel axis, compressed to 12.5 Hz by a strided conv, and quantized by **one shared** 16×2048 RVQ with factorized cosine-similarity lookup. Every layer therefore carries both semantic and acoustic information, rather than splitting by stream.
  - The decoder is fully causal: a Vocos-style Transformer with RoPE and a 25-frame sliding window (2.0 s; derived: 25/12.5) feeds an ISTFT head. It has a bounded receptive field and no look-ahead, for streaming (p.4).
- **Backbone vocabulary** (p.4). cb0 is promoted into the LM vocabulary as 2,048 contiguous tokens (ids [V, V+2048), eq.3), so text and cb0 share one head and one softmax and can interleave freely.
- **Input path** (p.5). Each codebook has its own embedding table. The 16 vectors are summed, passed through the **RVQ Adaptor** (token-wise residual blocks with pre-norm and SwiGLU, down-projections zero-initialized), and added to the cb0 token embedding.
  - cb0 is therefore represented twice: as the vocabulary token and inside the summed frame.
  - Text positions receive nothing from this path ("position gating").
- **RVQ Code Predictor** (pp.2, 5). A lightweight 4-layer causal Transformer along the codebook axis.
  - Its prefixes are the backbone hidden state zₜ (linearly projected) and cb0; it then emits cb1…cb15 autoregressively.
  - A stop-gradient toggle on the zₜ prefix controls whether the residual loss reaches the backbone.
  - The factorization p(cₜ,₀|z<ₜ,c<ₜ)·∏ₖ p(cₜ,ₖ|zₜ,cₜ,₀..ₖ₋₁) (eq.1) is exact, because zₜ already encodes all past codebooks through the summed embeddings. The cost is that the predictor's output never feeds back into the backbone (p.5).
- **Loss** (p.5). L = L_tok + λ·L_sp, where L_tok covers text and cb0 and L_sp **sums** over the 15 residual codebooks. λ is the only weight changed between stages.
- **Why time–depth, not a delay pattern** (p.2). A delay pattern would need a multi-stream output interface and would make the pretrained LLM "directly predict every residual layer and receive all acoustic losses". Time–depth lets the LLM handle the semantically concentrated cb0 and long-range planning, while "a small module contains residual acoustic modeling and its gradients."
- **Interleaving** (p.6). Generation samples interleave **one text token with two audio frames**, i.e. 6.25 text tokens per second of audio (derived).
- **Control format** (p.3). Three fields:
  - ROLE: speaker identity and voice;
  - DIRECTOR: scene and intent;
  - SCRIPT: time-ordered speaker-tagged speech segments with optional `(description)`, plus `[description]` entries for sound and music events.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training (p.8)
- **Objectives.**
  - RVQ with **quantizer dropout 0.5**. This is "deliberately moderate": more dropout strengthens the coarse layers but slightly degrades full-depth reconstruction.
  - A distillation branch regresses SSL teacher features from the quantized latent. Because this runs under quantizer dropout, semantic information concentrates in the earliest codebooks.
  - GAN training with a multi-period waveform discriminator and a multi-resolution spectral discriminator, plus feature-matching, multi-scale mel and commitment losses.
- **Data and schedule.**
  1. Pretraining on **~700k h** of speech, music and sound events with a modality-balanced sampler.
  2. Decoder-only refinement: encoder, quantizer and semantic branch frozen; only the acoustic decoder and discriminators update. This improves reconstruction while leaving tokens unchanged, so downstream models stay compatible.
- SSL model identity, steps, LR and reconstruction metrics: not reported.

### Stage 1: modality alignment (p.9; Tab.1)
- **Data:** ASR and S2TT, supervising only the *text* output.
- **Trainable:** input-side audio embeddings and the adaptor. Backbone, LM head and code predictor sit at LR 0. The token embedding gets a **0.1× multiplier** so the new audio-code rows can adapt.
- **Zero drift.** The zero-init adaptor is an identity map at init; with a frozen backbone, text drift is "strictly zero", so no text replay is needed.
- **Hyperparameters:** batch **4.19M tokens** (derived: exactly 256 × 16,384); LR 2×10⁻⁴ → 2×10⁻⁵, cosine. Sequence length is 16,384 (implied by the Stage-4 extension and the batch arithmetic). Token count: not reported.

### Stage 2: audio understanding (p.9)
- The full model is unfrozen and trained on audio-understanding tasks mixed **1:1 with the text corpus**, with no audio targets supervised.
- **Tasks.** p.6 lists these understanding tasks for pretraining as a whole: ASR, S2TT, audio captioning, paralinguistic QA, accent/dialect ID and interleaved multi-turn speech dialogue.
- **Learning rate:** constant 2×10⁻⁵, a plateau the later stages continue without re-warming. From-scratch modules keep a **10× LR multiplier** and are excluded from weight decay.
- **Batch:** 12.58M tokens (derived: 768 × 16,384).

### Stage 3: generation with a detached code predictor (main run) (pp.6, 9)
- **Mixture:** text : TTS : interleaved dialogue = **3:1:2**. Text is 50%, and TTS is "1/6 of every micro-batch, or one third of the audio-generation streams". Interleaved dialogue is therefore 1/3 of each micro-batch (derived).
- **Generation tasks** in pretraining (p.6): TTS, singing synthesis, lyrics-conditioned music, caption-conditioned sound and speech-to-speech translation.
- **Detachment.** The predictor's conditioning input is detached from the backbone hidden states. The 15-codebook loss "is far stronger in magnitude than the token-level objective", and the predictor is still random, so its gradient "would reshape the representation built over the previous two stages."
  - With the detach, the acoustic loss trains only the predictor, while the backbone learns from the token objective (text + cb0).
  - This "decouples learning acoustic detail from learning the semantic and prosodic plan", a second safeguard for text ability.
- **Hyperparameters:** λ = 1.0; batch 12.58M; LR 2×10⁻⁵ constant.

### Stage 4: long-context joint cool-down (p.9)
- **Detach removed.** The residual loss now reaches the backbone, so hidden states become good acoustic conditioning. This is deferred until the predictor has converged, and the low cool-down LR limits how much its gradients alter the backbone.
- **λ 1.0 → 0.1**, to keep the summed 15-codebook loss from being "an order of magnitude above the token-level objective".
- **Sequence 16,384 → 32,768** via context parallelism, covering roughly 44 minutes of audio (derived: 43.7 min).
- **Batch** 25.17M tokens (derived: 768 × 32,768); **LR** 2×10⁻⁵ → 1.5×10⁻⁵, cosine.
- Mixture: p.6 says "the generation stages" use 3:1:2; we read this as Stages 3–4.

**Pretraining total: ~2.7T LLM tokens** across the 4 stages (pp.6, 8). One temporal audio step counts as one token, i.e. 12.5 tokens/s (Tab.1 caption). The per-stage split is not reported.

### Stage 5: SFT (pp.9–10)
- **Setup.** Full-parameter, with the joint configuration carried over: detachment off and λ = 0.1.
- **Data:** ~**5,000 h** of audio covering music, sound effects, voice design, singing and speech. The music and text-conditioned singing subsets are generated by the StepAudio Music Model. The TTS part includes **1,500 h** of curated natural conversational speech (p.6).
- **Hyperparameters:**
  - audio embedding, adaptor and code predictor keep a 10× LR multiplier;
  - sequence 16,384; batch size 64 (units not stated; ≈1.05M tokens/step if these are packed 16K sequences, derived);
  - cosine LR 1×10⁻⁵ → 1×10⁻⁶.

### Stage 6: RL, GRPO (p.10)
- **Purpose:** improve instruction following.
- **Data:** same taxonomy and conditioning formats as SFT (TTS, voice design, vocal, music, sound, speech+sound mixtures).
  - Speech examples include an exact transcript plus role and style requirements; other tasks use concrete descriptions of acoustic events and performance.
  - Ambiguous or weakly verifiable instructions are removed to reduce reward noise.
- **Reward.**
  - An audio-understanding model ([27], Step-Audio-R1.5) captions each output yᵢ. A text LLM scores caption vs instruction from 0 to 100; this is repeated 4 times and averaged (s̄ᵢ).
  - For speech, ASR gives an error eᵢ (CER or WER). **Rᵢ = (s̄ᵢ/100)·exp(−3·eᵢ) if eᵢ ≤ 0.5, else 0.** Tasks without a transcript use an ASR term of 1.
  - The product prevents "an output with severely incorrect linguistic content from receiving a high score solely because its acoustic style matches."
- **Optimisation.**
  - **G = 16** samples per instruction, with group-normalized advantages and no value model.
  - **DAPO-style dynamic sampling:** groups with reward std σ_R < 0.02 are dropped.
  - The clipped token-level ratio is applied over every generated position **and every codebook ℓ = 0…15**, normalized by 16|yᵢ|. RL therefore updates both the backbone (cb0) and the predictor (cb1–15).
  - Small KL coefficient β to the SFT reference, relying mainly on clipping; LR 10⁻⁶ → 10⁻⁷.
- **Not reported:** steps, batch size, ε_clip, β value, RL data size, and before/after-RL numbers.

## Data pipeline

- **Pretraining** (p.6). A "generation-centric mixture while retaining the text corpus".
  - Generation tasks: TTS, singing synthesis, lyrics→music, caption→sound, and speech-to-speech translation (S2ST).
  - Understanding tasks: ASR, S2TT, captioning, paralinguistic QA, accent/dialect ID and interleaved multi-turn speech dialogue. These "anchor RVQ representations in language" and expose timbre, emotion and accent.
  - Hours per task and the format of the interleaved dialogue: not reported.
- **Natural-speech TTS SFT** (p.6). It focuses on "tone, rhythm, and spontaneous vocal behaviors found in real conversations", unlike StepAudio 2.5 TTS's instruction control.
  - Step-Audio-R1.5 writes TTS-oriented captions (recording context, noise level, speaker traits) used to select natural conversational speech.
  - A paralinguistic-aware ASR transcribes and verifies candidates while keeping tone and hesitation cues.
  - Result: **1,500 h**.
- **Other SFT domains** (pp.6–7):
  - *Speech→vocal pairs* from TTS cloning of singers and singing voice conversion.
  - *Voice design* from recorded and synthesized speech.
  - *Vocals* from source-separated songs.
  - *Music* generated by the StepAudio Music Model, with multi-form ZH/EN prompts.
  - *Sound* from libraries plus LLM-scripted speech–sound scenes mixed from separately synthesized stems.

  These sets are filtered with Audiobox-Aesthetics PQ/CE, SingMOS, timbre similarity and ASR cross-checks. None of this is relevant to duplex speech.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Tokenizer | ~700k h (p.8) | not reported | GAN pretraining + decoder-only refinement |
| PT Stage 1 | not reported | not reported | batch 4.19M tok (=256×16K); frozen backbone |
| PT Stage 2 | not reported | not reported | batch 12.58M (=768×16K); 1:1 text |
| PT Stage 3 | not reported | not reported | batch 12.58M; 3:1:2 text:TTS:dialogue; main run |
| PT Stage 4 | not reported | not reported | batch 25.17M (=768×32K); context parallel |
| PT total | **~2.7T tokens** (pp.6, 8) | not reported | ≥50% text from Stage 2 on |
| SFT | ~5,000 h | not reported | batch 64 × 16K sequence |
| GRPO | G=16 | not reported | LR 1e-6 → 1e-7 |

Scale anchor (derived): at 12.5 Hz, 1 h of audio = 45,000 temporal tokens. Our 10K h ≈ **0.45B frames per channel**, about 0.02% of SA3-Gen's 2.7T pretraining tokens.

## Preserving text intelligence / knowledge

- **Six mechanisms:**
  1. Only cb0 enters the LLM vocabulary, and residual acoustics live in a small predictor (pp.2, 4).
  2. Stage 1 uses a zero-init adaptor with a frozen backbone, so drift is exactly zero (p.9).
  3. **50% text replay** in every step from Stage 2 onward (p.9).
  4. The predictor is detached in Stage 3 (p.9).
  5. End-to-end acoustic gradients arrive only at the low-LR cool-down, with λ = 0.1 (p.9).
  6. From-scratch modules get a 10× LR multiplier and no weight decay, while the backbone stays at 2e-5 (p.9).
- **Evidence (Tab.3, p.11).** Text benchmarks after pretraining, interference-aware recipe vs a 3-stage baseline (ASR → understanding+generation → cool-down, *without* the adaptor):

  | Benchmark | Baseline | Interference-aware |
  |---|---|---|
  | FinEval | 65.86 | 71.68 |
  | C-Eval | 66.34 | 71.92 |
  | MMLU | 64.99 | 69.20 |
  | CMMLU | 66.69 | 71.73 |
  | MATH | 36.62 | 45.07 |
  | GSM8K | 67.94 | 74.05 |
  | BBH | 59.61 | 67.03 |
  | HumanEval | 47.56 | 54.27 |

  The gains are +4.2 to +8.5 points (derived).
- **Knowledge in generated speech.** SpeechMMLU in the T2S setting reaches 58.76 with the adaptor vs 12.13 without (Tab.2).
- **Caveats.**
  - The paper gives no row for the original text LLM, so absolute retention is unknown.
  - The baseline differs at least in adaptor and stage structure. Whether it used detachment or the same text ratio is not stated.

## Key results

- **After pretraining, with adaptor (Tab.2):** AISHELL-1 CER 3.00; LibriSpeech-clean WER 3.41; MMAU 51.70; CoVoST BLEU En→Zh 30.56 / Zh→En 18.59; SpeechMMLU (T2S) 58.76.
- **TTS human-likeness, Chinese** (Fig.2, p.12). Six systems, one voice each, 1,500 pairwise comparisons, judged on "sounds more like a real person".

  | System | Elo | StepAudio 3 Gen win / tie / loss (100 trials) |
  |---|---|---|
  | **StepAudio 3 Gen** | **1755.3** | — |
  | Qwen-Audio-3.0-TTS-Plus | 1544.3 | 73 / 9 / 18 |
  | StepAudio 2.5 TTS | 1504.0 | 78 / 8 / 14 |
  | Doubao-App | 1490.3 | 79 / 9 / 12 |
  | Inworld-TTS-2 | 1209.6 | 90 / 5 / 5 |
  | MiniMax-Speech-2.8-HD | 1181.4 | 90 / 8 / 2 |

  The aggregate head-to-head win rate is **82.0%** (410 / 39 / 51 of 500).
- **InstructTTSEval, judged by Gemini 3.1 Pro (Tab.4).** 1,000 ZH + 1,000 EN texts, each under three conditions (6,000 utterances per system).

  | System | EN APS / DSD / RP | EN avg | ZH APS / DSD / RP | ZH avg |
  |---|---|---|---|---|
  | **StepAudio 3 Gen** | 82.4 / 85.5 / 65.2 | **77.7** | 88.5 / 91.5 / 75.5 | **85.2** |
  | Qwen3-TTS-12Hz-1.7B-VoiceDesign | 74.7 / 83.8 / 61.9 | 73.5 | 79.0 / 86.7 / 56.2 | 74.0 |
  | MOSS-VoiceGenerator | 59.5 / 79.0 / 56.1 | 64.9 | 63.7 / 78.1 / 53.4 | 65.1 |
- **Voice design, human preference** (Fig.3, pp.13–14). 490 comparisons in ZH and EN.

  | System | Elo | StepAudio 3 Gen win rate (49 trials) |
  |---|---|---|
  | **StepAudio 3 Gen** | **1668.5** | — |
  | Qwen3-TTS-12Hz-1.7B-VoiceDesign | 1571.5 | 65.3% |
  | MOSS-VoiceGenerator | 1495.0 | 71.4% |
  | Ming-omni-tts-0.5B | 1409.6 | 79.6% |
  | VoiceSculptor | 1243.4 | 85.7% |

  The aggregate win rate is 75.5% (148 W / 9 T / 39 L).
- **Not evaluated with objective metrics.** Music (>60 s coherent pieces), singing, sound and "vibe speech" are shown qualitatively only (p.14). No objective TTS WER or speaker similarity is reported; the authors argue these metrics miss expressiveness (p.12).

## Ablations & lessons

- **The input adaptor for multi-codebook frames matters a lot** (Tab.2, after pretraining):

  | Metric | w/o adaptor | w/ adaptor |
  |---|---|---|
  | AISHELL-1 CER | 5.25 | **3.00** |
  | LibriSpeech WER | 6.00 | **3.41** |
  | MMAU | 40.70 | **51.70** |
  | CoVoST En→Zh BLEU | 12.05 | **30.56** |
  | CoVoST Zh→En BLEU | 5.99 | **18.59** |
  | SpeechMMLU (T2S) | 12.13 | **58.76** |

  The stated cause: summed RVQ embeddings "need not match the statistics of pretrained text embeddings" (p.2).
  - *Our inference:* in T2S the model reads back its own generated frames through this input path. So the input-side representation of generated audio plausibly governs whether knowledge survives during speech generation.
- **A staged curriculum with text replay preserves text ability** (Tab.3; numbers in the previous section). The largest gains are MATH +8.45 and BBH +7.42 (derived). It is a package comparison with no per-mechanism ablation.
- **Design arguments without isolated numbers:**
  - detach the randomly initialized acoustic head, then rebalance λ 10× when gradients reach the backbone (p.9; see Stage 3–4);
  - time–depth over delay pattern, to keep acoustic losses out of the LLM (p.2).
- **Tokenizer:** quantizer dropout 0.5 trades coarse-layer strength against full-depth reconstruction (p.8). Decoder-only refinement improves audio without invalidating trained LMs (p.8). Neither has numbers.
- **The RL reward needs a content gate.** Multiplicative ASR gating stops style-matching outputs with wrong words from scoring high (p.10). No RL ablation or before/after numbers are given.
- **Natural conversational SFT data targets human-likeness.** The model beats its predecessor StepAudio 2.5 TTS by 251 Elo (derived) and 78 to 14 head-to-head, which the authors attribute to focusing on conversational "tone, rhythm, and spontaneous vocal behaviors" (pp.6, 12). This is confounded with all other changes.
- **Evaluation.** Low CER is compatible with flat prosody, so they use arena-style preference instead (p.12).

## Implications for our setup

1. **Keep the codebook layout; copy the loss schedule.** The Qwen3 12 Hz tokenizer has the same 12.5 Hz × 16 × 2048 shape. Qwen3-TTS and Qwen3-Omni already predict cb0 with the backbone and cb1–15 with an MTP module (`qwen3-tts.md`, `qwen3-omni.md`), which is SA3-Gen's time–depth split.
   - Make sure our duplex LLM keeps that split: it predicts cb0 only, with no flattened or delayed 16-codebook prediction.
   - The transferable piece is the schedule. If the depth/MTP head is new, or re-attached to a different backbone (a Qwen3 text LLM rather than the TTS talker), train it on **detached** conditioning first. Then enable end-to-end gradients at low LR with λ ≈ 0.1; otherwise the summed 15-codebook loss dominates the token loss.
2. **Add the input adaptor.** Qwen3-TTS already sums the codebook embeddings on input (`qwen3-tts.md`). SA3-Gen adds a zero-init residual SwiGLU adaptor on that sum, then adds the result to the cb0 token embedding.
   - Without it, SpeechMMLU-T2S is 12.13; with it, 58.76.
   - It is cheap, and it shapes how the LLM reads both the user channel and its own generated channel.
3. **Curriculum.**
   - (a) a frozen-Qwen3 alignment stage on ASR/S2TT;
   - (b) unfreeze with **1:1 text replay**;
   - (c) a generation stage with the acoustic head detached, still 50% text;
   - (d) a joint cool-down.

   Unlike the paper, compare against the original Qwen3's text scores after every stage.
4. **Text-guided speech.** Interleave **1 text token per 2 audio frames** at 12.5 Hz in the model channel (p.6).
5. **Scale gap.** ≈0.45B frames per channel vs 2.7T tokens (derived). Do not expect to learn acoustic generation from the duplex data alone. Lean on TTS-pretrained Qwen3 parts and add public ASR/TTS corpora; reserve the duplex data for interaction.
6. **RL template.** GRPO with G=16 and reward = judge score × exp(−3·WER), gated at WER ≤ 0.5. Drop zero-variance groups, take ratios over all codebooks, and use LR 1e-6 → 1e-7. For duplex, the WER term could compare the model's generated audio against its own text channel.
7. **Naturalness SFT.** Filter real conversational speech with a captioner and a paralinguistic-preserving ASR; 1,500 h was the size used. Our duplex corpus could serve if cleaned this way.
8. **Decoder.** A causal Vocos/ISTFT decoder with a 2 s window streams without flow matching. Consider it if the Qwen3 decoder's latency bites.

## Open questions / unclear points

- **Not reported:** backbone identity and size, per-stage tokens, steps, hardware, the SSL encoder, and the SFT batch units.
- **No codec or objective TTS evaluation.** There are no reconstruction or codebook-usage metrics and no standard-set WER/SIM, so comparison with Qwen3-TTS rests on a Gemini judge and small arenas. The human-likeness arena is Chinese-only with one voice per competitor.
- **The Tab.3 comparison is confounded.** The baseline lacks the adaptor and uses 3 stages; its detachment and text ratio are not stated. There is no original-LLM row and no single-factor ablation.
- **Source of the adaptor's T2S gain.** It may come from reading back self-generated frames or from cleaner understanding; the paper does not analyze this.
- **Mixture details.** Understanding tasks are listed for pretraining (§3.1) but absent from the 3:1:2 Stage-3/4 mix; are they dropped after Stage 2? The format and size of the "interleaved multi-turn speech dialogue" data, and whether it includes any overlapping speech, are not given.
- **RL is unquantified.** There are no reward curves, no pre/post-RL scores and no step counts.
- **Link to the Realtime generator.** The paper does not say whether this is StepAudio 3 Realtime's speech generator, despite the same release week.
