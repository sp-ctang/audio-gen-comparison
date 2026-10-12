# Voxtral TTS

| Field | Value |
|---|---|
| Org | Mistral AI (mistral.ai; Hugging Face org "mistralai") (p.1) |
| Date / arXiv | arXiv 2603.25551v2, 6 Apr 2026 |
| Category | TTS (multilingual zero-shot voice cloning) + codec |
| Backbone | Ministral 3B architecture, initialized from Ministral 3B (p.5, p.7). The released checkpoint is named "Voxtral-4B-TTS-2603" (p.1); no parameter breakdown is given |
| Audio input | Voxtral Codec encoder: 24 kHz audio → 12.5 Hz frames of 37 tokens each. Each frame has 1 semantic VQ token (8192 entries) and 36 acoustic FSQ values (21 levels each). The per-codebook embeddings are summed into one frame embedding (p.2, p.5) |
| Audio output | 12.5 Hz (80 ms per frame). A linear head predicts the semantic token autoregressively (8192 + <EOA>, trained with cross-entropy). A 3-layer bidirectional flow-matching (FM) transformer, conditioned on the backbone hidden state, predicts the 36 acoustic values (8 Euler steps, CFG α=1.2). These are quantized to 21 levels and fed back as discrete tokens (p.5–6) |
| Waveform decoder | Voxtral Codec decoder: mirrored causal conv + sliding-window transformer, 12.5 → 100 Hz → 240-sample patches. The whole codec is ≈300M params (p.4–5) |
| Duplex mechanism | n/a (TTS) |
| Open weights / code | Weights on Hugging Face, CC BY-NC (p.1). Served through vLLM-Omni (p.11). Training code not mentioned |
| Relevance to us | **High.** It has a hybrid head (autoregressive semantic + flow-matching acoustic) and compares it against a depth transformer and MaskGIT. It shows DPO on mixed discrete/flow outputs, down-weights the loss on silent frames, freezes text embeddings, and reports real streaming latency |

## TL;DR

- **Codec.** 2.14 kbps at 12.5 Hz: one Whisper-distilled semantic VQ token plus 36 FSQ acoustic dims per frame. It beats Mimi-16cb at a similar bitrate on every objective metric (Tab.2).
- **Hybrid generation.** The backbone (initialized from Ministral 3B) predicts only the semantic stream; a small flow-matching (FM) transformer generates the acoustics. FM beat a depth transformer and MaskGIT in human evaluations, "especially on expressivity" (p.6).
- **Pretraining.** Samples are ⟨reference, transcript, target⟩ tuples with pseudo-labels. Text embeddings are frozen and silent frames are down-weighted (p.7). Data hours are not reported.
- **DPO.** Semantic DPO plus flow-DPO. Mean WER drops 2.47 → 1.84 (derived from Tab.6), UTMOS rises +0.06, volume tapering is fixed, and Hindi regresses.
- **Results.** Preferred over ElevenLabs Flash v2.5 68.4% of the time for cloning (Tab.5). On one H200: 70 ms latency at concurrency 1, and RTF 0.302 at concurrency 32 (Tab.8).

## Architecture

**Codec encoder (p.2–5, Tab.1).** 240-sample patches (100 Hz) go through a causal conv to 1024 dims. Then 4 blocks, each with 2 causal sliding-window transformer layers and a strided causal conv (×2 downsampling in 3 of the 4 blocks). The output is a 292-dim latent.

**Quantization.** The latent splits into two parts:
- 256 semantic dims, quantized by an 8192-entry VQ (applied to 50% of samples during training);
- 36 acoustic dims, each a 21-level FSQ (50% quantized, 25% dithered, 25% unquantized during training).

**Codec losses.**
- ASR distillation: a cosine loss between the post-VQ semantic embeddings and frozen Whisper's last-layer decoder states. These are soft-aligned via DTW-selected cross-attention heads, so no aligner or transcripts are needed.
- 8 STFT discriminators. L1 feature matching replaces the GAN generator loss.
- Decaying L1 and STFT reconstruction losses.

**TTS head (p.6).**
- The FM head takes three inputs: the backbone hidden state h, a sinusoidal timestep t, and the current acoustic state x_t ∈ R^36. Each has its own projection, which beat DiT-style AdaLN.
- h is dropped 10% of the time to enable classifier-free guidance (CFG). CFG then needs only an extra FM pass, not an extra backbone pass.
- The Fig.2 caption says the semantic *token* feeds the FM head; Sec. 2.3 and the figure use h.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **Objective:** α·feature + β·ASR + γ(L1 + STFT) + δ·commit, with α=1, β=1, γ=0.9999^t, δ=0.1 (p.5).
- "All decisions are ablated", but no ablations are shown.
- **Not reported:** data, steps, LR, compute.

### Stage 1: pretraining
**Data (p.6):**
- Audio paired with transcripts pseudo-labelled by Voxtral Mini Transcribe.
- Each sample is a tuple (A1 reference, T2 transcript, A2 target), joined by the special tokens <next> and <repeat>.
- A1 and A2 are single-speaker segments from the same speaker, not necessarily adjacent, each up to 180 s; A1 is at least 1 s.
- The loss covers A2 only. The model works best with 3–25 s prompts.

**Objective:** cross-entropy on semantic tokens plus conditional flow matching on the acoustic values.

**Init:** the backbone starts from Ministral 3B. The FM head, audio embeddings and output projections start random.

**Training fixes (p.7):**
- Text-embedding layers are frozen "to improve robustness to text tokens that appear with low-frequency".
- Non-speech frames (per a voice-activity-detection model) get a lower loss weight. Extremely long silences get weight 0 "to avoid overfitting to silence".
- An LLM rewrites transcripts so the model sees both normalized and un-normalized text.

**Not reported:** hours, language mix (9 languages supported), steps, LR, batch, compute. Text-only share: none mentioned.

### Stage 2: DPO (post-training)
**Objective (p.7):**
- Standard DPO on the semantic stream, β=0.1.
- Flow-DPO (Ziv et al.) on the acoustic stream, with t sampled per token and the same t and x0 used for policy and reference. β=0.5, because training "is sensitive to the flow-DPO loss".
- Normalizing by length caused instability.

**Schedule:** LR 8e-8. The pretraining loss on high-quality speech is kept alongside DPO. Training runs for 1 epoch, because "training longer on synthetic data led to more robotic speech".

**Pair data:**
- Inputs are held-out single-speaker voices plus texts from Mistral Small Creative. That model was prompted with the voice prompt's transcript and random personas to "continue or reply to the conversational context".
- Several samples are generated per input and ranked by WER, speaker similarity, loudness consistency, UTMOS-v2 and LM-judge metrics.
- Pair count: not reported.

## Data pipeline
- Real speech is pseudo-transcribed by an in-house ASR model, cut into same-speaker reference/target pairs, and rewritten by an LLM (p.6–7).
- Sources, hours and the conversational/read split are not reported.
- No multi-speaker, overlap or duplex data.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Codec | not reported | not reported | ≈300M params |
| Pretraining | not reported | not reported | init Ministral 3B |
| DPO | 1 epoch; size not reported | not reported | LR 8e-8 |
| Serving | n/a | 1× H200. Concurrency 1: 70 ms latency, RTF 0.103. Concurrency 32: 552 ms, RTF 0.302, 1430.78 char/s, 0% wait (Tab.8) | CUDA graphs cut latency 133 → 70 ms (Tab.7) |

## Preserving text intelligence / knowledge
- The backbone starts from Ministral 3B, and its text embeddings are frozen during pretraining (p.7). The stated reason is robustness to rare tokens, not keeping knowledge.
- No text benchmarks.
- Indirect sign of text understanding: with no emotion label ("implicit steering"), Voxtral wins 58.3% against ElevenLabs Flash v2.5 and 55.4% against v3, but only 37.1% against Gemini 2.5 Flash TTS (Tab.4).

## Key results

**Codec, Expresso (Tab.2).**

| Codec | kbps | Mel ↓ | PESQ | ESTOI | ASR-WER | SIM |
|---|---|---|---|---|---|---|
| Voxtral Codec | 2.1 | 0.545 | 3.05 | 0.882 | 10.66 | 0.843 |
| Mimi 16 codebooks | 2.2 | 0.618 | 2.67 | 0.865 | 11.01 | 0.829 |
| Mimi 32 codebooks | 4.4 | 0.552 | 3.18 | 0.910 | 10.25 | 0.902 |

**Automatic metrics (Tab.3).**
- Seed-TTS: WER 1.23, UTMOS 4.11, SIM 0.628, against ElevenLabs v3 at 1.26 / 3.92 / 0.392.
- MiniMax English: 0.63 / 4.30 / 0.786.
- SIM is higher than both ElevenLabs models in every language.

**Human evaluations.**
- Zero-shot cloning vs Flash v2.5: 68.4% win rate overall, ranging from Dutch 49.4% to Spanish 87.8% (Tab.5).
- Explicit emotion steering with flagship voices: 51.0% against ElevenLabs v3 and 35.4% against Gemini 2.5 Flash TTS (Tab.4).

## Ablations & lessons

**Acoustic head (p.6).** MaskGIT and a depth transformer "performed reasonably well, but were inferior to FM in human evaluations, especially on expressivity". FM is also cheaper:
- each frame needs a sequence of 3 inputs, against 38 for MaskGIT;
- it needs 8 function evaluations (NFEs), against 36 autoregressive steps for the depth transformer.
No numbers are given for the quality comparison.

**Number of NFEs (Fig.4, averaged over 10 test sets).**

| NFEs | WER | UTMOS | SIM |
|---|---|---|---|
| 2 | 30.1 | 1.44 | 0.421 |
| 4 | 6.5 | 2.58 | 0.656 |
| 8 | 6.7 | 3.48 | 0.732 |
| 16 | 7.0 | 3.48 | 0.735 |

**CFG strength (p.10).**
- Raising α from 1.0 to 1.4 improves objective metrics monotonically: WER 7.5 → 6.2, SIM 0.707 → 0.737.
- Human listeners found that high α over-adheres to the voice prompt and suppresses emotion implied by the text.
- α=1.2 is chosen for clean prompts; noisy in-the-wild prompts may want more.

**DPO results (Tab.6).**
- Large WER gains in German (4.08 → 0.83), French (5.01 → 3.22) and Dutch (3.10 → 1.99).
- Hindi gets worse (3.39 → 4.99).
- UTMOS rises +0.04 to +0.13. SIM moves by ≤±0.01.
- Qualitatively: fewer skipped words and hallucinations, and no volume tapering (p.10).

**DPO stability (p.7).** It needed a larger β for flow-DPO, no length normalization, a tiny LR, and the pretraining loss kept in the mix. Over-training on synthetic data makes speech robotic.

**Frozen text embeddings and silence down-weighting** are deliberate pretraining fixes (p.7). They are not ablated.

**Automatic metrics mislead (p.8).** ElevenLabs Flash wins most automatic metrics, but v3 wins the human evaluations. UTMOS "is only a loose proxy".

**Codec quality.** ASR-distilled semantic VQ plus FSQ acoustics at 2.1 kbps beats Mimi RVQ at 2.2 kbps on every objective metric (Tab.2).

## Implications for our setup

**Down-weight silence (our inference).** In duplex data, each channel is silent or listening for long stretches. Voxtral's VAD-based loss down-weighting (zero for long silences) is a cheap change that could stop silence tokens from dominating the audio loss and degrading continuation.

**Strengthen the per-frame acoustic head.** If the Qwen3 LLM predicts all codebooks through parallel heads, both Voxtral (FM beats depth transformer and MaskGIT) and MOSS-TTS (local head beats delay pattern) say a stronger per-frame head matters. For RVQ codes, the natural choice is a depth transformer. FM worked here because FSQ scalars can be treated as continuous values; that does not transfer directly to RVQ indices (our inference).

**Keep the backbone's target text-aligned (hypothesis).** Here the backbone predicts only one text-aligned (ASR-distilled) token per frame, and acoustics are offloaded to the FM head. A similar split might protect LLM knowledge. The paper does not evaluate this.

**Freeze text embeddings.** Freezing them during audio mid-training is a cheap knowledge safeguard to test.

**A ready post-training recipe.** Rejection sampling ranked by ASR-WER, speaker similarity, loudness, UTMOS and an LM judge. Then DPO at LR ~1e-7 for 1 epoch, with the pretraining loss mixed in. The persona-based "continue or reply" prompts already resemble dialogue.

**Tune CFG by ear.** CFG on the acoustic head is cheap and strongly affects quality, but tune it with human listening rather than UTMOS.

## Open questions / unclear points
- **Missing training details:** hours, languages, sources and conversational share of the training data; all training compute; pretraining LR and batch; DPO set size.
- **Parameter count:** the checkpoint name says 4B, but the backbone is 3B plus the FM head plus a ≈300M codec. No breakdown is given.
- **Fig.4 WER is inconsistent with the tables.** Fig.4's averages match Tab.3/Tab.6 for UTMOS (3.48) and SIM (0.732), but its WER (≈6.7%) is far above the Tab.6 means (derived: 2.47 pretrain, 1.84 DPO). It is unclear which ASR model or checkpoint Fig.4 uses.
- **DPO setup:** are the text embeddings still frozen during DPO?
- **Codec:** no ablations and no description of its training data.
