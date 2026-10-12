# VoxCPM2

| Field | Value |
|---|---|
| Org | VoxCPM Team / OpenBMB: Tsinghua SIGS HCSI Lab, THUNLP, ModelBest (p.1, p.20) |
| Date / arXiv | arXiv 2606.06928v1, 5 Jun 2026 |
| Category | TTS: multilingual, controllable, continuous-latent diffusion-autoregressive |
| Backbone | ≈2B total (p.9, Tab.1). The Text-Semantic LM (TSLM) is MiniCPM-4-1B (28 layers, H=2048), "built upon the MiniCPM-4 backbone" (p.3). The other modules are the Local Encoder (LocEnc: 12L, H1024), the Residual Acoustic LM (RALM: 8L, H2048, no positional encoding) and the Local Diffusion Transformer (LocDiT: 12L, H1024) |
| Audio input | AudioVAE V2 encoder: 16 kHz → 64-dim continuous latents at 25 Hz. LocEnc pools P=4 frames per patch, so the LM sees 6.25 Hz, or 160 ms per step (p.7–8) |
| Audio output | Continuous VAE latent patches (4 × 64 dims) at 6.25 Hz, with no codebooks. Per patch: TSLM state → FSQ "semantic skeleton" (512 dims × 9 levels) → RALM residual → LocDiT flow matching over the patch. A stop predictor ends generation (p.7–9) |
| Waveform decoder | AudioVAE V2 causal-CNN decoder: 25 Hz latents → 48 kHz ("implicit super-resolution"), stateful and streaming (p.8, p.12) |
| Duplex mechanism | n/a (TTS) |
| Open weights / code | Weights, fine-tuning code and inference tools under Apache 2.0 (p.1) |
| Relevance to us | **Medium.** Continuous latents differ from our discrete tokenizer. Still useful for its semantic/acoustic split inside the backbone, its 3-stage curriculum with high-quality (HQ) annealing, its continuation-vs-reference evidence, and its data scale (>2M h) |

## TL;DR

- **No discrete speech tokenizer.** A pretrained text LM plans semantics through an FSQ bottleneck. A residual acoustic LM adds detail, and a local DiT generates each 160 ms latent patch by flow matching. The whole stack trains end to end (p.7).
- **Scale.** 2B parameters, >2M h of speech, 30 languages plus 9 Chinese dialects, 48 kHz output, 6.25 Hz LM rate (p.1, p.4).
- **One backbone, several modes.** Basic TTS, voice design from text descriptions, reference cloning (optionally style-controlled) and continuation cloning differ only in input layout (Tab.2).
- **Curriculum.** Multilingual TTS pretraining → joint TTS plus controllable data → high-quality annealing SFT. No RL (p.10–11).
- **Headline results.** Seed-TTS-Eval: 1.84 WER / 75.3 SIM (EN) and 0.97 CER / 79.5 SIM (ZH). Best SIM on 22 of 24 MiniMax-MLS languages. 1.68% average WER on an internal 30-language set (Tab.3, Tab.7, p.15).

## Architecture

**Per-patch generation (Eq.1, p.7).**

- h_FSQ = FSQ(TSLM(text, E_<i)), where E_<i is the LocEnc-encoded acoustic history.
- h_res = RALM(TSLM text states, W_fuse[H_FSQ ∥ E]).
- LocDiT input: [μ_sem, μ_res, μ_t, previous clean patch, noisy current patch], with full attention. It predicts the velocity at the noisy positions.

**Changes from VoxCPM.**

- Concat-projection fusion replaces summation (p.8).
- The LocDiT gets separate conditioning tokens instead of one summed vector (p.8).
- FSQ width grows 256 → 512 (p.9).
- The RALM drops positional encoding (NoPE); the TSLM keeps RoPE (p.9).
- An isolated reference-audio segment [REF_START…REF_END] carries identity only. It needs no transcript and is excluded from the loss (p.9).

**AudioVAE V2 (p.8).**

- Causal CNN. The encoder downsamples by [2,5,8,8] = 640×. The decoder upsamples by [8,6,5,2,2,2] (derived: 1920×, 25 Hz → 48 kHz) and is deeper and wider.
- The decoder takes an optional output-sample-rate condition.
- Encoding at 16 kHz reuses VoxCPM's 16 kHz corpus and avoids latent mismatch across source sample rates.

**Inference (p.12).**

- Classifier-free guidance (CFG) α=2.0; the practical range is 1.5–3.0.
- Sway sampling and CFG-Zero*.
- Streaming patch by patch through the stateful decoder.
- The number of LocDiT solver steps is not reported.

## Training pipeline

**Common to all stages (p.10).**

- Loss: patch-level conditional flow matching plus a binary stop loss, both masked to the target segment. LM-side conditioning is dropped 10% of the time to enable CFG.
- Optimizer: AdamW with cosine decay and linear warmup.
- Trainable: the whole stack, end to end (p.7). VAE freezing is not stated.
- **Not reported:** LR, batch, steps, tokens, hardware. Text-only share: none mentioned.

### Stage 0: tokenizer / codec / decoder training
The AudioVAE V2 architecture is described, but its training data, losses and compute are not. It "follows" the original AudioVAE design (p.8).

### Stage 1: multilingual TTS pretraining
- Data: large-scale multilingual ⟨transcription, audio⟩ pairs, used for basic TTS and continuation cloning.
- Limits: audio ≤60 s; LM sequence ≤4096.
- Goal: "solid pronunciation and prosody across all 30 target languages" (p.10).

### Stage 2: joint TTS + controllable pretraining
- Keeps "a large proportion of plain TTS data to preserve base synthesis quality".
- Controllable data is added "at an increasing ratio":
  - description-annotated speech, for voice design;
  - ⟨reference, transcription, target⟩ triplets, for cloning.
- Context 8192; audio up to 3 min.
- Self-synthesized, content-decoupled data is injected mainly here.
- Stages 1–2 are "the majority of training compute" (p.10–11).

### Stage 3: high-quality annealing SFT
- Data: a curated high-quality expressive subset plus precisely annotated controllable data. The controllable share is "significantly larger", with more reference-based controllable cloning.
- Samples run 2 s to 5 min; context 8192.
- Sampling is balanced at the language level, with LR annealing.
- Only natively recorded speech is used, no self-synthesized audio (p.11).

### Stage 4: RL / preference
None. The authors attribute part of Fish Audio S2's WER lead to its "additional RL post-training" (p.14).

## Data pipeline

(p.11)

**Base corpus**

- >2M h, mostly Chinese and English. The other 28 languages range from about 1K to 50K h each.
- Pipeline: source separation → voice activity detection (VAD) → ASR-based transcript alignment → quality filtering. Thresholds are not given.

**Controllable data**

- Tens of thousands of hours of open-source expressive speech, plus several thousand hours of internally annotated data.
- Expressive clips are pre-screened with emotion classifiers.
- Free-form voice and style descriptions come from Step-Audio R1 and Gemini 2.5 Pro, checked by expert gender, age and emotion classifiers.
- References are same-session clips with speaker similarity >0.7, excluding the clip immediately before the target.

**Content–style decoupling**

The model clones an annotated voice and style onto unrelated text, keeping the original description. This stops style being inferred from the words and extends control to long-tail languages.

No conversational or multi-speaker data is described.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| AudioVAE V2 | not reported | not reported | n/a |
| S1 | share of >2M h; tokens not reported | not reported | context 4096, ≤60 s |
| S2 | + controllable (~10Ks h open, several K h internal) | not reported | context 8192, ≤3 min; S1+S2 = majority of compute |
| S3 | curated subset, size not reported | not reported | LR anneal |
| Inference | n/a | RTX 4090: RTF 0.30 (PyTorch), 0.13 (Nano-vLLM), ~8 GB (Tab.11) | 6.25 Hz LM rate |

## Preserving text intelligence / knowledge

- The TSLM is MiniCPM-4-1B. Voice and style instructions are plain text prefixes to it, with no extra encoders, so instruction following rests on the LM's text understanding (p.6, p.10).
- No text-only data and no text benchmarks.
- The paper never explicitly says that pretrained text weights are loaded; it is implied by "built upon the MiniCPM-4 backbone".

## Key results

**Seed-TTS-Eval (Tab.3)**

| Model | Params | EN WER/SIM | ZH CER/SIM | ZH-Hard CER/SIM |
|---|---|---|---|---|
| VoxCPM2 | 2B | 1.84/75.3 | 0.97/79.5 | 8.13/75.3 |
| VoxCPM1.5 | 0.8B | 2.12/71.4 | 1.18/77.0 | 7.74/73.1 |
| VoxCPM | 0.6B | 1.85/72.9 | 0.93/77.2 | 8.87/73.0 |

**Other benchmarks**

- **CV3-Eval (Tab.5):** behind Fish Audio S2 on most languages, e.g. EN 5.00 vs 2.43.
- **MiniMax-MLS, Whisper ASR (Tab.6–7):** best SIM on 22/24 languages. WER is weak on Arabic (13.05) and Hindi (19.70), and on the unseen Czech (24.13) and Romanian (21.58).
- **InstructTTSEval (Tab.9):** EN APS/DSD/RP 84.2/83.2/71.4; ZH 85.2/71.5/60.8.
- **MOS (Tab.12; 50 listeners, ~100 samples):** cloning N-MOS 4.78 and S-MOS 4.74 (highest). Qwen3-TTS scores 4.80/4.69.

## Ablations & lessons

**Conditioning pathway (Tab.4, Seed-TTS; WER or CER / SIM).**

| Recipe | test-EN | test-ZH | ZH-Hard |
|---|---|---|---|
| Continuation only | 1.01/77.7 | 1.97/72.6 | 8.16/72.4 |
| Reference only | 1.10/75.3 | 1.81/67.0 | 6.85/70.0 |
| Reference + Continuation | 0.99/79.5 | 1.94/75.2 | 7.44/74.9 |

- Using both pathways gives the best SIM everywhere: the prefix supplies prosodic and temporal alignment, and the isolated reference supplies identity.
- Without a temporal prefix, hard-case intelligibility improves at some cost in SIM (p.14).

**Scale within the family (Tab.3).** Moving from 0.8B to 2B also changed the data, languages and VAE, so effects are confounded.

- SIM: +3.9 EN, +2.5 ZH, +2.2 hard.
- EN WER 2.12 → 1.84, but hard-ZH CER worsened, 7.74 → 8.13.

**Claimed without numbers.**

- Patch size 2 → 4 (LM rate 12.5 → 6.25 Hz) cuts cost "while improving long-form stability" (p.9).
- NoPE in the RALM "improves long-utterance stability" (p.9).
- Separate LocDiT conditioning tokens "avoid early information collapse" (p.8).

**Asymmetric VAE (Tab.10).**

- Encoding at 16 kHz and decoding at 48 kHz roughly matches the 16k/16k VAE on 16k-band PESQ (3.906 vs 3.940).
- It is much better on full-band MelD-48k (1.335 vs 1.787).
- A native 44.1 kHz VAE wins full-band (1.139) but loses 16k-band PESQ (3.148).

**Curriculum hygiene.**

- Keep a large share of plain TTS when adding capabilities, "to prevent destabilizing the base synthesis quality" (p.10).
- Keep self-synthesized audio out of the final annealing stage (p.11).

**Low-resource languages are weakest, and the evaluation ASR matters.** Arabic and Hindi score 1.23 and 0.79 CER with Gemini ASR on the internal set, vs 13.05 and 19.70 WER with Whisper on MiniMax (p.15).

**SIM ≠ perception.** Speaker-embedding SIM "do[es] not always fully reflect perceived cloning quality" (p.19).

**A low LM rate keeps 2B fast.** RTF is 0.13, vs 0.15 for the 0.8B VoxCPM1.5 (Tab.11).

## Implications for our setup

**Separate semantics from acoustics (our inference).** In VoxCPM2 the pretrained LM produces only a coarse semi-discrete plan per 160 ms, and separate modules render the acoustics. If our Qwen3 LLM predicts every codebook directly, acoustic detail competes for its capacity. Offloading residual codebooks to a depth or residual module may help both continuation quality and knowledge retention. This is a hypothesis: VoxCPM2 reports no text-knowledge metric.

**Patching to lower the LM rate.** Grouping frames into patches with a local encoder and decoder could cut LM steps for our 12.5 Hz tokens. That matters more for two-stream duplex sequences (our inference).

**Curriculum template.**

1. Train on broad base data.
2. Add the new capability (duplex) at an increasing ratio, keeping plenty of base data.
3. Finish with an HQ annealing SFT on real, non-synthetic, balanced data.

**Conditioning.** An isolated voice reference plus the running continuation context gave the best SIM. A duplex system prompt could carry a reference segment for the agent's voice.

**Data-scale calibration (our inference).** Languages with about 1K–50K h inside a 2M h corpus are the model's weakest. Our 10K h duplex set is that size, so expect weak acoustics unless we add a much larger single-speaker continuation corpus.

## Open questions / unclear points

- **Missing training details:** LR, batch, steps, tokens and GPU-hours; per-stage data sizes and mixing ratios; AudioVAE V2 training; LocDiT solver steps.
- **Initialization:** does the TSLM load MiniCPM-4-1B weights, and is any text-only data mixed in?
- **Tab.4 vs Tab.3 inconsistency (verified in the PDF).** Tab.4's "Reference + Continuation" row is the stated default recipe, yet it reads EN 0.99/79.5 and ZH 1.94/75.2. Tab.3 gives EN 1.84/75.3 and ZH 0.97/79.5. The EN and ZH columns look swapped, and the hard-set numbers also differ (7.44/74.9 vs 8.13/75.3).
- **Baselines don't match their own reports.** Tab.3 lists MOSS-TTS at 1.85/73.4, while MOSS-TTS's own continuation result is 1.84/70.86.
- **Scope:** no multi-speaker, conversational or duplex evaluation.
