# MOSS-TTS (MOSS-TTS and MOSS-TTS-Local-Transformer)

| Field | Value |
|---|---|
| Org | SII-OpenMOSS Team: Shanghai Innovation Institute, MOSI Intelligence, Fudan University (p.1, p.24) |
| Date / arXiv | arXiv 2603.18090v2, 20 Mar 2026 |
| Category | TTS (+ audio tokenizer) |
| Backbone | Decoder-only LLM. MOSS-TTS (delay pattern) is 8B and MOSS-TTS-Local-Transformer is 1.7B (p.19, Tab.3). The text LLM it starts from is **not reported** |
| Audio input | MOSS-Audio-Tokenizer, a causal Transformer: 24 kHz → 12.5 fps with 32-layer RVQ. Per-layer embeddings are summed per frame (p.4–5, p.8) |
| Audio output | Discrete RVQ tokens at 12.5 Hz: 32 codebooks × 1024 plus a text-or-pad channel, so 33 channels per step (p.5, p.7). Two variants: (a) delay pattern with 33 linear heads; (b) a Local Transformer that autoregressively expands one backbone latent into the frame's 33 channels, Moshi-style (p.8–10) |
| Waveform decoder | The tokenizer's causal Transformer decoder (≈0.8B), streaming frame by frame (p.4–5) |
| Duplex mechanism | n/a (TTS) |
| Open weights / code | Yes: tokenizer and both generators on Hugging Face and GitHub (p.1, p.23). License not stated |
| Relevance to us | **High.** Same token regime as ours (12.5 Hz multi-codebook discrete tokens from an LLM). Controlled delay-vs-local head comparison, a podcast-heavy data pipeline at millions of hours, and clone-vs-continuation evidence |

## TL;DR

- **Recipe.** "Discrete audio tokens + autoregressive modeling + large-scale pretraining" (p.1). There is no SFT and no RL; all results come from pretrained checkpoints (p.19, p.21).
- **Tokenizer.** A 1.6B pure causal-Transformer RVQ-GAN trained from scratch. A jointly trained 0.5B LM head (ASR and captioning losses) makes its tokens text-aligned (p.4–7). Per the authors, it is the best open tokenizer for speech reconstruction in each bitrate regime from 0.75 to 4 kbps (p.17, Tab.2).
- **Head beats backbone size.** On the same tokenizer and data, the 1.7B Local-Transformer has higher speaker similarity (SIM) than the 8B delay-pattern model on Seed-TTS-eval. It also has lower WER/CER on all 9 CV3 languages (Tab.3–4).
- **Data and schedule.** "Millions of hours" of filtered single-speaker speech: 54.5% podcasts; 53% English, 42% Chinese (Fig.4). Training uses a 4-phase warmup–stable–decay (WSD) curriculum, 32k → 64k context (Tab.1).
- **Continuation prompting.** Continuation means prefixing the reference audio and its transcript. It beats explicit "Clone" conditioning on SIM in every model and language, and it slows speaker drift in hour-long outputs. Drift, not lexical failure, is the main long-form failure (p.19, p.22).

## Architecture

**Tokenizer (p.4–7).**
- *Encoder:* 68 causal Transformer blocks, 10-s sliding-window attention, RoPE.
  - Patchify steps (240 at input, then 2 after layers 12, 24 and 36) take 24 kHz down to 12.5 Hz.
  - Stage widths are 768/768/768/1280 with 12/12/12/32 blocks.
- *Decoder:* mirrors the encoder. The encoder and decoder are "approximately 0.8B parameters"; total capacity is 1.6B.
- *RVQ:* 32 layers of 1024 codes, factorized (dimension 8), L2-normalized. Quantizer dropout p=1.0 gives 125 bps per layer (derived), so 0.125–4 kbps.
- *Semantic supervision:* a 0.5B decoder-only LM (cites Qwen2.5) predicts text from the quantizer output. Tasks: ASR, multi-speaker ASR, captioning.
- *Acoustic losses:* multi-scale mel, multi-period plus complex-STFT discriminators (LSGAN), feature matching.
- *Loss weights:* λ_sem=20, λ_rec=15, λ_cmt=0.25, λ_code=1, λ_adv=1, λ_feat=2.

**Generators (p.7–10).**
- *Loss:* both variants use head-weighted cross-entropy, λ = (1, 3,3,3, 2,2,2, then 1 for the rest). This up-weights RVQ layers 1–6.
- *Delay pattern:* layer j is shifted j−1 frames, so a frame completes only after 32 backbone steps (derived from Eq.10). The authors call it the easiest to scale and deploy.
- *Local Transformer:* runs a 33-step AR loop inside each frame. Steady-state decoding is heavier, but first audio comes earlier. It supports bitrate control via Progressive Sequence Dropout. Its size is not reported.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **Objective:** joint multi-task loss (Eq.8). "All modules are optimized end-to-end without pretrained encoders or semantic teachers" (p.7). It is not stated whether the 0.5B LM loads pretrained weights.
- **Data:** "millions of hours" of public and in-the-wild speech, music and sound (p.4–5). Mixture not reported.
- **Schedule (p.7):**
  - 520k steps without discriminators at batch 1536, about 5 h audio per batch. Derived: ≈2.6M audio-hours seen.
  - Then 500k adversarial steps at batch 768.
- **Not reported:** optimizer, LR, hardware. Details are in the companion paper [11].

### Stage 1: TTS pretraining, four-phase WSD curriculum (P1–P4)
Data subsets (p.15):
- D_basic: the filtered corpus.
- D_clone: timbre pairs.
- D_dict: short-form entries.
- D_noise: noisy-text variants.
- D_phone: pinyin/IPA variants.

| Phase | Context | LR | Data |
|---|---|---|---|
| P1 | 32k | warmup to 2e-4, hold | D_basic only |
| P2 | 32k | hold 2e-4 | all subsets, D_clone "strongly upsample[d]" |
| P3 | 32k | linear 2e-4 → 2e-6 | all subsets, "normal proportions" |
| P4 | 64k | hold 2e-6 | long-form "heavily upsample[d]" |

- **Duration control:** every asset appears twice, with and without an explicit target token count (p.14).
- **Trainable:** backbone, embeddings and heads, end to end (p.9–10).
- **Not reported:** whether the tokenizer is frozen, mixture weights, steps, tokens, batch, optimizer, hardware. Text-only share: none mentioned.

### Stage 2: post-training
None. CV3 is evaluated "without any task-specific fine-tuning or post-training" (p.19), and duration control works "under a pretraining-only setup" (p.21).

## Data pipeline
Source: p.10–14 and Fig.3.

**Stage 1, preprocessing.**
- MossFormer2-SE-48K denoising, done to make diarization reliable, "not to produce the final training signal".
- RMS gain toward −20 dBFS (clamped to ±3 dB), then peak normalization.

**Stage 2, diarization.**
- DiariZen diarization; drop segments shorter than 0.1 s.
- Merge same-speaker neighbours with no gap threshold.
- Keep at most 1 h per recording.

**Stage 3, transcription and transcript QC.**
- MOSS-Transcribe-Diarize ASR.
- Rule filters: empty transcript, more than 6 repeats, or less than 20% linguistic content.
- An LLM diagnoses and repairs transcripts. Only those with the single tag [S1] are kept.

**Stage 4, joint filtering.**
- DNSMOS above 2.8 and AudioBox-PQ above 6.5, both computed on the *pre-denoising* audio.
- Whisper-large-v3 language ID must agree with an LLM's text language ID.
- Per-language bounds on characters per second.

**Synthesis.**
- Clone pairs: take 5 random crops of at most 30 s from each other same-speaker segment, and keep the best WavLM-Large match.
- Text-noise, pinyin/IPA and single-word supplements.

**Composition (Fig.4).**
- Domains: podcast 54.5%, online 12.9%, film/drama 12.9%, broadcast/news 12.1%, audiobook 4.6%, commentary 3.0%.
- Languages: English 53%, Chinese 42%, Spanish 1.43%, …
- About 45% of hours are in 1–10 min segments (read from the chart).
- Everything is single-speaker, so overlap is removed by construction.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Tokenizer, non-adversarial | 520k steps × ≈5 h/batch (derived ≈2.6M h seen) | not reported | batch 1536 |
| Tokenizer, adversarial | 500k steps | not reported | batch 768 |
| TTS P1–P4 | "millions of hours" corpus; tokens not reported | not reported | 32k → 64k context |
| Post-training | none | n/a | n/a |

## Preserving text intelligence / knowledge
Not addressed. The paper does not name the backbone's source LLM, mixes in no text-only data, and runs no text evaluation.

## Key results

**Tokenizer reconstruction (Tab.2, English/Chinese).**

| Tokenizer | kbps (codebooks) | SIM | PESQ-WB |
|---|---|---|---|
| MOSS | 1.0 (8) | 0.88/0.81 | 2.87/2.43 |
| Mimi | 1.1 (8) | 0.74/0.59 | 2.25/1.78 |
| MOSS | 2.0 (16) | 0.95/0.89 | 3.41/2.96 |
| Qwen3-TTS-Tokenizer | 2.2 (16) | 0.95/0.88 | 3.19/2.62 |
| MOSS | 4.0 (32) | 0.97/0.93 | 3.69/3.30 |

**Seed-TTS-eval (Tab.3).**

| Model | Mode | EN WER/SIM | ZH CER/SIM |
|---|---|---|---|
| MOSS-TTS 8B | Clone | 1.92/69.31 | 1.46/76.21 |
| MOSS-TTS 8B | Continuation | 1.84/70.86 | 1.37/76.98 |
| Local-Transformer 1.7B | Clone | 1.87/71.74 | 1.33/77.24 |
| Local-Transformer 1.7B | Continuation | 1.93/73.28 | 1.44/79.62 |
| Qwen3-TTS 1.7B | n/a | 1.50/71.45 | 1.33/76.72 |

**Duration control (Tab.5).** Mean error 0.71% (Chinese) and 0.72% (English).

**Ultra-long generation, longest buckets (Tab.6).**

| Bucket | Error, Clone / Continuation | SIM, Clone / Continuation |
|---|---|---|
| Chinese 10000+ | CER 3.41 / 1.86 | 60.1 / 63.0 |
| English 50000+ | WER 17.49 / 29.52 | 44.4 / 51.2 |

## Ablations & lessons

**Per-frame head beats 4.7× backbone size.** Local Transformer (1.7B) vs delay pattern (8B):
- English SIM: +2.4 in both modes.
- Chinese SIM: +1.0 (Clone) and +2.6 (Continuation) (Tab.3).
- Lower CER/WER in every CV3 language and both modes, e.g. Chinese Continuation 3.68 vs 4.26 (Tab.4).
- It also had "consistently lower per-layer token losses" (p.7).
- Caveat: model size and compute are confounded.

**Delay pattern stays the simpler, long-context workhorse.** It is the variant used for duration control and ultra-long generation (p.7, p.20).

**Continuation anchors speaker identity.**
- SIM gain over Clone: +1.5 English / +0.8 Chinese at 8B, and +1.5 / +2.4 at 1.7B (Tab.3).
- Much flatter 30-minute SIM curves (Fig.6).
- Not uniformly better on content: in the English 50000+ bucket, WER is 29.52 for Continuation vs 17.49 for Clone (Tab.6).

**Speaker drift is the long-horizon bottleneck.** English SIM falls from 64.6 to 44.4 across buckets in Clone mode, and English drifts earlier than Chinese (Tab.6, p.22).

**Curriculum claims (no numbers).**
- Training P1 on D_basic only gives a "substantially stronger initialization" than using the full mixture from step 0 (p.15).
- Oversampling clone pairs too long "biases the model toward prompt copying", so the mix is rebalanced during LR decay (p.15).
- Extending to long context late, at low LR, preserves short-form quality. Without heavy upsampling of long-form data, the 64k window goes unused (p.16).

**Controls can come from data formatting alone.** Duration control (≈0.7% error) and phoneme control (span error 1.00–5.84%) need no fine-tuning (Tab.5, Tab.7).

**Metric caveat.** Below about 2% WER, most remaining errors come from the ASR model, so SIM is the more informative metric (p.18–19).

## Implications for our setup

**Data scale (our inference).** Our 10K h is about two orders of magnitude below "millions of hours". MOSS shows podcasts can supply most of such a corpus after diarization. A large single-speaker *continuation* mid-training stage, before or interleaved with the duplex 10K h, is the best-supported lever on continuation quality.

**Head.** If the Qwen3 LLM emits all codebooks through parallel or delayed heads, this evidence favours a local or depth transformer. Up-weight the first codebooks in the loss (3,3,3,2,2,2).

**Tokenizer ceiling (our inference).** If our tokenizer is the Qwen3-TTS-Tokenizer measured in Tab.2 (12.5 Hz, 16 codebooks, 2.2 kbps), it is roughly on par with MOSS at 2 kbps. Reconstruction is then unlikely to be our main bottleneck.

**Duplex generation is continuation.** MOSS's schedule fits:
- WSD learning rate.
- Hard conditioning introduced while the LR is high.
- Mixture rebalanced during decay.
- Context extended last, at low LR.

**Post-training isn't the gap (our inference).** Strong continuation here comes from pretraining alone, so our weakness more likely lies in data, head or loss.

**Text conditioning (our inference).** All results here are text-conditioned. Without an aligned text (inner-monologue) channel, our duplex outputs are not comparable to these WERs.

## Open questions / unclear points
- **Backbone init.** The 8B and 1.7B sizes match Qwen3 sizes, but the paper never states the initialization.
- **Training budget.** Tokens, steps, batch, optimizer, GPU-hours and per-phase mixture weights are all missing.
- **Which audio is trained on.** Denoised or original is not stated.
- **Local Transformer details.** Its size, and the number of RVQ layers used at evaluation, are not reported.
- **Tab.6 units.** The units of the text-length buckets are not stated.
- **Cross-paper mismatch.** VoxCPM2 reports MOSS-TTS at 1.85/73.4 (English) and 1.20/78.8 (Chinese) on Seed-TTS, which does not match this report.
