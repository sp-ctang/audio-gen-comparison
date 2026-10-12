# Continuous Audio Language Models (Kyutai CALM, incl. Pocket TTS)

| Field | Value |
|---|---|
| Org | Kyutai. First author also at UMR STMS (IRCAM–CNRS–Sorbonne) (p.1) |
| Date / arXiv | arXiv:2509.06926v3, 13 Jan 2026 (v1 Sep 2025, per the arXiv ID) (p.1) |
| Category | method: continuous-latent audio LM, applied to speech continuation, TTS, music continuation and text-to-music. Also releases Pocket TTS |
| Backbone | **Speech continuation:** 2.2B (2560-d, 24 layers, 20 heads), initialized from **Helium-1**, Kyutai's 2B multilingual text LM. **TTS:** 302M (1024-d, 24 layers), from scratch. **Music:** 1.35B (the MusicGen-Medium backbone); a 3B variant also exists (Tab.14, p.23; p.8; p.19) |
| Audio input | Causal VAE-GAN derived from Mimi. Speech: 24 kHz → 12.5 Hz, 32-dim Gaussian latent. Music: 32 kHz → 25 Hz, 128-dim (Tab.13, p.22) |
| Audio output | One continuous latent per frame. A small MLP head samples it in 1 step: a consistency model, or LSD (Lagrangian Self-Distillation) for TTS. The speech head has 6 layers, MLP width 512, 10M params. The discrete baseline is the same backbone with a 701M RQ-Transformer head over 8 RVQ (Tab.14) |
| Waveform decoder | The VAE decoder (causal convolutions + Transformer) |
| Duplex mechanism | n/a. Single-stream speech continuation with an inner-monologue text stream that leads the audio by 2 frames (160 ms) (p.8) |
| Open weights / code | Pocket TTS: github.com/kyutai-labs/pocket-tts (p.1). Training code "will" be released (p.11). Speech-continuation weights: not reported |
| Relevance to us | **High.** With the same text-LM-initialized backbone and the same inner monologue, a continuous head beats an RVQ depth transformer on both acoustic quality and *meaningfulness* of speech continuation |

## TL;DR

- CALM replaces RVQ tokens + RQ-Transformer with VAE latents + a small MLP head trained with continuous-time consistency, so each frame is generated in 1 step. Backbone and head are trained jointly (p.5–7).
- **Speech continuation results.** The backbone is Helium-1 init with an inner monologue, sampled at temperature 0.8. Against the 8-RVQ RQ-Transformer baseline (Tab.2, p.8):
  - text perplexity (PPX) of the transcribed speech: 23.8 vs 26.8;
  - acoustic MOS: 3.45 vs 2.75;
  - meaningfulness Elo: 2023 vs 1870;
  - speed: the sampler is 12.3× faster and full inference 1.3× faster.
- The 32-dim speech VAE matches or beats 8-RVQ Mimi on reconstruction and on ABX phonetic discriminability (Tab.1).
- Noise injection and a short-context transformer are essential for music: FAD is 0.93 with both and 8.38 without either. They brought no gain for speech, so the speech model omits them (p.8, Tab.5).
- Four practical tricks:
  - Gaussian temperature;
  - head batch multiplier;
  - latent CFG;
  - latent distillation.

  Together they yield **Pocket TTS**: 100M parameters, faster than real time on a laptop CPU (p.7–8, App.F).

## Architecture

- **VAE.** A Mimi-like causal encoder/decoder with a KL-regularized Gaussian bottleneck instead of RVQ.
  - Speech: WavLM is distilled (cosine loss) into the *entire* latent, whereas Mimi distils only into codebook 1. Training uses adversarial + feature-matching losses with no reconstruction loss. Reconstruction losses hurt speech and help music (p.5–6, p.8, Tab.13).
- **Backbone.** A causal Transformer produces z_long. Two options exist (p.6):
  - **Noise injection** (training only): x̃ = √k·ε + √(1−k)·x, with k ~ U(0,1).
  - **Short-context Transformer:** reads the last K = 10 *clean* latents and outputs z_short.
  - Both are used for music only (Tab.14).
- **Head.** An MLP consistency model is conditioned on z_long + z_short. It is trained with the continuous-time consistency loss with adaptive weighting (Lu & Song 2025) on the TrigFlow path (p.6–7).
  - The LSD alternative splits each batch: flow-matching loss on 75%, LSD loss on 25% (p.17).
- **Speech continuation.** Text tokens and latents are summed at the input. A linear layer on z_long predicts text, which also conditions the head (p.8).
- **Sampling-side tricks:**
  - **Gaussian temperature:** noise std is scaled by √τ (p.7).
  - **Latent CFG:** Z = Z∅ + α(Z_C − Z∅), applied on backbone outputs (p.7).
  - **Latent distillation:** an L2 loss trains a (smaller) student backbone to output Z_CFG, while the teacher's head is kept (p.7–8).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training

- **Speech VAE:**
  - Format: 24 kHz, 12.5 Hz, 32 dims; conv ratios 6,5,4,4,4; 8+8 Transformer layers; 10 s context.
  - Optimization: batch 64 × 12 s; KL weight 0.01; distillation weight 25; cosine LR 8e-4.
  - Steps and GPUs: not reported (Tab.13).
- **Music VAE:** 32 kHz, 25 Hz, 128 dims, with reconstruction loss and no semantic distillation (Tab.13). At least 96 dims are needed to match a 32-RVQ codec (Tab.7, p.17).

### Stage 1: speech-continuation CALM (the mid-training analogue)

- **Init:** Helium-1 (2B). All modules are trained jointly (p.6–8).
- **Data:** "French and English speech data following Défossez et al. (2024b)", i.e. Moshi (p.8).
  - Hours, sources, text-only share: not reported.
  - Text alignment method: not reported.
- **Optimization:**
  - 300 s samples, batch 144, 150k steps, 48 H100.
  - AdamW (0.9, 0.95), cosine schedule, head batch multiplier 8.
  - Tab.14 lists two learning rates: 5e-5 (backbone block) and 2e-4 (training block).
  - No noise injection and no short-context transformer (Tab.14).
- **Audio seen:** ≈1.8M h (derived: 150k × 144 × 300 s = 6.48e9 s).
- **Baseline:** identical backbone and data, with a 701M RQ-Transformer over 8-RVQ Mimi (Tab.14).

### Stage 2: TTS CALM (from scratch)

- **Model:** 302M backbone + 10M head (Tab.14); Tab.3 reports 313M total. Text enters as a prefix (SentencePiece, 4k vocabulary), along with an audio-prompt prefix (p.9).
- **Data:** 88k h of public English (App.D).
- **Optimization:** batch 128 × 60 s, 400k steps, 8 H100, LR 1e-4, cosine (Tab.14).
- **Audio seen:** ≈853k h (derived: 400k × 128 × 60 s), about 9.7 epochs (derived).

### Stage 3: latent CFG + latent distillation → Pocket TTS

- The teacher is the 313M model with a 24-layer backbone; latent CFG uses α = 1.5. The student is a 6-layer backbone that keeps the same MLP head (p.8, p.20).
- Size: 90M generative + 20M VAE. Steps and compute: not reported (p.20).

### Stage 4: music CALM

- **Model:** 1.35B backbone; 601M head (12 layers); 113M short-context transformer; noise injection on.
- **Data:** 400K songs ≈ 20K h from LAION-Disco-12M.
- **Optimization:** batch 48 × 30 s, 500k steps, 16 H100, LR 1e-4 (p.10, Tab.14).

## Data pipeline

- **Speech continuation:** no detail beyond "following Moshi". In the inner monologue, each word sits at the timestep where it is spoken (p.8).
- **TTS:** AMI, Earnings22, GigaSpeech, SPGISpeech, TED-LIUM, VoxPopuli, LibriHeavy and Emilia, 88k h in total (App.D).
- **No duplex, overlap or synthetic-dialogue data.**
- **Evaluation:**
  - **PPX:** 1,000 × 30 s generations are transcribed with Whisper and scored by Mistral-7B.
  - **VERT:** a diversity score (geometric mean of self-BLEU and auto-BLEU; lower = more diverse).
  - **Human:** 50 raters each scored 10 of 50 English prompts (3 s prompt, 30 s clips) (p.8–9, p.21).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Speech VAE | n.r. (batch 64 × 12 s) | n.r. | LR 8e-4 (Tab.13) |
| Speech continuation | ≈1.8M h seen (derived); 150k steps | 48 H100 × n.r. | Helium-1 init (Tab.14) |
| TTS | 88k h data; ≈853k h seen (derived) | 8 H100 × n.r. | 400k steps |
| Pocket TTS distillation | n.r. | n.r. | 24 → 6 layers (p.20) |
| Music | ≈20K h; 500k × 48 × 30 s | 16 H100 × n.r. (Fig.3 axis spans ~4 days) | (p.10) |

n.r. = not reported.

## Preserving text intelligence / knowledge

- **Mechanism:** Helium-1 init plus an inner-monologue text stream placed 160 ms ahead of the audio, described as "a semantic scaffold" (p.8).
- **Key observation:** CALM beat the RVQ baseline on meaningfulness even though both had the text stream. The authors' hypothesis: "less model capacity in the backbone being allocated to audio manipulation, allowing more for text prediction" (p.9).
- **Not reported:** text-only replay data, knowledge or QA benchmarks, or how much of Helium-1's text ability is retained.

## Key results

- **Speech VAE vs Mimi 8 RVQ** at 12.5 Hz (Tab.1):

  | Metric | VAE | Mimi 8 RVQ |
  |---|---|---|
  | MOSNet | 3.15 | 3.11 |
  | ABX (↓) | 8.1% | 9.4% |
  | PESQ | 2.42 | 2.13 |
  | STOI | 0.90 | 0.87 |
  | MUSHRA | 66.0 | 57.7 |

- **Speech continuation** (Tab.2; PPX and VERT lower is better; reference audio scores PPX 20.2, MOS 4.02, Elo 2180):

  | Model | Temp | PPX | VERT | MOS | Elo |
  |---|---|---|---|---|---|
  | RQ-Transformer | 1.0 | 52.4 | 36.3 | 2.42 | 1841 |
  | RQ-Transformer | 0.8 | 26.8 | 33.1 | 2.75 | 1870 |
  | CALM | 1.0 | 42.9 | 34.3 | 2.82 | 1947 |
  | CALM | 0.8 | 23.8 | 31.2 | 3.45 | 2023 |

- **TTS, LibriSpeech test-clean** (Tab.3). CALM-LSD 313M scores WER 1.81, SIM 0.52, MUSHRA 61.1. It beats DSM 750M (1.95 / 0.67 / 60.2) and F5-TTS (2.42 / 0.66 / 54.7) on WER and MUSHRA, but trails both on SIM. The reference scores MUSHRA 61.8.
- **Pocket TTS, 100M** (Tab.12): WER 1.84, audio-quality Elo 2016, speaker-similarity Elo 1898, and faster than real time on CPU (Apple M3, Intel Core Ultra 7 165H). For comparison, DSM 750M: 1.84 / 1959 / 2037; Chatterbox Turbo: 3.24 / 2055 / 2012.

## Ablations & lessons

- **Continuous beats discrete at equal backbone and data.** CALM wins on every metric at both temperatures (Tab.2). A 10M MLP head beats a 701M RQ-Transformer head (Tab.14).
- **Sampling temperature is first-order for both model types.** Lowering T from 1.0 to 0.8 cuts PPX from 52.4 to 26.8 (RQ-Transformer) and from 42.9 to 23.8 (CALM), and raises CALM's MOS from 2.82 to 3.45 (Tab.2). Gaussian temperature tracks discrete temperature in its effect on speaker diversity (Fig.2).
- **The RVQ sampler is expensive.** It takes 26.7% of speech inference time vs 2.9% for the consistency head (Tab.2); in music, 57.7% vs 6.6% (Tab.4).
- **Error accumulation, music (Tab.5):**

  | Variant | FAD |
  |---|---|
  | Full model | 0.93 |
  | Without noise injection | 1.63 |
  | Without short-context transformer | 4.03 |
  | Without both (≈ MAR) | 8.38 |

- **Error accumulation, speech:** speech with an inner monologue needed neither component (p.8).
- **Head batch multiplier.** Without it, FAD is 1.32 vs 0.93, and convergence is slower (Fig.3). N = 8 was the memory limit (p.10).
- **Only consistency works at ≤10 steps.** At 4 steps, FAD is 0.71 (consistency) vs 28.83 (TrigFlow). TrigFlow wins only at 100 steps (0.64), and is then slower than real time (Tab.9). Consistency degrades with more steps (2.05 at 100).
- **LSD beats consistency for ~300M TTS.** MUSHRA 61.1 vs 46.8, WER 1.81 vs 1.93, SIM 0.52 vs 0.40 (E.5 table, p.19).
- **Latent CFG.**
  - On text: WER 2.39 → 1.81, MUSHRA 55.5 → 61.1.
  - Also on the audio prefix: SIM rises to 0.54 but MUSHRA drops to 56.2 (E.5).
- **Distillation 24 → 6 layers keeps WER.** Pocket TTS scores 1.84, the same as DSM 750M (Tab.12).
- **Scaling to 3B helps both model types.** FAD goes 0.71 → 0.62 (CALM) and 1.06 → 0.98 (RQ-Transformer); the gap persists (Tab.10).

## Implications for our setup

Setup: Qwen3 LLM + Qwen3 tokenizer/vocoder (discrete), ~10K h duplex mid-training; weak continuation and semantics.

1. **Sweep decoding temperature and top-p first.** Tab.2 shows T 1.0 → 0.8 halves PPX for an RVQ model, so part of a "weak continuation" can be a decoding artifact.
2. **Audit where acoustic detail is modelled.** CALM's explanation for its meaningfulness gain is backbone capacity freed from audio. If the Qwen3 LLM itself carries most of the codebook prediction, move the acoustic detail into a separate head (depth, flow, or consistency).
3. **The larger lever: a continuous output.**
   - Predict VAE latents with a 1-step head (consistency, LSD, or drifting; see the Pocket TTS web note) conditioned on the LLM's hidden state.
   - This needs a semantic VAE (Mimi-style, with WavLM distilled into the whole latent: ABX 8.1 vs 9.4%) to replace the Qwen3 vocoder on the output side. The input side can stay as is.
4. **Keep a word-aligned inner monologue** leading audio by about 2 frames. Both CALM and its baseline rely on it (see the DSM note for the alignment recipe).
5. **Scale.** CALM's speech model saw ≈1.8M h-equivalent of audio (derived) in 300 s contexts, starting from a 2B text LM. 10K h of duplex data cannot carry continuation quality alone: put large-scale single-stream speech with text first.
6. **Use a modest backbone LR when starting from a text LLM.** CALM lists 5e-5 for the Helium-initialized backbone, vs 1e-4 to 2e-4 elsewhere (Tab.14). This is not ablated.
7. **Test the tokenizer before blaming the LM.** For example, check whether Qwen3 tokens carry semantic content with an ABX test.

## Open questions / unclear points

- **Speech-continuation data:** amount, composition and text-only share are not reported, and neither is the alignment method.
- **Two learning rates for speech continuation** (5e-5 vs 2e-4, Tab.14) are not explained.
- **Text knowledge:** no text or spoken-QA evaluations, so Helium-1 knowledge retention is unknown.
- **Speaker-similarity puzzle:** passing the reference through the VAE drops WavLM similarity to 0.57, despite good reconstruction metrics (Tab.3).
- **Pocket TTS claims vs Tab.12:**
  - "Lowest WER" is in fact tied with DSM (1.84).
  - Chatterbox Turbo has the highest audio-quality Elo (2055).
  - Pocket TTS has the lowest speaker-similarity Elo (1898).
  - No ground-truth row exists in Tab.12.
- **VAE size:** 20M (p.20) vs the Kyutai tech report's 10M decoder + 18M encoder (https://kyutai.org/blog/2026-01-13-pocket-tts/).
- **Table references:** the E.5 text cites "Tab. 10" for the TTS ablation, but that table is unnumbered.
- **RTF convention:** Tab.9's RTF is time over duration (lower = faster), the opposite of DSM's convention.
- **Untested settings:** dialogue, duplex and overlapping speech.
