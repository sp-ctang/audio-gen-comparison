# VibeVoice

| Field | Value |
|---|---|
| Org | Microsoft Research (p.1) |
| Date / arXiv | arXiv 2508.19205v1, 26 Aug 2025 |
| Category | TTS: long-form multi-speaker conversational ("podcast") generation, with a continuous speech tokenizer |
| Backbone | Qwen2.5 at 1.5B and 7B (VibeVoice-1.5B / -7B) (p.3) |
| Audio input | Voice prompts go through the acoustic σ-VAE tokenizer (24 kHz → 7.5 Hz continuous latents, causal). Generated speech is re-encoded by both the acoustic and the semantic tokenizer into a "hybrid" input (p.3, Fig.2) |
| Audio output | One continuous acoustic VAE latent per step at 7.5 Hz. A 4-layer token-level diffusion head, conditioned on the LLM hidden state, produces it (DDPM noise prediction, CFG 1.3, 10 DPM-Solver++ steps) (p.3) |
| Waveform decoder | Acoustic tokenizer decoder (mirror of the encoder, ≈340M params) (p.3) |
| Duplex mechanism | n/a. Turns are generated one after another in a single stream with speaker IDs; overlapping speech is "not explicitly model[led]" (p.5) |
| Open weights / code | Code on GitHub, weights on Hugging Face (p.1). "Research and development purposes only" (p.6) |
| Relevance to us | **Medium.** The only paper in this set that generates multi-speaker conversation (≤4 speakers, ≤90 min) from an LLM backbone, and it shows LLM-scaling evidence. But it gives almost no data or compute details and does not model overlap |

## TL;DR

- **Next-token diffusion.** A Qwen2.5 LLM reads voice-prompt latents plus role-tagged scripts. It autoregressively generates 7.5 Hz continuous acoustic latents, each produced by a 4-layer diffusion head (p.2–3).
- **Very low frame rate.** The causal σ-VAE tokenizer downsamples 3200×, which gives roughly 2 speech tokens per BPE text token (p.2). That allows up to 90 min with 4 speakers in a 64K context (p.1).
- **Scaling the LLM.** Going from 1.5B to 7B raises average subjective MOS from 3.54 to 3.76 and SIM from 0.548 to 0.692, with WER roughly flat (Tab.1).
- **Podcast test.** On 8 scripts (~1 h total), VibeVoice-7B beats Gemini 2.5 Pro preview TTS (avg 3.66) and ElevenLabs v3 alpha (3.40) on MOS, with the lowest WER (Tab.1).
- **Gaps.** Training data, steps, LR and compute are not reported. There is no post-training. English and Chinese only.

## Architecture

**Acoustic tokenizer (p.3)**
- A σ-VAE: the variance is fixed (σ ~ N(0, C_σ)) rather than learned, "to mitigate potential variance collapse" when the latents are modeled autoregressively.
- Mirror-symmetric encoder and decoder, each 7 stages of Transformer-style blocks in which self-attention is replaced by 1D depth-wise *causal* convolutions, for streaming.
- 6 downsampling layers reduce 24 kHz audio by 3200×, to 7.5 Hz.
- About 340M params each for encoder and decoder. Trained with DAC losses and discriminators.
- Latent dimension: not reported.

**Semantic tokenizer (p.3)**
- Same hierarchy as the acoustic encoder, without the VAE.
- Trained on ASR as a proxy task, with Transformer decoder layers that are discarded afterwards.
- Used only to encode generated speech for the next step's input.

**LLM input (p.3)**
- The sequence is: [Speaker_k: z_k] voice-prompt latents, then [Speaker_k: T_k] scripts, then the generated speech.
- Each generated segment is encoded by both tokenizers and the results are combined (Fig.2 draws a "+").

**Diffusion head (p.3)**
- 4 layers, conditioned on the hidden state h_i, trained to predict the noise added to the clean acoustic latent.
- Inference uses CFG (scale 1.3) and DPM-Solver++ with 10 steps.
- Derived: 90 min ≈ 40,500 latents (90 × 60 × 7.5), which fits a 64K context alongside the text.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- Acoustic tokenizer: DAC objective and discriminators.
- Semantic tokenizer: ASR objective.
- **Not reported:** data, steps, LR, compute.

### Stage 1: LLM + diffusion-head training
- The tokenizers are frozen. Only the LLM (initialized from Qwen2.5 1.5B or 7B) and the diffusion head train (p.3).
- Curriculum on input sequence length: 4,096 → 65,536 tokens (p.3).
- Objective: diffusion noise-prediction loss on the acoustic latents. How turn and end tokens are predicted is not described.
- Data: "primarily trained on long-form speech" (p.4). Hours, sources and conversational share are not reported. Text-only share: none mentioned.
- **Not reported:** LR, batch, steps.

### Stage 2: post-training
None reported.

## Data pipeline
- What is stated: English and Chinese only (p.5), "primarily … long-form speech" (p.4), and speaker-tagged script input.
- Nothing is said about synthetic dialogue generation, backchannels or silence handling.
- Overlap is explicitly not modelled (p.5).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Tokenizers | not reported | not reported | Acoustic: ≈340M encoder + ≈340M decoder |
| LLM + diffusion head | not reported | not reported | Context curriculum 4K → 64K |
| Inference | n/a | not reported | 10 diffusion steps per 133 ms latent (derived: 1/7.5 s) |

## Preserving text intelligence / knowledge
- The LLM is initialized from Qwen2.5 and fully trainable. It "interpret[s] complex user inputs, including detailed text sentences and role assignments" (p.2).
- No text-only data, no frozen layers, no text evaluation.
- Qualitatively, the 7B model gives "richer timbre, more natural intonation, and enhanced transfer capabilities, such as in cross-lingual applications" (p.2).

## Key results

**Long conversational test (Tab.1).** 8 transcripts totalling about 1 h; 24 annotators, each listening to about 6 h (p.4).

| Model | Realism | Richness | Pref. | Avg | WER Whisper/Nemo | SIM |
|---|---|---|---|---|---|---|
| VibeVoice-7B | 3.71 | 3.81 | 3.75 | 3.76 | 1.29/1.95 | 0.692 |
| VibeVoice-1.5B | 3.59 | 3.59 | 3.44 | 3.54 | 1.11/1.82 | 0.548 |
| Gemini 2.5 Pro TTS | 3.55 | 3.78 | 3.65 | 3.66 | 1.73/2.43 | – |
| ElevenLabs v3 α | 3.34 | 3.48 | 3.38 | 3.40 | 2.39/2.47 | 0.623 |
| Higgs Audio V2 | 2.95 | 3.19 | 2.83 | 2.99 | 5.94/5.97 | 0.543 |
| Sesame CSM | 2.89 | 3.03 | 2.75 | 2.89 | 2.66/3.05 | 0.685 |

Objective metrics only: Nari Dia WER 11.96, SIM 0.541; MoonCast WER 2.81, SIM 0.562.

**Short utterances (Tab.2, SEED test sets, 1.5B only).**
- test-zh: CER 1.16, SIM 0.744.
- test-en: WER 3.04, SIM 0.689.
- For comparison, 25–50 Hz systems score 1.98 (Spark-TTS) and 2.57 (CosyVoice 2) on test-en.

**Tokenizer reconstruction (Tab.3, LibriTTS test-clean).** VibeVoice's single latent stream at 7.5/s scores PESQ 3.068, STOI 0.828, UTMOS 4.181. For comparison:
- WavTokenizer at 75 tokens/s: 2.373 / 0.914 / 4.049.
- Encodec with 8 codebooks (600 tokens/s): 2.72 / 0.939 / 3.04.

## Ablations & lessons

**LLM scale.** With the same architecture, going from 1.5B to 7B changes Preference +0.31, Realism +0.12, Richness +0.22 and SIM +0.144, while WER is slightly worse (1.11 → 1.29, Whisper) (Tab.1). The text's claim that 7B is better "on all objective metrics … while maintaining a comparable WER" is loosely worded (p.4).

**Separate acoustic and semantic tokenizers.** "In our experiments, generating long-form speech benefits from this separate design" (p.3). No numbers are given.

**7.5 Hz frame rate.**
- It cuts decoding steps and makes hour-long contexts feasible (p.4).
- Chinese short-utterance quality holds up, but English WER trails the 25–50 Hz systems (Tab.2).
- The tokenizer's STOI (0.828) is below most higher-rate codecs (e.g., Encodec-8 0.939, WavTokenizer-75 0.914), although its PESQ and UTMOS are the best in Tab.3.

**Fixed-variance σ-VAE.** Chosen so continuous latents stay modellable by AR plus diffusion (p.3). Not ablated.

**Long-form robustness separates systems.** Dialogue models built for shorter outputs degrade badly on hour-long scripts: Dia WER 11.96, Higgs 5.94 (Tab.1).

**No overlap.** The authors list overlapping speech as unmodelled (p.5). Generating turns sequentially in one stream does not capture duplex phenomena.

## Implications for our setup

**Conversational style is learnable, but this is not duplex.** VibeVoice shows that an LLM-initialized AR model can produce natural multi-speaker turn-taking at hour scale. It is single-stream and has no overlap, so it does not address two-stream duplex. Podcast-style data of this kind could feed our continuation mid-training (our inference).

**Backbone size matters for perceptual quality.** 7B vs 1.5B is the largest MOS and SIM effect in the paper. If our Qwen3 backbone is small, part of our continuation weakness may be capacity. Because data and compute are not reported, this is only directional (our inference).

**Feed back a text-aligned view of the model's own speech (hypothesis, not ablated).** Re-encoding generated speech with an ASR-trained encoder and adding it to the acoustic latents gives the LLM a text-aligned view of its own output. That could help our semantic-knowledge problem if our tokenizer's codes are weakly text-aligned.

**Frame rate drives context cost.** Two duplex streams at 12.5 Hz cost 25 positions per second, against VibeVoice's 7.5 (derived).

**Little to borrow for data or compute planning.** The paper reports neither.

## Open questions / unclear points
- Training data (hours; sources; conversational vs read; real vs synthetic), steps and compute.
- How speaker turns and end-of-generation are predicted, and whether the LLM also emits text or speaker tokens.
- How acoustic and semantic features are fused (sum or concat; dimensions), the latent dimension, and the value of C_σ.
- The subjective test is small (8 scripts). Fig.1 bar values differ from Tab.1 by about 0.01.
