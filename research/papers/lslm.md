# Language Model Can Listen While Speaking (LSLM)

| Field | Value |
|---|---|
| Org | SJTU X-LANCE Lab + ByteDance (p.1) |
| Date / arXiv | 5 Aug 2024, arXiv 2408.02622v1 (p.1) |
| Category | method: full-duplex (interruption-only) modeling on top of a token-based TTS LM. Not a dialogue S2S model |
| Backbone | 106M decoder-only Transformer (12 layers, 12 heads, d=768, FFN 3072), first trained from scratch as a TTS model. No text LLM (p.6, p.10) |
| Audio input | Listening channel: streaming vq-wav2vec (34M, 20-layer 1-D CNN) continuous features → linear projection (p.5–6). Frame rate not reported |
| Audio output | A single layer of vq-wav2vec discrete tokens, plain AR (no RVQ, no NAR stage) (p.4). Text-conditioned (TTS). Token rate and vocab size not reported |
| Waveform decoder | GAN-based token-to-waveform vocoder from UniCATS [12]; an acoustic prompt supplies the timbre (p.5–6) |
| Duplex mechanism | Dual-channel: listening features fused into the speaking decoder at every step; an IRQ token stops speech when interrupted (p.5–6). Stop-only: no response, backchannel or turn re-entry |
| Open weights / code | Demo page only (p.1). Code and weights not mentioned |
| Relevance to us | low–medium: tiny TTS-scale study, but clean fusion and freezing ablations for adding a listening stream |

## TL;DR

- Defines full-duplex modeling as P(r_t \| R_<t, S_<t, C): each next speaking token is conditioned on the real-time listening stream S (p.4, Eq.3).
- Speaking is an AR TTS over single-layer vq-wav2vec tokens. Listening uses the same encoder's streaming continuous features. An IRQ token is added to the vocabulary (p.4–6).
- **Middle fusion wins.** It adds the listening features to the input of every Transformer block. TTS WER (clean): middle 4.05, early 33.56, late 4.37 (p.9, Tab.2).
- With unseen speakers and words plus noise, the speaking quality and interruption detection both drop: WER 8.50 (vs 4.28 for vanilla TTS) and F1 85.15 (p.9, Tab.3).
- The best setup loads the pretrained TTS backbone and the SSL encoder and fine-tunes both. Freezing the listening encoder hurts interruption recall (p.10, Tab.4).

## Architecture

- **Speaking:** a decoder-only AR TTS conditioned on text C. Targets are vq-wav2vec tokens. A single token layer was chosen so generation never waits for an AR→NAR handoff (p.4). The vocoder takes the acoustic prompt for timbre, so the AR model can concentrate on semantic content (p.5).
- **Listening:** the same SSL encoder, which must be streaming. Its continuous embedding goes through a linear projection and is not discretized (p.5–6).
- **Fusion variants** (p.6, Fig.3):
  - Early: combine the channels at the input embeddings.
  - Middle: add the listening channel to the input of each Transformer block, together with the hidden states and positional embeddings.
  - Late: combine at the output logits, before the softmax.
- **Inference:** top-p 0.99, temperature 1.0 (p.7). The IRQ probability stays below 1e-3 while there is no interruption. It rises shortly after an interruption starts, until IRQ gets sampled (p.9, Fig.4).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
vq-wav2vec is used as a pretrained SSL encoder and quantizer (p.6). The vocoder comes from UniCATS [12]. Its training is not described here.

### Stage 1: Vanilla TTS
The decoder is trained from scratch on LibriTTS (585 h) as a plain TTS model, reaching WER 4.28 (p.7, p.10 Tab.4). Its hyperparameters are not reported separately.

### Stage 2: listening-while-speaking (FDM) training
- **Objective:** Eq.11. NLL of the speaking tokens up to IRQ (when there is a turn-taking event) or up to EOS (p.6).
- **Trainable modules:** in the best configuration, both the TTS backbone and the SSL encoder are loaded from pretrained weights and fine-tuned. The projection is new (p.6, p.10).
- **Data:** LibriTTS 585 h TTS pairs, plus interruption clips, plus MUSAN Freesound noise.
  - Each sample independently gets noise (p=0.5) and an interruption (p=0.5) in the listening channel.
  - When interrupted, the target emits IRQ µ=0.5 s after the interruption starts and drops the remaining speech tokens (p.7).
- **Optimizer:** AdamW, max LR 5e-4, no weight decay, batch size 4, 5,000 warmup steps then cosine decay, 20 epochs. The checkpoint with the best validation loss is kept (p.7).
- **Compute:** not reported.

## Data pipeline

- **Command-based FDM** (speaker-dependent): the interruption word "Honey" is synthesized in the timbres of 22 "boutique" SEED-TTS speakers (p.7).
- **Voice-based FDM** (speaker-independent): one-second Speech Commands clips, split 8:1:1 into 51,088 / 6,798 / 6,835 clips. No test speaker appears in training (p.7).
- **Noise:** the Freesound part of MUSAN, including high-frequency sounds (phone ringing, explosions) and low-frequency ones (white noise, traffic). It exists so the model cannot learn the trivial rule of stopping on any input (p.7).
- All overlap is synthetic mixing. There is no conversational data, no backchannels and no turn re-entry.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| TTS (stage 1) | LibriTTS 585 h | not reported | trained from scratch |
| FDM (stage 2) | LibriTTS 585 h + interruption/noise clips, 20 epochs | not reported | batch 4, LR 5e-4 (p.7) |

## Preserving text intelligence / knowledge

n/a. There is no LLM; the "context" is the text to be synthesized.

## Key results

- **Command FDM** (Tab.2, p.9), values clean / noisy listening channel:

| Model | TTS WER | Interruption F1 | Note |
|---|---|---|---|
| Vanilla TTS | 4.28 | – | no listening channel |
| Early fusion | 33.56 / 34.99 | 98.10 / 97.20 | |
| Middle fusion | 4.05 / 4.51 | 98.00 / 97.38 | |
| Late fusion | 4.37 / 6.87 | 97.89 / 94.89 | precision 93.06 under noise |

- **Voice FDM, middle fusion** (Tab.3, p.9):
  - Clean: WER 5.33, P/R/F1 95.21 / 95.78 / 95.50.
  - Noise: WER 8.50, P/R/F1 87.69 / 82.77 / 85.15.
- **Protocol** (p.7–8): Whisper large-v3 WER on LibriTTS-testsetB (500 utterances, 37 unseen speakers). An interruption counts as handled if the model stops within 1 s ([0, 2µ]). 1,000 interaction test utterances, half interrupted.

## Ablations & lessons

- **Where the listening stream enters matters.** Early fusion (merged input embeddings) wrecked speech generation: WER 33.56 vs 4.05 for middle fusion. The authors attribute this to the model being unable to tell the two channels apart (p.8, Tab.2). Late fusion kept WER low but lost the ability to tell noise from voice (precision 93.06 under noise) (p.8).
- **Training regime** (Tab.4, p.10; speaking module / listening module):

| Speaking | Listening | WER | F1 | Recall |
|---|---|---|---|---|
| scratch | frozen | 4.82 | 97.89 | |
| scratch | fine-tuned | 4.67 | 95.79 | |
| frozen | frozen | 6.64 | 90.18 | 83.60 |
| frozen | fine-tuned | 4.64 | 97.89 | |
| fine-tuned | frozen | 4.46 | 94.44 | 92.54 |
| fine-tuned | fine-tuned | 4.05 | 98.00 | |

  - Lesson: initialize from the pretrained speaker and keep it trainable.
  - Also fine-tune the listening encoder. The authors' explanation is that the SSL encoder never saw noisy mixtures in pretraining (p.10).
- **The listening stream leaks into speaking quality.** With a clean listening channel, adding it slightly *improved* TTS WER (4.05 vs 4.28). With noisy input or unseen speakers it degraded WER to 8.50 (p.8–9).
- **Generalizing to unseen speakers and words is the hard part.** F1 is 95.50 (clean) / 85.15 (noise) in the voice-based setting, vs 98.00 / 97.38 for a fixed command (p.8–9).

## Implications for our setup

- **Test how the user stream is injected.** If our duplex model sums the user-stream embedding into the input (early fusion), LSLM is small-scale evidence that this can damage the assistant stream's generation. Cheap diagnostic: compare assistant-stream continuation quality with the user stream set to silence, to real user audio and to noise. In LSLM, WER moved from 4.05 (clean) to 4.51–8.50 depending on what was in the listening channel.
- **Keep the user-stream encoder trainable.** A frozen pretrained encoder that never saw overlapped or noisy conversational audio is a bottleneck: recall was 83.6–92.5 when frozen vs 98.2 when fine-tuned.
- **Their interruption recipe is cheap but narrow.** p=0.5 noise, p=0.5 interruption, stop target 0.5 s after onset. It is a usable augmentation but only teaches "stop". It says nothing about content quality or semantic knowledge.
- **Scale caveat.** 106M parameters, 585 h, a from-scratch TTS model. This is not evidence about knowledge at LLM scale.

## Open questions / unclear points

- The vq-wav2vec token rate and vocabulary size are not reported. It is also not stated whether the listening features are taken before or after quantization.
- The exact combination operator for early and late fusion (sum vs concatenation) is not specified beyond Fig.3.
- No latency measurement beyond the 1 s success window, and no compute.
