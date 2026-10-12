# SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models

| Field | Value |
|---|---|
| Org | Fudan University (p.1) |
| Date / arXiv | arXiv 2308.16692v2, 23 Jan 2024; ICLR 2024 (p.1) |
| Category | codec / tokenizer. Also introduces the SLMTokBench benchmark and USLM, a zero-shot TTS model built on the tokenizer |
| Backbone | Tokenizer: EnCodec-style convolutional encoder-decoder; the encoder ends in a 2-layer BiLSTM, the decoder uses LSTM; 8 RVQ layers. USLM: AR + NAR models, each a 12-layer Transformer decoder (16 heads, d=1024, FFN 4096), trained from scratch (p.4, p.6, p.16) |
| Audio input | 16 kHz waveform; strides (2, 4, 5, 8) give 50 Hz frames (derived: 16,000 / 320) (p.6, p.16) |
| Audio output | 8 RVQ codebooks. **RVQ-1 is distilled toward HuBERT and carries content**; RVQ-2:8 are residual and carry timbre and other paralinguistics. Codebook size not reported. USLM: an AR model predicts RVQ-1 from phonemes; a NAR model predicts RVQ-2:8 (p.4–6) |
| Waveform decoder | The SpeechTokenizer decoder, a mirror of the encoder, GAN-trained with MS-STFT + multi-period + multi-scale discriminators (p.5, p.16) |
| Duplex mechanism | n/a |
| Open weights / code | Code and models at github.com/ZhangXInFD/SpeechTokenizer (p.1) |
| Relevance to us | medium-high: direct evidence that a semantically supervised first codebook makes tokens far more text-aligned at no reconstruction cost |

## TL;DR

- **Idea:** distill HuBERT into RVQ layer 1 of an EnCodec-style codec, and let the residual layers carry everything else (p.2, p.4–5).
- **RVQ-1 becomes as text-aligned as HuBERT units.** On SLMTokBench, RVQ-1 mutual information with text (MI) is 30.9–32.9 vs 31.2 for HuBERT units and 16.5 for EnCodec's RVQ-1. Phone purity (PNMI) is 0.71 vs 0.43 (HuBERT KM500) vs 0.28 (EnCodec RVQ-1) (p.8, p.18).
- **Full 8-layer reconstruction is on par with or better than EnCodec:** WER 5.04 vs 5.11, MUSHRA 90.55 vs 79.86 (p.7).
- **Hierarchical TTS beats VALL-E:** AR(RVQ-1) + NAR(RVQ-2:8) scores WER 6.5 vs 7.9, speaker similarity 0.84 vs 0.75, MOS 3.63 vs 3.08 (p.8).
- **Design ablations** (p.8, p.14–15): continuous HuBERT features beat discrete HuBERT units as the teacher; a per-dimension ("D-axis") cosine loss beats a per-timestep one; a BiLSTM beats an LSTM for text alignment.

## Architecture

- **Encoder** (p.16):
  - Conv layer (32 channels, kernel 7).
  - 4 residual blocks with strided downsampling (strides 2, 4, 5, 8); channels double at each downsampling.
  - 2-layer BiLSTM, then a conv layer to the embedding dimension D.
- **Decoder:** mirrors the encoder, using transposed convs and an LSTM (p.16).
- **RVQ** follows EnCodec (p.16): EMA codebook updates (decay 0.99), replacement of dead codes, straight-through estimator, commitment loss.
- **Semantic distillation on the RVQ-1 output** (p.4–5). Two variants:
  - Continuous: maximize the cosine similarity, *per feature dimension across time*, between the projected RVQ-1 output and HuBERT layer-9 (L9) features or the average over all HuBERT layers. L = −(1/D)·Σ_d log σ(cos(A·Q1^(:,d), S^(:,d))).
  - Pseudo-label: cross-entropy against HuBERT units.
- **Generator loss** (p.5): time-domain L1 + multi-scale mel L1+L2 (64 bins, windows 2^5 to 2^11) + hinge GAN loss + feature matching + commitment + distillation. The λ weights are not reported.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **Data:** LibriSpeech, random 3 s crops (p.6).
- **Hardware / schedule:** 2×A800, 20 epochs, max LR 4e-4, batch 20 per GPU (p.6).

### Stage 1: USLM (zero-shot TTS on top of the tokens)
- **Data:** MLS English, 44k h, clips of 3–14 s (p.6).
- **AR model:** predicts RVQ-1 conditioned on phonemes.
- **NAR model:** predicts RVQ-2:8 conditioned on phonemes, a 3 s acoustic prompt, and the earlier layers (p.6).
- **Hardware / schedule:** 8×A800, 500k steps, max LR 5e-4. Batch is 7,500 tokens per GPU (AR) and 5,000 tokens per GPU (NAR) (p.6).
- **Baseline:** VALL-E with EnCodec 24 kHz at 6 kbps, trained on the same setup (p.6).

## Data pipeline

- LibriSpeech for the tokenizer, MLS-English for TTS, everything at 16 kHz (p.6).
- **How SLMTokBench measures text alignment** (p.3, p.13):
  - A 2×1024 BiLSTM probe with CTC loss is trained on LibriSpeech train-clean-100 (batch 32, LR 1e-4, 200k steps).
  - MI is estimated on dev-clean with a vCLUB bound; probe WER is measured on the test set.
- **How it measures information preservation** (p.3):
  - Tokens are resynthesized and transcribed with Whisper medium-en (WER).
  - Speaker similarity uses WavLM-TDNN.
  - 300 LibriSpeech test utterances.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Tokenizer | LibriSpeech (hours not stated), 20 epochs | 2×A800; time not reported | batch 20/GPU, LR 4e-4 |
| USLM AR / NAR | MLS-En 44k h, 500k steps | 8×A800; time not reported | 7.5k / 5k tokens per GPU, LR 5e-4 |

## Preserving text intelligence / knowledge

n/a: there is no text LLM. The related claim is the paper's thesis: tokens that align strongly with text (high MI and PNMI) let an LM model content more accurately. USLM's content WER is 6.5 vs 7.9 for VALL-E (p.8).

## Key results

**SLMTokBench** (Tab.3, p.8). MI and probe WER measure text alignment; resynthesis WER and speaker similarity measure information preservation:

| Tokens | MI | Probe WER | Resynth WER | Speaker sim |
|---|---|---|---|---|
| HuBERT KM500 | 31.2 | 9.88 | 16.26 | 0.77 |
| EnCodec RVQ-1 | 16.5 | 61.52 | 38.34 | 0.92 |
| EnCodec RVQ-1:8 | 23.6 | 30.91 | 5.11 | 0.98 |
| SpeechTokenizer RVQ-1 (HuBERT-average teacher) | 30.9 | 15.58 | 9.57 | 0.74 |
| SpeechTokenizer RVQ-1:8 (HuBERT-average teacher) | 29.7 | 16.03 | 5.04 | 0.97 |
| SpeechTokenizer RVQ-1 (L9 teacher) | 32.9 | 12.68 | 14.17 | 0.73 |
| SpeechTokenizer RVQ-1:8 (L9 teacher) | 31.6 | 13.12 | 5.31 | 0.97 |
| SpeechTokenizer RVQ-1 (HuBERT-units teacher) | 24.2 | 34.13 | 20.02 | 0.72 |
| SpeechTokenizer RVQ-1:8 (HuBERT-units teacher) | 25.1 | 30.71 | 5.84 | 0.95 |

**Reconstruction** (Tab.2, p.7):

| | WER | ViSQOL | MUSHRA |
|---|---|---|---|
| Ground truth | 4.58 | – | 91.46 |
| EnCodec | 5.11 | 4.37 | 79.86 |
| SpeechTokenizer | 5.04 | 4.30 | 90.55 |

**Zero-shot TTS on VCTK** (Tab.4, p.8):

| | WER | Speaker sim | MOS | SMOS |
|---|---|---|---|---|
| Ground truth | 1.9 | 0.93 | 4.5 | 3.96 |
| VALL-E | 7.9 | 0.75 | 3.08 | 3.31 |
| USLM | 6.5 | 0.84 | 3.63 | 3.45 |

**PNMI** (Tab.8, p.18): HuBERT KM500 0.43; EnCodec RVQ-1 0.28; SpeechTokenizer RVQ-1 0.71. More than 600 of EnCodec's RVQ-1 codes are never used (p.18).

## Ablations & lessons

- **Continuous teacher beats discrete units.** HuBERT L9 features vs HuBERT units: MI 32.9 vs 24.2, and full resynthesis WER 5.31 vs 5.84 (Tab.3, p.8).
- **Semantic distillation conflicts with reconstruction.** The purer L9 teacher gives higher MI but worse resynthesis WER than the all-layer average (5.31 vs 5.04). The authors attribute this to a task conflict between distillation and reconstruction (p.8).
- **D-axis vs T-axis distillation loss:** RVQ-1 MI 30.9 vs 26.65; full resynthesis WER 5.04 vs 5.29 (Tab.7, p.15).
- **BiLSTM vs LSTM:** RVQ-1 MI 30.9 vs 27.60; probe WER 15.58 vs 20.71 (Tab.6, p.14). Our inference: a bidirectional encoder is non-causal, so this tokenizer is not streamable as-is. The paper does not discuss streaming.
- **Disentanglement is only partial.** In one-shot voice conversion, taking the reference's RVQ-2, RVQ-2:4 or RVQ-2:8 raises speaker similarity (0.72 / 0.80 / 0.82) but also WER (2.6 / 11.7 / 35.4). Content leaks into layers 2–8 (Tab.5, p.9).
- **A content-only first layer drops timbre:** speaker similarity 0.74 from SpeechTokenizer RVQ-1 vs 0.92 from EnCodec RVQ-1. Residual layers restore it to 0.97 (Tab.3).
- **Transfers to unseen languages** (German, Chinese), qualitatively only (p.19).

## Implications for our setup

- **Probe our Qwen3 tokenizer first.** Run SLMTokBench-style checks on the primary codebook and on all codebooks: CTC-probe MI and WER, plus PNMI.
  - If the codebook the LLM predicts first is not semantically supervised, the LLM has to learn phonetic content from acoustic codes. That is a plausible contributor to weak semantics in our generated speech.
  - SpeechTokenizer shows distillation roughly doubles MI (30.9 vs 16.5) and raises PNMI 2.5× (0.71 vs 0.28) over a plain acoustic codec.
- **Factorize generation.** Let the backbone predict the semantic layer, and a lighter module (depth transformer or NAR) predict the acoustic residuals. USLM shows this improves both content (WER 6.5 vs 7.9) and voice similarity (0.84 vs 0.75) compared with flat acoustic-token modeling.
- **If we train or fine-tune a tokenizer:** distill from continuous SSL features, not units; use the per-dimension cosine loss; keep the encoder causal for duplex streaming.
- **Semantic tokens alone lose timbre** (speaker similarity 0.74). Residual codebooks or a prompt-conditioned decoder are needed for voice consistency in continuation.

## Open questions / unclear points

- Codebook size, bitrate, the λ loss weights, and how many LibriSpeech hours were used.
- Which teacher the released model uses. The Tab.2 numbers match the HuBERT-average row of Tab.3.
- MI units are not stated.
- No causal or streaming variant is evaluated.
