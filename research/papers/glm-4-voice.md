# GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot

| Field | Value |
|---|---|
| Org | Zhipu.AI and Tsinghua University (p.1) |
| Date / arXiv | 3 Dec 2024; arXiv:2412.02612v1 (p.1) |
| Category | half-duplex S2S (spoken chatbot) |
| Backbone | GLM-4-9B-Base, vocabulary expanded with speech tokens, continually pretrained (p.6) |
| Audio input | GLM-4-Voice-Tokenizer: whisper-large-v3 fine-tuned with average pooling plus a VQ bottleneck in the middle of the encoder. 12.5 Hz, single codebook, 175 bps (⇒ 2^14 = 16,384 codes; derived: 175 bps ÷ 12.5 Hz = 14 bits). Causal convolution and block-causal attention allow streaming (p.3–4, Tab.1) |
| Audio output | The same 12.5 Hz single-codebook tokens, in the same vocabulary as text. Output follows the "streaming thoughts" pattern of 13 text tokens then 26 speech tokens (p.5) |
| Waveform decoder | CosyVoice-style: speech-token encoder → conditional flow matching → mel → HiFi-GAN. Streams in blocks of b = 0.8 s, so ≥10 tokens are needed before the first audio (p.4–5) |
| Duplex mechanism | n/a (half duplex; input and output stream, but turns are sequential) |
| Open weights / code | github.com/THUDM/GLM-4-Voice; HF THUDM/glm-4-voice-9b (p.1) |
| Relevance to us | **High.** The reference recipe for synthetic interleaved speech–text data built from text corpora to transfer knowledge, with 1T tokens at 30% text and spoken-QA and StoryCloze evals |

## TL;DR

- A 12.5 Hz single-codebook supervised semantic tokenizer, derived from ASR, is used for both input and output. Sharing one token space lets the model pretrain on unsupervised speech with plain next-token prediction (p.3).
- Continued pretraining from GLM-4-9B-Base covers 1T tokens (p.6, Tab.2):
  - 30% text-only data.
  - About one epoch of 700K h of real unsupervised speech, plus ASR/TTS pairs.
  - The rest is synthetic interleaved speech–text, generated from text pretraining data by a "text-to-token" model.
- Base-model results (p.7–8):
  - Spoken StoryCloze: S→S 62.4 / S→T 76.3.
  - Topic-StoryCloze: S→S 82.9 / S→T 93.6.
  - Spoken QA, S→T vs S→S: Web Questions 32.2 / 15.9; Llama Questions 64.7 / 50.7; TriviaQA 39.1 / 26.5.
- SFT uses a "streaming thoughts" template, in which text leads speech in chunks of 13 text : 26 speech tokens. The loss is decoupled: speech output is trained for 20 epochs and text output for 4 (p.5, 7–8).
- Main finding: S→T beats S→S on every task, so "textual guidance is still necessary for intelligent speech chatbots" (p.8).
- How the synthetic data is made (text-to-token model, span selection) is described in the companion paper Zeng et al. 2024 [45] (arXiv:2411.17607), not here.

## Architecture

**Tokenizer** (p.3–4)
- whisper-large-v3 plus average pooling plus VQ, placed in the middle of the encoder.
- The codebook is learned by EMA (decay 0.99) with dead-code reset; commitment coefficient 10.0.
- Variants exist at 50, 25, 12.5 and 6.25 Hz. Codebook size grows as the frame rate drops, to offset the information lost to pooling.

**Why 12.5 Hz was chosen** (Tab.1, p.4–5)

| Tokenizer | ASR WER, LS clean / other | ASR CER, AISHELL-1 | Recon WER | VisQOL | MOSNet |
|---|---|---|---|---|---|
| GLM-4-Voice 12.5 Hz (175 bps) | 2.10 / 4.90 | 3.02 | 8.43 | 2.52 | 3.39 |
| GLM-4-Voice 50 Hz (600 bps) | 1.85 / 3.78 | 2.70 | 6.24 | 2.67 | 3.38 |
| Mimi 12.5 Hz (1.1 kbps) | — | — | 8.36 | 2.82 | 2.89 |

- The authors call 12.5 Hz "an optimal balance between efficiency and quality."
- The 6.25 Hz row in the paper looks corrupted: identical numbers appear in both the ASR and reconstruction columns.

**Decoder** (p.4–5)
- CosyVoice decoder architecture: token encoder, conditional flow matching, HiFi-GAN.
- The token encoder and flow-matching model are trained from scratch: first pretrained on all the unsupervised multi-speaker speech, then fine-tuned on one high-quality speaker.
- For streaming, the decoder is fine-tuned on truncated audio (the first n·b seconds). At inference, the previous (n−1)·b seconds serve as the prompt. b = 0.8 s.

**LLM**
- GLM-4-9B with an expanded vocabulary and no extra heads ("minimal modifications") (p.3).
- The authors argue that a single codebook avoids multi-layer generation machinery and "helps preserve the model's text processing capabilities" (p.3).

**Inference** (p.5–6)
- S2S is split into speech→text (A_t) and speech+text→speech (A_s).
- "Streaming thoughts" alternates 13 text tokens with 26 speech tokens (ratio 1:2), so text always runs ahead of speech.
- The first audio needs 13 text + 10 speech = 23 decoded tokens (p.6).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training

**Tokenizer** (p.4)
- Fine-tuned from whisper-large-v3 for 2 epochs, batch 4096, LR 1e-5.
- Data:
  - ASR sets: LibriSpeech, GigaSpeech, MLS-Eng, Wenet, CommonVoice, AISHELL-1.
  - A 10K-h proprietary Chinese ASR set.
  - 700K h of speech pseudo-labeled by whisper-large-v3 (EN) and paraformer-large (ZH).
  - Supervised : pseudo-labeled samples = 1:3.

**Decoder:** two stages (multi-speaker pretraining, then single-speaker fine-tuning) plus streaming truncation training (p.4–5).

**Compute:** not reported.

### Stage 1: Joint speech–text pretraining (1T tokens)

- **Initialization:** GLM-4-9B-Base with an expanded vocabulary (p.6). That all parameters are trained is implied but not stated.
- **Data** (Tab.2, p.6). "Consumed" = dataset size × epochs (derived).

| Data | Speech tokens | Text tokens | Epochs | Consumed (derived) |
|---|---|---|---|---|
| Interleaved speech–text | 455B | 279B | 0.90 | ≈661B |
| Speech-only (700K h) | 31B | – | 2.10 | ≈65B |
| ASR + TTS | 11B | 3.5B | 2.07 | ≈30B |
| Text-only | – | 10T | 0.03 | ≈300B |

  - Total consumed ≈1.06T, matching the stated 1T.
  - Text-only share ≈28% (derived), vs the stated "fixed sampling ratio of 30% text data."
  - Overall ≈497B speech and ≈558B text tokens (derived).
  - 31B speech-only tokens ≈ 689K h (derived: 31B ÷ 12.5 Hz ÷ 3600), consistent with the 700K h figure.
  - 455B synthetic interleaved speech tokens ≈ 10M h-equivalent (derived: 455B ÷ 12.5 ÷ 3600). These tokens were synthesized and never vocoded.
- **Inconsistency:** the text says "one epoch each of unsupervised speech and supervised speech-text data" (p.6), but Tab.2 lists 2.10 and 2.07 epochs.
- **Optimizer:** AdamW (β1 = 0.9, β2 = 0.95); sequence length 8192; LR decays linearly from 6e-5 to 6e-6 (p.6).
- **Not reported:** batch, steps, warmup, compute.
- Data and procedure are the same as in Zeng et al. [45] (p.6).

### Stage 2: Supervised fine-tuning

**Data** (p.7; hours and sample counts not reported)
1. Multi-turn spoken dialogues derived from text dialogue data.
   - Filtered; code and math removed.
   - Responses shortened and made easy to speak.
   - Speech outputs synthesized; the TTS engine is not named.
   - User inputs are diversified with recordings read by annotators.
2. Speech-style-controlled dialogues covering speed, emotion and dialect.

**Format** (p.7): each turn holds user speech Q_s, user text Q_t, text answer A_t and speech answer A_s, arranged in the streaming-thoughts template.

**Decoupled loss** (p.7–8)
- Each sample is split into two copies:
  - Speech→text, with the speech-output loss masked.
  - Speech+text→speech, with the text-output loss masked.
- Reason: given speech input, text output is learned faster than speech output.

**Hyperparameters** (p.8)
- 20 epochs on speech output and 4 epochs on text output.
- LR 1e-5 → 1e-6; weight decay 0.1; dropout 0.5 on hidden layers; gradient clipping at 1.0.

### RL
- None.

## Data pipeline

- **Interleaved data:** "synthesized from text pre-training data" by a text-to-token model (Fig.2) that maps text to speech tokens directly, without making waveforms (p.1–2, 5–6). Span selection, granularity, interleave ratio and how the text-to-token model is trained are all deferred to [45].
- **Unsupervised speech:** 700K h of real speech (p.6); sources not given here.
- **ASR/TTS pairs:** used in pretraining; datasets not listed here.
- **SFT:** text-derived dialogues with synthesized speech, annotator-recorded user speech, and style-controlled dialogues (p.7).
- **Evaluation contexts and continuations:** synthesized with VolcEngine multi-speaker TTS (p.7).
- **Duplex:** no duplex, overlap or backchannel data.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Tokenizer | ASR sets + 10K h ZH + 700K h pseudo-labeled; 2 epochs | not reported | batch 4096, LR 1e-5 (p.4) |
| Decoder | all unsupervised speech, then one speaker | not reported | streaming fine-tune |
| Pretraining | 1T tokens (≈497B speech + ≈558B text, derived) | not reported | seq 8192; LR 6e-5 → 6e-6 |
| SFT | not reported | not reported | 20 epochs speech / 4 epochs text |

## Preserving text intelligence / knowledge

- **Text replay:** 30% text-only data (≈300B tokens, derived), plus 279B text tokens inside the interleaved data (p.6).
- **Synthetic interleaving from text corpora** is the main transfer device, intended "to efficiently transfer knowledge from text to speech modalities" (p.1).
- **Single low-rate codebook,** argued to protect text ability (p.3).
- **Text-first generation:** speech is conditioned on text that has already been generated (p.5).
- **Measured retention:** no text benchmark before vs after, so degradation is not quantified.

## Key results

**Base model, speech language modeling** (Tab.3, p.7)

| Model | Topic-StoryCloze S→S / S→T | StoryCloze S→S / S→T |
|---|---|---|
| GLM-4-Voice (9B) | 82.9 / 93.6 | 62.4 / 76.3 |
| Moshi | 83.0 / – | 60.8 / – |
| Spirit-LM | 82.9 / 88.6 | 61.0 / 64.6 |
| TWIST | 66.6 / – | 53.3 / – |

**Base model, spoken QA** (Tab.4, p.8)

| Model | Web Questions S→T / S→S | Llama Questions S→T / S→S | TriviaQA S→T / S→S |
|---|---|---|---|
| GLM-4-Voice | 32.2 / 15.9 | 64.7 / 50.7 | 39.1 / 26.5 |
| Moshi | 26.6 / 9.2 | 62.3 / 21.0 | 22.8 / 7.3 |

**Base model, ASR/TTS** (Tab.5, p.8)
- ASR: LibriSpeech clean / other WER 2.82 / 7.66; AISHELL-1 CER 2.46.
- TTS WER: LibriTTS 5.64; Seed-TTS en 2.91, zh 2.10.

**Chat model** (Tab.6, p.9)
- GPT-4o-judged scores: General QA 5.40 (MT-Bench-style rating, 1–10); Knowledge 5.20 (accuracy on 100 Web Q / Llama Q / TriviaQA items, scaled to 0–10). Output was restricted to English tokens for this eval (p.8–9).
- UTMOS 4.45; ASR-WER between speech and text output 5.74.
- Comparisons: Llama-Omni 3.50 / 3.90 / 3.92 / 9.18; Moshi 2.42 / 3.60 / 3.90 / 7.95.

**Cross-paper numbers for the same model**
- MiMo-Audio report (p.21, Tab.8): Big Bench Audio S2T 44.80 / S2S 42.70.
- Covo-Audio report (p.15, Tab.6): VCB-Bench general knowledge 41.79.
- MiMo-Audio report (p.9, Tab.1): the GLM tokenizer reconstructs with speaker similarity 0.12 (EN) / 0.33 (ZH).

## Ablations & lessons

- **Speech answers recover only part of the text-answer knowledge.**
  - In spoken QA, S→S accuracy is 49% (Web Questions), 68% (TriviaQA) and 78% (Llama Questions) of S→T accuracy (derived from Tab.4, p.8).
  - The authors conclude that text guidance is required and adopt text-first decoding (p.8).
  - They also note that the gap is much smaller than Moshi's, especially on Llama Questions (p.8).
- **Frame-rate trade-off at the tokenizer level** (Tab.1, p.4).
  - Going from 50 Hz to 12.5 Hz barely changes ASR (LS-clean WER 2.10 vs 1.85).
  - Reconstruction WER worsens from 6.24 to 8.43, and VisQOL from 2.67 to 2.52. MOSNet is unchanged (3.38 vs 3.39).
  - LM sequences become 4× shorter.
  - There is no LM-level frame-rate ablation.
- **In SFT, speech output learns more slowly than text output.** The fix is separate loss-masked copies of each sample, with 20 vs 4 epochs and heavy regularization (dropout 0.5, weight decay 0.1) (p.7–8).
- **The streaming ratio was chosen empirically.** A 13:26 text:speech ratio keeps text ahead; 26 speech tokens per block lets the model produce a coherent piece of content before synthesizing it (p.5). No numbers are given.
- **Speech pretraining matters more than SFT alone** (confounded comparison).
  - On the knowledge score, Mini-Omni, which has no speech pretraining, gets 1.10 vs 5.20 for GLM-4-Voice (Tab.6, p.9).
  - The authors attribute this to the missing speech pretraining (p.3). Model sizes and data differ, so this is not a controlled ablation.
- **Text-derived synthetic interleaving matches real-speech scale on continuation** (cross-model comparison).
  - S→S Topic-StoryCloze 82.9 is on par with Moshi's 83.0; S→S StoryCloze 62.4 is above Moshi's 60.8 (p.7).
  - GLM used only 700K h of real speech, while the paper notes Moshi scaled natural speech to 7M h (p.1–2).

## Implications for our setup

- **Synthetic interleaved data is the most direct lever for knowledge in generated speech.**
  - GLM-4-Voice used ≈455B synthetic speech tokens (≈10M h-equivalent, derived) against only 700K h of real speech (p.6).
  - For a Qwen3 stack, we could train a text→Qwen3-audio-token model (or use a Qwen3 TTS front-end and keep its tokens without vocoding), then convert spans of text pretraining data into audio tokens.
  - This targets weak semantic knowledge in generated speech head-on. Span-sampling details have to come from [45].
- **Text replay of at least ~30%.** GLM used about 300B text-only tokens (p.6). Pure duplex-speech mid-training removes this anchor.
- **Text-first output.** Generating 13 text tokens before each 26 speech tokens (p.5) is a strong semantic scaffold. For duplex, the equivalent is a text "inner monologue" stream that leads the model's audio channel.
- **SFT mechanics if the speech side lags:** decoupled loss-masked copies and more epochs on speech output (p.7–8).
- **Benchmarks to reproduce on our checkpoints:**
  - Spoken StoryCloze and Topic-StoryCloze, S→S and S→T, as continuation-quality probes.
  - Spoken Web Questions, Llama Questions and TriviaQA, S→T and S→S, to measure the knowledge gap.
  - GLM's base numbers are a 9B / 1T-token reference.
- **Scale gap.** GLM consumed ≈497B speech tokens; 10K h at 12.5 Hz is ≈0.45B (derived: 10K × 3600 × 12.5), about 1,100× less.
- **Tokenizer caveat.** GLM's semantic tokens reconstruct speaker identity poorly (EN SIM 0.12; MiMo report p.9). That hurts voice and prosody continuity in continuation, which matters for duplex. Where the Qwen3 tokenizer sits on this semantic-vs-acoustic trade-off should guide the choice between GLM-style and MiMo-style recipes.

## Open questions / unclear points

- How the text-to-token model is built and trained; span selection; interleave granularity; language split. All are in [45], not this paper.
- Sources of the 700K h of unsupervised speech and of the ASR/TTS pairs used in pretraining.
- Batch size, steps, warmup, hardware and GPU-hours.
- SFT data volume, the TTS engine used for assistant speech, and the number of style categories.
- Text-capability retention: no text benchmarks.
- Internal inconsistencies: "one epoch each" (p.6) vs 2.10 / 2.07 epochs in Tab.2; the 6.25 Hz row of Tab.1 looks duplicated.
