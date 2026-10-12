# MiMo-Audio: Audio Language Models are Few-Shot Learners

| Field | Value |
|---|---|
| Org | Xiaomi, LLM-Core team (p.1) |
| Date / arXiv | No date or arXiv ID printed in the report. It cites Aug-2025 work (MMAU-Pro, arXiv 2508.13992), so it is from Aug 2025 or later. Code: github.com/XiaomiMiMo/MiMo-Audio (p.1) |
| Category | half-duplex S2S / audio foundation model (pretrained speech–text LM plus an instruct model) |
| Backbone | MiMo-7B-Base (Xiaomi's own text LLM): 36 layers, d=4096, FFN 11008, 32 heads, context 8192 (p.11–12, Tab.2). Fully trained in PT Stage 2 and SFT |
| Audio input | MiMo-Audio-Tokenizer (1.2B params, trained from scratch): 24 kHz → 100 Hz mel → 25 Hz encoder → 20-layer RVQ. The LM uses the first 8 codebooks, i.e. 200 tokens/s at 1.55 kbps (p.4, 6–7, 9, Tab.1). A patch encoder groups 4 frames into one LLM position, giving 6.25 Hz (p.5, 10–11) |
| Audio output | The same 8 RVQ codebooks at 25 Hz (vocab 1024, 1024, then 128 each). For each 6.25 Hz LLM step, a 16-layer patch decoder (local AR transformer) generates 4 frames × 8 codebooks with a MusicGen-style delay pattern 0-1-…-7 (p.11–12, Tab.2–3) |
| Waveform decoder | The tokenizer's own audio decoder (mirror of the encoder, causal attention for streaming) plus a Transformer-based Vocos-style vocoder (ISTFT), fine-tuned with GAN losses (p.6–9) |
| Duplex mechanism | n/a (half duplex; no streaming or duplex dialogue is described) |
| Open weights / code | Model checkpoints and the full evaluation suite are released (p.1). SpeechMMLU is released on HF (p.14) |
| Relevance to us | **High.** It is the only paper in this group that shows a data-scale threshold (~0.7T tokens) for emergent speech-side knowledge and continuation. It also has the smallest text→speech "modality gap" (3.4 pts on SpeechMMLU) |

## TL;DR

- MiMo-7B-Base is continually pretrained on >100M hours of in-the-wild audio, using a high-fidelity 8-codebook RVQ tokenizer instead of semantic-only tokens. Frames are patched to 6.25 Hz so the LLM sees audio at a text-like rate (p.4–5, 10).
- Pretraining has two stages (p.13–14, Tab.3):
  - **Stage 1 (understanding):** 2.6T tokens (1.2T text + 1.4T speech-related), loss on text only.
  - **Stage 2 (understanding + generation):** 5T tokens (2.6T text + 2.4T audio), loss on text (weight 100) and on RVQ tokens (weights 12 down to 1).
  - Text is 46–52% of tokens (derived: 1.2/2.6 and 2.6/5).
- Few-shot abilities stay near zero until about 0.7T training tokens, then jump within the next ~0.3T tokens and plateau (p.16, Fig.1). This holds for 5-shot SpeechMMLU T2S/S2S, 16-shot voice conversion and 16-shot S2ST.
- Base model on 5-shot SpeechMMLU: S2S 69.1 vs T2T 72.5, a modality gap of 3.4 pts. Other bases: Step-Audio2-mini 22.3, Kimi-Audio 58.9, Baichuan-Audio 39.2 (p.17, Tab.6). Speech continuation is shown with demos only; no quantitative continuation metric is given (p.18).
- Post-training is a 100B-token SFT. Dialogue speech is synthesized by an in-house MiMo-TTS-7B trained on >7M h (p.5, 19). Big Bench Audio: S2T 72.9, S2S 60.2 (p.21, Tab.8). There is no RL; it is listed as future work (p.24).
- There is no duplex model. There are no ablations of mixture, tokenizer or patch size, and compute is not reported.

## Architecture

**Tokenizer** (1.2B params; p.4, 6–7)
- Encoder: a 32-layer bidirectional Transformer (20 heads, d=1280, FFN 5120, RoPE, GELU) between 2× downsampling layers.
- Layer-3 hidden states are added to the final-layer output "to mitigate the conflict between semantic and acoustic information" (p.7).
- 20-layer RVQ: the first two codebooks have 1024 entries, the rest 128.
- Decoder: mirrors the encoder but uses causal self-attention (for streaming).
- Vocoder: Vocos with a Transformer backbone instead of ConvNeXt (16 layers, 16 heads, d=256, FFN 1024). Sliding-window attention with windows [40, 10] gives receptive fields of [6.4 s, 1.6 s] (p.7).

**Patch encoder** (p.10–11)
- Sums the 8 codebook embeddings of each frame.
- Runs a 6-layer bidirectional Transformer (d=1024, FFN 4096) over the G=4 frames of a patch.
- Concatenates the 4 frame outputs and projects them linearly to the LLM width.
- Head count: the text says 64 heads; Tab.2 says 16.

**LLM** (p.11)
- MiMo-7B-Base. Each input position is either a text-token embedding or an audio patch.
- Each output hidden state goes either to the text head or to the patch decoder.

**Patch decoder** (p.11–12, Tab.2–3)
- 16-layer causal Transformer (d=1024, FFN 4096, 64 heads).
- Shares the 8 embedding tables with the patch encoder and has 8 output heads.
- Generates the next patch autoregressively with delays D=[0..7], so one patch takes 4+7=11 decoder steps. This matches the "context length 11" in Tab.2.
- Without the delay, predicting all codebooks at once "often leads to poor audio generation quality" (p.11).

**Sequence design**
- Text tokens and audio patches are interleaved and trained with next-token / next-patch prediction (p.10, Eq.10).
- The stated reason for patching: audio frames carry much less information than text tokens, so grouping them reduces the mismatch "and facilitate[s] cross-modal knowledge transfer" (p.10).
- A context of 8192 positions holds about 21.8 min of pure audio at 6.25 Hz (derived: 8192 ÷ 6.25 Hz).

**Design principle:** "lossless" information flow. The authors explicitly reject the semantic-token designs of GLM-4-Voice, Kimi-Audio and Step-Audio 2 because they discard paralinguistic information (p.4).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training

**0a. Unified representation learning** (p.7–8)
- All tokenizer parameters are trained from scratch together with a jointly trained LLM, also from scratch. That LLM's size is not reported.
- Data: ">11 million hours" (p.7). The intro says a "10-million-hour corpus" (p.4).
- Loss (Eq.1–3) = 10 × A2T (next-token prediction of transcription/caption text from quantized audio) + 1 × multi-scale mel L1 (2^5 to 2^7 mel bins) + 1 × commitment.

**0b. Adversarial fine-tuning** (p.8–9)
- Encoder and RVQ are frozen "to preserve the semantic structure of the audio token space."
- Decoder and vocoder are trained with MPD and MS-STFT discriminators (hinge GAN plus feature matching). Weights: recon 1.0, adv 1.0, fm 2.0 (Eq.4–7).

**Reconstruction quality** with the 8 codebooks used downstream (Seed-TTS-Eval; Tab.1, p.9):

| Split | PESQ-NB / WB | SIM | STOI |
|---|---|---|---|
| ZH | 3.30 / 2.71 | 0.89 | 0.93 |
| EN | 3.02 / 2.43 | 0.85 | 0.92 |

- These are the best scores in the table at a comparable bitrate. Baselines include Mimi (1.1 kbps, EN SIM 0.74) and the GLM-4-Voice tokenizer (0.175 kbps, EN SIM 0.12).
- Optimizer, LR, batch, steps and compute for the tokenizer: not reported.

### Stage 1: Understanding training (speech–text pretraining, loss on text only)
- **Trainable:** patch encoder + LLM. The patch decoder is not used yet (p.13).
- **Data:** 2.6T tokens = 1.2T text + 1.4T "speech-related" tokens, counted at 6.25 Hz (p.13).
  - Formats: speech–text interleaved, ASR, general audio captioning, text-only pretraining.
  - Text share: 46% (derived: 1.2/2.6).
  - 1.4T tokens ≈ 62M h (derived: 1.4T ÷ 6.25 Hz ÷ 3600). This is an upper bound, because "speech-related" may include the text side of ASR and caption pairs.
- **Loss:** text tokens only; loss weights 1-0-…-0 (p.14, Tab.3).
- **Hyperparameters** (p.14, Tab.3):
  - LR: patch encoder 2e-4, LLM 3e-5, constant schedule, 1% warmup.
  - Batch 16.8M tokens; context 8192.
  - About 155K steps (derived: 2.6T ÷ 16.8M, assuming a single pass).
- **Compute:** not reported.

### Stage 2: Understanding–generation joint training
- **Trainable:** everything (patch encoder, LLM, patch decoder) (p.14).
- **Data:** 5T tokens = 2.6T text + 2.4T audio, at 6.25 Hz (p.14).
  - Seven formats: speech continuation, speech–text interleaved, ASR, TTS, general audio captioning, instruction-following TTS, text pretraining.
  - Text share: 52% (derived).
  - 2.4T audio tokens ≈ 107M h (derived: 2.4T ÷ 6.25 ÷ 3600). This is consistent with the ">100M hours" claim.
- **Loss:** text weight 100; RVQ codebook weights 12, 8, 6, 4, 2, 2, 1, 1; delays 0–7 (p.14, Tab.3).
- **Hyperparameters:** LR 2e-4 for patch encoder/decoder and 3e-5 for the LLM, cosine schedule, 1% warmup. Batch 16.8M tokens; context 8192. About 298K steps (derived: 5T ÷ 16.8M).
- **Compute:** not reported.

### Stage 3: Post-training (SFT) → MiMo-Audio-7B-Instruct
- All parameters are fine-tuned on 100B tokens (p.19).
- Six task formats: ASR, TTS, audio understanding, spoken dialogue, instruction-following TTS, text dialogue.
  - ASR, TTS and text dialogue come from open-source data.
  - The other three use the curated sets described in Sec.4.1.
- Loss weights are the same as Stage 2 (p.19).
- LR: 5e-5 for patch encoder/decoder, 1e-5 for the LLM, cosine, 1% warmup. Context 8192; batch 2.1M tokens (p.19, Tab.3). About 48K steps (derived: 100B ÷ 2.1M).
- "Thinking" (CoT) data is included for both understanding and generation (p.5). Its volume and format are not reported.
- Per-task mixture ratios: not reported.
- **RL:** none. RL is planned to fix dialogue instability and thinking hallucinations (p.24).

## Data pipeline

**Pretraining audio**
- Volume: "hundreds of millions of hours" of in-the-wild audio collected (p.12); "over 100 million hours" used (p.1, 5).
- Sources: podcasts, audiobooks, news, interviews, conference recordings.
- Topics: daily communication, entertainment, business, arts, science (p.12).

**Processing:** normalization, speaker diarization, VAD, ASR and audio-quality assessment (p.13).

**Labeling** (p.13)
- Semantic: a text-quality model scores ASR transcripts on conversational quality, knowledge density and logical reasoning.
- Non-semantic: an in-house audio-captioning model describes timbre, emotion/style and background environment.

**Curation:** threshold filtering for noise, low quality and unsafe content, plus quality-weighted sampling that combines both score types (p.13).

**Interleaved speech–text data:** listed as a task format in both stages (p.13–14). How it is built is not reported: real aligned speech vs TTS, granularity and volume are all unknown.

**Post-training data** (p.19)
- Audio understanding: open-source sets, cleaned and augmented by an LLM pipeline.
- Speech generation: caption-based instructions over a high-quality subset of the pretraining audio.
- Spoken dialogue: screened text dialogues are colloquially "stylistically rewritten," then voiced with in-house MiMo-TTS. MiMo-TTS-7B was trained on >7M h (p.5). Voice prompts are sampled from a large timbre library.

**Duplex:** no duplex, overlap or backchannel data.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Tokenizer (0a + 0b) | >11M h (p.7); "10M h" (p.4) | not reported | 1.2B params; A2T + reconstruction, then GAN |
| PT Stage 1 (understanding) | 2.6T tok (1.2T text) | not reported | batch 16.8M, ctx 8192; ≈155K steps (derived: 2.6T ÷ 16.8M) |
| PT Stage 2 (understanding + generation) | 5T tok (2.6T text; 2.4T audio ≈ 107M h, derived) | not reported | ≈298K steps (derived: 5T ÷ 16.8M) |
| SFT | 100B tok | not reported | batch 2.1M; ≈48K steps (derived) |
| MiMo-TTS-7B (data synthesis) | >7M h (p.5) | not reported | voices the SFT dialogues |

## Preserving text intelligence / knowledge

**Mechanisms**
1. Text-only data is kept at about 46% of tokens in Stage 1 and 52% in Stage 2 (derived from p.13–14).
2. Text loss has weight 100, versus at most 12 for any RVQ codebook (p.14, Tab.3).
3. The LLM's LR (3e-5) is much lower than the new patch modules' LR (2e-4) (p.14).
4. Stage 1 trains understanding with text loss only, before any audio-generation loss (p.13).
5. Patching to 6.25 Hz is meant to "facilitate cross-modal knowledge transfer" (p.10).

**Measured text retention:** only SpeechMMLU T2T = 72.5 for MiMo-Audio-7B-Base (p.17, Tab.6). The paper claims "only a minor degradation in text performance" (p.5), but it does not report MiMo-7B-Base's own score on this split. The size of the drop is therefore not quantified.

**Knowledge transfer into speech** (5-shot SpeechMMLU, Tab.6, p.17):
- MiMo: S2S 69.1, S2T 69.5, T2S 71.5, T2T 72.5.
- Modality gap (T2T − S2S): MiMo 3.4; Step-Audio2-mini-Base 22.3; Kimi-Audio-Base 58.9; Baichuan-Audio-Base 39.2.

## Key results

**Base model** (5-shot / 16-shot ICL; p.17, Tab.6). SpeechMMLU has 8,549 items over 34 subjects, voiced by a commercial TTS (p.15).

| SpeechMMLU | S2S | S2T | T2S | T2T |
|---|---|---|---|---|
| MiMo-Audio-7B-Base | 69.1 | 69.5 | 71.5 | 72.5 |
| Step-Audio2-mini-Base | 51.8 | 67.8 | 63.4 | 74.1 |
| Kimi-Audio-Base | 11.8 | 67.9 | 0.0 | 70.7 |
| Baichuan-Audio-Base | 31.9 | 29.9 | 16.7 | 71.1 |

- MMAU (5-shot): overall 66.0 (speech 67.6, sound 65.2, music 65.3), vs 60.3 for Step-Audio2-mini-Base.
- Speech continuation: demos only (p.18). No sStoryCloze, tStoryCloze, sWUGGY or sBLIMP numbers.

**Instruct model** (p.21–22, Tab.8–9)
- Audio understanding: MMAU 74.90; MMAU-Pro 53.35; MMAR 63.60; MMSU 61.70 (62.88 with thinking).
- Big Bench Audio: S2T 72.90 / S2S 60.20.
  - gpt-4o-audio-preview: 70.20 / 67.20.
  - Kimi-Audio-Instruct: 59.40 / 51.00.
  - GLM-4-Voice: 44.80 / 42.70.
- MultiChallenge Audio: S2T 15.15 / S2S 10.10.
- TTS:
  - Seed-TTS-Eval WER: ZH 1.96, EN 5.37, ZH-hard 14.14.
  - InstructTTSEval overall: EN 72.59, ZH 70.52 (GPT-4o-mini-tts: 68.50 / 51.07).
- ASR WER: LibriSpeech test-clean 3.76, AISHELL-1 1.78. These are worse than Step-Audio2-mini (1.87 / 0.95) and Kimi-Audio (2.13 / 0.62).

## Ablations & lessons

The paper has no controlled ablations: no mixture, tokenizer, patch-size, loss-weight or data-scale ablations beyond one training-curve figure. The lessons below come from that figure and from cross-model comparisons.

- **Speech-side knowledge and few-shot S2S skills appear abruptly at a data threshold.**
  - Evidence: 5-shot SpeechMMLU T2S/S2S, 16-shot voice conversion and 16-shot S2ST are near zero until about 0.7T training tokens, then jump and plateau (p.16, Fig.1).
  - Approximate values read off Fig.1 (p.1):
    - S2S: ≈6% at ~0.67T → ≈55% at ~0.83T → ≈66% at ~1.0T.
    - T2S: ≈0% through ~0.83T → ≈60% at ~1.0T → ≈71% by ~1.5T.
    - S2ST COMET: ≈43 → ≈73 between ~0.67T and ~1.0T.
    - Voice-conversion similarity: ≈11 → ≈34 over the same span.
  - Knowledge in generated speech (T2S/S2S) and S2S generation skills "emerge together at similar training scales" (p.18).
  - The axis is labeled "Train Tokens." The paper does not say which stage it covers or whether text tokens are included.
  - 0.7T tokens ≈ 15–31M h of audio (derived: 0.7T ÷ 6.25 Hz ÷ 3600 = 31M h if every token is audio; ≈15M h if Stage 2's 48% audio share applies).
- **Understanding speech is not the same as answering in speech; the generation side needs its own scale.**
  - Under the same protocol, Kimi-Audio-Base (semantic tokens, 13M h) scores S2T 67.9 but S2S 11.8 and T2S 0.0. MiMo's S2S (69.1) is close to its S2T (69.5) (p.17, Tab.6).
  - Caveat: this is a few-shot base-model protocol, and output-format effects may penalize other models.
- **High-rate multi-codebook audio plus 4-frame patching did not block knowledge transfer.** Despite 200 tokens/s, MiMo has the smallest modality gap (3.4) (p.17). The authors credit the architecture (p.17), but no ablation separates architecture from data scale.
- **Loss weighting strongly favors text and coarse codebooks:** text 100; RVQ 12/8/6/4/2/2/1/1 (p.14, Tab.3). There is no ablation, but this is a concrete recipe for keeping the text pathway dominant while learning 8 codebooks.
- **Curriculum: understanding first (text-only loss), then joint generation** (p.13–14). This is what the authors use to "maximally preserve" text ability (p.13). No ablation.
- **Separate learning rates:** new modules 2e-4 vs LLM 3e-5 in pretraining (6.7× lower for the LLM); 5e-5 vs 1e-5 in SFT (p.14, Tab.3).
- **Naturalistic multi-speaker pretraining yields multi-speaker continuation.** The model continues debates (two consistent speakers), talk shows (with audience cheers), livestreams, recitation, singing and dialects (p.18). The evidence is qualitative only.
- **The post-SFT dialogue model is still unstable.**
  - Reported problems: timbre discontinuities, unstable audio quality, mispronunciations (especially symbols and formulas), inconsistent system-prompt compliance and unstable style control (p.24).
  - Thinking helps speech understanding (MMSU 61.70 → 62.88) but hurts sound and music because of hallucinations during thinking (p.21, 24).
- **A modality gap reopens after SFT.** Big Bench Audio S2S is 12.7 pts below S2T (derived: 72.9 − 60.2, p.21), even though the base model's SpeechMMLU gap was only 3.4.

## Implications for our setup

- **Data scale is the first suspect for both problems.**
  - MiMo's speech-side knowledge (T2S/S2S) and in-context S2S abilities were essentially zero until about 0.7T tokens, roughly 15–31M h (derived), of a mix that was about 50% text (p.13–16).
  - Our 10K h is about 0.2–0.45B tokens (derived: 10K h × 3600 × 6.25–12.5 Hz), roughly 1,500–3,000× below that knee (derived: 15–31M h ÷ 10K h).
  - Recommendation: treat duplex conversational data as a fine-tuning-scale ingredient. Add a large general speech–text continued-pretraining stage (real long-form speech plus interleaved data) before duplex training.
- **Keep about 50% text tokens and weight text loss heavily.** MiMo used 46–52% text, with text-loss weight 100 vs ≤12 per codebook (p.13–14). A duplex-only mid-training run without text replay is the opposite regime.
- **Stage the curriculum and split learning rates.** Run an understanding-only stage (text loss) before adding audio-generation loss. Use a lower LLM LR (3e-5) than for new audio modules (2e-4) (Tab.3).
- **Bring the backbone's audio rate down to text-like rates.** If the Qwen3 tokenizer emits ≥12.5 Hz frames with several codebooks, consider grouping frames into 6.25 Hz patches. Add a small local decoder (MiMo uses 16 layers) with a codebook delay pattern for intra-patch generation (p.10–12). MiMo has no patch-size ablation, so treat this as a hypothesis to test.
- **Track the same diagnostic.**
  - Build a T2T / S2T / T2S / S2S SpeechMMLU split (the dataset is public, p.14) and plot it against training tokens.
  - This separates "weak knowledge in generated speech" (T2S, S2S) from "weak understanding" (S2T), and shows whether a run is before or after the emergence knee.
- **Continuation needs naturalistic, long-form, multi-speaker audio.** MiMo used podcasts, interviews, debates and talk shows (p.12, 18), with an 8192-position context (≈21.8 min of audio, derived). Duplex corpora are in-domain for this but are tiny by comparison.
- **Don't expect SFT alone to close the gap.** MiMo-Instruct still has a ~13-pt S2T–S2S gap on Big Bench Audio and dialogue-quality problems that the authors plan to address with RL (p.21, 24).

## Open questions / unclear points

- What Fig.1's "Train Tokens" axis counts: Stage 2 only or cumulative, all tokens or audio only. Without this, the ~0.7T threshold cannot be converted to hours reliably.
- How interleaved speech–text data is built (real vs TTS, word/sentence/span granularity) and how much of it there is: not reported.
- Per-format mixture ratios in Stage 1 (4 formats) and Stage 2 (7 formats), including the speech-continuation share.
- The size of the LLM jointly trained with the tokenizer, and the tokenizer's LR, steps and compute.
- MiMo-7B-Base's own score on SpeechMMLU-T2T or MMLU, needed to quantify text degradation.
- Internal inconsistencies:
  - Tokenizer data: 10M h (p.4) vs >11M h (p.7).
  - Patch-encoder heads: 64 (p.11) vs 16 (Tab.2).
  - Tab.2 lists 7 vocab sizes for 8 channels.
- Whether Instruct S2S answers are text-guided (text before speech, or thinking) or speech-only: not described.
- No quantitative continuation metrics and no hardware or compute figures.
