# Qwen3-TTS Technical Report (Qwen3-TTS; Qwen-TTS-Tokenizer-25Hz / -12Hz)

| Field | Value |
|---|---|
| Org | Qwen Team, Alibaba (p.1) |
| Date / arXiv | Title page dated 2026-01-23; arXiv:2601.15621v1, 22 Jan 2026 (p.1) |
| Category | TTS + codec |
| Backbone | "Qwen3 LM family" (p.4), in 0.6B and 1.7B sizes (Tab.1, p.2). The 1.7B talker config (28 layers, hidden 2048, 16 heads / 8 KV heads, FFN 6144, text vocab 151,936) matches Qwen3-1.7B on all of these dimensions (derived from https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/config.json and https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json). The initialization checkpoint is not named. A learnable speaker encoder is trained jointly with the backbone (p.4; config enc_dim 2048) |
| Audio input | Text via the Qwen tokenizer, plus an optional instruction. Voice is taken from either (i) a speaker embedding of reference audio or (ii) an in-context text–speech pair (p.4–5) |
| Audio output | **25 Hz variant:** 1 codebook of 32,768 entries at 25 Hz; the backbone predicts one token per step through a linear head (p.3–4, Tab.3). **12 Hz variant:** 12.5 Hz frames, 16 codebooks of 2048 entries each. Codebook 0 is semantic; the other 15 form an acoustic RVQ stack. The backbone predicts codebook 0 from the summed codebook embeddings; an MTP module predicts the 15 residual codebooks (p.3–4, Fig.3, Tab.4). MTP config: 5 layers, hidden 1024 |
| Waveform decoder | **25 Hz:** block-attention flow-matching DiT (code → mel, 1-block lookahead, 8-token blocks), then a modified BigVGAN (mel → wave, 130 ms right context) (p.3, p.6). **12 Hz:** fully causal, left-context-only decoder: window transformer + upsampling ConvNet (Fig.2b). It needs no speaker vector and no diffusion (p.2–4). Output is 24 kHz, 1920 samples per frame (model card config) |
| Duplex mechanism | n/a (TTS). "Dual-track" means a text track and a codec track per position, for streaming text in / audio out (p.4) |
| Open weights / code | Apache 2.0 (p.1). **Public:** Qwen3-TTS-Tokenizer-12Hz (0.2B; model card: https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz) and the 12 Hz models 1.7B-Base/-CustomVoice/-VoiceDesign and 0.6B-Base/-CustomVoice (GitHub: https://github.com/QwenLM/Qwen3-TTS, released 2026-01-22). **25 Hz** tokenizer and models: the paper says they are released (p.1), but the HF repo returned 401 on 2026-10-12 and the README says other models will come later |
| Relevance to us | **High.** The primary source on the Qwen3 speech tokenizers, and the only direct comparison of 12.5 Hz vs 25 Hz tokens under autoregressive modeling |

## TL;DR

- **Two tokenizers** (p.3–4):
  - **25 Hz:** a single codebook of 32,768, made by inserting VQ into Qwen2-Audio's ASR-trained encoder, then fine-tuning with mel reconstruction.
  - **12.5 Hz:** a fully causal, Mimi-style GAN codec with 16 layers — 1 WavLM-guided semantic codebook plus 15 RVQ acoustic layers.
- **12 Hz tokenizer beats same-shape baselines.** At the same 16 × 2048 @ 12.5 Hz budget it outperforms Mimi and FireRedTTS-2 on every reconstruction metric (Tab.4).
- **The TTS model** (p.4–5):
  - Qwen3 backbone with dual-track input and an MTP, trained on >5M h in 10 languages.
  - Pretraining: general → high-quality continued pretraining (CPT) → 32K-token context.
  - Post-training: DPO → GSPO → speaker fine-tuning.
- **Frame rate has a trade-off.** 12.5 Hz gives better short-form WER; 25 Hz is more stable on speech over 10 minutes (Tab.5, Tab.10).
- **Latency.** First packet is 97–101 ms with 12 Hz vs 138–150 ms with 25 Hz (Tab.2).
- **Not disclosed:** hyperparameters, compute, and the tokenizers' training data.

## Architecture

**Qwen-TTS-Tokenizer-25Hz** (p.2–3, Fig.2a, Tab.3)
- 1 codebook × 32,768 entries at 25 Hz (40 ms). That is 375 bps (derived: 15 bits × 25).
- **Encoder.** Qwen2-Audio's encoder is split in two, with a resampling layer and a VQ layer inserted at an intermediate depth. The pipeline is encoder-part-2 → VQ → encoder-part-1 → Qwen2-Audio LLM, which outputs the transcript. A convolutional mel decoder reconstructs mels from the tokens during training only.
- **Rationale** (p.2): purely semantic tokens lack expressiveness, while purely acoustic tokens burden the LLM with low-level detail and long-horizon error accumulation. So this tokenizer mixes both kinds of cue.
- **Detokenizer** (p.3, p.6):
  - A flow-matching DiT maps codes to mel; a modified BigVGAN maps mel to waveform.
  - Streaming uses a block attention mask: the current block, 3 blocks back and 1 block ahead (the paper calls this "4 blocks").
  - Blocks are 8 tokens. The first DiT chunk needs 16 LM tokens. Each 8 tokens produce 320 ms of mel, and BigVGAN adds 130 ms of right look-ahead, so the first packet carries ~190 ms of audio.
  - The authors concede that the single codebook limits ultra-low-latency use and general synthesis (p.2).
- Whether the DiT is conditioned on a speaker embedding or prompt mel is not reported.

**Qwen-TTS-Tokenizer-12Hz** (p.2–4, Fig.2b, Tab.4)
- 12.5 Hz (80 ms frames). 16 codebooks × 2048 entries: codebook 0 semantic, codebooks 1–15 an acoustic RVQ. That is 200 codes/s or 2.2 kbps (derived: 16 × 11 bits × 12.5).
- **Design and training:**
  - Uses Mimi's semantic/acoustic disentangled quantization.
  - Trained as a GAN directly on waveforms, with a multi-scale mel reconstruction loss.
  - A WavLM teacher guides codebook 0 through a cosine-similarity loss (Fig.2b).
- **Fully causal, no look-ahead.** The encoder is a downsampling ConvNet + window transformer; the decoder is a window transformer + upsampling ConvNet.
- **No speaker vector or diffusion.** The authors say the extra capacity lets a small causal ConvNet reconstruct waveforms without either (p.2).
- **Released config** (model card):
  - 24 kHz in and out, 1920-sample hop.
  - Encoder: causal conv, 8 transformer layers with window 250, configured with 32 quantizers but encoder_valid_num_quantizers = 16.
  - Decoder: 8 layers with window 72, upsampling [8,5,4,3] × [2,2].
  - semantic_codebook_size is 4096, against 2048 in Tab.4.
  - 0.2B parameters in total.

**TTS model** (p.4–5, Fig.3)
- Qwen3 backbone, Qwen text tokenizer, and a jointly trained speaker encoder.
- **Dual-track input.** Text and acoustic tokens are "concatenated along the channel axis". Fig.3 shows a text row and a codec row at every position:
  - [instruction text | codec pad];
  - optional think tokens;
  - an optional speaker embedding;
  - then [target text | codec embedding].
  - On each new text token, the model immediately predicts the matching acoustic tokens (p.4).
- **25 Hz model:** the backbone combines text features with the previous speech tokens and predicts the current token through a linear head. A chunk-wise DiT renders the result.
- **12 Hz model:** the backbone predicts codebook 0 from the aggregated (summed, Fig.3) codebook features. The MTP then generates codebooks 1–15 sequentially per frame (Fig.3).
- **Voice cloning:** from a speaker embedding of reference speech (real-time), or by in-context learning on a text–speech pair, which keeps prosody better (p.5). The released API takes ref_audio + ref_text; x_vector_only_mode uses the embedding alone, at reduced quality (GitHub).
- **Voice design:** instructions are prepended, and a probabilistically activated "thinking pattern" is used in training to improve instruction following (p.5).
- **Streaming packets:** a 12 Hz packet is 4 tokens = 320 ms. This is a scheduling choice; a single token can in principle be decoded on its own (p.6).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **25 Hz tokenizer** (p.3):
  - Stage 1: continue pretraining Qwen2-Audio on ASR with the resampling and VQ layers inserted.
  - Stage 2: fine-tune the whole model together with the convolutional mel decoder. The reconstruction loss injects acoustic information into the tokens.
  - Not reported: DiT/BigVGAN training, data, steps, compute.
- **12 Hz tokenizer** (p.3–4): GAN + multi-scale mel loss + WavLM distillation into codebook 0; 15-layer RVQ; causal throughout. Data, steps and compute: not reported.

### Stage 1: General pretraining (S1)
- More than 5M h of multilingual speech, used to learn a monotonic text→speech mapping (p.5). Ten languages (p.2).
- Context 8,192 (implied by S3).
- Not reported: trainable parts, optimizer, LR, batch size, compute.

### Stage 2: High-quality CPT (S2)
- A dedicated pipeline stratifies data by quality, and pretraining continues on the high-quality subset.
- This reduces the hallucinations caused by noisy S1 data and markedly improves quality (p.5). Size not reported.

### Stage 3: Long-context (S3)
- Maximum length raised from 8,192 to 32,768, with long speech upsampled (p.5).
- At 12.5 Hz, 32,768 positions is ≈43.7 min of audio (derived).

### Stage 4: Post-training (p.5)
1. DPO on human-feedback preference pairs over multilingual samples.
2. GSPO with rule-based rewards, for capability and stability across tasks.
3. Lightweight speaker fine-tuning on the base model, giving the CustomVoice variants.

Sizes and steps: not reported. Whether the "-Base" checkpoints include stages 1–2 of post-training is unclear.

## Data pipeline

- More than 5M h across 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian (p.1; README).
- Quality stratification uses "a dedicated pipeline", with no details (p.5).
- Preference pairs come from human feedback (p.5).
- The target-speaker voice "Aiden" was fine-tuned only on monolingual data (p.10).
- Sources, filtering, synthetic data and labels: not reported.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| 25 Hz tokenizer | not reported | not reported | ASR stage → reconstruction stage |
| 12 Hz tokenizer | not reported | not reported | GAN + WavLM distillation |
| S1 general | >5M h (p.5) | not reported | 8,192 ctx |
| S2 high-quality CPT | not reported | not reported | quality-stratified subset |
| S3 long context | not reported | not reported | 32,768 ctx |
| Post-training | not reported | not reported | DPO → GSPO → speaker FT |
| Inference | — | one unnamed accelerator; internal vLLM (V0) + torch.compile + CUDA Graph (p.5) | Tab.2 |

## Preserving text intelligence / knowledge

Not evaluated.
- The paper says only that, being built on Qwen3, the model inherits robust text comprehension for voice design (p.5), and it adds the thinking pattern.
- No text benchmark is reported after training on >5M h with a pure TTS objective. Whether the backbone keeps any LLM knowledge is unknown; our inference is that it is not designed to.

## Key results

- **Tokenizer reconstruction** (LibriSpeech test-clean; Tab.4):

  | Tokenizer | Config | PESQ-WB | STOI | UTMOS | SIM |
  |---|---|---|---|---|---|
  | Qwen 12 Hz | 16 × 2048 @ 12.5 Hz | 3.21 | 0.96 | 4.16 | 0.95 |
  | Mimi | 16 × 2048 @ 12.5 Hz | 2.88 | 0.94 | 3.87 | 0.87 |
  | XY-Tokenizer | 8 × 1024 @ 12.5 Hz | 2.41 | 0.91 | 3.98 | 0.83 |

- **25 Hz tokens, ASR WER** (CV-EN / CV-CN / Fleurs-EN / Fleurs-CN; Tab.3):
  - after stage 1: 7.51 / 10.73 / 3.07 / 4.23
  - after stage 2: 10.40 / 14.99 / 4.14 / 4.67
- **Seed-TTS WER** (zh / en; Tab.5):

  | Model | zh | en |
  |---|---|---|
  | 12 Hz-1.7B | **0.77** | **1.24** |
  | 12 Hz-0.6B | 0.92 | 1.32 |
  | 25 Hz-1.7B | 1.10 | 1.49 |
  | 25 Hz-0.6B | 1.18 | 1.64 |
  | CosyVoice 3 | 0.71 | 1.45 |

- **Multilingual and cross-lingual:**
  - Best speaker similarity in all 10 languages (Tab.6).
  - Chinese→Korean WER 4.82 vs 14.4 for CosyVoice3 (Tab.7).
- **Voice design** (InstructTTSEval-ZH, 12 Hz-1.7B; Tab.8): APS 85.2, DSD 81.1, RP 65.1.
- **Long speech** (>10 min; long-zh / long-en WER; Tab.10): 25 Hz 1.517 / 1.225 vs 12 Hz 2.356 / 2.812.
- **Latency** (Tab.2):
  - First packet at concurrency 1 / 6: 12 Hz-1.7B 101 / 333 ms; 25 Hz-1.7B 150 / 523 ms.
  - Real-time factor at concurrency 1: 0.313 vs 0.253.

## Ablations & lessons

There are no formal ablations; these lessons come from comparing variants.
- **(a) Semantic vs acoustic content in tokens.** Adding mel reconstruction to the 25 Hz tokenizer raised ASR-from-tokens WER (Fleurs-EN 3.07 → 4.14; CV-EN 7.51 → 10.40). Tokens became less text-like but more expressive, a trade-off the authors accept (Tab.3, p.6).
- **(a) Same-shape codec, better quality.** With the same bit budget as Mimi (16 × 2048 @ 12.5 Hz), Qwen's training choices still raised reconstruction quality: PESQ-WB 3.21 vs 2.88, SIM 0.95 vs 0.87 (Tab.4). SIM 0.95 means timbre survives in the codes.
- **(a) Frame rate under AR modeling.** 12.5 Hz consistently beats 25 Hz on short-form WER (Tab.5); the authors attribute this to coarser frames easing long-term dependencies (p.7). Over 10 minutes the result reverses: 25 Hz gets 1.517 / 1.225 vs 2.356 / 2.812, which the authors attribute to the semantic tokens (p.11).
- **(e) Model size.** Going from 0.6B to 1.7B improves both tokenizers' models (12 Hz: 0.92 / 1.32 → 0.77 / 1.24) (Tab.5, p.7).
- **(e) Latency.** The left-context-only 12 Hz decoder costs 4–5 ms per packet regardless of load. The 25 Hz DiT + BigVGAN costs 25–147 ms and needs lookahead (Tab.2, p.6).
- **(a) Data quality and post-training.** High-quality CPT reduces hallucinations, and post-training lifts 12 Hz-1.7B to SOTA test-en WER (p.5, p.7). Neither claim comes with before/after numbers.
- **(d) Speaker fine-tuning generalizes.** Fine-tuning on one monolingual voice transfers to all 10 languages, beating GPT-4o-Audio-Preview on 7 of 10 (Tab.9).

## Implications for our setup

**What each tokenizer demands from an LM that models it directly in full duplex** (numbers derived from Tab.3–4, p.3–6, and configs):

| | Qwen-TTS-Tokenizer-12Hz (public) | Qwen-TTS-Tokenizer-25Hz (not public as of 2026-10-12) |
|---|---|---|
| Frame / codebooks | 80 ms; 16 × 2048 | 40 ms; 1 × 32,768 |
| Codes/s and bitrate per stream | 200 codes/s; 2.2 kbps | 25 tokens/s; 375 bps |
| Codes per 80 ms tick, two streams | 32 | 4 |
| Backbone steps per second | 12.5, plus 15 sequential MTP steps per frame | 25 |
| Acoustic detail the LM must predict | essentially all of it: 15 of 16 codes per frame (93.75% of bits) are acoustic RVQ, covering timbre, prosody, room and noise | partial: tokens are ASR-derived with acoustic cues added; the DiT/BigVGAN fills in the rest |
| Decoder needs a speaker prompt? | No: it maps codes directly to 24 kHz audio, so voice identity must come from the LM | Not reported |
| Decoder lookahead | none beyond the current frame; decode ≈1 ms per frame (derived: 4 ms per 4-token packet, Tab.2) | ≥1 block (8 tokens = 320 ms) + 130 ms BigVGAN ≈ 450 ms; first output needs 16 tokens (640 ms) |
| Measured strength | short-form WER, low latency | long-horizon stability |

1. **For full duplex, 12 Hz is the only workable choice of the two.** A duplex LM produces frames in lockstep with wall-clock time and cannot run ahead the way a TTS model does. So the 25 Hz decoder's ≈450 ms lookahead would become response latency. The 12 Hz codec's per-frame (80 ms) causal decode fits.
2. **The cost of 12 Hz is that the LM owns all acoustics.** There is no speaker vector in the decoder, and the codec faithfully keeps paralinguistics and the acoustic environment (README). The LM therefore has to predict room, noise and any channel cross-talk in our duplex recordings, as well as voice identity. Recommendations:
   - clean or separate the agent channel;
   - condition on a speaker embedding (as Qwen3-TTS does) or fine-tune to one voice;
   - follow S2 and finish on a high-quality subset.
3. **Model the 16 codebooks hierarchically, as Qwen does.** Feed the summed codebook embeddings in, predict codebook 0 with the backbone head, and generate codebooks 1–15 with a small AR module (Qwen: 5 layers, hidden 1024, per the model-card config). One option, which is our suggestion and not from the paper, is to up-weight the loss on codebook 0, since it carries the content.
4. **Codebook 0 is WavLM-distilled, not ASR-supervised** (unlike the 25 Hz tokens). Its text-alignment is not reported. In Qwen3-TTS the text token is always given at the same position, so the LM never has to invent content in codes. This is the strongest argument for putting a text track (the dual-track layout) on our agent stream.
5. **Long sessions drift at 12.5 Hz.** The >10-minute WER is 2.36 / 2.81 vs 1.52 / 1.23 for the semantic 25 Hz tokens (Tab.10). Duplex conversations are long, so plan for long-context training (S3-style, 32K positions ≈ 43.7 min of a single stream, derived) and add a text track to anchor the content.
6. **Data scale.** Our 10K h is 0.2% of S1's >5M h, or 450M frames (7.2B codes) per stream (derived: 10,000 h × 3,600 s × 12.5 Hz, × 16 codebooks). Don't try to learn rendering from that; reuse Qwen3-TTS-12Hz-1.7B, which already maps text to these exact codes. Its backbone matches Qwen3-1.7B's layer, hidden, head and FFN sizes (per the configs), and its code predictor can initialize our depth module. It could also serve as the Talker in a Thinker–Talker design behind our Qwen3 LLM.
7. **Stability levers.** DPO plus GSPO with rule-based rewards (p.5) is the Qwen recipe; ASR-WER-based rewards are the cheapest to build.

## Open questions / unclear points

- The DiT receptive field is called "4 blocks" but listed as current + 3 back + 1 ahead (p.3). Qwen2.5-Omni used 2 back + 1 ahead.
- Long-speech WERs differ between the text (1.533 / 1.571) and Tab.10 (1.517 / 1.225) (p.11).
- Tab.5 lists "-Base" models but the text attributes the 1.24 to post-training (p.7).
- The preset-voice count is printed as a placeholder "x" (p.2).
- Configs: semantic_codebook_size 4096 vs 2048 in Tab.4; the encoder is configured with 32 quantizers, 16 of them valid.
- Is the 12 Hz tokenizer identical to Qwen3-Omni's codec? The configs match on codebook parameters but differ in decoder hidden sizes (see qwen3-omni.md).
- Not reported: whether the 25 Hz DiT needs a speaker prompt; the dual-track alignment policy (pads when text and audio lengths differ; whether "concatenated" means summed); the speaker-encoder architecture; tokenizer training data.
