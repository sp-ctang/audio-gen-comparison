# Nemotron-Labs-Audex-30B-A3B ("Audex"): Unified Audio Intelligence Without Regressing on Text Intelligence

| Field | Value |
|---|---|
| Org | NVIDIA |
| Date / arXiv | 2026-06-08 (p.1 header); arXiv 2607.05196v2 (7 Jul 2026) |
| Category | omni: unified audio–text understanding plus speech/audio generation. Not duplex. S2S is evaluated only as a cascade run inside the one model (ASR → text reasoning → TTS) |
| Backbone | Nemotron-Cascade-2-30B-A3B: hybrid Mamba2-Transformer MoE, 52 layers, d 2688, 128 routable / 6 active experts, ~3B active. Initialized from Cascade-2's checkpoint **after text SFT** (p.2, p.4). Audex-2B uses a 2B dense Transformer (28 layers, d 2048), trained text-only with Cascade-2's data and recipe (p.4) |
| Audio input | AF-Whisper (Whisper-Large-v3 architecture, finetuned in Audio Flamingo 3). 16 kHz input, 25 Hz features (dim 1280) over 30 s windows, 2-layer MLP adapters into the LLM embedding space (p.5) |
| Audio output | Codec tokens added to the vocabulary and predicted by the normal LM head with plain cross-entropy. **Speech:** X-Codec2, 50 Hz, single FSQ codebook of 65,536 → 50 tok/s. **Non-speech:** X-Codec, first 4 of 8 RVQ layers (1,024 codes each), flattened → 200 tok/s. No depth transformer, delay pattern or MTP (p.5) |
| Waveform decoder | Speech: a new streaming causal-ConvNeXt (Vocos-style) decoder replaces X-Codec2's non-causal decoder. "Small look-ahead", size not reported; 16 kHz output. Non-speech: X-Codec decoder plus an enhancement VAE (BigVGAN-v2 recipe) that upsamples to 48 kHz (p.5–6) |
| Duplex mechanism | n/a: duplex is "outside the scope of this paper" (p.18) |
| Open weights / code | 30B-A3B and 2B checkpoints released (p.1). Code release not stated |
| Relevance to us | **High.** The most detailed public recipe for adding speech generation to a strong LLM with near-zero measured text regression (text-replay ratios, embedding freezing, stage order, text RL afterwards). Caveat: no speech continuation, no native S2S, no duplex |

## TL;DR

- **No audio pretraining.** Audex adds audio understanding, TTS and text-to-audio (TTA) to a 30B-A3B reasoning MoE purely through multi-stage SFT: 1.09M h of paired audio–text data (157.4B audio tokens), mixed with the backbone's own 303B-token text-SFT set. Text-only Cascade RL plus multi-domain on-policy distillation (MOPD) follows (p.1, p.8–10).
- **Text regression is ≤~2.5 points on most benchmarks.** Backbone → Audex: MMLU-Pro 79.8→78.9, GPQA 76.1→74.9, AIME25 92.4→91.2. The exceptions are IFBench (82.9→77.8) and NIAH@1M (99.0→83.4) (Tab.1). For contrast, Qwen3-Omni-30B-A3B-Thinking scores 73.7 on AIME25 against 85.0 for Qwen3-30B-A3B-Thinking-2507 (p.3).
- **What preserved text ability:**
  - frozen text embeddings while warming up the new tokens;
  - **44–75% text tokens in every stage**, raised when audio tasks emit text;
  - a curriculum of generation first, understanding second.
  
  A one-shot mix needed 88% text and still destroyed long context (p.9–11, p.26–27).
- **Text-only RL afterwards left audio metrics essentially intact** while restoring instruction following, alignment and long context (Tab.15).
- **"S2S" (BigBenchAudio 90.0) is ASR → text reasoning → TTS inside one model.** Knowledge reaches speech through text, not through native speech generation (p.16, p.23–24).

## Architecture

- **Single decoder.** Audio enters as continuous encoder features through the adapters. Text and codec tokens share one autoregressive output space. There is no thinker–talker split. The design is compatible with Megatron-LM and vLLM (p.2, Fig.1).
- **Vocabulary.** 131,072 base entries, plus 65,536 speech codes, 8,192 audio codes (all 8 RVQ layers reserved) and 5 special tokens (`<speechgen_start/end>`, `<audiogen_start/end>`, `<audiogen>`). That totals 204,805, padded to 205,312. New embeddings are initialized N(0, 0.02) (p.4–5).
- **Separate codecs.** Speech uses a low-rate single-layer codec; general audio uses RVQ at a higher token budget (p.5).
- **Task formats are all chat templates.** Audio QA, ASR and AST put `<sound>` in the user turn and answer in non-thinking mode. TTS takes the user turn `<|text to speech|> Generate speech for this transcription. {text}` and the assistant emits `<speechgen_start>` + codec tokens + `<speechgen_end>`. TTA works the same way from a caption (p.6–7, Fig.2). **Text and speech tokens are never interleaved in the output**, and speech is always generated from a complete transcript.
- **Loss.** Cross-entropy over trainable tokens. Packed samples use "square averaging": each sample's summed loss is divided by sqrt(length), and the total is normalized by the sum of sqrt(lengths) (Eq.2, p.8). MoE routing uses aux-loss-free balancing (update rate 1e-3) plus a load-balancing loss with coefficient 1e-6 (p.8).
- **Classifier-free guidance (CFG).** For 10% of speech/audio-generation samples, the transcript or caption is replaced by a random number of untrainable pad tokens. At inference, conditional/unconditional pairs are decoded in the same vLLM batch and a custom logits processor combines them (p.8, p.12).

## Training pipeline

Every stage that touches the LLM is "SFT" on paired data: "it is possible to build a best-in-class audio LLM without audio- or speech-related pretraining" (p.9). The release uses the **multi-stage** path. The table below is from Tab.3 (p.10) in epochs. I converted epochs to tokens using Tab.2 per-epoch sizes: text 303.0B, ASR+AST 39.3B, audio understanding 29.9B, TTS 81.0B, TTA 24.7B. The derived text shares reproduce the paper's "Text weights (%)" row exactly, so the shares are by tokens.

| Stage | Text ep | ASR+AST | Aud-Und | TTS | TTA | Text share | Trainable | GBS | LR max→min | WD | Tokens (derived) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Text SFT (Cascade-2, reused) | 1.6 | 0 | 0 | 0 | 0 | 100% | LLM | 64 | 5e-5→5e-6 | 0.1 | ≈485B |
| Audio warmup | 0.5 | 0 | 2.0 | 1.0 | 2.0 | 44% | MLP adapters + audio-token emb. | 64 | 2e-3→1e-5 | 0 | ≈342B |
| Audio Gen. SFT | 0.5 | 0 | 0 | 1.0 | 1.0 | 59% | LLM | 16 | 2e-5→1e-6 | 0.05 | ≈257B |
| Audio Gen. + Und. SFT | 1.5 | 1.0 | 2.0 | 1.0 | 1.0 | 69% | MLP + LLM | 64 | 2e-5→1e-6 | 0.05 | ≈659B |
| *Alt: consolidated SFT (after the same warmup recipe, applied to the base model)* | 2.0 | 1.0 | 2.0 | 1.0 | 1.0 | 75% (later 88%) | MLP + LLM | 64 | 2e-5→1e-6 | 0.05 | ≈811B |

Shared settings: 5% warmup (200 steps for text SFT), cosine schedule, AdamW (0.9, 0.999; β2 0.98 for text SFT), gradient clip 1.0 (Tab.3, p.11). The audio encoder is always frozen (p.9). Steps and per-stage time are not reported. The multi-stage audio path totals ≈1.26T data tokens, against ≈1.15T for the single-stage path (derived; these counts include prompt tokens and encoder positions).

### Stage 0: tokenizer / codec / decoder training
The encoder is AF-Whisper, used as is and frozen. X-Codec2 and X-Codec are off the shelf. The new pieces are the streaming causal speech decoder and the 48 kHz enhancement VAE for non-speech; their data and compute are not reported (p.5–6).

### Stage 1: text SFT
Nemotron-Cascade-2's own SFT checkpoint, reused. Data is 33M samples and 303.0B tokens, filtered to ≤200K tokens per sample (Tab.2, p.6, p.9).

### Stage 2: audio warmup
The vocabulary is extended and the encoder and adapters attached. Only the adapters and **audio-token** embeddings train. **Text-token embeddings are frozen by gradient masking, and the LLM is frozen.** LR is 2e-3 on a mixed audio + text set (p.9, p.26). The paper does not say what the text data does here; presumably it shapes only the audio-token output rows.

### Stage 3: Audio Gen. SFT
The full LLM is unfrozen and trained on TTS + TTA with 59% text. Audio-to-text tasks are deliberately excluded because token generation is "entirely new" and "require[s] more training" (p.9).

### Stage 4: Audio Gen. + Audio Und. SFT
Adapters and LLM train; ASR, AST and audio understanding are added, with text raised to 69%. The output is the "Audex SFT" checkpoint (p.9). An extra stage then "remove[s] speech prompting abilities" (voice cloning) for responsible release. Its recipe is not reported (p.9 fn.4).

### Alternative (ablated, not released): single-stage consolidated SFT
Audio warmup runs on the pretrained base (with no text SFT), followed by one SFT stage over all data (p.9).

### Stage 5: text-only Cascade RL + MOPD
The Nemotron-Cascade-2 sequence with **no audio data**: IF-RL → multi-domain RL → MOPD → RLHF → long-context RL (p.10). Hyperparameters, rewards and compute are not reported. RL is applied only to the 30B model; Audex-2B is the SFT checkpoint (p.11 fn.5).

## Data pipeline

- **Totals (Tab.2, p.8).**

  | Task | Samples | Hours | Audio tokens | Text tokens |
  |---|---|---|---|---|
  | TTS | 147M | 421K | 75.8B | 5.2B |
  | TTA | 12M | 34K | 24.3B | 0.4B |
  | Audio understanding | 49M | 308K | 27.7B | 2.2B |
  | ASR | 95M | 201K | 18.1B | 4.4B |
  | AST | 58M | 128K | 11.5B | 5.3B |
  | Text SFT | 33M | — | — | 303.0B |
  | **Total** | **394M** | **1,092K** | **157.4B** | **320.5B** (477.9B combined) |

  The counts imply 25 tok/s for input audio (the encoder rate), 50 tok/s for TTS and about 200 tok/s for TTA (derived).
- **ASR/AST.** Permissively licensed public corpora, with DNS-Challenge-5 noise and RIR augmentation. ASR targets are a language tag plus the transcript. AST targets are the source language, the transcript, then the English translation (p.6).
- **TTS.** Public TTS corpora plus clean ASR clips, 0.1–120 s and mostly under 20 s. About 8% is re-voiced into one fixed Tortoise voice with Qwen3-TTS-12Hz-1.7B-Base to create a fixed-voice mode (p.6).
- **Audio understanding** follows AF3 and AF-Next (single-turn QA, ≤900 s). **TTA** data is filtered and fixed at 10 s (p.6).
- No conversational, expressive or synthetic-dialogue speech, and no duplex phenomena. Audio is pre-tokenized (base64), with online knapsack packing and a buffer of 8,192 (p.12).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Codec decoder, VAE | not reported | not reported | |
| Text SFT (reused) | ≈485B (derived) | not reported | from Cascade-2 |
| Audio warmup → Gen. → Gen.+Und. | ≈342B → 257B → 659B (derived) | 512× H100, BF16 (p.11); time not reported | TP4/EP32/CP8 + SP, 262,144-token context |
| Cascade RL + MOPD | not reported | not reported | text-only |
| Dataset per epoch | 1,092K h; 157.4B audio + 320.5B text tokens | | Tab.2 |

## Preserving text intelligence / knowledge

- Start from a strong *post-SFT* text checkpoint and replay **the backbone's own SFT data**: the same 303B tokens (p.2, p.6).
- Text share by tokens: 44% in warmup, 59% in generation, 69% in Gen + Und, and 75→88% for the single-stage alternative (Tab.3, p.26–27).
- Freeze text embeddings and the LLM during new-token warmup; train the LLM at 2e-5 afterwards (p.9).
- Learn generation first; add text-output understanding tasks later and with more replay (p.26).
- Finish with text-only RL. Changes from the SFT checkpoint to the final model: IFBench 52.4→77.8, ArenaHard 59.4→81.6, AA-LCR 27.0→39.3, MMLU-Pro 77.5→78.9 (Tab.15).
- Text drift during audio SFT, comparing Tab.16 (after warmup) with Tab.15 (after SFT): MMLU-Pro / GPQA / AIME25 go from 78.9 / 73.8 / 93.2 to 77.5 / 73.8 / 91.7. That is a loss of ≤1.5 points across ≈0.9T tokens of audio-heavy SFT (derived; assumes identical eval settings in both tables).

## Key results

- **Text (backbone → Audex, Tab.1):**
  - AIME25 92.4 → 91.2 (with tool-integrated reasoning, TIR: 98.6 → 98.3)
  - HMMT 94.6 → 92.2
  - LiveCodeBench v6 87.2 → 85.3
  - GPQA-D 76.1 → 74.9
  - ArenaHard v2 83.5 → 81.6
  - IFBench 82.9 → 77.8
  - NIAH 256K|1M 100|99.0 → 99.4|83.4
  - SWE-Verified 50.2 → 48.2
- **TTS, Seed-TTS-Eval en (Tab.6):**
  - Fixed-voice WER: 1.70 for 30B and 2.04 for 2B, against Qwen3-Omni-Instruct 1.39 and MiMo-Audio 5.37.
  - Prompted voice: WER 2.07 but **SIM 45.3**, against F5-TTS 64.7 and Seed-TTS 76.2. The authors blame the codec choice and recipe (p.12).
- **ASR / AST / audio understanding** (Tab.1, Tab.8, Tab.11):
  - OpenASR average WER 6.82 (Qwen3-Omni-Instruct 5.72); LibriSpeech clean|other 1.34|3.06.
  - Fleurs xx→en BLEU 34.0.
  - MMAU 75.6, MMAR 63.2, MMSU 63.4, which lags SOTA on MMAR and MMSU.
- **TTA FD_openl3 (Tab.7):** AudioCaps 66.9 and SongDescriber 62.7. The SFT checkpoint's 60.9 and 58.1 are SOTA among public-data models.
- **Speech interaction (in-model cascade):**
  - BigBenchAudio 90.0 (2B: 64.3), against Qwen3-Omni Flash 58.7 and Moshi 4.4 (Tab.13).
  - VoiceBench SD-QA 60.2 against 78.1 for Qwen3-Omni-Thinking; IFEval 88.0 against 80.6 (Tab.14).

## Ablations & lessons

- **Freezing text embeddings during warmup is the largest effect on text preservation** (Tab.16, p.26):

  | Warmup variant | MMLU-Pro | GPQA-D | AIME25 |
  |---|---|---|---|
  | Freeze text embeddings | 78.9 | 73.8 | 93.2 |
  | Unfreeze all | 71.2 | 62.3 | 71.8 |
  | Freeze, then unfreeze | 65.1 | – | – |
  | Unfreeze, with more text | 59.7 | – | – |

  Unfrozen runs "recover after 10%–20%" of the next stage but are "not comparable" to the frozen run (Fig.5). Warmup LR was 2e-3.
- **Audio tasks with text outputs need more text replay.** In Gen + Und, 0.56 text gave "considerable degradation" and 0.69 did not (Fig.6). In the generation stage 0.59 sufficed, by the paper's account, because audio targets there are not text (p.25–26). The comparison was run over only ~30% of the stage.
- **One-shot mixing is unstable.** At 0.75 text the single-stage run showed a text "cliff" at 50–60% of training, and again at 60–80% after a reseeded restart. 0.88 fixed it (Fig.7, p.26–27).
- **Staging protects long context.** NIAH 256K|1M was 6.0|0.0 single-stage against 99.3|86.8 multi-stage. The authors' explanation: long text and long audio are "too different" to learn simultaneously (Tab.4, p.10–11).
- **Other single-stage effects** (Tab.4, p.10): slightly better ASR (Fleurs-ML 4.79 vs 5.07); worse TTA (AudioCaps FD 69.4 vs 60.9), BigBenchAudio (84.6 vs 88.0) and HMMT (87.1 vs 92.2). It is "much harder to stabilize", though cheaper in epochs.
- **Text-only RL barely affects audio** (Tab.15, p.25):
  - MOPD temporarily hurt noisy ASR (2.75→3.32) and MMSU (63.3→60.9); both recovered after two more stages (2.92, 63.4).
  - Fixed-voice TTS WER went 1.76→1.70; BigBenchAudio rose 88.0→90.0 as text reasoning improved.
  - TTA FD degraded (60.9→66.9).
- **LLM size drives cascade speech intelligence; the encoder drives ASR.** 2B vs 30B: BigBenchAudio 64.3 vs 90.0, but OpenASR 7.14 vs 6.82 (p.15, Tab.13).
- **Decoding for text-conditioned speech:**
  - TTS is best at temperature 0.1 and top-k 80. CFG is optional: λ≈1.5 helps slightly at twice the compute.
  - TTA needs CFG λ∈[3,5] and temperature 1.0 (Fig.4, Fig.8–9, p.14, p.28).
- **Variable-length generation targets destabilized training**, giving random durations and content (tested on TTA). A fixed 10 s length was used instead (p.13–14).
- **The single-codebook 50 Hz codec setup** gives excellent intelligibility (fixed-voice WER 1.70; prompted 2.07) but weak speaker similarity (prompted SIM 45.3). The authors suspect the codec and recipe (Tab.6, p.12).

## Implications for our setup

- **Our duplex-only mid-training is the opposite of what Audex needed.** Audex never went below 44% text tokens, and needed ≥69% once audio tasks emitted text. Our ~10K h of duplex-only speech will likely erode Qwen3 knowledge, which fits our symptom. **Action:** replay high-quality text, ideally instruction-style data near Qwen3's post-training distribution, at about 50–70% of tokens. Eval MMLU-Pro, GPQA, AIME-style and NIAH at every checkpoint and watch for cliffs, not just drift.
- **Warm up new speech-token embeddings in isolation.** If the Qwen3 audio codes are new vocabulary rows, train only those rows (plus any adapter) with the LLM and text embeddings frozen (Audex: LR 2e-3). Then unfreeze the LLM at a low LR (2e-5, WD 0.05). Touching text embeddings early cost Audex 7–21 points.
- **Stage it.** Learn speech-token generation with text replay first, then add listening, understanding and duplex. A two-stream duplex model needs long contexts, and Audex shows a one-shot mix can silently destroy long-context ability.
- **Knowledge in speech arrived via text.** Audex's BigBenchAudio 90.0 comes from transcribe → reason in text → TTS, and nothing here shows that pure speech-token continuation inherits LLM knowledge. This argues for a text or inner-monologue stream that leads or accompanies our speech tokens.
- **Continuation quality likely needs much more clean speech-generation data.** Audex used 421K h of TTS (75.8B speech tokens), about 42 times our total, to reach WER 1.70. Adding a TTS-format (text → speech tokens) corpus alongside duplex data is the closest Audex analogue.
- **Check decoding before blaming training.** Audex's best text-conditioned speech-token sampling was temperature 0.1 and top-k 80; high-temperature sampling may explain part of our "weak continuation".
- **Text RL or on-policy distillation from the original Qwen3 is a plausible repair step,** since text RL left Audex's speech skills intact. It is untested when speech is generated in the same pass as reasoning, as in duplex.
- **Use square-root loss averaging when packing long duplex sequences with short text samples** (Eq.2). It is low-cost, but Audex did not ablate it.

## Open questions / unclear points

- No steps, wall-clock or GPU-hours per stage; only "512 H100" overall (p.11). The packed sequence length per stage is not stated.
- Why text data appears in warmup when every text-side parameter is frozen is not explained.
- The single-stage checkpoint in Tab.4: was it trained at 0.75 or 0.88 text? Tab.3 says 75%; App. C.2 later uses 0.88.
- The prompted-TTS training format and the "remove speech prompting" stage are undocumented. The SIM number predates that removal.
- The streaming decoder's look-ahead, latency and training data are not reported.
- Nothing is tested on speech continuation, interleaved speech–text, speech generated without a full transcript, or spoken multi-turn dialogue. Applying these findings to duplex S2S is extrapolation.
- No 2B text-only baseline scores are given, so 2B regression is unknown. The claims that early fusion and MoE help are cited, not tested (p.17–18).
