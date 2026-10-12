# Qwen3-Omni Technical Report (Qwen3-Omni-30B-A3B)

| Field | Value |
|---|---|
| Org | Qwen Team, Alibaba (p.1) |
| Date / arXiv | 2025-09-22 (p.1); no arXiv ID is printed in the provided PDF |
| Category | omni (any input → text + streaming speech); half-duplex streaming S2S |
| Backbone | **Thinker:** MoE 30B-A3B, LLM initialized from Qwen3 (Tab.1, p.6; p.7). **Talker:** MoE 3B-A0.3B, init not reported (Tab.1). Released config: Thinker 48 layers, hidden 2048, 128 experts / 8 active; Talker 20 layers, hidden 1024, 128 experts / 6 active, codec vocab 3072 (model card config: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json) |
| Audio input | AuT encoder (~0.6B; 650M in Tab.1), trained from scratch on 20M h of supervised audio. 16 kHz input, 128-bin mel, 10 ms hop, 8× Conv2D downsampling → **continuous** features at 12.5 Hz (80 ms per frame) (p.4–5) |
| Audio output | RVQ codec tokens at **12.5 Hz** (p.2), **16 codebooks**: Fig.2 labels them 0–15 (p.3); config num_code_groups 16, codebook size 2048. The backbone's linear head predicts codebook 0; an 80M dense "MTP" module then predicts residual codebooks 1–15 autoregressively, in a fixed number of steps (depth-transformer style) (p.5–6) |
| Waveform decoder | **Code2Wav:** "lightweight causal ConvNet", 200M params, left context only, emits waveform after every frame; 3 ms per code (Tab.1–2, p.5–6). Config: 8 attention layers (sliding window 72) + conv upsampling 8·5·4·3·2·2 = 1920 samples per frame at 24 kHz (model card config; product derived) |
| Duplex mechanism | **n/a.** Turn-based streaming only (chunked input prefill, streaming output). The paper never mentions duplex, barge-in or backchannels. The README advertises natural turn-taking but describes no mechanism (GitHub: https://github.com/QwenLM/Qwen3-Omni) |
| Open weights / code | Apache 2.0: -Instruct (Thinker + Talker), -Thinking (Thinker only), -Captioner (p.1; GitHub) |
| Relevance to us | **High.** Reference design for the Qwen3 12.5 Hz / 16-codebook codec + MTP + causal decoder we build on, and the main evidence on adding audio while keeping text ability |

## TL;DR

- **Thinker–Talker split.** The Thinker writes text. The Talker renders speech from two inputs: its own embeddings of the text tokens, and audio/visual hidden states taken from the Thinker. Unlike Qwen2.5-Omni, it no longer reads the Thinker's text hidden states (p.3).
- **Speech output path.**
  - 12.5 Hz frames, 16 codebooks. The backbone predicts codebook 0, the MTP predicts codebooks 1–15, and causal Code2Wav decodes each frame.
  - Theoretical cold-start first packet: 234 ms for audio, 547 ms for video (Tab.1–2).
- **Pretraining** (p.7):
  - S1: encoder alignment, LLM frozen.
  - S2: ~2T tokens (text 0.57T, audio 0.77T, image 0.82T, video 0.05T, video-audio 0.05T), all parameters trainable.
  - S3: 32,768-token context.
- **Text ability.** A controlled base-model study shows no text or vision degradation (Tab.16). The post-trained models still trail the text-only Qwen3 "2507" checkpoints (Instruct-2507 / Thinking-2507) on most text benchmarks (Tab.4–5).
- **Talker training, 4 stages:** multimodal-context→speech pretraining, then high-quality continued pretraining (CPT) plus long context, then DPO, then speaker fine-tuning (p.8). No hours, hyperparameters or compute are given anywhere.
- **No duplex.** There is no duplex capability, data or evaluation. Speech output is evaluated only as TTS (p.13).

## Architecture

**Thinker**
- Qwen3-initialized MoE LLM with a 151,643-token BPE vocabulary (p.4).
- TM-RoPE splits rotary angles 24/20/20 across time/height/width.
- Audio gets one temporal ID per 80 ms, anchored to absolute time. Qwen2.5-Omni instead used fixed 2-s chunks (p.5).
- Handles audio up to 40 min per instance (p.1).

**AuT encoder** (Fig.3, p.4)
- An attention encoder–decoder ASR model: 3 downsampling Conv2D blocks, 32 encoder layers and 8 decoder layers.
- Uses dynamic attention windows of 1–8 s so prefill can be cached in real time.
- Only the encoder is kept in Qwen3-Omni.

**Talker conditioning** (p.3, p.5, Fig.2)
- **Text tokens.** The Talker embeds the prompt and the Thinker's sampled tokens itself. The stated reason is that, for text, discrete tokens and embeddings carry the same information.
- **Audio/vision hidden states** "from middle layers" of the Thinker. The config sets accept_hidden_layer = 24 of 48. These are needed for audio–video-coordinated speech, e.g., keeping prosody and timbre in speech translation.
- **Its own previous frame**, fed back as the **sum of the 16 codebook embeddings**.
- The context also holds the history's text and the current turn's streamed text.
- Because text is decoupled, RAG, tool calls or safety filters can edit the text before it is spoken. The Thinker and Talker also take separate system prompts, for content style and audio style respectively (p.3).

**Codebook prediction** (p.5–6)
- The backbone sees the aggregated features of the current frame and its linear head predicts codebook 0.
- The MTP then fills in the residual codebooks. It is a dense transformer that runs a fixed number of autoregressive steps with a fixed KV cache. Config: 5 layers, hidden 1024, vocab 2048.
- Qwen argues this captures the full acoustic detail, which is what lets the vocoder be a small ConvNet.

**Relation to Qwen-TTS-Tokenizer-12Hz** (derived by comparing the two model-card configs)
- The paper never names or describes its codec.
- Code2Wav's released config matches the decoder of the public Qwen3-TTS-Tokenizer-12Hz on every codebook parameter:
  - 16 × 2048 quantizers, 1 of them semantic;
  - codebook_dim 512, decoder_dim 1536;
  - 8 layers, sliding window 72;
  - 1920-sample hop.
- The code-predictor shape and the codec special IDs also match Qwen3-TTS-12Hz (pad 2148, bos 2149, eos 2150).
- Hidden and FFN sizes differ: 1024/3072 vs 512/1024. Code2Wav alone is 200M parameters (Tab.1), while the full tokenizer is listed at 0.2B (model card: https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz).
- **Conclusion:** same code format, but identical codebooks or weights are not established.

**Streaming** (p.5–7)
- Chunked prefill. The Talker prefills each chunk asynchronously as the Thinker finishes it.
- MoE reduces KV-cache IO.
- Generation is left-context-only, so waveform can be emitted after every Talker token. One token = 80 ms of audio.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **AuT:** trained from scratch on 20M h. Mix: 80% Chinese/English pseudo-labeled ASR, 10% other-language ASR, 10% audio understanding (p.4). Optimizer and compute: not reported.
- **Vision encoder:** from Qwen3-VL, initialized from SigLIP2-So400m, ~543M params (p.5).
- **Speech codec and Code2Wav:** training not described in this paper (see qwen3-tts.md).

### Stage 1: Encoder alignment (S1)
- LLM frozen. Each encoder is trained separately against it, adapter first, then the encoder (p.7).
- The older joint encoder+adapter training on a frozen LLM was dropped: the encoder may compensate for the frozen LLM and degrade perception (p.7).
- Data: audio-text and image-text pairs, size not reported. Context 8,192 (implied).
- LR, batch size, compute: not reported.

### Stage 2: General (S2)
- All parameters trainable, ~2T tokens: text 0.57T, audio 0.77T, image 0.82T, video 0.05T, video-audio 0.05T (p.7).
- The parts sum to 2.26T (derived). Shares (derived): text ≈25%, audio ≈34%.
- Unimodal and cross-modal data are mixed from early pretraining. The paper calls this the "key ingredient" for non-degradation (p.1, p.7).
- Varied natural-language task prompts (p.7).
- Hyperparameters and compute: not reported.

### Stage 3: Long context (S3)
- Context raised from 8,192 to 32,768, with more long audio and video (p.7).
- Gains are claimed without numbers. Size not reported.

### Stage 4: Thinker post-training (p.7–8)
1. Light SFT in ChatML on text, visual, audio and mixed dialogues.
2. Strong-to-weak distillation: off-policy first, then on-policy. In the on-policy phase, student samples are trained to minimize KL divergence to the logits of Qwen3-32B or Qwen3-235B-A22B.
3. GSPO with rule-based rewards plus LLM-judge rewards (Qwen3; Qwen2.5-VL for visual tasks; references supplied).

Sizes and steps: not reported.

### Stage 5: Talker (p.8)
1. "hundreds of millions of speech data with multimodal context" teach a monotonic multimodal→speech mapping. The unit is not stated; hours not reported.
2. CPT on high-quality data, alongside long-context training. It reduces hallucinations caused by the noisy stage-1 data and improves quality.
3. DPO on multilingual speech preference pairs, for generalization and stability.
4. Speaker fine-tuning: specific voices, expressiveness, controllability.

Frozen parts and initialization: not reported.

### Stage 6: Captioner
Fine-tuned on a large dataset of detailed audio descriptions; size not reported (p.8).

## Data pipeline

Barely disclosed.
- Languages: 119 for text, 19 for speech input, 10 for speech output (Tab.3).
- AuT relies on pseudo-labeled ASR (p.4).
- Talker stage-1 data was noisy enough to cause hallucinations, fixed by a high-quality CPT stage (p.8).
- No dialogue synthesis, overlap, backchannel, interruption or silence handling is described.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| AuT | 20M h (p.4) | not reported | encoder ~0.6B |
| S1 | not reported | not reported | LLM frozen |
| S2 | ~2T tokens (0.77T audio) (p.7) | not reported | all params, 8,192 ctx |
| S3 | not reported | not reported | 32,768 ctx |
| Thinker post-training | not reported | not reported | SFT → distillation → GSPO |
| Talker | "hundreds of millions" of samples (p.8) | not reported | CPT → DPO → speaker FT |
| Inference | — | unnamed GPU; vLLM + torch.compile + CUDA Graph (p.6) | Tab.2 |

## Preserving text intelligence / knowledge

**Mechanisms**
- Content is produced as Thinker **text**; the Talker only renders it.
- Modalities are mixed from early pretraining, with ≈25% text in S2 (derived).
- S1 aligns the encoders against a frozen LLM.
- Distillation from much larger Qwen3 teachers, then GSPO (p.7–8).

**Controlled evidence** (Tab.16, p.14–15)
- Setup: same parameter count, same schedule, identical text and vision corpora, matched FLOPs. The only difference is that the Omni model adds audio and audio-visual data.
- Omni-Base vs Qwen3-30B-A3B-Base:
  - MMLU 81.69 vs 81.24
  - MMLU-Pro 61.57 vs 61.81
  - GSM8K 91.36 vs 90.83
  - MATH 60.42 vs 60.84
  - EvalPlus 73.96 vs 69.70
- Omni is equal or better on 8 of 13 text benchmarks; every loss is ≤0.96 points (derived).
- Stated conclusions (p.15):
  - no language degradation;
  - text helps audio and vision, but audio/vision give no measurable text gain;
  - audio improves MMMU and OCR.
- Run at one scale only, because of cost.

**Caveat after post-training**
- Omni-Instruct vs Qwen3-30B-A3B-Instruct-2507 (Tab.4): lower on 10 of 11 text benchmarks.
  - MMLU-Redux 86.6 vs 89.3
  - ZebraLogic 76.0 vs 90.0
  - IFEval 81.0 vs 84.7
  - It wins only AIME25 (65.0 vs 61.3).
- Omni-Thinking vs Qwen3-30B-A3B-Thinking-2507 (Tab.5): lower on 11 of 12.
  - AIME25 73.7 vs 85.0
  - BFCL-v3 63.2 vs 72.4
- The paper calls this "on par" (p.9). The comparison is uncontrolled.

**Speech in vs speech out**
- Spoken-input chat: VoiceBench 85.5 (Instruct) / 88.8 (Thinking) (Tab.7).
- Nothing measures the knowledge content of the *generated speech*.

## Key results

- **Latency** (Tab.2, 1 / 4 / 6 concurrent streams):
  - First packet 234 / 728 / 1,172 ms.
  - At 1 stream, 234 = 72 preprocessing + 88 Thinker TTFT + 57 Talker TTFT + 14 MTP + 3 decode (derived).
  - Talker 140 / 125 / 110 frames/s; real-time factor 0.47 / 0.56 / 0.66.
- **ASR** (Tab.6): LibriSpeech clean/other 1.22 / 2.48 WER; Fleurs 19-language average 5.33.
- **Audio reasoning** (Tab.7): MMAU 77.5 (Gemini-2.5-Pro 77.4).
- **Zero-shot TTS** (SEED WER, zh/en; Tab.13): 1.07 / 1.39. CosyVoice 3: 0.71 / 1.45. Qwen2.5-Omni: 1.42 / 2.33.
- **Multilingual and cross-lingual cloning:**
  - English WER 1.069, speaker similarity 0.773 (Tab.14).
  - Chinese→Korean WER 5.13 vs 14.4 for CosyVoice3 (Tab.15).

## Ablations & lessons

- **(a) Generation:** the multi-codebook + MTP + ConvNet Talker cut SEED WER from 1.42/2.33 (Qwen2.5-Omni) to 1.07/1.39 (Tab.13). This is not controlled: MoE, data and training all changed at once.
- **(a) Data quality and RL:** high-quality CPT after noisy large-scale pretraining reduces hallucinations (p.8), and RL "significantly" improves stability (p.13). Neither comes with numbers.
- **(a) Reasoning mode hurts perception:** Thinking does worse than Instruct on ASR, e.g., LibriSpeech-other 4.38 vs 2.48 and Fleurs average 8.63 vs 5.33 (Tab.17 vs Tab.6).
- **(b) Early mixing works at base level** (Tab.16), and audio adds nothing to text scores (p.15). Base-level parity does not fully carry through post-training (Tab.4–5).
- **(b) Encoder alignment:** align the adapter before the encoder, so the encoder cannot compensate for the frozen LLM (p.7). Rationale only.
- **(d) Scale:** 0.77T audio tokens in S2 and 20M h for AuT. If all 0.77T were 12.5 Hz frames, that would be ≈17.1M h (derived upper bound).
- **(e) Small Talker suffices:** 0.3B active params + 80M MTP + 200M decoder reach TTS WER that matches or beats specialist TTS systems (Tab.1, Tab.13).
- **(e) RTF check:** the paper's formula reproduces 0.47 = (1/75 + 1/140 + 0.014 + 0.003)/0.08 (derived).
- **(c) Duplex:** no evidence.

## Implications for our setup

1. **Separate knowing from speaking.** Qwen3-Omni never asks a model to produce knowledge directly in codec tokens: content is Thinker text, and the Talker conditions on text tokens. Our LM has to carry semantics through 16-codebook frames learned from 10K h, which is the likely root of our weak semantics. Two fixes:
   - a Thinker–Talker split, where the Qwen3 LLM emits text and a Talker renders codes (e.g., initialized from Qwen3-TTS-12Hz; see qwen3-tts.md);
   - a time-aligned text ("inner monologue") track on the agent stream.
2. **Text replay is required.** Qwen reached parity only by mixing early with ≈25% text, and still lost up to 14 points after post-training (Tab.4). Speech-only mid-training on an already-trained LLM is riskier still (see the Qwen2.5-Omni text drop in qwen2.5-omni.md).
   - Keep a sizeable text-only share plus paired speech–text tasks.
   - Track text benchmarks during mid-training.
   - On-policy KL distillation to the original Qwen3, as in Stage 4, is a cheap repair.
3. **Data gap.** Qwen's speech rendering rests on "hundreds of millions" of Talker samples, plus the 20M h AuT corpus on the input side. Qwen3-TTS used more than 5M h. Our 10K h is 2–3+ orders of magnitude smaller. Use the duplex data for interaction behavior, not to learn rendering from scratch.
4. **Codec modeling recipe.** For each 80 ms frame:
   - input = sum of the 16 codebook embeddings;
   - codebook 0 from the backbone head;
   - codebooks 1–15 from a small autoregressive depth module (Qwen: 80M parameters, 5 layers).
   Predicting all 16 codebooks in parallel, or flattening them (200 codes/s per stream, derived), is a prime suspect for weak continuation.
5. **No speaker prompt in the decoder.** All timbre lives in the LM's 15 acoustic codebooks. Qwen pins voices through speaker fine-tuning plus speaker-ID tokens (config: chelsie 2301 / ethan 2302 / aiden 2303). Condition the agent stream on a speaker ID or embedding, or finish with single-voice fine-tuning; otherwise expect timbre drift.
6. **Latency budget.** Qwen's MTP takes 14–18 ms and decode 3–5 ms per frame (Tab.2), leaving ≈57–63 ms of each 80 ms tick for the backbone (derived). A dense backbone that also ingests the user stream has to fit in that.
7. **User input stream.** Qwen encodes user audio as continuous AuT features, not codec tokens. These run on the same 80 ms clock as the output codec, so they suit a frame-synchronous duplex layout. AuT's 1–8 s attention windows (p.4) mean encoding is chunked rather than frame-causal, so verify the streaming delay.
8. **Quality stages.** Finish with a high-quality-only stage and DPO on ASR-WER-ranked pairs, mirroring Talker stages 2–3.
9. **No duplex recipe here.** This paper supplies only the streaming components.

## Open questions / unclear points

- Is the Talker's codec the released Qwen3-TTS-Tokenizer-12Hz, with interchangeable codes? Not stated. Test with an encode → Code2Wav round trip.
- The configs list semantic_codebook_size 4096 while the paper says 2048. The Talker vocab (3072, special IDs from 2148) implies codebook 0 uses ≤2148 IDs (derived).
- Not reported: Talker initialization and frozen parts; the unit behind "hundreds of millions"; how DPO pairs were built.
- Text says S2 is ~2T tokens, but the listed parts sum to 2.26T.
- p.11 credits VoiceBench 89.5 to 30B-A3B-Thinking; Tab.7 gives it to Flash-Thinking (30B-A3B-Thinking: 88.8).
- Tab.16 is "FLOPs-matched" even though the Omni model sees extra audio. What was cut is not stated.
- Latency is "theoretical" and the hardware is unnamed. The turn-detection mechanism is not described.
