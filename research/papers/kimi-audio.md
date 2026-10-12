# Kimi-Audio Technical Report

| Field | Value |
|---|---|
| Org | Kimi Team (Moonshot AI; github.com/MoonshotAI) (p.1) |
| Date / arXiv | 25 Apr 2025; arXiv:2504.18425v1 (p.1) |
| Category | half-duplex S2S + audio understanding ("universal audio foundation model") |
| Backbone | Qwen2.5-7B. The shared bottom layers and the text head are initialized from it; the audio head (transformer layers) is randomly initialized (p.5, 12). The layer split between shared layers and heads is not reported |
| Audio input | Sum of two embeddings at 12.5 Hz (p.3–4): (a) GLM-4-Voice supervised semantic tokens (VQ inside a Whisper encoder; 12.5 Hz, single codebook) and (b) Whisper-large-v3 continuous features (50 Hz), downsampled to 12.5 Hz by an adaptor |
| Audio output | GLM-4-Voice semantic tokens (12.5 Hz, 1 codebook) from a dedicated audio head that runs in parallel with the text head. Text and audio are length-matched with blank tokens; audio is delayed by 6 blank tokens (p.4–5, 10–11) |
| Waveform decoder | MoonCast-style detokenizer: tokens upsampled 4× → flow matching to 50 Hz mel → BigVGAN vocoder. Chunk-wise autoregressive streaming with a look-ahead of n=4 future tokens (p.5) |
| Duplex mechanism | n/a (half duplex; a server-side VAD commits the user turn, p.13) |
| Open weights / code | Code, checkpoints and evaluation toolkit (github.com/MoonshotAI/Kimi-Audio; Kimi-Audio-Evalkit) (p.1, 16) |
| Relevance to us | **Medium-high.** It gives a fully specified recipe for large-scale real-audio speech–text continued pretraining with explicit task weights and 50% text. However, an independent eval (in the MiMo-Audio report) shows its base model's knowledge does not reach speech output |

## TL;DR

- Qwen2.5-7B is continually pretrained on ~13M h of real audio (585B audio tokens) plus 585B text tokens, for one epoch (p.6, 10–12, Tab.3). There are seven tasks:
  - text-only
  - audio-only
  - ASR
  - TTS
  - three audio–text interleaving variants built from diarized segments of real audio
- I/O is hybrid (p.2–5):
  - Input: discrete semantic tokens plus continuous Whisper features.
  - Output: semantic tokens plus text, from parallel heads.
- SFT uses about 300K h with natural-language instructions, 2–4 epochs per source (p.9, 12). Dialogue audio comes from in-house Kimi-TTS (trained on ~1M h, then improved with RL) and Kimi-VC, which convert assistant turns to one voice actor.
- Results are strong on understanding and audio→text chat (p.16–19):
  - LibriSpeech WER 1.28 / 2.42.
  - MMAU sound 73.27.
  - VoiceBench average 76.93.
  - Human-rated speech conversation: 3.90/5 vs 4.06 for GPT-4o.
- Gaps: no numeric ablations, no text-retention numbers, no model-training compute, no speech-continuation or speech-output knowledge evals.

## Architecture

**Tokenizer** (p.3–4)
- Reuses GLM-4-Voice's supervised semantic tokenizer: a VQ layer inside the Whisper encoder, 12.5 Hz, single codebook.

**Continuous input path** (p.4, 12)
- Whisper-large-v3 features at 50 Hz pass through an adaptor that downsamples to 12.5 Hz.
- These features are added to the semantic-token embeddings. This path is input-only.
- The Whisper module is frozen for the first ~20% of pretraining tokens, then fine-tuned jointly.

**Audio LLM** (p.5, 10)
- "A significant portion of the original transformer bottom layers" is shared.
- On top sit two parallel heads made of transformer layers: a text head (from Qwen2.5) and an audio head (random init).
- When both modalities are output, text and audio are generated position-aligned and padded with blanks to equal length.

**Detokenizer** (p.5)
- Flow matching with a chunk-wise causal mask (example chunk size 1 s). Earlier chunks act as the prompt.
- To fix artifacts at chunk boundaries, each chunk is decoded with the next n=4 tokens appended, and only the current chunk's mel is kept. This look-ahead needs no training and delays only the first chunk.

**Serving** (p.13–14)
- Separate tokenizer, LLM and detokenizer services behind load balancers, connected over WebRTC.
- A scheduler stores conversation history as tokens.
- Turns are triggered by VAD.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **Tokenizer:** reused from GLM-4-Voice, not retrained (p.3).
- **Detokenizer**, trained in three steps (p.12):
  1. Pretrain flow matching and the vocoder on ~1M h of the pretraining audio.
  2. Chunk-wise fine-tuning with dynamic chunk sizes from 0.5 s to 3 s, on the same data.
  3. Fine-tune on studio recordings of the single "Kimi-Audio speaker."
- LR, steps and compute: not reported.

### Stage 1: Speech–text continued pretraining
- **Initialization:** Qwen2.5-7B, with the vocabulary extended by semantic audio tokens and special tokens (p.12).
- **Tasks and weights** (Tab.3, p.11). Notation: a = continuous+discrete audio input, a_d = semantic tokens, t = text; "/" marks two streams generated in parallel.

| Task | Sequence | Loss on | Weight |
|---|---|---|---|
| Text-only | t | t | 7 |
| Audio-only | a_d | a_d | 1 |
| ASR | a1, t1, a2, t2, … | t | 1 |
| TTS | t1, a_d1, t2, a_d2, … | a_d | 1 |
| Audio→semantic interleave | a1, a_d2, a3, a_d4, … | a_d | 1 |
| Audio→text interleave | a1, t2, a3, t4, … | t | 1 |
| Audio→semantic+text interleave | a1, a_d2/t2, a3, a_d4/t4, … | both | 2 |

  - Text-only is 50% of total task weight (derived: 7/14).
  - The third interleave variant delays audio by 6 blank tokens.
- **Tokens:** 585B audio + 585B text, one epoch (p.12). 585B audio tokens ≈ 13.0M h (derived: 585B ÷ 12.5 Hz ÷ 3600), consistent with seeing the 13M-h corpus about once.
- **Optimizer:** AdamW; LR 2e-5 → 2e-6 cosine; 1% warmup (p.12).
- **Not reported:** batch size, sequence length, steps, hardware.

### Stage 2: SFT
- **Volume:** about 300K h of SFT data (p.12).
- **Instruction format** (p.12):
  - Natural-language instructions instead of task tokens.
  - Each instruction exists in a text version and a Kimi-TTS audio version; one is chosen at random.
  - 200 LLM-written ASR instructions and 30 instructions for every other task.
- **Epochs:** 2–4 per source, "based on comprehensive ablation experiments" (results not shown; p.12, Tab.1–2).
- **Optimizer:** AdamW; LR 1e-5 → 1e-6 cosine; 10% warmup (p.12).
- **Not reported:** batch, steps, compute.

### RL
- No RL is applied to the audio LLM. RL is used only to improve Kimi-TTS, the data synthesizer (p.9).

## Data pipeline

**Pretraining corpus** (p.6, 11)
- About 13M h of raw audio (audiobooks, podcasts, interviews), including music, environmental sound and multilingual content.
- Text comes from MoonLight (p.11), though p.6 cites Kimi k1.5 for the same data.

**Processing pipeline** (p.6–7, Fig.3)
1. **Speech enhancement:** BSRNN at 48 kHz. Because enhancement removes environmental sound and music, original and enhanced audio are mixed 1:1 during pretraining (p.6).
2. **Diarization:** PyAnnote, then three post-processing rules:
   - Merge clusters whose speaker embeddings have cosine > 0.6.
   - Split segments into 1.5 s chunks and reassign a chunk when its cosine to the neighboring chunk is < 0.5.
   - Merge adjacent same-speaker segments until a segment exceeds 27 s or the silence gap exceeds 2 s.
3. **Transcription:** language ID with Whisper-large-v3; keep English and Mandarin only. English is transcribed by Whisper-large-v3. Mandarin is transcribed by Paraformer-Zh, with punctuation inserted from timestamp gaps (0.5–1.0 s → comma, >1.0 s → period) (p.7).

**Interleaving granularity:** interleaving alternates between diarized speaker segments of real audio (≤27 s each) and their ASR transcripts (p.10–11). It is neither TTS-synthesized nor word-level.

**Pipeline throughput:** 30 cloud instances with 240 NVIDIA L20 GPUs in total process about 200K h of raw audio per day (p.7).

**SFT data** (p.7–10)
- **Audio understanding:** open-source datasets (Tab.1) plus 55K h of in-house ASR and 5.2K h of in-house AAC/AQA. Tab.1 sums to about 301.5K h, of which about 291K h is ASR (derived: sum of Tab.1 rows).
- **Speech conversation** (volume not reported):
  - User queries: written by an LLM, then voiced by Kimi-TTS using more than 125K timbres.
  - Assistant responses: rendered in one voice actor's timbre through Kimi-TTS and Kimi-VC.
  - Kimi-TTS: zero-shot from a 3 s prompt; an LLM plus a flow-matching decoder; trained on ~1M h, then RL.
  - Kimi-VC: based on Seed-VC with timbre-shifting perturbation, fine-tuned on the actor's data.
  - The actor recorded more than 20 styles/emotions at 5 intensity levels in a studio (p.9).
- **Audio-to-text chat:** about 12.7M samples from 10 text SFT sets (derived: sum of Tab.2; Infinity-Instruct alone is 7M), each trained for 2 epochs. Before TTS of the user turns, the text is processed:
  - Math, code, tables and long content are filtered out.
  - The text is rewritten colloquially.
  - Complex single-turn items are split into multi-turn ones (p.9–10).
- **Duplex:** no duplex data.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Data processing | ~13M h raw | 240× L20 (30 instances); ~200K h/day | p.7 |
| Detokenizer | ~1M h | not reported | 3 sub-stages (p.12) |
| Continued pretraining | 585B audio + 585B text tokens (≈13M h, derived); 1 epoch | not reported | LR 2e-5 → 2e-6 |
| SFT | ~300K h; 2–4 epochs per source | not reported | LR 1e-5 → 1e-6 |
| Kimi-TTS (data synthesis) | ~1M h | not reported | plus RL |

## Preserving text intelligence / knowledge

**Mechanisms**
- The shared layers and text head are kept from the LLM. Audio generation gets its own randomly initialized head, so it does not overwrite the text head (p.5).
- Text-only data has task weight 7 of 14 (50%), and 585B text tokens are seen (p.11–12).
- Text is generated in parallel with audio, so spoken output is guided by the text head (p.2, 10).

**Measured retention**
- No text benchmark (MMLU etc.) is reported.
- An independent eval in the MiMo-Audio report (p.17, Tab.6) gives Kimi-Audio-Base SpeechMMLU T2T 70.7 and S2T 67.9, but S2S 11.8 and T2S 0.0. In the base model, knowledge reaches text output but not speech output, at least under few-shot prompting.

## Key results

**ASR WER** (Tab.4, p.16)
- LibriSpeech clean / other: 1.28 / 2.42.
- AISHELL-1: 0.60; AISHELL-2 ios: 2.56.
- WenetSpeech meeting / net: 6.28 / 5.37.
- Fleurs zh / en: 2.69 / 4.44.

**Audio understanding** (Tab.5, p.17)
- MMAU music / sound / speech: 61.68 / 73.27 / 60.66.
- MELD 59.13; VocalSound 94.85; Nonspeech7k 93.93; TUT2017 65.25; CochlScene 79.84 / 80.99.

**Audio-to-text chat** (Tab.6, p.18)
- OpenAudioBench: AlpacaEval 75.73, Llama Questions 79.33, Reasoning QA 58.02, TriviaQA 62.10, Web Questions 70.20.
- VoiceBench: AlpacaEval 4.46, CommonEval 3.97, SD-QA 63.12, MMSU 62.17, OpenBookQA 83.52, IFEval 61.10, AdvBench 100.00; average 76.93 (Qwen2.5-Omni 72.83; GLM-4-Voice 57.17).

**Speech conversation, human ratings 1–5** (Tab.7, p.19)

| Model | Speed | Accent | Emotion | Empathy | Style | Avg |
|---|---|---|---|---|---|---|
| Kimi-Audio | 4.30 | 3.45 | 4.27 | 3.39 | 4.09 | 3.90 |
| GPT-4o | | | | | | 4.06 |
| GLM-4-Voice | | | | | | 3.65 |

**Cross-paper numbers for the same model**
- MiMo-Audio report (p.21, Tab.8): Kimi-Audio-Instruct Big Bench Audio S2T 59.40 / S2S 51.00.
- Covo-Audio report (p.15, Tab.6): VCB-Bench general knowledge 54.47.

**Not evaluated:** S2S knowledge QA and speech continuation (StoryCloze, sWUGGY, sBLIMP).

## Ablations & lessons

The paper reports no numeric ablations. These are the engineering lessons it states.

- **Speech enhancement damages audio-understanding data.** It strips environmental sound and music, so original and enhanced audio are mixed 1:1 (p.6).
- **Parallel text+audio generation needs an audio delay.** Predicting the first semantic tokens at the same time as the first text tokens is hard. Prepending 6 blank tokens fixes this; 6 was chosen to trade quality against latency (p.11).
- **Naive chunked detokenization leaves boundary glitches.** Chunk-wise AR flow matching plus a 4-token look-ahead removes them (p.5).
- **Freeze, then unfreeze, the continuous encoder.** Whisper stays frozen for about 20% of pretraining tokens (p.12).
- **SFT epochs per source: 2–4**, chosen by internal ablations whose numbers are not shown (p.12).
- **Hours alone do not produce knowledge in speech output** (cross-paper evidence). After 13M h and 585B audio tokens of real-audio speech–text pretraining with semantic output tokens, the base model scores S2T 67.9 but T2S 0.0 and S2S 11.8 on SpeechMMLU (MiMo report p.17, Tab.6).
  - The cause cannot be identified from either paper. It is not a lack of generation tasks: 5 of the 7 non-text weight units put loss on audio tokens (audio-only, TTS, a→a_d, a→a_d/t; derived from Tab.3).
  - Candidate explanations, none tested:
    - Semantic tokens are produced by a separate, randomly initialized audio head.
    - Interleaving is only at segment level and only from real speech, which is less knowledge-dense than text corpora. GLM-4-Voice instead synthesizes interleaved data from text corpora.
    - A few-shot format mismatch under MiMo's evaluation protocol.
- **The authors' own outlook** (p.21):
  - Transcripts miss paralinguistics, so audio captions should be added.
  - Semantic tokens lose acoustic detail while acoustic tokens lack semantics, so a unified representation is needed.
  - Data generated by ASR and TTS caps the model at ASR/TTS quality, so training should move toward native audio data.

## Implications for our setup

- **Data scale.** Kimi used about 13M h of real audio plus an equal number of text tokens (585B each) in one epoch (p.6, 12). That is about 1,300× our 10K h (derived: 13M ÷ 10K). Our duplex-only 10K h covers essentially one of Kimi's seven tasks: audio-only continuation of conversational speech.
- **A concrete mixture we can copy.** Use 50% text-only, then audio-only, ASR, TTS and three interleaved variants at weights 1:1:1:1:1:2 (p.11).
  - Kimi's interleaving uses real speech with ASR transcripts at the diarized-segment level, which we can build from our own duplex corpus: transcribe both channels and create a→t, a→a_d and a→(a_d/t) alternations per turn.
- **A parallel or leading text stream.** Kimi pairs audio output with a parallel text head (p.5) and delays audio by 6 tokens (p.11). Both are cheap to add on a Qwen3 backbone. Kimi has no ablation showing this helps; the supporting evidence is GLM-4-Voice's S→T > S→S results.
- **Caveat: audio-generation tasks on real speech were not enough.**
  - Kimi already put most of its non-text weight on audio-token losses (TTS, audio-only, a→a_d, a→a_d/t), yet its base model was near zero on knowledge in speech output under MiMo's eval.
  - Real-speech interleaving and ASR/TTS pairs alone may not carry text-corpus knowledge into speech output.
  - Consider adding interleaved data synthesized from text corpora (as GLM-4-Voice does) and/or much more audio (as MiMo does).
  - Track T2S/S2S directly rather than assuming they follow from S2T.
- **Input side.** Adding continuous encoder features to discrete-token embeddings improves perception (p.4). If the Qwen3 tokenizer is discrete-only, a continuous input branch for the user channel in duplex is worth trying.
- **Streaming decoder.** If the Qwen3 decoder shows chunk-boundary artifacts in streaming, Kimi's training-free 4-token look-ahead is a cheap fix (p.5).

## Open questions / unclear points

- How many layers are shared and how large the audio head is; the paper only says "the first several layers" (p.5).
- Batch size, sequence length, steps and hardware for pretraining and SFT.
- Whether task weights are sampling ratios by tokens or by examples. Also, p.12 lists the weights as "1 : 7 : 1 : 1 : 1 : 1 : 2", which does not match the row order of Tab.3.
- The volume of speech-conversation SFT data, and whether "~300K h" includes it. Tab.1 alone already sums to about 301.5K h (derived).
- Any text-capability retention numbers (no MMLU or similar).
- The text-corpus citation is inconsistent: Kimi k1.5 [65] on p.6 vs MoonLight [44] on p.11.
