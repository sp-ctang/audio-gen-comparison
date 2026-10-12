# Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling (Kyutai DSM: DSM-ASR / DSM-TTS)

| Field | Value |
|---|---|
| Org | Kyutai (p.1) |
| Date / arXiv | arXiv:2509.08753v2, 29 Sep 2025, preprint (p.1) |
| Category | method (streaming seq2seq: ASR + TTS). Not a dialogue model, but it is the text–audio alignment recipe behind Moshi-style inner monologue and behind the Unmute cascade |
| Backbone | **DSM-ASR:** 2.6B (2048-d, 48 layers, 32 heads); a 300M variant also exists (p.7, p.19). **DSM-TTS:** 1B (2048-d, 16 layers); 1.8B with the depth transformer (p.7). **Public TTS:** 300M (p.23). No text-LLM initialization is reported |
| Audio input | Mimi: 24 kHz → 12.5 Hz, RVQ with up to 32 codebooks × 2,048 entries (p.4–5) |
| Audio output | (TTS) Mimi RVQ, 12.5 Hz, Q = 32 (p.23). A per-frame RQ "depth" transformer (1024-d, 4 layers per codebook, weights shared after codebook 8) generates the codebooks (p.7), with a 2-step acoustic delay (p.23) |
| Waveform decoder | Mimi decoder |
| Duplex mechanism | n/a. Text and audio are time-aligned streams with a constant delay between them; MAIN/OTHER turn tokens let the TTS render two-speaker dialogue (p.6) |
| Open weights / code | github.com/kyutai-labs/delayed-streams-modeling (p.1); huggingface.co/kyutai/tts-0.75b-en-public (p.23). The best TTS ships with precomputed speaker embeddings only (p.12) |
| Relevance to us | **High.** It is the recipe for word-level alignment of a text stream to audio frames with a delay (i.e., how to build and time an inner monologue), plus evidence that pseudo-labelling at scale works |

## TL;DR

- **Alignment happens in preprocessing.** Word timestamps place the text on the 12.5 Hz audio grid, and a decoder-only multistream Transformer models the streams jointly. Delaying the text gives ASR; delaying the audio gives TTS (p.3–6, Fig.1).
- **The delay sets the latency/quality trade-off.** One ASR model trained on random delays plus a delay embedding stays within ~300 ms of the requested delay and beats fixed-delay models (p.9, Fig.4).
- **The student beats the teacher.** ASR pretraining is hard distillation of whisper-timestamped on 2.5M h of EN/FR audio. Average WER goes 8.1 (teacher) → 7.1 → 6.4 after 28k h of ground-truth fine-tuning (p.7, p.9).
- **Strong results in both directions.** DSM-ASR averages 6.4 short-form WER, the only streaming model among the top OpenASR systems (Tab.1). DSM-TTS has the lowest long-form WER among open models (EN 1.72 / FR 2.96) and serves 100× real-time on one H100 (p.9–11).
- **TTS needs to see upcoming words.** Removing the 2-word lookahead stream raises English WER from 1.60% to 3.51% (Tab.15).

## Architecture

- **Components:** a causal Transformer backbone; summed per-stream input embeddings; a per-step sampler. The loss is computed only on output streams. The sampler "could also be a flow or diffusion model" for continuous outputs, which is the CALM direction (p.4).
- **Text stream:** a word starting at s seconds occupies frame i = floor(12.5·s) with a WORD token, followed by its subword tokens; all other frames are PAD. Predicting PAD/WORD learns word boundaries; timestamps are precise to 80 ms (p.5).
- **DSM-ASR:** text is delayed by τ. Optionally τ is drawn from U[0.25, 4] s per sequence, with a cosine embedding of τ added to the inputs (p.5, p.7).
- **DSM-TTS:** audio is delayed by τ. Four additions (p.6):
  - an **action stream** predicts when to feed the next word;
  - a **lookahead stream** carries the word 2 ahead;
  - **speaker conditioning** takes up to 5 speakers, each as a 10 s clip through a Mimi-encoder-initialized encoder, fed via cross-attention;
  - **classifier-free guidance** (CFG) on text and speaker, α = 2.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training

- Mimi is used as given; no codec training is described (p.4).
- Text tokenizers are trained on transcripts: 8k tokens FR/EN for TTS, 4k tokens EN for ASR (p.7).
- The speaker encoder has frozen conv layers and a Transformer trained end-to-end (p.6).
- A 300M prefix-LM normalizes transcript formatting; its training details are not reported (p.5).

### Stage 1: DSM-ASR pretraining

- **Data:** 2.5M h of public EN+FR audio, transcribed with word timestamps by whisper-timestamped (which wraps Whisper Medium) (p.7, p.9). No text-only data.
- **Segments:** random 90 s (1,125 frames, derived).
- **Optimization:** AdamW, cosine with warmup, peak LR 4e-4, weight decay 0.1; 1.6M steps on 48 H100s, with codebook dropout. Batch size and wall-clock time are not reported (p.7, p.9).

### Stage 2: DSM-ASR fine-tuning

- **Short-form:** 28k h of public ground-truth sets (LibriSpeech, VoxPopuli, GigaSpeech, AMI, SPGISpeech, TED-LIUM, Earnings22), CER-filtered; only 0.5% of examples exceed 30.5 s. 100k updates, batch 128, 16 H100s (p.7, p.19).
- **Long-form:** an 8:1:1 mix of concatenated LibriSpeech (≤6 min, ≈1k h), ≈22k h of 5-min synthetic dialogues from a preliminary DSM-TTS, and the short-form mix. 25k updates, batch 32, 16 H100s (p.7, p.19).

### Stage 3: DSM-TTS pretraining

- **Delay:** 1.28 s = 16 steps (p.6). The paper is inconsistent: p.7 says "1.6 s, or 16 steps".
- **Dropout:** 20% independent dropout of text and of speaker, for CFG (p.6).
- **Data:** 2.5M h, implied rather than stated (one pretraining paragraph covers both ASR and TTS, p.7).
- **Optimization:** 150 s extracts; AdamW, peak LR 2e-4, weight decay 0.1; 750k updates, batch 64, 32 H100s (p.7).
- **Audio seen:** ≈2.0M h (derived: 750k × 64 × 150 s), about 0.8 epoch.

### Stage 4: DSM-TTS CFG-distillation fine-tune

- An α input (1–4) bakes CFG into the model, so the open model works only with Kyutai's closed speaker embeddings. 2,400 updates at LR 1e-6 (p.7–8).

### Stage 5 (public-data variant)

- **Model:** 300M backbone, Q = 16 or 32, compact depth transformer, τ = 1.28 s. Voice cloning is by audio prefix: 20% of samples start with no text, 10% have an empty prefix, and text is dropped 20% of the time (p.23).
- **Data:** 88k h (AMI, Earnings22, GigaSpeech, SPGISpeech, TED-LIUM, VoxPopuli, LibriHeavy, Emilia; WER-filtered).
- **Optimization:** 500k updates, batch 64, 16 H100s (p.23).

## Data pipeline

- **Timestamps:**
  - Pseudo-labels come from whisper-timestamped (p.5).
  - Ground-truth transcripts get timestamps by DTW alignment to Whisper output; LibriSpeech uses Montreal Forced Aligner instead (p.5, p.19).
  - The prefix-LM harmonizes formatting.
- **Turns:** Pyannote diarization supplies the MAIN/OTHER turn tokens (p.6). Overlap, backchannels and interruptions are not discussed, and dialogue audio is one mixed stream.
- **Synthetic data:** DSM-TTS generated dialogues (≈22k h) were reused as ASR training data (p.19). The evaluation dialogue scripts were LLM-written (p.20).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| ASR pretrain | 2.5M h pseudo-labelled; 90 s; 1.6M steps | 48 H100 × n.r. | LR 4e-4 (p.7) |
| ASR short-form FT | 28k h; 100k × 128 | 16 H100 × n.r. | (p.7) |
| ASR long-form FT | ≈1k h + ≈22k h synthetic + short-form (8:1:1); 25k × 32 | 16 H100 × n.r. | (p.7, p.19) |
| TTS pretrain | ≈2.0M h seen, ≈9.0e10 frames (derived) | 32 H100 × n.r. | LR 2e-4 (p.7) |
| TTS CFG distillation | 2,400 updates | n.r. | LR 1e-6 (p.8) |
| Public TTS | 88k h; 500k × 64 | 16 H100 × n.r. | (p.23) |

n.r. = not reported.

## Preserving text intelligence / knowledge

- **Not addressed in the paper.** These are task models with 4k/8k text vocabularies; there is no text-LLM initialization and no text-only data (p.7).
- **Knowledge is kept by not touching the LLM.** DSM-ASR + DSM-TTS + Gemma 3 form a voice chat app with sub-second latency (unmute.sh) (p.11).
- **Later Kyutai use (outside the paper).** Kyutai used this "multi-stream conversational TTS" to synthesize knowledge-heavy training dialogues for MoshiRAG (https://kyutai.org/blog/2026-04-30-moshi-rag/).

## Key results

- **Short-form ASR** (Tab.1): DSM-ASR averages 6.4 WER (LS-clean 1.7, AMI 12.2). Streaming baselines: Parakeet-TDT-v2 7.0, Whisper large-v3 9.4. Best non-streaming: Canary-Qwen-2.5B 5.6.
- **Long-form ASR** (Tab.2): DSM-ASR 7.9 vs Parakeet (streaming) 7.8.
- **Timestamps** (Tab.12): F1 0.73, vs CrisperWhisper 0.85 and Whisper large-v3 0.22.
- **TTS WER** (Tab.3): long-form EN 1.72 (dialogues 1.15), FR 2.96. Turn-by-turn EN 2.06 vs Chatterbox 1.95.
- **Speaker similarity** (EN, turn-by-turn, Tab.4): 80.90, vs CSM 74.44 and ElevenLabs Multilingual v2 56.40.
- **Human quality** (MUSHRA, EN, Tab.5): 54.8, vs Chatterbox 67.7.
- **Speed:** TTS has 150 ms latency at batch 1 and 137.3× real-time throughput at batch 64 (Tab.6). ASR serves 400 real-time streams per H100 (p.9).
- **Public-data TTS:** LS test-clean WER 1.68 / SIM 0.71, vs F5-TTS 2.42 / 0.66 (Tab.7).

## Ablations & lessons

- **Longer delay, better quality.** ASR WER falls from ≈2.1% at ≈0.75 s to ≈1.6% at ≈2.4 s (read off Fig.4; the test set is not named) (p.9).
- **Train on random delays.** One delay-conditioned model beats every fixed-delay model and hits its target delay within ~300 ms (p.9).
- **Pseudo-labels are enough.** The student beats the teacher before any ground truth: average WER 8.1 → 7.1. The authors credit data diversity and low-temperature sampling (p.9).
- **Lookahead matters.** Without the lookahead stream, WER rises 1.60 → 3.51% (EN) and 3.02 → 3.25% (FR); speaker similarity is unchanged (Tab.15).
- **Learned timing beats fixed padding.** Fixed padding gives EN WER 2.32–2.86% and speaker similarity 0.691–0.715, vs 1.60% / 0.743 with the action stream. The authors blame unnatural prosody (Tab.15).
- **A slower frame rate beats more tokens.** Mimi (32 RVQ, 12.5 Hz, 400 tok/s) beats EnCodec (8 RVQ, 75 Hz, 600 tok/s): WER 1.68 vs 2.45%, SIM 0.71 vs 0.68 (Tab.14).
- **Codebook depth has limited and uneven value.**
  - ASR needs ≥24 of 32 quantizers and gains nothing above that (App.F).
  - For TTS, 32 beats 16 codebooks on LibriSpeech (WER 1.68 vs 1.95) but not on Seed-en (1.71 vs 1.58) (Tab.7–8).
  - Per step, the depth transformer costs 25 ms against the backbone's 5 ms (p.23).
- **Scale matters for ASR.** 300M averages 8.2 WER vs 6.4 at 2.6B (Tab.11).
- **Long context trades voice similarity for WER.** 150 s training enables long-form generation; long-form inference lowers WER (1.72 vs 2.06) but also lowers speaker similarity (76.46 vs 80.90) (Tab.3–4).

## Implications for our setup

Setup: Qwen3 LLM + Qwen3 tokenizer/vocoder, ~10K h duplex mid-training; weak continuation and semantics.

1. **Build the inner monologue DSM-style.**
   - Word-timestamp both channels (whisper-timestamped or forced alignment).
   - At the tokenizer's frame rate, put WORD + subwords at each word's onset frame and PAD elsewhere.
   - Automatic timestamps at 80 ms precision are adequate at scale.
2. **Tune how far the model's own text leads its audio** (TTS direction).
   - Quality rises with lead and lookahead (Fig.4, Tab.15). DSM-TTS uses ≥1.28 s plus a 2-word lookahead; dialogue models use ~2 frames (CALM/Moshi).
   - Sweep the lead, or train on random leads with a lead embedding and choose the lead at inference.
   - A longer lead adds response latency.
3. **Add a delayed user-transcript stream** (ASR direction) as an auxiliary target for listening comprehension. Its quality depends strongly on the delay.
4. **Scale up before the duplex stage.** DSM-TTS saw ≈2.0M h; even the 88k h public model matches F5-TTS/CosyVoice3 WER. Our 10K h is ~1/250 of DSM's corpus (derived). Mid-training likely needs large pseudo-labelled, word-aligned monologue and conversation speech before the duplex data.
5. **Generate knowledge-rich dialogues.** A DSM-style dialogue TTS can voice LLM-written scripts, as Kyutai did for 22k h of ASR data and later for MoshiRAG. Caveat: it outputs one mixed stream, while duplex training needs per-speaker channels.
6. **Spend fewer output codebooks.** Many-codebook depth decoding dominates step time; consider fewer codebooks or a continuous head (see the CALM note).
7. **Use a cascade as the knowledge ceiling.** Unmute is the reference to measure our spoken QA against (p.11).

## Open questions / unclear points

- **Inconsistent delays:** 1.28 s / 16 steps (p.6, p.23), "1.6 s, or 16 steps" (p.7), and 2 s / 25 steps in the latency calculation (p.22).
- **Not reported:** ASR pretraining batch size, wall-clock times, whether the English-only ASR used only the English part of the 2.5M h, and any LLM initialization.
- **Claims vs. tables:**
  - "Lowest WER across all languages" (p.10) is contradicted on the small subset: ElevenLabs Multilingual v2 scores lower in EN (2.01 vs 2.35), and Chatterbox beats turn-by-turn DSM (1.95 vs 2.06) (Tab.3).
  - The 300M ASR is called "considerably better" than Whisper Medium, yet scores 8.2 vs 8.1 (Tab.11).
- **Gaps:** overlapping speech is not handled or discussed, and delay conditioning is tested only for ASR.
