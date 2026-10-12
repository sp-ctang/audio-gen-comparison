# Moshi: a speech-text foundation model for real-time dialogue

| Field | Value |
|---|---|
| Org | Kyutai, Paris (p.1) |
| Date / arXiv | arXiv:2410.00037v2, 2 Oct 2024 (p.1) |
| Category | full-duplex S2S (the paper also contributes a text LLM, Helium, and a codec, Mimi) |
| Backbone | 7B "Temporal Transformer" (32 layers, d=4096, MLP 11,264, 32 heads; Helium context 4,096 tokens, Moshi context 3,000 frames ≈ 4 min) initialized from **Helium**, Kyutai's own 7B text LLM trained from scratch on 2.1T English tokens; plus a randomly initialized 6-layer "Depth Transformer" (d=1024, MLP 4,096, 16 heads) (p.3, p.8 Tab.1, p.20) |
| Audio input | Mimi codec (causal, streaming): 24 kHz mono → 12.5 Hz frames, 8 codebooks × 2,048 entries, 1.1 kbps. The user stream and Moshi's own stream are tokenized separately (p.10–11, p.15) |
| Audio output | Mimi tokens at 12.5 Hz: codebook 1 is a semantic VQ distilled from WavLM, codebooks 2–8 are an acoustic RVQ. For each frame, the Depth Transformer autoregressively predicts **text token → semantic → 7 acoustic** tokens, with separate weights per codebook. Acoustic codebooks lag 2 frames in pretraining and 1 frame afterwards (p.8 Tab.1, p.14, p.17) |
| Waveform decoder | Mimi decoder: causal SEANet transposed convolutions plus a causal Transformer, streaming in 80 ms frames (p.10–11) |
| Duplex mechanism | Two parallel token streams (Moshi and user) in one RQ-Transformer, with 17 tokens per 80 ms step (1 text + 8 Moshi audio + 8 user audio). There is no turn segmentation. When Moshi is silent it outputs "natural silence" audio tokens and PAD text tokens (p.17, p.30) |
| Open weights / code | Mimi and Moshi are released, with code at github.com/kyutai-labs/moshi (p.1, p.41–42). The paper does not say whether Helium is released |
| Relevance to us | **High.** It is the canonical open full-duplex recipe and has the same shape as ours (text LLM → audio pretraining → multi-stream post-training → duplex FT → instruct FT). It includes ablations on exactly our two problems: continuation quality and knowledge retention |

## TL;DR

- Moshi learns speech from a **single-stream audio pretraining stage on 7M hours** (1M steps, 16 h of audio per batch, 5-min sequences, 50% of steps on text-only batches), and this happens *before* it sees any duplex data. Duplex ability is added afterwards in three stages: diarization-simulated two-stream data (100k steps), 2,000 h of real two-channel Fisher (10k steps), and >20k h of synthetic two-channel instruct dialogues (30k steps) (p.8 Tab.1; p.18–21).
- **Inner Monologue** is the most important design choice. Moshi predicts a time-aligned text stream for its own speech, one text token per frame, before that frame's semantic and acoustic tokens. On continuations, it cuts transcript NLL from 3.65 to 2.77 and raises transcript length from 602 to 1,920 characters (p.25 Tab.6). It roughly triples spoken-QA accuracy (p.29 Tab.8).
- **Keeping 50% text-only batches** during audio pretraining preserves knowledge: Audio TriviaQA is 22.8 with them and 18.3 without (p.29 Tab.8). MMLU still falls from 54.3 (Helium) to 49.8 after audio pretraining and 49.7 after instruct tuning (p.22, p.27).
- **Mimi** is a causal 12.5 Hz, 1.1 kbps codec. WavLM is distilled into a separate ("split") semantic VQ, and the codec is trained with adversarial losses only. It scores MUSHRA 81.0, against 58.8 with the standard loss mix (p.24 Tab.4).
- The **RQ-Transformer** (large temporal model plus small depth model) uses a 100× loss weight on the semantic token, separate depth weights per codebook, and a 1–2 frame acoustic delay. Latency is 160 ms in theory and 200 ms in practice (p.1, p.25–27).
- **Compute is not reported** beyond "H100 GPUs, using FSDP and activation checkpointing" (p.20).

## Architecture

**Helium (p.7–8).** A decoder-only Transformer with RMSNorm, RoPE, a 4,096-token context, FlashAttention and SiLU-gated GLU feed-forward blocks. The tokenizer is SentencePiece unigram with 32k entries (mostly English), single-digit number splitting and byte fallback (p.7). Size: 7B, 32 layers, d=4096 (p.8 Tab.1).

**Mimi (p.9–12).**
- SEANet convolutional encoder with strides (4,5,6,8) plus a final stride-2 convolution, giving a 12.5 Hz latent with D=512 (p.10).
- Causal Transformers sit before and after quantization: 8 layers, 8 heads, d=512, MLP 2,048, RoPE, LayerScale initialized at 0.01 (p.10).
- The latent is projected from 512 to 256 dims before quantization. There are 8 quantizers × 2,048 entries, i.e. 1.1 kbps (p.11).
- **Split RVQ:** a plain VQ carries the WavLM-distilled semantic level, and a separate 7-level RVQ runs in parallel. Their outputs are summed (p.12).
- Fully causal, with an 80 ms frame and stride (p.11).

**RQ-Transformer (p.12–14).**
- The Temporal Transformer runs once per 80 ms frame. Its input is the sum of K learned embeddings of the previous frame's tokens, and it outputs a context vector z_s.
- The first sub-token (text) is read from z_s by a linear head. The Depth Transformer then predicts the remaining sub-tokens autoregressively, using **different weights for each codebook index** (p.14).
- Flattening the codebooks instead would cost 100 tokens/s, i.e. 30k steps for 5 minutes of audio (p.13).

**Acoustic delay (p.14).** At frame s the model emits the semantic token for frame s and the acoustic tokens for frame s−τ. τ=2 in pretraining and τ=1 thereafter (p.8 Tab.1). Latency is the 80 ms codec frame plus τ × 80 ms, giving 160 ms (p.26–27).

**Multi-stream (p.15–17).**
- Each frame holds V = [text_Moshi, sem_Moshi, ac_Moshi×7, sem_user, ac_user×7], so K = 2Q+1 = 17 (p.17).
- At inference, Moshi samples its text token and its 8 audio tokens; the user's tokens come from the microphone.
- The model also predicts the user stream, which lets it generate whole simulated dialogues for offline evaluation (p.17).

**Inner Monologue (p.15–17).**
- The text stream is the SentencePiece tokenization of a Whisper transcript of Moshi's own audio. Each word's tokens start at that word's frame (from Whisper word timestamps), PAD fills the gaps, and EPAD is inserted just before the next word. EPAD splits "the decision of ending a word, and which one should follow, into two steps" (p.16).
- About **65% of text-stream tokens are padding** in English conversation (p.16), because speech carries about 3–4 text tokens/s against 12.5 frames/s (p.13).
- The user's text is not modeled (p.16).
- Shifting text relative to audio turns the same model into streaming ASR (audio leads) or streaming TTS (text leads) (p.16; App. C, p.56–58).
- Forcing an EPAD makes Moshi start talking immediately (p.17).

**Loss (p.21, Eq.7).** Per frame, the loss is CE(text) plus the weighted mean of the audio CEs, so the text token counts as much as all audio tokens combined. The weights are α=100 for semantic tokens and α=1 for acoustic tokens.

## Training pipeline

Every Moshi stage uses AdamW (weight decay 0.1, β1 0.9, β2 0.95) on H100s with FSDP and activation checkpointing (p.20). Both transformers are trained in every stage, each with its own LR (p.8 Tab.1). Mimi is trained separately; no joint training is described. Warmup lengths are not reported.

### Stage 0: tokenizer / codec / decoder training (Mimi)
- **Objective:** adversarial losses only (feature-matching plus multi-scale STFT discriminator; reconstruction losses are dropped), plus a cosine-distance distillation loss (p.11–12). The distillation maps the semantic VQ output, linearly projected to 1,024 dims, onto WavLM-large embeddings that are average-pooled **non-causally** (kernel 8, stride 4) down to 12.5 Hz.
- **Regularization:** quantizer dropout, and quantization is applied only 50% of the time per sequence (p.11).
- **Data:** not reported. Training uses random 12 s windows, with the Transformers' context capped at 10 s (p.11).
- **Optimizer:** AdamW, LR 8e-4, β1 0.5, β2 0.9, weight decay 5e-2 on the Transformer parameters only, EMA 0.99; batch 128; 4M steps (p.11).
- **Audio processed:** about 1.7M h (derived: 4M × 128 × 12 s).
- **Compute:** not reported.

### Stage 1: Helium text pretraining
- 2.1T tokens (p.3): 500k steps × 4.2M-token batches (derived product = 2.1T).
- Context 4,096; cosine LR with peak 3e-4 and linear warmup (p.8, p.20).
- Compute: not reported.

### Stage 2: Moshi audio pretraining (single stream)
- **Init:** the Temporal Transformer starts from Helium; the Depth Transformer is random (p.20).
- **Audio data:** the 7M-hour "unsupervised audio dataset", mostly English, transcribed with Whisper large-v3 and resampled to 24 kHz mono (p.18). All speakers are mixed into one audio stream, and the text stream holds the words of all speakers.
- **Text share:** half the steps are text-only batches of 1.2M tokens drawn from the Helium dataset, with a **separate optimizer state** (p.8 caption, p.20).
- **Batch:** 16 h of audio made of 5-min items. That is about 192 items and 720k frames per batch (derived).
- **Schedule:** 1M steps, cosine LR with linear warmup. LR is 3e-5 for the Temporal Transformer (10× below Helium's peak) and 2e-4 for the Depth Transformer (p.8, p.20).
- **Tricks (p.20):**
  - text tokens are masked with probability 30%;
  - the text-to-audio delay is randomized in [−0.6, +0.6] s;
  - LR is multiplied by 0.75 on the text embedding and text output layer for audio batches;
  - PAD tokens get half weight in the loss;
  - acoustic delay τ=2.
- **Totals (derived, assuming the 1M steps include the text-only steps):** about 500k audio steps × 16 h ≈ 8M h, i.e. about 1.1 passes over the 7M h. Text: about 500k × 1.2M ≈ 0.6T tokens.

### Stage 3: Post-training with simulated multi-stream (diarization)
- **Data:** the same 7M h, diarized with PyAnnote (p.20).
  - A random speaker becomes the main speaker ("Moshi"). Its activity mask splits each waveform into the main speaker and a residual that may contain several speakers. The two waveforms are encoded separately and used as the two streams.
  - The text stream holds only the main speaker's words, with text delay 0.
- **Schedule:** 100k steps, 8 h batches, fixed LR 3e-6 (temporal) and 5e-5 (depth); 10% text-only batches; τ=1 (p.8, p.20).
- **Totals (derived):** about 0.72M h of audio and about 12B text tokens.
- **Stated limitation:** this data has no overlap, and the inactive stream is perfectly silent (p.20–21).

### Stage 4: Fisher fine-tuning (real full-duplex)
- **Data:** Fisher, 2,000 h of telephone conversations with each side on its own channel. The 8 kHz audio is upsampled to 24 kHz with AudioSR (p.18). A random side becomes Moshi. Text covers only the main speaker and is timestamped with whisper-timestamped (Whisper medium) (p.18).
- **Schedule:** 10k steps, 40-min batches, LR 2e-6 (temporal) and 4e-6 (depth). No text batches (p.21).
- **Totals (derived):** about 6.7k h processed, i.e. about 3.3 epochs.

### Stage 5: Instruction fine-tuning (synthetic conversations)
- **Data:** more than 20k h of synthetic two-channel speech (p.18; see Data pipeline).
- **Augmentation of the user stream (p.21):**
  - random gain of −24 to +15 dB, 50% of the time;
  - DNS-challenge noise at −30 to +6 dB relative level, 30% of the time, interleaved with silent stretches of up to 30 s with probability 50%;
  - echo (Moshi's stream scaled by U[0, 0.2] and delayed U[100, 500] ms) plus reverb, applied together 30% of the time.
- **Schedule:** 30k steps, 2.7 h batches, LR 2e-6 for both transformers (p.21).
- **Totals (derived):** about 81k h processed, at most about 4 epochs.

### Side model: streaming multi-stream TTS (instruct-data generator)
- Shares Stage 2 with Moshi. Its post-training puts audio 2 s behind text (p.21).
- It is fine-tuned on a 170 h "supervised multi-stream dataset" of natural and scripted two-channel conversations. Both speakers' text goes in one stream, separated by <bos>/<eos> (p.21, p.56).
- LibriSpeech test-clean WER 4.7% (p.31).
- Moshi itself never trains on the 170 h (p.21).

## Data pipeline

- **Text (p.9, p.17):**
  - Mix: 12.5% curated sources (Wikipedia from five dumps, 2017–2022; Wikibooks; Wikisource; Wikinews; StackExchange; peS2o) and 87.5% CommonCrawl (10 crawls, 2018-30 to 2023-40).
  - Deduplication: line-level with FNV-1a hashing and a Bloom filter, then fastText fuzzy dedup that removes blocks of ≥3 duplicate lines.
  - Language ID: fastText, keeping documents scoring ≥0.85.
  - Quality: a fastText classifier with 9 classes (high-quality sources and domains), applied per line and averaged with length weighting.
- **Audio (p.18, p.54–56):**
  - 7M h with Whisper large-v3 transcripts. Source composition is not reported.
  - Audio-fingerprint deduplication: constellation-map hashes; segments that match ≥10 times form a duplicate signature set, which is used to filter segments at training time.
  - In the regurgitation study, deduplication drops prompted regurgitation of the most frequent segment from 98.4% to 0% (p.37 Tab.13).
- **Duplex data:**
  - diarization-simulated streams (p.20);
  - Fisher (p.18);
  - the 170 h two-channel set, used only for the TTS and to fine-tune Helium on real conversation transcripts (p.18).
- **Synthetic instruct data (p.18–20):**
  - Text instruct data such as OpenHermes "proved to be ill-suited": URLs could not be rendered by TTS, and bullet points and long enumerations do not follow an oral flow (p.18).
  - Instead, Helium fine-tuned on OpenHermes plus real conversation transcripts writes the scripts. Script types:
    - **Knowledge conversations:** a Wikipedia or StackExchange paragraph → a 2-sentence summary → a transcript prompted with "Moshi is knowledgeable about the topic. Use some backchanneling. Use short turns." (p.19).
    - **Identity:** conversations about Moshi and Kyutai.
    - **Voice-style instructions:** single-turn sentences, monologues or poems in a random style about a random entity; role-play prompted as "a {voice} {character} who…", with "a lot of backchanneling" (p.19).
    - **Clarifications:** misspelled user questions, after which Moshi asks the user to repeat or clarify.
    - **False premises:** Moshi answers "No" and corrects the user.
    - **Basic math, grammar and trivia QA:** added because Moshi initially failed at things like adding numbers (p.20).
    - **Safety refusals** (p.20).
  - Rendering: the multi-stream TTS voices Moshi's side, conditioned on one voice actor (">70" speaking styles on p.18; Table 19 lists 92). The user voice is randomized per example (p.18, p.67).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Mimi | 4M steps × 128 × 12 s ≈ 1.7M h processed (derived); dataset not reported | not reported | adversarial-only + WavLM distillation |
| Helium | 2.1T text tokens (500k × 4.2M) | H100, FSDP; GPU count and time not reported | — |
| Audio pretrain | 1M steps; ≈8M h audio + ≈0.6T text tokens (derived) | not reported | 7M-h corpus, 50% text steps |
| Post-train (diarized) | 100k × 8 h; ≈0.72M h + ≈12B text tokens (derived) | not reported | 10% text steps |
| Fisher FT | 10k × 40 min ≈ 6.7k h (derived) | not reported | 2,000 h real two-channel |
| Instruct FT | 30k × 2.7 h ≈ 81k h (derived) | not reported | >20k h synthetic |
| TTS | shares the audio pretrain; later steps not reported | not reported | 170 h FT |

For inference, practical latency is 200 ms; the hardware is not stated (p.1). The demo serves 8-bit weights (p.33).

## Preserving text intelligence / knowledge

**Mechanisms:**
1. Start from a strong text LLM.
2. Run 50% text-only batches in audio pretraining and 10% in post-training, with a separate optimizer state (p.20).
3. Keep the backbone LR low: 3e-5, then 3e-6, then 2e-6, against 3e-4 for Helium (p.8).
4. Use Inner Monologue, so that the LM "speaks" through its own text vocabulary.
5. Fine-tune on oral-style, knowledge-grounded synthetic dialogues, plus explicit math and trivia QA (p.18–20).

**Evidence:**
- **MMLU:** 54.3 → 49.8 after single-stream pretraining → 49.7 after multi-stream instruct; 48.7 when Moshi's voice is synthetic (p.27 Tab.7). For comparison, Spirit-LM scores 36.9 (p.27).
- **Spoken QA (Web Q / LlaMA Q / Audio TriviaQA):** Moshi 26.6 / 62.3 / 22.8, against Helium on the text versions 32.3 / 75.0 / 56.4 (p.29 Tab.8).
- The TriviaQA gap is attributed to multi-sentence and syntactically unusual questions that the oral-style fine-tuning data does not cover (p.30).

## Key results

| Benchmark | Result |
|---|---|
| Helium text (p.22 Tab.2) | MMLU 54.3; ARC-e/c 79.6/55.9; HellaSwag 76.3; TriviaQA 59.9/72.6; NQ 23.3 (Llama 2 MMLU 45.3) |
| Mimi (p.24 Tab.4) | 1.1 kbps, 12.5 Hz, causal: ABX 8.7%, MUSHRA 81.0. Comparisons: SpeechTokenizer 4 kbps 74.3; SemantiCodec 1.3 kbps 64.8; RVQGAN 1.5 kbps 31.3; ground truth 90.6 |
| Audio LM: sWUGGY / sBLIMP / sTopic-StoryCloze (= tStoryCloze) / sStoryCloze (p.27 Tab.7) | Single-stream pretrain 72.6 / 58.8 / 83.0 / 60.8. After multi-stream instruct 63.0 / 55.2 / 83.6 / 62.7. Audio-only, warm start 74.3 / 58.9 / 81.8 / 58.7. Audio-only, cold start 74.8 / 59.9 / 80.9 / 56.9. Spirit-LM 69.0 / 58.3 / 82.9 / 61.0 |
| Spoken QA: Web Q / LlaMA Q / Audio TriviaQA (p.29 Tab.8) | Moshi 26.6 / 62.3 / 22.8. Without IM 9.2 / 21.0 / 7.3. SpeechGPT 6.5 / 21.6 / 14.8. Spectron 6.1 / 22.9 / – |
| Generated dialogues from 10-s Fisher prompts: DialoGPT PPL; IPU / pause / gap / overlap in seconds (p.30 Tab.9) | temp 0.8: 41.9; 35.1 / 13.2 / 12.5 / 1.2. temp 1.0: 79.3; 50.8 / 7.0 / 4.5 / 4.1. Ground truth: 59.6; 51.1 / 6.4 / 4.2 / 3.3. dGSLM: 195.9; 41.4 / 13.8 / 10.7 / 6.1 |
| Streaming TTS / ASR with 2 s delay, LibriSpeech test-clean WER (p.31–32) | TTS 4.7% (Vall-E 5.9%, NaturalSpeech 3 1.81%). ASR 5.7% (streaming FastConformer 3.6%) |
| Voice consistency (p.38) | Over 100 h of generated dialogue, 98.7% of Moshi's segments are closer to Moshi's reference voice than to the user's. No drift between 20 and 45 s |
| ALERT safety (p.36) | 83.05 overall |

## Ablations & lessons

**(a) Continuation quality**
- **Inner Monologue gives the biggest gain in continuation quality.** In the setting with delay [0,2,…,2], semantic weight 100 and per-codebook depth weights, adding IM cuts transcript NLL from 3.65 to 2.77 and raises length from 602 to 1,920 characters on continuations of 3 s prompts (p.25 Tab.6). The authors call it "the most drastic improvement" (p.26). Length matters because "weak models typically collapse to silence" (p.26).
- **Up-weighting the semantic token and using per-codebook depth weights both help.**
  - With per-codebook weights, raising the semantic weight from 1 to 100 moves NLL from 4.09 to 3.65 and length from 519 to 602.
  - At weight 100, switching from shared to per-codebook depth weights moves NLL from 3.75 to 3.65 and length from 538 to 602 (p.25 Tab.6).
  - The motivation was that per-level losses "were conflicting with one another" (p.26).
- **A small acoustic delay is a cheap stability gain.** Delays of 0, 1 and 2 frames give NLL 4.36 / 4.12 / 4.09 and length 486 / 529 / 519 (p.25 Tab.6). The final model pretrains at τ=2 and fine-tunes at τ=1, for 160 ms latency (p.27).
- **The RQ-Transformer is required under tight latency.**
  - With delay pattern [0,2,…,2], perplexity is 135.4 with independent heads and 36.8 with the depth transformer.
  - With the 640 ms staircase pattern [0..7], it is 42.2 versus 40.3 (p.25 Tab.5; p.26).
- **The tokenizer's semantic content governs how learnable it is.**
  - Without WavLM distillation, ABX is 23.3%: the tokens are "not amenable to capturing linguistic content" (p.23).
  - Distillation lowers ABX to 6.5% but drops MUSHRA from 65.9 to 57.8.
  - The split semantic VQ gives ABX 8.1% and MUSHRA 64.0 (p.24 Tab.3).
  - Non-causal pooling of the distillation targets was "critical" (p.12). Removing the encoder Transformer raises ABX to 10.8% (p.24 Tab.3).
- **Codec perceptual quality depends on adversarial-only training and the decoder Transformer.**
  - Adversarial-only training raises MUSHRA from 58.8 to 81.0 even though VisQOL falls from 2.82 to 1.84 (p.24 Tab.4).
  - Removing the decoder Transformer drops MUSHRA from 64.0 to 48.4 (p.24 Tab.3).
  - VisQOL and MOSNet correlate poorly with MUSHRA (p.24–25).
- **Textless metrics can mislead for dialogue models.** After instruct tuning, sWUGGY falls from 72.6 to 63.0 while sStoryCloze rises from 60.8 to 62.7. The authors saw no lexical degradation in practice. They attribute the drop to scoring on the augmented, noisy user stream and prefer spoken QA (p.28–29).

**(b) Knowledge transfer**
- **Inner Monologue also gives the biggest gain in spoken knowledge.** Without and with IM: Web Q 9.2 → 26.6; LlaMA Q 21.0 → 62.3; Audio TriviaQA 7.3 → 22.8 (p.29 Tab.8). The cost is one extra token per frame (17 instead of 16) (p.30).
- **Text replay matters.** Dropping the 50% text-only batches in pretraining lowers spoken QA from 26.6 to 23.2 (Web Q), 62.3 to 61.3 (LlaMA Q) and 22.8 to 18.3 (Audio TriviaQA) (p.29 Tab.8). The authors conclude that "a warm start from Helium is not sufficient" (p.30). Only 0% versus 50% was tested.
- **Most knowledge loss happens during audio pretraining.** MMLU goes 54.3 → 49.8 → 49.7 (p.22, p.27). Later stages barely move it, even with 10% or 0% text batches.
- **Text-LM initialization alone helps little; joint text-audio training adds semantic coherence.**
  - Audio-only, cold start versus warm start: sStoryCloze 56.9 → 58.7, sWUGGY 74.8 → 74.3.
  - With joint text-audio pretraining, sStoryCloze reaches 60.8, but sWUGGY is 72.6 (p.27 Tab.7).
- **The style of fine-tuning data caps how knowledge comes out.** The TriviaQA gap (22.8 for Moshi against 56.4 for Helium on text) is traced to oral-style fine-tuning (p.30). Simple arithmetic needed explicit synthetic QA data (p.20).
- **Language ability is the fragile part under quantization.** At W4A8 with block 32, Moshi's MMLU falls from 49.7 to 42.2, while Helium's only falls from 54.3 to 52.97 (p.33 Tab.10–11). Audio quality holds up down to 4 bits (p.34).

**(c) Full-duplex behavior**
- **Simulated duplex data is necessary but not sufficient.** Diarized streams have no overlap and a silent inactive channel, which is why the real-overlap Fisher stage follows (p.20–21). Neither stage is ablated numerically.
- **Sampling temperature trades content against conversational dynamics.**
  - At temperature 0.8, DialoGPT perplexity is 41.9, but the gap is 12.5 s and the overlap 1.2 s.
  - At 1.0, perplexity is 79.3, with gap 4.5 s and overlap 4.1 s, which is closest to the ground truth (59.6 / 4.2 / 3.3) (p.30 Tab.9).
- **A fixed assistant voice in the instruct data is enough to control the output voice.** 98.7% of segments match Moshi's reference (10,249 vs 133) (p.38). Using a synthetic Moshi voice instead of the actor costs a little: MMLU 49.7 → 48.7 (p.27 Tab.7).

**(d) Data**
- **Deduplicating pretraining audio removes regurgitation.** When prompted, the rate of regurgitating the most frequent segment falls from 98.4% to 0% at temperature 0.8 (p.37 Tab.13).

## Implications for our setup

1. **Our 10K h of duplex data should not be the main speech-learning stage.**
   - Moshi learns continuation from 7M h of single-stream audio, about 8M h processed. That is roughly 700× our corpus (derived: 7M/10K).
   - It adds duplex behavior afterwards with only 2,000 h of real two-channel audio plus synthetic dialogues. Our corpus is already 5× Moshi's real duplex data.
   - Weak continuation is the expected result of skipping a large single-stream speech-text stage. We should add one before the duplex stage.
   - Moshi's diarization trick turns any single-channel corpus into pseudo-duplex training data (p.20).
2. **Add an Inner Monologue text stream if we lack one.** This is the strongest evidence-backed fix for both of our problems: about 3× spoken QA and 3.2× continuation length.
   - It requires word timestamps for all 10K h (Moshi used whisper-timestamped).
   - Use PAD/EPAD at the codec frame rate, place the text token first in each frame, set PAD loss weight to 0.5, mask text 30% of the time and use a random ±0.6 s text-audio delay during pretraining (p.16, p.20).
   - If we already have a text stream, check our alignment and padding scheme against Moshi's.
3. **Use text replay and low LRs.**
   - About 50% text-only steps during speech-text continued pretraining and about 10% in duplex post-training, with a separate optimizer state.
   - Backbone LR about 10× below its pretraining peak; new audio modules at a higher LR (2e-4 in Moshi).
   - Track MMLU of the backbone after every stage.
   - Caveat: Moshi replayed Helium's own pretraining data. We cannot replay Qwen3's, and Moshi gives no evidence about substitute text mixtures.
4. **Check the tokenizer's structure.**
   - If the Qwen3 tokenizer has multiple codebooks: up-weight the semantic or first codebook (×100), use a depth decoder with per-codebook weights, and add a 1–2 frame acoustic delay.
   - If it is flattened or interleaved, two streams multiply the backbone's sequence length; Moshi's per-frame RQ design avoids that (p.13).
   - Measure the tokenizer's phonetic ABX. Moshi uses ABX as its proxy because it "has been shown to be a strong predictor" of downstream LM coherence (citing Lakhotia et al.) (p.23). Without semantic distillation, Mimi's ABX was 23.3%; with it, 8.1%.
5. **Get knowledge from the instruct data, not from natural duplex audio.**
   - Generate oral-style, knowledge-grounded dialogues (from Wikipedia and StackExchange contexts) with backchannels and short turns, false-premise corrections, clarification requests, math and trivia QA, and refusals.
   - Synthesize them in two channels with a fixed assistant voice and random user voices, and augment the user channel.
   - Moshi used more than 20k h of this.
6. **Adopt Moshi's diagnostics** for regression tracking:
   - transcript NLL and length of continuations from 3 s prompts (catches collapse to silence);
   - spoken Web Q / LlaMA Q / TriviaQA against the text backbone;
   - turn-taking statistics from Table 9, with a temperature sweep from 0.8 to 1.0.
7. **Duplex timing will probably need a later RL stage.** A Kyutai follow-up finds that even full-recipe Moshi mis-times turns: Full-Duplex-Bench pause TOR 0.445 and interruption latency 1.377 s (see `multifaceted-interactivity-alignment.md`).

## Open questions / unclear points
- **Compute:** GPU count, wall-clock and total GPU-hours are not reported for any stage, and the hardware behind the 200 ms practical latency is not stated.
- **Mimi:** its training data is not reported. Its Transformer context is given as "250 frames (20 seconds)" on p.10 but "10s (250 frames before the last downsampling layer)" on p.11.
- **Step accounting:** it is unclear whether the "1 million training steps" include the text-only steps. The derived totals assume they do.
- **Sequence length:** Table 1 gives a 3,000-step (4 min) context, while §4.4 says 5-min batch items. How the two are reconciled is not stated.
- **Corpus details:** the composition of the 7M h (sources, speech share, languages) and how many steps Helium was fine-tuned for script generation are not reported.
- **Loss on the user stream:** whether the multi-stream loss includes user-stream tokens is implied (the user stream is predicted, p.17) but not stated.
- **Missing ablations:** no ablation isolates the diarization post-training or the Fisher stage, and no text ratio other than 0% and 50% is tested.
- **Minor inconsistencies:** Mimi VisQOL values differ slightly between Table 3 and Table 17 for identical configurations (e.g. 2.82 vs 2.72). Speaking styles are given as ">70" on p.18 but Table 19 lists 92.
