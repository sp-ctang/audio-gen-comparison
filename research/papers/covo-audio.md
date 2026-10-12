# Covo-Audio Technical Report (Covo-Audio / Covo-Audio-Chat / Covo-Audio-Chat-FD)

| Field | Value |
|---|---|
| Org | Tencent (Tencent AI Lab) (p.1) |
| Date / arXiv | arXiv:2602.09823v2, 16 Mar 2026 (v2) (p.1) |
| Category | half-duplex S2S (Covo-Audio-Chat) plus a full-duplex S2S variant (Covo-Audio-Chat-FD) |
| Backbone | Qwen2.5-7B-Base, with its text vocabulary extended by discrete audio tokens (p.5) |
| Audio input | Whisper-large-v3 encoder (50 Hz). An adapter of three downsampling modules (each: 2 linear layers + 1 conv, stride 2) reduces it to 6.25 Hz continuous features (p.4). FD mode uses a chunk-streaming encoder (p.8) |
| Audio output | Single-codebook VQ tokens (codebook 16,384) at 25 Hz from a WavLM-large-based tokenizer trained with ASR, TTS-reconstruction and pitch losses (p.4–5). Audio tokens live in the LLM vocabulary and are interleaved with text tokens in one stream, not in parallel streams (p.2, 5) |
| Waveform decoder | Two stages: flow matching turns tokens into continuous acoustic latents, then a BigVGAN vocoder produces 24 kHz audio (following MiniMax-Speech) (p.5) |
| Duplex mechanism | See the **Stage 4** subsection for full detail. In short: the user's continuous stream and the model's token stream are interleaved in 0.16 s chunks (1:4). State tokens THINK / SHIFT / BREAK control listening and speaking. Duplex data appears in pretraining (5B tokens) and in a single-step SFT |
| Open weights / code | Covo-Audio-Chat weights plus inference pipeline (github.com/Tencent/Covo-Audio; HF tencent/Covo-Audio-Chat) (p.1). Release of the base and FD models is not stated |
| Relevance to us | **High.** The closest analogue to our setup: a 7B Qwen-family backbone, about 2T tokens of speech–text pretraining with 40% text, full-duplex data mixed into pretraining, and a duplex model that keeps most of the half-duplex model's intelligence |

## TL;DR

- **Architecture.** Qwen2.5-7B-Base with a Whisper encoder (6.25 Hz continuous input) and 25 Hz single-codebook output tokens. Output is text interleaved with audio, so speech is text-guided. Multi-turn context is built recursively from the user's features plus the model's earlier outputs (p.4–5).
- **Pretraining (~2T tokens; p.5–6, Tab.1).**
  - Stage 1: adapter-only ASR alignment on 200K h (30B tokens).
  - Stage 2: full model on ≈1,985B tokens (derived: sum of Tab.1), drawn from ~8M h of audio:

| Task | Tokens |
|---|---|
| Text-only | 800B (≈40%) |
| Interleave | 540B |
| Audio-only | 240B |
| TTS | 160B |
| "Speech continuation" | 160B |
| ASR | 80B |
| Full-duplex | 5B |

- **"Hierarchical tri-modal interleaving"** places continuous audio, discrete audio and text in one sequence, at both phrase and sentence/utterance granularity (p.3, 6).
- **Base results:** A2A topic-StoryCloze 83.3, A2T 95.7, T2T 99.4 (Qwen2.5-7B-Base: 98.6); sBLIMP 61.6; sWUGGY 74.90 (p.12, Tab.3).
- **Chat SFT (50K steps; p.6–7).**
  - Mix: general intelligence 0.4, spoken dialogue 0.3, speech understanding 0.1, speech generation 0.1, audio understanding 0.1.
  - Data: 10M internal text instructions plus TTS-voiced copies.
  - On-policy KL distillation from a stronger text LLM's top-20 logits.
- **Full duplex keeps most of the dialogue quality.**
  - URO-Bench mean (10 subtasks), English: 83.75 → 80.01; Chinese: 84.37 → 82.40 (derived from Tab.5 vs Tab.8).
  - Behavior success rates: turn-taking 99.7%, pause handling 97.6%, backchanneling 93.89%, interruption 96.81% (p.16, Tab.9).

## Architecture

**Input** (p.4)
- Whisper-large-v3 (50 Hz) followed by the adapter, giving 6.25 Hz continuous features.
- In dialogue, user audio enters only as continuous features. Discrete audio tokens appear on the input side only as the model's own earlier outputs, or in some pretraining formats (a_d→a_d, a_d→t→a_d).
- Notation used in the paper: a_c = continuous audio features, a_d = discrete audio tokens, t = text.

**Output tokenizer** (p.4–5)
- WavLM-large plus one VQ layer: codebook 16,384, 25 Hz.
- Pretrained on a "massive multilingual dataset" with three losses: ASR (semantic grounding), TTS reconstruction (acoustic fidelity) and pitch (prosody).
- Reconstruction metrics: not reported.

**Decoder** (p.5)
- Flow matching to continuous latents, then a BigVGAN vocoder at 24 kHz.

**LLM** (p.2, 5)
- Qwen2.5-7B-Base with an extended vocabulary.
- Outputs one stream of text and audio tokens. The authors chose interleaved over parallel streams as "more adherent to the oracle LLM philosophy."
- Multi-turn context: user continuous features plus the model's earlier text and audio tokens are prefixed as history ("recursive context-filling").

**Conditioning tokens** (Fig.2, p.4)
- Sequences contain prompt tokens, speaker tokens, prompt-audio features and a start token.
- The speaker token supports the intelligence–speaker decoupling (Stage 3b).

**Context length** (derived): 8192 positions ≈ 5.5 min at 25 Hz output tokens (8192 ÷ 25), or ≈21.8 min at 6.25 Hz input (8192 ÷ 6.25).

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **Encoder:** pretrained Whisper-large-v3, used as is (p.4).
- **Output tokenizer:** WavLM-large + VQ, trained with ASR + TTS-reconstruction + pitch losses on a large multilingual set (p.4–5).
- **Decoder:** flow matching + BigVGAN (p.5).
- **Not reported:** data size, steps, LR and compute for the tokenizer and the decoder.

### Stage 1: Modality bridging (ASR alignment)
- **Trainable:** adapter only. The Whisper encoder and the LLM are frozen (p.5).
- **Data:** a 200,000-h multilingual ASR set; 30B tokens (p.5, Tab.1).
- **Schedule:** 50K steps; cosine; peak LR 1e-4; 40K warmup steps (p.5).
- About 600K tokens per step (derived: 30B ÷ 50K).
- **Compute:** not reported.

### Stage 2: Speech–text modality fusion (main mid-training)

**Trainable:** adapter + LLM, "jointly unfrozen" (p.6). The encoder is presumably still frozen.

**Data by task** (Tab.1, p.5):

| Task | Format | Tokens |
|---|---|---|
| ASR | a_c→t | 80B |
| TTS | t→a_d | 160B |
| Audio-only | a_c→a_d, a_d→a_d | 240B |
| Speech continuation | a_c→t (as printed) | 160B |
| Speech–text interleave (a) | a_c→t\|a_d, a_c→t→a_c | 360B |
| Speech–text interleave (b) | a_d→t→a_d, a_c→t→a_d | 180B |
| Text-only | t→t | 800B |
| Full-duplex | a_c→t\|a_d | 5B |

- Sum ≈1,985B (derived). Shares: text-only 40.3%, interleave 27.2%, full-duplex 0.25% (derived).
- Corpus: about 8M h of "diverse audio and speech" plus a multi-domain text corpus, "totaling 3T tokens" (p.6). It is ambiguous whether 3T is the text corpus or the total.

**Interleaving design** (p.6)
- Two patterns:
  - Sequential: a_c→t→a_d.
  - Parallel: a_c→t|a_d.
- Hierarchical granularity: phrase-level segments for fine alignment plus utterance/sentence-level segments for long-form semantic integrity.
- This is presented as an improvement over word/character-level interleaving (which they attribute to GLM-4-Voice). Their argument is that word-level interleaving is "highly sensitive to tokenization or forced-alignment errors."

**Hyperparameters** (p.6)
- Sequence length 8192; cosine; peak LR 3e-5; 500K steps; 20K warmup.
- About 4.0M tokens per step, i.e. ≈485 sequences of 8192 tokens (derived: 1,985B ÷ 500K steps, assuming full packing).

**Compute:** not reported.

### Stage 3: Post-training → Covo-Audio-Chat (half duplex)

**Mixture** (Tab.2, p.6)

| Task group | Ratio | Formats |
|---|---|---|
| General intelligence | 0.4 | t→t, t→a_d, a_c→t, a_c→t\|a_d |
| Spoken dialogue | 0.3 | a_c→t\|a_d |
| Speech understanding | 0.1 | |
| Speech generation | 0.1 | t→a_d |
| Audio understanding | 0.1 | |

**Intelligence data** (p.7)
- 10M internal text instructions (commonsense, math, code, specialized knowledge) for text-to-text training.
- A random subset is voiced by a "high-fidelity neural TTS" (not named). Voices are sampled from thousands of speech prompts, and outputs are WER-filtered.
- These voiced copies create text-to-audio, audio-to-text and audio-to-audio pairs.

**Distillation** (p.7): KL-based on-policy distillation during text-to-text training. A stronger text LLM (not named) supplies top-20 logits as soft targets "to prevent intelligence degradation."

**Colloquial style** (p.7)
- A moderate-scale subset is rewritten conversationally by a text LLM before TTS.
- Different system prompts mark colloquial vs formal style.

**Empathy** (p.7): dialogues covering 7 emotions (joy, anger, sadness, fear, disgust, depression, surprise), voiced with matching emotional speech.

**Hyperparameters** (p.7): sequence length 8192; cosine; peak LR 1e-5; 50K steps; 1K warmup. Batch: not reported.

### Stage 3b: Intelligence–speaker decoupling (voice transfer)

**Method** (p.7–8, Fig.3)
1. Dialogue data is generated with thousands of random speakers, so intelligence is not tied to one voice.
2. Recordings of the target TTS speaker are reformatted into pseudo-dialogues: a constructed prompt, query and text response, plus the authentic speech response.
3. The loss on the text response is masked in these pseudo-dialogues, so they teach the voice but not the content.

**Evidence**
- Plugging TTS-only speakers straight into dialogue gave a "significant decline in performance and reduced naturalness" (p.7). No numbers are given.
- With the method, Covo-Audio-Chat-TTS is close to Covo-Audio-Chat on URO-Bench (10-subtask mean; derived from Tab.5, p.14):
  - English: 83.75 → 81.77.
  - Chinese: 84.37 → 83.55.
  - The worst subtask is English Gsm8kEval (−7.9).

### Stage 4: Full duplex → Covo-Audio-Chat-FD

**Model changes** (p.8)
- The encoder becomes chunk-streaming.
- The user stream and the model stream are chunk-interleaved at 1:4: one 6.25 Hz input frame per four 25 Hz output tokens, so each chunk is 0.16 s (p.8, footnote 3).

**Behavior protocol** (p.8–9, Fig.4b)
- Listening: the model emits THINK tokens while the user speaks.
- Taking the turn: it emits SHIFT when the user's turn ends.
- Speaking: each chunk is one text token followed by four speech tokens. The model keeps receiving the user stream, so it can handle barge-ins and backchannels.
- Stopping: it emits BREAK and returns to listening when interrupted or when its turn ends normally.

**Data** (p.8)
- Half-duplex spoken-dialogue sets are converted to a dual-channel format.
- Barge-ins and backchannels are inserted "in proper positions according to semantic dynamics."
- Volume and overlap/timing statistics: not reported.

**Training** (p.8)
- Full-duplex data is already in pretraining: 5B tokens in Stage 2 (Tab.1).
- Chat-FD is then trained "directly from the pretrained foundation using both half-duplex and full-duplex datasets," in one step instead of OmniFlatten-style progressive stages.
- The authors state, without numbers: "We found this simple-yet-effective approach can yield more competitive results."
- LR and steps: not reported.

### Separate track: speech and audio understanding (SFT + RL)

**Speech understanding** (p.9–10)
- QA data built from LibriSpeech and other ASR sets, VoxCeleb, CN-Celeb, MELD, Common Voice and AudioSkills speech subsets.
- Three modality configurations: text-to-text QA, speech-to-text QA, speech-to-speech QA. Queries and responses are TTS-synthesized.
- Stratified sampling with priority age → language → gender, to reduce demographic bias.

**Audio understanding** (p.10–11), three steps:
1. AudioSkill (8M audio–text pairs): 32K steps; AdamW; LR 3e-6 for the LLM and 1e-5 for the adapter; weight decay 5e-4; 100 warmup steps, then cosine.
2. CoT SFT on AudioMCQ + AF-Think (1M instances; final answers extracted with DeepSeek V3.1): 2 epochs, 1K warmup.
3. GRPO on AVQA. Reward = accuracy + format + consistency (LLM judge) + thinking quality (LLM judge, 0–1 in steps of 0.2).

It is not stated whether these understanding results come from the same checkpoint as Covo-Audio-Chat. The paper calls it an "SFT-based speech understanding model" (p.17).

## Data pipeline

- **Pretraining corpus** (p.5–6): ~8M h of audio plus multi-domain text, and the 200K-h multilingual ASR set for Stage 1. Sources, languages and cleaning are not described.
- **Interleaving** (p.6): phrase-level plus sentence/utterance-level segments. Not reported: whether the interleaved speech is real aligned audio or TTS, and how phrases are segmented.
- **Post-training dialogue** (p.7): fully synthetic.
  - Text instructions are voiced by TTS with thousands of voice prompts and WER filtering.
  - An LLM rewrites a subset into colloquial style.
  - Emotion-aligned TTS is used for the empathy data.
- **Duplex data** (p.8): dual-channel conversions of the half-duplex dialogues with semantically placed barge-ins and backchannels.
  - Silence and overlap modeling details are not reported.
  - For evaluation, interruption and backchannel audio is sampled from a naturally recorded corpus (p.16).
- **Voice data** (p.7–8): lightweight TTS recordings of the target speaker, wrapped in pseudo-dialogues.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Stage 1 (adapter only) | 30B tokens; 200K h ASR | not reported | 50K steps; LR 1e-4 |
| Stage 2 (adapter + LLM) | ≈1,985B tokens (800B text-only); ~8M h audio | not reported | 500K steps; LR 3e-5; ≈4M tokens/step (derived) |
| Chat SFT | not reported (50K steps) | not reported | LR 1e-5 |
| FD SFT | not reported | not reported | single step from the pretrained base |
| Audio understanding | 8M pairs (32K steps) + 1M CoT × 2 epochs + GRPO | not reported | |

## Preserving text intelligence / knowledge

**Mechanisms**
- 800B text-only tokens, 40% of Stage 2 (derived), included "to preserve foundational linguistic knowledge" (p.5–6).
- Adapter-only alignment with a frozen LLM comes before full unfreezing (p.5).
- Text precedes or accompanies audio in every output format: a_c→t|a_d in half duplex, and a text token before every 4 speech tokens in full duplex (p.6, 8).
- In SFT, "general intelligence" is 40% of the mix, and text-to-text training uses KL distillation from a stronger text LLM (p.6–7).

**Measured retention**
- The only text metric: T2T topic-StoryCloze 99.4 vs 98.6 for Qwen2.5-7B-Base (p.12, Tab.3). No MMLU-style text benchmark is reported.

**Knowledge in speech is still the weakest axis**
- VCB-Bench general knowledge (GK) is 49.95 (p.15, Tab.6), vs:
  - Qwen3-Omni 66.86
  - GPT-4o Audio 55.81
  - Kimi-Audio 54.47
  - MiMo-Audio 48.70
  - GLM-4-Voice 41.79
- The authors call this "a deliberate prioritization of core interaction capabilities" (p.14).

## Key results

**Base model, speech–text modeling** (Tab.3, p.12)

| Model | A2A-tSC | A2T-tSC | T2T-tSC | sBLIMP | sWUGGY |
|---|---|---|---|---|---|
| Covo-Audio-Base | 83.3 | 95.7 | 99.4 | 61.6 | 74.90 |
| GLM-4-Voice-Base | 82.9 | 93.6 | – | – | – |
| Step-Audio-2-mini-Base | – | 95.5 | 98.5 | 52.4 | 52.70 |
| SIMS | 88.3 | 94.0 | 98.0 | 59.8 | 75.36 |
| AlignSLM | 86.8 | – | – | 62.3 | – |
| Moshi | 83.0 | – | – | 58.8 | – |

**Base model, ASR/TTS** (Tab.4, p.12)
- Covo-Audio-Base: AISHELL-1 1.96; LibriSpeech clean / other 1.96 / 4.55; Seed-TTS WER en 2.44, zh 1.73.
- GLM-4-Voice-Base: AISHELL-1 2.46; LibriSpeech 2.82 / 7.66; Seed-TTS en 2.91, zh 2.10.

**Chat, URO-Bench basic track** (Tab.5, p.14)
- Chinese: SQuAD 77.34, OpenbookQA 83.60, APE 68.42, MLC 80.69, AlpacaEval 90.02, Wildchat 90.41, HSK5 88.00.
- English: Gsm8kEval 85.68, GaokaoEval 89.11, StoralEval 82.59, TruthfulEval 67.60, AlpacaEval 85.73, WildchatEval 86.82.

**Chat, VCB-Bench** (Tab.6, p.15)
- Instruction following: text instructions (TIF) 93.07 / English 89.94; speech instructions (SIF) 67.41 / English 60.86; multi-turn dialogue 87.70.
- Knowledge: general knowledge 49.95; math/logic 79.34; dialogue comprehension 64.95.
- Robustness: speaker variation 88.94; environmental noise 87.13; content variation 90.37.

**Empathy, VStyle** (Tab.7, p.15)
- en / zh: anger 4.86 / 4.89; sadness 4.83 / 4.93; anxiety 4.97 / 5.00; joy 4.80 / 4.89.
- These are LLM-judge scores. In a subjective listening test the model was still below Doubao (p.15).

**Full duplex** (Tab.8–9, p.16)
- URO-Bench English: Repeat 94.64, GaokaoEval 73.82, Gsm8kEval 80.47, AlpacaEval 84.25, WildchatEval 88.34. Moshi scores 1.26, 6.71, 3.09, 48.48 and 48.18 on the same tasks.
- URO-Bench Chinese: SQuAD 75.16, OpenbookQA 79.37, AlpacaEval 84.90.
- Behavior success rates:

| Model | Turn-taking | Pause handling | Backchanneling | Interruption |
|---|---|---|---|---|
| Covo-Audio-Chat-FD | 99.7 | 97.6 | 93.89 | 96.81 |
| Moshi | 96.8 | 53.2 | – | – |
| Freeze-Omni | 99.1 | 51.2 | – | – |

**Speech and audio understanding** (Tab.10–13, p.18, 34)
- ASR average WER 4.71.
- CoVoST2: en-zh 49.84, zh-en 26.77.
- AIR-Bench average 80.86.
- MMAU test-mini 75.30; MMSU 66.64; MMAR 55.30.

## Ablations & lessons

- **Full duplex on a strong half-duplex base costs little intelligence.**
  - URO-Bench 10-subtask means, half → full duplex: English 83.75 → 80.01 (−3.7); Chinese 84.37 → 82.40 (−2.0) (derived from Tab.5 vs Tab.8).
  - The biggest single drop, GaokaoEval −15.3, is attributed to long silences inside that test set that trigger premature responses (p.16, footnote). Excluding it, English goes 83.15 → 80.70 (derived).
  - Reasoning drops: Gsm8kEval −5.2 and TruthfulEval −3.1 (English); OpenbookQA −4.2 (Chinese).
- **Pause handling is where full duplex loses intelligence.** The remaining half-vs-full-duplex gap "largely stems from" early responses at short intra-turn pauses, which "correlates directly with its pause-handling success metric" (97.6%) (p.16).
- **Full-duplex tokens can be a tiny share of pretraining.**
  - 5B of ≈1,985B Stage-2 tokens (0.25%, derived), plus a single SFT on half- and full-duplex data.
  - That combination gives 99.7% turn-taking and 96.81% interruption handling (p.5, 8, 16).
  - There is no ablation of removing full-duplex data from pretraining.
- **Single-step FD SFT reportedly beats multi-stage progressive training.** This is a claim with no numbers (p.8).
- **Cheap voice transfer works if TTS data is wrapped in dialogue context and text loss is masked.**
  - Chat-TTS is within about 1–2 pts of Chat on URO-Bench means (derived: −2.0 English, −0.8 Chinese), with a −7.9 outlier on English Gsm8kEval (p.14–15).
  - Naive transfer "significantly" degraded quality; no numbers are given (p.7).
- **About 40% text-only tokens keep text ability intact.** T2T topic-StoryCloze is 99.4 vs 98.6 for the base LLM (p.12; share derived).
- **Speech continuation lags understanding, and RL-trained speech LMs still lead on it.**
  - A2A topic-StoryCloze is 83.3, vs 88.3 for SIMS and 86.8 for AlignSLM. The authors note those two used extra RL (p.12–13).
  - sWUGGY is 74.90 vs 75.36 for SIMS.
  - A2T topic-StoryCloze is 95.7 vs A2A 83.3, so the model understands a story in speech much better than it continues one in speech (12.4-pt gap, derived).
  - The authors argue specialized A2A models "exhibit signs of overfitting on continuation tasks" (p.12–13).
- **Knowledge stays weak after SFT.** General knowledge on VCB-Bench is 49.95 vs 66.86 for Qwen3-Omni (Tab.6, p.15), despite text-to-text distillation and a 40% general-intelligence SFT share.
- **Hierarchical (phrase + sentence) interleaving** is argued to avoid alignment errors and preserve long-form semantics (p.6). There is no ablation.
- **LLM-as-judge overrates semantic content on empathy.** Subjective listening ranks the model below Doubao (p.15).

## Implications for our setup

- **This is the closest template to ours. The missing piece is probably the general stage, not more duplex data.**
  - Covo ran about 2T tokens of speech–text pretraining (~8M h of audio plus 800B text-only tokens) before any dialogue SFT. Full-duplex sequences were only 0.25% (5B tokens) of that (p.5–6).
  - Our 10K h of duplex speech is roughly the same order as Covo's 5B FD tokens: ≈37K–111K h-equivalent (derived: 5B ÷ (12.5–37.5 tokens/s) ÷ 3600, assuming the 0.16 s chunk layout on p.8).
  - So our duplex volume may already be adequate; the general stage that precedes it is what we lack.
- **Keep about 40% text-only tokens in mid-training,** and run an adapter-only alignment phase with a frozen LLM first (p.5–6).
- **Text-guided duplex output.** One text token per four speech tokens in each 0.16 s chunk, plus explicit THINK / SHIFT / BREAK state tokens (p.8–9). Covo-Audio, GLM-4-Voice (text-first) and Kimi-Audio (parallel text head) all generate text alongside or ahead of speech. MiMo-Audio does not describe this. Among these papers it is the standard answer to weak knowledge in generated speech.
- **Inject knowledge into duplex data synthetically.** Convert knowledge-rich half-duplex dialogues (text → TTS) into dual-channel form with barge-ins and backchannels inserted at semantically appropriate points (p.8), and mix them with our real 10K h.
- **Measure and protect pause handling.** Premature responses at intra-turn pauses directly lower measured intelligence (p.16). Evaluate on test sets with long silences.
- **Post-training for knowledge.**
  - Make about 40% of the SFT mix general intelligence (text-to-text plus TTS-voiced audio copies).
  - Add on-policy KL distillation from a stronger text teacher's top-20 logits (p.6–7).
  - With Qwen3 we could distill from a larger Qwen3 text model.
- **Voice.** Use intelligence–speaker decoupling (masked-text-loss pseudo-dialogues built from TTS recordings) to adopt a target voice without re-synthesizing dialogue data (p.7–8).
- **Eval suite to adopt.**
  - Base checkpoints: topic-StoryCloze (A2A / A2T / T2T), sBLIMP, sWUGGY.
  - Chat: URO-Bench and VCB-Bench (the latter has a general-knowledge split).
  - Full duplex: turn-taking, pause-handling, backchannel and interruption success rates (p.11–16).

## Open questions / unclear points

- Whether the interleaved speech is real aligned audio or TTS, how phrases are segmented, and how many tokens use each granularity.
- The "Speech Continuation" task is printed as a_c→t in Tab.1, which looks like text output. What is actually predicted?
- "3T tokens" (p.6) vs "2T tokens" (p.5): is 3T the size of the text corpus or the total?
- For FD SFT: data volume, hyperparameters, the half:full-duplex ratio, the overlap and backchannel statistics inserted, and whether the user channel includes real recorded speech.
- Within FD chunks, how the text token is handled once the response text runs out (padding?). The half-duplex output interleave ratio is also not stated.
- Tokenizer and decoder training data, sizes and reconstruction metrics.
- No compute or hardware figures at all.
- Whether the speech/audio-understanding results come from Covo-Audio-Chat or from separately fine-tuned checkpoints.
- No numeric ablations for hierarchical interleaving, KL distillation, full-duplex data in pretraining, or single-step FD SFT.
