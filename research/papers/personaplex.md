# PersonaPlex: voice and role control for full-duplex conversational speech models

| Field | Value |
|---|---|
| Org | NVIDIA (Roy, Raiman, Lee, Ene, Kirby, S. Kim, J. Kim, Catanzaro) |
| Date / arXiv | Accepted at ICASSP 2026 (p.1). arXiv:2602.06053, as cited in the VoiceChat-TTS references (p.7). |
| Category | full-duplex S2S (a fine-tune of Moshi) |
| Backbone | Moshi. The weights are initialized from Moshi and fine-tuned, keeping its Temporal Transformer + Depth Transformer and the Mimi codec (p.2, p.4). The released checkpoint is named personaplex-7b-v1 (p.5). |
| Audio input | Mimi codec tokens of the user audio. Inputs are three streams: user audio, agent text, agent audio (p.2). The frame rate is 12.5 Hz (derived: 2,048 tokens = 163.84 s, p.4). The number of codebooks is not restated. |
| Audio output | Agent text (inner monologue) plus Mimi tokens on the agent-audio stream. The Depth Transformer predicts the codebooks within each frame (Fig.1, p.2). |
| Waveform decoder | Mimi decoder, inherited from Moshi |
| Duplex mechanism | Moshi-style parallel streams consumed and generated every frame. It adds a Hybrid System Prompt prefix: a voice prompt on agent audio, then a text prompt on agent text, with the user channel set to a 440 Hz sine wave during the prompt (p.2). |
| Open weights / code | nvidia/personaplex-7b-v1 on Hugging Face (p.5). The Service-Duplex-Bench release is "planned" (p.4). |
| Relevance to us | **medium-high**. It is the cheapest documented duplex fine-tuning recipe on synthetic data (48 A100-hours) and includes a data-scale ablation. It is also a clear counter-example: duplex fine-tuning does not add knowledge. |

## TL;DR

- PersonaPlex is Moshi fine-tuned so that one model can take on any role (from a text prompt) and any voice (zero-shot cloning from a speech sample). The conditioning is a "hybrid system prompt" written into Moshi's existing streams, with no architecture change (p.2).
- Training data: 2,250 hours (derived), all synthetic.
  - 1,840 h of customer-service dialogues (105,410 dialogues).
  - 410 h of question-answering dialogues (39,322 dialogues).
  - Transcripts come from Qwen-3-32B and GPT-OSS-120B; audio from the Dia and Chatterbox TTS models (p.2–4).
- Training cost: 24,576 steps, batch 32, sequences of 163.84 s, learning rate 2e-6 to 4e-6, 6 hours on 8×A100 (p.4).
- Results (paper model): best human-rated naturalness of five systems (DMOS 3.90), voice similarity 0.57 versus Moshi's 0.10, and role adherence 4.48, second only to Gemini's 4.73 (Tab.1, Tab.4).
- The released checkpoint adds 1,217 hours of real Fisher conversation with LLM-written prompts, synthetic voices, and Chatterbox-only rendering. This improved pause handling and backchanneling (Tab.6). VoiceChat's Table 1 baseline numbers come from this released checkpoint.
- Knowledge does not move. VoiceBench is 30.6 for PersonaPlex against 29.5 for Moshi (third-party numbers, VoiceChat Tab.3).

## Architecture

**Moshi base (p.2, Fig.1).**
- Three input streams per frame: user audio, agent text, agent audio.
- The Temporal Transformer models time; the Depth Transformer predicts each frame's codec tokens. Mimi is the neural audio codec.
- The model autoregressively generates text and audio while receiving live user audio.
- The paper does not restate Moshi's sizes, codebook count or token delays.

**Hybrid System Prompt (p.2).** Two segments are concatenated in time before the dialogue, separated by pauses (Fig.1):

| Segment | Agent audio | Agent text | User channel |
|---|---|---|---|
| Voice prompt | speech sample of the target voice | PAD | 440 Hz sine wave |
| Text prompt | silence | role/scenario text between `<system>` delimiters | 440 Hz sine wave |

- The sine wave on the user channel is there to stabilize conditioning. Custom text/audio delimiters mark the boundary between prompt and dialogue.
- Later agent turns are generated in the prompt's voice, which gives zero-shot cloning.
- **Order does not matter.** The authors observe no difference either way. They put the voice prompt first so it can be prefilled and cached when cloning is not needed, which reduces latency (p.2).
- **Loss:**
  - The loss is masked over the system prompt.
  - Following Moshi, non-semantic audio tokens are down-weighted to 0.02 and padded text tokens to 0.3 (p.2).

**How it relates to the other NVIDIA work.**
- Unlike NemotronLabs VoiceChat, PersonaPlex is end-to-end: its LLM generates the audio tokens.
- VoiceChat uses it as a baseline (VoiceChat Tab.1, Tab.3). VoiceChat-TTS re-voices its text stream to isolate decoder quality (VoiceChat-TTS Tab.2).
- Authors overlap: Roy and Jaehyeon Kim also wrote VoiceChat-TTS, and Roy contributed to VoiceChat.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training

Inherited from Moshi: the Mimi codec and the Moshi weights used for initialization (p.4). Nothing is retrained here, and this paper gives no details of how those were trained.

### Stage 1: hybrid-prompt fine-tuning on synthetic dialogues (paper model)

- **Objective:** Moshi's next-token loss on agent text and agent audio, with the system prompt masked and the 0.02 / 0.3 down-weighting above (p.2).
- **Trainable modules:** both transformers, with separate learning rates (depth 4e-6, temporal 2e-6) (p.4). Whether Mimi or the embeddings are frozen is not stated.
- **Data:** about 2,250 hours, all synthetic (p.4):
  - 1,840 h of service dialogues, 105,410 dialogues, about 63 s on average (derived);
  - 410 h of QA dialogues, 39,322 dialogues, about 38 s on average (derived).
- **Sequence length:** at most 2,048 tokens, which is 163.84 s (p.4).
- **Optimization:** Adam with cosine annealing, 24,576 steps, batch size 32 (p.4).
- **Compute:** 6 hours on 8×A100 (p.4), i.e. 48 A100-hours (derived).
- **Audio seen:** at most about 1.61B frames, or about 35.8k hours of audio. That is up to about 16 passes over the 2,250-hour set (derived upper bound, assuming every sequence is full length).

### Stage 1′: released checkpoint (Appendix A, p.5)

- **Added real data:** 7,303 Fisher English conversations, 1,217 hours, about 10 minutes each (derived).
  - The goal was more natural backchannels, expressions and emotional responses.
  - Each conversation gets a GPT-OSS-120B-written prompt at one of three detail levels (minimal persona, plus topic, plus detailed backstory), to balance generalization against precise instruction following.
- **Voices:** all synthetic dialogues now use synthetic voices sampled from TortoiseTTS, with Praat pitch and formant augmentation, instead of real-speaker corpora. The stated reason is data privacy.
- **Rendering:** all synthetic dialogues (service and QA) are re-rendered with Chatterbox, replacing the Dia/Chatterbox mix.
- **Not reported:** hyperparameters, steps, mixture ratios, and whether training continued from the paper model.

No RL or preference stage. Post-training alignment and tool integration are listed as future work (p.4).

## Data pipeline

**Transcripts (p.2–3).** All generated with Qwen-3-32B and GPT-OSS-120B.
- **Service scenarios:** sample a domain (e.g., restaurant, bank), then a scenario (refund, information request, general enquiry), then a high-level description. An LLM expands it into a two-speaker transcript plus an agent role context (name, company, facts such as plans and prices). Training scenarios are kept distinct from the benchmark's.
- **QA scenarios:** two-turn question-answering dialogues across topics, where the second question may change topic or follow up. They use one fixed role prompt: a wise, friendly teacher who answers clearly and engagingly.

**Voices (p.3).** 26,296 single-speaker voice samples from VoxCeleb, Libriheavy, LibriTTS, CommonAccent and Fisher. Of these, 2,630 are held out to test speaker similarity.

**Audio rendering (p.3).**
- **Service dialogues:** Dia, a multi-speaker TTS that generates both speakers jointly from a transcript while cloning two voice samples. It better captures timing, interruptions and room tone.
- **QA dialogues:** Chatterbox (single-speaker, zero-shot cloning), one random voice per role, then stitched together.
  - Stitching inserts silence padding between user and agent turns to simulate natural turn-taking.
  - *Negative* silence makes turns overlap, which simulates barge-in and interruption.
  - The authors point to SALM-Duplex [12] as prior validation of this approach.

**Evaluation sets (p.1, p.3–4).** Full-Duplex-Bench (400 questions) plus the new Service-Duplex-Bench: 50 service roles × 7 single-turn probes = 350 questions, covering proper nouns, context details, unfulfillable requests, rudeness, unspecified and unrelated requests (Tab.3).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Moshi pretraining (inherited) | not reported here | not reported here | — |
| Fine-tuning (paper model) | 2,250 h of data; ≤35.8k h / ≤1.61B frames processed (derived) | 8×A100 × 6 h = 48 A100-hours | 24,576 steps × batch 32 × 2,048 tokens |
| Released checkpoint | + 1,217 h Fisher, re-rendered synthetic data | not reported | — |

## Preserving text intelligence / knowledge

Not addressed directly. The fine-tune is gentle (LR 2e-6 / 4e-6, 24.6k steps), and role prompts are forced onto the agent-text stream (p.2), which we read as reusing the inner-monologue LM's instruction following. Outcome evidence comes from other papers:

- VoiceBench average 30.6 versus 29.5 for Moshi. MMSU 24.9, OBQA 24.4, SD-QA 18.8, IFEval 12.0 (VoiceChat Tab.3, measured by the DuplexCascade authors on the released checkpoint).
- AdvBench refusal is 8.1, below Moshi's 44.2 (leaderboard number, different evaluator). The fine-tune may have eroded refusal behavior; the paper mentions no safety data.

## Key results

**Naturalness and voice cloning** (Tab.1, p.3). DMOS from Amazon Mechanical Turk (202 raters / 1,616 samples on Service-Duplex-Bench; 152 / 1,216 on FDB User Interruption, p.4). SSIM is WavLM-TDNN similarity to the voice prompt.

| System | DMOS, FDB | DMOS, Service-Duplex-Bench | SSIM |
|---|---|---|---|
| PersonaPlex | 3.90 ± 0.15 | 3.59 ± 0.12 | 0.57 |
| Gemini | 3.72 ± 0.14 | 3.22 ± 0.14 | 0.00 |
| Qwen-2.5-Omni | 3.70 ± 0.13 | 2.37 ± 0.20 | 0.07 |
| Freeze-Omni | 3.51 ± 0.18 | 2.38 ± 0.21 | 0.05 |
| Moshi | 3.11 ± 0.15 | 2.83 ± 0.13 | 0.10 |

**Full-Duplex-Bench 1.0, paper model vs released checkpoint** (Tab.2 p.3, Tab.6 p.5). TOR = takeover rate.

| Metric | Paper model | Released | Moshi | Gemini |
|---|---|---|---|---|
| Pause TOR, synthetic ↓ | 0.584 | 0.358 | 0.985 | 0.255 |
| Pause TOR, CANDOR ↓ | 0.662 | 0.431 | 0.980 | 0.310 |
| Backchannel TOR ↓ | 0.327 | 0.273 | 1.000 | 0.091 |
| Backchannel frequency ↑ | 0.025 | 0.042 | 0.001 | 0.012 |
| Backchannel JSD ↓ | 0.649 | 0.662 | 0.957 | 0.896 |
| Smooth-turn TOR ↑ | 0.992 | 0.908 | 0.941 | 0.655 |
| Smooth-turn latency (s) ↓ | 0.070 | 0.170 | 0.265 | 1.301 |
| Interruption TOR ↑ | 1.000 | 0.950 | 1.000 | 0.891 |
| Interruption quality (GPT-4o) ↑ | 4.210 | 4.290 | 0.765 | 3.376 |
| Interruption latency (s) ↓ | 0.400 | 0.240 | 0.257 | 1.183 |

**Service-Duplex-Bench, GPT-4o mean** (Tab.4, p.4): Gemini 4.73, PersonaPlex 4.48, Freeze-Omni 4.02, Qwen-2.5-Omni 2.76, Moshi 1.75. Qwen-2.5-Omni was run with Freeze-Omni's voice-activity detector (p.4 fn.).

**Released-checkpoint DMOS** (Tab.7, p.5; a separate rater pool, not comparable with Tab.1): PersonaPlex 2.95 ± 0.25, Qwen-2.5-Omni 2.81, Gemini 2.80, Freeze-Omni 2.51, Moshi 2.44.

## Ablations & lessons

- **Data-scale ablation** (Tab.5, p.4). Fraction of the 2,250 h used:

  | Data used | SSIM | FDB GPT-4o | Service-Duplex-Bench GPT-4o |
  |---|---|---|---|
  | 0% (Moshi) | 0.10 | 0.77 | 1.75 |
  | 25% (≈560 h, derived) | 0.54 | 4.44 | 4.20 |
  | 50% | 0.56 | 4.52 | 4.24 |
  | 100% | 0.57 | 4.21 | 4.48 |

  Voice cloning and post-interruption response quality saturate with a few hundred hours; the FDB score even dips at 100%. Role adherence keeps rising with data (p.4).
- **The synthesis engine matters for voice cloning.** Rendering all synthetic dialogues with Chatterbox raised SSIM from 0.57 to 0.65 (p.5). This is confounded with the switch to synthetic voices and the added Fisher data.
- **Real conversational data improves listening behavior, at some cost in responsiveness.** Going from the paper model to the released one, which adds Fisher with LLM prompts among other changes:
  - Gains: pause TOR 0.584 → 0.358 (synthetic) and 0.662 → 0.431 (CANDOR); backchannel frequency 0.025 → 0.042; interruption latency 0.400 → 0.240 s.
  - Costs: smooth-turn TOR 0.992 → 0.908; smooth-turn latency 0.070 → 0.170 s; interruption TOR 1.000 → 0.950 (Tab.2 vs Tab.6).
  - All changes were made at once, so none of these effects can be attributed to a single change.
- **Duplex fine-tuning reshapes behavior but not knowledge.** The 2,250-hour synthetic set alone takes role adherence from 1.75 to 4.48 (Tab.5). The released checkpoint adds 1,217 hours of Fisher on top, about 3.5k hours in total (derived). Yet VoiceBench stays at Moshi's level (30.6 vs 29.5, cross-paper).
- **The joint model's own audio is the weak link.** A dedicated TTS re-voicing the same text halves WER (5.00 → 2.42) (VoiceChat-TTS Tab.2, cross-paper).
- **A strong turn-taking prior survives a gentle fine-tune.** Moshi's responsiveness is kept and improved: smooth-turn latency 70 ms versus Moshi's 265 ms. Pause handling stays weak in the paper model (pause TOR 0.58–0.66) and improves only with the released checkpoint's combined changes, which include Fisher.

## Implications for our setup

Our setup: Qwen3 LLM backbone, Qwen3 audio tokenizer and its vocoder/decoder, mid-trained on about 10K hours of duplex conversational speech. Problems: weak audio continuation and weak semantic knowledge in the generated speech.

- **More duplex conversation hours won't fix weak knowledge.** PersonaPlex is the cleanest case: thousands of hours of synthetic and real dialogue leave VoiceBench at its base model's level. Knowledge has to come from the backbone and a knowledge-bearing speech-text stage, as in VoiceChat's TTS-rendered pretraining-text CPT. Our 10K-hour duplex mid-training should be treated as behavior data.
- **Behavior is cheap.** A few hundred to about 2k hours of well-designed synthetic dialogue plus about 48 GPU-hours at a very low learning rate were enough to move turn-taking, voice and role on top of a strong duplex prior. If our base already has good duplex dynamics, post-training doesn't need much scale.
- **Hybrid system prompt for Qwen3 duplex:** a voice-prompt segment on the agent-audio stream and a text role prompt on the agent-text stream, with a constant user-channel filler (they use a sine wave) and the loss masked. This gives persona and voice control with no architecture change, and the voice segment can be prefix-cached.
- **Recipe for synthesizing duplex dialogues:** LLM-written two-speaker transcripts with a role context, rendered either by a joint two-speaker TTS (captures timing, overlaps, room tone) or by single-speaker TTS stitched with positive or negative gaps for barge-in. Chatterbox-only rendering gave better speaker consistency.
- **Put prompts on our real data.** Annotate our 10K hours of real duplex speech with LLM-written system prompts at several levels of detail, as they did for Fisher. That turns unconditioned conversations into instruction-conditioned ones.
- **Loss weighting on audio tokens:** Moshi's 0.02 weight on non-semantic codebooks pushes capacity toward semantics. If our audio continuation is weak, check how our acoustic-codebook loss is weighted and whether the acoustic stream is undertrained.
- **Evaluation:** Service-Duplex-Bench for role adherence; human DMOS on FDB user-interruption audio; WavLM-TDNN SSIM against the voice prompt.

## Open questions / unclear points

- **Base model details:** Mimi codebook count, acoustic delay and model size (7B is inferred only from the checkpoint name) are not restated. Which modules were frozen is not stated.
- **Released-checkpoint training:** steps, learning rates, the Fisher-to-synthetic mixture, and whether it was trained from Moshi or from the paper model are not reported.
- **Fisher handling:** how were Fisher's two channels assigned to user and agent, and how were the ~10-minute calls cropped to the 163.84-s window?
- **Not reported:** stitching gap/overlap distributions; WER of generated speech. VoiceBench and AdvBench come only from third parties.
- **Comparability:** Gemini and Qwen-2.5-Omni were run with different turn-detection setups, and the DMOS rater pools differ between Tab.1 and Tab.7.
- **The FDB GPT-4o dip** at 100% data versus 50% (4.21 vs 4.52) is unexplained.
