# Step-Audio 2 Technical Report

| Field | Value |
|---|---|
| Org | StepFun Audio Team, with external contributors from NTU Singapore (p.18) |
| Date / arXiv | arXiv:2507.16632, v3 of 27 Aug 2025 (p.1) |
| Category | half-duplex S2S and audio understanding (end-to-end LALM with tool use) |
| Backbone | "a textual LLM"; name and size **not reported**, only that it has "fewer parameters than Step-Audio", which was 130B (pp.1–2, 7). The open **mini** variant uses a Qwen2.5-7B init and the Qwen2-Audio encoder (p.19). |
| Audio input | In-house encoder pretrained on ASR, speaker age/gender and audio-event tasks. 25 Hz output, **frozen for all training**. A 2× downsampling adaptor feeds 12.5 Hz continuous features into the LLM (p.6). |
| Audio output | Discrete CosyVoice 2 tokens; the vocabulary is extended by 6.6K audio tokens (p.7). They are interleaved with text tokens "at a fixed ratio" and padded at the end (p.6). The ratio and token rate are not stated. One token stream; no multi-codebook scheme is described. |
| Waveform decoder | Flow matching to Mel, then a HiFi-GAN vocoder. The flow-matching model has a CNN encoder layer after each self-attention module and is trained on 200,000 h of high-quality speech (p.7). Frozen (Fig.3). |
| Duplex mechanism | n/a; half duplex. Deployment uses a VAD module to filter input speech (p.7). |
| Open weights / code | GitHub; **Step-Audio 2 mini** is open (p.19). The StepEval-Audio-Paralinguistic and -Toolcall benchmarks are released (pp.10, 12). |
| Relevance to us | **Medium-high.** The most detailed per-task token budget in the Step lineage that StepAudio 3 builds on. But it is half duplex and has no ablations. |

## TL;DR

- **Architecture.** A frozen 25 Hz encoder and an adaptor (12.5 Hz) feed an LLM that outputs **text interleaved with CosyVoice 2 audio tokens**, decoded by flow matching + HiFi-GAN. User audio enters as continuous features; the model's own past outputs re-enter history as discrete text and audio tokens (p.6, Fig.3).
- **Pretraining.** **1.356T tokens over 21 days** in four phases: adaptor-only ASR alignment (100B), vocabulary extension (256B; derived: 128B + 128B), main (800B) and cooldown (200B). Every phase after alignment is **50% text** (p.7). The headline totals are "680B text tokens and 8M hours of real and synthesized audio" (p.3).
- **Post-training:**
  - SFT on 4B tokens for one epoch, including synthesized conversations, ~1K scripts per tool and reasoning cold-start sets;
  - RL in three stages: **PPO with a binary length reward** (60 iterations) → **PPO with a learned reward model** (120 iterations) → **GRPO** (400 iterations) (pp.8–9).
- **Tools.** Web search, weather and date/time, plus a novel **audio search** over hundreds of thousands of voice clips, used to switch timbre or speaking style (p.6).
- **Results:**
  - ASR average 3.14% WER (EN) / 3.08% CER (ZH);
  - StepEval-Paralinguistic 83.09; MMAU 78.0;
  - URO-Bench basic 83.32 (ZH) / 83.90 (EN), where GPT-4o Audio scores 78.59 / 84.54 (Tabs.1–3, 6).

## Architecture

- **Components** (p.6, Fig.3): audio encoder → audio adaptor → LLM decoder → audio detokenizer.
  - The encoder is pretrained on ASR, speaker age/gender and audio-event detection, outputs 25 Hz and stays **frozen**.
  - The adaptor downsamples 2× to 12.5 Hz.
- **Output.** The LLM emits text and CosyVoice 2 audio tokens interleaved at a fixed ratio, padded to fill it. The audio tokens are extracted and passed to the detokenizer (p.6).
- **History.** For the next turn, input audio *features* and output *interleaved tokens* are pre-filled as history (p.6). The model's own prior speech is therefore re-read as discrete tokens, while user speech is read as latent features (Fig.3).
- **Detokenizer** (p.7). Flow matching produces Mel from audio tokens, then HiFi-GAN produces the waveform. A CNN encoder layer after each self-attention block "significantly improves" Mel reconstruction, pronunciation and timbre similarity; no numbers are given.
- **Tools** (pp.6–7): audio search, date/time, weather and web search. Retrieved content is appended after the input audio features before speech is generated.
- **Real-time use** relies on a VAD front end shared with Step-Audio and Step-Audio-AQAA (p.7). There is no native duplex.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
- **Encoder:** pretrained on ASR, age/gender and audio-event tasks; sizes and data not reported (p.6).
- **Tokenizer:** reused from CosyVoice 2 (p.6).
- **Detokenizer:** the flow-matching module is trained on **200,000 h** of high-quality speech (p.7).

### Stage 1: alignment (p.7)
- **Data:** 100B tokens of ASR.
- **Trainable:** adaptor only; encoder and LLM frozen.
- **Schedule:** 12K steps at sequence length 8,192, LR 10⁻⁴ → 2×10⁻⁵. That is ≈8.3M tokens per step, about 1K sequences (derived).

### Stage 2: vocabulary extension (p.7)
- The text tokenizer gains 6.6K audio tokens, which are "properly embedded" while textual ability is preserved.
- **Data:** 256B tokens (derived):
  - 128B text;
  - 128B audio = 80B TTS + 32B speech-to-speech (S2S) conversation + 16B utterance-level text-speech interleaved continuation.
- **Sequence length:** 16,384.
- **Separate LRs:** LLM 2×10⁻⁵, adaptor 5×10⁻⁵, embedding 5×10⁻⁵, output layer 4×10⁻⁵.

### Stage 3: main pretraining (p.7)
- 800B tokens at a unified LR of 2×10⁻⁵.
- **400B text.**
- **400B audio** (derived sum):

  | Task | Tokens |
  |---|---|
  | S2S conversation | 150B |
  | TTS | 120B |
  | utterance-level text-speech interleaved continuation | 45B |
  | ASR | 42B |
  | text-to-speech translation | 30B |
  | speech-to-text translation | 8B |
  | speech-to-text continuation | 5B |

### Stage 4: cooldown (p.7)
- 200B tokens with LR 2×10⁻⁵ → 5×10⁻⁶.
- **100B high-quality text.**
- **100B audio** (derived sum):

  | Task | Tokens |
  |---|---|
  | multilingual and dialect ASR | 24.6B |
  | TTS | 12.4B |
  | speech-to-text translation | 3.6B |
  | paralinguistic understanding | 2.4B |
  | S2S conversation *(synthesized)* | 36B |
  | utterance-level text-speech interleaved conversation *(synthesized)* | 15B |
  | S2S translation *(synthesized)* | 6B |

  The synthesized rows come from a conversational speech synthesis pipeline with a library of **~50k speakers**.

Total: 100 + 256 + 800 + 200 = 1,356B (consistent with p.7). The encoder is frozen throughout; the LLM trains from Stage 2 on.

### Stage 5: SFT (p.8)
- **Schedule:** 4B tokens of text and audio, one epoch, LR 10⁻⁵ → 10⁻⁶.
- **Data:**
  - ASR: GigaSpeech, WenetSpeech and in-house data, multilingual and dialect.
  - AudioSet and AudioCaps reformatted into audio QA.
  - An in-house **detailed speech captioning** set covering 11 paralinguistic and environmental aspects.
  - Professionally labeled in-house TTS.
  - CoVoST 2 zh↔en for speech translation.
  - High-quality text conversations, rewritten by several LLMs into colloquial scripts with random emotion and speed instructions, then synthesized.
  - ~1K tool-use scripts per tool, with explicit and implicit invocations, synthesized.
  - Two **reasoning cold-start sets**:
    - complex acoustic scenes from mixed AudioSet/AudioCaps clips;
    - emotion-annotated synthesized conversations.

    In both, a text reasoning LLM writes QA with step-by-step traces from the mixing recipes or scripts.

### Stage 6: RL (pp.8–9)
- **PPO stage 1.** The binary reward is 1 if reasoning is "appropriately concise (neither empty nor excessively long)", else 0. 60 iterations, global batch 64, actor LR 1×10⁻⁶, critic LR 2.5×10⁻⁶.
- **PPO stage 2.** A **learned reward model** scores response quality. 120 iterations with the same batch and LRs.
- **GRPO.** 400 iterations "to further improve the model's audio perceptual abilities".
- Reward-model training, RL prompt data size and KL settings: not reported.
- The citation attached to PPO and GRPO ([54]) is actually the DPO paper (reference list, p.15).

## Data pipeline

- **Conversational speech synthesis pipeline** with ~50k speakers. It produces the cooldown's S2ST, interleaved-conversation and S2S-conversation data (6B / 15B / 36B, p.7), and the SFT dialogues and tool scripts (p.8). Whether the main phase's 150B of conversation data is synthesized is not stated.
- **Dialogue scripts** are LLM-rewritten from text chat into colloquial style, with random emotion and speed tags (p.8).
- **Audio search library:** "hundreds of thousands of speeches" with transcripts and descriptions (p.6).
- **Real vs synthetic** shares of the "8 million hours" are not reported (p.3).
- **No overlap, backchannel or interruption data**: the model is half duplex.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| Detokenizer (flow matching) | 200k h speech (p.7) | not reported | |
| Stage 1: alignment | 100B ASR tokens, 12K steps, 8K sequence | not reported | adaptor only |
| Stage 2: vocab extension | 256B (128B text + 128B audio), 16K sequence | not reported | per-module LRs |
| Stage 3: main | 800B (400B text) | not reported | LR 2e-5 |
| Stage 4: cooldown | 200B (100B text) | not reported | LR 2e-5 → 5e-6 |
| Pretraining total | **1.356T tokens** | **21 days** (p.7); GPUs not reported | ≈64.6B tokens/day (derived) |
| SFT | 4B tokens, 1 epoch | not reported | |
| RL | 60 + 120 PPO iterations (batch 64) + 400 GRPO iterations | not reported | |

## Preserving text intelligence / knowledge

- **Recipe.**
  - The LLM is frozen during alignment.
  - New rows are trained at a higher LR (5×10⁻⁵ / 4×10⁻⁵) than the LLM body (2×10⁻⁵).
  - Text is **exactly 50% of tokens in every phase after alignment** (derived: 128/256, 400/800, 100/200).
  - The output stream is text-interleaved, so text tokens lead the speech.
- **Claim.** The model keeps "its textual performance inherited from the initial textual LLM" (p.8). **No text benchmark is reported** to back this up.
- **Text-token mismatch.** Text tokens summed across Stages 2–4 come to 628B (derived), vs "680 billion tokens of text data" on p.3. The difference is unexplained; it may include SFT or count differently.
- **Knowledge in spoken answers** is measured only indirectly: URO-Bench Understanding and Reasoning tracks (Tab.6) and tool calling, which brings in fresh knowledge via web search (Tab.5).

## Key results

- **ASR (Tab.1).**
  - English average 3.14% WER (LibriSpeech clean 1.17 / other 2.42).
  - Chinese average 3.08% CER (AISHELL 0.63).
  - In-house accents and dialects average 8.85, vs 14.66 for Doubao LLM ASR.
- **Paralinguistics (Tab.2).** StepEval-Audio-Paralinguistic, 550 clips over 11 dimensions: **83.09**, vs Kimi-Audio 49.64, Qwen-Omni 44.18, GPT-4o Audio 43.45 and Step-Audio-AQAA 36.91.
- **MMAU v05.15.25 test-mini (Tab.3):** 78.0 (sound 83.5 / speech 76.9 / music 73.7), vs Omni-R1 77.0 and Audio Flamingo 3 73.1.
- **Translation (Tab.4):** CoVoST 2 average 39.26 BLEU; CVSS speech-to-speech average 30.87.
- **Tool calling (Tab.5).** StepEval-Audio-Toolcall, with speech input. Trigger precision/recall: audio search 86.8/99.5, web search 88.4/95.5. Qwen3-32B with text input gets only 67.5/98.5 on audio search.
- **URO-Bench (Tab.6).** ASR-mediated: Whisper transcribes, GPT-4o-mini judges.

  | Language | Track | Step-Audio 2 (U / R / O) | GPT-4o Audio |
  |---|---|---|---|
  | ZH | basic | **83.32** (91.05 / 75.45 / 86.08) | 78.59 |
  | ZH | pro | **68.25** | 67.10 |
  | EN | basic | 83.90 (92.72 / 76.51 / 84.92) | **84.54** |
  | EN | pro | 66.07 | **67.51** |

  U = understanding, R = reasoning, O = oral conversation.

## Ablations & lessons

The paper contains **no controlled ablations**. Usable comparative evidence:

- **Backbone and encoder drive spoken reasoning more than perception.** Step-Audio 2 mini (Qwen2.5-7B + Qwen2-Audio encoder) was trained on the *same data*, with only the web-search tool (p.19). Comparing full vs mini (Tabs.7–11):

  | Metric | Full | Mini |
  |---|---|---|
  | ASR EN avg | 3.14 | 3.50 |
  | ASR ZH avg | 3.08 | 3.19 |
  | CoVoST BLEU | 39.26 | 39.29 |
  | URO-Bench EN basic | **83.90** | 74.36 |
  | – EN reasoning sub-score | **76.51** | 60.12 |
  | URO-Bench ZH basic | **83.32** | 77.81 |
  | – ZH reasoning sub-score | **75.45** | 64.53 |
  | MMAU | **78.0** | 73.2 |

  Perception and translation saturate with data; reasoning and knowledge in speech track the backbone.
  - Exceptions: mini scores higher on ZH pro (69.57 vs 68.25) and on vocal-sound paralinguistics (76 vs 68).
- **vs the 130B predecessor Step-Audio-AQAA** (Tabs.2, 3, 6):

  | Metric | AQAA | Step-Audio 2 |
  |---|---|---|
  | Paralinguistics | 36.91 | 83.09 |
  | MMAU | 49.7 | 78.0 |
  | URO EN basic | 71.11 | 83.90 |

  The authors credit the latent (continuous) audio encoder and reasoning-centric RL (p.1). This is not controlled: data, size and post-training all changed.
- **Conversation data at scale.** S2S conversation is the single largest audio category: 150B of 400B audio tokens in the main phase and 36B of 100B in cooldown (derived 37.5% / 36%). The cooldown portion is synthesized with the ~50k-speaker library; the source of the main-phase portion is not stated (p.7).
- **Latency-aware RL ordering.** First constrain reasoning length (binary reward), then optimize quality (reward model), then perception (GRPO) (pp.8–9). No before/after numbers.
- **Detokenizer.** CNN-augmented flow matching improves pronunciation and timbre (p.7). No numbers.

## Implications for our setup

1. **A token budget we can scale down.** Even at 1.356T tokens, StepFun spent **50% on text** in every post-alignment phase. Inside the audio half they explicitly included **continuation tasks**:
   - speech-to-text continuation: 5B;
   - utterance-level text-speech interleaved continuation: 16B + 45B;
   - interleaved conversation: 15B.

   These appear aimed at our weak spot, audio continuation (the paper does not define their formats). Our 10K h duplex corpus has none of this structure. Building interleaved-continuation and TTS samples from it, and from public ASR corpora, is cheap.
2. **Alignment first, with the LLM frozen.** Train the adaptor only, on ASR. Then unfreeze with higher LRs on new embedding and output rows (p.7).
3. **Text-led output.** Interleaving text with audio tokens lets the text LLM's knowledge drive speech. This is the opposite of making the LLM predict audio alone.
4. **The backbone sets the ceiling for knowledge in speech.** On the same data, the 7B mini (which also swaps the encoder) loses ~16 points on EN spoken reasoning but almost nothing on ASR or translation. If our Qwen3 backbone is small, expect that ceiling; use the largest Qwen3 that meets the latency budget.
5. **RL recipe.** Run cheap length-control RL first, then RM-based PPO, then GRPO for perception. Duplex itself must come from elsewhere (StepAudio 3 Realtime and Chronological Thinking).

## Open questions / unclear points

- **Model details missing:** backbone identity and size, the text:audio interleave ratio, and the audio-token rate (not restated here) are not given.
- **Data and compute missing:** hardware, the real vs synthetic split of the 8M hours, and the 680B-vs-628B text-token gap are not explained.
- **Text retention is unverified:** no text benchmarks are reported.
- **RL details missing:** reward-model data and architecture, RL prompt counts and KL settings; and the PPO/GRPO citation points to DPO.
- **Undefined tasks.** The formats of "utterance-level text-speech interleaved continuation" and "speech-to-text continuation" are not defined.
