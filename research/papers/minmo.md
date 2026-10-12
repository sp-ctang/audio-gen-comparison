# MinMo: A Multimodal Large Language Model for Seamless Voice Interaction

| Field | Value |
|---|---|
| Org | FunAudioLLM Team, Tongyi Lab, Alibaba Group |
| Date / arXiv | arXiv 2501.06282v1 (10 Jan 2025) |
| Category | full-duplex S2S of the "aligned" type: a text LLM plus a streaming voice decoder plus an external duplex (turn-taking) predictor. The LLM core is turn-based; full duplex comes from the predictor, and separate AEC and VAD modules are still needed (p.23) |
| Backbone | Qwen2.5-7B-Instruct, ~8B total. Never fully fine-tuned: frozen in most stages and LoRA-updated in one (p.7–8, p.10) |
| Audio input | SenseVoice-Large encoder (~636M), then an input projector (2 Transformer layers + 1 CNN layer with 2× downsampling, ~170M) (Tab.1, p.8). Frame rate not stated in this paper |
| Audio output | CosyVoice 2 speech tokens, generated autoregressively by a separate ~370M "Voice Token LM" initialized from CosyVoice 2's LM. Its input interleaves **5 "semantic vectors" : 15 speech tokens**. Semantic vectors are the Output Projector's outputs computed from the LLM's last-layer hidden states, the sampled text-token embeddings and user-input "query" embeddings. Token rate and codebook count are not stated in this paper (p.7) |
| Waveform decoder | CosyVoice 2's pretrained streaming token2wav: chunk-aware flow matching (tokens → mel) plus a mel vocoder, working in chunks of 15 tokens (p.7) |
| Duplex mechanism | "Full Duplex Predictor": 1 Transformer layer + a linear-softmax head (~18M, randomly initialized). It reads LLM hidden embeddings and predicts whether to respond, or to stop the ongoing output and listen. It is the only module trained in the duplex stage, on 4K h (p.7, p.11) |
| Open weights / code | "will be released soon" (p.1); project page funaudiollm.github.io/minmo |
| Relevance to us | **Medium.** A useful counterpoint: duplex behavior and good spoken QA with the LLM's weights barely touched, plus a cheap recipe for duplex labels from real conversations. Its architecture (text-first, half-duplex core) differs from a native single-model duplex S2S |

## TL;DR

- About 8B parameters: Qwen2.5-7B-Instruct, a SenseVoice-Large encoder, a CosyVoice 2-based streaming voice decoder and a duplex predictor. Trained in **4 alignment stages** (speech→text, text→speech, speech→speech, duplex) on **~1.4M h**. The Tab.2 sum is 1,443.5K h (derived): 1,258.4K h speech-to-text, 171K h TTS, 10.1K h S2S and 4K h duplex (p.1, p.8).
- **The LLM is protected by never being fully trained.** It is frozen during encoder alignment, LoRA-updated during S2T instruction tuning, and frozen during all speech-output and duplex stages. No text-only replay is described, and no text benchmarks (MMLU etc.) are reported. The evidence for "minimal loss" is LLM-judged spoken chat: Alpaca 6.48 against ASR+Qwen2.5 6.59, and ChitChat 7.20 against 7.18 (Tab.20).
- **Speech is rendered from text by a separate pretrained TTS LM** fed with LLM hidden states. Spoken QA still loses **11–15 points** going from S2T to S2S: Llama Q 78.9 → 64.1, Web Q 55.0 → 39.9 (Tab.19).
- **Duplex runs on 4K h:** ≈3K h of real conversations (AliMeeting, Fisher, in-house) plus ≈1K h simulated, labelled with simple heuristics. Results: turn-taking positive F1@K=10 of 0.80–1.0, back-channel accuracy 0.71–0.81, and latency of ~600 ms in theory and ~800 ms in practice (p.1, p.10, Tab.21–22).
- No optimizer, LR, batch, step or GPU numbers are reported anywhere.

## Architecture

- **Path:** Voice Encoder → Input Projector → LLM, with LoRA in one stage (Fig.3, Tab.1). The LLM produces text. **For every 5 text tokens**, those tokens and their last-layer hidden states go through the Output Projector (a ~6M linear layer) to the Voice Token LM, which emits 15 speech tokens. Token2wav then synthesizes each 15-token chunk in streaming fashion (p.7).
- **Voice decoder input (p.7).**
  - Per turn, the user-input embeddings and the LLM's last-layer hidden states are concatenated along the feature dimension to form "query embeddings".
  - These are concatenated along the sequence with the 5 sampled-token embeddings and their hidden states, then projected into "semantic vectors".
  - Rationale: hidden states carry context and instruction (style) information; the sampled tokens are "more precise and consistent with the generated text".
- **Decoder rhythm.** The sequence alternates 5 semantic vectors with 15 speech tokens and is trained with teacher forcing. When the text ends, a "turn of speech" token switches to speech-only generation, which runs until "end of speech" (p.7).
- **Latency (p.8, p.22, Tab.21; two L20 GPUs, BF16).** Decoder latency is 5·d_llm + 15·d_lm + 15·d_syn. Full-duplex assistant turn-taking costs 250 ms (predictor) + 150 ms (first 5 text tokens; 95 ms for 1) + 70 ms (first 15 speech tokens) + 130 ms (token2wav first packet) ≈ **600 ms**. Practice is ≈800 ms, and speech-to-text latency is ≈100 ms (p.1).

## Training pipeline

The modules trained at each stage are given in the stage subsections below. **Hyperparameters (LR, batch, steps, sequence length, LoRA rank) and compute are not reported for any stage.**

### Stage 0: tokenizer / codec / decoder training
None of this is new. The SenseVoice-Large encoder, the CosyVoice 2 LM, the flow-matching model and the vocoder are all pretrained (p.6–7). For data generation the team also fine-tuned CosyVoice's base model on 2 h of one selected speaker ("CosyVoice-SFT"), which became the assistant voice (p.9).

### Stage 1: speech-to-text alignment (p.10)
- **1a, Pre-align.** Only the randomly initialized Input Projector trains, on about 1/10 of the Full-Align data. This "prevents the randomly initialized parameters from having large-gradient influences on the pre-trained Voice Encoder."
- **1b, Full-Align.** Input Projector + Voice Encoder train on all S2T data, about 1.2M h covering ASR, S2TT, LID, contextual biasing, SER, AED, speaker analysis, spoken-language smoothing and S2T chat. **The LLM is frozen.**
- **1c, SFT.** About **1.3M samples**, with the **LLM updated via LoRA**, to teach instruction following. Mixture proportions for 1b and 1c appear in Fig.4 (pie charts) and are not recoverable from the text.

### Stage 2: text-to-speech alignment (p.11)
First the Output Projector trains alone; then the Output Projector and Voice Token LM train jointly. **Everything else is frozen.** Data: 170K h of TTS (the CosyVoice 2 training data, in zh/en/ko/ja) plus ~1K h of instruction-controlled TTS (p.9, p.11).

### Stage 3: speech-to-speech alignment (p.11)
Same trainable modules, with the LLM still frozen. Data: about 10K h of paired multi-turn S2S dialogue, plus style-controlled dialogues covering dialect, speaking rate and emotion. "Even without updating the LLM, just by leveraging embeddings aligned with a small-scale instruction dataset (<150 hours), the large model can still learn quite effective control capabilities" (p.11).

### Stage 4: duplex interaction alignment (p.11)
**Only the Full Duplex Predictor trains**, on 4K h. It decides (1) whether to respond to the current user query, and (2) whether to stop ongoing audio output to listen. Prediction granularity and chunk size are not reported.

## Data pipeline

- **Hours by task (Tab.2, p.8):**
  - **S2T:** ASR 630K, S2TT 451K, LID 34K, contextual-bias ASR 50K, SER 48K, AED 11K, speaker analysis 24K, spoken-language smoothing 0.4K, S2T chat 10K.
  - **TTS:** speech synthesis 170K; instruct TTS 1K.
  - **S2S:** chat 10K; style-controllable chat 0.1K.
  - **Speech-to-ControlToken:** full duplex 4K.
- **S2T chat** (ChatML format with the audio inside the user turn, p.9). Two sources: open-source multi-turn text data with user turns synthesized by zero-shot TTS, and real ASR utterances used as queries with LLM-generated text responses (p.20).
- **Instruct TTS.** Human-labelled attributes (emotion, rate, dialect, role-play) are expanded into natural-language instructions by Qwen-Max. Outputs are prefixed with a style tag, e.g. "(Sad) …" (p.9, Tab.3).
- **S2S chat (~10K h, all synthetic speech; p.9–10):**
  - Alpaca and ShareGPT text chats. User turns use CosyVoice zero-shot cloning for voice diversity; assistant turns use CosyVoice-SFT (one fixed voice).
  - Real ASR speech as user queries, with Qwen-Max response text rendered by CosyVoice-SFT, for robustness to real input audio.
  - Style data (~100 h). Qwen-Max writes style-controllable multi-turn dialogues. User turns use CosyVoice zero-shot; assistant turns use CosyVoice 2 with instruction prompts plus a small recorded expressive corpus used as zero-shot prompts.
- **Duplex (~4K h; p.10).**
  - Sources: ≈3,000 h of real voice interaction (AliMeeting, Fisher, in-house) and ≈1,000 h simulated from the open MOSS dataset and synthesized in-house text dialogues.
  - Labels are heuristic:
    - *Assistant turn-taking* starts at the end of the user's turn.
    - *User turn-taking* starts a gap T after the assistant's turn ends, with T ~ N(0.6, 0.4²). The text extraction reads "N(0.6, 0.42)"; units are presumably seconds.
    - *User back-channel* samples are moments when the speaker designated as user "is unable to interrupt the other speaker".
  - Overlap is not modelled in the LLM; only the predictor sees these labels.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| 1a Pre-align | ~1/10 of S2T data (≈126K h, derived) | not reported | projector only |
| 1b Full-Align | ≈1.26M h S2T (Tab.2) | not reported | encoder + projector |
| 1c SFT | ~1.3M samples | not reported | LoRA on LLM |
| 2 TTS align | 171K h | not reported | output projector + voice-token LM |
| 3 S2S align | ~10.1K h | not reported | same modules |
| 4 Duplex | 4K h | not reported | predictor only (~18M) |
| Inference | — | 2× L20, BF16 | ~600 ms theoretical duplex latency |

## Preserving text intelligence / knowledge

- **Method:** stepwise freezing. The LLM is frozen in Full-Align and in stages 2–4, and gets LoRA only during S2T SFT (p.10–11). S2T chat data (10K h) is included explicitly "to transfer the dialog capabilities of the base model to the speech modality" (p.20). **No text-only replay data is mentioned.**
- **Evidence:** Tab.20, with Qwen-Max judging on a 0–10 scale:

  | | Alpaca (reasoning) | ChitChat |
  |---|---|---|
  | Ground truth | 7.73 | 7.62 |
  | ASR + Qwen2.5 cascade | 6.59 | 7.18 |
  | MinMo | 6.48 | 7.20 |

  Spoken-language smoothing is on par with Qwen2.5-7B, though that task was fully fine-tuned (Tab.15). **No MMLU-style text benchmarks are reported.**
- **Self-reported erosion.** "Integration of multiple speech tasks and … LoRA training … have somewhat diminished the logical generation capabilities" of the LLM, with a bigger drop on Alpaca than ChitChat (p.20). Limitations add that LoRA-only updates leave instruction following weak, and suggest "more high-quality text data for more comprehensive updates" (p.23).

## Key results

- **Spoken QA, S2T / S2S accuracy** (Tab.19):

  | Model | Llama Q | TriviaQA* | Web Q |
  |---|---|---|---|
  | MinMo | 78.9 / 64.1 | 48.3 / 37.5 | 55.0 / 39.9 |
  | GLM-4-Voice | 64.7 / 50.7 | 39.1 / 26.5 | 32.2 / 15.9 |
  | Moshi | 62.3 / 21 | 22.8 / 7.3 | 26.6 / 9.2 |
  | Freeze-Omni (S2T only) | 72.0 | 53.9 | 44.7 |

  S2S answers are scored by ASR; TriviaQA* is a non-standard 1,000-sample subset.
- **Duplex predictor, positive F1 at K = 1 / 5 / 10, and back-channel accuracy** (Tab.22):

  | Test set | Assistant turn-taking | User turn-taking | Back-channel acc. |
  |---|---|---|---|
  | AliMeeting | .61 / .75 / .80 | .48 / .94 / 1.0 | .71 |
  | Fisher | .67 / .84 / .88 | .43 / .95 / .999 | .81 |
  | Simulation | .79 / .96 / .985 | .26 / .82 / .99 | – |

- **Duplex latency (Tab.23).** Tab.21 and Tab.23 put assistant turn-taking at ≈250 ms on average (88.8–448.8 ms) and user turn-taking at ≈660 ms. The p.21 prose swaps these labels.
- **TTS on Seed-TTS-Eval ("Copy:" prompt), Tab.17:** zh CER 2.48 / NMOS 3.69 and en WER 2.90 / NMOS 3.56, against CosyVoice 2-SFT at 2.06 / 3.73 and 3.19 / 3.71. Only the ~20% of cases where the LLM copied the text exactly were scored (p.19).
- **Instruction-following voice generation** (in-house, 122 turns): 98.4% overall, against 63.1% for GLM-4-Voice (Tab.18).
- **Understanding:**
  - ASR: Fleurs 10-language average 4.13, Common Voice 7.37, LibriSpeech clean/other 1.74/3.89 (Tab.5).
  - S2TT: Fleurs xx→en 29.13 BLEU, CoVoST2 38.92 (Tab.6).
  - LID: Fleurs 85.3% (Tab.7).

## Ablations & lessons

There are almost no controlled ablations. The lessons come from design choices and observations.

- **Text-first rendering still loses knowledge in speech.** The S2T→S2S drop is 14.8 (Llama Q), 10.8 (TriviaQA) and 15.1 (Web Q) points (derived, Tab.19). The authors blame TTS of structured or specialized-vocabulary answers, plus ASR used in scoring (p.20).
- **A frozen LLM keeps casual chat quality but loses some reasoning quality.** MinMo matches the cascade on ChitChat (7.20 vs 7.18) and is slightly below on Alpaca (6.48 vs 6.59) (Tab.20).
- **Style control needs very little data when the LLM is frozen.** Fewer than 150 h of style S2S data, training only the decoder-side modules, gave 98.4% instruction accuracy (p.11, Tab.18). Hidden states, not just text tokens, carry the style instruction to the decoder (p.7).
- **Align the projector before unfreezing the encoder.** Pre-align on 1/10 of the data protects the pretrained encoder from gradients through randomly initialized projector weights (p.10). The paper makes this claim without numbers.
- **The LLM's text faithfulness limits the speech path.** Asked to "Copy:" text, the LLM reproduced it exactly in only ~20% of cases, even with teacher forcing (p.19). The limitations cite long-tail pronunciation errors from one-to-many LLM tokens and special symbols. They also note that conditioning the decoder only on hidden embeddings "restricts the transmission of historical information" (p.23).
- **Duplex trade-off.**
  - The predictor is tuned to be very sensitive to user speech: user turn-taking F1@K=5 is 0.94–0.95 on real data. The cost is in back-channel accuracy, which reaches only 0.71–0.81: "a certain trade-off between the two" (p.21).
  - Assistant turn-taking on real human–human data (F1@K=1 0.61–0.67) is much worse than on simulated data (0.79), attributed to noise, rate and pause variability (p.21).
- **The semantic duplex predictor dominates latency:** 250 ms of the 600 ms budget (Tab.21).

## Implications for our setup

- **Duplex behavior without exposing the LLM to duplex data.** MinMo trains only an 18M head on frozen LLM states, from 4K h, so duplex training cannot hurt the LLM's knowledge by construction. Our approach (full mid-training on 10K h of duplex speech) puts all the duplex learning into the backbone, which is the likely source of knowledge loss. Middle-ground options: freeze the backbone or use LoRA or low LR in the duplex stage, add light duplex or turn-taking heads, and heavily replay text.
- **10K h of dialogue speech was used here only to align a speech renderer, never the LLM.** MinMo's ~10K h of S2S data (all TTS-synthesized from text chats) trained only the output projector and voice-token LM; content comes from the frozen LLM. This shows ~10K h is enough to align speech output to a frozen LLM. It says nothing about whether 10K h can teach content: by design that data never touches the LLM.
- **Continuation quality came from a dedicated, large-scale speech generator.** MinMo leans on CosyVoice 2's LM (trained on the 170K h TTS set) and still drops 11–15 points from S2T to S2S. If our LLM must generate Qwen3 tokenizer tokens itself, we need comparable speech-generation data (≥10⁵ h), or a fallback: a pretrained talker/TTS LM fed with LLM hidden states and text tokens.
- **Reusable duplex labelling recipe for real two-speaker corpora** such as Fisher, AliMeeting and in-house data:
  - assistant turn start = the other speaker's turn end;
  - barge-in timing gap ~ N(0.6 s, 0.4²);
  - back-channel examples = failed interruptions.

  These could annotate our 10K h with control or turn labels at no cost.
- **Evaluate the S2T/S2S gap explicitly.** Run spoken QA (Llama Q, Web Q, TriviaQA) in both modes with ASR scoring, so we can separate knowledge loss in the LLM from loss in speech rendering.
- **Expect real-conversation turn-taking to be harder than simulated.** Keep real human–human data in the duplex mix (MinMo used ≈3K h real out of 4K).

## Open questions / unclear points

- No LR, batch, steps, sequence length, LoRA rank/targets, GPU count or training time for any stage.
- Fig.4 mixture proportions for Full-Align and SFT are not recoverable from the text. The SFT sample count (1.3M) cannot be converted to hours.
- Encoder frame rate and CosyVoice 2 token rate and codebooks are not stated in this paper.
- p.11 calls the duplex set "4,000 hours of long-form human-human spoken conversation", but p.10 says ≈1,000 h is simulated from text dialogues.
- How the predictor runs in time is not described: chunk size, how often it is called, and whether it sees the system's own speech while speaking. Separate AEC and VAD are required (p.23).
- The T-distribution parameters are ambiguous in extraction ("N(0.6, 0.42)"), and the latency labels conflict between Tab.23 and the p.21 prose.
- It is not stated which stage uses the 10K h S2T chat data; presumably Stage 1.
- There are no text-only benchmarks, so the "minimal loss" claim rests on two in-house LLM-judged sets.
