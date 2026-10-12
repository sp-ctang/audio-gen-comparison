# Mini-Omni2

| Field | Value |
|---|---|
| Org | Inspirai + Tsinghua University (p.1) |
| Date / arXiv | arXiv 2410.11190v3, 5 Nov 2024 (p.1) |
| Category | omni (vision + speech + text in; text + speech out), with command-based interruption |
| Backbone | Qwen2-0.5B base, ported to LitGPT. Vocab extended with 7 × 4,160 audio sub-heads to 181,120 entries (p.4, p.6). Derived: the text vocab is 152,000 (Fig.4 axis) |
| Audio input | Whisper-small encoder (frozen) → Llama-MLP adapter, continuous features (p.4, p.7). For interruption only, the incoming audio is SNAC-tokenized and fed as tokens (p.3, p.8) |
| Audio output | SNAC tokens, 7 layers, generated in parallel with text: 8 tokens per step, with a one-step delay between layers ("Text-Instruct Delay Parallel Decoding") (p.6). SNAC frame rate not reported |
| Waveform decoder | SNAC decoder (p.2, p.6) |
| Duplex mechanism | Command-based interruption. The model reads SNAC tokens of the incoming stream and emits a frame-level status token, irq or n-irq. On irq it stops speaking and starts listening (p.3, p.8) |
| Open weights / code | github.com/gpt-omni/mini-omni2. The paper promises to open-source all models and datasets (p.1, p.10) |
| Relevance to us | low: 0.5B, interruption-only duplex, almost no quantitative evaluation. A few design observations are still useful |

## TL;DR

- **A continuation of Mini-Omni.** Qwen2-0.5B plus frozen CLIP ViT-B/32 and frozen Whisper-small, with MLP adapters and parallel text + SNAC output (p.4–6).
- **Three training stages:** adapters only → LLM trained on text-output QA → LLM + adapters trained on audio output and interruption (p.7–8, Fig.5).
- **Text QA (Open-Orca) is mixed into every stage where the LLM is trainable**, to keep its reasoning ability (p.8).
- **Duplex is detection of the command "Stop Omni".** The authors judge full-duplex training not yet stable enough (p.3).
- **ASR is the only reported metric:** LibriSpeech test-clean / test-other WER 4.8 / 9.8 (p.9, Tab.2).

## Architecture

- **Vision:** CLIP ViT-B/32 gives 49 patch features + 1 global feature = 50 features, then a single-layer LlamaMLP adapter (p.4).
- **Audio:** Whisper-small encoder, then an adapter (p.4). All adapters are Llama-MLPs with intermediate size 4,864 (p.9).
- **Input layout** (p.6):
  - Visual features first, then Whisper features, then a special token where the response starts.
  - Single-modality features are replicated across layers 1–7 so they stay prominent when the per-layer embeddings are averaged.
- **Output** (p.6):
  - Per step: 1 text token + 7 SNAC layers, with the layers delayed one step from each other (Fig.3 staircase).
- **"Batch" trick** (p.6–7):
  - Two samples run together: one needs a text + audio answer, the other a text-only answer.
  - The text tokens of the first are replaced by the text output of the second.
  - The goal is to transfer text reasoning into audio responses.

## Training pipeline

All stages ran on 8×A100 with a cosine schedule, 1,500 warmup steps, global batch 192, and one epoch over each stage's full data (p.8–9). Steps and hours are not reported.

### Stage 0: tokenizer / codec / decoder training
None. SNAC, Whisper-small and CLIP are used off the shelf (p.2, p.4).

### Stage 1: multimodal encoder adaptation
- **Trainable:** adapters only. The LLM and the encoders are frozen (Fig.5).
- **Data:** ASR and image-caption data.
- **Goal:** make the multimodal features resemble text-token embeddings, and keep later changes to the LLM small (p.7).
- **LR:** 2e-5 to 1e-3 (p.8).

### Stage 2: modality alignment
- **Trainable:** the LLM. Adapters are frozen.
- **Data:** image and audio QA with text-only answers, plus text-to-text QA (p.7–8).
- **LR:** 2e-6 to 2e-4 (p.8).

### Stage 3: post-training (audio output + interruption)
- **Trainable:** the LLM and the adapters (Fig.5).
- **Data:** all stage 1–2 tasks, now with audio-token outputs, plus the interruption task (p.8).
- **LR:** 2e-6 to 2e-5 (p.8–9).

## Data pipeline

Datasets (Tab.1, p.9):

| Task | Stages | Datasets |
|---|---|---|
| ASR | 1–3 | LibriTTS 586 h, VCTK 44 h, MLS 8,000 h (total 8,630 h, derived) |
| Text QA | 2–3 | Open-Orca, 2,000K (the text says the "first 1.5 million" pairs, p.8) |
| Audio QA | 2–3 | Moss-002-sft, 1,500K, TTS'd |
| Visual QA | 2–3 | ALLaVA-4V, 800K (the text says 400,000, p.8) |
| Voice QA | final | Alpaca-GPT4 55k, identity 2k, QAassistant 27k, hh-rlhf 367k, Trivia single/multiple-choice 17k / 20k, OpenAssistant 2k |

VoiceAssistant-400K is also said to be used "continuously" (p.8), but it is not in Tab.1.

**Spoken-dialogue synthesis** (p.9):
- User side: CosyVoice zero-shot TTS, with a random voice prompt drawn from the ASR corpora.
- Assistant side: a single fixed voice from an internal TTS.

**Interruption data** (p.8–9):
- Noise sources: LibriTTS speech used as "human noise", and MUSAN (music, voices, white noise, urban noise).
- Construction: the noise is stream-encoded and decoded; a random segment is taken; "Stop Omni" in a random timbre is appended; then a 0–10 s tail is added.
- Labels: n-irq before the tail, irq on the tail.

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| 1–3 | per Tab.1; 1 epoch per stage | 8×A100; time not reported | global batch 192, 1,500 warmup |

## Preserving text intelligence / knowledge

- The adapter-only stage 1 is meant to minimize changes to the LLM core (p.7).
- Text-only QA is included whenever LLM weights are unfrozen (p.8), and a text-to-text task is added in stages 2–3 (p.7).
- Text comes first: stage 2 establishes reasoning with text answers; stage 3 then extends them to speech (p.7–8).
- The text answer drives the audio through the delayed parallel decoding and the batch trick (p.6–7).
- **Not measured:** no knowledge or QA benchmark is reported, so none of this is quantified.

## Key results

- **ASR** (Tab.2, p.9), LibriSpeech WER:

| Model | test-clean | test-other | dev-clean | dev-other |
|---|---|---|---|---|
| Mini-Omni2 | 4.8 | 9.8 | 4.7 | 9.4 |
| Mini-Omni | 4.5 | 9.7 | 4.6 | 9.2 |
| Whisper-small (authors' reproduction) | 4.4 | 10.1 | 4.6 | 10.3 |
| VITA | 8.14 | 18.41 | 7.57 | 16.57 |

- **Not reported:** spoken QA, speech-output quality, latency, interruption accuracy. Only case studies are shown (p.10).

## Ablations & lessons

There are no controlled ablations. What the authors observed:
- **They avoided discrete tokens for input understanding.** With token input, the audio loss stayed high during training, and the tokens for a segment changed substantially depending on the content at both ends. ASR from tokens was worse than from Whisper features (p.4). Whisper features even gave some Chinese understanding without any Chinese training (p.4).
- **Adding vision slightly hurt ASR** compared with Mini-Omni (test-clean 4.8 vs 4.5), which they attribute to the smaller share of audio data (p.9).
- **Putting semantic features (vision, then Whisper) before the generated response** reportedly trains more easily and gives more natural responses (p.6). This is qualitative.
- **Full-duplex training was not stable enough** in their hands, which is why they fell back to command-based interruption (p.3).

## Implications for our setup

- **Weak evidence for continuous features on the user stream.** If our listening/user stream is fed as codec tokens, Mini-Omni2's observation (high audio loss, context-dependent tokens, worse ASR) suggests trying continuous encoder features for input and keeping codec tokens only for output.
- **Check whether our duplex model has an inner text stream** (our inference, not tested in the paper). The text-leads-audio delay pattern is the standard way to let the LLM's text competence drive speech. A duplex model that predicts audio tokens without an inner text stream is a likely contributor to weak semantics.
- **Keep a text-only QA stream in every stage where the LLM trains**, and start with an adapter-only stage to limit early drift. This is their stated policy, unmeasured.
- **Not a reference for natural duplex behaviour.** It is command-only.

## Open questions / unclear points

- Whether training starts from Qwen2-0.5B or from the Mini-Omni checkpoint; the figures label the LM "Mini-Omni" (p.7).
- Dataset-size inconsistencies between text and table: Open-Orca 1.5M vs 2,000K; ALLaVA 400K vs 800K.
- SNAC frame rate, total tokens or steps, and interruption detection accuracy.
