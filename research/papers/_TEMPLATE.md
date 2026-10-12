# <Paper / model name>

| Field | Value |
|---|---|
| Org | |
| Date / arXiv | |
| Category | full-duplex S2S / half-duplex S2S / omni / audio understanding / TTS / codec / benchmark / method |
| Backbone | size; which LLM it was initialized from |
| Audio input | encoder or tokenizer; frame rate |
| Audio output | token type; frame rate; # codebooks; how multiple codebooks are predicted (depth transformer / delay pattern / MTP / flow head …) |
| Waveform decoder | |
| Duplex mechanism | n/a if half duplex |
| Open weights / code | |
| Relevance to us | high / medium / low, with a one-line reason |

## TL;DR

3–6 bullets.

## Architecture

## Training pipeline

One subsection per stage, in order. For each stage give: the objective; which modules are trainable and which are frozen; the data (type, size in hours and/or tokens, real vs synthetic, sources, mixture ratios including the text-only share); sequence length; optimizer, LR, batch size and steps; compute (GPUs × hours, tokens seen). Write "not reported" when a value is missing. Cite page numbers, e.g. (p.7) or (p.7, Tab.3). Mark derived numbers "(derived: …)".

### Stage 0: tokenizer / codec / decoder training
### Stage 1: …

## Data pipeline

How the data was collected, cleaned and labelled, and how synthetic dialogues were generated (especially duplex features: overlap, backchannels, interruptions, silence).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|

## Preserving text intelligence / knowledge

## Key results

Only the numbers that matter for comparison. Name each benchmark.

## Ablations & lessons

This is the most important section. One bullet per lesson, each with its evidence (number, table, page).

## Implications for our setup

Our setup: Qwen3 LLM backbone + Qwen3 audio tokenizer + its vocoder/decoder, mid-trained on ~10K h of duplex conversational speech. The current problems are weak audio-continuation quality and weak semantic knowledge in generated speech.

## Open questions / unclear points
