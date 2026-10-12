# Full-duplex S2S training: research notes

**Goal.** Plan mid-training and post-training for a full-duplex speech-to-speech model built from open-source Qwen3 components: the LLM backbone, the audio tokenizer and the vocoder/decoder. The current data is ~10K h of duplex conversational speech. Early runs show weak audio continuation and weak semantic knowledge in generated speech.

## Start here

1. **[synthesis.md](synthesis.md)**: what the literature says, organized around our two symptoms. Includes the ten key findings, the canonical pipeline, duplex mechanisms, frontier systems and Qwen3-specific implications.
2. **[plan.md](plan.md)**: draft v0 plan. Covers the intake questionnaire about our stack, cheap diagnostics, architecture options A–D, data buckets, the stage plan with compute estimates, the eval suite and a ranked experiment backlog.
3. **[training-recipes.md](training-recipes.md)**: comparison tables of architecture, data and compute per stage, text-preservation mechanisms, the S2T→S2S knowledge gap, duplex metrics and generation-head evidence.

## Paper notes (`papers/`)

Every note follows [`papers/_TEMPLATE.md`](papers/_TEMPLATE.md) and cites page and table numbers. Derived numbers are marked.

| Group | Notes |
|---|---|
| Priority full-duplex (2026) | [stepaudio3-realtime](papers/stepaudio3-realtime.md) · [nemotronlabs-voicechat](papers/nemotronlabs-voicechat.md) · [raon-speech](papers/raon-speech.md) · [minicpm-o-4.5](papers/minicpm-o-4.5.md) |
| StepFun lineage | [stepaudio3-gen](papers/stepaudio3-gen.md) · [step-audio-2](papers/step-audio-2.md) · [chronological-thinking](papers/chronological-thinking.md) |
| NVIDIA | [voicechat-tts](papers/voicechat-tts.md) · [personaplex](papers/personaplex.md) · [full-duplex-bench-v3](papers/full-duplex-bench-v3.md) · [audex](papers/audex.md) |
| Kyutai | [moshi](papers/moshi.md) · [multifaceted-interactivity-alignment](papers/multifaceted-interactivity-alignment.md) (GRPO for duplex) · [kyutai-dsm](papers/kyutai-dsm.md) · [kyutai-calm](papers/kyutai-calm.md) |
| Qwen (our base) | [qwen3-omni](papers/qwen3-omni.md) · [qwen3-tts](papers/qwen3-tts.md) (tokenizer details) · [qwen2.5-omni](papers/qwen2.5-omni.md) |
| Large-scale speech-text pretraining | [mimo-audio](papers/mimo-audio.md) · [kimi-audio](papers/kimi-audio.md) · [glm-4-voice](papers/glm-4-voice.md) · [covo-audio](papers/covo-audio.md) |
| Understanding / text preservation | [moss-audio](papers/moss-audio.md) · [minmo](papers/minmo.md) |
| Generation (TTS / codec) | [moss-tts](papers/moss-tts.md) · [voxcpm2](papers/voxcpm2.md) · [voxtral-tts](papers/voxtral-tts.md) · [vibevoice](papers/vibevoice.md) · [speechtokenizer](papers/speechtokenizer.md) |
| Earlier duplex systems | [syncllm](papers/syncllm.md) · [lslm](papers/lslm.md) · [freeze-omni](papers/freeze-omni.md) · [mini-omni2](papers/mini-omni2.md) · [flexduo](papers/flexduo.md) |
| Method | [multi-stream-llms](papers/multi-stream-llms.md) |

## Web notes (`web/`)

Claims in these notes are labelled [stated] (official source), [measured] (third-party) or [inferred].

| Topic | Note |
|---|---|
| OpenAI GPT-Live (incl. Desh Raj's "Dissecting GPT-Live" and the live-delegation docs) | [gpt-live](web/gpt-live.md) |
| Google Gemini 3.8 Live and the native-audio lineage | [gemini-live](web/gemini-live.md) |
| xAI Grok Voice | [xai-grok-voice](web/xai-grok-voice.md) |
| Sesame CSM | [sesame-csm](web/sesame-csm.md) |
| Nari Labs (serving VoiceChat-11B) | [nari-labs](web/nari-labs.md) |
| Breeze-TTS-2 (uses the Qwen3-TTS 12 Hz tokenizer) | [breeze-tts-2](web/breeze-tts-2.md) |
| Kyutai Pocket TTS / drifting (+ MoshiRAG, Hibiki-Zero) | [kyutai-pocket-tts-drifting](web/kyutai-pocket-tts-drifting.md) |
| Frontier systems, US/EU (Thinking Machines, Nova Sonic, Hume, LFM2-Audio, J-Moshi, HiPLEX, …) | [frontier-duplex-systems-west](web/frontier-duplex-systems-west.md) |
| Frontier systems, China/Asia academia (Qwen3.5-Omni, LongCat, Seed, AdaptDuplex, DuplexOmni, DuplexPO, MOSS-Speech, …) | [frontier-duplex-systems-china-academia](web/frontier-duplex-systems-china-academia.md) |
| Methodology evidence: scaling, interleaving, text retention, token design, duplex data, RL, architectural workarounds (~40 papers) | [speech-lm-training-methodology](web/speech-lm-training-methodology.md) |

## Other files in the repo

- `../papers/`: source PDFs, the GPT-Live MHTML, and `links.md`.
- `../index.html`, `../audio_model_comparison_v2.jsx`, `../vocoders.jsx`: an earlier interactive comparison of waveform-generation approaches.

## Conventions

- **n.r.** = not reported.
- **(d)** = derived by us.
- **[web]** in the synthesis = from web sources rather than a paper read in full.
- Accessed dates for web sources: 2026-10-12.
