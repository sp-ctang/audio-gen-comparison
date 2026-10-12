# Breeze TTS 2 (BreezeBlue): open-weight 3B TTS on a Qwen3 backbone with the Qwen3-TTS 12 Hz tokenizer

Labels: **[stated]** comes from an official BreezeBlue source, including its published config files. **[measured]** is a third-party evaluation. **[inferred]** is an inference, and the tag says whose.

## Summary

- **Who made it.** BreezeBlue, whose legal licensor is RESONIA, INC. of Delaware; its HF org lists 10 members [stated]. Despite the name, nothing links it to MediaTek Research's "Breeze" family [inferred: me].
- **Release [stated].** Benchmarks and blog on 7 Aug 2026; weights and code on 25 Aug 2026.
- **The model [stated].**
  - 3B parameters, English and Chinese.
  - Three modes: voice clone (reference audio plus transcript), voice design (natural-language description, no reference audio) and voice direction (clone plus delivery instructions).
  - Inline vocal events such as `(laugh)` and `(sigh)`.
- **Speed and memory on an H100 [stated].** Time to first audio under 40 ms and a real-time factor of 0.32. It needs about 7.7 GiB eager, or 14.4 GiB on the CUDA-graph fast path.
- **Architecture, from the shipped configs** [stated configs; reading inferred: me]. It is a **CSM-style model**:
  - The backbone has exactly Qwen3-1.7B's shape and predicts codebook 0.
  - A 12-layer CSM-shaped depth decoder predicts the other codebooks.
  - A T5Gemma-2 text encoder conditions on text and instructions.
  - The audio tokenizer is **Qwen3-TTS-Tokenizer-12Hz** (12.5 Hz, 16 codebooks, 24 kHz).
  - If our tokenizer is Qwen3-TTS-Tokenizer-12Hz, this is essentially our component stack, used for TTS.
- **Not disclosed.** Training data, hours, stages and RL. There is no technical report.
- **Evals.**
  - #1 open-weights in the Artificial Analysis (AA) Speech Arena at about 1,215 Elo (±17), #6 overall [measured: AA, via orcarouter].
  - Leads BreezeBlue's own voice-design and voice-direction benchmarks [stated, vendor-run].
- **Licence.** Weights are research/non-commercial only, and the licence explicitly bans using the model *or its outputs* to train, distill or improve other models (narrow research/non-commercial exception) [stated].

## Sources (accessed 2026-10-12)

**Model card and configs**
- https://huggingface.co/BreezeBlue/Breeze-TTS-2
- https://huggingface.co/BreezeBlue/Breeze-TTS-2/tree/main
- https://huggingface.co/BreezeBlue/Breeze-TTS-2/raw/main/config.json
- https://huggingface.co/BreezeBlue/Breeze-TTS-2/raw/main/audio_tokenizer/config.json
- https://huggingface.co/BreezeBlue/Breeze-TTS-2/raw/main/LICENSE (v1.1, 1 Sep 2026)
- HF discussion: https://huggingface.co/BreezeBlue/Breeze-TTS-2/discussions/1
- HF org page: https://huggingface.co/BreezeBlue

**BreezeBlue**
- Code: https://github.com/breezeblue-ai/breeze-tts
- Blog: https://breezeblue.ai/breeze-tts-2
- Benchmarks:
  - https://breezeblue.ai/benchmarks/tts-voice-design
  - https://breezeblue.ai/benchmarks/tts-voice-direction
  - https://breezeblue.ai/benchmarks/tts-latency
- Benchmark datasets:
  - https://huggingface.co/datasets/BreezeBlue/TTS-Voice-Design-Benchmark
  - https://huggingface.co/datasets/BreezeBlue/TTS-Voice-Direction-Benchmark
- Paper entry, a blog mirror only: https://paperswithcode.co/paper/109782

**Configs used for comparison**
- Qwen3-1.7B: https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json
- Qwen3-TTS: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/config.json
- CSM-1B, via mirror: https://huggingface.co/unsloth/csm-1b/raw/main/config.json
- Qwen3-TTS report: https://arxiv.org/abs/2601.15621

**Third-party**
- Arena Elo: https://www.orcarouter.ai/blog/breeze-tts-2-tops-open-weights-speech-arena
- https://www.mindstudio.ai/blog/breeze-tts-2-open-weight-model
- AA leaderboard: https://artificialanalysis.ai/text-to-speech/leaderboard (seen only via search snippets)
- Unrelated MediaTek model with a similar name: https://huggingface.co/MediaTek-Research/BreezyVoice

**Could not find**
- Any technical report or training description.
- A primary AA leaderboard row (JS-rendered), so Elo values come from secondary reports.
- A syndicated press release (HTTP 404).

## What it is

**Positioning [stated].** A real-time, expressive TTS aimed at games, companions, storytelling and voice agents.

**Languages [stated, with a conflict].**
- The open model is bilingual, English and Chinese, in one checkpoint.
- The blog advertises "51 languages", which appears to describe the hosted platform.

**Interfaces [stated].**
- `infer.py` for the command line.
- A single-concurrency streaming API in `breeze_infer.api`, outputting 24 kHz mono 16-bit PCM.
- An HF ZeroGPU demo space.

**Repository [stated].**
- Two safetensors shards, 4.96 GB + 2 GB, stored in BF16/F32.
- An `audio_tokenizer/` folder holding a 682 MB codec.
- A 33.4 MB `tokenizer.json`.

## Architecture

The top-level class is `BreezeForConditionalGeneration` (transformers 4.57.3). The pipeline has four stages: text encoder, backbone prefill, backbone decode, then depth decoder and codec [stated]. Each part's config, set against the model it most resembles:

**Backbone** (`model_type: qwen3`) [stated config]

| Field | Breeze backbone | Qwen3-1.7B LLM | Qwen3-TTS-12Hz-1.7B talker |
|---|---|---|---|
| layers / hidden / FFN | 28 / 2048 / 6144 | 28 / 2048 / 6144 | 28 / 2048 / 6144 |
| heads / KV heads / head_dim | 16 / 8 / 128 | 16 / 8 / 128 | 16 / 8 / – |
| vocab | 151,936 | 151,936 | 3,072 codec + 151,936 text |
| max_position_embeddings | **40,960** | **40,960** | 32,768 |
| rope_theta / tied embeddings | 1e6 / true | 1e6 / true | 1e6 / – |

**Depth decoder** [stated config]

| Field | Breeze | CSM-1B decoder | Qwen3-TTS code predictor |
|---|---|---|---|
| layers | **12** | 4 | 5 |
| hidden / FFN | 1024 / 8192 | 1024 / 8192 | 1024 / – |
| heads / KV heads / head_dim | 8 / 2 / 128 | 8 / 2 / 128 | 16 / 8 / – |
| max positions / rope_theta | 33 / 500,000 | 33 / 500,000 | (max length 20) / – |
| per-codebook vocab | 2,051 | 2,051 | – |

**Text encoder [stated config].** `t5gemma2_text`: 26 layers, d=1152, 4 heads / 1 KV head, vocabulary 262,158. It adds:
- speaker tokens `[S0]`–`[S9]`
- instruction delimiters `<ins_bos>` / `<ins_eos>`
- an audio placeholder token (id 262,144) and an audio EOS token (id 262,145)

**Codec [stated config].** `audio_tokenizer/config.json` gives:
- `model_type: qwen3_tts_tokenizer_12hz`, `Qwen3TTSTokenizerV2Model`
- 24 kHz audio at 12.5 Hz
- encoder quantizers 32; decoder quantizers 16; semantic quantizers 1
- codebook size 2048; semantic codebook size 4096
- encoder and decoder transformers of 8 layers at d=512
- the model card credits "Qwen3-TTS by the Alibaba Qwen Team"
- the main config uses `num_codebooks: 16` and `audio_vocab_size: 2051`

**My reading of these configs [inferred: me]**
- **The backbone is Qwen3-1.7B.** Every field matches the Qwen3-1.7B LLM config, including `max_position_embeddings` 40,960, which differs from the Qwen3-TTS talker's 32,768. So it is almost certainly initialized from Qwen3-1.7B (or at least shaped like it), not fine-tuned from Qwen3-TTS. However, `backbone_model_name_or_path` is null, so the initialization is unconfirmed.
- **The model is built on HF's CSM template.**
  - The depth decoder is CSM-1B's decoder configuration with 12 layers instead of 4.
  - The main config still carries `codec_config._name_or_path: "kyutai/mimi"` and a `MimiModel` entry, which is residue of the CSM template, even though the shipped codec is Qwen3-TTS-12Hz.
- **Text goes through the encoder, not the backbone.** Text and instructions are tokenized with a Gemma-family tokenizer: the 33.4 MB `tokenizer.json` and the 262k vocabulary point that way. A T5Gemma-2 encoder, whose dimensions match Gemma-3-1B, then encodes them. How the backbone consumes the encoder output (prefix embeddings or cross-attention) is not documented. The backbone's Qwen text vocabulary may be vestigial.
- **The 3B count adds up.** Roughly 1.7B backbone + ~1B encoder + ~0.4B depth decoder, heads and embeddings ≈ 3B, consistent with the stated 3B and a ~7 GB BF16-mixed checkpoint.
- **Comparison with Qwen3-TTS.** Breeze keeps the Qwen3-TTS frame structure (12.5 Hz, codebook 0 plus 15 residual codebooks). It spends about 2.4× more depth on the residual predictor than Qwen3-TTS (12 vs 5 layers), and it adds a separate pretrained text encoder.

## Training data & compute

**Nothing disclosed [stated by omission].** The model card, README, blog and benchmark pages give no hours, sources, language mix, training stages, compute or RL/DPO details.

**Background on the reused tokenizer [stated: Qwen3-TTS report, Jan 2026].**
- Qwen3-TTS trained on "over 5 million hours of speech data spanning 10 languages".
- Its 12 Hz tokenizer is a 16-layer multi-codebook design with "immediate first-packet emission (97 ms)".
- Both tokenizers are Apache-2.0.

## Post-training / alignment

**Not described.**
- Voice design from a description, and direction such as accent, emotion, "physiological state" or role acting, imply instruction-tuning data that pairs speech with natural-language style descriptions. Whether any preference or RL stage was used is unknown [inferred: me].
- The licence restricts cloning real voices without consent [stated]. The model card describes no technical safeguards such as watermarking [stated by omission].

## Evals & numbers

**AA Speech Arena, "Provider Voices" (late Aug 2026) [measured: AA, via orcarouter and search-result snippets].**
- Breeze TTS 2: 1,215 Elo ±17, #6 overall and #1 open-weights.
- Open-weights rivals: Fish Audio S2 Pro 1,125 (another report: 1,121), Mistral Voxtral 1,010 on the controlled-voice arena.
- Proprietary systems:

| Model | Elo |
|---|---|
| Cartesia Sonic 3.6 | 1,283 |
| Qwen-Audio-3.0-TTS-Plus | 1,238 |
| ElevenLabs Eleven v3 | 1,177 |

- A later secondary snapshot shows Breeze at 1,205 and #7.

**Voice Design benchmark (vendor-run) [stated].** 1,000 English tasks across 29 media types.
- Role Fit: reviewers rate age, texture, temperament and delivery from 1 to 5; clips that fail a transcript gate are excluded. Whether the reviewers are human or a model is not specified.
- Transcript Pass: whether the clip clearly speaks its intended line.
- Diversity: WavLM embeddings with average-linkage clustering.

| Model | Role Fit | Voice diversity | Transcript pass |
|---|---|---|---|
| Breeze TTS 2 | 78.02 | 90.8% | 98.5% |
| MiMo-V2.5-TTS | 72.78 | 90.2% | 96.3% |
| Inworld Voice Design | 64.61 | 90.3% | 98.2% |
| Fun-AudioGen-VD | 63.82 | 90.1% | 98.6% |
| Eleven v3 | 58.88 | 90.1% | 99.7% |
| Qwen3-TTS-VD | 58.48 | 90.8% | 99.6% |
| VoxCPM2 | 55.28 | 90.5% | 98.5% |

The blog instead quotes "Voice Diversity (708)" and "39% more distinct voices" than the next model, so the leaderboard and blog use different diversity readouts [stated; inconsistency noted: me].

**Voice Direction benchmark (vendor-run) [stated].**
- 700 English tasks: 25 synthetic reference voices, 9 direction categories.
- Voice Direction Score (VDS): a holistic 1–5 score; the judge type is not specified.
- Speaker similarity (SIM): WavLM-Large ECAPA cosine.

| Model | VDS | SIM |
|---|---|---|
| Breeze TTS 2 | 4.25 | 0.67 |
| MiMo-v2.5-TTS | 3.76 | 0.66 |
| StepAudio 2.5 TTS | 3.48 | 0.64 |
| Qwen-Audio 3.0-TTS-Plus | 3.33 | 0.65 |
| Inworld TTS-2 | 3.01 | 0.71 |
| VoxCPM2 | 2.49 | 0.70 |

Breeze wins on following direction but not on raw speaker similarity [inferred: me, from the vendor table].

**Hosted latency (vendor-run) [stated].**
- Method: the 40 longest Seed-TTS-Eval prompts, serial requests from 3 US regions, 3 warm-up requests.
- Breeze was tested at its highest request priority.

| Provider | TTFA p50 | TTFA p95 |
|---|---|---|
| Breeze TTS 2 | 127.2 ms | 159.3 ms |
| Speechify Simba 3.2 | 82.5 ms | 126.9 ms |
| ElevenLabs Flash v2.5 | 143.6 ms | 191.6 ms |
| xAI TTS | 289.7 ms | 351.3 ms |
| ElevenLabs v3 | 602.1 ms | 753.4 ms |

Locally on an H100, TTFA is under 40 ms and RTF 0.32 [stated].

**No standard objective metrics are published** [stated by omission]: no Seed-TTS-eval WER/CER, no SIM on standard sets, no MOS.

## Lessons for our setup

1. **This is the closest public analogue of our component stack.** A Qwen3-1.7B-shaped LLM backbone plus Qwen3-TTS-Tokenizer-12Hz and its decoder reach the top open-weight naturalness in a blind arena. So the Qwen3 tokenizer and vocoder are not a naturalness ceiling. Our weak continuation quality more likely comes from modeling capacity, data or decoding than from the codec [inferred: me].
2. **Spend capacity on the residual-codebook predictor.** Breeze uses a 12-layer depth decoder, against 4 in CSM and 5 in Qwen3-TTS. If our continuations sound smeared or unstable while the semantic content is right, try a deeper code predictor before scaling the backbone [inferred: me].
3. **The CSM recipe ports to Qwen3 parts.** Breeze appears to be HF's `CsmForConditionalGeneration` template adapted to the Qwen3 backbone and tokenizer. That template already supports training with backbone-only frames (label `-101`) and `depth_decoder_labels_ratio` amortization (see `sesame-csm.md`). It is a quick way to prototype the backbone-plus-depth-decoder split on our components [inferred: me].
4. **Use a dedicated text-understanding module.** Breeze conditions on a pretrained T5Gemma-2 encoder rather than relying on the audio backbone for text semantics. In our duplex setting the analogue is keeping a strong text pathway (a text channel or thinker) so that knowledge does not have to survive audio-token training [inferred: me].
5. **Borrow its evals for expressivity.** The 1,000-task voice-design and 700-task voice-direction sets are public and include a WavLM-based diversity and SIM protocol. They are useful for measuring style control in generated speech. Check the dataset licences first [inferred: me].
6. **Licensing.** Breeze outputs cannot be used to create training data for our model, except for a research/non-commercial derivative. Do not use it as a synthetic-data generator for a commercial model [stated].

## Open questions

- Was the backbone initialized from Qwen3-1.7B weights, from scratch, or from another checkpoint? Was the T5Gemma-2 encoder frozen?
- How many hours were used, in which languages, how much was conversational or expressive data, and how much was synthetic? Were there separate pretraining, SFT and RL stages for design and direction?
- Why does the codec encoder have 32 quantizers when only 16 are modeled? How is the 4096-entry semantic codebook used?
- Who rated Role Fit and VDS (humans or an LLM/audio judge), and how many raters were there?
- Is the "51 languages" claim a hosted-model feature beyond the open checkpoint?
- Relation to MediaTek Research: none found. MediaTek's BreezyVoice (Jan 2025) is a CosyVoice-derived Taiwanese-Mandarin TTS with bopomofo control, Apache-2.0 [stated: HF card], an unrelated line.
