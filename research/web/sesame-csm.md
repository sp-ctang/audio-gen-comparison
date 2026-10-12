# Sesame: Conversational Speech Model (CSM) and later releases

Labels: **[stated]** comes from an official Sesame source. **[measured]** is a third-party evaluation. **[inferred]** is an inference, and the tag says whose.

## Summary

- **What CSM is.** The 27 Feb 2025 post introduced CSM, a contextual speech generator built from two autoregressive Llama-architecture transformers [stated]:
  - A **backbone** reads interleaved text and audio and predicts Mimi codebook 0 for each frame.
  - A small **audio decoder** predicts codebooks 1…N-1 for that frame.
- **Training [stated].**
  - Three sizes: 1B backbone with 100M decoder, 3B with 250M, and 8B with 300M.
  - About 1M hours of "predominantly English" public audio, transcribed, diarized and segmented.
  - 5 epochs at sequence length 2048, about 2 minutes of audio.
- **Compute amortization [stated].** The decoder is trained on a random 1/16 of frames, while codebook 0 is trained on every frame. Sesame saw "no perceivable difference in audio decoder losses".
- **Evaluation [stated].** WER and speaker similarity (SIM) are near saturation. Sesame added two new evals, homograph disambiguation and pronunciation-continuation consistency, plus a CMOS study:
  - Without context, listeners showed no clear preference between CSM and real speech.
  - With 90 s of context, they still preferred real speech.
- **Self-declared limitations [stated].**
  - Weak outside English.
  - "does not take advantage of the information present in the weights of pre-trained language models".
  - "does not model the structure of the conversation itself".
  - The stated future direction is fully duplex models.
- **Open release (13 Mar 2025, Apache-2.0) [stated].** CSM-1B: a Llama-3.2-1B-shaped backbone plus a 4-layer 100M decoder over 32 Mimi codebooks. It "cannot generate text".
- **What came after [stated].** No new model paper or open model weights followed; TurnBench is a benchmark paper. The record is:
  - $250M Series B (Oct 2025)
  - iOS preview (May 2026)
  - **TurnBench** turn-taking benchmark (Aug 2026)
  - GA app with an "updated CSM" and tool-using agents (6 Oct 2026)
  - eyewear planned for 2027
- **For us.** The CSM split puts semantic codebook 0 on the big backbone and the residual codebooks on a small per-frame decoder trained on a subset of frames. It maps directly onto the Qwen3-TTS 12.5 Hz multi-codebook tokenizer, if that is the one we use. Breeze-TTS-2 does exactly this (see `breeze-tts-2.md`). CSM also shows that voice quality alone took about 1M hours, roughly 100× our data [inferred: me].

## Sources (accessed 2026-10-12)

**Primary**
- https://www.sesame.com/journal/crossing-the-uncanny-valley-of-voice

**Code and model**
- https://github.com/SesameAILabs/csm, including the raw `models.py` and `generator.py`
- https://huggingface.co/sesame/csm-1b (gated; config read through the mirror https://huggingface.co/unsloth/csm-1b)
- https://huggingface.co/docs/transformers/model_doc/csm

**Later Sesame posts**
- https://www.sesame.com/journal (index)
- https://www.sesame.com/journal/turnbench (28 Aug 2026)
- https://arxiv.org/abs/2608.25218 (the TurnBench paper)
- https://www.sesame.com/journal/voice-your-curiosity (27 May 2026)
- https://www.sesame.com/journal/a-new-form-of-eyewear (6 Oct 2026)

**Third-party**
- TechCrunch, open-source release: https://techcrunch.com/2025/03/13/sesame-the-startup-behind-the-viral-virtual-assistant-maya-open-sources-its-base-ai-model
- RoadtoVR, Series B: https://roadtovr.com/former-oculus-ai-smart-glasses-startup-sesame-raises-250m/
- Contrary Research: https://research.contrary.com/company/sesame-ai
- Speechmatics fine-tuning guide: https://blog.speechmatics.com/sesame-finetune
- testingcatalog: https://www.testingcatalog.com/sesame-updates-voice-agents-with-their-own-computers/
- Beebom review (search snippet): https://beebom.com/sesame-ai-voice-companion-maya-experience-like-talking-to-real-person/
- UMD talk abstract by a Sesame engineer: https://www.cs.umd.edu/class/spring2026/cmsc731/LEC/emre
- a16z podcast summary: https://getcoai.com/news/sesames-cto-reveals-how-theyre-building-real-time-voice-ai-that-talks-like-humans

**Could not access**
- Numeric values in the post's figures: they are images, and the text gives no numbers.
- aiwiki.ai (HTTP 429).
- The gated `sesame/csm-1b` config (HTTP 401). I used the unsloth mirror instead.

## What it is

**Purpose.** CSM is a "contextual TTS". It generates the next speaker's audio conditioned on the conversation's text and audio history, aiming for "voice presence". Sesame names four components of voice presence [stated]:
- emotional intelligence
- conversational dynamics
- contextual awareness
- consistent personality

**The demo.**
- The public demo voices, Maya and Miles, went viral in Feb–Mar 2025 [stated via TechCrunch].
- The README says CSM "cannot generate text" and suggests pairing it with a separate LLM [stated].
- Community analysis concluded the demo runs ASR, then an LLM, then CSM, with the conversation passed as context [inferred: r/LocalLLaMA commenters].
- Beebom reports that Sesame itself said the system is not truly full-duplex, because it processes your speech after you stop talking [stated via Beebom].

## Architecture

**Design [stated].**
- **Tokenizer.** Mimi split-RVQ at 12.5 Hz (80 ms frames): one semantic codebook, described as "speaker-invariant representations of semantic and phonetic features", plus N–1 acoustic codebooks.
- **No delay pattern.** Sesame rejects it because "time-to-first-audio scales poorly because an RVQ tokenizer with N codebooks requires N backbone steps before decoding the first audio chunk".
- **Division of labour.** Llama-tokenized text and audio are interleaved and fed to the backbone, which predicts codebook 0. The decoder uses one linear head per codebook to predict codebooks 1…N-1 for the same frame.

**Open CSM-1B, from the code and config [stated]**

| Part | Configuration |
|---|---|
| Backbone (`llama3_2_1B` flavor) | 16 layers, d=2048, FFN 8192, 32 heads / 8 KV heads, `max_seq_len=2048` |
| Decoder (`llama3_2_100M` flavor) | 4 layers, d=1024, FFN 8192, 8 heads / 2 KV heads |
| Audio vocabulary | 32 Mimi codebooks (`mimi.set_num_codebooks(32)`), 2,051 entries each |
| Link between them | a linear projection from backbone width to decoder width; `codebook0_head` on the backbone |
| Text side | Llama-3.2-1B tokenizer with a `[speaker]text` template |
| Sampling defaults | 80 ms per generation step; temperature 0.9, top-k 50; maximum 90 s of audio |

- Outputs carry an imperceptible watermark.
- HF lists about 2B total parameters, which includes the embeddings and heads.

**Training support in Transformers (v4.52.1, 20 May 2025) [stated].** The HF port supports training:
- Label value `-101` marks frames that train only the backbone, using the codebook-0 target.
- `depth_decoder_labels_ratio` sets the fraction of frames the depth decoder trains on. Together these implement the 1/16 amortization.

## Training data & compute

**Data [stated].**
- About 1M hours after filtering, from "publicly available audio" that Sesame transcribed, diarized and segmented.
- Predominantly English. Other languages appear only through "data contamination".
- Sources are not named. TechCrunch: "It's unclear what data Sesame used".

**Schedule [stated].** 5 epochs at sequence length 2048, about 2 minutes. Model sizes are Tiny 1B/100M, Small 3B/250M and Medium 8B/300M. GPU count and hours are not disclosed.

**Scale [inferred: me, arithmetic].**
- 1M h × 3,600 s × 12.5 Hz ≈ 45B audio frames per epoch, so about 225B backbone audio positions over 5 epochs.
- With 1/16 amortization, the decoder sees about 14B frames, or about 440B residual-codebook targets at 31 codebooks.
- For comparison, our 10K h is about 0.45B frames per channel per epoch.

**Initialization.** The post lists not using pretrained-LM weights as a limitation [stated]. So the Llama-shaped backbone was presumably trained from scratch [inferred: me].

**Amortization [stated].** Sesame says the 1/16 scheme "alleviates the memory bottleneck while preserving the fidelity of the full RVQ codebooks".

**Fine-tuning practice [stated: Speechmatics].** Speechmatics' public recipe keeps the 1/16 subset, with `decoder_loss_weight` 0.5 (loss = 0.5·c0 + 0.5·decoder), LR 3e-5 and weight decay 0.002. It warns that fine-tuning is "fairly sensitive to the choice of hyperparameters".

## Post-training / alignment

**CSM.** The post describes no RL or preference stage.

**Later product updates [stated].**
- **May 2026.** Agents were "trained" for conversational flow. They can "run multiple parallel searches while speaking" and can pivot mid-sentence.
- **Oct 2026.** The "updated" CSM has better expressivity and pronunciation, "particularly in long conversations". The release also claims "reduced sycophancy" and fewer repetitive phrases, with writers and producers shaping "voice and personality as one system".

No method or data is disclosed for any of these.

## Evals & numbers

**Objective evals [stated].**
- **WER and SIM.** Both are near-human or "saturated", so Sesame argues they no longer discriminate between systems.
- **Homograph disambiguation** ("text understanding"). 200 samples over 5 homographs (lead, bass, tear, wound, row), 2 variants each. Pronunciation is checked with the phoneme recognizer `wav2vec2-lv-60-espeak-cv-ft`.
- **Pronunciation-continuation consistency** ("audio understanding"). 200 samples over 10 words with regional variants, e.g. "route" as /raʊt/ or /ruːt/. It checks whether the model keeps the pronunciation the context established.
- **Baselines.** Play.ht, ElevenLabs and OpenAI were run with default voices. The plotted accuracies are not readable from the text.
- **Scaling.** "performance improves with larger models".

**CMOS [stated].**
- Set-up: the Expresso dataset; 80 paid raters, about 15 examples each.
- Without context: "no clear preference between generated and real speech".
- With 90 s of audio and text context: raters preferred the real recordings. The remaining gap is in contextual prosody.

**TurnBench (Aug 2026; arXiv 2608.25218, with academic co-authors including Hung-yi Lee and Shinji Watanabe) [stated].**
- **Data.** 30 h of dual-channel English dyadic speech: 154 dialogues by 106 voice actors in 53 pairs, recorded in separate booths. A 104-h training set was hand-labelled under the same protocol.
- **Labels.** Triple annotation. Four event types (Turn, Interruption, Backchannel, Mid-turn pause) and a 17-category taxonomy, with Fleiss' κ = 0.78.
- **Licence.** Non-commercial, and voice cloning is prohibited.

TurnBench results for selected systems, showing end-of-turn (EOT) and interruption (INT) recall and false-positive rate (FPR) [measured: Sesame running 14 systems]:

| System | EOT recall | EOT FPR | INT recall | INT FPR |
|---|---|---|---|---|
| OpenAI Realtime, server VAD | 0.955 | 0.525 | 0.990 | 0.458 |
| OpenAI Realtime, semantic VAD | 0.303 | 0.018 | 0.484 | 0.271 |
| Kyutai Semantic VAD | 0.773 | 0.059 | 0.898 | 0.081 |
| Voice Activity Projection | 0.845 | 0.055 | 0.945 | 0.107 |
| Smart Turn v3 | 0.752 | 0.047 | 0.107 | 0.093 |
| Gemini 3.1 Live | 0.657 | 0.022 | – | – |
| Moshi | 0.233 | 0.044 | – | – |

**TurnBench findings [stated].**
- In smooth floor transfers, humans start speaking a median 151 ms *before* the other turn ends. No system matches this without excessive false positives.
- The best system takes "nearly a second" to commit to an interruption.
- Sesame's own system is not on the board.

**Company facts [stated, via secondary reports].**
- Funding: Series A of $47.5M led by a16z (Feb 2025, per Contrary); Series B of $250M led by Spark and Sequoia (24 Oct 2025), for $307.6M in total (per RoadtoVR).
- Products: iOS preview in 39 countries (27 May 2026); iOS and Android GA (6 Oct 2026); glasses due in 2027.
- Research direction: a Sesame research engineer's April 2026 talk names turn-taking, full-duplex, data at scale and latency budgeting as open problems [stated: UMD abstract].

## Lessons for our setup

1. **Use the CSM split for our tokenizer.** This assumes our tokenizer is Qwen3-TTS-Tokenizer-12Hz. Its config lists 1 semantic quantizer and 16 decoder quantizers at 12.5 Hz, so the split maps directly:
   - the Qwen3 backbone predicts the semantic codebook 0 every frame
   - a small per-frame depth decoder predicts the residual codebooks
   This keeps first-audio latency to one backbone step [inferred: me]. Breeze-TTS-2 is evidence this works on a Qwen3-1.7B-shaped backbone.
2. **Amortize the decoder, then check whether it is the weak part.** Training the decoder on 1/16 of frames frees memory for long (~2 min) duplex contexts.
   - Sesame's "no perceivable loss difference" was measured at 1M h. At 10K h the residual decoder may be under-trained.
   - If continuation sounds poor while semantic tokens look fine (e.g., good ASR of the codebook-0 path, poor acoustic quality), we should raise the decoder's frame ratio, depth or data before blaming the backbone [inferred: me].
3. **Expect a data gap for continuation quality.** Sesame used about 1M h with 5 epochs just to make a contextual TTS sound human. Our 10K h duplex set is about 1% of that. Pre-training on large monologue, podcast or conversational corpora before duplex mid-training is the obvious fix [inferred: me].
4. **CSM gives no help on knowledge.** It was not initialized from an LLM and cannot produce text; the demo's content came from a separate LLM. For our weak semantic knowledge, CSM points to keeping a text LLM in the loop as the content source, through a text channel or thinker–talker. Speech generation then conditions on text plus audio context [inferred: me].
5. **Adopt the cheap diagnostics.**
   - Homograph and pronunciation-consistency tests use 200 samples each with a phoneme recognizer. They separate text understanding from audio-context understanding. They are directly reusable and probe the "semantic knowledge in speech" problem [inferred: me].
   - Run CMOS *with context*, not just isolated utterances.
6. **Evaluate turn-taking with TurnBench.** It is a clean benchmark for end-of-turn and interruption timing, and the 104 h of labelled training data is useful for diagnostics. The non-commercial licence likely bars commercial training use [inferred: me].
7. **Note the industry signal.** By Sesame's own account its 2025 demo was not full-duplex. Its 2026 releases make no full-duplex claim while it publishes turn-taking research. Even a well-funded voice-first team has not visibly shipped production full-duplex [inferred: me].

## Open questions

- Which audio sources and how much conversational data were in the 1M h? How much compute went into the 8B Medium?
- Which model size powers the demo and app? What changed in the 2026 "updated CSM": new tokenizer, LLM initialization, RL?
- What are the numbers in the homograph, consistency and CMOS figures?
- Has Sesame trained a full-duplex model internally, given TurnBench and the hiring focus, and why is it not on TurnBench?
- Does the 1/16 amortization still hold when the backbone is initialized from a pretrained LLM and data is small?
