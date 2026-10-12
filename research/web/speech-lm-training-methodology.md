# Speech-LM training methodology: what the controlled evidence says (Q1–Q7)

*Web literature note, compiled 2026-10-12.*

**Scope.** This note collects controlled experiments, scaling studies, ablations, benchmarks and surveys, not system descriptions. Systems that have their own notes (Moshi, Qwen-Omni, Qwen3-TTS, MOSS, PersonaPlex, Freeze-Omni, …) appear here only for their ablations.

**Sources.** Numbers come from full texts: arXiv HTML, or the local PDF text extracts of the papers in `../../papers/`. Anything else is marked *[abstract only]*.

**Tags.**
- **[controlled]**: an ablation or matched comparison inside one paper.
- **[scaling]**: a scaling-law study.
- **[benchmark]**: a benchmark result.
- **[system]**: a system-report claim.

**Notation.** S→S means speech in, speech out; S→T means speech in, text out; T2T means text in, text out. sSC and tSC are spoken StoryCloze and spoken topic-StoryCloze. GenPPL is the perplexity of ASR transcripts of generated speech under a reference text LM.

**Our setup.** A Qwen3 LLM, a Qwen3 audio tokenizer and its decoder, mid-trained on ~10K h of duplex speech. Continuation and knowledge are both weak.

## Bottom line for our setup

1. **Our symptoms match "speech-only training at small data".**
   - Speech-only LMs gain linguistic ability with compute up to ~10³× more slowly than text LMs.
   - At a matched 10K h, a speech-only model's continuations reach GenPPL ≈180. With an aligned text stream the same model reaches ≈53.
   - Going to 45K h of speech moves speech-only only to 149.
   - **More duplex hours will not fix knowledge. Text anchoring will** (Q1).
2. **Knowledge in generated speech needs either a text channel or ~10¹¹ interleaved tokens.**
   - Inner monologue roughly triples spoken QA.
   - With no interleaved speech–text pretraining, spoken QA is ≈0.
   - Going from 100B to 600B synthetic interleaved tokens moves S→S Llama-Q from 37 to 51 (Q1, Q2).
3. **Protect Qwen3's text ability explicitly.**
   - On Qwen3-8B, a frozen-backbone stage before unfreezing gave +7.5 tSC, +6.5 sSC and +4.4 MMLU.
   - Replay of text and earlier data is the strongest anti-forgetting tool, even at tiny ratios.
   - On-policy distillation from the model's own text mode beats SFT for closing the speech-vs-text gap (Q3, Q6).
4. **Keep the backbone's audio stream semantic and coarse.**
   - Each extra acoustic codebook in the backbone stream lowers content quality.
   - Up-weighting the semantic codebook ~100× recovers HuBERT-level linguistics.
   - Under low-latency delays, a depth/MTP head is mandatory (Q4).
5. **Duplex data is for timing and behaviour, not knowledge.**
   - Turn-taking gains saturate at ~0.5–2K h.
   - Training on human–human data alone wrecks assistant semantics (VoiceBench 18 vs 65).
   - Use LLM-scripted synthetic dialogue for content and real two-channel data for timing (Q5).
6. **If end-to-end knowledge stays weak, keep a strong text LLM in the loop.** Options with evidence: a tandem "oracle" stream (KAME: MT-Bench 2.05 → 6.43 at 0 s median latency), asynchronous RAG, or a thinker–talker split (Q7).

---

## Q1. Data scale: how much speech or interleaved data before continuation and knowledge transfer work?

**Key findings**

- **Speech-only (textless) SLMs scale very slowly.**
  - Setup: HuBERT 25 Hz units, ≤160K h (10.9B tokens), models from 20M to 823M.
  - The downstream-vs-compute exponents for speech are smaller than for text: 3.1× on sBLIMP, 1.6× on topic cloze and 2.7× on story cloze. The authors read this as up to ~10³× more compute per unit of syntactic gain.
  - Best model (823M): sBLIMP/tSC/sSC 61.3/78.0/56.7, vs 80.0/97.5/76.2 for Pythia-6.9B on the text versions.
  - Synthetic spoken TinyStories (72K h) beat audiobooks on the semantic metrics at every scale.
  - Coarser BPE-style units improved the loss but not downstream scores; sSC did not scale.

  [scaling] — Cuervo & Marxer, arXiv 2404.00685 (2024).
- **Initializing from a text LM is the largest lever, and it shrinks the value of extra hours.**
  - TWIST (text-LM initialization) trained on 10% of ~150K h matches or beats a randomly initialized model trained on 100%.
  - At 1.3B and 25 Hz, sWUGGY/sBLIMP:

    | Initialization | Data | sWUGGY | sBLIMP |
    |---|---|---|---|
    | TWIST | ~15K h (10%) | 82.28 | 56.81 |
    | TWIST | ~150K h (100%) | 82.49 | 57.05 |
    | Random | ~150K h (100%) | 81.94 | 56.52 |

  - TWIST reaches the same perplexity in ~¼ of the updates.
  - Even TWIST-13B reaches only sSC 55.4 / tSC 76.4, against humans at 79.9 / 90.2.

  [controlled] — Hassid et al., arXiv 2305.13009 (NeurIPS 2023).
- **Interleaving with text changes the scaling regime: spend compute on parameters, not tokens.**
  - At 2e20 FLOPs the best interleaved model had 7B parameters trained on 4.2B tokens. Textless scaling laws would suggest 463M parameters on 72B tokens.
  - SIMS-7B is initialized from Qwen2.5-7B and trained on 15B tokens, ~50–60% of them speech (84K h real + 30K h synthetic). It beats far larger training runs:

    | Model | Training tokens | sBLIMP | sSC | tSC |
    |---|---|---|---|---|
    | SIMS-7B | 15B | 59.8 | 66.7 | 88.3 |
    | SpiritLM | 100B | 58.3 | 61.0 | 82.9 |
    | Moshi | 720B | 58.8 | 60.8 | 83.0 |

  - The text-model family matters: Qwen2.5 and Llama-3.2 were best, and Qwen2.5-0.5B beat SmolLM2-1.7B.
  - At 0.5B, adding synthetic speech to real speech raised sSC from 54.94 to 59.81 and tSC from 72.15 to 81.51.

  [scaling] — Maimon et al., arXiv 2504.02398 (COLM 2025).
- **Knowledge in directly generated speech needs ~10¹¹ interleaved tokens.**
  - Setup: a 9B model initialized from GLM-4, trained on 1T tokens: 30% text and 600B synthetic interleaved tokens. A text→speech-token model was run over FineWeb-Edu, so no audio was rendered.
  - With **no interleaving**, S→S Llama-Q is 2.3, i.e. ≈0.
  - Scaling interleaved data:

    | Interleaved tokens | Web-Q S→S | Llama-Q S→S | TriviaQA S→S | Llama-Q S→T |
    |---|---|---|---|---|
    | 100B | 9.3 | 37.0 | 11.7 | 60.0 |
    | 200B | 13.3 | 44.0 | 18.7 | 63.0 |
    | 600B | 15.9 | 50.7 | 26.5 | 64.7 |

  - At 1.5B, removing the plain-speech or the plain-text data barely mattered. Removing interleaving was catastrophic.

  [controlled] — Zeng et al., arXiv 2411.17607 (ICLR 2025). *Rough derivation: 1 h at 12.5 Hz is 45K frames, and ~60% of interleaved tokens are speech. So 100B interleaved tokens ≈ 10⁶ hour-equivalents, about 100× our 10K h.*
- **At ~10K h, a text stream buys far more than 4.5× more speech does.**
  - Setup: Flow-SLM, 456M parameters, initialized from OpenELM, trained on matched ~10K h datasets.
  - Emilia 10K h, GenPPL:

    | Model | GenPPL |
    |---|---|
    | Speech-only | 182.1 |
    | Speech + text stream (text leads by 2 frames) | **52.7** |
    | Text-only (same data) | ≈28–30 |
    | Llama-Mimi-style flattened RVQ, reproduced at 241M on the same data | 282.6 |
    | Published Llama-Mimi-8B (240K h) | 52.6 |

  - Scaling to 45K h moved speech-only to 149.3; speech+text stayed at 51.6. Scaling the model to 1.16B moved speech-only to 170.2.
  - Conversational Emilia beat audiobooks and TTS-rendered C4 as training data.
  - Cost of the text stream: speaker similarity 0.47 → 0.41 and UTMOS 3.19 → 3.07.

  [controlled] — Chou, Zhou & Livescu, arXiv 2609.14743 (SLT 2026).
- **Very large speech-only scale buys prosody, not long-form semantics.**
  - Apple's continuous-diffusion SLMs scale cleanly up to 16B parameters on tens of millions of hours. They produce emotive, multi-speaker speech, but "long-form linguistic coherence remains a significant challenge". [scaling] — arXiv 2604.24416 (2026).
  - MiMo-Audio reports few-shot ability emerging only beyond 100M h. [system]
- **Cheap continuation boosters: synthetic in-domain speech, plus DPO on judged continuations.**
  - Slam (Qwen2.5-0.5B, one GPU-day):
    - Adding sTinyStories: tSC 71.14 → 78.01, GenPPL 145.4 → 88.3.
    - Adding 30 minutes of DPO on 47K synthetic pairs: tSC 82.04, GenPPL 62.8.
    - Adding "diverse" real data *without* synthetic data slightly hurt (tSC 70.66).
    - Caveat: the evaluation is story-domain.

    [controlled] — arXiv 2502.15814 (2025).
  - Align-SLM ran RLAIF on TWIST-7B continuations (LLM-scored ASR transcripts, curriculum DPO): sSC 55.1 → 61.1, tSC 75.4 → 86.8, human MMOS 3.48 → 3.73. [controlled] — arXiv 2411.01834.

**Benchmarks**
- Likelihood probes:
  - sWUGGY (lexical) and sBLIMP (syntax), from ZeroSpeech 2021.
  - sStoryCloze (adversarial ending; human 79.9) and tStoryCloze (random ending; human 90.2), from TWIST.
  - SALMon (acoustic consistency).
- Generation quality: GenPPL, or an LLM judge on transcripts.
- Knowledge: Llama Questions (300 items) and Spoken WebQuestions (both from Spectron, arXiv 2305.15255), plus TTS-rendered TriviaQA. Report S→T and S→S separately.
- Moshi notes that textless probes poorly predict dialogue quality, and that duplex instruction tuning *lowered* sWUGGY from 72.6 to 63.0. So measure the base and the duplex checkpoints separately.

**Implication for our setup**
- 10K h of speech-only duplex continuation sits in the GenPPL ≈150–200 regime, and the data curves there are shallow.
- In order of priority:
  1. Add an aligned text stream for the assistant.
  2. Before duplex training, insert a single-stream interleaved stage of 10¹⁰–10¹¹ tokens. Generate it from text corpora with a text→Qwen3-audio-token model; Qwen3-TTS can provide this.
  3. Prefer a bigger Qwen3 to more tokens.
  4. Add a DPO/RLAIF pass on continuations.

## Q2. Speech–text interleaving and alignment

**Key findings**

- **Interleaving is what makes knowledge transfer happen.**
  - SpiritLM: word-level modality switches, 100B tokens split ⅓ speech, ⅓ text, ⅓ aligned. Removing interleaving dropped speech topic-StoryCloze from 82.9 to 74.0, S→T StoryCloze from 88.6 to 71.9, and T→S from 72.7 to 57.5. [controlled] — arXiv 2402.05755.
  - Zeng et al.: with no interleaving, spoken QA ≈0 (Q1).
- **How much interleaving.** About ⅓ of training tokens interleaved, with 20–40% of words rendered as speech, is the best-supported setting.
  - Sternberg et al. tested interleaved shares of 1/3, 2/3 and 5/6 of training tokens. They measured "implicit transcription": how well the spoken word can be decoded as text from intermediate layers.

    | Training mix | Implicit transcription (current-word Recall@10) |
    |---|---|
    | 1/3 interleaved | 61.9% |
    | 2/3 interleaved | 49.1% |
    | 5/6 interleaved | 2.7% |
    | Speech + text, no interleaving | 3.8% |
    | Speech-only | 0.2% |
    | Random init + interleaving | 2.2% |

  - Implicit transcription correlates with spoken factual recall (ρ = 0.70). [controlled analysis] — arXiv 2606.22473 (2026).
  - Zeng et al. used Poisson span lengths (λ = 10 words). A speech share of 0.2–0.4 of words works; beyond that range there is "significant degradation".
  - **No head-to-head word-level vs sentence-level pretraining study was found.** For text placement in the *output stream* of a duplex model, see Q6: FLM-Audio and AdaptDuplex favour sentence-level or bounded text-ahead over strict word alignment.
- **A text-first channel in the output (inner monologue) is the single biggest win.**
  - Moshi, continuations from 3-s prompts: inner monologue cut transcript NLL from 3.65 to 2.77 and grew output length from 602 to 1,920 characters. It also tripled spoken QA (Q7). [controlled] — arXiv 2410.00037 (Tables 6 and 8).
  - Spectron: removing the text losses raised continuation log-PPL from 126 to 714. [controlled]
  - OmniFlatten, full-duplex on Qwen2-0.5B with 2K h synthetic data. Qwen-max judged ASR transcripts of the spoken replies:

    | Variant | English | Chinese |
    |---|---|---|
    | Text + speech streams | 3.92 | 5.15 |
    | Speech-only | 2.19 | 3.06 |

    [controlled] — arXiv 2410.17799.
- **Interleaved beats parallel decoding at matched base model, data and tokenizer.**
  - Phi-4-MM-3.8B: Llama-Q 57.33 vs 46.67; Web-Q 26.20 vs 19.67; text–speech consistency WER 5.11 vs 17.94.
  - "Early-stop interleaving" keeps most of the gain at ~75% of the sequence length.

  [controlled] — arXiv 2506.04518 (2025).
  - PSLM (parallel text+speech) matches text-first chain-of-modality quality at roughly half the latency (0.34 vs 0.67 s). [controlled] — arXiv 2406.12428.
- **Text must lead audio with enough context.**
  - IST-LM (TTS): a 1:3 text:speech chunk ratio was best (WER 3.60 vs 3.35 non-streaming). At 1:4, speech overtakes text and WER rises to ~5.7. [controlled] — arXiv 2412.16102.
  - GLM-4-Voice alternates 13 text tokens with 26 speech tokens, so its text always runs ahead.
  - Kyutai DSM uses a 1.28 s text lead for TTS and a 2.5 s delay for ASR; delay conditioning beats any fixed delay.
- **Interleaved output hides an output-mode gap.**
  - In Step-Audio-2-mini, the internal text answer is 42.87 pp less accurate on Spoken-MQA when the model is also speaking.
  - Distilling on-policy from its own S2T mode (JO-OPD) cut the gap to 16.26 pp and raised spoken-answer accuracy from 28.67% to 36.16%.
  - Matched SFT baselines did worse.

  [controlled] — arXiv 2609.15313 (2026).
- **Speech decoders that read text-LLM states work with little data (half-duplex).** EchoX (~6K h) scores S2S Llama-Q/Web-Q/TriviaQA of 63.3/40.6/35.0 at 8B. Removing its "echo" training costs the 3B model 13–14 points per benchmark. [controlled] — arXiv 2509.09174.

**Implication for our setup**
- Put Qwen3 text tokens in the assistant stream, leading audio by ~1–2 frames.
- Pretrain with ~⅓ interleaved tokens, rendering 20–40% of words as speech.
- Monitor the output-mode gap (text-only answers vs joint text+speech answers on the same spoken questions). Fix it with on-policy distillation rather than SFT.

## Q3. Preserving text intelligence during mid-training

**Key findings**

- **Without replay, forgetting is catastrophic. Replay is the best fix, even at tiny ratios.**
  - Setup: Llama-3.2-11B (vision tower removed), LoRA r=64, trained ASR → TTS → spoken QA.
  - Text-only (T2T) accuracy collapsed: Llama-Q 70.0 → 14.3, Web-Q 61.5 → 3.6, TriviaQA 78.9 → 6.0, IFEval 67.1 → 9.2.
  - Replay restored it to 66.3/55.2/66.4/47.5. Each stage mixed in earlier data, including the LLM's own text-instruction data, at 0.5% of the stage size.
  - Merging alone reached only 19.3; LoRA down-scaling only 15.0.
  - Even with replay, speech output lags badly: Llama-Q T2T 66.3, S2T 50.3, S2S 28.7.

  [controlled] — Hsiao et al., arXiv 2505.17496 (Interspeech 2025).
- **Text share vs retention across recipes.** The recipes differ, so treat this as indicative.

  | Recipe | Text share | MMLU / knowledge |
  |---|---|---|
  | OpusLM-7B | 50% | 63.4 → 59.0 |
  | Moshi | 50% text batches | 54.3 → 49.7; removing the text batches cut TriviaQA 22.8 → 18.3 |
  | SpiritLM | ⅓ | 46.2 → 36.9 |
  | Ichigo | 10% | 69.4 → 63.8 |
  | VoiceTextBlender (LoRA SFT) | 15% | 35.5 → 34.8 |
  | Speech-only SFT | 0% | → chance; GSM8K ≈0 |

  Sources: arXiv 2506.17611, 2410.00037, 2402.05755, 2410.15316, 2410.17485.
- **Freeze first, then unfreeze. This was tested on Qwen3-8B.**
  - MOSS-Speech Stage 1 trains only the new speech parts on a frozen Qwen3-8B. Against training everything from the start:

    | Variant | tSC | sSC | MMLU |
    |---|---|---|---|
    | Frozen stage, then unfreeze all | 85.20 | 63.12 | 66.50 |
    | No freezing | 77.66 | 56.60 | 62.11 |
    | No freezing, no layer split | 77.12 | 55.80 | 60.97 |
    | Qwen3-8B (text) | – | – | 76.60 |

  - Unfreezing layer by layer (FP-Layerwise) kept MMLU at 68.82.

  [controlled] — arXiv 2510.00499 (Table 7).
- **LoRA does not protect against speech-only data.**
  - SmolLM2-360M and 1.7B on 48K h of ASR data, average text-benchmark change:

    | Method | 360M | 1.7B |
    |---|---|---|
    | Full fine-tuning | −22.2% | −32.6% |
    | LoRA | −22.0% | −35.7% |
    | New layers inserted into the frozen LLM | −1.9% | −8.3% |

    The inserted-layer method reached near-equal ASR. [controlled] — Yano et al., arXiv 2604.00489 (2026).
  - SSR-Connector: LoRA's MMLU "steadily declined"; mixing in text limited the drop to ~2 points. [controlled] — arXiv 2410.00168.
- **Distillation and self-generated targets.**
  - DeSTA2.5-Audio: when the backbone writes its own training targets, the forgetting rate is −0.24 to −2.15. With another LLM's targets under LoRA it is −10.11. [controlled] — arXiv 2507.02768.
  - SALAD: KL distillation to the text LLM on aligned content, plus 1% targeted synthetic data, over 141.6K h. The understanding gap fell to 6.2%, vs Qwen2-Audio 17.8%, GLM-4-Voice 20.1% and Qwen2.5-Omni 5.0%. Plain next-token training *increased* cross-modal misalignment as data scaled. [controlled] — arXiv 2510.13632.
  - Both are input-side results. For speech output, see JO-OPD (Q2) and X-OPD (Q6).
- **Merging is a finisher, not a fix.**
  - TASTE2 TIES-merged the Instruct−Base task vector (top 30%, λ = 0.5) into its speech-trained base before SFT. Llama-Q reached 56.3 against a 57.3 text reference (98.2% retention), vs 92.4% when starting directly from Instruct. [system ablation] — arXiv 2609.08956.
  - In Hsiao et al., merging alone was weak.

**Implication for our setup**
1. Never train a stage without text. Use 15–50% text, ideally Qwen3's own SFT-style outputs.
2. Start mid-training with Qwen3 frozen, training only embeddings, heads and adapters. Then unfreeze, possibly layer-wise.
3. Add a distillation term toward Qwen3's text mode: KL at aligned text positions, or on-policy.
4. After SFT, try merging the Qwen3-Instruct task vector back in.
5. At every checkpoint, track T2T/S2T/S2S on the same QA items, plus MMLU and IFEval.

## Q4. Audio token design for LM-native generation

**Our tokenizers** (Qwen3-TTS, arXiv 2601.15621):
- **Tokenizer-12Hz.**
  - 12.5 Hz, 16 codebooks of 2048 entries (≈2.2 kbps).
  - Codebook 0 is semantic, distilled from WavLM; codebooks 1–15 are acoustic RVQ (residual vector quantization).
  - Decoded by a causal ConvNet with a 97 ms first packet.
  - Reconstruction on LibriSpeech, PESQ-WB / UTMOS / speaker similarity: 3.21 / 4.16 / 0.95, against Mimi-16 at 2.88 / 3.87 / 0.87.
  - In Qwen3-TTS the backbone predicts codebook 0 and an MTP module predicts the 15 residual codebooks.
- **Tokenizer-25Hz.**
  - Single codebook, ASR-supervised on the Qwen2-Audio encoder.
  - Decoded by a block-wise flow-matching DiT plus BigVGAN.
- **12 Hz vs 25 Hz inside Qwen3-TTS** (within-report comparison):
  - 12 Hz wins on short-form WER: Seed test-zh/en at 1.7B is 0.77/1.24 vs 1.10/1.49.
  - 25 Hz wins on long-form; the authors write that "semantic tokens may be more beneficial for maintaining stability over extended sequences".

**Key findings**

- **Pure acoustic tokens give chance-level SLM linguistics. Distilled codecs approach SSL tokens only if the semantic codebook's loss is up-weighted.**
  - Setup: identical 357M SLMs with TWIST init, trained on LibriHeavy 56K h.

    | Tokenizer | sBLIMP | sWUGGY | tSC |
    |---|---|---|---|
    | HuBERT-25Hz | 60.9 | 70.5 | 71.5 |
    | EnCodec | 51.1 | 51.3 | 48.2 |
    | Mimi-8 | 52.3 | 62.2 | 54.3 |
    | Mimi-8, semantic loss ×100 | 60.2 | 67.6 | 68.5 |
    | WavTokenizer (single codebook) | 51.2 | 54.6 | 52.8 |

  - Distilled codecs win on acoustic consistency (SALMon).

  [controlled] — Mousavi et al., "Discrete Audio Tokens: More Than a Survey!", arXiv 2506.10274 (2025).
- **Text-aligned or ASR-supervised tokens give the largest semantic gains.**
  - TASTE (one speech embedding per text token, ~150 bps) raised sSC/tSC from 54.4/63.0 with S3 tokens to 64.2/88.9. [controlled] — arXiv 2504.07053.
  - In GLM-4-Voice, an ASR-supervised 12.5 Hz single-codebook tokenizer at 175 bps matches Mimi's resynthesis WER (8.43 vs 8.36). [system]
  - Caveat: "semantic" SSL-distilled tokens are mostly *phonetic*, not lexical (arXiv 2603.10371, abstract only).
- **Good reconstruction does not mean easy-to-model tokens. LM-aware codec training helps.** A LLaMA-3-8B was continued-pretrained on ~400K h with codecs retrained on identical data.
  - XCodec, SpeechTokenizer, XCodec2 and BigCodec all scored sSC/tSC ≈63–64/72–73.
  - Adding an AR-predictability regularizer plus a 6.25 Hz semantic layer reached 70.1/76.9, with unchanged codec WER and speaker similarity.

  [controlled] — arXiv 2609.04237 (2026).
- **A lower frame rate helps the LM down to ~12.5 Hz.**
  - Zeng et al. swept 50/25/12.5/6.25 Hz at a fixed token budget: lower was better. 6.25 Hz was rejected because resynthesis WER rose from 8.43 to 14.41.
  - TWIST: 25 Hz beat 50 Hz on sWUGGY (81.42 vs 70.69).
  - Syllable-rate units (4–8 Hz; SyllableLM, Sylber) match HuBERT probes with 5× fewer tokens. But Sylber's GenPPL was much worse (183 vs 86), and Cuervo & Marxer found coarser units scale worse downstream.

  [controlled] — arXiv 2411.17607, 2305.13009, 2410.04029, 2509.26634.
- **The more acoustic codebooks the backbone must predict, the worse the content.**
  - Llama-Mimi flattens codebooks into one stream (sequence) of 1.3B LM tokens. With 2/4/8 codebooks, content quality was 3.53/3.01/2.54 while speaker similarity was 0.201/0.346/0.474.
  - Up-weighting the semantic loss ×100 traded speaker similarity (0.346 → 0.196) for tSC (64.0 → 68.4).

  [controlled] — arXiv 2509.14882. DASB reaches the same conclusion: more codebooks give better reconstruction and worse downstream performance (arXiv 2406.14294).
- **Multi-codebook prediction under low latency needs a depth transformer.**
  - Moshi Table 5, independent heads vs RQ/depth transformer:
    - MusicGen-style full delay [0..7] (640 ms): heads 42.2 vs depth transformer 40.3, nearly equal.
    - Low-latency delay [0,2,…] (240 ms): heads **135.4** vs depth transformer 36.8.
  - Moshi Table 6, continuation transcript NLL:
    - Acoustic delay of 0/1/2 frames: 4.36/4.12/4.09.
    - Semantic loss ×100: 3.75; adding depthwise parameters: 3.65.

  [controlled] — arXiv 2410.00037.
  - MusicGen: delay patterns roughly match flattening at ¼ of the steps, while parallel prediction is much worse (FAD 0.96 vs 0.86 vs 2.58). [controlled] — arXiv 2306.05284.
  - MTP done right helps.
    - VocalNet, WER: naive MTP 18.23, its own MTP 6.46, next-token prediction 10.62.
    - VITA-Audio: MTP modules cut time to first audio from 236 to 53 ms at a small QA cost (42.6 → 40.0).

    [controlled] — arXiv 2504.04060, 2505.03739.
- **Continuous latents with a flow or consistency head are viable, and faster than an RQ-transformer.**
  - Kyutai CALM, same 2B stack with inner monologue, CALM vs an 8-codebook RQ-transformer:
    - transcript PPL 23.8 vs 52.4;
    - acoustic-quality MOS 3.45 vs 2.42;
    - sampler 12.3× faster.

    [controlled] — arXiv 2509.06926.
  - At a matched 10K h, a continuous-feature speech-only model (GenPPL 182) beats flattened RVQ (283) but stays far behind speech+text (53). Continuous output alone does not fix semantics (Chou et al.).

**Implication for our setup**
- With Tokenizer-12Hz:
  - Never flatten all 16 codebooks into the backbone.
  - Have the backbone predict codebook 0 (plus text), and a depth/MTP module predict codebooks 1–15. Add 1–2 frames of acoustic delay.
  - Weight codebook 0 roughly 100× relative to the acoustic codebooks.
  - Evaluate continuation on codebook 0 plus transcript GenPPL.
- If continuation stays weak, consider the 25 Hz ASR-supervised single-codebook tokenizer as the backbone's stream. It is more semantic and more stable on long-form, but its DiT decoder adds latency. A CALM-style continuous head is the other option.

## Q5. Full-duplex data

**Real two-channel corpora**

| Corpus | Lang | Hours | Audio | Licence / access |
|---|---|---|---|---|
| Fisher English Pt1+2 (LDC2004S13/2005S13) | EN | 984 + 975 (11.7K calls ≤10 min) | 8 kHz, 2-ch | LDC fee; non-member licences research-only |
| Switchboard-1 R2 (LDC97S62) | EN | ~260 (~2.4K calls) | 8 kHz, 2-ch | LDC |
| Fisher Spanish / HKUST Mandarin / CALLHOME / CALLFRIEND | ES / ZH / many | ~163 / ~149 / small | 8 kHz, 2-ch | LDC |
| CANDOR | EN | ~850 (1,656 video-call dyads) | per-participant tracks | By application (BetterUp) |
| Seamless Interaction (Meta 2025, arXiv 2506.22554) | (not stated on card) | 4,000+ (65K interactions; partly acted) | 48 kHz, per-participant | CC-BY-NC 4.0 |
| MagicData-RAMC | ZH | 180 | 16 kHz | CC BY-NC-ND 4.0 |
| AMI (headset mics) / AliMeeting | EN / ZH | 100 / 119 | 16 kHz | CC BY 4.0 / CC BY-SA 4.0 |
| DuplexChat (podcasts separated with DialogueSidon, arXiv 2607.04941) | EN / JA | 282.6K / 132.7K | 16 kHz pseudo-stereo | URLs and timestamps only (CC BY 4.0) |

Mono dialogue sources that would need separation: J-CHAT (76K h JA; CC-BY-NC, gated), OpenDialog (EN 5.1K h + ZH 1.8K h; CC-BY-NC) and Emilia (101K h). Emilia's pipeline splits audio into single-speaker segments, so it cannot be used as duplex data as-is.

**Key findings**

- **Diarized pseudo-stereo is a pretraining proxy only.**
  - Moshi ran Whisper + PyAnnote over 7M h. The result "contains no overlap, and the stream of an inactive speaker is perfectly silent".
  - Real interaction was learned from Fisher: ~2K h, upsampled from 8 to 24 kHz with AudioSR.
  - On top of that, 20K+ h of synthetic dialogue with noise, gain and echo augmentation.

  [system] — arXiv 2410.00037.
- **Proper overlap separation pays off even at small scale.** The Sommelier pipeline:
  - Steps: Sortformer diarization (DER 7.16% vs 8.40% for pyannote), SepReformer applied only to overlapped spans, then a 3-ASR ROVER ensemble.
  - On 0 dB fully overlapped mixtures, WER drops from 48.9% to 15.6%.
  - LoRA-tuning moshiko on just ~83 h of the cleaned data improved Full-Duplex-Bench: backchannel turn-over rate 1.000 → 0.291 and interruption score 0.765 → 3.684. Interruption latency got worse, 0.257 → 1.065 s.
  - Floor-holds longer than a minute destabilized training, so turns were kept ≤10 s.

  [system] — arXiv 2603.25750 (ACL 2026 Industry).
  - DialogueSidon jointly restores and separates dialogue audio. On in-the-wild audio it reaches WER 13.86% vs 43.79% for the GENESES baseline, and its authors argue that separators trained on monologue mixtures do not transfer to dialogue. [method] — arXiv 2604.09344.
- **Synthetic recipes.**
  - Data volumes:
    - SyncLLM: 212K h of Bark TTS.
    - OmniFlatten: 2K h of CosyVoice from 390K text sessions.
    - PersonaPlex: 2.25K h scripted by Qwen3-32B and GPT-OSS-120B, voiced with Dia/Chatterbox; barge-in is simulated with "negative-duration silence".
    - SALM-Duplex: 27K h.
    - MultiTalk: 57.6K h, with turn-length priors, 0.2–0.6 s overlaps at normal transitions and 1–2 s at interruptions.
    - DuplexSLA: ~370K h.
    - Raon: 106K h synthetic plus 13K h real. Part of the synthetic data re-voices its earlier SpeechLM training samples "to preserve the model's prior knowledge".
  - Most dialogue TTS systems (FireRedTTS-2, MoonCast, SoulX-Podcast, Dia) produce sequential turns with **no overlap**. CoVoMix2 and stereo ZipVoice-Dialog do model overlap.
- **How much duplex data: turn-taking saturates early; role and content keep improving.**
  - PersonaPlex, adding 0/25/50/100% of its 2.25K h to Moshi:

    | Data | Speaker similarity | Full-Duplex-Bench GPT-4o score | Role adherence |
    |---|---|---|---|
    | 0% (Moshi) | 0.10 | 0.77 | 1.75 |
    | 25% (~560 h) | 0.54 | 4.44 | 4.20 |
    | 50% | 0.56 | 4.52 | 4.24 |
    | 100% | 0.57 | 4.21 | 4.48 |

    [controlled] — arXiv 2602.06053.
  - MultiTalk at 0/25/50/75/100% of 3.2K h scored 7.95/10.39/11.78/12.77/13.15. [controlled]
- **Synthetic vs real.**
  - Stitched synthetic dialogue has 0% overlapping transitions, against 42.8% in real dialogue (arXiv 2608.16053).
  - LLM-scripted dialogue has backchannels in 0.6% of turns, against 19% of utterances in Switchboard.
  - Real human–human data alone teaches timing but wrecks semantics. A Qwen3-4B duplex controller trained on Fisher + Switchboard got turn-taking F1 0.618 but VoiceBench **18.30**. A decoupled mix got 0.654 and 64.84, against 64.93 for the base LLM. [controlled] — arXiv 2609.03321 (EMNLP 2026).
  - Fisher-learned timing transfers imperfectly to CANDOR: SyncLLM's turn-taking correlation drops from 0.41–0.43 to 0.23–0.31. [system]

**Implication for our setup**
- Treat the 10K h duplex set as a *timing and behaviour* resource. It is already 5–20× the point where published turn-taking gains saturate.
- Keep interleaved, monologue and text data in the duplex mix.
- Make dialogue content assistant-like with LLM-scripted dialogues voiced by Qwen3-TTS. Inject overlaps and backchannels at rates fitted to Fisher or CANDOR statistics.
- Check telephone 8 kHz audio against the tokenizer's 24 kHz input; Moshi upsampled with AudioSR.
- Most open corpora are non-commercial or research-only.

## Q6. Duplex post-training, alignment and evaluation

**Key findings: SFT**

- **Route data by capability: human dialogues supervise states, assistant dialogues supervise content.** This is the decoupled mix from Q5 (Fisher + Switchboard alone: VoiceBench 18.30; decoupled: 64.84).
  - ASPIRin: plain SFT of Moshi on 43 h of natural dialogue collapsed the interruption-reply score from 3.894 to 0.440. [controlled] — arXiv 2604.10065.
- **Use a curriculum: turn-based → basic duplex → full duplex.**
  - AdaptDuplex (Qwen3-Omni-30B-A3B), FDB-v1.5 interruption handling: mixed SFT 0.62 at 1.65 s; curriculum 0.82 at 0.33 s; curriculum + GRPO 0.85 at 0.30 s. [controlled] — arXiv 2609.29217.
  - OmniFlatten, speech-reply score: 2.59 training straight to full duplex, 3.54 without the half-duplex stage, 3.92 with the full curriculum.
  - FLM-Audio: removing the semi-duplex stage cut its instruction-following score from 6.58 to 4.59 (arXiv 2509.02521).
- **Text placement in the output stream.**
  - AdaptDuplex: a bounded text lead (≤2 tokens) vs strict word alignment improved WER 19.10 → 8.93% and UTMOSv2 2.90 → 3.65.
  - FLM-Audio: sentence-level text vs word-level alignment improved HellaSwag 58.3 → 61.6. [controlled]
- **Down-weight silence and use explicit state tokens.**
  - BayLing-Duplex: with uniform loss weights, turn success within 3 s was 60.3%. Weighting the assistant's speech 10× and silence 0.1× raised it to 88.9%. [controlled] — arXiv 2606.14528.
  - SALMONN-omni, explicit vs implicit state tokens: Llama-Q 82.0 vs 76.3. [controlled] — arXiv 2505.17060.

**Key findings: RL and preference optimization**

- **Flat RL over all tokens trades content for timing. Separate the "when to speak" decision from "what to say", and keep a semantic reward.**
  - ASPIRin: flat GRPO made backchannel turn-over worse (0.436 → 0.709) and increased repetition. Its decoupled variant reached 0.364.
  - HiPLEX: pause turn-over rate 0.579 → 0.306 and judge score 3.773 → 4.083. Without the semantic reward the judge score fell to 2.332. [controlled] — arXiv 2610.07727.
  - DuplexPO applied GRPO only on dynamics-critical windows. On FDB-v3, yield went 64.8 → 100% and latency 7.33 → 0.24 s, while Llama-Q improved 72.0 → 75.3. DPO did worse than GRPO here. [controlled] — arXiv 2607.07148.
- **Preferences mined from real user conversations improve content. Apply the loss to the text stream only.**
  - Moshi with length-normalized DPO on ~93K AI-judged pairs: QA 36.1 → 39.2 and safety 70.9 → 77.8.
  - Adding audio tokens to the loss *hurt* (QA ≈35–36).
  - Replies became longer.

  [controlled] — arXiv 2506.21463 (ICML 2025).
- **On-policy distillation from the backbone's own text model gives the strongest knowledge recovery.**
  - X-OPD: Qwen3-Omni-30B-A3B student, Qwen3-30B-A3B text teacher, 27.8K prompts (232 h).
  - The speech-input drop vs the teacher fell from 11.29% to 3.43%; Big Bench Audio went 85.67 → 93.41.
  - On the same data, SFT made the gap *worse* (22.76%), and so did offline KD (19.62%).

  [controlled] — arXiv 2603.24596 (Interspeech 2026).
- **Other quality rewards.**
  - SpeechAlign, iterative DPO on codec tokens: WER 7.2 → 6.0.
  - Multi-reward RLAIF: an audio reward took UTMOS from 2.16 to 3.06 and WER from 6.1 to 3.3, while combining all rewards diluted emotion gains (arXiv 2404.05600, 2601.19063).

**Reward models and judges**
- General audio LLMs are weak spoken-dialogue judges. GPT-4o-audio reaches 57.9% and Gemini-2.5-Pro 76.4%, against 96.7% for the trained SDiaReward model (arXiv 2603.14889).
- A duplex generative reward model on Qwen2.5-Omni-7B scores 77.3% on real dialogues vs GPT-4o's 68.2% (arXiv 2604.14920).
- Single quality metrics (UTMOS, DNSMOS) get reward-hacked; composite rewards resist this (arXiv 2609.13150, *abstract only*).

**Evaluation suites**

| Suite | Measures | Representative finding |
|---|---|---|
| Full-Duplex-Bench v1 (2503.04721) | pause, backchannel, turn-taking, interruption | Moshi pause turn-over rate 0.985; interruption relevance 0.765 vs Freeze-Omni 3.615 |
| FDB v1.5 (2507.23159) | overlap types | Moshi "unknown" response to user backchannels 0.92 |
| FDB v2 (2510.07838) | multi-turn, examiner agent | Moshi turn-taking 3.73 ≈ GPT-Realtime 3.74, but instruction-following 2.67 vs 3.81 |
| FDB v3 (2604.04847) | tool use with real disfluencies | GPT-Realtime Pass@1 0.600; self-correction hardest |
| Talking Turns (2503.01174) | turn-taking events | Moshi barges in on the AI 46.5% of the time (human 15.9%) |
| Game-Time (2509.26388) | timing instructions | most models fail timed silence |
| URO-Bench (2502.17810) | understanding, reasoning, chat; speech output | GLM-4-Voice reasoning 55.46 vs Whisper+GLM-4 76.29 |
| VoiceBench (2410.17196) | spoken instructions | 2024 end-to-end models lost 25–40 points from text to speech input |
| Big Bench Audio (Artificial Analysis) | 1,000 BBH questions in audio | Moshi 4%; top systems 97–100% |
| τ-Voice (2603.13686) | duplex tool-agent tasks | 31–51% vs GPT-5 text 85% |
| MTR-DuplexBench (2511.10262) | multi-round | Moshi instruction-following falls 68.0 → 41.9% over 10 rounds |
| Spoken QA (Llama-Q / Web-Q / TriviaQA) | knowledge | report S→T and S→S |

Absolute numbers depend on protocol. Changing only the prompt delay moves Moshi's FDB-v1 pause turn-over rate from 0.985 to 0.467, so compare models only within one paper.

**Implication for our setup**
- **SFT.** Use separate data for state targets and content targets. Run a turn-based or half-duplex curriculum stage. Let text lead audio by a bounded amount, down-weight silence, and use explicit state tokens.
- **Knowledge recovery.** Run on-policy distillation from Qwen3 text (X-OPD/JO-OPD), not more SFT. Then text-stream-only DPO.
- **Interactivity RL.** Use GRPO on a separate control action with hand-shaped timing rewards, always paired with a semantic reward.
- **Evaluation.** Use FDB v1/v1.5/v2 for interaction. Use S→T vs S→S spoken QA, URO-Bench, VoiceBench and Big Bench Audio for content.

## Q7. Architectures that work around weak knowledge

**Key findings**

- **Tandem designs ("cascaded in the loop") recover most knowledge with no added time to first audio. The cost moves to when the key content arrives.**
  - **KAME.** A Moshi-style front-end receives a fourth stream of "oracle" text from an asynchronous back-end LLM, updated every 100–500 ms. It was trained on 56.6K synthetic dialogues with simulated oracles. MT-Bench (reasoning/STEM/humanities):

    | System | MT-Bench | Median latency |
    |---|---|---|
    | Moshi, fine-tuned on the same data | 2.05 | 0.0 s |
    | KAME + GPT-4.1 | 6.43 | 0.0 s |
    | Unmute cascade | 7.70 | 2.1 s |

    The back-end LLM can be swapped without retraining. [controlled] — arXiv 2510.02327 (ICASSP 2026).
  - **MoshiRAG.** Asynchronous references take Moshi's Llama-Q/Web-Q/TriviaQA from 62.3/26.6/22.8 to 80.6/68.9/78.2 (GPT-4.1 references). For comparison, GPT-4o-Audio scores 88.4/81.0/90.6. Time to first audio stays 0 s, but the answer keyword arrives ~3.1 s after the question. Multi-step math transfers poorly (GSM8K 66.2 → 33.9). [controlled] — arXiv 2604.12928 (2026).
  - **NVIDIA front-end/back-end.** A duplex front-end delegates to a Qwen3 back-end. FDB-v3 response quality is 54.0 with Qwen3-30B-A3B and 67.0 with Qwen3-235B-A22B. Filler speech starts at ~0.7–0.8 s; the real answer comes at 2.8–3.1 s. [benchmark] — arXiv 2609.19334.
  - **StreamRAG.** Querying while the user is still talking took AudioCRAG accuracy from 11.1% (no retrieval) to 26.3% (retrieve after the user finishes) to 34.2% (streaming), and cut tool latency by 20.7%. The streaming variant was also post-trained, so part of the gain is confounded. [controlled] — arXiv 2510.02044.
- **Thinker–talker designs: how much knowledge survives depends on the training recipe.**
  - Qwen3-Omni, matched-FLOPs 30B-A3B base models, text-only vs omni:

    | Benchmark | Text-only | Omni |
    |---|---|---|
    | MMLU | 81.24 | 81.69 |
    | GSM8K | 90.83 | 91.36 |

    Audio was mixed in from early in pretraining, which is not our situation. The talker is conditioned on streamed text, so externally supplied text can be injected. [controlled] — arXiv 2509.17765 (Table 16).
  - Qwen2.5-Omni lost text ability: MMLU-Pro 47.0 vs 56.3 for Qwen2.5-7B (arXiv 2503.20215).
  - LLaMA-Omni2 loses ~10 points from text to speech output (Llama-Q 70.3 → 60.7). Its read:write ratio trades latency against quality: 457 ms / UTMOS 3.98 vs 799 ms / 4.32 (arXiv 2505.02625).
  - Freeze-Omni keeps its LLM frozen and retains knowledge (Llama-Q 72.0 speech input vs 77.7 text input), but takes turns in 0.953 s vs Moshi's 0.265 s.
- **Text-guided decoding beats direct speech generation.**
  - Moshi's inner monologue: Web-Q/Llama-Q/TriviaQA 9.2/21.0/7.3 → 26.6/62.3/22.8, for 17 instead of 16 tokens per step.
  - SCoT (a block-wise duplex model): adding a text step per 2-s block cut transcript perplexity from 313.8 to 42.3 (arXiv 2510.02066).
- **"Think while listening or speaking" helps reasoning, not facts.**
  - STITCH (GLM-4-Voice): math average 62.98 → 78.04 at no added latency to first audio, close to think-first (79.12). TriviaQA moved only 46.0 → 50.0 (arXiv 2507.15375).
  - Mind-Paced Speaking: Spoken-MQA 70.6 → 92.8, but runs two LLMs (arXiv 2510.09592).
  - FLAIR (full-duplex latent thinking): Llama-Q 73.0 → 78.0 but TriviaQA 53.8 → 51.2, with turn-taking latency +60 ms (arXiv 2603.17837).
  - SHANKS: interrupting the user's math mistakes became valid 26.8% → 63.9% of the time (arXiv 2510.06917).
- **Head-to-head on URO-Bench:** GLM-4-Voice 69.09 vs Whisper + GLM-4-9B-Chat 84.24 overall. VERA shows voice systems averaging 11.3% vs 54.0% for text (arXiv 2509.26542).

**Implication for our setup.** Keep the Qwen3 text path in the loop. In order of preference:
1. **Inner-monologue text in the duplex model.** The cheapest change, worth ~3× on spoken QA.
2. **A KAME-style oracle or thinker stream from Qwen3-Instruct.** It can prime or override the duplex model's text channel.
3. **Background delegation or RAG for hard queries.** Use filler speech, and measure *keyword delay*, not only time to first audio.

Use think-while-speaking for reasoning, not for missing facts.

---

## Reading list

**Data scale and scaling (Q1)**
- Cuervo & Marxer, *Scaling Properties of Speech Language Models* (2024) — https://arxiv.org/abs/2404.00685
- Maimon et al., *Scaling Analysis of Interleaved Speech-Text Language Models* (COLM 2025) — https://arxiv.org/abs/2504.02398
- Zeng et al., *Scaling Speech-Text Pre-training with Synthetic Interleaved Data* (ICLR 2025) — https://arxiv.org/abs/2411.17607
- Chou, Zhou & Livescu, *Quantifying the Generation Modality Gap in Speech-Text LMs* (SLT 2026) — https://arxiv.org/abs/2609.14743
- Hassid et al., *TWIST* (NeurIPS 2023) — https://arxiv.org/abs/2305.13009
- Maimon et al., *Slamming* (2025) — https://arxiv.org/abs/2502.15814
- Lin et al., *Align-SLM* (2024) — https://arxiv.org/abs/2411.01834

**Interleaving and alignment (Q2)**
- Nguyen et al., *SpiritLM* (2024) — https://arxiv.org/abs/2402.05755
- Sternberg, Maimon & Adi, *Interleaved Speech LMs Latently Work in Text* (2026) — https://arxiv.org/abs/2606.22473
- Wu et al., *Towards Efficient Speech-Text Jointly Decoding* (2025) — https://arxiv.org/abs/2506.04518
- Tan et al., *JO-OPD: Reducing the Output-Mode Gap* (2026) — https://arxiv.org/abs/2609.15313
- Zhang et al., *OmniFlatten* (2024) — https://arxiv.org/abs/2410.17799

**Preserving text intelligence (Q3)**
- Hsiao et al., *Mitigating Catastrophic Forgetting in SLMs* (Interspeech 2025) — https://arxiv.org/abs/2505.17496
- Zhao et al., *MOSS-Speech* (2025; Qwen3-8B freeze ablation) — https://arxiv.org/abs/2510.00499
- Cuervo et al., *SALAD: Closing the Gap Between Text and Speech Understanding* (2025) — https://arxiv.org/abs/2510.13632
- Yano et al., *Multimodal Depth Up-Scaling* (2026) — https://arxiv.org/abs/2604.00489
- Lu et al., *DeSTA2.5-Audio* (2025) — https://arxiv.org/abs/2507.02768

**Tokens and output heads (Q4)**
- Mousavi et al., *Discrete Audio Tokens: More Than a Survey!* (2025) — https://arxiv.org/abs/2506.10274
- Sugiura et al., *Llama-Mimi* (2025) — https://arxiv.org/abs/2509.14882
- Défossez et al., *Moshi* (2024; Tables 5–8) — https://arxiv.org/abs/2410.00037
- Rouard et al., *Continuous Audio Language Models (CALM)* (2025) — https://arxiv.org/abs/2509.06926
- Tseng et al., *TASTE* (2025) — https://arxiv.org/abs/2504.07053
- Yang et al., *LM-aware codec training (ARDDS)* (2026) — https://arxiv.org/abs/2609.04237

**Duplex data (Q5)**
- Jung et al., *Sommelier* (2026) — https://arxiv.org/abs/2603.25750
- Nakata et al., *DialogueSidon* (2026) — https://arxiv.org/abs/2604.09344
- Roy et al., *PersonaPlex* (2026; data-scale ablation) — https://arxiv.org/abs/2602.06053
- Li & Chu, *Decoupling Turn-Taking from Semantics* (2026) — https://arxiv.org/abs/2609.03321

**Post-training and evaluation (Q6)**
- Cao et al., *X-OPD* (Interspeech 2026) — https://arxiv.org/abs/2603.24596
- Wu et al., *Aligning Spoken Dialogue Models from User Interactions* (ICML 2025) — https://arxiv.org/abs/2506.21463
- *DuplexPO* (2026) — https://arxiv.org/abs/2607.07148
- *HiPLEX* (2026) — https://arxiv.org/abs/2610.07727
- *AdaptDuplex* (2026) — https://arxiv.org/abs/2609.29217
- Full-Duplex-Bench v1 / v1.5 / v2 / v3 — https://arxiv.org/abs/2503.04721, https://arxiv.org/abs/2507.23159, https://arxiv.org/abs/2510.07838, https://arxiv.org/abs/2604.04847
- *URO-Bench* — https://arxiv.org/abs/2502.17810
- *VoiceBench* — https://arxiv.org/abs/2410.17196

**Architectures (Q7)**
- Kuroki et al., *KAME* (2025) — https://arxiv.org/abs/2510.02327
- Chien et al., *MoshiRAG* (2026) — https://arxiv.org/abs/2604.12928
- Hu et al., *Frontend–backend tool calls in full-duplex speech models* (2026) — https://arxiv.org/abs/2609.19334
- Chiang et al., *STITCH* (2025) — https://arxiv.org/abs/2507.15375
- Arora et al., *Stream RAG* (2025) — https://arxiv.org/abs/2510.02044

*Caveats.*
- The shared web-search quota ran out partway through. Some Sep–Oct 2026 items were found through arXiv listings and read only as abstracts; they are marked.
- Absolute benchmark numbers from different papers use different protocols. Weigh within-paper deltas over cross-paper comparisons.
