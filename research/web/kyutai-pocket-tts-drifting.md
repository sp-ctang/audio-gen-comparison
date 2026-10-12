# Kyutai: Pocket TTS with a drifting objective (+ related Kyutai/Gradium work)

**Claim labels**

- **[stated]** said by Kyutai or Gradium on the cited page;
- **[inferred]** my inference;
- **[third-party]** a source from neither organization.

All pages accessed 2026-10-12.

## Summary

- **What changed.** [stated] Kyutai swapped the one-step sampler-head objective of Pocket TTS, a continuous audio LM (CALM), from Lagrangian self-distillation (LSD) to **drifting** (arXiv 2602.04770).
  - They believe it is the first autoregressive model, and the first speech model, trained with drifting.
  - It matches LSD "on every metric we track" at the same inference cost, with no Jacobian-vector products (blog).
- **Release.** [stated] The model ships as `english_drifting_26-09`: a 6-layer student distilled from a 24-layer drifting teacher, with guidance baked in.
  - On LibriSpeech test-clean (1,127 items, guidance 1, temperature 0.3) it scores 0.90% WER / 4.37 UTMOS. The default LSD English model scores 0.90% / 4.36 (blog; PR #329).
- **Cost.** [stated] It needed two things:
  - a **learned kernel temperature**;
  - **≥ 128 rows per step**: with 64 rows it never reached high quality within 250k steps.

  Per-frame distance normalization is "a speed lever rather than a requirement". Drifting reaches the same quality as LSD at about **2× the training cost**.
- **Takeaway for us.** [inferred] The head objective is not where perceived quality comes from. The transferable lessons concern decoding knobs, training dynamics, and the need for a semantic latent space.

## Sources (URLs)

1. https://kyutai.org/blog/2026-09-28-pocket-tts-drifting/: "Building a Pocket TTS with a drifting objective" by Manu Orsini, Samy Aïmeur and Nicolas Dufour.
   - The page is dated 27 Sep 2026; the blog index lists 2026-09-28.
   - **Access note:** WebFetch returned only the title and navigation because the body renders client-side. I read the full text in a browser.
2. https://github.com/kyutai-labs/pocket-tts/pull/329: the merged PR "Drifting sampler head: training recipe and english_drifting_26-09".
3. https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/training/configs/drifting.yaml and https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/pocket_tts/config/english_drifting_26-09.yaml
4. https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md and the repo README (https://github.com/kyutai-labs/pocket-tts)
5. https://kyutai.org/blog/2026-01-13-pocket-tts/ (tech report); https://kyutai.org/blog/2026-08-25-pocket-tts-training/; https://kyutai.org/blog/2026-05-04-pocket-tts-multilingual/ (has no data or compute numbers)
6. [third-party] https://arxiv.org/abs/2602.04770: Deng, Li, Li, Du, He, "Generative Modeling via Drifting". It reports one-step ImageNet-256 FID 1.54 (latent) and 1.61 (pixel).
7. **Not used:**
   - Community forks found by search (github.com/kadirnar/drifting-tts, github.com/ishine/drifting-tts) and aggregator pages.
   - A WebFetch summary claiming the default English model has "18 layers". The README does not support it.
   - Gradium blog pages 2–4, which I did not browse one by one; I ran site-filtered searches instead.

## What it is / architecture

- **Pipeline.** [stated] (blog, tech report)
  - A Mimi-based codec turns 24 kHz audio into one **32-dim continuous latent per 80 ms frame** (12.5 Hz).
  - A backbone transformer reads the voice prompt, the text and the past latents, and outputs z_t.
  - A small MLP head maps (z_t, ε) to the next latent in a single forward pass. Only the head's objective changed; the codec and backbone architecture did not.
- **Released student.** [stated] The backbone has d_model 1024, 16 heads and 6 layers. The head has depth 6, width 512, and no time conditioning. The text lookup table has 4,000 entries, and the default temperature is 0.3 (config; PR).
  - Size: 90M generative model + 10M codec decoder = 100M, plus an 18M encoder that runs once per voice prompt (tech report).
- **The drifting objective.** [stated] The field is V(x) = attraction toward data − repulsion from the other generated samples, with kernel k(x, y) = exp(−‖x − y‖/τ).
  - The loss is E‖y − stopgrad(y + V(y))‖², so each gradient step moves samples along V: "no second network, no solver, no Jacobian-vector product".
  - For each frame, N generated candidates (32–64) are compared against **one** positive.
- **Why it is easier in CALM than for images.** [stated]
  - No feature encoder is needed. Mimi distils WavLM, so L2 distance between latents is meaningful (ABX ≈ 8%).
  - The next-frame conditional distribution is "heavily-conditioned, low-dimensional, mostly-unimodal".
  - The catch: there is only one positive per context.
- **Recipe.** [stated] Three new ingredients:
  - **Learned τ.** An auxiliary loss trains τ to maximize the log kernel mass on the true positive. τ anneals to 0.055–0.057 in every run; starting at 1.0 is fastest.
  - **Per-frame normalization.** Each context's distances are divided by their own mean (the paper uses one scalar per batch).
  - **128 rows per step.**

  Pocket TTS's existing sampler tricks are kept as they are:
  - Gaussian temperature (scale ε by √t);
  - latent CFG: z = z_uncond + γ(z_cond − z_uncond);
  - latent distillation, with the head left untouched.

## Training data & compute

- **Blog ablations.** [stated] All runs use the "same 24-layer backbone, the same data and the same recipe" on 4 H100s. They are evaluated every 25k steps, with UTMOS reported as mean ± sd over 3 seeds.
  - The blog does not name the main corpus. One ablation table mentions HiFiTTS-2, "our full TTS corpus" and a 2 kh HiFiTTS-2 subset.
- **`drifting.yaml` teacher config.** [stated]
  - 24 layers from scratch; effective batch 128; 400k steps.
  - AdamW at LR 2e-4, **held constant**: "annealing the learning rate before that jump stopped it". Warmup 4k steps, betas (0.9, 0.999).
  - Weight decay 0, because it "would also decay the learned temperature".
  - Clip 3.0; EMA 0.999; audio ≤ 60 s, voice prompt ≤ 5 s; text and voice dropout 0.2.
  - τ starts at 10, with 64 negatives.
  - The default data is the 200 h quickstart subset; the "reference numbers need --hours 2000".
  - [inferred] This shipped config (τ₀ = 10, 64 negatives) matches the blog's *earlier* recipe, not its best one (τ₀ = 1.0, 32 negatives).
- **PR #329.** [stated] The teacher reaches 1.01% WER / 4.26 UTMOS (cfg 1.5, temperature 0.3). The student is CFG-distilled at a guidance of 2.0.
- **LSD reference.** [stated] From the training README; LibriSpeech test-clean, cfg 2.0, temperature 0.3:

  | Model | Data | WER | SIM | UTMOS |
  |---|---|---|---|---|
  | 24 layers | 31,700 h HiFiTTS-2 | 0.82% | 0.929 | 4.33 |
  | 24 layers | 2,000 h subset | 0.94% | 0.929 | 4.32 |
  | 16 layers | 31,700 h | 0.83% | 0.927 | 4.31 |
  | Distilled 6-layer student | — | 0.76% | 0.921 | 4.35 |

  Training on 2,000 h with 8×H100 takes ≈5.3 h to 200k steps; distillation takes ≈2.6 h.
- **Original Pocket TTS.** [stated] Trained on 88k h of public English (AMI, Earnings22, GigaSpeech, SPGISpeech, TED-LIUM, VoxPopuli, LibriHeavy, Emilia).
  - Data guidance: 100 h is the minimum and 1,000+ h gives a strong model. 2,000 h reaches production level.
  - Word alignment uses wav2vec2-CTC forced alignment; whisper-timestamped is "noisier".
  - Training on 8×H100 takes 10–20 h and costs "less than $200" on rented hardware. WER drops below 1% at ~50k steps, and the voice "stops sounding synthetic" at ~200k steps.

## Results

- **Head-to-head.** [stated] Same backbone, 1,127 items, one-step sampling. LSD scores 0.91% WER / 4.31 UTMOS / 0.927 SIM; drifting scores 0.96% / 4.32 / 0.926. Kyutai calls the gap "within evaluation noise".
- **Ablations.** [stated] Speed is measured as the first evaluation reaching UTMOS ≥ 4.0.

  **τ initialization**

  | τ setting | First UTMOS ≥ 4.0 | Outcome |
  |---|---|---|
  | Learned, init 1.0 | 150k steps | UTMOS 4.29 at 250k |
  | Learned, init 0.05 | 200k steps | 4.16 |
  | Paper's fixed set | never by 250k | UTMOS 2.17 |
  | Fixed at the converged 0.056 | never | collapse: WER 15–20%, UTMOS 2.10 |
  | Init 15–100 | never | diverges |

  **Batch rows per step**

  | Rows per step | First UTMOS ≥ 4.0 | Wall-clock |
  |---|---|---|
  | 64 | never (UTMOS 3.77) | — |
  | 128 | 150k steps | 12.7 h |
  | 256 | 100k steps | 20 h |

  **Normalization:** per-frame reaches quality at 200k steps (UTMOS 4.29); per-batch at 300k (4.11).

  **Latent CFG γ** (WER):

  | γ | 1.0 | 1.5 | 2.0 | 3.0 |
  |---|---|---|---|---|
  | WER | 1.43% | 1.03% | 1.25% | 15.7% |

  At γ = 3.0 the EOS decision over-commits and utterances get truncated.

  **Sampling temperature 1.0 → 0.3:** WER 1.32 → 1.03%, UTMOS 4.06 → 4.29; SIM stays at 0.924–0.925.
- **Cost.** [stated] To first reach UTMOS ≥ 4.0 on 4 H100s:
  - LSD at 64 rows needs 4.9–5.6 h (11–13M rows);
  - drifting at 128 rows needs 10–12 h (16–19M rows).
- **Training dynamics.** [stated] "WER saturates within the first few percent of a run". UTMOS then plateaus near 3.7 and jumps above 4.2 "within a few thousand steps", under both objectives.

## Lessons for our setup

1. **The backbone drives perceived quality.** [stated] "The head objective decides how the head covers the conditional distribution; the backbone decides most of what we perceive."
   - [inferred] Our weak continuation is a backbone, data or decoding problem. Swapping the sampler objective will not fix semantics.
2. **Decoding knobs are cheap and large.** [stated] γ = 1.5 removed more than a quarter of the word errors. Temperature 0.3 removed a fifth and added 0.2 UTMOS. γ = 3 broke EOS.
   - [inferred] Sweep temperature, top-p and CFG before retraining.
   - [inferred] In duplex, an over-committed end decision would show up as premature turn-yielding. Monitor it when adding guidance.
3. **Quality arrives late, as a phase transition, and the LR schedule can kill it.** [stated] (LSD notes in the training README)
   - It needs LR 2e-4: 1e-4 "never gets there in 400k steps".
   - Batch size only shifts the timing. An effective batch of 48 arrives at 200–300k steps, against ≥ 64 as the stated requirement. Drifting needs 128.
   - Decaying the LR to zero before ~200k steps "stalls the transition".
   - Neither corpus size nor depth (16/24/32 layers) moves the transition.
   - [inferred] Check that our runs were not annealed on a plateau. Log a perceptual metric (UTMOS) and a semantic metric (transcript LLM perplexity) at every checkpoint.
4. **The latent space must be semantic.** [stated] One-step heads work because Mimi's WavLM-distilled latents make L2 distance meaningful.
   - [inferred] Moving to continuous outputs would mean replacing the Qwen3 vocoder on the output side with a semantically distilled VAE.
5. **Train big, then distill.** [stated] A 24-layer teacher distilled to 6 layers beats training a 6-layer model from scratch (student WER 0.76% vs teacher 0.82%).
   - [inferred] This is useful later for serving a duplex model at low latency.
6. **TTS is data-efficient; continuation without text is not.** [stated] Going from 2,000 h to 31,700 h barely changes text-conditioned TTS (WER 0.94 vs 0.82%).
   - [inferred] Free continuation without a text prompt, which is our problem, is the setting that needs large pretraining corpora (see the CALM and Moshi notes).
   - [inferred] Pocket TTS is single-speaker TTS and says nothing directly about duplex behaviour.

## Related Kyutai/Gradium work (brief)

- **CALM** (note `../papers/kyutai-calm.md`; https://arxiv.org/abs/2509.06926) and **DSM**, which covers Kyutai TTS/STT and the Unmute cascade (note `../papers/kyutai-dsm.md`; https://kyutai.org/blog/2025-05-22-unmute/).
- **Post-training for interactivity** (Jun 2026). Sources: https://kyutai.org/blog/2026-06-10-interactivity/, https://arxiv.org/abs/2606.11167.
  - [stated] GRPO on Moshi and PersonaPlex with rule-based rewards for pause handling, turn-taking, backchannels and interruptions, plus an LLM-judge content reward. Training segments were mined with VAD from the 4,000 h Seamless Interaction corpus.
  - [stated] Results for Moshi:
    - Full-Duplex-Bench v1: turn-taking TOR 0.739 → 0.958, pause TOR 0.528 → 0.463.
    - FDB v2: instruction-following 2.44 → 2.91.
  - [inferred] This is a ready template for our RL stage.
- **MoshiRAG** (Apr 2026). Sources: https://kyutai.org/blog/2026-04-30-moshi-rag/, https://arxiv.org/abs/2604.12928, https://github.com/kyutai-labs/moshi-rag.
  - [stated] Full-duplex Moshi emits a `<ret>` token to trigger an asynchronous text retrieval. The retrieval back end reads Moshi's own transcript plus the user's, transcribed by Gradium's streaming ASR.
  - The retrieved text is encoded and summed into the input embeddings, using the ≥ 2 s natural gap before the key information in an answer.
  - Training dialogues are synthetic and voiced with Kyutai's multi-stream TTS.
  - Results: TriviaQA 22.8 → 78.2, HaluEval 10.5 → 51.3; keyword delay rises 2.1 → 3.1 s.
  - [inferred] This is the most direct remedy for weak knowledge.
- **Hibiki-Zero** (Feb 2026). Sources: https://kyutai.org/blog/2026-02-12-hibiki-zero/, https://arxiv.org/abs/2602.11072, https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16.
  - [stated] A 3B multistream model on Mimi at 12.5 Hz. It uses sentence-level alignment only, then GRPO with BLEU-based intermediate rewards that trade quality against latency.
  - A new input language needed < 1,000 h. FR→EN reaches ASR-BLEU 28.7 at an average lag of 6.1 s.
  - [inferred] RL can replace word-level alignment engineering.
- **Gradium** (https://gradium.ai/).
  - [stated] Products: streaming TTS and STT, voice cloning, voice design, live translation, and **Phonon**, an on-device TTS.
  - [stated] Phonon has ~100M parameters, is "building on continuous audio language models", and reaches Seed-TTS-en WER 1.48% → 1.00% (May 2026) (https://gradium.ai/blog/evaluating-phonon).
  - [stated] Gradium's explainer (May 27, updated Sep 10, 2026) says "Speech-to-speech exists at the research and prototype stage", and lists Gradium's own offering as streaming STT + TTS (https://gradium.ai/content/cascaded-voice-agent-vs-speech-to-speech-2026).
  - I found no Gradium full-duplex release.
