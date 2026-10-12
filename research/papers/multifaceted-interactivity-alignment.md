# Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models

| Field | Value |
|---|---|
| Org | Kyutai and Gradium, Paris (p.1) |
| Date / arXiv | arXiv:2606.11167v1 [cs.CL], 9 Jun 2026 (p.1) |
| Category | method: RL (GRPO) post-training of full-duplex S2S models for interactivity |
| Backbone | No new model. Starts from released **Moshi** (7B speech-text LM) and **PersonaPlex** (a Moshi extension with text-prompt role control and audio-prompt voice cloning; size not restated) (p.6) |
| Audio input | Inherited from the base models (for Moshi, Mimi tokens; see `moshi.md`). Not restated in this paper |
| Audio output | Inherited. The RL objective uses only the probabilities of the parallel **text** stream; audio tokens are sampled but get no direct gradient (p.4) |
| Waveform decoder | Inherited. Rollouts are decoded to waveform so that VAD and ASR rewards can be computed (p.4) |
| Duplex mechanism | Inherited parallel two-stream modeling: π_θ(y, w \| x) predicts the agent's audio tokens y and text tokens w frame by frame, given the user stream x (p.3) |
| Open weights / code | Checkpoints released at huggingface.co/kyutai/moshika-rl-seamless and kyutai/personaplex-rl-seamless (p.1). Training code is not mentioned |
| Relevance to us | **High for the later post-training stage:** a cheap RL recipe whose data is mined from the same kind of two-channel human conversations we already have. **Low for our current problems:** it does not touch continuation quality or knowledge, and it assumes the model has a parallel text stream |

## TL;DR

- The method is **GRPO on short segments mined automatically** from two-channel human conversation corpora. It targets four axes: pause handling, turn-taking, backchanneling and user interruption (p.3–5).
  - Each axis has a rule-based reward computed by VAD on the decoded rollout audio.
  - Turn-taking and interruption also get an **ASR → LLM-judge content reward**.
- Only text-token probabilities enter the objective, because the text stream controls both the timing and the content of speech (p.4).
- **It is cheap:** 100 epochs × 32 segments × 16 rollouts on 32 H100s, LR 2e-7, KL coefficient β = 0.01 (p.6, p.13). That is about 51k rollouts per run (derived).
- **Moshi on Full-Duplex-Bench (FDB) v1, base → RL on Fisher / RL on Seamless (p.7 Tab.1):**
  - pause takeover rate (TOR) 0.445 → 0.226 / 0.307;
  - turn-taking TOR 0.739 → 0.966 / 0.958;
  - interruption latency 1.377 → 0.461 / 0.409 s;
  - post-interruption GPT-4o relevance score 3.440 → 3.575 / 3.630.
- **The gains carry over to real-time multi-turn evaluation (FDB v2).** For example, PersonaPlex + RL (Seamless) Daily turn-taking rises from 3.327 to 4.017 (p.7 Tab.2).
- Further findings:
  - The LLM-judge reward is the most important ingredient: removing it causes the largest ablation drop (p.8 Tab.3).
  - Seamless works better than Fisher as RL data.
  - RL on Fisher hurt PersonaPlex's safety behavior.
  - Speech quality (UTMOSv2) is unchanged (p.7–8, p.14).

## Architecture

There are no architectural changes. The base formulation is π_θ(y_{1:N}, w_{1:N} | x_{1:N}) = Π_n π_θ(y_n, w_n | x_≤n, y_<n, w_<n). Here x is the user's token stream, y the agent's audio tokens, and w a parallel text stream "to guide the semantic content" and "implicitly control when the model speaks" (p.3).

Model-specific inputs (p.6):
- **Moshi:** 3 s of silence is prepended to the input so Moshi can say its opening phrase first (following ASPIRin).
- **PersonaPlex:** the official standard text prompt plus a 3 s female voice prompt, used identically in training and inference.

## Training pipeline

### Stage 0: tokenizer / codec / decoder training
None: everything is inherited from the base models.

### Stage 1: supervised base training
Inherited: the method starts from the released Moshi and PersonaPlex checkpoints. The paper argues that RL is needed because token-likelihood training "does not directly optimize interaction-level behaviors", and because exposure bias makes models less robust in real conversations (p.1).

### Stage 2: GRPO interactivity alignment
- **Loop (p.3–4):**
  - For each sample, draw an axis ℓ ∈ {pause, turn, bc, int} (the sampling distribution is not reported).
  - Draw a segment from that axis's training set D_ℓ and encode speaker X's channel as the user input.
  - Sample G=16 completions, decode each one, and score it with R_ℓ.
  - The advantage is (r − mean) / std across the group.
- **Objective (p.4):** a PPO-style clipped surrogate plus β·KL against the frozen pre-RL model. The importance ratio and the objective use only the text tokens; the KL is the exact KL at each position.
- **Context (p.4, p.13):**
  - With probability 0.5, the recording that precedes the segment is prepended as context, with length drawn from U[0, l_max].
  - l_max rises linearly from 0 to 30 s over the 100 epochs.
  - The loss is masked on the context.
- **Data (p.5):** separate runs on two corpora:
  - **Fisher:** 2,000 h of telephone conversations.
  - **Seamless Interaction:** Improvised (1,300 h of actors) plus Naturalistic (2,700 h of general participants), combined.
  - Up to 2,000 segments per axis, so at most 8,000 per corpus (derived).
- **Rollout sampling (p.13):** temperature 0.7 for text and 0.8 for audio; top-k 250 for audio.
- **Optimizer (p.6, p.13):** AdamW (β 0.9/0.95), weight decay 0.1, gradient clipping at 2.0, LR 2e-7 with cosine schedule, clip ε = 0.2, KL β = 0.01.
- **Batching (p.13):** each epoch takes 32 segments (groups), one group per GPU, with G=16. Mini-batches of size 1 give 16 updates per group per epoch.
  - Derived per run: 3,200 segment draws, 51,200 rollouts and about 1,600 optimizer steps (assuming the per-GPU updates are synchronized under FSDP).
- **Trainable vs frozen:** not reported. Training uses FSDP, and gradients flow only through text-token log-probs.
- **Compute:** 32 × H100; wall-clock not reported.

## Data pipeline

- **Utterance annotation (p.4–5):**
  - Silero VAD on each channel gives inter-pausal units (IPUs) and silences.
  - Consecutive IPUs from the same speaker are merged into one utterance unless separated by more than 1.0 s of silence.
  - An internal silence of ≤1.0 s counts as a "pause".
- **Segment-extraction rules (p.4).** Minimum utterance duration τ_min is 4.0 s for pause, 5.0 s for turn and backchannel, and 3.0 s for interruption (p.5).
  - **Pause:** a single X utterance of at least τ_min that contains at least one internal pause, while Y stays silent.
  - **Turn:** an X utterance followed by a Y utterance, both at least τ_min, with a gap of ≤0.4 s.
  - **Backchannel:** an X utterance of at least τ_min during which Y produces only short (≤1 s) utterances.
  - **Interruption:** an X → Y → X → Y sequence in which X's second utterance starts before Y's first one ends; all four utterances at least τ_min.
- **Rewards (p.5).** VAD is run on the generated audio.
  - **R_pause** = −1 if any generated speech interval lasts more than 1 s, else 0.
  - **R_turn** = −d, where d is the delay from the end of X's turn to the first generated utterance longer than 1 s. If the model never speaks, d is the remaining segment duration.
  - **R_bc** = F1. Generated speech intervals of ≤1 s count as backchannels; one within ±1 s of a ground-truth backchannel is a true positive. Unmatched backchannels and longer "takeovers" are false positives.
  - **R_int** = −d, where d runs from the end of X's interrupting utterance to the model's next speech.
- **Content reward R_llm (turn and interruption axes only):**
  - Parakeet-TDT-0.6B-v2 transcribes both channels, and **Qwen3-235B-A22B** rates the response for contextual relevance and naturalness.
  - The text calls this a "three-point" scale (1–3, p.5–6). The prompt in Fig.3 uses 0 = irrelevant, silence or gibberish; 1 = generic; 2 = natural and grounded (p.13).
  - It is combined with the delay reward by reward-decoupled normalization (GDPO): each reward is standardized separately within the group, and the advantages are summed with equal weight (p.5).
- **No synthetic data:** there are no TTS dialogues and no noise augmentation. Real conversations are used "without artificial bias" (p.4).

## Compute summary

| Stage | Tokens / hours | Hardware × time | Notes |
|---|---|---|---|
| GRPO, per run (4 main runs = 2 models × 2 corpora, plus 7 ablation runs on Moshi/Fisher) | 100 × 32 × 16 = 51,200 rollouts (derived). Segments of ≥3–5 s plus ≤30 s context. ≤8,000 segments mined per corpus out of 2,000 h (Fisher) or 4,000 h (Seamless) | 32 × H100; time not reported | Each turn/interruption rollout also needs ASR and a 235B-MoE judge call; that cost is not reported |

## Preserving text intelligence / knowledge

**Safeguards:** the content reward from the LLM judge, the KL penalty to the reference model (β = 0.01), and a very small LR of 2e-7 (p.5–6, p.13).

**Evidence:**
- On FDB v1, the post-interruption GPT-4o score (0–5) rises for Moshi from 3.440 to 3.575 (Fisher) / 3.630 (Seamless), and for PersonaPlex from 4.500 to 4.520 / 4.533 (p.7 Tab.1).
- By contrast, the concurrent ASPIRin, which ran GRPO on Moshi without a content reward, reported a drop from 3.89 to 3.73 on a 1–5 scale (p.7).
- FDB v2 instruction-following rises in most conditions; for example, Moshi on Correction goes from 2.189 to 2.706 / 2.895 (p.7 Tab.2).

**Not measured:** knowledge benchmarks such as spoken QA or MMLU are not reported.

**Regression:** after RL on Fisher, PersonaPlex's FDB v2 Safety scores fall: turn 3.841 → 3.695, instruct 3.596 → 3.288, task 3.260 → 3.000 (p.7 Tab.2). Instead of refusing, the model gives fragmented cooperative replies such as "Yeah" (p.15).

## Key results

FDB v1 (p.7 Tab.1). TOR = takeover rate; latencies are in seconds.

| Model | Pause TOR, synth / Candor (↓) | BC TOR ↓ / Freq ↑ / JSD ↓ | Turn TOR ↑ / latency ↓ | Interrupt TOR ↑ / GPT-4o ↑ / latency ↓ |
|---|---|---|---|---|
| Moshi | 0.445 / 0.528 | 0.255 / 0.074 / 0.824 | 0.739 / 0.162 | 0.920 / 3.440 / 1.377 |
| + RL (Fisher) | 0.226 / 0.417 | 0.091 / 0.095 / 0.789 | 0.966 / 0.121 | 1.000 / 3.575 / 0.461 |
| + RL (Seamless) | 0.307 / 0.463 | 0.145 / 0.101 / 0.794 | 0.958 / 0.160 | 1.000 / 3.630 / 0.409 |
| PersonaPlex | 0.482 / 0.444 | 0.182 / 0.046 / 0.841 | 0.958 / 0.219 | 0.940 / 4.500 / 0.271 |
| + RL (Fisher) | 0.328 / 0.361 | 0.127 / 0.122 / 0.783 | 0.950 / 0.079 | 1.000 / 4.520 / 0.187 |
| + RL (Seamless) | 0.350 / 0.356 | 0.073 / 0.112 / 0.786 | 0.975 / 0.086 | 0.995 / 4.533 / 0.223 |
| dGSLM (cited) | 0.934 / 0.935 | 0.691 / 0.015 / 0.934 | 0.975 / 0.352 | 0.917 / 0.201 / 2.531 |
| Freeze-Omni (cited) | 0.642 / 0.481 | 0.636 / 0.001 / 0.997 | 0.336 / 0.953 | 0.867 / 3.615 / 1.409 |

FDB v2 (p.7 Tab.2): scored 1–5 by a Gemini 2.5 Flash judge, with GPT-Realtime as examiner, 60 s dialogues and fast pacing (p.6, p.14). T = turn-taking fluency, I = instruction following.

| Model | Daily T / I | Correction T / I / Task | Entity T / I / Task | Safety T / I / Task |
|---|---|---|---|---|
| Moshi | 3.284 / 2.221 | 3.248 / 2.189 / 2.340 | 3.951 / 2.537 / 2.440 | 3.839 / 2.831 / 2.720 |
| + RL (Fisher) | 3.397 / 2.502 | 3.957 / 2.706 / 2.820 | 4.110 / 2.626 / 2.640 | 3.858 / 3.058 / 2.820 |
| + RL (Seamless) | 3.442 / 2.615 | 4.003 / 2.895 / 3.300 | 3.965 / 2.609 / 2.740 | 4.161 / 3.503 / 3.440 |
| PersonaPlex | 3.327 / 2.861 | 3.803 / 2.945 / 3.080 | 3.748 / 3.130 / 3.200 | 3.841 / 3.596 / 3.260 |
| + RL (Fisher) | 3.627 / 2.915 | 3.840 / 3.026 / 3.500 | 4.055 / 3.562 / 3.700 | 3.695 / 3.288 / 3.000 |
| + RL (Seamless) | 4.017 / 3.197 | 4.501 / 3.369 / 3.620 | 4.647 / 4.059 / 3.840 | 4.511 / 3.780 / 3.280 |

Speech quality, overall UTMOSv2 (p.14 Tab.4): Moshi 2.56 → 2.58 (Fisher) / 2.64 (Seamless); PersonaPlex 2.56 → 2.51 / 2.62.

## Ablations & lessons

All ablations are on Moshi + RL (Fisher) (p.8 Tab.3). Metrics are FDB v1 pause TOR (Candor), backchannel JSD, turn latency and interruption GPT-4o, followed by FDB v2 Daily turn / instruct. The full model scores 0.42 / 0.79 / 0.12 s / 3.58, then 3.40 / 2.50.

- **Pause and turn data must be trained together.**
  - Without pause data, pause TOR jumps to 0.74 while latency drops to the best value, 0.05 s: the model "learns to speak at every opportunity".
  - Without turn data, pause TOR is the best (0.29), but latency rises to 0.30 s and GPT-4o falls to 3.28: the model becomes overly conservative (p.8).
- **The LLM-judge reward is the most important component.**
  - Without it: GPT-4o 3.05, Daily turn 3.00, instruct 2.18, and also pause TOR 0.48 and latency 0.17 s.
  - The authors call this the "largest degradation across nearly all metrics" (p.8).
  - This matches prior delay-only RL (ASPIRin), which degraded semantic quality (p.7).
- **Backchannel data buys backchannel timing at a small cost elsewhere.**
  - Without it, JSD is the worst (0.83) and latency is 0.22 s.
  - However, GPT-4o (3.67) and Daily turn (3.61) are *higher* than for the full model. The authors do not discuss this.
- **Interruption data protects content after barge-in.** Without it, GPT-4o is 3.42 and Daily instruct 2.24.
- **The context window and its schedule help with multi-turn generalization, but with trade-offs.**
  - With no context at all, Daily scores are 3.33 / 2.21, against 3.40 / 2.50 for the full model.
  - Without the schedule (always 30 s of context), pause TOR is 0.51 and instruct 2.41, but GPT-4o (3.70) and Daily turn (3.50) are better.
- **Training on short segments transfers to real-time multi-turn dialogue.** FDB v2 improves broadly (Tab.2).
  - In a case study, base Moshi responded only after about 6 s (App. D.1, p.15).
  - After RL, turn transitions took 0.76 s and 0.08 s (p.8 Fig.2).
- **The RL corpus matters.** Seamless beats Fisher on FDB v2 for both models; for example, PersonaPlex Correction turn is 4.501 with Seamless against 3.840 with Fisher (Tab.2). The authors credit "greater variety and more consistent dialogue structure" (p.7). On FDB v1, however, Fisher wins some Moshi metrics: pause TOR 0.226 vs 0.307, and turn latency 0.121 vs 0.160 s (Tab.1).
- **The RL corpus's interaction style leaks into behavior.** Fisher's "cooperative and casual" style overwrote PersonaPlex's safety behavior. The authors suggest safety-aware rewards (p.8–9, p.15).
- **Speech quality is preserved.** UTMOSv2 is flat. The authors credit three things: the rewards depend on VAD applied to the actual audio, the judge works from ASR transcripts, and the KL penalty keeps the model close to the reference (p.14).
- **The gains are not quite uniform.**
  - PersonaPlex + RL (Fisher) turn-taking TOR goes 0.958 → 0.950.
  - Moshi + RL (Seamless) turn latency is flat (0.162 → 0.160 s).
  - PersonaPlex + RL (Fisher) Safety turn-taking falls, even though the text claims turn-taking fluency improves in all four tasks (p.7).

## Implications for our setup

1. **Do this last.** The method aligns interactivity only. It reports no spoken-QA or MMLU results, and audio quality stays flat. It will not fix our weak continuation or weak knowledge; fix the base model first (see `moshi.md`).
2. **Still, plan for it.** Even full-recipe, SFT-only Moshi mis-times turns: pause TOR 0.445, turn TOR 0.739, interruption latency 1.377 s. A cheap RL pass fixes most of this, so we should expect our duplex model to need one too.
3. **Our 10K h corpus is the data source.**
   - Apply the same Silero-VAD utterance rules and τ_min thresholds to our two-channel recordings. We need separated channels per speaker.
   - The paper used at most 8,000 segments per corpus and only 3,200 draws per run. The bottleneck is reward engineering, not data.
4. **We need a parallel text stream.**
   - The objective uses only text-token probabilities, and the authors say the method "is not directly applicable" to models without a text stream (p.9).
   - If we add an inner-monologue stream (see `moshi.md`), the recipe ports directly. Otherwise we would have to define the importance ratio over audio tokens, which is untested.
5. **Starting configuration to copy:**
   - G=16, LR 2e-7 with cosine schedule, KL β = 0.01, clip ε = 0.2, gradient clipping at 2;
   - temperature 0.7 for text and 0.8 for audio, top-k 250;
   - 100 epochs × 32 groups on 32 H100s;
   - the 0→30 s context curriculum;
   - always the ASR + LLM-judge content reward with decoupled normalization;
   - pause and turn axes trained jointly.
6. **Audit our corpus's conversational style.** Casual, cooperative telephone-style data eroded refusals here, so add a safety reward or mix in more structured data.
7. **Evaluate on FDB v1 and v2.** For v1, use the corrected backchannel script, which resamples to 16 kHz (p.13). For v2, note that the authors suppressed the model's audio to the examiner while the examiner was speaking (p.14). Comparisons with other published numbers need this context.

## Open questions / unclear points

- **Sampling and data counts:** the probabilities for sampling each axis and the actual number of segments per axis (beyond "up to 2,000") are not reported.
- **Judge scale:** the LLM judge's scale is inconsistent: "three-point" or 1–3 in the text (p.5–6), 0–2 in the prompt (p.13).
- **Training details:** which parameters are trained is not stated. That the KL covers only the text distribution is implied but not stated.
- **Interruption reward:** R_int does not explicitly reward stopping speech during the barge-in. If the model talks through the interruption, it is not specified how its "next speech" is defined. Interruption TOR reaches 1.000, but a model that never yields might still score well.
- **Cost and evaluation:** wall-clock time and the cost of ASR and judge inference are not reported. There is no human evaluation; all metrics are automated (p.9).
- **Base checkpoint:** the paper says only "Moshi", but the release is named "moshika-…". The variant (and its voice) is not stated.
- **Reproducibility:** the base PersonaPlex numbers did not match the original paper's (pause TOR 0.358 and turn latency 0.170 there) (p.6 fn.7).
- **Not tested:** combining Fisher and Seamless, scaling the number of segments or epochs, and DPO or offline-preference baselines on the same base model.
