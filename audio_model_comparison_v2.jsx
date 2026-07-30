const models = [
  {
    name: "Moshi",
    org: "Kyutai",
    date: "Sep 2024",
    params: "7B (Helium LLM)",
    paradigm: "Codec tokens (no mel)",
    tokenizer: "Mimi — custom streaming neural audio codec. Conv encoder + Transformer + conv decoder. Split RVQ: 1 semantic codebook + 7 acoustic codebooks. 12.5 Hz frame rate, 1.1 kbps. Trained with WavLM distillation for semantic grounding.",
    tokenizerShort: "Mimi codec (split RVQ, 8 codebooks, 12.5Hz)",
    waveformMethod: "Mimi codec decoder (conv + Transformer)",
    waveformDetail: "The Mimi decoder mirrors the encoder: convolutional upsampling layers (transposed convolutions) with an 8-layer Transformer in the middle. Takes dequantized RVQ latents and reconstructs 24 kHz waveform directly — no mel or separate vocoder. Fully causal/streaming with 80ms latency.",
    melInvolved: "No",
    vocoderType: "Codec decoder (conv + Transformer hybrid)",
    streamingStrategy: "Fully causal. 80ms frame size = 80ms algorithmic latency. Encoder and decoder both stream natively. No stitching needed — continuous latent stream.",
    outputSR: "24 kHz",
    highlight: "First real-time full-duplex speech LLM. Codec-native: never touches mel.",
    family: "codec",
  },
  {
    name: "GLM-4-Voice",
    org: "Zhipu AI",
    date: "Oct 2024",
    params: "9B (GLM-4-9B)",
    paradigm: "Semantic tokens → Flow Matching → Mel → Vocoder",
    tokenizer: "Supervised speech tokenizer — VQ added to Whisper-large-v3 encoder. Codebook size scales with sampling rate reduction. 12.5 Hz discrete tokens. Bilingual (Chinese + English).",
    tokenizerShort: "VQ-Whisper tokenizer (12.5 Hz)",
    waveformMethod: "CosyVoice-based decoder (Flow Matching + vocoder)",
    waveformDetail: "GLM-4-Voice-Decoder is retrained from CosyVoice architecture, supporting streaming inference. It uses a conditional flow matching model to convert semantic tokens into mel spectrograms, then a vocoder synthesizes waveform. Can start generating with as few as 10 audio tokens, reducing conversation latency significantly.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching (CosyVoice-based) + vocoder",
    streamingStrategy: "Streaming inference with 10-token minimum to start. Based on CosyVoice's chunk-aware causal flow matching. Low-latency streaming enabled by early token emission.",
    outputSR: "22.05 kHz",
    highlight: "First open-source end-to-end voice chatbot with emotion/dialect control. CosyVoice decoder enables streaming from just 10 tokens.",
    family: "flow+vocoder",
  },
  {
    name: "CosyVoice 2",
    org: "Alibaba FunAudioLLM",
    date: "Dec 2024",
    params: "0.5B (Qwen2.5-0.5B backbone)",
    paradigm: "Text → Semantic tokens → Flow Matching → Mel → Vocoder",
    tokenizer: "FSQ-based supervised speech tokenizer (Finite Scalar Quantization). 100% codebook utilization. Trained on aligned ASR data. 50 Hz frame rate. No forced alignment or phonemizers needed.",
    tokenizerShort: "FSQ speech tokenizer (50 Hz)",
    waveformMethod: "Chunk-aware causal Flow Matching → HiFi-GAN",
    waveformDetail: "LLM backbone (Qwen2.5-0.5B) autoregressively generates semantic tokens from text. A chunk-aware causal flow matching model (CFM with UNet-like Transformer blocks) converts tokens → mel spectrograms. HiFi-GAN vocoder then generates waveform. Supports both streaming and non-streaming in a single model.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching (chunk-aware CFM) + HiFi-GAN",
    streamingStrategy: "Chunk-aware causal flow matching with look-ahead convolution. Streaming mode achieves sub-100ms latency with near-lossless quality. KV cache + SDPA optimizations.",
    outputSR: "22.05 kHz",
    highlight: "Foundation TTS — reused by GLM-4-Voice, MiniCPM-o, Step-Audio 2. Most popular open-source TTS (~19k GitHub stars). Human-parity naturalness.",
    family: "flow+vocoder",
  },
  {
    name: "Step-Audio (v1)",
    org: "StepFun",
    date: "Feb 2025",
    params: "130B LLM + 3B speech decoder",
    paradigm: "Tokens → Flow Matching → Mel → Vocoder",
    tokenizer: "Dual-codebook: linguistic tokenizer (16.7 Hz, 1024-entry codebook) + semantic/acoustic tokenizer (25 Hz, 4096-entry codebook). Interleaved at 2:3 ratio.",
    tokenizerShort: "Dual codebook (linguistic + acoustic, 2:3 interleave)",
    waveformMethod: "Flow Matching DiT → BigVGAN v2",
    waveformDetail: "A 3B speech decoder LLM processes the dual-codebook tokens. A Flow Matching module (DiT architecture) generates mel spectrograms conditioned on tokens, reference audio, and speaker embeddings. BigVGAN v2 converts mel → 24 kHz waveform. Classic two-stage: flow matching for high-quality mel, GAN vocoder for fast waveform.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching DiT + BigVGAN v2 (GAN)",
    streamingStrategy: "Streaming-aware with speculative response generation (40% commit rate). Text-based context management with 14:1 compression.",
    outputSR: "24 kHz",
    highlight: "Largest LLM backbone (130B). Hybrid flow+GAN gives high fidelity.",
    family: "flow+vocoder",
  },
  {
    name: "Sesame CSM",
    org: "Sesame AI",
    date: "Mar 2025",
    params: "1B / 3B / 8B (Llama backbone)",
    paradigm: "Codec tokens (no mel)",
    tokenizer: "Mimi split-RVQ tokenizer (same as Moshi). 1 semantic codebook + N-1 acoustic codebooks at 12.5 Hz. Interleaved text + audio token input to backbone.",
    tokenizerShort: "Mimi codec (split RVQ, 12.5 Hz)",
    waveformMethod: "Mimi codec decoder (inv. quant → Transformer → CNN)",
    waveformDetail: "Two-stage generation: Llama backbone generates semantic (first) codebook, then smaller Llama audio decoder predicts remaining acoustic codebooks. RVQ codes fed through Mimi's decoder: inverse quantization → Transformer → transposed convolutions → 24 kHz waveform. No mel spectrogram involved.",
    melInvolved: "No",
    vocoderType: "Mimi codec decoder (Transformer + CNN)",
    streamingStrategy: "Context-aware: processes interleaved conversation history (text + audio tokens from prior turns). Mimi decoder is causal/streaming. Fine-tuned variant used for interactive demo.",
    outputSR: "24 kHz",
    highlight: "Focus on 'voice presence' — conversational context shapes prosody. Llama backbone + Mimi codec. Open-sourced CSM-1B.",
    family: "codec",
  },
  {
    name: "Qwen2.5-Omni",
    org: "Alibaba Qwen",
    date: "Mar 2025",
    params: "7B",
    paradigm: "Speech tokens → DiT → Mel → Vocoder",
    tokenizer: "qwen-tts-tokenizer — custom speech codec. Dual-track autoregressive Talker generates tokens from Thinker hidden states. End-to-end trained without forced alignment.",
    tokenizerShort: "qwen-tts-tokenizer",
    waveformMethod: "Sliding-window DiT (Flow Matching) → BigVGAN",
    waveformDetail: "Speech tokens decoded via DiT using Flow Matching → mel spectrograms. Sliding-window block attention (4-block receptive field: 2 lookback + 1 lookahead) for streaming chunk generation. BigVGAN converts mel chunks to waveform, also using fixed receptive field.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "DiT (Flow Matching) + BigVGAN",
    streamingStrategy: "Block-based sliding window. DiT generates mel in chunks with limited receptive field. BigVGAN also chunked. True streaming with controlled latency.",
    outputSR: "24 kHz",
    highlight: "Thinker-Talker architecture. End-to-end joint training. TMRoPE for audio-video sync.",
    family: "flow+vocoder",
  },
  {
    name: "Kimi-Audio",
    org: "Moonshot AI",
    date: "Apr 2025",
    params: "7B (Qwen2.5-based)",
    paradigm: "Semantic tokens → Flow Matching → Mel → Vocoder",
    tokenizer: "GLM-4-Voice supervised tokenizer. VQ inside Whisper encoder. 12.5 Hz semantic tokens. Also extracts continuous acoustic vectors via Whisper for richer input.",
    tokenizerShort: "GLM-4-Voice tokenizer (VQ-Whisper, 12.5 Hz)",
    waveformMethod: "Flow Matching → BigVGAN",
    waveformDetail: "Audio detokenizer (MoonCast architecture): (1) flow-matching module converts 12.5 Hz semantic tokens → 50 Hz mel spectrograms, (2) BigVGAN vocoder → waveform. Chunk-wise autoregressive streaming with look-ahead to prevent boundary artifacts.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching + BigVGAN",
    streamingStrategy: "Chunk-wise autoregressive streaming with look-ahead. Each chunk decoded with small future context window. Solves boundary discontinuity without explicit cross-fade.",
    outputSR: "24 kHz",
    highlight: "13M hours pretraining. Elegant chunk+look-ahead streaming. Hybrid input (continuous + discrete).",
    family: "flow+vocoder",
  },
  {
    name: "Step-Audio 2",
    org: "StepFun",
    date: "~Aug 2025",
    params: "7B (Qwen2.5-based)",
    paradigm: "Tokens → Flow Matching → Mel → Vocoder",
    tokenizer: "S3Tokenizer (CosyVoice family). 50 Hz discrete codes from 16 kHz speech. Text and audio tokens interleaved at fixed ratio.",
    tokenizerShort: "S3Tokenizer (CosyVoice, 50 Hz)",
    waveformMethod: "Flow Matching + HiFi-GAN",
    waveformDetail: "S3Tokenizer for tokenization. Audio detokenizer uses flow-matching/diffusion to predict mel, then HiFi-GAN for mel→waveform. End-to-end: no ASR+LLM+TTS pipeline.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching + HiFi-GAN",
    streamingStrategy: "True end-to-end streaming. Continuous input (raw waveform) → discrete output (acoustic tokens). Interleaved modality alignment.",
    outputSR: "24 kHz",
    highlight: "Open-source (Apache 2.0). End-to-end replaces v1's pipeline. 7B vs 130B.",
    family: "flow+vocoder",
  },
  {
    name: "Qwen3-Omni",
    org: "Alibaba Qwen",
    date: "Sep 2025",
    params: "30B total / 3B active (MoE)",
    paradigm: "Multi-codebook tokens → Causal ConvNet",
    tokenizer: "Qwen-TTS-Tokenizer-12Hz — 12.5 Hz, 16-layer multi-codebook. First codebook = semantic; rest = acoustic. Talker predicts zeroth codebook, MTP module generates residuals.",
    tokenizerShort: "Qwen-TTS-Tokenizer-12Hz (16-layer multi-codebook)",
    waveformMethod: "Lightweight causal ConvNet (Code2Wav)",
    waveformDetail: "Replaces Qwen2.5-Omni's DiT+BigVGAN with lightweight causal ConvNet (Code2Wav). 16-layer multi-codebook captures rich acoustic detail, so reconstruction is simplified. Streams from first codec frame. Much lower FLOPs and latency than DiT.",
    melInvolved: "No — direct codec→waveform",
    vocoderType: "Causal ConvNet (Code2Wav)",
    streamingStrategy: "Frame-by-frame from first codec frame. MTP outputs all residual codebooks per step, Code2Wav immediately synthesizes. No chunking needed. 234ms first-packet latency.",
    outputSR: "24 kHz",
    highlight: "Eliminated mel+DiT bottleneck. ConvNet decoder dramatically faster. MoE. 234ms first-packet.",
    family: "codec",
  },
  {
    name: "GPT-4o Realtime",
    org: "OpenAI",
    date: "Oct 2024 → GA Aug 2025",
    params: "Undisclosed",
    paradigm: "Undisclosed (end-to-end, native audio)",
    tokenizer: "Undisclosed. GPT-4o natively processes audio tokens — not Whisper+GPT+TTS pipeline. Likely proprietary audio codec integrated into the model.",
    tokenizerShort: "Proprietary (native audio tokens)",
    waveformMethod: "Undisclosed (proprietary)",
    waveformDetail: "Architecture not published. GPT-4o processes audio natively within a single model. Realtime API streams audio I/O over WebRTC/WebSocket. Likely uses learned audio codec with neural decoder, but specifics unknown.",
    melInvolved: "Unknown",
    vocoderType: "Proprietary (not disclosed)",
    streamingStrategy: "WebRTC/WebSocket streaming. Server-side VAD. Sub-second response latency. Function calling during audio interaction.",
    outputSR: "24 kHz (PCM16/G.711)",
    highlight: "First major commercial native-audio LLM. Industry benchmark. Proprietary — no published architecture.",
    family: "proprietary",
  },
  {
    name: "Grok Voice",
    org: "xAI",
    date: "Dec 2025 (API)",
    params: "Undisclosed",
    paradigm: "Undisclosed (end-to-end, single model)",
    tokenizer: "Undisclosed. Processes speech and generates expressive output within single model. Handles paralinguistic cues natively.",
    tokenizerShort: "Proprietary (single-model audio)",
    waveformMethod: "Undisclosed (proprietary)",
    waveformDetail: "Architecture not published. Single-model processing rather than STT→LLM→TTS. 80+ voices, 28 languages, voice cloning from ~1 min audio. <1s time-to-first-audio. Top Big Bench Audio ranking.",
    melInvolved: "Unknown",
    vocoderType: "Proprietary (not disclosed)",
    streamingStrategy: "WebSocket streaming. Server VAD. <700ms response latency. Same infrastructure as Tesla in-car and Starlink.",
    outputSR: "24 kHz (WAV/MP3/μ-law)",
    highlight: "Sub-second latency. Voice cloning <2 min. Powers Tesla voice. No architecture published.",
    family: "proprietary",
  },
  {
    name: "MiMo-Audio",
    org: "Xiaomi",
    date: "Dec 2025",
    params: "7B (MiMo-7B-Base)",
    paradigm: "Codec tokens → Vocos vocoder",
    tokenizer: "MiMo-Audio-Tokenizer — from scratch. Transformer encoder (bidir, 1280d) + 20-layer RVQ (first 2: 1024 entries, rest: 128). Causal Transformer decoder. 25 Hz. Layer-3 hidden states summed with final layer.",
    tokenizerShort: "MiMo-Audio-Tokenizer (20-layer RVQ, 25 Hz)",
    waveformMethod: "Vocos-style vocoder (Transformer backbone)",
    waveformDetail: "Audio decoder reconstructs continuous representation from RVQ tokens. Vocos-style vocoder (Transformer backbone instead of ConvNeXt) operates in STFT domain — predicts magnitude+phase → iSTFT to waveform. Efficient via sequence packing.",
    melInvolved: "No (STFT domain, not mel)",
    vocoderType: "Vocos variant (Transformer, STFT-domain)",
    streamingStrategy: "Patch encoder: 4 timesteps → 6.25 Hz for LLM. Patch decoder: generates 25 Hz RVQ via delayed-generation scheme (staggered codebook layers).",
    outputSR: "24 kHz",
    highlight: "100M+ hours training. Few-shot emergent capabilities. STFT-domain avoids mel bottleneck.",
    family: "stft",
  },
  {
    name: "MiniCPM-o 4.5",
    org: "OpenBMB / Tsinghua",
    date: "Feb 2026",
    params: "9B total",
    paradigm: "Speech tokens → Flow Matching → Waveform",
    tokenizer: "CosyVoice2-based tokenizer. LLM hidden states → TTS projector → TTS Llama (20-layer) → interleaved text+speech tokens. TDM syncs all streams on ms timeline.",
    tokenizerShort: "CosyVoice2 tokenizer + TTS Llama (20L)",
    waveformMethod: "Streaming Flow Matching (Token2Wav)",
    waveformDetail: "Streaming flow-matching decoder (Token2Wav from CosyVoice2/Step-Audio2) converts speech tokens → 24 kHz waveform. Interleaves text+speech for full-duplex. End-to-end differentiable: encoders → LLM → speech decoder all connected via hidden states.",
    melInvolved: "Likely (flow matching → mel → vocoder)",
    vocoderType: "Flow Matching (Token2Wav from CosyVoice2/Step-Audio2)",
    streamingStrategy: "Full-duplex via TDM. Input (video+audio) and output (text+speech) simultaneous. 1 Hz proactive decision-making.",
    outputSR: "24 kHz",
    highlight: "Full-duplex omni-modal. Proactive interaction. 9B on-device. End-to-end gradient propagation.",
    family: "flow+vocoder",
  },
  {
    name: "Raon-Speech",
    org: "KRAFTON",
    date: "Apr 2026",
    params: "9B (Qwen3 backbone)",
    paradigm: "Codec tokens (no mel)",
    tokenizer: "Mimi codec with 32 quantizers (extended from 8). ECAPA-TDNN for speaker embeddings. 3-stage: encoder-decoder alignment → SpeechLM pretraining → multi-reward DPO.",
    tokenizerShort: "Mimi codec (32 quantizers) + ECAPA-TDNN",
    waveformMethod: "Mimi codec decoder",
    waveformDetail: "Mimi codec decoder: inverse quantization of 32-codebook RVQ → Transformer → transposed convolutions → waveform. Extended 32 codebooks (vs Moshi's 8) provide richer acoustic representation. Speaker conditioning via ECAPA-TDNN for voice cloning.",
    melInvolved: "No",
    vocoderType: "Mimi codec decoder (Transformer + CNN, 32 codebooks)",
    streamingStrategy: "Causal streaming with sub-second TTFT. Faster-than-real-time on single GPU. SpeechChat variant: full-duplex with backchanneling and interruption handling.",
    outputSR: "24 kHz",
    highlight: "#1 open <10B speech LLM (EN+KR). 32-codebook Mimi for richer reconstruction. Multi-reward DPO.",
    family: "codec",
  },
  {
    name: "TML-Interaction-Small",
    org: "Thinking Machines Lab",
    date: "May 2026",
    params: "276B / 12B active (MoE)",
    paradigm: "dMel input → Flow head output (encoder-free)",
    tokenizer: "No codec or tokenizer. Audio input = dMel features through lightweight embedding layer. No standalone encoder (no Whisper). Images = 40×40 patches via hMLP. All co-trained from scratch.",
    tokenizerShort: "dMel embeddings (no codec, encoder-free)",
    waveformMethod: "Flow head (jointly trained decoder)",
    waveformDetail: "Encoder-free early fusion: audio output decoded by a flow head from Transformer hidden states. No separate codec, no Mimi, no BigVGAN — flow head is part of unified model. All components jointly trained from scratch in single gradient flow. Likely targets mel/STFT internally.",
    melInvolved: "Likely (flow head → mel/STFT internally)",
    vocoderType: "Flow head (jointly trained, not fully disclosed)",
    streamingStrategy: "200ms micro-turns: continuously processes 200ms input while generating 200ms output, interleaved on same clock. Dual-model: interaction model (real-time) + background model (deep reasoning/tools) share context asynchronously.",
    outputSR: "Not disclosed",
    highlight: "First 'interaction model'. 200ms micro-turns. Encoder-free fusion. Full-duplex voice+video+text+tools.",
    family: "flow",
  },
];

const familyColors = {
  codec: { bg: "#1a2a1a", fg: "#7cc87c", label: "Codec-native" },
  "flow+vocoder": { bg: "#2a2218", fg: "#c8a060", label: "Flow + Vocoder" },
  stft: { bg: "#1a1a2a", fg: "#7c8cc8", label: "STFT-domain" },
  flow: { bg: "#2a1a2a", fg: "#c87cc8", label: "Flow (direct)" },
  proprietary: { bg: "#1e1e1e", fg: "#888888", label: "Proprietary" },
};

const columns = [
  { key: "paradigm", label: "Overall Paradigm" },
  { key: "tokenizerShort", label: "Audio Tokenizer" },
  { key: "waveformMethod", label: "Token → Waveform" },
  { key: "melInvolved", label: "Mel?" },
  { key: "vocoderType", label: "Vocoder Type" },
  { key: "streamingStrategy", label: "Streaming / Stitching" },
  { key: "outputSR", label: "SR" },
];

function AudioModelComparison() {
  const [expanded, setExpanded] = React.useState(null);
  const [view, setView] = React.useState("table");
  const [familyFilter, setFamilyFilter] = React.useState("all");

  const toggleExpand = (idx) => setExpanded(expanded === idx ? null : idx);
  const filtered = familyFilter === "all" ? models : models.filter((m) => m.family === familyFilter);

  return (
    <div style={{ fontFamily: "'IBM Plex Mono', 'SF Mono', monospace", background: "#0a0a0f", color: "#e0ddd5", minHeight: "100vh", padding: "20px 14px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; }
        .ht { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: #f0ece4; margin: 0 0 2px 0; letter-spacing: -0.5px; }
        .hs { font-size: 11.5px; color: #6b6860; margin: 0 0 14px 0; }
        .ctrls { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; align-items: center; }
        .tb { display: flex; gap: 2px; background: #151520; border-radius: 5px; padding: 2px; }
        .tbtn { padding: 5px 12px; border: none; background: transparent; color: #6b6860; font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; cursor: pointer; border-radius: 3px; transition: all 0.15s; white-space: nowrap; }
        .tbtn.a { background: #1e1e2e; color: #c8e060; }
        .tbtn:hover:not(.a) { color: #a0a090; }
        .fp { display: flex; flex-wrap: wrap; gap: 4px; margin-left: 8px; }
        .pl { padding: 4px 10px; border: 1px solid #1e1e2e; border-radius: 12px; font-size: 10px; cursor: pointer; transition: all 0.15s; background: transparent; font-family: 'IBM Plex Mono', monospace; }
        .pl:hover { border-color: #3a3a4e; }
        .pl.a { border-color: currentColor; }
        .cb { font-size: 10px; color: #4a4840; margin-left: 8px; }
        .tw { overflow-x: auto; border: 1px solid #1e1e2e; border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; min-width: 1100px; font-size: 10.5px; }
        thead th { background: #12121c; color: #8b8878; font-weight: 500; text-align: left; padding: 7px 9px; border-bottom: 1px solid #1e1e2e; position: sticky; top: 0; font-size: 9px; text-transform: uppercase; letter-spacing: 0.7px; z-index: 1; }
        thead th:first-child { min-width: 115px; position: sticky; left: 0; z-index: 3; }
        tbody tr { border-bottom: 1px solid #131318; transition: background 0.12s; }
        tbody tr:hover { background: #13131f; }
        td { padding: 7px 9px; vertical-align: top; line-height: 1.4; }
        td:first-child { position: sticky; left: 0; background: #0a0a0f; z-index: 1; border-right: 1px solid #1e1e2e; }
        tbody tr:hover td:first-child { background: #13131f; }
        .mn { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #f0ece4; font-size: 11.5px; }
        .mo { color: #6b6860; font-size: 9px; }
        .md { color: #4a4840; font-size: 9px; }
        .tg { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: 500; }
        .tn { background: #1a2a1a; color: #7cc87c; }
        .ty { background: #2a2218; color: #c8a060; }
        .tm { background: #1e1e2e; color: #8888aa; }
        .cg { display: flex; flex-direction: column; gap: 8px; }
        .cd { background: #111118; border: 1px solid #1e1e2e; border-radius: 6px; overflow: hidden; cursor: pointer; transition: border-color 0.15s; }
        .cd:hover { border-color: #2a2a3e; }
        .ch { padding: 10px 13px; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
        .cl { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; min-width: 0; }
        .cp { font-size: 10px; color: #6b6860; padding: 2px 7px; background: #0a0a0f; border-radius: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 280px; }
        .fd { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
        .ca { color: #4a4840; font-size: 15px; transition: transform 0.15s; flex-shrink: 0; user-select: none; }
        .ca.o { transform: rotate(90deg); }
        .cbdy { padding: 0 13px 13px; display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 7px; }
        .cf { padding: 7px 9px; background: #0d0d15; border-radius: 4px; }
        .cfl { font-size: 8.5px; text-transform: uppercase; letter-spacing: 0.7px; color: #5b5850; margin-bottom: 3px; }
        .cfv { font-size: 10.5px; line-height: 1.5; color: #c8c4b8; }
        .chi { grid-column: 1 / -1; padding: 7px 9px; background: #0f1018; border-left: 2px solid #c8e060; border-radius: 0 4px 4px 0; }
        .leg { margin-top: 14px; padding: 11px 13px; background: #111118; border: 1px solid #1e1e2e; border-radius: 6px; font-size: 10px; line-height: 1.65; color: #8b8878; }
        .leg strong { color: #c8c4b8; }
      `}</style>

      <h1 className="ht">Audio Model Waveform Generation — Comprehensive Comparison</h1>
      <p className="hs">{models.length} models · How each converts internal representation → raw audio waveform</p>

      <div className="ctrls">
        <div className="tb">
          <button className={`tbtn ${view === "cards" ? "a" : ""}`} onClick={() => setView("cards")}>Cards</button>
          <button className={`tbtn ${view === "table" ? "a" : ""}`} onClick={() => setView("table")}>Table</button>
        </div>
        <div className="fp">
          <button className={`pl ${familyFilter === "all" ? "a" : ""}`} style={{ color: "#e0ddd5" }} onClick={() => setFamilyFilter("all")}>All</button>
          {Object.entries(familyColors).map(([k, v]) => (
            <button key={k} className={`pl ${familyFilter === k ? "a" : ""}`} style={{ color: v.fg }} onClick={() => setFamilyFilter(k)}>{v.label}</button>
          ))}
        </div>
        <span className="cb">{filtered.length} shown</span>
      </div>

      {view === "table" ? (
        <div className="tw">
          <table>
            <thead><tr><th>Model</th>{columns.map((c) => <th key={c.key}>{c.label}</th>)}</tr></thead>
            <tbody>
              {filtered.map((m, i) => (
                <tr key={i}>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                      <span className="fd" style={{ background: familyColors[m.family]?.fg }} />
                      <div>
                        <div className="mn">{m.name}</div>
                        <div className="mo">{m.org} · {m.date}</div>
                        <div className="md">{m.params}</div>
                      </div>
                    </div>
                  </td>
                  {columns.map((c) => (
                    <td key={c.key}>
                      {c.key === "melInvolved" ? (
                        <span className={`tg ${m[c.key] === "No" || m[c.key].startsWith("No") ? "tn" : m[c.key].startsWith("Likely") || m[c.key] === "Unknown" ? "tm" : "ty"}`}>{m[c.key]}</span>
                      ) : m[c.key]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="cg">
          {filtered.map((m) => {
            const fi = models.indexOf(m);
            return (
              <div className="cd" key={fi} onClick={() => toggleExpand(fi)}>
                <div className="ch">
                  <div className="cl">
                    <span className="fd" style={{ background: familyColors[m.family]?.fg }} />
                    <div style={{ minWidth: 0 }}>
                      <span className="mn">{m.name}</span>
                      <span className="mo" style={{ marginLeft: 6 }}>{m.org} · {m.date} · {m.params}</span>
                    </div>
                    <div className="cp">{m.waveformMethod}</div>
                  </div>
                  <span className={`ca ${expanded === fi ? "o" : ""}`}>›</span>
                </div>
                {expanded === fi && (
                  <div className="cbdy">
                    <div className="cf"><div className="cfl">Audio Tokenizer</div><div className="cfv">{m.tokenizer}</div></div>
                    <div className="cf"><div className="cfl">Waveform Generation</div><div className="cfv">{m.waveformDetail}</div></div>
                    <div className="cf"><div className="cfl">Vocoder Type</div><div className="cfv">{m.vocoderType}</div></div>
                    <div className="cf"><div className="cfl">Mel Intermediate?</div><div className="cfv"><span className={`tg ${m.melInvolved === "No" || m.melInvolved.startsWith("No") ? "tn" : m.melInvolved.startsWith("Likely") || m.melInvolved === "Unknown" ? "tm" : "ty"}`}>{m.melInvolved}</span></div></div>
                    <div className="cf"><div className="cfl">Streaming / Stitching</div><div className="cfv">{m.streamingStrategy}</div></div>
                    <div className="cf"><div className="cfl">Output</div><div className="cfv">{m.outputSR}</div></div>
                    <div className="chi"><div className="cfl">Key Insight</div><div className="cfv" style={{ color: "#c8e060" }}>{m.highlight}</div></div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="leg">
        <strong>Paradigm families (color-coded):</strong><br />
        <span style={{ color: "#7cc87c" }}>●</span> <strong>Codec-native</strong> — Moshi, Sesame CSM, Qwen3-Omni, Raon-Speech: rich RVQ tokens (8–32 codebooks) decoded directly by lightweight ConvNet or Transformer+CNN. Skips mel. Lowest latency path.<br />
        <span style={{ color: "#c8a060" }}>●</span> <strong>Flow + Vocoder</strong> — GLM-4-Voice, Step-Audio, Kimi-Audio, Qwen2.5-Omni, CosyVoice 2, MiniCPM-o: flow matching → mel → GAN vocoder (BigVGAN/HiFi-GAN). Proven, high-fidelity, but two-stage.<br />
        <span style={{ color: "#7c8cc8" }}>●</span> <strong>STFT-domain</strong> — MiMo-Audio: Vocos-style magnitude+phase → iSTFT. Avoids both mel and codec bottlenecks.<br />
        <span style={{ color: "#c87cc8" }}>●</span> <strong>Flow (direct)</strong> — Thinking Machines: encoder-free early fusion with jointly-trained flow head. No separate codec or vocoder.<br />
        <span style={{ color: "#888" }}>●</span> <strong>Proprietary</strong> — GPT-4o Realtime, Grok Voice: architecture not publicly documented.<br /><br />
        <strong>Key trends (2024→2026):</strong> The field moved from Flow+Vocoder toward Codec-native. Multi-codebook representations (16–32 layers) make mel spectrograms unnecessary, enabling causal frame-by-frame streaming with zero boundary stitching. CosyVoice 2 is the most reused component — its flow matching decoder appears in GLM-4-Voice, MiniCPM-o, and Step-Audio 2. Thinking Machines' encoder-free approach (May 2026) may represent the next shift: no codec, no mel, no standalone vocoder — just a single model with a flow head.
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<AudioModelComparison />);
