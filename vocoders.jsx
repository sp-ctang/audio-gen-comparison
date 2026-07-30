import { useState } from "react";

const models = [
  {
    name: "Moshi",
    org: "Kyutai",
    date: "Sep 2024",
    params: "7B (Helium LLM)",
    category: "codec",
    paradigm: "Codec tokens (no mel)",
    tokenizerShort: "Mimi codec (split RVQ, 12.5 Hz)",
    waveformMethod: "Mimi codec decoder (conv + Transformer)",
    waveformDetail: "The Mimi decoder mirrors the encoder: convolutional upsampling (transposed convolutions) with an 8-layer Transformer in the middle. Dequantized RVQ latents → 24 kHz waveform directly. No mel, no separate vocoder. Fully causal/streaming at 80ms latency.",
    melInvolved: "No",
    vocoderType: "Codec decoder (conv + Transformer hybrid)",
    streamingStrategy: "Fully causal. 80ms frame = 80ms algorithmic latency. Continuous latent stream, no stitching.",
    outputSR: "24 kHz",
    highlight: "First real-time full-duplex speech LLM. Codec-native: never touches mel.",
  },
  {
    name: "GPT-4o Realtime",
    org: "OpenAI",
    date: "Oct 2024 (GA Aug 2025)",
    params: "Undisclosed",
    category: "proprietary",
    paradigm: "Native audio tokens (end-to-end, proprietary)",
    tokenizerShort: "Proprietary (likely learned audio tokens)",
    waveformMethod: "Proprietary end-to-end (native audio generation)",
    waveformDetail: "Architecture undisclosed. OpenAI states it natively processes and generates audio — not a cascaded ASR→LLM→TTS pipeline. Directly outputs audio preserving emotion, emphasis, accents. Built on GPT-4o with specialized audio pretraining. Realtime API (WebRTC/WebSocket) and audio Chat Completions API.",
    melInvolved: "Unknown (proprietary)",
    vocoderType: "Proprietary (native audio generation)",
    streamingStrategy: "WebRTC/WebSocket real-time streaming. Sub-second latency. Full-duplex capable. No public internal details.",
    outputSR: "24 kHz (PCM16/G.711)",
    highlight: "First major commercial native-audio LLM. Set the bar for voice AI. Architecture unpublished.",
  },
  {
    name: "GLM-4-Voice",
    org: "Zhipu AI",
    date: "Dec 2024",
    params: "9B (GLM-4-9B)",
    category: "flow",
    paradigm: "Semantic tokens → Flow Matching (CosyVoice)",
    tokenizerShort: "Whisper + VQ (single codebook, 12.5 Hz, 175 bps)",
    waveformMethod: "CosyVoice-based Flow Matching decoder",
    waveformDetail: "GLM-4-Voice-Decoder is retrained from CosyVoice's flow-matching architecture. Takes discrete semantic tokens (single codebook, 12.5 Hz) → flow matching → mel → vocoder. Streaming starts with as few as 10 tokens. Block-wise: decoder processes first n×b seconds, using prior blocks as prompt.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching (CosyVoice-based) + vocoder",
    streamingStrategy: "Block-wise: min delay = block size b. Starts with just 10 tokens. Prior blocks serve as prompt for continuity.",
    outputSR: "22/24 kHz",
    highlight: "Ultra-low bitrate (175 bps). Emotion/dialect control via instruction. VQ-in-Whisper approach reused by Kimi-Audio.",
  },
  {
    name: "CosyVoice 2/3",
    org: "Alibaba FunAudioLLM",
    date: "Dec 2024 / Dec 2025",
    params: "0.5B (Qwen2 LLM backbone)",
    category: "flow",
    paradigm: "Text → LLM → Semantic tokens → Flow Matching → Mel → Vocoder",
    tokenizerShort: "FSQ speech tokenizer (25 Hz, supervised semantic)",
    waveformMethod: "Chunk-aware causal Flow Matching → HiFi-GAN vocoder",
    waveformDetail: "Two-stage: (1) Qwen2-0.5B LLM decodes text into supervised semantic tokens via Finite Scalar Quantization (FSQ). (2) Chunk-aware causal flow matching → mel conditioned on speaker embedding + reference audio. Pretrained vocoder converts mel → waveform. Unified streaming/non-streaming. RL-optimized in v3.",
    melInvolved: "Yes — 80-bin mel spectrogram",
    vocoderType: "Causal Flow Matching + HiFi-GAN-like vocoder",
    streamingStrategy: "Chunk-aware causal flow matching: lossless streaming vs offline. 150ms latency. KV cache + SDPA.",
    outputSR: "24 kHz",
    highlight: "Foundation for GLM-4-Voice, MiniCPM-o, Step-Audio 2. 150ms streaming. RL in v3. Zero-shot cloning.",
  },
  {
    name: "Step-Audio (v1)",
    org: "StepFun",
    date: "Feb 2025",
    params: "130B LLM + 3B speech decoder",
    category: "flow",
    paradigm: "Tokens → Flow Matching DiT → Mel → BigVGAN",
    tokenizerShort: "Dual codebook (linguistic 16.7 Hz + acoustic 25 Hz)",
    waveformMethod: "Flow Matching DiT → BigVGAN v2",
    waveformDetail: "3B speech decoder LLM processes dual-codebook tokens. Flow Matching DiT generates mel conditioned on tokens, reference audio, speaker embeddings. BigVGAN v2 converts mel → waveform. Two-stage: flow for quality, GAN for speed.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching DiT + BigVGAN v2 (GAN)",
    streamingStrategy: "Speculative generation (40% commit rate). Text context management (14:1 compression). Chunk-based overlap.",
    outputSR: "24 kHz",
    highlight: "Largest LLM backbone (130B). Hybrid flow+GAN.",
  },
  {
    name: "Sesame CSM",
    org: "Sesame AI",
    date: "Mar 2025",
    params: "1B / 3B / 8B (Llama backbone)",
    category: "codec",
    paradigm: "Codec tokens (Mimi RVQ) → Mimi decoder",
    tokenizerShort: "Mimi split-RVQ (12.5 Hz, 1.1 kbps)",
    waveformMethod: "Mimi codec decoder (conv + Transformer)",
    waveformDetail: "Two-stage transformer: Llama backbone predicts semantic codebook (C0), then 100M-param audio decoder generates remaining acoustic codebooks. Aggregated embeddings → Mimi decoder (inverse quantization → Transformer → transposed CNN) → waveform. No mel.",
    melInvolved: "No",
    vocoderType: "Mimi codec decoder (conv + Transformer)",
    streamingStrategy: "Autoregressive frame-by-frame. Context-aware: fed interleaved text+audio history for conversational coherence.",
    outputSR: "24 kHz",
    highlight: "\"Voice presence\" focus — conversational expressivity. Reuses Moshi's Mimi codec. Open-source 1B.",
  },
  {
    name: "Qwen2.5-Omni",
    org: "Alibaba Qwen",
    date: "Mar 2025",
    params: "7B",
    category: "flow",
    paradigm: "Speech tokens → Sliding-window DiT → Mel → BigVGAN",
    tokenizerShort: "qwen-tts-tokenizer (custom codec)",
    waveformMethod: "Sliding-window DiT (Flow Matching) → BigVGAN",
    waveformDetail: "Talker generates speech tokens from Thinker hidden states. DiT via flow matching → mel using sliding-window block attention (2 lookback + 1 lookahead blocks). BigVGAN converts mel chunks to waveform with fixed receptive field.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "DiT (Flow Matching) + BigVGAN",
    streamingStrategy: "Block-based sliding window. DiT and BigVGAN both chunked with limited receptive field. True streaming.",
    outputSR: "24 kHz",
    highlight: "Thinker-Talker architecture. End-to-end joint training. TMRoPE for audio-video sync.",
  },
  {
    name: "Kimi-Audio",
    org: "Moonshot AI",
    date: "Apr 2025",
    params: "7B (Qwen2.5-based)",
    category: "flow",
    paradigm: "Semantic tokens → Flow Matching → Mel → BigVGAN",
    tokenizerShort: "GLM-4-Voice tokenizer (VQ in Whisper, 12.5 Hz)",
    waveformMethod: "Flow Matching → BigVGAN",
    waveformDetail: "MoonCast-based detokenizer: (1) flow matching converts 12.5 Hz semantic tokens → 50 Hz mel, (2) BigVGAN generates waveform. Chunk-wise autoregressive streaming with look-ahead to avoid boundary artifacts.",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching + BigVGAN",
    streamingStrategy: "Chunk-wise autoregressive with look-ahead. Each chunk has small future context. No cross-fade needed.",
    outputSR: "24 kHz",
    highlight: "13M hours pretraining. Elegant chunk+look-ahead solves boundary artifacts. Hybrid input.",
  },
  {
    name: "Gemini 2.5 Native Audio",
    org: "Google DeepMind",
    date: "Jun 2025 (updated Dec 2025)",
    params: "Undisclosed (Flash variant)",
    category: "proprietary",
    paradigm: "Native audio (end-to-end, proprietary)",
    tokenizerShort: "Proprietary (native audio tokens)",
    waveformMethod: "Proprietary end-to-end native audio",
    waveformDetail: "Architecture undisclosed. Google says it 'reasons and generates speech natively in audio.' Preserves tone, emotion, non-speech vocalizations. Style control via natural language (accents, whisper). Multi-speaker in single generation. 30 HD voices, 24 languages.",
    melInvolved: "Unknown (proprietary)",
    vocoderType: "Proprietary (native audio generation)",
    streamingStrategy: "Live API real-time streaming. Context-aware proactive audio. Simultaneous tool calling during speech. Low-latency full-duplex.",
    outputSR: "24 kHz",
    highlight: "Affective dialog (responds to user's tone). Live speech-to-speech translation. Context-aware silence.",
  },
  {
    name: "Step-Audio 2 Mini",
    org: "StepFun",
    date: "~Aug 2025",
    params: "7-8B (Qwen2.5-based)",
    category: "flow",
    paradigm: "Tokens → Flow Matching + HiFi-GAN",
    tokenizerShort: "S3Tokenizer (CosyVoice, 50 Hz)",
    waveformMethod: "Flow Matching + HiFi-GAN",
    waveformDetail: "Uses S3Tokenizer (CosyVoice family) at 50 Hz from 16 kHz speech. Flow-matching module → mel → HiFi-GAN waveform. True end-to-end: no separate ASR+LLM+TTS pipeline. Much smaller than v1 (7B vs 130B).",
    melInvolved: "Yes — mel is intermediate",
    vocoderType: "Flow Matching + HiFi-GAN",
    streamingStrategy: "End-to-end streaming. Interleaved modality alignment at fixed ratio.",
    outputSR: "24 kHz",
    highlight: "Open-source (Apache 2.0). End-to-end replaces v1's 130B pipeline.",
  },
  {
    name: "Qwen3-Omni",
    org: "Alibaba Qwen",
    date: "Sep 2025",
    params: "30B total, 3B active (MoE)",
    category: "codec",
    paradigm: "Multi-codebook tokens → Causal ConvNet (Code2Wav)",
    tokenizerShort: "Qwen-TTS-Tokenizer-12Hz (16-layer multi-codebook)",
    waveformMethod: "Lightweight causal ConvNet (Code2Wav)",
    waveformDetail: "Major departure from Qwen2.5-Omni: replaces DiT+BigVGAN with lightweight causal ConvNet. Talker predicts C0 via linear head, MTP module generates all 16 residual codebooks. Rich representation → simple decoder suffices. Streams from first codec frame. Dramatically lower FLOPs.",
    melInvolved: "No — direct codec→waveform",
    vocoderType: "Causal ConvNet (Code2Wav)",
    streamingStrategy: "Frame-by-frame from first frame. MTP outputs all codebooks per step, Code2Wav synthesizes immediately. 234ms first-packet.",
    outputSR: "24 kHz",
    highlight: "Eliminated mel+DiT bottleneck. ConvNet dramatically faster. MoE architecture. 234ms first-packet.",
  },
  {
    name: "MiMo-Audio",
    org: "Xiaomi",
    date: "Dec 2025",
    params: "7B (MiMo-7B-Base)",
    category: "stft",
    paradigm: "Codec tokens → Vocos (STFT-domain)",
    tokenizerShort: "MiMo-Audio-Tokenizer (20-layer RVQ, 25 Hz)",
    waveformMethod: "Vocos-style vocoder (Transformer backbone, iSTFT)",
    waveformDetail: "Audio decoder reconstructs continuous representation from RVQ tokens, then Vocos-style vocoder predicts STFT magnitude+phase, iSTFT → waveform. Transformer backbone (not ConvNeXt) enables efficient sequence packing. Avoids mel bottleneck entirely.",
    melInvolved: "No (STFT domain, not mel)",
    vocoderType: "Vocos variant (Transformer-based, STFT-domain)",
    streamingStrategy: "Patch encoder: 4 timesteps → 6.25 Hz for LLM. Patch decoder: delayed-generation for 25 Hz RVQ. Causal decoder attention.",
    outputSR: "24 kHz",
    highlight: "100M+ hours training. Few-shot emergent capabilities. STFT-domain avoids mel bottleneck.",
  },
  {
    name: "MiniCPM-o 4.5",
    org: "OpenBMB / Tsinghua",
    date: "Feb 2026",
    params: "9B total",
    category: "flow",
    paradigm: "Speech tokens → Flow Matching (Token2Wav)",
    tokenizerShort: "CosyVoice2 tokenizer + TTS Llama (20L)",
    waveformMethod: "Streaming Flow Matching (Token2Wav)",
    waveformDetail: "LLM hidden states → TTS projector → TTS Llama (20L) → interleaved text+speech tokens. Token2Wav (from CosyVoice2/Step-Audio2) converts speech tokens via streaming flow matching → waveform. End-to-end differentiable: gradient flows through entire pipeline.",
    melInvolved: "Likely yes (flow matching → mel → vocoder)",
    vocoderType: "Flow Matching (Token2Wav from CosyVoice2)",
    streamingStrategy: "Full-duplex via TDM. Input+output streams simultaneous. 1 Hz proactive decisions. Interleaved tokens.",
    outputSR: "24 kHz",
    highlight: "Full-duplex omni-modal (vision+audio+text). Proactive interaction. 9B on-device.",
  },
  {
    name: "Raon-Speech",
    org: "KRAFTON",
    date: "Apr 2026",
    params: "9B (Qwen3-based)",
    category: "codec",
    paradigm: "Mimi codec tokens → Mimi decoder",
    tokenizerShort: "Mimi codec (32 quantizers) + ECAPA-TDNN speaker enc",
    waveformMethod: "Mimi codec decoder",
    waveformDetail: "Built on Qwen3 (36L, 4096d) + Qwen3OmniMoeAudioEncoder (24L) + Mimi codec (32 quantizers). LLM generates Mimi RVQ tokens → Mimi conv+Transformer decoder → waveform. Speaker conditioning via ECAPA-TDNN embeddings. Multi-reward DPO post-training.",
    melInvolved: "No",
    vocoderType: "Mimi codec decoder (conv + Transformer)",
    streamingStrategy: "Causal streaming + interleaved speech-text. Full-duplex SpeechChat variant. Prefill-based TTS continuation.",
    outputSR: "24 kHz",
    highlight: "#1 globally among <10B speech LLMs (42 benchmarks). Multi-reward DPO. Full-duplex variant.",
  },
  {
    name: "Raon-OpenTTS",
    org: "KRAFTON",
    date: "Apr 2026",
    params: "0.3B / 1B (F5-TTS DiT)",
    category: "flow",
    paradigm: "Text → F5-TTS DiT → Mel → HiFi-GAN",
    tokenizerShort: "Character-level (vocab 5,512)",
    waveformMethod: "F5-TTS DiT → HiFi-GAN (16 kHz)",
    waveformDetail: "F5-TTS DiT architecture. Character-level text (text_dim=512). DiT generates 80-channel log mel (16 kHz, hop=256). HiFi-GAN vocoder (LibriTTS) converts mel → waveform. 510K hours quality-filtered open-data training.",
    melInvolved: "Yes — 80-channel log mel",
    vocoderType: "F5-TTS DiT + HiFi-GAN",
    streamingStrategy: "Non-autoregressive DiT generation. Standard mel → vocoder pipeline.",
    outputSR: "16 kHz",
    highlight: "Fully open-weight AND open-data at scale. 510K hours public data. Matches closed-data SOTA.",
  },
  {
    name: "Grok Voice Think Fast",
    org: "xAI",
    date: "Apr 2026",
    params: "Undisclosed",
    category: "proprietary",
    paradigm: "Native audio (end-to-end, proprietary)",
    tokenizerShort: "Proprietary",
    waveformMethod: "Proprietary (concurrent reasoning + audio)",
    waveformDetail: "Architecture undisclosed. xAI describes 'Think Fast' as handling input, reasoning, and output near-simultaneously. Custom Voices extract speaker embeddings from ~1 min of speech. Separate STT and TTS APIs also available. #1 on Tau Voice Bench.",
    melInvolved: "Unknown (proprietary)",
    vocoderType: "Proprietary",
    streamingStrategy: "Real-time concurrent processing. Optimized for noisy environments. $0.05/min Voice Agent API.",
    outputSR: "Unknown",
    highlight: "#1 Tau Voice Bench. Custom Voices (clone ~1 min). Enterprise voice agents. Concurrent reasoning+speech.",
  },
  {
    name: "TML-Interaction-Small",
    org: "Thinking Machines Lab",
    date: "May 2026",
    params: "276B total, 12B active (MoE)",
    category: "proprietary",
    paradigm: "Multi-stream micro-turns (200ms, trained from scratch)",
    tokenizerShort: "Proprietary audio tokenizer (200ms chunks)",
    waveformMethod: "Proprietary native audio (200ms micro-turns)",
    waveformDetail: "Trained from scratch on continuous audio+video. Processes 200ms input while generating 200ms output in parallel. Audio, video, text as concurrent token streams on same clock. Time is first-class. No external VAD or dialog manager. Interactivity is part of the model itself, not bolted on.",
    melInvolved: "Unknown (proprietary)",
    vocoderType: "Proprietary (native, trained from scratch)",
    streamingStrategy: "200ms micro-turn design. Continuous interleaving. Persistent GPU-memory streaming sessions. ~400ms response latency.",
    outputSR: "Unknown",
    highlight: "Mira Murati's lab. Interaction-native (not bolted on). 200ms micro-turns. Simultaneous speech. Time-aware.",
  },
];

const columns = [
  { key: "paradigm", label: "Overall Paradigm" },
  { key: "tokenizerShort", label: "Audio Tokenizer" },
  { key: "waveformMethod", label: "Token → Waveform" },
  { key: "melInvolved", label: "Mel?" },
  { key: "vocoderType", label: "Vocoder Type" },
  { key: "streamingStrategy", label: "Streaming / Stitching" },
  { key: "outputSR", label: "SR" },
];

const categoryColors = {
  codec: { bg: "#1a2a1a", fg: "#7cc87c", label: "Codec-native" },
  flow: { bg: "#2a2218", fg: "#c8a060", label: "Flow+Vocoder" },
  stft: { bg: "#1a1a2a", fg: "#6ca0d8", label: "STFT-domain" },
  proprietary: { bg: "#2a1a2a", fg: "#b888c8", label: "Proprietary" },
};

function AudioModelComparison() {
  const [expanded, setExpanded] = useState(null);
  const [view, setView] = useState("cards");
  const [filter, setFilter] = useState("all");

  const toggleExpand = (idx) => setExpanded(expanded === idx ? null : idx);
  const filtered = filter === "all" ? models : models.filter((m) => m.category === filter);

  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', 'SF Mono', 'Fira Code', monospace",
      background: "#0a0a0f", color: "#e0ddd5", minHeight: "100vh", padding: "24px 16px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        *{box-sizing:border-box}
        .ht{font-family:'Space Grotesk',sans-serif;font-size:21px;font-weight:700;color:#f0ece4;margin:0 0 4px;letter-spacing:-.5px}
        .hs{font-size:11.5px;color:#6b6860;margin:0 0 14px}
        .ctrls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
        .tb{display:flex;gap:2px;background:#151520;border-radius:6px;padding:3px}
        .tbtn{padding:5px 12px;border:none;background:transparent;color:#6b6860;font-family:'IBM Plex Mono',monospace;font-size:10.5px;cursor:pointer;border-radius:4px;transition:all .2s}
        .tbtn.a{background:#1e1e2e;color:#c8e060}
        .cb{font-size:9.5px;color:#4a4840;margin-left:3px}
        .tw{overflow-x:auto;border:1px solid #1e1e2e;border-radius:8px}
        table{width:100%;border-collapse:collapse;min-width:1200px;font-size:10.5px}
        thead th{background:#12121c;color:#8b8878;font-weight:500;text-align:left;padding:7px 8px;border-bottom:1px solid #1e1e2e;position:sticky;top:0;font-size:9px;text-transform:uppercase;letter-spacing:.6px}
        thead th:first-child{min-width:130px;position:sticky;left:0;z-index:2}
        tbody tr{border-bottom:1px solid #151520;transition:background .15s}
        tbody tr:hover{background:#13131f}
        td{padding:7px 8px;vertical-align:top;line-height:1.4}
        td:first-child{position:sticky;left:0;background:#0a0a0f;z-index:1;border-right:1px solid #1e1e2e}
        tbody tr:hover td:first-child{background:#13131f}
        .mn{font-family:'Space Grotesk',sans-serif;font-weight:600;color:#f0ece4;font-size:12px}
        .mo{color:#6b6860;font-size:9.5px}
        .md{color:#4a4840;font-size:9px}
        .tg{display:inline-block;padding:2px 6px;border-radius:3px;font-size:9px;font-weight:500}
        .tg-n{background:#1a2a1a;color:#7cc87c}
        .tg-y{background:#2a2218;color:#c8a060}
        .tg-m{background:#1e1e2e;color:#8888aa}
        .cat{display:inline-block;padding:1px 5px;border-radius:3px;font-size:8.5px;font-weight:500;margin-top:2px}
        .cg{display:flex;flex-direction:column;gap:8px}
        .cd{background:#111118;border:1px solid #1e1e2e;border-radius:7px;overflow:hidden;cursor:pointer;transition:border-color .2s}
        .cd:hover{border-color:#2a2a3e}
        .ch{padding:11px 13px;display:flex;justify-content:space-between;align-items:center}
        .cl{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
        .cp{font-size:10px;color:#6b6860;padding:2px 6px;background:#0a0a0f;border-radius:3px;max-width:320px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .ca{color:#4a4840;font-size:15px;transition:transform .2s;flex-shrink:0}
        .ca.o{transform:rotate(90deg)}
        .cb2{padding:0 13px 13px;display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:9px}
        .cf{padding:8px 10px;background:#0d0d15;border-radius:5px}
        .cfl{font-size:8.5px;text-transform:uppercase;letter-spacing:.6px;color:#5b5850;margin-bottom:3px}
        .cfv{font-size:10.5px;line-height:1.5;color:#c8c4b8}
        .chi{grid-column:1/-1;padding:8px 10px;background:#0f1018;border-left:2px solid #c8e060;border-radius:0 5px 5px 0}
        .lg{margin-top:14px;padding:11px 13px;background:#111118;border:1px solid #1e1e2e;border-radius:7px;font-size:10px;line-height:1.7;color:#8b8878}
        .lg strong{color:#c8c4b8}
      `}</style>

      <h1 className="ht">Audio Model Waveform Generation — {models.length} Models Compared</h1>
      <p className="hs">How each model converts internal representations → raw audio waveform (2024–2026)</p>

      <div className="ctrls">
        <div className="tb">
          <button className={`tbtn ${view==="cards"?"a":""}`} onClick={()=>setView("cards")}>Cards</button>
          <button className={`tbtn ${view==="table"?"a":""}`} onClick={()=>setView("table")}>Table</button>
        </div>
        <div className="tb">
          <button className={`tbtn ${filter==="all"?"a":""}`} onClick={()=>setFilter("all")}>All<span className="cb">{models.length}</span></button>
          {Object.entries(categoryColors).map(([k,v])=>{
            const c=models.filter(m=>m.category===k).length;
            return <button key={k} className={`tbtn ${filter===k?"a":""}`} onClick={()=>setFilter(k)} style={filter===k?{color:v.fg}:{}}>{v.label}<span className="cb">{c}</span></button>
          })}
        </div>
      </div>

      {view==="table"?(
        <div className="tw"><table><thead><tr><th>Model</th>{columns.map(c=><th key={c.key}>{c.label}</th>)}</tr></thead>
        <tbody>{filtered.map((m,i)=>{const cc=categoryColors[m.category];return(
          <tr key={i}><td><div className="mn">{m.name}</div><div className="mo">{m.org}</div><div className="md">{m.date} · {m.params}</div><span className="cat" style={{background:cc.bg,color:cc.fg}}>{cc.label}</span></td>
          {columns.map(c=><td key={c.key}>{c.key==="melInvolved"?<span className={`tg ${m[c.key].startsWith("No")?"tg-n":m[c.key].startsWith("Unknown")||m[c.key].startsWith("Likely")?"tg-m":"tg-y"}`}>{m[c.key]}</span>:m[c.key]}</td>)}</tr>
        )})}</tbody></table></div>
      ):(
        <div className="cg">{filtered.map((m,i)=>{const cc=categoryColors[m.category];const gi=models.indexOf(m);return(
          <div className="cd" key={gi} onClick={()=>toggleExpand(gi)}>
            <div className="ch"><div className="cl"><div><span className="mn">{m.name}</span><span className="mo" style={{marginLeft:7}}>{m.org} · {m.date}</span></div><span className="cat" style={{background:cc.bg,color:cc.fg}}>{cc.label}</span><div className="cp">{m.paradigm}</div></div><span className={`ca ${expanded===gi?"o":""}`}>›</span></div>
            {expanded===gi&&<div className="cb2">
              <div className="cf"><div className="cfl">Audio Tokenizer</div><div className="cfv">{m.tokenizerShort}<br/><span style={{color:"#6b6860",fontSize:9.5}}>{m.params}</span></div></div>
              <div className="cf"><div className="cfl">Waveform Generation</div><div className="cfv">{m.waveformDetail}</div></div>
              <div className="cf"><div className="cfl">Vocoder Type</div><div className="cfv">{m.vocoderType}</div></div>
              <div className="cf"><div className="cfl">Mel Intermediate?</div><div className="cfv"><span className={`tg ${m.melInvolved.startsWith("No")?"tg-n":m.melInvolved.startsWith("Unknown")||m.melInvolved.startsWith("Likely")?"tg-m":"tg-y"}`}>{m.melInvolved}</span></div></div>
              <div className="cf"><div className="cfl">Streaming / Stitching</div><div className="cfv">{m.streamingStrategy}</div></div>
              <div className="cf"><div className="cfl">Output</div><div className="cfv">{m.outputSR}</div></div>
              <div className="chi"><div className="cfl">Key Insight</div><div className="cfv" style={{color:"#c8e060"}}>{m.highlight}</div></div>
            </div>}
          </div>
        )})}</div>
      )}

      <div className="lg">
        <strong>Architecture families ({models.length} models):</strong><br/>
        <span style={{color:"#7cc87c"}}>● Codec-native</span> — Rich codec tokens (RVQ / multi-codebook); lightweight decoder reconstructs waveform directly. No mel. (Moshi, Sesame CSM, Qwen3-Omni, Raon-Speech)<br/>
        <span style={{color:"#c8a060"}}>● Flow Matching + Vocoder</span> — Tokens → flow matching / DiT → mel → GAN vocoder. Most common open-source pattern. (GLM-4-Voice, CosyVoice, Step-Audio, Kimi-Audio, Qwen2.5-Omni, MiniCPM-o, Raon-OpenTTS)<br/>
        <span style={{color:"#6ca0d8"}}>● STFT-domain</span> — Vocos-style: predict STFT magnitude+phase, iSTFT. Skips mel. (MiMo-Audio)<br/>
        <span style={{color:"#b888c8"}}>● Proprietary</span> — Native end-to-end audio, architecture undisclosed. (GPT-4o Realtime, Gemini Native Audio, Grok Voice, TML Interaction)<br/><br/>
        <strong>2024→2026 trend:</strong> The field is moving from mel-based Flow+Vocoder pipelines toward codec-native and end-to-end approaches. Qwen's evolution is emblematic: 2.5-Omni (DiT+BigVGAN) → 3-Omni (lightweight ConvNet), enabled by richer multi-codebook representations. The proprietary models (OpenAI, Google, xAI, TML) have gone further — training audio generation end-to-end from scratch, making the vocoder concept disappear entirely into the model weights. TML's "interaction model" represents the most radical departure: time itself is tokenized into 200ms micro-turns, with audio generation as an intrinsic capability rather than a downstream module.
      </div>
    </div>
  );
}
