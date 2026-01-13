# app.py — Tigrinya TTS (HF-hosted model) + Stable Streamlit UI (no sidebar)
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from huggingface_hub import snapshot_download
from scipy.signal import resample_poly
from transformers import AutoProcessor

from tigrinya_g2p import g2p, G2PConfig, parse_rules

try:
    from transformers import VitsModel
    HAS_VITS = True
except Exception:
    HAS_VITS = False


# ----------------------------
# Page config + style (hide sidebar/nav/menu)
# ----------------------------
st.set_page_config(page_title="Tigrinya TTS", layout="wide")

st.markdown(
    """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}

/* Remove sidebar & left nav */
[data-testid="stSidebar"] {display:none !important;}
[data-testid="stSidebarNav"] {display:none !important;}

/* Tighten layout */
.block-container {padding-top:0.6rem; padding-bottom:1.0rem; padding-left:1.0rem; padding-right:1.0rem;}
h1,h2,h3 {margin:0.2rem 0 0.35rem 0;}
.small {color: rgba(49, 51, 63, 0.70); font-size:0.92rem;}
.card {border:1px solid rgba(49,51,63,0.15); border-radius:14px; padding:14px; background:rgba(255,255,255,0.65);}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Tigrinya Text-to-Speech")
st.markdown('<div class="small">Text → (G2P optional) → Fine-tuned TTS (from Hugging Face) → Play + Download</div>', unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
if not HAS_VITS:
    st.warning("VitsModel is not available in this transformers build. Please ensure transformers supports VitsModel.")


# ----------------------------
# Hidden keep-alive ping (every 5 seconds) while page is open
# Note: This does NOT fully prevent Streamlit Cloud sleep; use UptimeRobot for true keep-alive.
# ----------------------------
st.components.v1.html(
    """
<script>
  const interval = 5000; // 5 seconds
  setInterval(() => {
    fetch(window.location.href, { cache: "no-store" }).catch(() => {});
  }, interval);
</script>
""",
    height=0,
)


# ----------------------------
# Audio utilities
# ----------------------------
def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(a)) + 1e-9)
    if peak <= 0:
        return a
    gain = target_peak / peak
    return np.clip(a * gain, -1.0, 1.0)


def trim_edges_silence(audio: np.ndarray, sr: int, thr_db: float = -42.0, pad_ms: int = 60) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.size < int(0.2 * sr):
        return a

    frame = max(256, int(0.02 * sr))
    hop = frame // 2

    def rms_db(x):
        r = np.sqrt(np.mean(x * x) + 1e-12)
        return 20.0 * np.log10(r + 1e-12)

    active = []
    for i in range(0, len(a) - frame, hop):
        if rms_db(a[i:i + frame]) > thr_db:
            active.append(i)

    if not active:
        return a

    start = max(0, min(active) - int((pad_ms / 1000.0) * sr))
    end = min(len(a), max(active) + frame + int((pad_ms / 1000.0) * sr))
    return a[start:end]


def add_silence(sr: int, ms: int) -> np.ndarray:
    return np.zeros((int(sr * ms / 1000.0),), dtype=np.float32)


def safe_resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(audio, dtype=np.float32)
    from math import gcd
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(np.asarray(audio, dtype=np.float32), up=up, down=down).astype(np.float32)


_SPLIT_RE = re.compile(r"(?<=[።.!?፧፨])\s+|[\r\n]+")
def split_text(text: str) -> List[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(text.strip()) if p.strip()]
    return parts if parts else [text.strip()]


# ----------------------------
# HuggingFace model reference
# ----------------------------
@dataclass(frozen=True)
class HFModelRef:
    repo_id: str
    revision: str = "main"
    subdir: str = ""  # e.g. "round_01"


@st.cache_resource
def download_model_from_hf(ref: HFModelRef) -> str:
    token = st.secrets.get("HF_TOKEN", None) if hasattr(st, "secrets") else None

    allow_patterns = [f"{ref.subdir}/**"] if ref.subdir.strip() else None

    # IMPORTANT: skip huge training artifacts if your HF repo contains checkpoints
    ignore_patterns = [
        "**/checkpoint-*/**",
        "**/optimizer*.bin",
        "**/scheduler*.bin",
        "**/random_states*.pkl",
        "**/trainer_state.json",
        "**/rng_state.pth",
        "**/training_args.bin",
    ]

    local_root = snapshot_download(
        repo_id=ref.repo_id,
        revision=ref.revision,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    return os.path.join(local_root, ref.subdir) if ref.subdir.strip() else local_root


@st.cache_resource
def load_processor(model_dir: str):
    return AutoProcessor.from_pretrained(model_dir)


@st.cache_resource
def load_vits(model_dir: str, device: str):
    if not HAS_VITS:
        raise RuntimeError("VitsModel not available in this transformers version.")
    m = VitsModel.from_pretrained(model_dir)
    m.to(device)
    m.eval()
    return m


def synth_one(
    model_dir: str,
    text: str,
    device: str,
    noise_scale: float,
    noise_scale_duration: float,
    length_scale: float,
) -> Tuple[np.ndarray, int]:
    proc = load_processor(model_dir)
    model = load_vits(model_dir, device)

    inputs = proc(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        if hasattr(model, "generate_speech"):
            wav = model.generate_speech(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noise_scale=noise_scale,
                noise_scale_duration=noise_scale_duration,
                length_scale=length_scale,
            ).detach().cpu().numpy()
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            wav = out.waveform.detach().cpu().numpy()

    wav = np.squeeze(wav).astype(np.float32)
    sr = int(getattr(model.config, "sampling_rate", 16000))
    return wav, sr


# ----------------------------
# Config (top expander, not sidebar)
# ----------------------------
with st.expander("Model & Runtime Settings", expanded=False):
    c1, c2, c3 = st.columns([1.5, 0.8, 0.8], gap="large")
    with c1:
        repo_id = st.text_input("Hugging Face repo_id", value="husnainbinmunawar/tigrinya-tts-model")
        subdir = st.text_input("Model subfolder (optional)", value="round_01")
    with c2:
        revision = st.text_input("Revision", value="main")
    with c3:
        st.write(f"Device: **{device}**")


# ----------------------------
# Main UI
# ----------------------------
left, right = st.columns([1.55, 1.0], gap="large")

with left:
    text = st.text_area(
        "Tigrinya Text",
        height=190,
        placeholder="ኣብዚ ጽሑፍ ኣእትዉ። ድምጺ ንፍጠር።",
    )

    st.markdown("#### G2P")
    g2p_enabled = st.toggle("Enable G2P", value=True)

    g2p_mode = st.selectbox(
        "Mode",
        ["normalize", "rules", "uroman", "auto"],
        index=0,
        disabled=not g2p_enabled,
    )
    rules_text = st.text_area(
        "Rules (optional) — one per line: FROM => TO",
        height=110,
        placeholder="Example:\nword1 => word1_fixed\n# comments allowed",
        disabled=not (g2p_enabled and g2p_mode in ["rules", "auto"]),
    )
    show_processed = st.toggle("Show processed text", value=True, disabled=not g2p_enabled)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Voice Controls")

    length_scale = st.slider("Speed (length_scale)", 0.78, 1.15, 0.90, 0.01)
    noise_scale = st.slider("noise_scale", 0.12, 0.55, 0.24, 0.01)
    noise_scale_duration = st.slider("noise_scale_duration", 0.12, 0.55, 0.30, 0.01)

    st.markdown("---")
    split_long = st.toggle("Auto-split long text", value=True)
    gap_ms = st.slider("Gap between chunks (ms)", 0, 250, 70, 10, disabled=not split_long)

    st.markdown("---")
    do_trim = st.toggle("Trim start/end silence", value=True)
    trim_thr = st.slider("Trim threshold (dB)", -60, -20, -42, 1, disabled=not do_trim)
    do_norm = st.toggle("Normalize peak", value=True)

    st.markdown("</div>", unsafe_allow_html=True)

gen_btn = st.button("Generate Voice", use_container_width=True)

# ----------------------------
# Generate
# ----------------------------
if gen_btn:
    if not repo_id.strip():
        st.error("Hugging Face repo_id is required.")
        st.stop()
    if not text.strip():
        st.error("Please enter Tigrinya text.")
        st.stop()
    if not HAS_VITS:
        st.error("VitsModel not available. Adjust transformers version in requirements.txt.")
        st.stop()

    ref = HFModelRef(repo_id=repo_id.strip(), revision=revision.strip() or "main", subdir=subdir.strip())

    rules = parse_rules(rules_text) if rules_text.strip() else []
    cfg = G2PConfig(
        mode=g2p_mode if g2p_enabled else "normalize",
        rules=rules,
        uroman_path=os.environ.get("UROMAN_PATH", ""),
        uroman_lang="tir",
        keep_original_on_fail=True,
    )
    processed = g2p(text, cfg) if g2p_enabled else text

    if show_processed and g2p_enabled:
        st.markdown("**Processed Text**")
        st.code(processed, language="text")

    parts = split_text(processed) if split_long else [processed]

    with st.spinner("Downloading model (first run may take time)..."):
        model_dir = download_model_from_hf(ref)

    with st.spinner("Synthesizing..."):
        all_audio: List[np.ndarray] = []
        sr_ref: Optional[int] = None

        for i, part in enumerate(parts):
            wav, sr = synth_one(
                model_dir=model_dir,
                text=part,
                device=device,
                noise_scale=noise_scale,
                noise_scale_duration=noise_scale_duration,
                length_scale=length_scale,
            )

            if do_trim:
                wav = trim_edges_silence(wav, sr, thr_db=float(trim_thr), pad_ms=60)
            if do_norm:
                wav = normalize_peak(wav, target_peak=0.95)

            if sr_ref is None:
                sr_ref = sr
            elif sr != sr_ref:
                wav = safe_resample(wav, sr, sr_ref)

            all_audio.append(wav)
            if split_long and gap_ms > 0 and i != len(parts) - 1:
                all_audio.append(add_silence(sr_ref, gap_ms))

    out_sr = int(sr_ref or 16000)
    out_audio = np.concatenate([np.asarray(a, dtype=np.float32) for a in all_audio]) if all_audio else np.zeros((0,), np.float32)

    st.success(f"Done — {len(out_audio)/out_sr:.2f}s @ {out_sr} Hz")
    wav_bytes = to_wav_bytes(out_audio, out_sr)

    st.audio(wav_bytes, format="audio/wav")
    st.download_button(
        "Download WAV",
        data=wav_bytes,
        file_name="tigrinya_tts.wav",
        mime="audio/wav",
        use_container_width=True,
    )

    st.caption("For true keep-alive on Streamlit Cloud, use an external ping (e.g., UptimeRobot) every 5 minutes.")
