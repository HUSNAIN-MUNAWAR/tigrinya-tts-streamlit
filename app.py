# app.py — Tigrinya TTS (HF-hosted VITS) + Stable Streamlit UI (no sidebar)
# - No sidebar / no settings in UI
# - Robust HF loading (works whether your model is in repo root or in a subfolder)
# - Safe G2P call (never blocks synthesis; falls back to raw text)
# - Hidden keep-alive ping (5s) while page is open
# - Professional, tight UI

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from math import gcd

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


# =============================================================================
# Hidden config (NOT shown in UI)
# =============================================================================
REPO_ID: str = os.getenv("HF_REPO_ID", "husnainbinmunawar/tigrinya-tts-model")
REVISION: str = os.getenv("HF_REVISION", "main")

# If your HF repo really has a folder "round_01/" that contains model files, set SUBDIR="round_01".
# If you uploaded from inside round_01 using `hf upload ... .`, the files are likely at repo root -> SUBDIR=""
SUBDIR: str = os.getenv("HF_SUBDIR", "round_01")

# Allow automatic fallback to repo root if SUBDIR doesn't exist on HF
AUTO_FALLBACK_TO_ROOT: bool = os.getenv("HF_SUBDIR_FALLBACK", "1") == "1"

# Keep-alive ping: runs in browser while page is open
KEEP_ALIVE: bool = os.getenv("KEEP_ALIVE", "1") == "1"
KEEP_ALIVE_MS: int = int(os.getenv("KEEP_ALIVE_MS", "5000"))

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Page config + style (hide sidebar/nav/menu)
# =============================================================================
st.set_page_config(page_title="Tigrinya TTS", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}

/* Remove sidebar & nav completely */
[data-testid="stSidebar"] {display:none !important;}
[data-testid="stSidebarNav"] {display:none !important;}
[data-testid="collapsedControl"] {display:none !important;}

/* Layout */
.block-container {padding-top:0.65rem; padding-bottom:1.0rem; padding-left:1.0rem; padding-right:1.0rem;}
h1,h2,h3 {margin:0.15rem 0 0.35rem 0;}
.small {color: rgba(49, 51, 63, 0.70); font-size:0.92rem; line-height:1.25;}
.card {border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:14px; background:rgba(255,255,255,0.65);}
.hr {height:1px; background:rgba(49,51,63,0.10); margin:12px 0;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Tigrinya Text-to-Speech")
st.markdown(
    '<div class="small">Text → (G2P optional) → Fine-tuned VITS (Hugging Face) → Play + Download</div>',
    unsafe_allow_html=True,
)

if not HAS_VITS:
    st.error("VitsModel is not available in this transformers build. Please use a transformers version that includes VitsModel.")
    st.stop()


# =============================================================================
# Hidden keep-alive ping (every 5 seconds) while page is open
# Note: This does NOT fully prevent Streamlit Cloud sleep. For true keep-alive,
# use an external ping service (e.g., UptimeRobot) every 5 minutes.
# =============================================================================
if KEEP_ALIVE:
    st.components.v1.html(
        f"""
<script>
  const interval = {KEEP_ALIVE_MS};
  setInterval(() => {{
    fetch(window.location.href, {{ cache: "no-store" }}).catch(() => {{}});
  }}, interval);
</script>
""",
        height=0,
    )


# =============================================================================
# Audio utilities
# =============================================================================
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
        if rms_db(a[i : i + frame]) > thr_db:
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
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(np.asarray(audio, dtype=np.float32), up=up, down=down).astype(np.float32)


# safer split regex (use unicode escapes)
_SPLIT_RE = re.compile(r"(?<=[\u1362.!?\u1367\u1368])\s+|[\r\n]+")
def split_text(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    return parts if parts else [text]


# =============================================================================
# Hugging Face model reference + download/cache
# =============================================================================
@dataclass(frozen=True)
class HFModelRef:
    repo_id: str
    revision: str = "main"
    subdir: str = ""


def _get_hf_token() -> Optional[str]:
    # Streamlit secrets (recommended) or env var
    try:
        tok = st.secrets.get("HF_TOKEN", None)
    except Exception:
        tok = None
    return tok or os.getenv("HF_TOKEN")


@st.cache_resource(show_spinner=False)
def download_model_from_hf(ref: HFModelRef) -> str:
    """
    Downloads model files from Hugging Face and returns local directory path.
    Uses ignore patterns to avoid pulling huge training checkpoints if present.
    """
    token = _get_hf_token()

    allow_patterns = [f"{ref.subdir}/**"] if ref.subdir.strip() else None
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


@st.cache_resource(show_spinner=False)
def load_processor(model_dir: str):
    # IMPORTANT: model_dir must be an actual local directory
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Invalid model directory: {model_dir}")
    return AutoProcessor.from_pretrained(model_dir)


@st.cache_resource(show_spinner=False)
def load_vits(model_dir: str, device: str):
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Invalid model directory: {model_dir}")
    m = VitsModel.from_pretrained(model_dir)
    m.to(device)
    m.eval()
    return m


def resolve_model_dir() -> str:
    """
    Resolve the correct model directory robustly:
    - Try SUBDIR (e.g., 'round_01')
    - If missing and AUTO_FALLBACK_TO_ROOT enabled, retry root
    """
    # First attempt: configured subdir
    ref1 = HFModelRef(repo_id=REPO_ID, revision=REVISION, subdir=SUBDIR.strip())
    try:
        return download_model_from_hf(ref1)
    except Exception as e1:
        if not AUTO_FALLBACK_TO_ROOT or not SUBDIR.strip():
            raise e1

        # Fallback: root
        ref2 = HFModelRef(repo_id=REPO_ID, revision=REVISION, subdir="")
        return download_model_from_hf(ref2)


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


# =============================================================================
# Main UI
# =============================================================================
left, right = st.columns([1.55, 1.0], gap="large")

with left:
    st.subheader("Input")
    text = st.text_area(
        "Tigrinya Text",
        height=210,
        placeholder="ኣብዚ ጽሑፍ ኣእትዉ። ድምጺ ንፍጠር።",
        label_visibility="collapsed",
    )

    st.subheader("G2P")
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

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    split_long = st.toggle("Auto-split long text", value=True)
    gap_ms = st.slider("Gap between chunks (ms)", 0, 250, 70, 10, disabled=not split_long)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    do_trim = st.toggle("Trim start/end silence", value=True)
    trim_thr = st.slider("Trim threshold (dB)", -60, -20, -42, 1, disabled=not do_trim)
    do_norm = st.toggle("Normalize peak", value=True)

    st.markdown("</div>", unsafe_allow_html=True)

gen_btn = st.button("Generate Voice", use_container_width=True)


# =============================================================================
# Generate
# =============================================================================
if gen_btn:
    if not REPO_ID.strip() or REPO_ID.count("/") > 1:
        st.error("Invalid Hugging Face repo_id. Must be 'repo' or 'namespace/repo'.")
        st.stop()
    if not text.strip():
        st.error("Please enter Tigrinya text.")
        st.stop()

    # 1) G2P: NEVER block synthesis
    rules = parse_rules(rules_text) if (rules_text or "").strip() else []
    cfg = G2PConfig(
        mode=g2p_mode if g2p_enabled else "normalize",
        rules=rules,
        uroman_path=os.environ.get("UROMAN_PATH", ""),
        uroman_lang="tir",
        keep_original_on_fail=True,
        digit_strategy="separate_words",
    )

    try:
        processed = g2p(text, cfg) if g2p_enabled else text
        processed = processed.strip() if processed else text
    except Exception:
        processed = text

    if show_processed and g2p_enabled:
        st.markdown("**Processed Text**")
        st.code(processed, language="text")

    # 2) Split
    parts = split_text(processed) if split_long else [processed]
    if not parts:
        st.error("No text to synthesize after preprocessing.")
        st.stop()

    # 3) Resolve model dir (robust subdir/root)
    with st.spinner("Loading model (first run may take time)..."):
        try:
            model_dir = resolve_model_dir()
        except Exception as e:
            st.error(
                "Failed to download/resolve model files from Hugging Face.\n\n"
                f"Repo: {REPO_ID} | Revision: {REVISION} | Subdir: '{SUBDIR}'\n\n"
                f"Error: {e}"
            )
            st.stop()

    # 4) Synthesize
    with st.spinner("Synthesizing..."):
        all_audio: List[np.ndarray] = []
        sr_ref: Optional[int] = None

        for i, part in enumerate(parts):
            try:
                wav, sr = synth_one(
                    model_dir=model_dir,
                    text=part,
                    device=DEVICE,
                    noise_scale=noise_scale,
                    noise_scale_duration=noise_scale_duration,
                    length_scale=length_scale,
                )
            except Exception as e:
                st.error(f"Synthesis failed on chunk {i+1}/{len(parts)}: {e}")
                st.stop()

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
    out_audio = (
        np.concatenate([np.asarray(a, dtype=np.float32) for a in all_audio])
        if all_audio
        else np.zeros((0,), np.float32)
    )

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

    st.caption("Tip: For reliable 24/7 uptime on Streamlit Cloud, use an external ping (e.g., UptimeRobot) every 5 minutes.")
