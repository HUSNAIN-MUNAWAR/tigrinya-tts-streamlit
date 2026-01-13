# app.py
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st

from tigrinya_g2p import g2p, G2PConfig, parse_rules
from inference import synthesize_text, to_wav_bytes


# ----------------------------
# Hidden config (no settings shown on UI)
# ----------------------------
REPO_ID = os.getenv("HF_REPO_ID", "husnainbinmunawar/tigrinya-tts-model")
REVISION = os.getenv("HF_REVISION", "main")
SUBDIR = os.getenv("HF_SUBDIR", "")  # IMPORTANT: set "" if files are in repo root

KEEP_ALIVE_MS = int(os.getenv("KEEP_ALIVE_MS", "5000"))


# ----------------------------
# Page + style (no sidebar/nav + no scroll as much as possible)
# ----------------------------
st.set_page_config(page_title="Tigrinya TTS", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}
[data-testid="stSidebar"] {display:none !important;}
[data-testid="stSidebarNav"] {display:none !important;}
[data-testid="collapsedControl"] {display:none !important;}

/* Compact layout */
.block-container {padding-top:0.55rem; padding-bottom:0.8rem; padding-left:0.9rem; padding-right:0.9rem; max-width: 1200px;}
h1 {margin:0.05rem 0 0.3rem 0;}
.small {color: rgba(49, 51, 63, 0.70); font-size:0.90rem; line-height:1.25;}
.card {border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:12px; background:rgba(255,255,255,0.65);}
.hr {height:1px; background:rgba(49,51,63,0.10); margin:10px 0;}
/* Make Generate button colored */
div.stButton > button {
  background: linear-gradient(90deg, #2D6CDF, #1DB9A6) !important;
  color: white !important;
  border: 0 !important;
  padding: 0.65rem 1.0rem !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}
div.stButton > button:hover { filter: brightness(0.97); }
</style>
""",
    unsafe_allow_html=True,
)

# Internal keep-alive while tab is open (does not keep it awake with zero visitors)
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

st.title("Tigrinya Text-to-Speech")
st.markdown('<div class="small">Fine-tuned VITS (HF) → Generate audio → Play + Download</div>', unsafe_allow_html=True)


# ----------------------------
# Helpers
# ----------------------------
_SPLIT_RE = re.compile(r"(?<=[\u1362.!?\u1367\u1368])\s+|[\r\n]+")

def split_text(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(t) if p.strip()]
    return parts if parts else [t]


# ----------------------------
# UI (compact)
# ----------------------------
left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.markdown("#### Text")
    text = st.text_area(
        "Text",
        height=170,
        placeholder="ኣብዚ ጽሑፍ ኣእትዉ።",
        label_visibility="collapsed",
    )

    st.markdown("#### G2P")
    g2p_enabled = st.toggle("Enable G2P", value=True)

    c1, c2 = st.columns([1.0, 1.0], gap="medium")
    with c1:
        g2p_mode = st.selectbox("Mode", ["normalize", "rules", "uroman", "auto"], index=0, disabled=not g2p_enabled)
    with c2:
        show_processed = st.toggle("Show processed", value=False, disabled=not g2p_enabled)

    rules_text = st.text_area(
        "Rules (optional): FROM => TO (one per line)",
        height=90,
        placeholder="Example:\nword1 => word1_fixed",
        disabled=not (g2p_enabled and g2p_mode in ["rules", "auto"]),
    )

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Voice Settings")

    length_scale = st.slider("Speed", 0.78, 1.15, 0.90, 0.01)
    noise_scale = st.slider("Noise", 0.12, 0.55, 0.24, 0.01)
    noise_scale_duration = st.slider("Noise duration", 0.12, 0.55, 0.30, 0.01)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    split_long = st.toggle("Auto-split long text", value=True)
    gap_ms = st.slider("Gap (ms)", 0, 250, 70, 10, disabled=not split_long)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")  # spacing
gen_btn = st.button("Generate Voice", use_container_width=True)


# ----------------------------
# Generation
# ----------------------------
if gen_btn:
    if not text.strip():
        st.error("Please enter text.")
        st.stop()

    # G2P (never block synthesis)
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
        st.code(processed, language="text")

    parts = split_text(processed) if split_long else [processed]
    if not parts:
        st.error("No text to synthesize.")
        st.stop()

    # Progress UI
    progress = st.progress(0)
    status = st.empty()

    def cb(pct: int, msg: str):
        progress.progress(int(pct))
        status.markdown(f"<div class='small'>{msg}</div>", unsafe_allow_html=True)

    # Synthesize chunk-by-chunk (shows progress)
    audios: List[np.ndarray] = []
    sr_ref: Optional[int] = None

    for idx, part in enumerate(parts, start=1):
        status.markdown(f"<div class='small'>Chunk {idx}/{len(parts)}…</div>", unsafe_allow_html=True)
        opts = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "subdir": SUBDIR,
            "noise_scale": noise_scale,
            "noise_scale_duration": noise_scale_duration,
            "length_scale": length_scale,
        }
        try:
            wav, sr, _dbg = synthesize_text(part, opts, cb=cb)
        except Exception as e:
            st.error(f"Synthesis failed: {e}")
            st.stop()

        # Keep it simple: assume same sr for all chunks; if mismatch, you can add resample later.
        if sr_ref is None:
            sr_ref = sr
        audios.append(wav)

        # add small gap
        if split_long and gap_ms > 0 and idx != len(parts):
            audios.append(np.zeros((int(sr_ref * gap_ms / 1000.0),), dtype=np.float32))

    progress.progress(100)
    status.markdown("<div class='small'>Ready.</div>", unsafe_allow_html=True)

    out_sr = int(sr_ref or 16000)
    out_audio = np.concatenate([np.asarray(a, dtype=np.float32) for a in audios]) if audios else np.zeros((0,), np.float32)

    wav_bytes = to_wav_bytes(out_audio, out_sr)
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("Download WAV", wav_bytes, file_name="tigrinya_tts.wav", mime="audio/wav", use_container_width=True)
