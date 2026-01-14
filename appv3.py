# app.py — Queued Tigrinya TTS (NO DB). Stable queue across Streamlit reruns.
from __future__ import annotations

import os
import re
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from tigrinya_g2p import g2p, G2PConfig, parse_rules
from inference import synthesize_text, to_wav_bytes


# ----------------------------
# Hidden config
# ----------------------------
REPO_ID = os.getenv("HF_REPO_ID", "husnainbinmunawar/tigrinya-tts-model")
REVISION = os.getenv("HF_REVISION", "main")
SUBDIR = os.getenv("HF_SUBDIR", "")

KEEP_ALIVE_MS = int(os.getenv("KEEP_ALIVE_MS", "5000"))

MAX_CHARS = int(os.getenv("MAX_CHARS", "1200"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "10"))
QUEUE_MAX_JOBS = int(os.getenv("QUEUE_MAX_JOBS", "200"))

HEARTBEAT_SEC = int(os.getenv("HEARTBEAT_SEC", "15"))
POLL_MS = int(os.getenv("POLL_MS", "900"))

JOB_TTL_SEC = int(os.getenv("JOB_TTL_SEC", "1200"))
CLEANUP_EVERY_SEC = int(os.getenv("CLEANUP_EVERY_SEC", "20"))


# ----------------------------
# UI basics
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

.block-container {padding-top:0.55rem; padding-bottom:0.8rem; padding-left:0.9rem; padding-right:0.9rem; max-width: 1200px;}
h1 {margin:0.05rem 0 0.3rem 0;}
.small {color: rgba(49, 51, 63, 0.70); font-size:0.90rem; line-height:1.25;}
.card {border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:12px; background:rgba(255,255,255,0.65);}
.hr {height:1px; background:rgba(49,51,63,0.10); margin:10px 0;}
.badge {padding:6px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.15); background:rgba(255,255,255,0.75); font-size:0.85rem;}
.toprow {display:flex; align-items:flex-start; justify-content:space-between; gap:12px;}

div.stButton > button, div.stFormSubmitButton > button {
  background: linear-gradient(90deg, #2D6CDF, #1DB9A6) !important;
  color: white !important;
  border: 0 !important;
  padding: 0.65rem 1.0rem !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover { filter: brightness(0.97); }
</style>
""",
    unsafe_allow_html=True,
)

# Keep-alive while tab is open (does NOT guarantee 24/7 uptime)
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


def _safe_rerun_after(ms: int) -> None:
    """Rerun without relying on deprecated APIs. Works on old+new Streamlit."""
    ms = int(max(300, ms))
    time.sleep(ms / 1000.0)
    if hasattr(st, "rerun"):
        st.rerun()
        return
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
        return
    # last resort: hard reload
    st.components.v1.html(f"<script>window.location.reload()</script>", height=0)
    st.stop()


def _get_session_id() -> str:
    if "sid" not in st.session_state:
        st.session_state["sid"] = uuid.uuid4().hex
    return st.session_state["sid"]


# ----------------------------
# Queue state stored in cache_resource (CRITICAL)
# ----------------------------
@dataclass
class Job:
    job_id: str
    created_ts: float
    session_id: str
    text_chunks: List[str]
    opts: Dict

    pct: int = 0
    msg: str = "Queued"
    done: bool = False
    error: Optional[str] = None
    wav_bytes: Optional[bytes] = None
    sr: Optional[int] = None
    updated_ts: float = 0.0


@dataclass
class QueueState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(init=False)

    queue: List[str] = field(default_factory=list)
    jobs: Dict[str, Job] = field(default_factory=dict)
    active_job_id: Optional[str] = None

    aud_lock: threading.Lock = field(default_factory=threading.Lock)
    audience: Dict[str, float] = field(default_factory=dict)

    started: bool = False

    def __post_init__(self):
        self.cond = threading.Condition(self.lock)


def _cleanup_loop(state: QueueState) -> None:
    while True:
        time.sleep(max(5, CLEANUP_EVERY_SEC))
        now = time.time()
        with state.lock:
            # Drop completed jobs after TTL
            for jid in list(state.jobs.keys()):
                j = state.jobs[jid]
                idle = now - float(j.updated_ts or j.created_ts)
                if j.done and idle > JOB_TTL_SEC:
                    state.jobs.pop(jid, None)

            # Keep queue consistent
            state.queue[:] = [jid for jid in state.queue if jid in state.jobs]


def _job_cb(state: QueueState, job_id: str):
    def cb(pct: int, msg: str):
        with state.lock:
            j = state.jobs.get(job_id)
            if j and not j.done:
                j.pct = int(max(0, min(100, pct)))
                j.msg = str(msg)
                j.updated_ts = time.time()
    return cb


def _run_one_job(state: QueueState, job: Job) -> None:
    with state.lock:
        job.pct = 5
        job.msg = "Loading model / warming up…"
        job.updated_ts = time.time()

    cb = _job_cb(state, job.job_id)

    audios: List[np.ndarray] = []
    sr_ref: Optional[int] = None

    for idx, part in enumerate(job.text_chunks, start=1):
        with state.lock:
            job.msg = f"Generating chunk {idx}/{len(job.text_chunks)}…"
            job.updated_ts = time.time()

        wav, sr, _dbg = synthesize_text(part, job.opts, cb=cb)

        if sr_ref is None:
            sr_ref = sr
        audios.append(wav)

        gap_ms = int(job.opts.get("gap_ms", 0) or 0)
        if gap_ms > 0 and idx != len(job.text_chunks):
            audios.append(np.zeros((int(sr_ref * gap_ms / 1000.0),), dtype=np.float32))

    out_sr = int(sr_ref or 16000)
    out_audio = (
        np.concatenate([np.asarray(a, dtype=np.float32) for a in audios])
        if audios else np.zeros((0,), np.float32)
    )
    wav_bytes = to_wav_bytes(out_audio, out_sr)

    with state.lock:
        job.wav_bytes = wav_bytes
        job.sr = out_sr
        job.pct = 100
        job.msg = "Done"
        job.done = True
        job.updated_ts = time.time()


def _worker_loop(state: QueueState) -> None:
    while True:
        with state.cond:
            while not state.queue:
                state.cond.wait()

            job_id = state.queue.pop(0)
            state.active_job_id = job_id
            job = state.jobs.get(job_id)

        if job is None:
            with state.lock:
                state.active_job_id = None
            continue

        try:
            _run_one_job(state, job)
        except Exception as e:
            with state.lock:
                job.error = str(e)
                job.done = True
                job.msg = "Failed"
                job.pct = 100
                job.updated_ts = time.time()
        finally:
            with state.lock:
                state.active_job_id = None


def _start_threads(state: QueueState) -> None:
    with state.lock:
        if state.started:
            return
        state.started = True

    threading.Thread(target=_worker_loop, args=(state,), daemon=True).start()
    threading.Thread(target=_cleanup_loop, args=(state,), daemon=True).start()


@st.cache_resource
def get_state() -> QueueState:
    s = QueueState()
    _start_threads(s)
    return s


def _heartbeat_and_count(state: QueueState) -> int:
    sid = _get_session_id()
    now = time.time()
    with state.aud_lock:
        state.audience[sid] = now
        dead_before = now - (HEARTBEAT_SEC * 2.5)
        for k in list(state.audience.keys()):
            if state.audience[k] < dead_before:
                state.audience.pop(k, None)
        return len(state.audience)


def _queue_snapshot(state: QueueState) -> Tuple[Optional[str], List[str]]:
    with state.lock:
        return state.active_job_id, list(state.queue)


def _queue_position(state: QueueState, job_id: str) -> Tuple[int, int]:
    active, q = _queue_snapshot(state)
    total = (1 if active else 0) + len(q)
    if active == job_id:
        return 0, total
    if job_id in q:
        return q.index(job_id) + 1, total
    return -1, total


def _get_job(state: QueueState, job_id: str) -> Optional[Job]:
    with state.lock:
        return state.jobs.get(job_id)


def _enqueue_job(state: QueueState, job: Job) -> None:
    with state.cond:
        # depth guard
        if len(state.queue) >= QUEUE_MAX_JOBS:
            raise RuntimeError("Queue is full. Please try again later.")
        state.jobs[job.job_id] = job
        state.queue.append(job.job_id)
        state.cond.notify()


# ----------------------------
# Text splitting
# ----------------------------
_SPLIT_RE = re.compile(r"(?<=[\u1362.!?\u1367\u1368])\s+|[\r\n]+")

def split_text(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(t) if p.strip()]
    return parts if parts else [t]


# ----------------------------
# State + top bar (audience/queue via fragment so it updates without killing sessions)
# ----------------------------
state = get_state()

# Streamlit 1.52+ supports fragments; this updates ONLY the badges, not the entire app.
@st.fragment(run_every=f"{HEARTBEAT_SEC}s")
def audience_widget():
    audience = _heartbeat_and_count(state)
    active_id, queue_list = _queue_snapshot(state)
    queue_depth = len(queue_list) + (1 if active_id else 0)
    st.markdown(
        f"""
        <div style="display:flex; gap:10px; justify-content:flex-end; flex-wrap:wrap;">
          <div class="badge">Live audience: <b>{audience}</b></div>
          <div class="badge">Queue depth: <b>{queue_depth}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


title_col, badge_col = st.columns([1.8, 1.0], vertical_alignment="top")
with title_col:
    st.markdown(
        """
<div class="toprow">
  <div>
    <h1 style="margin:0;">Tigrinya Text-to-Speech</h1>
    <div class="small">Queued generation: users wait in line; stable on Streamlit Community.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with badge_col:
    audience_widget()


# ----------------------------
# Job UI (polls while waiting/running)
# ----------------------------
def _show_job_ui(job_id: str) -> None:
    job = _get_job(state, job_id)
    if not job:
        # Silent reset only (no annoying messages)
        st.session_state.pop("job_id", None)
        return

    pos, total = _queue_position(state, job_id)

    st.progress(int(job.pct))
    if pos == 0:
        st.markdown(f"<div class='small'><b>Running</b> — {job.msg}</div>", unsafe_allow_html=True)
    elif pos > 0:
        st.markdown(
            f"<div class='small'><b>Queued</b> — position {pos} of {total}. {job.msg}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div class='small'>{job.msg}</div>", unsafe_allow_html=True)

    if job.done:
        if job.error:
            st.error(f"Generation failed: {job.error}")
        elif job.wav_bytes:
            st.session_state["last_wav_bytes"] = job.wav_bytes
            st.audio(job.wav_bytes, format="audio/wav")
            st.download_button(
                "Download WAV",
                job.wav_bytes,
                file_name="tigrinya_tts.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        st.session_state.pop("job_id", None)
        return

    _safe_rerun_after(POLL_MS)


# If user already has a job, show it and do NOT allow new submission yet
if "job_id" in st.session_state:
    _show_job_ui(st.session_state["job_id"])
    if "last_wav_bytes" in st.session_state:
        st.markdown("<div class='small'>Last generated output:</div>", unsafe_allow_html=True)
        st.audio(st.session_state["last_wav_bytes"], format="audio/wav")
    st.stop()


# ----------------------------
# Main UI inside a FORM (prevents typing interruptions from reruns)
# ----------------------------
with st.form("tts_form", clear_on_submit=False):
    left, right = st.columns([1.6, 1.0], gap="large")

    with left:
        st.markdown("#### Text")
        text = st.text_area(
            "Text",
            height=170,
            placeholder="ኣብዚ ጽሑፍ ኣእትዉ።",
            label_visibility="collapsed",
        )
        if text and len(text) > MAX_CHARS:
            st.warning(f"Max {MAX_CHARS} chars (free tier protection). Please shorten.")

        st.markdown("#### G2P")
        g2p_enabled = st.toggle("Enable G2P", value=True)

        c1, c2 = st.columns([1.0, 1.0], gap="medium")
        with c1:
            g2p_mode = st.selectbox("Mode", ["normalize", "rules", "auto"], index=0, disabled=not g2p_enabled)
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

    st.markdown("")
    gen_btn = st.form_submit_button("Generate Voice", use_container_width=True)


# ----------------------------
# Enqueue new job
# ----------------------------
if gen_btn:
    if not (text or "").strip():
        st.error("Please enter text.")
        st.stop()
    if len(text) > MAX_CHARS:
        st.error(f"Too long. Max {MAX_CHARS} characters.")
        st.stop()

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
        processed = (processed.strip() if processed else text.strip())
    except Exception:
        processed = text.strip()

    if show_processed and g2p_enabled:
        st.code(processed, language="text")

    parts = split_text(processed) if split_long else [processed]
    parts = [p for p in parts if p.strip()]

    if not parts:
        st.error("No text to synthesize.")
        st.stop()
    if len(parts) > MAX_CHUNKS:
        st.warning(f"Too many chunks ({len(parts)}). Limiting to {MAX_CHUNKS}.")
        parts = parts[:MAX_CHUNKS]

    session_id = _get_session_id()
    opts = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "subdir": SUBDIR,
        "noise_scale": float(noise_scale),
        "noise_scale_duration": float(noise_scale_duration),
        "length_scale": float(length_scale),
        "gap_ms": int(gap_ms) if split_long else 0,
        "uroman_lang": "tir",
        "force_uroman": False,
        "min_tokens": 3,
    }

    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        created_ts=time.time(),
        session_id=session_id,
        text_chunks=parts,
        opts=opts,
        pct=0,
        msg="Queued…",
        done=False,
        updated_ts=time.time(),
    )

    try:
        _enqueue_job(state, job)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.session_state["job_id"] = job_id
    _show_job_ui(job_id)


# Last generated audio (stable)
if "last_wav_bytes" in st.session_state:
    st.markdown("<div class='small'>Last generated output:</div>", unsafe_allow_html=True)
    st.audio(st.session_state["last_wav_bytes"], format="audio/wav")
