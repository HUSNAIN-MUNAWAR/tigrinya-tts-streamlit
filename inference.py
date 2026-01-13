# inference.py
from __future__ import annotations

import io
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

try:
    from transformers import VitsModel
except Exception:
    VitsModel = None


ProgressCB = Optional[Callable[[int, str], None]]  # (percent, message)


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str = "main"
    subdir: str = ""  # "round_01" if files are inside that folder


# ----------------------------
# Progress
# ----------------------------
def _emit(cb: ProgressCB, pct: int, msg: str) -> None:
    if cb is None:
        return
    try:
        cb(int(max(0, min(100, pct))), str(msg))
    except Exception:
        pass


# ----------------------------
# HF helpers
# ----------------------------
def _hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or None


def _default_cache_dir() -> Optional[str]:
    # If HF_HOME is set, HF will place cache there. Otherwise default (~/.cache/huggingface)
    return os.getenv("HF_HOME") or None


def download_inference_files(spec: ModelSpec, cb: ProgressCB = None) -> str:
    """
    Download only minimal inference artifacts and return a LOCAL DIR.
    Skips training checkpoints completely.
    """
    if not spec.repo_id or spec.repo_id.count("/") > 1:
        raise ValueError("Invalid repo_id; expected 'repo' or 'namespace/repo'.")

    token = _hf_token()
    cache_dir = _default_cache_dir()

    # Minimal inference artifacts only (avoid checkpoint folders)
    if spec.subdir.strip():
        p = spec.subdir.strip().rstrip("/")
        allow_patterns = [
            f"{p}/model.safetensors",
            f"{p}/config.json",
            f"{p}/preprocessor_config.json",
            f"{p}/tokenizer_config.json",
            f"{p}/special_tokens_map.json",
            f"{p}/vocab.json",
            f"{p}/added_tokens.json",
            f"{p}/*.json",
        ]
    else:
        allow_patterns = [
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "added_tokens.json",
            "*.json",
        ]

    ignore_patterns = [
        "checkpoint-*/**",
        "**/checkpoint-*/**",
        "**/optimizer*.bin",
        "**/scheduler*.bin",
        "**/random_states*.pkl",
        "**/trainer_state.json",
        "**/rng_state.pth",
        "**/training_args.bin",
    ]

    _emit(cb, 5, "Checking model files on Hugging Face…")
    local_root = snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        cache_dir=cache_dir,
    )

    local_dir = os.path.join(local_root, spec.subdir) if spec.subdir.strip() else local_root
    if not os.path.isdir(local_dir):
        raise RuntimeError(f"Invalid local model directory: {local_dir}")

    weights = os.path.join(local_dir, "model.safetensors")
    config = os.path.join(local_dir, "config.json")
    if not os.path.isfile(weights) or not os.path.isfile(config):
        raise RuntimeError(
            "Inference files not found in resolved directory. "
            "Ensure model.safetensors and config.json are present in repo root or SUBDIR."
        )

    _emit(cb, 25, "Model files cached locally.")
    return local_dir


# ----------------------------
# uroman (for Ethiopic/non-Roman text)
# ----------------------------
def _find_uroman_pl() -> Optional[str]:
    """
    Returns path to uroman.pl if available.
    Priority:
      1) env UROMAN_PL
      2) env UROMAN_PATH + /uroman.pl
      3) python package uroman (pip install uroman) - locate its uroman.pl
    """
    p = (os.getenv("UROMAN_PL") or "").strip()
    if p and os.path.isfile(p):
        return p

    upath = (os.getenv("UROMAN_PATH") or "").strip()
    if upath:
        cand = os.path.join(upath, "uroman.pl")
        if os.path.isfile(cand):
            return cand

    # Try python package `uroman`
    try:
        import uroman  # type: ignore
        pkg_dir = os.path.dirname(uroman.__file__)
        cand = os.path.join(pkg_dir, "uroman.pl")
        if os.path.isfile(cand):
            return cand
        # Some distributions keep it under `bin/`
        cand2 = os.path.join(pkg_dir, "bin", "uroman.pl")
        if os.path.isfile(cand2):
            return cand2
    except Exception:
        pass

    return None


def _uromanize(text: str, lang: str = "tir") -> str:
    uroman_pl = _find_uroman_pl()
    if not uroman_pl:
        raise RuntimeError("uroman.pl not found. Install `uroman` or set UROMAN_PATH/UROMAN_PL.")

    cmd = ["perl", uroman_pl]
    if lang:
        cmd += ["-l", lang]

    p = subprocess.run(
        cmd,
        input=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "uroman failed.")

    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError("uroman returned empty output.")
    return out


def _looks_non_roman(text: str) -> bool:
    # Heuristic: Ethiopic block present -> non-roman
    return any(0x1200 <= ord(ch) <= 0x137F for ch in text)


# ----------------------------
# Model + processor caches
# ----------------------------
@lru_cache(maxsize=2)
def load_processor(model_dir: str):
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Invalid model dir: {model_dir}")
    return AutoProcessor.from_pretrained(model_dir)


@lru_cache(maxsize=2)
def load_model(model_dir: str, device: str):
    if VitsModel is None:
        raise RuntimeError("transformers VitsModel is not available. Pin transformers to a version that includes VitsModel.")
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Invalid model dir: {model_dir}")

    m = VitsModel.from_pretrained(model_dir)
    m.to(device)
    m.eval()
    return m


# ----------------------------
# Audio helper
# ----------------------------
def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


# ----------------------------
# Main synthesis
# ----------------------------
def synthesize_text(
    text: str,
    opts: Dict,
    cb: ProgressCB = None,
) -> Tuple[np.ndarray, int, Dict]:
    """
    Returns: (audio_float32, sample_rate, debug_dict)

    opts:
      repo_id, revision, subdir
      noise_scale, noise_scale_duration, length_scale
      device (optional)
      uroman_lang (optional, default "tir")
      force_uroman (optional bool)
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty text.")

    repo_id = opts.get("repo_id", "")
    revision = opts.get("revision", "main")
    subdir = opts.get("subdir", "")

    device = opts.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    noise_scale = float(opts.get("noise_scale", 0.24))
    noise_scale_duration = float(opts.get("noise_scale_duration", 0.30))
    length_scale = float(opts.get("length_scale", 0.90))

    uroman_lang = (opts.get("uroman_lang") or "tir").strip()
    force_uroman = bool(opts.get("force_uroman", False))

    spec = ModelSpec(repo_id=repo_id, revision=revision, subdir=subdir)

    _emit(cb, 10, "Caching model files…")
    model_dir = download_inference_files(spec, cb=cb)

    _emit(cb, 30, "Loading processor/model…")
    proc = load_processor(model_dir)
    model = load_model(model_dir, device)

    # If the text is Ethiopic (non-Roman), romanize (Streamlit Cloud warning indicates this is needed)
    # We only do this when:
    #   - forced, OR
    #   - heuristically non-Roman
    # If uroman isn't available, we proceed without it but then enforce empty-token guard.
    original_t = t
    did_uroman = False
    if force_uroman or _looks_non_roman(t):
        try:
            _emit(cb, 40, "Applying uroman romanizer…")
            t = _uromanize(t, lang=uroman_lang)
            did_uroman = True
        except Exception:
            # continue; guard below will catch empty input_ids
            t = original_t

    _emit(cb, 50, "Tokenizing…")
    inputs = proc(t, return_tensors="pt")

    input_ids = inputs.get("input_ids", None)
    if input_ids is None or input_ids.numel() == 0 or input_ids.shape[-1] == 0:
        raise RuntimeError(
            "Tokenizer produced empty input_ids (no tokens). "
            "Fix: ensure `uroman` is installed on Streamlit Cloud and romanize Ethiopic text. "
            "Your requirements.txt already includes `uroman`; re-deploy and try again."
        )

    input_ids = input_ids.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    _emit(cb, 70, "Generating waveform…")
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

    _emit(cb, 95, "Finalizing…")
    dbg = {
        "model_dir": model_dir,
        "device": device,
        "repo_id": repo_id,
        "revision": revision,
        "subdir": subdir,
        "did_uroman": did_uroman,
    }
    _emit(cb, 100, "Done.")
    return wav, sr, dbg
