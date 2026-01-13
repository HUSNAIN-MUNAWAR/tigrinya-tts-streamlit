# inference.py
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

try:
    from transformers import VitsModel
except Exception as e:
    VitsModel = None


ProgressCB = Optional[Callable[[int, str], None]]  # (percent, message)


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str = "main"
    subdir: str = ""  # "round_01" if the inference files are actually inside that folder


def _emit(cb: ProgressCB, pct: int, msg: str) -> None:
    if cb is None:
        return
    try:
        cb(int(max(0, min(100, pct))), str(msg))
    except Exception:
        pass


def _hf_token() -> Optional[str]:
    # Prefer env; Streamlit secrets can inject to env as well.
    return os.getenv("HF_TOKEN") or None


def _default_cache_dir() -> Optional[str]:
    """
    Streamlit Community containers typically keep /home/appuser/.cache during runtime.
    Not guaranteed permanent across restarts, but best practice is to use HF cache.
    """
    return os.getenv("HF_HOME") or None  # if set, HF will use it


def download_inference_files(spec: ModelSpec, cb: ProgressCB = None) -> str:
    """
    Download only the minimal inference artifacts and return a LOCAL DIR.
    Skips training checkpoints completely.
    """
    if not spec.repo_id or spec.repo_id.count("/") > 1:
        raise ValueError("Invalid repo_id; expected 'repo' or 'namespace/repo'.")

    token = _hf_token()
    cache_dir = _default_cache_dir()

    # Only inference artifacts; DO NOT PULL checkpoint-* folders
    allow_patterns = []
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

    # Must exist for inference
    weights = os.path.join(local_dir, "model.safetensors")
    config = os.path.join(local_dir, "config.json")
    if not os.path.isfile(weights) or not os.path.isfile(config):
        raise RuntimeError(
            "Inference files not found in resolved directory. "
            "Ensure model.safetensors and config.json are present in repo root or SUBDIR."
        )

    _emit(cb, 25, "Model files cached locally.")
    return local_dir


@lru_cache(maxsize=2)
def load_processor(model_dir: str):
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Invalid model dir: {model_dir}")
    return AutoProcessor.from_pretrained(model_dir)


@lru_cache(maxsize=2)
def load_model(model_dir: str, device: str):
    if VitsModel is None:
        raise RuntimeError("transformers VitsModel is not available. Fix transformers version.")
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Invalid model dir: {model_dir}")

    m = VitsModel.from_pretrained(model_dir)
    m.to(device)
    m.eval()
    return m


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


def synthesize_text(
    text: str,
    opts: Dict,
    cb: ProgressCB = None,
) -> Tuple[np.ndarray, int, Dict]:
    """
    Returns: (audio_float32, sample_rate, debug_dict)
    opts required:
      repo_id, revision, subdir
      noise_scale, noise_scale_duration, length_scale
      device (optional)
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

    spec = ModelSpec(repo_id=repo_id, revision=revision, subdir=subdir)
    model_dir = download_inference_files(spec, cb=cb)

    _emit(cb, 35, "Loading processor/model into memory…")
    proc = load_processor(model_dir)
    model = load_model(model_dir, device)

    inputs = proc(t, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    _emit(cb, 55, "Generating waveform…")
    with torch.no_grad():
        # Some transformers builds do NOT have generate_speech
        if hasattr(model, "generate_speech"):
            wav = model.generate_speech(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noise_scale=noise_scale,
                noise_scale_duration=noise_scale_duration,
                length_scale=length_scale,
            )
            wav = wav.detach().cpu().numpy()
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # out.waveform is the correct fallback for many versions
            wav = out.waveform.detach().cpu().numpy()

    wav = np.squeeze(wav).astype(np.float32)
    sr = int(getattr(model.config, "sampling_rate", 16000))
    _emit(cb, 95, "Finalizing…")

    dbg = {"model_dir": model_dir, "device": device, "repo_id": repo_id, "revision": revision, "subdir": subdir}
    _emit(cb, 100, "Done.")
    return wav, sr, dbg
