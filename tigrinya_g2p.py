# tigrinya_g2p.py
# Robust UTF-8 safe normalization + optional rules/uroman.
# Fixes:
# - Prevents regex encoding crashes (explicit Ethiopic unicode ranges)
# - Keeps punctuation definitions stable (no accidental mojibake)
# - Backward-compatible _normalize_digits signature (accepts digit_strategy / strategy)
# - Never breaks the pipeline: g2p() always returns something (unless configured otherwise)

from __future__ import annotations

import os
import re
import unicodedata
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


# ----------------------------
# Regex + normalization helpers (UTF-8 safe)
# ----------------------------
_WS_RE = re.compile(r"[ \t\u00A0\u2000-\u200B\u202F\u205F\u3000]+")
_NEWLINES_RE = re.compile(r"\r\n|\r|\n")
_MULTI_SPACE = re.compile(r"\s{2,}")

# Ethiopic punctuation + common punctuation
_PUNCT = r"።፣፡፥፧፨!?.,:;()\[\]{}\"'"
_SPACE_BEFORE_PUNCT = re.compile(rf"\s+([{_PUNCT}])")
_SPACE_AFTER_PUNCT = re.compile(rf"([{_PUNCT}])([^\s{_PUNCT}])")

_MULTI_PUNCT = re.compile(r"([!?.,])\1{2,}")  # !!!! -> !!
_MULTI_STOP = re.compile(r"[.]{3,}")          # ..... -> ...

# Digits
_DIGIT_RE = re.compile(r"\d+")

# Ethiopic Unicode ranges:
# - Ethiopic: U+1200..U+137F (basic Ethiopic block inc. punctuation)
# - Ethiopic Supplement: U+1380..U+139F
# - Ethiopic Extended: U+2D80..U+2DDF
# - Ethiopic Extended-A: U+AB00..U+AB2F
_ETHIOPIC_RANGES = r"\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F"

# Separate digits from letters to avoid "word12" sticking.
# NOTE: The character class includes ASCII letters + Ethiopic ranges above.
_LETTER_DIGIT = re.compile(rf"([A-Za-z{_ETHIOPIC_RANGES}])(\d)")
_DIGIT_LETTER = re.compile(rf"(\d)([A-Za-z{_ETHIOPIC_RANGES}])")

# Punctuation normalization map (safe)
PUNCT_MAP: Dict[str, str] = {
    "。": "።",
    "，": "፣",
    "؛": ";",
    "؟": "?",
    "“": '"',
    "”": '"',
    "„": '"',
    "’": "'",
    "‘": "'",
    "‐": "-",
    "–": "-",
    "—": "-",
}


@dataclass
class G2PConfig:
    mode: str = "normalize"  # normalize | rules | uroman | auto
    rules: Optional[List[Tuple[str, str]]] = None
    uroman_path: str = ""    # folder containing uroman.pl
    uroman_lang: str = "tir"
    keep_original_on_fail: bool = True
    digit_strategy: str = "separate_words"  # keep | spell_digits | separate_words
    max_text_len: int = 5000


# ----------------------------
# Rules helpers
# ----------------------------
def parse_rules(rules_text: str) -> List[Tuple[str, str]]:
    rules: List[Tuple[str, str]] = []
    for ln in (rules_text or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "=>" in ln:
            a, b = ln.split("=>", 1)
        elif "\t" in ln:
            a, b = ln.split("\t", 1)
        else:
            continue
        a = a.strip()
        b = b.strip()
        if a:
            rules.append((a, b))
    return rules


def _apply_rules(text: str, rules: List[Tuple[str, str]]) -> str:
    out = text
    for a, b in rules:
        out = out.replace(a, b)
    return out


# ----------------------------
# Normalization helpers
# ----------------------------
def _unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _normalize_whitespace(text: str) -> str:
    text = _NEWLINES_RE.sub("\n", text or "")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _normalize_punct(text: str) -> str:
    # map variants
    text = "".join(PUNCT_MAP.get(ch, ch) for ch in (text or ""))

    # remove spaces before punctuation and ensure a space after punctuation when followed by text
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT.sub(r"\1 \2", text)

    text = _MULTI_PUNCT.sub(r"\1\1", text)
    text = _MULTI_STOP.sub("...", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _normalize_digits(text: str, strategy: str = "separate_words", **kwargs) -> str:
    """
    Normalizes digits in a backward-compatible way.

    Supported:
      - _normalize_digits(text, strategy="keep")
      - _normalize_digits(text, digit_strategy="keep")   (older call sites)
      - _normalize_digits(text, strategy="spell_digits") etc.
    """
    # Back-compat: allow digit_strategy=...
    if "digit_strategy" in kwargs and kwargs["digit_strategy"]:
        strategy = kwargs["digit_strategy"]

    strategy = (strategy or "separate_words").lower().strip()
    text = text or ""

    if strategy == "keep":
        # Ensure digits don’t stick to letters: "abc12" -> "abc 12"
        text = _LETTER_DIGIT.sub(r"\1 \2", text)
        text = _DIGIT_LETTER.sub(r"\1 \2", text)
        return _MULTI_SPACE.sub(" ", text).strip()

    if strategy == "spell_digits":
        # "123" -> "1 2 3"
        def repl(m: re.Match) -> str:
            s = m.group(0)
            return " " + " ".join(list(s)) + " "

        text = _DIGIT_RE.sub(repl, text)
        text = _MULTI_SPACE.sub(" ", text)
        return text.strip()

    # default: separate_words
    text = _LETTER_DIGIT.sub(r"\1 \2", text)
    text = _DIGIT_LETTER.sub(r"\1 \2", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def normalize_only(text: str, digit_strategy: str = "separate_words") -> str:
    text = _unicode_normalize(text)
    text = _normalize_whitespace(text)
    text = _normalize_punct(text)
    # call with digit_strategy for compatibility with older internal use
    text = _normalize_digits(text, digit_strategy=digit_strategy)
    return text


# ----------------------------
# Uroman integration (optional)
# ----------------------------
def _run_uroman(text: str, uroman_path: str, lang: str) -> str:
    uroman_path = (uroman_path or os.environ.get("UROMAN_PATH", "")).strip()
    if not uroman_path:
        raise RuntimeError("UROMAN_PATH not set.")
    uroman_pl = os.path.join(uroman_path, "uroman.pl")
    if not os.path.isfile(uroman_pl):
        raise RuntimeError(f"uroman.pl not found in: {uroman_path}")

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


# ----------------------------
# Main G2P function (never breaks pipeline)
# ----------------------------
def g2p(text: str, cfg: Optional[G2PConfig] = None) -> str:
    cfg = cfg or G2PConfig()
    raw = (text or "")[: int(cfg.max_text_len)]

    try:
        base = normalize_only(raw, digit_strategy=cfg.digit_strategy)

        mode = (cfg.mode or "normalize").lower().strip()
        rules = cfg.rules or []

        if mode == "normalize":
            return base

        if mode == "rules":
            return _apply_rules(base, rules)

        if mode == "uroman":
            return _run_uroman(base, cfg.uroman_path, cfg.uroman_lang)

        if mode == "auto":
            if rules:
                return _apply_rules(base, rules)
            try:
                return _run_uroman(base, cfg.uroman_path, cfg.uroman_lang)
            except Exception:
                return base

        # unknown mode -> fallback
        return base

    except Exception:
        # Never break pipeline
        if cfg.keep_original_on_fail:
            return normalize_only(raw, digit_strategy=cfg.digit_strategy)
        return ""


# ----------------------------
# CLI (optional)
# ----------------------------
def main_cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Robust Tigrinya G2P (stdin -> stdout)")
    parser.add_argument("--mode", default="normalize", choices=["normalize", "rules", "uroman", "auto"])
    parser.add_argument("--uroman_path", default="")
    parser.add_argument("--uroman_lang", default="tir")
    parser.add_argument("--rules_file", default="")
    parser.add_argument("--digit_strategy", default="separate_words", choices=["keep", "spell_digits", "separate_words"])
    args = parser.parse_args()

    rules: List[Tuple[str, str]] = []
    if args.rules_file:
        with open(args.rules_file, "r", encoding="utf-8") as f:
            rules = parse_rules(f.read())

    cfg = G2PConfig(
        mode=args.mode,
        rules=rules,
        uroman_path=args.uroman_path,
        uroman_lang=args.uroman_lang,
        keep_original_on_fail=True,
        digit_strategy=args.digit_strategy,
    )


    inp = sys.stdin.read()
    sys.stdout.write(g2p(inp, cfg))


if __name__ == "__main__":
    main_cli()
