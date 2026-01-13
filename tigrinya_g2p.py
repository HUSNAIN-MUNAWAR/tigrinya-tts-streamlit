# tigrinya_g2p.py
from __future__ import annotations

import os
import re
import unicodedata
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

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

# Ethiopic Unicode ranges (safe + explicit)
# Basic Ethiopic block: U+1200..U+137F (includes most Tigrinya letters + punctuation)
_ETHIOPIC = r"\u1200-\u137F"

# Separate digits from letters to avoid "word12" sticking
_LETTER_DIGIT = re.compile(rf"([A-Za-z{_ETHIOPIC}])(\d)")
_DIGIT_LETTER = re.compile(rf"(\d)([A-Za-z{_ETHIOPIC}])")

PUNCT_MAP = {
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


def parse_rules(rules_text: str) -> List[Tuple[str, str]]:
    rules: List[Tuple[str, str]] = []
    for ln in rules_text.splitlines():
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


def _unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _normalize_whitespace(text: str) -> str:
    text = _NEWLINES_RE.sub("\n", text)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _normalize_punct(text: str) -> str:
    # map variants
    text = "".join(PUNCT_MAP.get(ch, ch) for ch in text)

    # remove spaces before punctuation and ensure a space after punctuation when followed by text
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT.sub(r"\1 \2", text)

    text = _MULTI_PUNCT.sub(r"\1\1", text)
    text = _MULTI_STOP.sub("...", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _normalize_digits(text: str, strategy: str = "separate_words", **kwargs) -> str:
    # Back-compat: allow digit_strategy=...
    if "digit_strategy" in kwargs and kwargs["digit_strategy"]:
        strategy = kwargs["digit_strategy"]

    strategy = (strategy or "separate_words").lower().strip()

    if strategy == "keep":
        text = _LETTER_DIGIT.sub(r"\1 \2", text)
        text = _DIGIT_LETTER.sub(r"\1 \2", text)
        return _MULTI_SPACE.sub(" ", text).strip()

    if strategy == "spell_digits":
        def repl(m):
            s = m.group(0)
            return " " + " ".join(list(s)) + " "
        text = _DIGIT_RE.sub(repl, text)
        return _MULTI_SPACE.sub(" ", text).strip()

    # default: separate_words
    text = _LETTER_DIGIT.sub(r"\1 \2", text)
    text = _DIGIT_LETTER.sub(r"\1 \2", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def normalize_only(text: str, digit_strategy: str = "separate_words") -> str:
    text = _unicode_normalize(text)
    text = _normalize_whitespace(text)
    text = _normalize_punct(text)
    text = _normalize_digits(text, strategy=digit_strategy)
    return text


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
    out = p.stdout.strip()
    if not out:
        raise RuntimeError("uroman returned empty output.")
    return out


def g2p(text: str, cfg: Optional[G2PConfig] = None) -> str:
    cfg = cfg or G2PConfig()
    raw = (text or "")
    raw = raw[: int(cfg.max_text_len)]

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

        return base
    except Exception:
        return normalize_only(raw, digit_strategy=cfg.digit_strategy) if cfg.keep_original_on_fail else ""


def main_cli():
    import argparse
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

    import sys
    inp = sys.stdin.read()
    sys.stdout.write(g2p(inp, cfg))


if __name__ == "__main__":
    main_cli()
