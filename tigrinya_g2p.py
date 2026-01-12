from __future__ import annotations

import os
import re
import unicodedata
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ----------------------------
# Normalization regex
# ----------------------------
_WS_RE = re.compile(r"[ \t\u00A0\u2000-\u200B\u202F\u205F\u3000]+")
_NEWLINES_RE = re.compile(r"\r\n|\r|\n")
_MULTI_SPACE = re.compile(r"\s{2,}")

# Space handling around punctuation:
# - Remove spaces before punctuation
# - Ensure exactly one space after punctuation when followed by a letter/number (not newline/end)
_PUNCT = r"።፣፡፥፧፨!?.,:;()\[\]{}\"'"
_SPACE_BEFORE_PUNCT = re.compile(rf"\s+([{_PUNCT}])")
_SPACE_AFTER_PUNCT = re.compile(rf"([{_PUNCT}])([^\s{_PUNCT}])")  # Punct immediately followed by text

_MULTI_PUNCT_LATIN = re.compile(r"([!?.,])\1{2,}")   # !!!!! -> !!
_MULTI_STOP = re.compile(r"[.]{3,}")                # ..... -> ...

# Digits (optional spacing)
_DIGIT_RE = re.compile(r"\d+")
_LETTER_DIGIT = re.compile(r"([A-Za-zኀ-ፚ])(\d)")
_DIGIT_LETTER = re.compile(r"(\d)([A-Za-zኀ-ፚ])")


# Ethiopic punctuation normalization map (safe)
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
    rules: List[Tuple[str, str]] = None
    uroman_path: str = ""    # folder containing uroman.pl (optional)
    uroman_lang: str = "tir" # uroman language tag (best guess)
    keep_original_on_fail: bool = True

    # extra knobs (safe defaults)
    digit_strategy: str = "separate_words"  # separate_words | keep | spell_digits
    rules_are_regex: bool = False           # if True: FROM is regex pattern
    max_text_len: int = 10000               # safety limit


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


def _unicode_normalize(text: str) -> str:
    # NFC is safest for Ethiopic scripts
    return unicodedata.normalize("NFC", text)


def _normalize_whitespace(text: str) -> str:
    text = _NEWLINES_RE.sub("\n", text)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _normalize_punct(text: str) -> str:
    # map punctuation variants
    text = "".join(PUNCT_MAP.get(ch, ch) for ch in text)

    # Remove spaces before punctuation
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)

    # Ensure a single space after punctuation when followed by normal text
    text = _SPACE_AFTER_PUNCT.sub(r"\1 \2", text)

    # Compact repeated latin punctuation
    text = _MULTI_PUNCT_LATIN.sub(r"\1\1", text)
    text = _MULTI_STOP.sub("...", text)

    # Final whitespace cleanup
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _normalize_digits(text: str, strategy: str) -> str:
    strategy = (strategy or "separate_words").lower().strip()

    if strategy == "keep":
        # Just ensure digits don’t stick to letters: "abc12" -> "abc 12"
        text = _LETTER_DIGIT.sub(r"\1 \2", text)
        text = _DIGIT_LETTER.sub(r"\1 \2", text)
        return _MULTI_SPACE.sub(" ", text).strip()

    if strategy == "spell_digits":
        # 123 -> "1 2 3" (safe, language-agnostic)
        def repl(m):
            s = m.group(0)
            return " " + " ".join(list(s)) + " "
        text = _DIGIT_RE.sub(repl, text)
        return _MULTI_SPACE.sub(" ", text).strip()

    # default: separate_words (recommended)
    # "abc12def" -> "abc 12 def"
    text = _LETTER_DIGIT.sub(r"\1 \2", text)
    text = _DIGIT_LETTER.sub(r"\1 \2", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def normalize_only(text: str, digit_strategy: str = "separate_words") -> str:
    text = _unicode_normalize(text)
    text = _normalize_whitespace(text)
    text = _normalize_punct(text)
    text = _normalize_digits(text, digit_strategy=digit_strategy)
    return text


def _apply_rules(text: str, rules: List[Tuple[str, str]], regex: bool) -> str:
    if not rules:
        return text

    out = text
    if regex:
        for pat, rep in rules:
            try:
                out = re.sub(pat, rep, out)
            except re.error:
                # If a bad regex is provided, ignore it safely
                continue
        return out

    # plain replace (fast)
    for a, b in rules:
        out = out.replace(a, b)
    return out


def _run_uroman(text: str, uroman_path: str, lang: str) -> str:
    """
    Optional uroman (perl). Raises on failure; caller falls back.
    """
    uroman_path = (uroman_path or "").strip() or os.environ.get("UROMAN_PATH", "").strip()
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
        timeout=20,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "uroman failed")

    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError("uroman returned empty output")
    return out


def g2p(text: str, cfg: Optional[G2PConfig] = None) -> str:
    """
    Robust G2P entry point.
    Never crashes the app. Always returns a string (possibly empty).
    """
    cfg = cfg or G2PConfig()
    raw = text or ""

    # safety cap
    if len(raw) > int(cfg.max_text_len or 10000):
        raw = raw[: int(cfg.max_text_len)]

    try:
        base = normalize_only(raw, digit_strategy=cfg.digit_strategy)

        mode = (cfg.mode or "normalize").lower().strip()
        rules = cfg.rules or []

        if mode == "normalize":
            return base

        if mode == "rules":
            return _apply_rules(base, rules, regex=bool(cfg.rules_are_regex))

        if mode == "uroman":
            # apply uroman over normalized text
            return _run_uroman(base, cfg.uroman_path, cfg.uroman_lang)

        if mode == "auto":
            # auto: rules if provided else try uroman else normalize
            if rules:
                return _apply_rules(base, rules, regex=bool(cfg.rules_are_regex))
            try:
                return _run_uroman(base, cfg.uroman_path, cfg.uroman_lang)
            except Exception:
                return base

        # unknown -> safe fallback
        return base

    except Exception:
        # Never break pipeline
        if cfg.keep_original_on_fail:
            return normalize_only(raw, digit_strategy=cfg.digit_strategy) if raw else ""
        return ""


# ----------------------------
# CLI entry (stdin -> stdout)
# ----------------------------
def main_cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Robust Tigrinya G2P (stdin -> stdout)")
    parser.add_argument("--mode", default="normalize", choices=["normalize", "rules", "uroman", "auto"])
    parser.add_argument("--uroman_path", default="")
    parser.add_argument("--uroman_lang", default="tir")
    parser.add_argument("--rules_file", default="", help="Optional rules file: 'FROM => TO' per line")
    parser.add_argument("--rules_regex", action="store_true", help="Treat FROM as regex pattern")
    parser.add_argument("--digit_strategy", default="separate_words", choices=["separate_words", "keep", "spell_digits"])
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
        rules_are_regex=args.rules_regex,
    )

    inp = sys.stdin.read()
    sys.stdout.write(g2p(inp, cfg))


if __name__ == "__main__":
    main_cli()
