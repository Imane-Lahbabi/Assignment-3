"""
Preprocessing for Darija n-gram language model.

Decisions:
- Darija appears in two distinct scripts in this corpus, so we train two separate
  models instead of mixing them:
    * Arabic-script Darija  -> sources: darija-wiki, goud.ma, story-data, Youtube
    * Latin-script Darija   -> sources: music-data, twitter
  Mixing would give a model that is bad at both.
- To keep training tractable on one machine we cap each corpus at a target size.
  story-data and twitter are huge (~380 MB each); we sample lines from them.
- Word-level tokenization (the classic n-gram unit). We lowercase only the Latin
  side. For Arabic we strip tatweel and keep the text as-is.
- Sentence boundaries: we treat each non-empty line as a sentence. This is not
  perfect for news articles but it's the standard pragmatic choice for this
  kind of multi-source corpus and matches the line-oriented nature of Twitter,
  music lyrics, and YouTube comments.
- We add <s> and </s> boundary tokens during training.
"""

import os
import re
import random
import unicodedata
from pathlib import Path

random.seed(42)

DATA_ROOT = Path("/home/data")
OUT_ROOT = Path("/home/processed")
OUT_ROOT.mkdir(exist_ok=True)

# Per-script source selection and per-source caps (in MB of raw text).
# Caps chosen to give a training corpus of ~30-50 MB per script, which is
# plenty for trigram estimation and still fits comfortably in RAM.
ARABIC_SOURCES = [
    ("darija-wiki",  None),    # ~4.6 MB, take all
    ("goud.ma",      None),    # ~3.3 MB, take all
    ("Youtube",      None),    # ~4.0 MB, take all
    ("story-data",    25),     # cap at 25 MB (out of 378)
]
LATIN_SOURCES = [
    ("music-data",   None),    # ~4.0 MB, take all
    ("twitter",       25),     # cap at 25 MB (mostly latin script) out of 389
]

# --- Character class helpers -------------------------------------------------

ARABIC_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
LATIN_RE  = re.compile(r'[A-Za-z]')
# Arabic tatweel (kashida) -- purely decorative, remove before tokenizing
TATWEEL = '\u0640'
# Arabic diacritics (fatha, damma, kasra, shadda, sukun, ...). Often missing
# in Darija text but when present they inflate vocabulary unnecessarily.
DIACRITICS_RE = re.compile(r'[\u064B-\u0652\u0670\u06D6-\u06ED]')

def script_of(line: str) -> str:
    """Return 'ar', 'la', or 'other' based on which script dominates the line."""
    ar = len(ARABIC_RE.findall(line))
    la = len(LATIN_RE.findall(line))
    if ar == 0 and la == 0:
        return 'other'
    return 'ar' if ar >= la else 'la'

# --- Normalization & tokenization -------------------------------------------

def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.replace(TATWEEL, '')
    text = DIACRITICS_RE.sub('', text)
    # normalize common Arabic punctuation to simple ASCII
    text = text.replace('،', ',').replace('؛', ';').replace('؟', '?')
    # strip zero-width chars
    text = re.sub(r'[\u200B-\u200F\u202A-\u202E\uFEFF]', '', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_latin(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Word tokenizer: keep digits (essential for Arabizi like "3lik", "7aja", "9lbi")
# Split on whitespace and on any char that is not a letter or digit.
WORD_RE_AR = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
WORD_RE_LA = re.compile(r"[A-Za-z0-9']+")

def tokenize_arabic(text: str):
    return WORD_RE_AR.findall(text)

def tokenize_latin(text: str):
    # keep apostrophes inside words (e.g. "f'tenue", "l'kerrata")
    return WORD_RE_LA.findall(text)

# --- Reading with caps -------------------------------------------------------

def iter_lines_capped(source_dir: Path, cap_mb):
    """Yield raw lines from every .txt under source_dir, stopping after cap_mb
    if cap_mb is not None. cap_mb is an approximate byte cap (UTF-8)."""
    cap_bytes = None if cap_mb is None else cap_mb * 1024 * 1024
    seen = 0
    files = sorted([p for p in source_dir.rglob("*.txt")])
    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line.strip():
                        continue
                    seen += len(line.encode('utf-8'))
                    yield line
                    if cap_bytes is not None and seen >= cap_bytes:
                        return
        except Exception as e:
            print(f"  ! skip {fp}: {e}")

# --- Build per-script corpus -------------------------------------------------

def build_corpus(script_label, sources, normalize_fn, tokenize_fn):
    print(f"\n=== Building {script_label} corpus ===")
    all_sentences = []  # list of token lists
    for subdir, cap in sources:
        src_path = DATA_ROOT / subdir
        if not src_path.exists():
            print(f"  ! missing {src_path}")
            continue
        print(f"  reading {subdir} (cap={cap} MB)")
        n_lines = 0
        n_kept = 0
        for raw in iter_lines_capped(src_path, cap):
            n_lines += 1
            if script_of(raw) != script_label:
                continue
            norm = normalize_fn(raw)
            toks = tokenize_fn(norm)
            if len(toks) < 3:   # skip too-short lines, useless for trigrams
                continue
            all_sentences.append(toks)
            n_kept += 1
        print(f"    scanned {n_lines:,} lines, kept {n_kept:,}")
    print(f"  TOTAL sentences kept: {len(all_sentences):,}")
    total_tokens = sum(len(s) for s in all_sentences)
    print(f"  TOTAL tokens: {total_tokens:,}")
    return all_sentences

def split_and_write(sentences, out_prefix, dev_frac=0.02, test_frac=0.02):
    random.shuffle(sentences)
    n = len(sentences)
    n_dev  = max(1, int(n * dev_frac))
    n_test = max(1, int(n * test_frac))
    dev  = sentences[:n_dev]
    test = sentences[n_dev:n_dev + n_test]
    train = sentences[n_dev + n_test:]
    for split_name, split_data in [('train', train), ('dev', dev), ('test', test)]:
        path = OUT_ROOT / f"{out_prefix}.{split_name}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            for toks in split_data:
                f.write(' '.join(toks) + '\n')
        print(f"  wrote {path.name}: {len(split_data):,} sentences, "
              f"{sum(len(s) for s in split_data):,} tokens")
    return len(train), len(dev), len(test)

# --- Run ---------------------------------------------------------------------

if __name__ == "__main__":
    ar_sents = build_corpus('ar', ARABIC_SOURCES, normalize_arabic, tokenize_arabic)
    split_and_write(ar_sents, "ar")

    la_sents = build_corpus('la', LATIN_SOURCES, normalize_latin, tokenize_latin)
    split_and_write(la_sents, "la")

    print("\nDone.")
