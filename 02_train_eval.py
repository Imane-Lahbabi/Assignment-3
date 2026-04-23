"""
Train unigram, bigram, trigram modified-KN language models on the two
Darija corpora (Arabic script and Latin script) and evaluate.
"""

import random
import time
from pathlib import Path

from ngram_lm import NGramLM

random.seed(42)
PROC = Path("/home/claude/processed")
MODELS = Path("/home/claude/models")
MODELS.mkdir(exist_ok=True)


def load_sents(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]


def train_and_eval(script_label, human_name):
    print(f"\n{'='*60}\n  {human_name}  ({script_label})\n{'='*60}")
    train = load_sents(PROC / f"{script_label}.train.txt")
    dev   = load_sents(PROC / f"{script_label}.dev.txt")
    test  = load_sents(PROC / f"{script_label}.test.txt")

    n_train_tokens = sum(len(s) for s in train)
    n_dev_tokens   = sum(len(s) for s in dev)
    n_test_tokens  = sum(len(s) for s in test)
    print(f"  train: {len(train):,} sentences / {n_train_tokens:,} tokens")
    print(f"  dev  : {len(dev):,} sentences / {n_dev_tokens:,} tokens")
    print(f"  test : {len(test):,} sentences / {n_test_tokens:,} tokens")

    # keep words seen >= 2 times; everything else -> <unk>
    results = {}
    for n in (1, 2, 3):
        print(f"\n  --- training {n}-gram model ---")
        t0 = time.time()
        lm = NGramLM(n=n, unk_threshold=1)
        lm.fit(train)
        t_fit = time.time() - t0
        print(f"    fit time       : {t_fit:,.1f} s")
        print(f"    vocabulary size: {lm.V:,}")
        print(f"    # {n}-gram types : {len(lm.counts[n]):,}")
        if n >= 2:
            print(f"    KN discounts   : D1={lm.discounts[2][0]:.3f}, "
                  f"D2={lm.discounts[2][1]:.3f}, D3+={lm.discounts[2][2]:.3f}"
                  f"  (bigram order)")
        if n == 3:
            print(f"                     D1={lm.discounts[3][0]:.3f}, "
                  f"D2={lm.discounts[3][1]:.3f}, D3+={lm.discounts[3][2]:.3f}"
                  f"  (trigram order)")

        t0 = time.time()
        ppl_dev,  _ = lm.perplexity(dev)
        ppl_test, _ = lm.perplexity(test)
        t_eval = time.time() - t0
        print(f"    dev  perplexity: {ppl_dev:,.2f}")
        print(f"    test perplexity: {ppl_test:,.2f}   (eval {t_eval:.1f}s)")
        results[n] = (ppl_dev, ppl_test)

        path = MODELS / f"{script_label}_{n}gram.pkl"
        lm.save(path)
        print(f"    saved -> {path.name}")

        # keep the trigram model around for sampling afterwards
        if n == 3:
            final_lm = lm

    print(f"\n  --- sampled generations (trigram, KN) ---")
    rng = random.Random(1)
    for i in range(5):
        toks = final_lm.generate(max_tokens=25, rng=rng)
        print(f"    [{i+1}] " + ' '.join(toks))

    return results


if __name__ == "__main__":
    ar_results = train_and_eval('ar', 'Arabic-script Darija')
    la_results = train_and_eval('la', 'Latin-script Darija (Arabizi)')

    print("\n\n" + "="*60)
    print("  SUMMARY — test-set perplexity")
    print("="*60)
    print(f"{'':<30} {'unigram':>10} {'bigram':>10} {'trigram':>10}")
    for label, results in [('Arabic-script Darija', ar_results),
                           ('Latin-script Darija (Arabizi)', la_results)]:
        print(f"{label:<30} "
              f"{results[1][1]:>10,.1f} "
              f"{results[2][1]:>10,.1f} "
              f"{results[3][1]:>10,.1f}")
