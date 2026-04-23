# Probabilistic n-gram language model for Darija

## 1. Decisions

**Data source selection — split by script.** The uploaded corpus (~820 MB from 6 sources) is actually *two* languages at the word-level view:

- **Arabic-script Darija** — `darija-wiki`, `goud.ma`, `story-data`, `Youtube`.
- **Latin-script Darija / Arabizi** — `music-data`, `twitter` (e.g. `"dima khouk la costa f'tenue"`, `"3dyani 3dyani"`).

Training a single mixed model would share no vocabulary between the two halves and be worse at both. I trained **one model per script**.

**Amount of data used for training.** Story-data (378 MB) and twitter (389 MB) are orders of magnitude larger than the other sources; I capped each at 25 MB to keep training tractable and to avoid letting story-data's style dominate the Arabic model. Final training-set sizes:

| Script       | Train sentences | Train tokens | Vocab (count > 1) |
|--------------|-----------------|-------------:|------------------:|
| Arabic       |         195,323 |    3,503,148 |           139,901 |
| Latin        |         124,006 |      901,972 |            59,029 |

Data was split **96% / 2% / 2%** into train / dev / test.

**Choice of n.** I trained **unigram, bigram, and trigram** and kept the **trigram (n = 3)** as the main model. Bigram already cuts perplexity by roughly 5–6×, and trigram gives another ~2× reduction. Going higher (4-gram, 5-gram) on this data size would expand the model file dramatically and give diminishing returns: most 4-grams would be singletons, which is what KN smoothing already backs off against. Trigram is the classical sweet spot.

**Smoothing.** Plain MLE is unusable (any unseen trigram gives 0 probability). Add-k / Laplace overweights the uniform base distribution and performs poorly on real text. I implemented **modified Kneser-Ney smoothing (Chen & Goodman, 1998)** from scratch:

- Three separate discounts D₁, D₂, D₃₊ per order, estimated from the count-of-counts N₁…N₄ via Y = N₁ / (N₁ + 2·N₂).
- Trigram interpolates with a **KN-bigram** lower-order distribution whose numerator uses continuation counts |{u' : c(u', u, w) > 0}|, not raw counts — this is the whole point of KN (the word "Francisco" gets most of its bigram mass only from "San", so its unigram *continuation* probability should be low).
- Bigram in turn interpolates with a **continuation-unigram** |{u : c(u, w) > 0}| / (# distinct bigram types).

Out-of-vocabulary handling: words seen only once in training are replaced by `<unk>`, so the model has a real OOV probability at test time. Sentences are padded with `<s> <s>` and `</s>`; perplexity is computed over all real tokens plus `</s>` (not charged for `<s>`).

## 2. Pipeline

```
data.rar
   │ unrar-free
   ▼
/home/data/{darija-wiki,goud.ma,music-data,story-data,twitter,Youtube}
   │ 01_preprocess.py
   │  - script-detect each line (Arabic vs Latin)
   │  - NFKC; strip tatweel + Arabic diacritics; lower-case Latin
   │  - word tokenizer (keeps digits: '3lik','7aja','9lbi' stay as one token)
   │  - cap story-data / twitter at 25 MB; 96/2/2 split
   ▼
/home/processed/{ar,la}.{train,dev,test}.txt
   │ 02_train_eval.py  +  ngram_lm.py
   │  - unk_threshold = 1 (singletons -> <unk>)
   │  - modified-KN, orders 1, 2, 3
   │  - perplexity on dev + test
   │  - sampled generations from trigram
   ▼
/home/models/{ar,la}_{1,2,3}gram.pkl
```

## 3. Results — perplexity on held-out test set

| Model                        | Unigram | Bigram | **Trigram** |
|------------------------------|--------:|-------:|------------:|
| Arabic-script Darija         | 4,327.7 |  693.6 |   **405.2** |
| Latin-script Darija (Arabizi)|   904.2 |  154.3 |    **77.3** |

Expected-shape result: perplexity drops monotonically with n in both cases, and by a large margin. The Latin-script model is much stronger (lower PPL) partly because its vocabulary is smaller (~59K vs ~140K) and partly because music lyrics and tweets are more repetitive than the more heterogeneous Arabic-script sources.

Modified-KN discounts that were learned (sanity check — they land in sensible ranges):

- Arabic bigram: D₁=0.74, D₂=1.15, D₃₊=1.44 — trigram: D₁=0.85, D₂=1.18, D₃₊=1.33
- Latin  bigram: D₁=0.54, D₂=1.64, D₃₊=1.73 — trigram: D₁=0.64, D₂=1.70, D₃₊=1.70

## 4. Example trigram generations

**Arabic:**
1. انا مالكي ياك لباس
2. ياقوت تبسمات و قربات عندها
3. مراد هههههه مانقدرش نحيد ليك يدك بغيت ولادنا الله اش هاد

**Latin (Arabizi):**
1. hadi hiya lekhra ma gha n3elmek
2. khawi lkhwa risquina
3. la kayn chi mochkil hhhhhhhh
4. zgelti train li dana

These are plausible Darija fragments (characteristic `hhh…` laughter, `ma … ch` negation, code-switching like `train`, `mochkil`), which is what a word-trigram of this size should be able to produce.

## 5. How to use a trained model

```python
from ngram_lm import NGramLM

m = NGramLM.load("models/la_3gram.pkl")

# Perplexity
ppl, n_tokens = m.perplexity([["had", "chi", "ma", "bghach", "y3awed"]])

# Probability of a single word given a history
p = m.prob("nari", history=("wa", "ana"))

# Sample a sentence
import random
tokens = m.generate(max_tokens=25, rng=random.Random(0))
print(" ".join(tokens))
```

## 6. Files

- `01_preprocess.py` — extraction, script-split, normalization, train/dev/test writer
- `ngram_lm.py` — `NGramLM` class (unigram / bigram / trigram + modified-KN), pure Python stdlib
- `02_train_eval.py` — training + evaluation driver for both scripts
- `03_use_model.py` — minimal example of loading and using a saved model
- `models/` — pickled models: `ar_{1,2,3}gram.pkl`, `la_{1,2,3}gram.pkl`
- `processed/` — `ar.{train,dev,test}.txt`, `la.{train,dev,test}.txt`
