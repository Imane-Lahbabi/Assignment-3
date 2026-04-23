"""
N-gram language model with modified Kneser-Ney smoothing.

Efficient version: every quantity that depends only on a context (sum of
counts, number of continuation types per count bucket) is precomputed once
at .fit() time so that .prob(w, hist) is O(1).

References: Chen & Goodman (1998); Jurafsky & Martin, Chapter 3.
"""

from collections import defaultdict, Counter
import math
import pickle

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramLM:
    def __init__(self, n=3, unk_threshold=1):
        assert n in (1, 2, 3)
        self.n = n
        self.unk_threshold = unk_threshold

        self.counts = {k: Counter() for k in range(1, n + 1)}

        # context sums and type-count buckets for orders 2 and 3
        self.ctx_total = {}
        self.ctx_n1 = {}
        self.ctx_n2 = {}
        self.ctx_n3 = {}

        # KN continuation stats
        self.bi_cont_num = Counter()     # (u, w) -> |{u' : c(u', u, w) > 0}|
        self.bi_cont_denom = Counter()   # u     -> sum over w of bi_cont_num[(u, w)]
        self.bicont_n1 = Counter()
        self.bicont_n2 = Counter()
        self.bicont_n3 = Counter()
        self.cont_uni_num = Counter()    # w -> |{u : c(u, w) > 0}|
        self.cont_uni_denom = 0          # total # distinct bigram types

        self.discounts = {}
        self.vocab = None
        self.V = 0
        self._uni_total = 0              # cached total unigram tokens

    # -------------------------------------------------------------- fit -----
    def fit(self, sentences):
        raw_uni = Counter()
        for s in sentences:
            raw_uni.update(s)
        self.vocab = {w for w, c in raw_uni.items() if c > self.unk_threshold}
        self.vocab.update([BOS, EOS, UNK])
        self.V = len(self.vocab)
        in_vocab = self.vocab.__contains__

        pad_left = [BOS] * (self.n - 1) if self.n > 1 else []
        pad_right = [EOS]

        c1 = self.counts[1]
        c2 = self.counts[2] if self.n >= 2 else None
        c3 = self.counts[3] if self.n >= 3 else None

        for s in sentences:
            toks = pad_left + [(w if in_vocab(w) else UNK) for w in s] + pad_right
            for w in toks:
                c1[(w,)] += 1
            if c2 is not None:
                for i in range(len(toks) - 1):
                    c2[(toks[i], toks[i+1])] += 1
            if c3 is not None:
                for i in range(len(toks) - 2):
                    c3[(toks[i], toks[i+1], toks[i+2])] += 1

        self._finalize()

    # --------------------------------------------------------- _finalize ----
    def _finalize(self):
        self._uni_total = sum(self.counts[1].values())

        # modified-KN discounts per order from counts of counts
        for k in range(2, self.n + 1):
            nr = Counter()
            for c in self.counts[k].values():
                if 1 <= c <= 4:
                    nr[c] += 1
            N1, N2, N3, N4 = nr[1], nr[2], nr[3], nr[4]
            if N1 == 0 or N2 == 0:
                self.discounts[k] = (0.75, 0.75, 0.75)
                continue
            Y = N1 / (N1 + 2 * N2)
            D1 = 1 - 2 * Y * (N2 / N1)
            D2 = 2 - 3 * Y * (N3 / N2) if N3 > 0 else 1.0
            D3 = 3 - 4 * Y * (N4 / N3) if N4 > 0 else 1.5
            D1 = max(0.0, min(D1, 0.9))
            D2 = max(D1, min(D2, 1.9))
            D3 = max(D2, min(D3, 2.9))
            self.discounts[k] = (D1, D2, D3)

        # bigram context stats
        if self.n >= 2:
            t = Counter(); a1 = Counter(); a2 = Counter(); a3 = Counter()
            for (u, w), c in self.counts[2].items():
                t[(u,)] += c
                if c == 1:   a1[(u,)] += 1
                elif c == 2: a2[(u,)] += 1
                else:        a3[(u,)] += 1
                self.cont_uni_num[w] += 1
            self.ctx_total[2] = t
            self.ctx_n1[2] = a1; self.ctx_n2[2] = a2; self.ctx_n3[2] = a3
            self.cont_uni_denom = len(self.counts[2])

        # trigram context stats and KN-bigram continuation
        if self.n >= 3:
            t = Counter(); a1 = Counter(); a2 = Counter(); a3 = Counter()
            bi_cont_num = Counter()
            for (u1, u2, w), c in self.counts[3].items():
                t[(u1, u2)] += c
                if c == 1:   a1[(u1, u2)] += 1
                elif c == 2: a2[(u1, u2)] += 1
                else:        a3[(u1, u2)] += 1
                bi_cont_num[(u2, w)] += 1
            self.ctx_total[3] = t
            self.ctx_n1[3] = a1; self.ctx_n2[3] = a2; self.ctx_n3[3] = a3
            self.bi_cont_num = bi_cont_num

            bi_cont_denom = Counter()
            b1 = Counter(); b2 = Counter(); b3 = Counter()
            for (u, w), v in bi_cont_num.items():
                bi_cont_denom[u] += v
                if v == 1:   b1[u] += 1
                elif v == 2: b2[u] += 1
                else:        b3[u] += 1
            self.bi_cont_denom = bi_cont_denom
            self.bicont_n1 = b1; self.bicont_n2 = b2; self.bicont_n3 = b3

    # ------------------------------------------------------- probabilities --
    @staticmethod
    def _disc(D, c):
        if c == 1: return D[0]
        if c == 2: return D[1]
        if c >= 3: return D[2]
        return 0.0

    def _p_cont_unigram(self, w):
        if self.n < 2 or self.cont_uni_denom == 0:
            return (self.counts[1].get((w,), 0) + 1) / (self._uni_total + self.V)
        num = self.cont_uni_num.get(w, 0)
        if num == 0:
            return 1.0 / self.V
        return num / self.cont_uni_denom

    def _p_kn_bigram(self, w, u):
        if self.n >= 3:
            denom = self.bi_cont_denom.get(u, 0)
            if denom == 0:
                return self._p_cont_unigram(w)
            num = self.bi_cont_num.get((u, w), 0)
            D = self.discounts[2]
            first = max(num - self._disc(D, num), 0) / denom
            n1 = self.bicont_n1.get(u, 0)
            n2 = self.bicont_n2.get(u, 0)
            n3 = self.bicont_n3.get(u, 0)
            lam = (D[0]*n1 + D[1]*n2 + D[2]*n3) / denom
            return first + lam * self._p_cont_unigram(w)

        denom = self.ctx_total[2].get((u,), 0)
        if denom == 0:
            return self._p_cont_unigram(w)
        c = self.counts[2].get((u, w), 0)
        D = self.discounts[2]
        first = max(c - self._disc(D, c), 0) / denom
        n1 = self.ctx_n1[2].get((u,), 0)
        n2 = self.ctx_n2[2].get((u,), 0)
        n3 = self.ctx_n3[2].get((u,), 0)
        lam = (D[0]*n1 + D[1]*n2 + D[2]*n3) / denom
        return first + lam * self._p_cont_unigram(w)

    def _p_kn_trigram(self, w, u1, u2):
        denom = self.ctx_total[3].get((u1, u2), 0)
        if denom == 0:
            return self._p_kn_bigram(w, u2)
        c = self.counts[3].get((u1, u2, w), 0)
        D = self.discounts[3]
        first = max(c - self._disc(D, c), 0) / denom
        n1 = self.ctx_n1[3].get((u1, u2), 0)
        n2 = self.ctx_n2[3].get((u1, u2), 0)
        n3 = self.ctx_n3[3].get((u1, u2), 0)
        lam = (D[0]*n1 + D[1]*n2 + D[2]*n3) / denom
        return first + lam * self._p_kn_bigram(w, u2)

    def prob(self, word, history):
        in_vocab = self.vocab.__contains__
        w = word if in_vocab(word) else UNK
        if self.n == 1:
            return (self.counts[1].get((w,), 0) + 1) / (self._uni_total + self.V)
        hist = tuple((h if in_vocab(h) else UNK) for h in history)
        if self.n == 2:
            return self._p_kn_bigram(w, hist[-1])
        return self._p_kn_trigram(w, hist[-2], hist[-1])

    # ------------------------------------------------------------- evaluate
    def perplexity(self, sentences):
        log_prob_sum = 0.0
        n_tokens = 0
        pad = (BOS,) * (self.n - 1) if self.n > 1 else ()
        in_vocab = self.vocab.__contains__
        uni_denom = self._uni_total + self.V if self.n == 1 else None

        for s in sentences:
            toks = list(pad) + [(w if in_vocab(w) else UNK) for w in s] + [EOS]
            for i in range(len(pad), len(toks)):
                w = toks[i]
                if self.n == 1:
                    p = (self.counts[1].get((w,), 0) + 1) / uni_denom
                elif self.n == 2:
                    p = self._p_kn_bigram(w, toks[i-1])
                else:
                    p = self._p_kn_trigram(w, toks[i-2], toks[i-1])
                if p <= 0.0:
                    p = 1e-12
                log_prob_sum += math.log(p)
                n_tokens += 1
        avg_nll = -log_prob_sum / max(1, n_tokens)
        return math.exp(avg_nll), n_tokens

    # ------------------------------------------------------------- generate
    def generate(self, max_tokens=30, rng=None, temperature=1.0):
        import random
        rng = rng or random.Random()

        if not hasattr(self, '_gen_index'):
            if self.n == 3:
                idx = defaultdict(list)
                for (u1, u2, w), c in self.counts[3].items():
                    idx[(u1, u2)].append((w, c))
                self._gen_index = dict(idx)
            elif self.n == 2:
                idx = defaultdict(list)
                for (u, w), c in self.counts[2].items():
                    idx[(u,)].append((w, c))
                self._gen_index = dict(idx)
            else:
                self._gen_index = None

        hist = [BOS, BOS] if self.n >= 3 else ([BOS] if self.n == 2 else [])
        out = []
        for _ in range(max_tokens):
            key = tuple(hist[-(self.n - 1):]) if self.n > 1 else ()
            candidates = self._gen_index.get(key) if self._gen_index is not None else None
            if not candidates and self.n == 3:
                # back off to bigram continuations of the last token
                candidates = [(w, c) for (u, w), c in self.counts[2].items()
                              if u == hist[-1]]
            if not candidates:
                break
            words = [w for w, _ in candidates]
            if self.n == 3:
                probs = [self._p_kn_trigram(w, hist[-2], hist[-1]) for w in words]
            elif self.n == 2:
                probs = [self._p_kn_bigram(w, hist[-1]) for w in words]
            else:
                probs = [self.prob(w, ()) for w in words]
            if temperature != 1.0:
                probs = [p ** (1.0 / temperature) for p in probs]
            s = sum(probs)
            if s <= 0:
                break
            probs = [p / s for p in probs]
            nxt = rng.choices(words, weights=probs, k=1)[0]
            if nxt == EOS:
                break
            if nxt != UNK:
                out.append(nxt)
            hist.append(nxt)
        return out

    # -------------------------------------------------------------- persist
    def save(self, path):
        state = {
            'n': self.n,
            'unk_threshold': self.unk_threshold,
            'counts': {k: dict(v) for k, v in self.counts.items()},
            'ctx_total': self.ctx_total,
            'ctx_n1': self.ctx_n1, 'ctx_n2': self.ctx_n2, 'ctx_n3': self.ctx_n3,
            'bi_cont_num': dict(self.bi_cont_num),
            'bi_cont_denom': dict(self.bi_cont_denom),
            'bicont_n1': dict(self.bicont_n1),
            'bicont_n2': dict(self.bicont_n2),
            'bicont_n3': dict(self.bicont_n3),
            'cont_uni_num': dict(self.cont_uni_num),
            'cont_uni_denom': self.cont_uni_denom,
            'discounts': self.discounts,
            'vocab': self.vocab,
            'V': self.V,
            '_uni_total': self._uni_total,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        m = cls(n=state['n'], unk_threshold=state['unk_threshold'])
        m.counts = {k: Counter(v) for k, v in state['counts'].items()}
        m.ctx_total = state['ctx_total']
        m.ctx_n1 = state['ctx_n1']; m.ctx_n2 = state['ctx_n2']; m.ctx_n3 = state['ctx_n3']
        m.bi_cont_num = Counter(state['bi_cont_num'])
        m.bi_cont_denom = Counter(state['bi_cont_denom'])
        m.bicont_n1 = Counter(state['bicont_n1'])
        m.bicont_n2 = Counter(state['bicont_n2'])
        m.bicont_n3 = Counter(state['bicont_n3'])
        m.cont_uni_num = Counter(state['cont_uni_num'])
        m.cont_uni_denom = state['cont_uni_denom']
        m.discounts = state['discounts']
        m.vocab = state['vocab']
        m.V = state['V']
        m._uni_total = state['_uni_total']
        return m


if __name__ == "__main__":
    toy = [
        "i love you".split(),
        "i love cats".split(),
        "you love me".split(),
        "cats love fish".split(),
        "i saw you".split(),
    ]
    for n in (1, 2, 3):
        m = NGramLM(n=n, unk_threshold=0)
        m.fit(toy)
        ppl, nt = m.perplexity([["i", "love", "you"]])
        print(f"n={n}  ppl on 'i love you' = {ppl:.3f}  tokens={nt}")
