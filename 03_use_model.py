"""
Example: load a trained model and use it to score / generate.
"""
from ngram_lm import NGramLM

m = NGramLM.load("/home/claude/models/la_3gram.pkl")

print("Vocabulary size:", m.V)
print("# trigram types:", len(m.counts[3]))

# Score a held-out sentence
sent = "had chi ma bghach y3awed".split()
ppl, nt = m.perplexity([sent])
print(f"Perplexity of '{' '.join(sent)}': {ppl:.2f} over {nt} tokens")

# Generate
import random
rng = random.Random(0)
for i in range(3):
    toks = m.generate(max_tokens=20, rng=rng)
    print(f"[sample {i+1}]", " ".join(toks))
