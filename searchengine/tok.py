import re
from nltk.stem.porter import PorterStemmer

_TOK = re.compile(r"[a-z0-9]+")
_ps = PorterStemmer()
_cache = {}

def stem(w):
    s = _cache.get(w)
    if s is None:
        s = _cache[w] = _ps.stem(w)
    return s

def tokenize(text):
    return [stem(w) for w in _TOK.findall(text.lower())]
