import ctypes, os, sys, pickle
import numpy as np
from .tok import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(HERE, "kernel.dll" if sys.platform == "win32" else "kernel.so"))
_lib.maxscore.restype = ctypes.c_int
_lib.maxscore.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p] * 2
ptr = lambda a: ctypes.c_void_p(a.ctypes.data)

class Searcher:
    def __init__(self, idx, tsv):
        L = lambda n: np.load(f"{idx}/{n}.npy", mmap_mode="r")
        self.D, self.I, self.off, self.tmax, self.lo = L("docids"), L("impacts"), L("offsets"), L("tmax"), L("lineoffs")
        self.vocab = pickle.load(open(f"{idx}/vocab.pkl", "rb"))
        self.N = len(self.lo); self.tsv = tsv

    def _terms(self, q):
        ids = {self.vocab[t] for t in tokenize(q) if t in self.vocab}
        return sorted(ids, key=lambda t: self.tmax[t])[-64:]           # ascending max impact

    def search(self, q, k=10):
        ts = self._terms(q)
        if not ts: return []
        S = np.array([self.off[t] for t in ts], np.int64); E = np.array([self.off[t + 1] for t in ts], np.int64)
        M = np.array([self.tmax[t] for t in ts], np.float32)
        oid = np.zeros(k, np.int32); osc = np.zeros(k, np.float32)
        n = _lib.maxscore(ptr(self.D), ptr(self.I), ptr(S), ptr(E), ptr(M), len(ts), k, ptr(oid), ptr(osc))
        return [(int(oid[i]), float(osc[i])) for i in range(n)]

    def brute(self, q, k=10):                                     # slow reference, for verification
        acc = np.zeros(self.N, np.float32)
        for t in self._terms(q):
            s, e = self.off[t], self.off[t + 1]; acc[self.D[s:e]] += self.I[s:e]
        top = np.argsort(-acc)[:k]
        return [(int(i), float(acc[i])) for i in top if acc[i] > 0]

    def text(self, doc):
        with open(self.tsv, "rb") as f:
            f.seek(int(self.lo[doc])); return f.readline().decode("utf8", "ignore").rstrip("\r\n").split("\t", 1)[-1]
