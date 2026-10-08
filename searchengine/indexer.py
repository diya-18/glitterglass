"""python -m searchengine.indexer data/mini.tsv indexes/mini [workers]"""
import sys, os, pickle, time
from collections import Counter
from multiprocessing import Pool
import numpy as np
from .tok import tokenize

K1, B = 0.9, 0.4

def process(args):
    start, lines = args
    vocab, ids, docs, tfs, dl = {}, [], [], [], []
    for i, ln in enumerate(lines):
        toks = tokenize(ln.decode("utf8", "ignore").split("\t", 1)[-1])
        dl.append(len(toks))
        for t, c in Counter(toks).items():
            ids.append(vocab.setdefault(t, len(vocab))); docs.append(start + i); tfs.append(c)
    return (list(vocab), np.array(ids, np.int32), np.array(docs, np.int32),
            np.array(tfs, np.float32), np.array(dl, np.int32))

def main(src, out, workers):
    t0 = time.time(); os.makedirs(out, exist_ok=True)
    offs = []
    def gen(size=5000):
        pos, buf, start = 0, [], 0
        with open(src, "rb") as f:
            for ln in f:
                offs.append(pos); pos += len(ln); buf.append(ln)
                if len(buf) == size:
                    yield start, buf; start += size; buf = []
        if buf: yield start, buf
    gv, parts, dls = {}, [], []
    with Pool(workers) as pool:
        for vocab, ids, docs, tfs, dl in pool.imap(process, gen()):
            remap = np.array([gv.setdefault(t, len(gv)) for t in vocab], np.int32)
            parts.append((remap[ids], docs, tfs)); dls.append(dl)
            print(f"\r{len(offs):>9,} passages", end="", flush=True)
    T = np.concatenate([p[0] for p in parts]); D = np.concatenate([p[1] for p in parts])
    F = np.concatenate([p[2] for p in parts]); del parts
    dl = np.concatenate(dls); N = len(dl); avg = dl.mean(); V = len(gv)
    df = np.bincount(T, minlength=V)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    order = np.argsort(T, kind="stable")            # keeps docids ascending inside each term
    T, D, F = T[order], D[order], F[order]
    impact = (idf[T] * F * (K1 + 1) / (F + K1 * (1 - B + B * dl[D] / avg))).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(df)]).astype(np.int64)
    tmax = np.maximum.reduceat(impact, offsets[:-1]).astype(np.float32)
    for name, arr in [("docids", D), ("impacts", impact), ("offsets", offsets),
                      ("tmax", tmax), ("lineoffs", np.array(offs, np.int64))]:
        np.save(f"{out}/{name}.npy", arr)
    pickle.dump(gv, open(f"{out}/vocab.pkl", "wb"))
    print(f"\nN={N:,} terms={V:,} postings={len(D):,}  built in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count())
