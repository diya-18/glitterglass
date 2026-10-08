"""python bench.py indexes/mini data/mini.tsv data  -> latency, exactness vs brute force, MRR@10"""
import sys, time
from collections import defaultdict
from searchengine.search import Searcher
idx, tsv, data = sys.argv[1:4]
S = Searcher(idx, tsv)
qs = [l.rstrip("\n").split("\t") for l in open(f"{data}/queries.dev.small.tsv", encoding="utf8")][:1000]
rel = defaultdict(set)
for l in open(f"{data}/qrels.dev.small.tsv"):
    qid, _, pid, _ = l.split()
    if int(pid) < S.N: rel[qid].add(int(pid))
lat, rr, bad, ev = [], [], 0, 0
for qid, q in qs:
    t = time.perf_counter(); hits = S.search(q, 10); lat.append((time.perf_counter() - t) * 1000)
    if len(lat) <= 200:
        ref = S.brute(q, 10)
        if [round(s, 2) for _, s in hits] != [round(s, 2) for _, s in ref]: bad += 1
    if qid in rel:
        ev += 1; rr.append(next((1 / (i + 1) for i, (d, _) in enumerate(hits) if d in rel[qid]), 0))
lat.sort()
print(f"queries={len(qs)}  p50={lat[len(lat)//2]:.2f}ms  p99={lat[int(len(lat)*.99)]:.2f}ms")
print(f"exactness: {200-bad}/200 match brute force")
print(f"MRR@10={sum(rr)/max(ev,1):.4f} on {ev} queries whose answer is in the subset")
