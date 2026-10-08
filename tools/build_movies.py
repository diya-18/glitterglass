"""python build_movies.py data/MovieSummaries data/movies.tsv
Merges plot_summaries.txt + movie.metadata.tsv into one line per film:
n <TAB> title <TAB> year <TAB> genres <TAB> plot   (everything after the 1st tab gets indexed)"""
import sys, os, json
src, out = sys.argv[1], sys.argv[2]
clean = lambda s: " ".join(s.split())
meta = {}
for ln in open(os.path.join(src, "movie.metadata.tsv"), encoding="utf8"):
    p = ln.rstrip("\n").split("\t")
    if len(p) < 9: continue
    try: genres = list(json.loads(p[8]).values())
    except ValueError: genres = []
    meta[p[0]] = (clean(p[2]), p[3][:4], clean(", ".join(genres[:4])))
n = 0
with open(out, "w", encoding="utf8", newline="\n") as f:
    for ln in open(os.path.join(src, "plot_summaries.txt"), encoding="utf8"):
        wid, _, plot = ln.rstrip("\n").partition("\t")
        if wid in meta and plot.strip():
            title, year, genres = meta[wid]
            f.write(f"{n}\t{title}\t{year}\t{genres}\t{clean(plot)}\n"); n += 1
print(f"{n:,} films written to {out}")
