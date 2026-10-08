"""python -m searchengine.server indexes/mini data/mini.tsv [port]"""
import sys, os, json, time, threading, re, random, math
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from .search import Searcher
from .tok import tokenize

idx, tsv = sys.argv[1], sys.argv[2]
port = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("PORT", 8080))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAGE = os.path.join(ROOT, "web", "index.html")
SEED = os.path.join(ROOT, "people.tsv")                       # permanent names: name<TAB>message
GB = os.environ.get("GUESTBOOK_FILE", os.path.join(ROOT, "guestbook.jsonl"))
S = Searcher(idx, tsv)
MIDX, MTSV = os.path.join(ROOT, "indexes", "movies"), os.path.join(ROOT, "data", "movies.tsv")
M = Searcher(MIDX, MTSV) if os.path.exists(os.path.join(MIDX, "vocab.pkl")) and os.path.exists(MTSV) else None
LOCK, PEOPLE, HITS = threading.Lock(), [], {}
STICK = ["🔥","💀","✨","👑","🦋","🎧","🌈","💿","⭐","🖤"]

def clean(s, n):
    return "".join(c for c in s if c.isprintable()).strip()[:n]

def load():
    if os.path.exists(SEED):
        for ln in open(SEED, encoding="utf8"):
            p = ln.rstrip("\r\n").split("\t")
            if p[0].strip(): PEOPLE.append({"name": clean(p[0], 40), "msg": clean(p[1], 140) if len(p) > 1 else ""})
    if os.path.exists(GB):
        for ln in open(GB, encoding="utf8"):
            try: PEOPLE.append(json.loads(ln))
            except ValueError: pass
load()

def snippet(plot, q):            # the sentence(s) of the plot that best match the query words
    qset = set(tokenize(q))
    sents = re.split(r"(?<=[.!?])\s+", plot) or [""]
    best = max(range(len(sents)), key=lambda i: len(qset & set(tokenize(sents[i]))))
    s = " ".join(sents[best:best + 2])
    return s if len(s) <= 360 else s[:360].rsplit(" ", 1)[0] + "…"

def norm(s): return " ".join(re.findall(r"[a-z0-9]+", s.lower()))
SLANG, IDF = [], {}
def load_slang():
    path = os.path.join(ROOT, "slang_source.tsv")
    if not os.path.exists(path): return
    df = {}
    for ln in open(path, encoding="utf8"):
        p = ln.rstrip("\r\n").split("\t")
        if len(p) < 4 or p[0].startswith("#"): continue
        p += [""] * (5 - len(p)); term, aliases, meaning, usage, tag = p[:5]
        f = [set(tokenize(term + " " + aliases)), set(tokenize(meaning)), set(tokenize(usage))]
        keys = {norm(x) for x in [term] + aliases.split(",") if norm(x)}
        SLANG.append({"term": term, "meaning": meaning, "usage": usage, "tag": tag, "keys": keys, "f": f})
        for t in set().union(*f): df[t] = df.get(t, 0) + 1
    for t, d in df.items(): IDF[t] = math.log(1 + (len(SLANG) - d + 0.5) / (d + 0.5))
load_slang()

def slang_search(q, k):
    nq, qt = norm(q), set(tokenize(q))
    if not SLANG: return []
    if not nq: return [random.choice(SLANG)]
    scored = []
    for e in SLANG:
        s = 100.0 if nq in e["keys"] else 0.0
        for t in qt: s += IDF.get(t, 0) * (6 * (t in e["f"][0]) + 2 * (t in e["f"][1]) + 0.5 * (t in e["f"][2]))
        if s > 0: scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:k]]

def people_for(q):
    ql = q.strip().lower()
    if len(ql) < 2 or len(ql.split()) > 3: return []
    words = {w for w in ql.replace(",", " ").split() if len(w) >= 2}
    out = []
    with LOCK:
        for p in reversed(PEOPLE):
            n = p["name"].lower()
            if n == ql or (len(ql) >= 3 and ql in n) or words & set(n.split()):
                out.append(p)
                if len(out) == 5: break
    return out

class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def _json(self, obj, code=200): self._send(json.dumps(obj).encode(), "application/json", code)
    def do_GET(self):
        u = urlparse(self.path); qs = parse_qs(u.query)
        if u.path == "/search":
            q = qs.get("q", [""])[0][:200]
            try: k = max(1, min(int(qs.get("k", ["10"])[0]), 50))
            except ValueError: k = 10
            t = time.perf_counter(); hits = S.search(q, k); ms = (time.perf_counter() - t) * 1000
            self._json({"q": q, "ms": round(ms, 2), "total_docs": S.N,
                        "people": people_for(q),
                        "hits": [{"id": d, "score": round(s, 3), "text": S.text(d)} for d, s in hits]})
        elif u.path == "/people":
            q = clean(qs.get("q", [""])[0], 40).lower()
            try: lim = max(1, min(int(qs.get("limit", ["24"])[0]), 150))
            except ValueError: lim = 24
            with LOCK: m = [p for p in reversed(PEOPLE) if not q or q in p["name"].lower()][:20 if q else lim]
            self._json({"matches": m})
        elif u.path == "/caps": self._json({"movies": M is not None, "slang": bool(SLANG)})
        elif u.path == "/slang":
            q = qs.get("q", [""])[0][:100]
            try: k = max(1, min(int(qs.get("k", ["6"])[0]), 12))
            except ValueError: k = 6
            t = time.perf_counter(); hits = slang_search(q, k); ms = (time.perf_counter() - t) * 1000
            self._json({"q": q, "ms": round(ms, 2), "total": len(SLANG),
                        "hits": [{x: e[x] for x in ("term", "meaning", "usage", "tag")} for e in hits]})
        elif u.path == "/movies":
            if M is None: return self._json({"error": "movie index not built"}, 404)
            q = qs.get("q", [""])[0][:300]
            try: k = max(1, min(int(qs.get("k", ["8"])[0]), 20))
            except ValueError: k = 8
            t = time.perf_counter(); hits = M.search(q, k); ms = (time.perf_counter() - t) * 1000
            out = []
            for d, s in hits:
                p = M.text(d).split("\t"); p += [""] * (3 - len(p))
                out.append({"id": d, "score": round(s, 2), "title": p[0], "year": p[1],
                            "genres": [g for g in p[2].split(", ") if g], "plot": snippet("\t".join(p[3:]), q)})
            self._json({"q": q, "ms": round(ms, 2), "total_docs": M.N, "hits": out})
        elif u.path == "/":          # re-read on every request: edit index.html, refresh, done
            self._send(open(PAGE, "rb").read(), "text/html; charset=utf-8")
        else: self.send_error(404)
    def do_POST(self):
        if urlparse(self.path).path != "/sign": return self.send_error(404)
        ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2048: return self._json({"error": "Message too big."}, 413)
        try: d = json.loads(self.rfile.read(n))
        except ValueError: return self._json({"error": "Bad request."}, 400)
        name, msg = clean(str(d.get("name", "")), 40), clean(str(d.get("msg", "")), 140)
        if not name: return self._json({"error": "Please enter a name."}, 400)
        try: hue = int(d.get("hue")) % 360
        except (TypeError, ValueError): hue = None
        sticker = d.get("sticker") if d.get("sticker") in STICK else ""
        try: font = int(d.get("font", 0)); font = font if 0 <= font < 4 else 0
        except (TypeError, ValueError): font = 0
        now = time.time()
        with LOCK:
            recent = [t for t in HITS.get(ip, []) if now - t < 3600]
            if len(recent) >= 5 or len(PEOPLE) >= 5000:
                return self._json({"error": "Too many signatures right now. Try again later."}, 429)
            HITS[ip] = recent + [now]
            e = {"name": name, "msg": msg, "t": int(now), "hue": hue, "sticker": sticker, "font": font}; PEOPLE.append(e)
            try:
                with open(GB, "a", encoding="utf8") as f: f.write(json.dumps(e) + "\n")
            except OSError: pass
        self._json(e)
    def log_message(self, *a): pass

print(f"serving {S.N:,} passages, {len(PEOPLE)} guestbook names, movies: {M.N if M else 'off'} on http://127.0.0.1:{port}")
ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
