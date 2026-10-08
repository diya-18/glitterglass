# glitterglass ✦

A tiny search engine built from scratch in **Python + C**, running on a CPU-only laptop, with a Y2K-style web interface.

**Live demo:** _add your Render link here_

## What it does

- **Passages:** searches 100,000 MS MARCO passages and returns the top 10 exact BM25 matches in about a millisecond.
- **Movies:** describe a scene and it finds the film, searching about 42,000 Wikipedia plot summaries and showing the sentences that matched.
- **Lingo:** a dictionary of Gen Z and Gen Alpha internet slang with original definitions and examples; search by word, phrase or meaning.
- **The sky:** a guestbook drawn as a constellation (plus an alphabetical index). Hover a star for a bell note, or play the whole sky as a melody.
- **Themes:** Y2K pastel and Frutiger Aero, each with light and dark versions.

## How it works

1. `tok.py` lowercases, tokenizes and stems text (Porter).
2. `indexer.py` builds an inverted index in parallel and precomputes a BM25 score for every word/passage pair.
3. `kernel.c` runs MaxScore pruning, skipping passages that can't reach the top 10, so results stay exact.
4. `search.py` memory-maps the index and calls the C code through `ctypes`.
5. `server.py` is a standard-library HTTP server with `/search`, `/movies`, `/slang`, `/people` and `/sign` endpoints.

## Tech stack

Python (standard library, numpy, NLTK) · C · HTML/CSS/JavaScript with no framework · Docker for hosting.

## Run locally (Windows)

```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m ziglang cc -O3 -shared -o searchengine\kernel.dll searchengine\kernel.c
python -m searchengine.indexer data\mini.tsv indexes\mini 4
python -m searchengine.server indexes\mini data\mini.tsv 8080
```

Open http://127.0.0.1:8080. Building the passage and movie data from the raw downloads is done with `build_movies.py` and the indexer; the raw files are not stored in this repo.

## Deploy

The `Dockerfile` compiles the C kernel on Linux and starts the server, which reads the `PORT` variable. On Render, create a **Web Service** from this repo with the **Docker** runtime. Free instances sleep after inactivity, and guestbook signatures made online are lost when the service redeploys. Names in `people.tsv` are permanent.

## Limits

It matches keywords, not meaning. The index is built once and is read-only. Slang changes fast, so the dictionary is a starter set.

## Credits and data

- Passages come from [MS MARCO](https://microsoft.github.io/msmarco/); check its terms before using the data beyond personal or research purposes.
- Movie plots come from the CMU Movie Summary Corpus, drawn from Wikipedia (Creative Commons Attribution-ShareAlike).
- Slang definitions and examples are original.
- Inspired by [RivaanRanawat/search-engine](https://github.com/RivaanRanawat/search-engine).
