# channel-chat

Ask a question about a YouTube channel; get an answer grounded in the
transcripts, with a clickable timestamp for every claim — or an honest
"I don't know."

Built on three pieces that already existed:
[`youtube-transcript-mcp`](../youtube-transcript-mcp) (transcript extraction),
[`transcript-rag`](../transcript-rag) (the Level-0 RAG walkthrough this
productionizes), and Claude for generation.

```
videos.txt  →  transcripts  →  chunks  →  Voyage embeddings  →  index.{json,npy}
                                                                      │
                             question  →  embed  →  cosine top-K  →  Claude  →  cited answer
```

## Design decisions worth defending

**Embeddings are precomputed, not computed per request.** The corpus only
changes when I re-ingest, so embedding belongs in the build, not the request
path. A question costs one small query embedding plus one Claude call.

**No vector database.** The whole index is a numpy matrix; retrieval is one
dot product against a few thousand rows — microseconds. A vector DB earns its
keep at a scale this isn't at yet. Swapping one in later is a contained change
to `retrieve()`.

**No `sentence-transformers`.** The original `transcript-rag` embedded locally,
which drags in torch (~2GB) and makes deployment painful. Moving embeddings to
an API call made the runtime image small enough to deploy anywhere.

**Ingest deps are separate from runtime deps.** `yt-dlp` and the MCP package
only run on my laptop, so the deployed service never installs them.

**The rate limit is in-memory and I know it's not real.** It resets on restart
and is per-instance. It exists so a public URL wired to my API key can't drain
credit overnight. Redis is the honest fix.

## Run it locally

```bash
cd /Users/one/projects/channel-chat
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-ingest.txt

export ANTHROPIC_API_KEY=sk-ant-...     # console.anthropic.com
export VOYAGE_API_KEY=pa-...            # dashboard.voyageai.com

# Pick the videos (writes videos.txt), then build the index.
python ingest.py --channel https://www.youtube.com/@YOURHANDLE --limit 10

uvicorn app:app --reload
# open http://127.0.0.1:8000
```

## Deploy (Render)

1. `git init && git add . && git commit -m "channel-chat"` — **commit
   `index.json` and `index.npy`**; the app reads them at startup.
2. Push to GitHub.
3. Render → New → Web Service → pick the repo. `render.yaml` supplies the build
   and start commands.
4. Set `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` in the Render dashboard.
5. Visit `/healthz` to confirm the index loaded.

Re-indexing is a re-run of `ingest.py` plus a commit — the deploy picks it up.

## Next (deliberately not done yet)

- Evals with [`llm-eval-harness`](../llm-eval-harness): retrieval hit-rate and
  answer faithfulness on a fixed question set.
- Measure p95 latency and cost per question.
- Any channel on demand (needs async ingestion + per-channel storage).
