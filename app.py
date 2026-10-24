"""
The deployed app: ask a question about the indexed channel, get a cited answer.

Runtime dependencies are deliberately small — fastapi, anthropic, voyageai,
numpy. No yt-dlp, no torch, no vector database. The index is a file built by
ingest.py and committed to the repo, so a cold start is just reading two files.
"""

import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv

# Read .env (gitignored) so keys live on disk, not in your shell history.
load_dotenv(Path(__file__).parent / ".env")
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from embeddings import embed_texts

HERE = Path(__file__).parent

ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-haiku-5-5")
TOP_K = int(os.environ.get("TOP_K", "4"))
MAX_PER_IP = int(os.environ.get("MAX_ASKS_PER_IP_PER_DAY", "20"))
MAX_GLOBAL = int(os.environ.get("MAX_ASKS_GLOBAL_PER_DAY", "300"))

SYSTEM_PROMPT = """You answer questions about a YouTube channel using ONLY the \
transcript excerpts provided. Rules:

1. If the excerpts contain the answer, answer clearly and briefly.
2. Cite the timestamp of every claim, like [1:23].
3. If the excerpts do NOT contain the answer, say exactly:
   "I don't know — the indexed videos don't cover that."
   Never use outside knowledge, and never guess.
"""

# ---------------------------------------------------------------------------
# Load the index once at startup.
# ---------------------------------------------------------------------------

try:
    INDEX = json.loads((HERE / "index.json").read_text())
    VECTORS = np.load(HERE / "index.npy")
    CHUNKS = INDEX["chunks"]
except FileNotFoundError:
    raise SystemExit(
        "No index found. Run `python ingest.py` locally and commit "
        "index.json + index.npy before deploying."
    )

if len(CHUNKS) != VECTORS.shape[0]:
    raise SystemExit(
        f"Index is inconsistent: {len(CHUNKS)} chunks but "
        f"{VECTORS.shape[0]} vectors. Re-run ingest.py."
    )

app = FastAPI(title="channel-chat")
_claude = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Crude abuse ceiling.
#
# This is in-memory on purpose: it resets on restart and is per-instance, so it
# is NOT real rate limiting. It exists so a public URL wired to my API key can't
# quietly drain credit overnight. Redis is the honest fix; this is the 10-line
# version that makes shipping tonight safe.
# ---------------------------------------------------------------------------

_hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


def over_limit(ip: str) -> str | None:
    today = date.today().isoformat()
    for stale in [d for d in _hits if d != today]:
        del _hits[stale]
    day = _hits[today]
    if day["__global__"] >= MAX_GLOBAL:
        return "This demo hit its daily cap. Try again tomorrow."
    if day[ip] >= MAX_PER_IP:
        return f"You've used today's {MAX_PER_IP} questions. Try again tomorrow."
    day["__global__"] += 1
    day[ip] += 1
    return None


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


# ---------------------------------------------------------------------------
# Retrieval + generation.
# ---------------------------------------------------------------------------


def retrieve(question: str) -> list[dict]:
    qvec = embed_texts([question], input_type="query")[0]
    scores = VECTORS @ qvec  # normalized vectors, so dot == cosine
    top = np.argsort(scores)[::-1][:TOP_K]
    out = []
    for i in top:
        chunk = dict(CHUNKS[int(i)])
        chunk["score"] = round(float(scores[int(i)]), 4)
        chunk["url"] = (
            f"https://www.youtube.com/watch?v={chunk['video_id']}"
            f"&t={int(chunk['start'])}s"
        )
        out.append(chunk)
    return out


def generate(question: str, retrieved: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['timestamp']}] (from \"{c['title']}\") {c['text']}" for c in retrieved
    )
    response = _claude.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},  # grounded extraction; keep it fast and cheap
        messages=[{
            "role": "user",
            "content": f"Transcript excerpts:\n\n{context}\n\nQuestion: {question}",
        }],
    )
    # Iterate blocks and keep only text. On current models the first block can be
    # a thinking block, so response.content[0].text is not safe.
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "chunks": len(CHUNKS),
        "videos": INDEX.get("video_count"),
        "embed_model": INDEX.get("model"),
        "answer_model": ANSWER_MODEL,
        "built_at": INDEX.get("built_at"),
    }


@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "Ask a question first."}, status_code=400)
    if len(question) > 500:
        return JSONResponse({"error": "Keep it under 500 characters."}, status_code=400)

    blocked = over_limit(client_ip(request))
    if blocked:
        return JSONResponse({"error": blocked}, status_code=429)

    retrieved = retrieve(question)
    return {"answer": generate(question, retrieved), "sources": retrieved}


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.replace("{{VIDEOS}}", str(INDEX.get("video_count", "?")))


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ask the channel</title>
<style>
  :root{--bg:#faf9f7;--fg:#1a1c20;--mut:#5f6570;--line:#e3e1dc;--card:#fff;--acc:#b05a1e}
  @media(prefers-color-scheme:dark){
    :root{--bg:#15171b;--fg:#ecebe8;--mut:#9aa0aa;--line:#2b2f36;--card:#1e2126;--acc:#e09a4f}
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}
  main{max-width:680px;margin:0 auto;padding:48px 20px 80px}
  h1{font-size:1.6rem;margin:0 0 6px;letter-spacing:-.01em}
  p.sub{color:var(--mut);margin:0 0 28px;font-size:.93rem}
  form{display:flex;gap:8px}
  input{flex:1;padding:13px 15px;font-size:1rem;color:var(--fg);background:var(--card);
    border:1px solid var(--line);border-radius:9px}
  input:focus{outline:2px solid var(--acc);outline-offset:1px}
  button{padding:13px 20px;font-size:1rem;font-weight:600;color:#fff;background:var(--acc);
    border:0;border-radius:9px;cursor:pointer}
  button:disabled{opacity:.55;cursor:default}
  #answer{margin-top:28px;white-space:pre-wrap}
  .err{color:#c2410c}
  h2{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--mut);
    margin:32px 0 10px}
  a.src{display:block;padding:11px 14px;margin-bottom:7px;background:var(--card);
    border:1px solid var(--line);border-radius:9px;color:var(--fg);text-decoration:none}
  a.src:hover{border-color:var(--acc)}
  a.src b{color:var(--acc);font-variant-numeric:tabular-nums}
  a.src span{display:block;color:var(--mut);font-size:.85rem;margin-top:3px}
  footer{margin-top:48px;color:var(--mut);font-size:.8rem}
</style>
<main>
  <h1>Ask the channel</h1>
  <p class="sub">Answers come only from the transcripts of {{VIDEOS}} indexed
    videos, with a timestamp for every claim. If it isn't in them, it says so.</p>

  <form id="f">
    <input id="q" placeholder="What does he say about shipping faster?" autocomplete="off" required>
    <button id="b">Ask</button>
  </form>

  <div id="answer"></div>
  <div id="sources"></div>
  <footer>Built with Claude + Voyage embeddings. Retrieval runs over a
    precomputed index — <a href="/healthz" style="color:var(--acc)">/healthz</a>.</footer>
</main>
<script>
const f=document.getElementById('f'),q=document.getElementById('q'),
      b=document.getElementById('b'),ans=document.getElementById('answer'),
      src=document.getElementById('sources');
f.addEventListener('submit',async e=>{
  e.preventDefault();
  b.disabled=true;b.textContent='Thinking…';ans.textContent='';src.innerHTML='';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q.value})});
    const d=await r.json();
    if(!r.ok){ans.innerHTML='<p class="err">'+(d.error||'Something broke.')+'</p>';}
    else{
      ans.textContent=d.answer;
      if(d.sources&&d.sources.length){
        src.innerHTML='<h2>Sources</h2>'+d.sources.map(s=>
          '<a class="src" target="_blank" rel="noopener" href="'+s.url+'">'+
          '<b>['+s.timestamp+']</b> '+s.title+
          '<span>'+s.text.slice(0,130).replace(/</g,'&lt;')+'…</span></a>').join('');
      }
    }
  }catch(err){ans.innerHTML='<p class="err">Network error. Try again.</p>';}
  b.disabled=false;b.textContent='Ask';
});
</script>
"""
