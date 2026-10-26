#!/usr/bin/env python3
"""
Daily arXiv cosmology digest (Cloud / GitHub Actions + Gemini API edition).

Fetches the latest astro-ph.CO announcement (arxiv.org/list/.../new), has Gemini read each
new paper (full text when available, not just the abstract), and writes a
self-contained HTML page with a short summary, the main conclusions and a
relevance score against research_profile.md.

Usage:
    GEMINI_API_KEY=... python3 arxiv_digest.py            # build today's digest (skips if already done)
    GEMINI_API_KEY=... python3 arxiv_digest.py --force    # rebuild even if it already exists
    GEMINI_API_KEY=... python3 arxiv_digest.py --limit 3  # only summarise 3 papers (for testing)
    python3 arxiv_digest.py --render-only                 # re-render HTML from cached summaries

Only uses the Python standard library. Calls the Gemini REST API directly via urllib.request.
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser

# ------------------------------------------------------------------ config ---
HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = ["astro-ph.CO"]          # add e.g. "gr-qc" if you like
# Ranked from most capable to fastest fallback.
# Note: Free-tier AI Studio keys have 0 RPM quota for gemini-4-argon/pro unless billing is linked;
# if a model returns 429 (quota=0), it is automatically skipped for the rest of the run.
MODEL_PRIORITY = [
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]
MODEL = os.environ.get("GEMINI_MODEL", MODEL_PRIORITY[0])
_DISABLED_MODELS = set()

BATCH_SCHEMA = None
MAX_WORKERS = 3                       # parallel Gemini calls (respects free-tier RPM limits)
SUMMARISE_TYPES = ("new", "cross")    # replacements are only listed
HEAD_CHARS = 30000                    # text sent to Gemini: start of paper ...
CONCL_CHARS = 15000                   # ... plus the conclusions section
PROFILE_FILE = os.path.join(HERE, "research_profile.md")
DIGEST_DIR = os.path.join(HERE, "digests")
CACHE_DIR = os.path.join(HERE, "cache")
USER_AGENT = "arxiv-digest/1.0 (personal daily reader; mailto:j.b.bendtsen@astro.uio.no)"

SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "conclusions": {"type": "array", "items": {"type": "string"}},
        "relevance": {"type": "integer", "minimum": 0, "maximum": 3},
        "relevance_reason": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "conclusions", "relevance", "relevance_reason", "tags"],
}

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "conclusions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "relevance": {"type": "INTEGER"},
        "relevance_reason": {"type": "STRING"},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary", "conclusions", "relevance", "relevance_reason", "tags"],
}

SYSTEM_PROMPT = """You are a friendly, clear-spoken cosmology mentor writing a daily arXiv digest for a first-year PhD student in cosmology.
The student has a solid physics/cosmology foundation (ΛCDM, expansion history, perturbation theory, N-body simulations) and works on beyond-ΛCDM simulations/emulators and inhomogeneous cosmology, but is NOT an expert in every specialized subfield of cosmology or astrophysics.

For the paper you are given, return:
- summary: 3-5 sentences written so a first-year cosmology PhD student can immediately grasp the big picture, even if the paper is outside their specialty. Start by briefly explaining the physical context or problem in accessible terms (defining any niche jargon, acronyms, or specialized phenomena in plain language), then explain what the authors actually did (method, data, simulations, or theoretical model) and why it matters. Do NOT paraphrase the abstract sentence by sentence; synthesize from the full text.
- conclusions: 2-5 short, clear bullet points stating the main takeaways and quantitative/qualitative results (include key numbers, constraints, or significances where given, and briefly state what those numbers mean physically).
- relevance: integer 0-3 for how relevant the paper is to the student's research profile below. 3 = directly on their topic/methods (they should read it today), 2 = closely related / useful method or result, 1 = loosely related background, 0 = unrelated. Be strict: most papers should be 0 or 1.
- relevance_reason: one clear sentence explaining the score with reference to the profile (empty string if relevance is 0).
- tags: 2-5 short topic tags (e.g. "weak lensing", "f(R)", "emulator", "BAO").
Use LaTeX in $...$ for maths if needed.

RESEARCH PROFILE OF THE STUDENT:
"""

# ----------------------------------------------------------------- helpers ---


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def fetch(url, timeout=60, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            log(f"  retry {url}: {e}")
            time.sleep(5 * (attempt + 1))


# --------------------------------------------------------------- arXiv feed ---


def _clean(s):
    """Strip tags/entities and collapse whitespace."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


def get_feed():
    """Return (announcement date 'YYYY-MM-DD', list of paper dicts).

    Reads arxiv.org/list/<cat>/new, which updates at the announcement
    (20:00 US Eastern, i.e. 01-02 Oslo time). The RSS feed only refreshes at
    midnight Eastern (05-06 Oslo time).
    """
    papers, seen, date = [], set(), None
    for cat in CATEGORIES:
        page = fetch(f"https://arxiv.org/list/{cat}/new?skip=0&show=2000").decode(
            "utf-8", "replace")
        m = re.search(r"Showing new listings for \w+, (\d+ \w+ \d{4})", page)
        if m:
            d = dt.datetime.strptime(m.group(1), "%d %B %Y").date().isoformat()
            date = max(date, d) if date else d
        # split the page into its sections (New / Cross / Replacement submissions)
        parts = re.split(r"<h3>\s*(New|Cross|Replacement) submissions", page)
        for kind, sec in zip(parts[1::2], parts[2::2]):
            atype = {"New": "new", "Cross": "cross", "Replacement": "replace"}[kind]
            for dt_html, dd in re.findall(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", sec, re.S):
                aid = re.search(r'href ?="/abs/([^"]+)"', dt_html)
                if not aid or aid.group(1) in seen:
                    continue
                aid = aid.group(1)
                seen.add(aid)
                title = re.search(r"list-title[^>]*>(.*?)</div>", dd, re.S)
                authors = re.search(r"list-authors'>(.*?)</div>", dd, re.S)
                subj = re.search(r"list-subjects'>(.*?)</div>", dd, re.S)
                abstract = re.search(r"<p class='mathjax'>(.*?)</p>", dd, re.S)
                papers.append({
                    "id": aid,
                    "title": _clean(title.group(1)).replace("Title:", "", 1).strip()
                    if title else aid,
                    "link": f"https://arxiv.org/abs/{aid}",
                    "authors": _clean(authors.group(1)).replace(" ,", ",") if authors else "",
                    "categories": re.findall(r"\(([a-z\-]+(?:\.[A-Za-z\-]+)?)\)",
                                             subj.group(1)) if subj else [],
                    "type": atype,
                    "abstract": _clean(abstract.group(1)) if abstract else "",
                })
    return date, papers


# ------------------------------------------------------------ full text ------


class _TextExtractor(HTMLParser):
    """Turn arXiv's LaTeXML HTML into plain text; maths -> $alttext$."""
    SKIP = {"script", "style", "nav", "header", "footer", "button"}
    BLOCK = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "li", "br",
             "tr", "figcaption", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.math = [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "math":
            if not self.math and not self.skip:
                alt = dict(attrs).get("alttext", "")
                self.out.append(f" ${alt}$ ")
            self.math += 1
        elif tag in self.BLOCK and not self.math:
            self.out.append("\n")
            if tag in ("h1", "h2", "h3", "h4"):
                self.out.append("\n## ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "math" and self.math:
            self.math -= 1

    def handle_data(self, data):
        if not self.skip and not self.math:
            self.out.append(data)

    def text(self):
        t = "".join(self.out)
        t = re.sub(r"[ \t]+", " ", t)
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def full_text(aid):
    """Full text from arXiv HTML, falling back to PDF via pdftotext. '' on failure."""
    try:
        raw = fetch(f"https://arxiv.org/html/{aid}", timeout=60, retries=2)
        p = _TextExtractor()
        p.feed(raw.decode("utf-8", "replace"))
        t = p.text()
        if len(t) > 3000:
            return t
    except Exception:  # noqa: BLE001
        pass
    try:
        pdf = fetch(f"https://arxiv.org/pdf/{aid}", timeout=90, retries=2)
        r = subprocess.run(["pdftotext", "-layout", "-", "-"], input=pdf,
                           capture_output=True, timeout=120)
        return r.stdout.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def trim_text(t):
    """Keep the start of the paper plus its conclusions; drop the references."""
    m = list(re.finditer(r"\n(## )?\s*(References|Bibliography|REFERENCES)\s*\n", t))
    if m and m[-1].start() > len(t) * 0.4:
        t = t[:m[-1].start()]
    if len(t) <= HEAD_CHARS + CONCL_CHARS:
        return t
    head = t[:HEAD_CHARS]
    concl = list(re.finditer(
        r"\n(## )?\s*(\d+\.?\s*|[IVX]+\.?\s*)?(Conclusions?|Summary|Discussion|"
        r"Summary and [Cc]onclusions?|Discussion and [Cc]onclusions?|"
        r"Conclusions? and [Dd]iscussion|Concluding [Rr]emarks)\b", t))
    concl = [c for c in concl if c.start() > HEAD_CHARS]
    tail = t[concl[-1].start():][:CONCL_CHARS] if concl else t[-CONCL_CHARS:]
    return head + "\n\n[... middle of paper omitted ...]\n\n" + tail


# --------------------------------------------------------------- Gemini -------


def summarise_agy(paper, agy_bin, agy_model, profile):
    cache = os.path.join(CACHE_DIR, paper["id"] + ".json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    text = trim_text(full_text(paper["id"]))
    source = "full text" if len(text) > 3000 else "abstract only"
    full_prompt = (
        SYSTEM_PROMPT + profile + "\n\n---\nPAPER TO SUMMARISE:\n"
        f"Title: {paper['title']}\nAuthors: {paper['authors']}\n"
        f"arXiv: {paper['id']}  Categories: {', '.join(paper['categories'])}\n\n"
        f"Abstract: {paper['abstract']}\n\n"
        + (f"FULL TEXT (may be truncated):\n{text}" if source == "full text" else "")
    )
    cmd = [
        agy_bin, "--model", agy_model, "--output-format", "json",
        "--json-schema", json.dumps(SCHEMA_JSON), "--disable-slash-commands",
        f"-p={full_prompt}"
    ]
    last_err = ""
    for attempt in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        try:
            out = json.loads(r.stdout)
            res = out.get("structured_output")
            if res is None:
                res = json.loads(re.search(r"\{.*\}", out.get("response", ""), re.S).group(0))
            res["relevance"] = max(0, min(3, int(res.get("relevance", 0))))
            res["source"] = source
            res["model"] = agy_model
            res["cost_usd"] = 0.0
            with open(cache, "w") as f:
                json.dump(res, f)
            return res
        except Exception as e:  # noqa: BLE001
            last_err = f"{e}; stderr: {r.stderr[-300:]} stdout: {r.stdout[-300:]}"
            time.sleep(5 * (attempt + 1))
    log(f"  FAILED {paper['id']}: {last_err}")
    return {"summary": "(Antigravity summary failed - see abstract.)", "conclusions": [],
            "relevance": 0, "relevance_reason": "", "tags": [], "source": "error",
            "model": agy_model, "cost_usd": 0}


def summarise(paper, api_key, profile):
    cache = os.path.join(CACHE_DIR, paper["id"] + ".json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    text = trim_text(full_text(paper["id"]))
    source = "full text" if len(text) > 3000 else "abstract only"
    prompt = (f"Title: {paper['title']}\nAuthors: {paper['authors']}\n"
              f"arXiv: {paper['id']}  Categories: {', '.join(paper['categories'])}\n\n"
              f"Abstract: {paper['abstract']}\n\n"
              + (f"FULL TEXT (may be truncated):\n{text}" if source == "full text" else ""))

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT + profile}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "temperature": 0.2,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    candidates = [MODEL] + [m for m in MODEL_PRIORITY if m != MODEL]
    last_err = ""
    for pass_num in range(3):
        models_to_try = [m for m in candidates if m not in _DISABLED_MODELS] or candidates
        for cur_model in models_to_try:
            if cur_model in _DISABLED_MODELS:
                continue
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}"
                   f":generateContent?key={api_key}")
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                parts = out["candidates"][0]["content"]["parts"]
                raw_json = next(p["text"] for p in reversed(parts) if "text" in p)
                res = json.loads(raw_json)
                res["relevance"] = max(0, min(3, int(res.get("relevance", 0))))
                res["source"] = source
                res["model"] = cur_model
                res["cost_usd"] = 0.0
                with open(cache, "w") as f:
                    json.dump(res, f)
                return res
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", "replace")[:400]
                last_err = f"HTTP {e.code} ({cur_model}): {err_body}"
                if e.code in (404, 400):
                    _DISABLED_MODELS.add(cur_model)
                elif e.code == 429:
                    if "limit: 0" in err_body:
                        _DISABLED_MODELS.add(cur_model)
                    else:
                        time.sleep(8 * (pass_num + 1))
                else:
                    time.sleep(4 * (pass_num + 1))
            except Exception as e:  # noqa: BLE001
                last_err = f"{cur_model}: {e}"
                time.sleep(4 * (pass_num + 1))
    log(f"  FAILED {paper['id']}: {last_err}")
    return {"summary": "(Gemini summary failed - see abstract.)", "conclusions": [],
            "relevance": 0, "relevance_reason": "", "tags": [], "source": "error",
            "cost_usd": 0}


# --------------------------------------------------------------- HTML ---------

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--text:#1d1d1f;--muted:#6b6b70;--border:#e3e3e0;
--accent:#2f5d8a;--hi3:#c0392b;--hi2:#d68910;--hi1:#7f8c8d;--hibg:#fff6e5;--chip:#eef2f6}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1e1f22;--text:#e8e8ea;
--muted:#9a9aa2;--border:#2e2f33;--accent:#7fb2e5;--hi3:#ff7b6b;--hi2:#f5b041;
--hi1:#95a5a6;--hibg:#2a2417;--chip:#2a2d33}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:1.7rem;margin:0 0 4px}
.sub{color:var(--muted);margin-bottom:20px}
.sub a{color:var(--accent)}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0;display:flex;
gap:8px;flex-wrap:wrap;align-items:center;border-bottom:1px solid var(--border);margin-bottom:18px}
.bar button{border:1px solid var(--border);background:var(--card);color:var(--text);
padding:6px 12px;border-radius:999px;cursor:pointer;font-size:.9rem}
.bar button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.bar input{flex:1;min-width:160px;padding:7px 12px;border-radius:999px;
border:1px solid var(--border);background:var(--card);color:var(--text);font-size:.9rem}
h2{font-size:1.15rem;margin:28px 0 10px;color:var(--muted);text-transform:uppercase;
letter-spacing:.04em}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
padding:16px 18px;margin:0 0 14px;border-left:5px solid var(--border)}
.card.r3{border-left-color:var(--hi3);background:linear-gradient(90deg,var(--hibg),var(--card) 40%)}
.card.r2{border-left-color:var(--hi2)}
.card h3{margin:0 0 4px;font-size:1.08rem;line-height:1.35}
.card h3 a{color:var(--text);text-decoration:none}
.card h3 a:hover{color:var(--accent);text-decoration:underline}
.meta{color:var(--muted);font-size:.85rem;margin-bottom:8px}
.meta a{color:var(--accent)}
.badge{display:inline-block;font-size:.75rem;font-weight:600;padding:2px 8px;
border-radius:999px;margin-right:6px;vertical-align:middle}
.b3{background:var(--hi3);color:#fff}.b2{background:var(--hi2);color:#111}
.b1{background:var(--chip);color:var(--muted)}
.why{font-size:.9rem;font-style:italic;color:var(--hi3);margin:4px 0 8px}
.r2 .why{color:var(--hi2)}
.card ul{margin:6px 0 4px;padding-left:20px}
.card li{margin:2px 0}
.tags span{display:inline-block;background:var(--chip);color:var(--muted);font-size:.75rem;
padding:2px 8px;border-radius:999px;margin:6px 4px 0 0}
details{margin-top:8px;font-size:.9rem;color:var(--muted)}
summary{cursor:pointer}
.repl li{margin:3px 0}.repl a{color:var(--accent)}
.empty{color:var(--muted);font-style:italic}
footer{color:var(--muted);font-size:.8rem;margin-top:40px}
"""

JS = """
const state={f:'all',q:''};
function apply(){document.querySelectorAll('.card').forEach(c=>{
 const okF=state.f==='all'||(state.f==='rel'&&+c.dataset.r>=2)||c.dataset.t===state.f;
 const okQ=!state.q||c.textContent.toLowerCase().includes(state.q);
 c.style.display=okF&&okQ?'':'none';});}
document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.bar button').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');state.f=b.dataset.f;apply();});
document.getElementById('q').oninput=e=>{state.q=e.target.value.toLowerCase();apply();};
"""

MATHJAX = ('<script>window.MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]]}};</script>'
           '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js">'
           '</script>')


def esc(s):
    return html.escape(s or "", quote=True)


def card(p):
    s = p.get("ai", {})
    r = int(s.get("relevance", 0))
    label = {3: "Highly relevant", 2: "Relevant", 1: "Related"}.get(r)
    badges = f'<span class="badge b{r}">{label}</span>' if label else ""
    if p["type"] != "new":
        badges += '<span class="badge b1">cross-list</span>'
    if s.get("source") == "abstract only":
        badges += '<span class="badge b1">from abstract</span>'
    why = (f'<div class="why">Why: {esc(s.get("relevance_reason"))}</div>'
           if r >= 2 and s.get("relevance_reason") else "")
    concl = "".join(f"<li>{esc(c)}</li>" for c in s.get("conclusions", []))
    tags = "".join(f"<span>{esc(t)}</span>" for t in s.get("tags", []))
    aid = esc(p["id"])
    return f"""<article class="card r{r}" data-r="{r}" data-t="{esc(p['type'])}">
<h3><a href="{esc(p['link'])}" target="_blank" rel="noopener">{esc(p['title'])}</a></h3>
<div class="meta">{badges}{esc(p['authors'])}<br>
<a href="https://arxiv.org/abs/{aid}" target="_blank" rel="noopener">arXiv:{aid}</a> ·
<a href="https://arxiv.org/pdf/{aid}" target="_blank" rel="noopener">PDF</a> ·
<a href="https://arxiv.org/html/{aid}" target="_blank" rel="noopener">HTML</a> ·
{esc(', '.join(p['categories']))}</div>
{why}<p>{esc(s.get('summary'))}</p>
{'<b>Main conclusions</b><ul>' + concl + '</ul>' if concl else ''}
<div class="tags">{tags}</div>
<details><summary>Abstract</summary><p>{esc(p['abstract'])}</p></details>
</article>"""


def render(date, papers, cost):
    summ = [p for p in papers if p["type"] in SUMMARISE_TYPES]
    repl = [p for p in papers if p["type"] not in SUMMARISE_TYPES]
    rel = sorted([p for p in summ if p["ai"].get("relevance", 0) >= 2],
                 key=lambda p: -p["ai"]["relevance"])
    rest = [p for p in summ if p not in rel]
    nice = dt.date.fromisoformat(date).strftime("%A %d %B %Y")
    body = [f"<h1>arXiv cosmology digest</h1>",
            f'<div class="sub">{esc(", ".join(CATEGORIES))} · announced {nice} · '
            f"{sum(p['type']=='new' for p in summ)} new, "
            f"{sum(p['type']!='new' for p in summ)} cross-lists, "
            f"<b>{len(rel)} relevant to you</b> · "
            f'<a href="archive.html">all digests</a></div>',
            '<div class="bar"><button class="on" data-f="all">All</button>'
            '<button data-f="rel">Relevant</button><button data-f="new">New</button>'
            '<button data-f="cross">Cross-lists</button>'
            '<input id="q" placeholder="Search titles, authors, summaries…"></div>',
            "<h2>Relevant to your research</h2>"]
    body += [card(p) for p in rel] or ['<p class="empty">Nothing flagged today.</p>']
    body += ["<h2>All other papers</h2>"] + [card(p) for p in rest]
    if repl:
        body.append(f"<h2>Replaced / updated ({len(repl)})</h2><ul class='repl'>")
        body += [f'<li><a href="{esc(p["link"])}" target="_blank" rel="noopener">'
                 f'{esc(p["title"])}</a> <span class="meta">({esc(p["id"])})</span></li>'
                 for p in repl]
        body.append("</ul>")
    models_used = sorted({p.get("ai", {}).get("model", MODEL) for p in summ if p.get("ai", {}).get("model")}) or [MODEL]
    body.append(f"<footer>Summaries written by Gemini ({', '.join(models_used)}) from the full text "
                f"where available; check the paper before citing anything. "
                f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}.</footer>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>arXiv digest {date}</title><style>{CSS}</style>{MATHJAX}</head>"
            f'<body><div class="wrap">{"".join(body)}</div><script>{JS}</script>'
            f"</body></html>")


def render_index():
    files = sorted(glob.glob(os.path.join(DIGEST_DIR, "20??-??-??.json")), reverse=True)
    rows = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        n_rel = sum(p.get("ai", {}).get("relevance", 0) >= 2 for p in d["papers"])
        n = sum(p["type"] in SUMMARISE_TYPES for p in d["papers"])
        rows.append(f'<li><a href="{d["date"]}.html">{d["date"]}</a> - {n} papers, '
                    f"<b>{n_rel} relevant</b></li>")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>arXiv digests</title><style>{CSS}</style></head><body>"
            f'<div class="wrap"><h1>arXiv cosmology digests</h1>'
            f'<p><a href="index.html">← Latest digest</a></p>'
            f'<ul class="repl">{"".join(rows)}</ul></div></body></html>')
    with open(os.path.join(DIGEST_DIR, "archive.html"), "w") as f:
        f.write(page)


def write_outputs(date, papers):
    cost = sum(p.get("ai", {}).get("cost_usd", 0) for p in papers)
    with open(os.path.join(DIGEST_DIR, f"{date}.json"), "w") as f:
        json.dump({"date": date, "papers": papers}, f, indent=1)
    page = render(date, papers, cost)
    # Write both index.html (so GitHub Pages root opens today's digest directly) and latest.html
    for name in (f"{date}.html", "latest.html", "index.html"):
        with open(os.path.join(DIGEST_DIR, name), "w") as f:
            f.write(page)
    render_index()
    return cost


# ----------------------------------------------------------------- main -------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--agy", action="store_true",
                    help="Use local Antigravity CLI (agy) with Google AI Pro instead of API key")
    ap.add_argument("--agy-model", default="gemini-4-argon-low",
                    help="Antigravity model when --agy is set (default: gemini-4-argon-low)")
    args = ap.parse_args()
    for d in (DIGEST_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)

    if args.render_only:
        for f in sorted(glob.glob(os.path.join(DIGEST_DIR, "20??-??-??.json"))):
            with open(f) as fh:
                d = json.load(fh)
            write_outputs(d["date"], d["papers"])
        log("Re-rendered all digests.")
        return

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not args.agy and not api_key:
        sys.exit("Error: GEMINI_API_KEY environment variable is not set (or pass --agy).")

    date, papers = get_feed()
    if not date or not papers:
        log("No announcement in the feed (weekend/holiday). Nothing to do.")
        return
    out_json = os.path.join(DIGEST_DIR, f"{date}.json")
    if os.path.exists(out_json) and not args.force:
        log(f"Digest for {date} already exists (arXiv has not announced anything new).")
        return

    if args.force:
        for f in glob.glob(os.path.join(CACHE_DIR, "*.json")):
            os.remove(f)

    with open(PROFILE_FILE) as f:
        profile = f.read()
    todo = [p for p in papers if p["type"] in SUMMARISE_TYPES]
    if args.limit:
        todo = todo[:args.limit]
        papers = todo + [p for p in papers if p["type"] not in SUMMARISE_TYPES]

    if args.agy:
        agy_bin = os.path.expanduser("~/.local/bin/agy")
        if not os.access(agy_bin, os.X_OK):
            agy_bin = "agy"
        log(f"Announcement {date}: {len(todo)} papers to summarise with Antigravity "
            f"({args.agy_model}) ({len(papers) - len(todo)} replacements listed).")
        with cf.ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(summarise_agy, p, agy_bin, args.agy_model, profile): p for p in todo}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                p = futs[fut]
                p["ai"] = fut.result()
                log(f"  [{i}/{len(todo)}] r={p['ai']['relevance']} {p['id']} {p['title'][:70]}")
    else:
        log(f"Announcement {date}: {len(todo)} papers to summarise with Gemini ({MODEL}) "
            f"({len(papers) - len(todo)} replacements listed).")
        with cf.ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(summarise, p, api_key, profile): p for p in todo}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                p = futs[fut]
                p["ai"] = fut.result()
                log(f"  [{i}/{len(todo)}] r={p['ai']['relevance']} {p['id']} {p['title'][:70]}")

    write_outputs(date, papers)
    for f in glob.glob(os.path.join(CACHE_DIR, "*.json")):
        if time.time() - os.path.getmtime(f) > 7 * 86400:
            os.remove(f)
    log(f"Done: {os.path.join(DIGEST_DIR, date + '.html')}")


if __name__ == "__main__":
    main()
