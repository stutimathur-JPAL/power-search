# app.py — Single-page, robust RCT power-input search (embeddings, zip-aware)
# Features: zip loader, strict filters, crash-proof extractor, study-level aggregation,
# clean citation & link display, diagnostics.

import os, io, re, zipfile, pickle, hashlib, math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============================ CONFIG ============================

DATA_DIR = "data"
PKL_NAME = "embeddings_dataframe.pkl"
ZIP_NAME = "embeddings_dataframe.pkl.zip"

# Set your embedding model in Streamlit Secrets for safety:
#   Settings → Secrets:  EMBED_MODEL="text-embedding-3-small" (or the exact one you used)
EMBED_MODEL = st.secrets.get("EMBED_MODEL") or "text-embedding-3-small"

TOPK_DEFAULT = 10           # number of studies to show
CHUNKS_PER_STUDY = 3        # (hidden) for internal scoring before aggregation
TABLE_BOOST = 0.25          # boost tables when power-mode ON
NUMERIC_BONUS = 0.05        # small bonus for sd/se/variance/icc/mde/power phrases
SHOW_EXCERPTS = False       # you said you don't want excerpts now

# =================================================================
# OpenAI client (optional). Put OPENAI_API_KEY in Streamlit Secrets.
# =================================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")) or None
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# ============================ HELPERS ============================

@st.cache_data(show_spinner=False)
def _read_pickle_bytes_from_zip(zip_path: str) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as zf:
        inner = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
        if not inner:
            raise FileNotFoundError(f"No .pkl inside {zip_path}. Found: {zf.namelist()}")
        with zf.open(inner[0]) as f:
            return f.read()

@st.cache_data(show_spinner=False)
def load_embeddings_df() -> pd.DataFrame:
    """Load DataFrame from data/embeddings_dataframe.pkl or .pkl.zip."""
    pkl_path = os.path.join(DATA_DIR, PKL_NAME)
    zip_path = os.path.join(DATA_DIR, ZIP_NAME)

    if os.path.exists(pkl_path):
        raw = open(pkl_path, "rb").read()
    elif os.path.exists(zip_path):
        raw = _read_pickle_bytes_from_zip(zip_path)
    else:
        raise FileNotFoundError(f"Missing {PKL_NAME} or {ZIP_NAME} in /{DATA_DIR}")

    # Try unpickle; if there is a numpy version mismatch, show a helpful error.
    try:
        df = pickle.loads(raw)
    except Exception as e:
        raise RuntimeError(
            "Could not unpickle the embeddings file. This usually means a NumPy version mismatch.\n"
            "Fix: Re-export in Colab with embeddings stored as plain Python lists, OR keep this app's "
            "requirements.txt (numpy==2.0.1) and rebuild.\n\n"
            f"Loader error: {e}"
        )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Embeddings file did not contain a pandas DataFrame.")

    # Normalize expected columns (we'll be flexible)
    required_text = "text"
    required_emb  = "embedding"
    if required_text not in df.columns:
        # try common alternatives
        for alt in ["chunk_text", "content", "body"]:
            if alt in df.columns:
                df = df.rename(columns={alt: required_text})
                break
    if required_emb not in df.columns:
        # try alternatives for embedding column
        for alt in ["embeddings", "vector", "vec"]:
            if alt in df.columns:
                df = df.rename(columns={alt: required_emb})
                break

    if required_text not in df.columns or required_emb not in df.columns:
        raise ValueError(f"Dataframe must contain '{required_text}' and '{required_emb}'. Found: {list(df.columns)}")

    # Make sure each embedding is a 1D list/array of same length
    def _to_list(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        # if it's a string like "[0.1, 0.2, ...]" try literal eval
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                import ast
                x = ast.literal_eval(v)
                if isinstance(x, (list, tuple, np.ndarray)):
                    return list(x)
            except Exception:
                pass
        return None

    df = df.dropna(subset=[required_text, required_emb]).copy()
    df["embedding"] = df["embedding"].apply(_to_list)
    df = df.dropna(subset=["embedding"]).reset_index(drop=True)

    # Infer embedding dim and validate
    dims = {len(e) for e in df["embedding"]}
    if not dims or len(dims) != 1:
        raise ValueError(f"Inconsistent embedding lengths in file: {sorted(list(dims)) if dims else 'none'}")
    emb_dim = list(dims)[0]

    # Light clean
    if "source_type" not in df.columns:
        df["source_type"] = "text"
    if "pub_id" not in df.columns:
        # synthesize a group id if missing
        df["pub_id"] = np.arange(len(df)).astype(str)

    # Add simple page number if absent (helps title detection)
    if "page" not in df.columns:
        df["page"] = np.nan

    return df, emb_dim

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def embed_query(q: str) -> Optional[np.ndarray]:
    if not client:
        return None
    try:
        r = client.embeddings.create(model=EMBED_MODEL, input=q.strip())
        return np.array(r.data[0].embedding, dtype=np.float32)
    except Exception as e:
        st.warning(f"Embedding service error: {e}")
        return None

# ---------- robust extractor for sd/se/variance/icc/mde/power ----------
NUM = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"

def extract_power_inputs(text: str) -> Dict[str, float]:
    """
    Extract study-design parameters only (avoid 'effect in SD units' lines).
    Crash-proof: always captures a number before converting to float.
    """
    t = " " + str(text).lower() + " "

    def pick(regexes, post_filter=None):
        for rgx in regexes:
            m = re.search(rgx, t, flags=re.I)
            if not m:
                continue
            span = m.group(1) if m.lastindex else m.group(0)
            n = re.search(NUM, span)
            if not n:
                continue
            try:
                x = float(n.group(0))
            except Exception:
                continue
            if post_filter and not post_filter(x, m):
                continue
            return x
        return None

    not_year   = lambda x, m: not (1900 <= x <= 2100)
    pct_0_200  = lambda x, m: 0 <= x <= 200
    icc_0_1    = lambda x, m: 0 <= x <= 1

    sd = pick([
        rf"(?:std\.?\s*dev(?:iation)?|standard\s+deviation|sd)\s*(?:=|:|\bof\b|\()\s*({NUM})\s*\)?\b(?!\s*(units|across|increase|improv|gain|impact|effect))",
        rf"\b({NUM})\s*(?:std\.?\s*dev(?:iation)?|standard\s+deviation|sd)\b(?!\s*(units|across|increase|improv|gain|impact|effect))"
    ], post_filter=not_year)

    se = pick([
        rf"(?:standard\s+error|se)\s*(?:=|:|\()\s*({NUM})\s*\)?\b",
        rf"\b({NUM})\s*(?:standard\s+error|se)\b"
    ], post_filter=not_year)

    variance = pick([
        rf"\bvariance\s*(?:=|:)?\s*({NUM})\b"
    ], post_filter=not_year)

    icc = pick([
        rf"\b(?:icc|intracluster|intra[-\s]?cluster\s+correlation(?:\s+coefficient)?)\s*(?:=|:)?\s*({NUM})\b"
    ], post_filter=icc_0_1)

    mde = pick([
        rf"\b(?:mde|minimum\s+detectable\s+effect(?:\s+size)?)\s*(?:=|:)?\s*({NUM})\s*%?\b",
        rf"\b({NUM})\s*%?\s*(?:mde|minimum\s+detectable\s+effect(?:\s+size)?)\b"
    ], post_filter=pct_0_200)

    power = pick([
        rf"\bpower\s*(?:=|:)?\s*({NUM})\s*%?\b",
        rf"\b({NUM})\s*%?\s*power\b"
    ], post_filter=pct_0_200)

    out = {}
    if sd is not None: out["sd"] = sd
    if se is not None: out["se"] = se
    if variance is not None: out["variance"] = variance
    if icc is not None: out["icc"] = icc
    if mde is not None: out["mde"] = mde
    if power is not None: out["power"] = power
    return out

def looks_like_power_query(q: str) -> bool:
    q = (q or "").lower()
    return any(k in q for k in [
        "sd","standard deviation","std dev",
        "se","standard error",
        "variance","icc","mde","minimum detectable","power"
    ])

def has_numeric_power_terms(text: str) -> bool:
    t = str(text).lower()
    if not re.search(NUM, t):
        return False
    return any(k in t for k in ["standard deviation","std dev","sd","se","variance","icc","mde","power"])

# ---------- title + citation helpers ----------

def guess_title_from_first_chunk(group: pd.DataFrame) -> str:
    # pick earliest page, then find a title-ish line before "abstract"
    g = group.sort_values("page", na_position="last")
    txt = str(g.iloc[0]["text"])
    lines = txt.splitlines()[:15]
    for line in lines:
        s = line.strip()
        if 20 <= len(s) <= 140 and s[0].isupper() and not re.match(r"^(Table|Figure|Appendix)\b", s, flags=re.I):
            return s
    # fallback: first sentence
    m = re.split(r"(?<=[.!?])\s+", txt.strip())
    return (m[0][:120] + "…") if m else f"Study {group.iloc[0]['pub_id']}"

def fmt_link(row: pd.Series) -> Optional[str]:
    url = row.get("url") or row.get("link")
    doi = row.get("doi") or row.get("DOI")
    if isinstance(url, str) and url.strip():
        return url.strip()
    if isinstance(doi, str) and doi.strip():
        d = doi.strip()
        if d.lower().startswith("http"):
            return d
        return f"https://doi.org/{d}"
    return None

def format_citation(row: pd.Series, titles: Dict[str,str]) -> str:
    pub = str(row.get("pub_id","")).strip()
    title = row.get("title")
    if not isinstance(title, str) or not title.strip():
        title = titles.get(pub) or f"Study {pub}"
    authors = row.get("authors") or row.get("author") or ""
    year = row.get("year") or row.get("date") or ""
    pieces = [title]
    if str(authors).strip():
        pieces.append(str(authors).strip())
    if str(year).strip():
        pieces.append(str(year).strip())
    return " · ".join(pieces)

# ============================ UI ============================

st.set_page_config(page_title="Power Inputs Search (Embeddings)", layout="wide")
st.title("Power Inputs Search (Embeddings)")

# Load DF
try:
    df, emb_dim = load_embeddings_df()
except Exception as e:
    st.error(str(e))
    st.stop()

# Diagnostics header
with st.expander("Diagnostics (for setup)", expanded=False):
    st.write("Rows:", len(df))
    st.write("Columns:", list(df.columns))
    st.write("Embedding dimension (from file):", emb_dim)
    st.write("Embedding model (secrets):", EMBED_MODEL)
    st.caption("If model dim ≠ file dim, set EMBED_MODEL in Settings → Secrets to the exact model used in Colab.")

# Sidebar controls
with st.sidebar:
    st.header("Search")
    query = st.text_input("Query", placeholder="e.g., sd of education studies in India")
    sectors = st.text_input("Sector filter (comma-sep)", value="")
    regions = st.text_input("Country/Region filter (comma-sep)", value="")
    topk = st.slider("How many studies", 3, 30, TOPK_DEFAULT, 1)
    power_mode = st.toggle("Power mode (prioritize sd/se/variance/icc/mde/power)", value=True)
    use_embeddings = st.toggle("Use embeddings (needs OPENAI_API_KEY in Secrets)", value=bool(client))

    st.caption("Tip: Fill **sectors** (education/health/…) and **regions** (India/Ghana/…) to avoid off-topic chunks.")

# Prepare embedding matrix (only if we’re going to use it)
EMB = None
if use_embeddings:
    try:
        # Build matrix from the same df to keep alignment safe
        EMB = np.vstack(df["embedding"].apply(lambda e: np.asarray(e, dtype=np.float32)).to_numpy())
    except Exception as e:
        st.error(f"Bad embeddings column. {e}")
        st.stop()

# Build titles per pub_id for nicer display
titles = {}
try:
    titles = {pid: guess_title_from_first_chunk(g) for pid, g in df.groupby("pub_id")}
except Exception:
    pass

# Run search
if st.button("Search") and query.strip():

    work = df.copy()
    q_norm = query.strip()

    # Hard filters by substrings in the chunk text (lowercase)
    qs_countries = [s.strip().lower() for s in regions.split(",") if s.strip()]
    qs_sectors   = [s.strip().lower() for s in sectors.split(",") if s.strip()]

    def passes_filters(row) -> bool:
        t = str(row.get("text","")).lower()
        if qs_countries and not any(c in t for c in qs_countries):
            return False
        if qs_sectors and not any(s in t for s in qs_sectors):
            return False
        # drop references/appendix unless question explicitly about power params
        if re.search(r"\b(references|bibliography|appendix)\b", t) and not looks_like_power_query(q_norm):
            return False
        # drop empty table stubs (common in OCR)
        if str(row.get("source_type","")).lower() == "table" and re.search(r"columns:\s*0\b", t):
            return False
        return True

    work = work[work.apply(passes_filters, axis=1)]
    if work.empty:
        st.warning("No chunks matched your filters. Try clearing Sector/Region or rephrasing.")
        st.stop()

    # Score
    if use_embeddings and client is not None:
        qvec = embed_query(q_norm)
        if qvec is None:
            st.warning("Embedding call failed; falling back to keyword overlap.")
            use_embeddings = False
        else:
            # dimension sanity (helps catch wrong model)
            if EMB.shape[1] != qvec.size:
                st.error(
                    f"Model/file dimension mismatch. Query dim={qvec.size} but file dim={EMB.shape[1]}. "
                    "Set EMBED_MODEL in Secrets to the exact model used to build the file."
                )
                st.stop()
            qv = normalize(qvec)
            E = EMB  # already aligned to df
            sims = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)) @ qv
            work = work.assign(score=sims)
    if not use_embeddings:
        # low-tech fallback: simple keyword overlap (safe)
        def keyword_overlap(q, txt):
            qw = set(w for w in re.findall(r"[A-Za-z]+", q.lower()) if len(w) > 2)
            tw = set(w for w in re.findall(r"[A-Za-z]+", str(txt).lower()) if len(w) > 2)
            return len(qw & tw) / (len(qw) or 1)
        work = work.assign(score=work["text"].apply(lambda t: keyword_overlap(q_norm, t)))

    # Power mode numeric boosts
    if power_mode and looks_like_power_query(q_norm):
        work["score"] = work["score"] + np.where(
            work["source_type"].str.lower() == "table", TABLE_BOOST, 0.0
        )
        work["score"] = work["score"] + work["text"].apply(lambda s: NUMERIC_BONUS if has_numeric_power_terms(s) else 0.0)

    # Extract power inputs per chunk (for later aggregation)
    if power_mode:
        work["power_inputs"] = work["text"].apply(extract_power_inputs)
    else:
        work["power_inputs"] = [{} for _ in range(len(work))]

    # ------------------------ aggregate to study level ------------------------
    # Take top chunk scores per pub_id, then use max as study score.
    def agg_power(dicts: List[Dict[str,float]]) -> Dict[str,float]:
        out = {}
        for d in dicts:
            for k,v in d.items():
                if k not in out:
                    out[k] = v
        return out

    # sort by score descending and keep a few best chunks per study
    work = work.sort_values("score", ascending=False)
    work["_rank_within_pub"] = work.groupby("pub_id").cumcount()
    best = work[work["_rank_within_pub"] < CHUNKS_PER_STUDY].copy()

    studies = (
        best.groupby("pub_id")
            .agg(
                study_score=("score", "max"),
                any_text=("text", "first"),
                source_type=("source_type", "first"),
                year=("year", "first"),
                authors=("authors", "first"),
                title=("title", "first"),
                doi=("doi", "first"),
                url=("url", "first"),
                power_inputs=("power_inputs", lambda s: agg_power(list(s)))
            )
            .reset_index()
            .sort_values("study_score", ascending=False)
            .head(int(topk))
    )
    # -------------------------------------------------------------------------

    if studies.empty:
        st.warning("No studies after aggregation. Try fewer filters.")
        st.stop()

    # Display: clean citation + link + power inputs (no excerpt)
    st.success(f"Top {len(studies)} studies")
    for _, row in studies.iterrows():
        pub = row["pub_id"]
        cit = format_citation(row, titles)
        link = fmt_link(row)
        score = float(row["study_score"])
        power_vals = row.get("power_inputs") or {}

        with st.container(border=True):
            # Title/citation line
            if link:
                st.markdown(f"### [{cit}]({link})  \n*{pub}* · Score **{score:.2f}**")
            else:
                st.markdown(f"### {cit}  \n*{pub}* · Score **{score:.2f}**")

            # Power inputs as compact badges
            if power_vals:
                cols = st.columns(len(power_vals))
                order = ["sd","se","variance","icc","mde","power"]
                kv = [(k, power_vals[k]) for k in order if k in power_vals] + [(k,v) for k,v in power_vals.items() if k not in order]
                for (k, v), c in zip(kv, cols):
                    try:
                        c.metric(k.upper(), f"{float(v):g}")
                    except Exception:
                        c.metric(k.upper(), str(v))
            else:
                st.caption("Power inputs: —")

            if SHOW_EXCERPTS:
                ex = str(row.get("any_text",""))
                ex = ex[:900] + ("…" if len(ex) > 900 else "")
                st.write(ex)
else:
    st.info("Type a query (e.g., **sd of education studies in India**), set Sector/Region, then click **Search**.")

# ===================== PATCH START (safe extractor + filter fix) =====================
# This replaces the old extractor that crashed on m.group(1), and removes the
# str.contains() warning by using a non-capturing regex.

import re
NUM = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"

def looks_like_power_query(q: str) -> bool:
    q = (q or "").lower()
    return any(k in q for k in [
        "sd","standard deviation","std dev",
        "se","standard error",
        "variance","icc","mde","minimum detectable","power"
    ])

def extract_power_inputs(text: str) -> dict:
    """
    Robustly extract SD / SE / variance / ICC / MDE / power as parameters.
    Avoids picking effect-size sentences like “0.19 SD gain”.
    Never crashes on m.group(1).
    """
    t = " " + str(text).lower() + " "

    def pick(regexes, post_filter=None):
        for rgx in regexes:
            m = re.search(rgx, t, flags=re.I)
            if not m:
                continue
            # get a numeric substring safely (use captured group if present)
            span = m.group(1) if m.lastindex else m.group(0)
            n = re.search(NUM, span)
            if not n:
                continue
            try:
                x = float(n.group(0))
            except Exception:
                continue
            if post_filter and not post_filter(x, m):
                continue
            return x
        return None

    not_year   = lambda x, m: not (1900 <= x <= 2100)
    pct_0_200  = lambda x, m: 0 <= x <= 200
    icc_0_1    = lambda x, m: 0 <= x <= 1

    # SD: parameter phrasing only; avoid effect-size “gain/impact/units”
    sd = pick([
        rf"(?:std\.?\s*dev(?:iation)?|standard\s+deviation|sd)\s*(?:=|:|\bof\b|\()\s*({NUM})\s*\)?\b(?!\s*(units|across|increase|improv|gain|impact|effect))",
        rf"\b({NUM})\s*(?:std\.?\s*dev(?:iation)?|standard\s+deviation|sd)\b(?!\s*(units|across|increase|improv|gain|impact|effect))"
    ], post_filter=not_year)

    # SE: **both patterns CAPTURE the number** (this was the crash)
    se = pick([
        rf"(?:standard\s+error|se)\s*(?:=|:|\()\s*({NUM})\s*\)?\b",
        rf"\b({NUM})\s*(?:standard\s+error|se)\b"
    ], post_filter=not_year)

    variance = pick([ rf"\bvariance\s*(?:=|:)?\s*({NUM})\b" ], post_filter=not_year)

    icc = pick([
        rf"\b(?:icc|intracluster|intra[-\s]?cluster\s+correlation(?:\s+coefficient)?)\s*(?:=|:)?\s*({NUM})\b"
    ], post_filter=icc_0_1)

    mde = pick([
        rf"\b(?:mde|minimum\s+detectable\s+effect(?:\s+size)?)\s*(?:=|:)?\s*({NUM})\s*%?\b",
        rf"\b({NUM})\s*%?\s*(?:mde|minimum\s+detectable\s+effect(?:\s+size)?)\b"
    ], post_filter=pct_0_200)

    power = pick([
        rf"\bpower\s*(?:=|:)?\s*({NUM})\s*%?\b",
        rf"\b({NUM})\s*%?\s*power\b"
    ], post_filter=pct_0_200)

    out = {}
    if sd is not None: out["sd"] = sd
    if se is not None: out["se"] = se
    if variance is not None: out["variance"] = variance
    if icc is not None: out["icc"] = icc
    if mde is not None: out["mde"] = mde
    if power is not None: out["power"] = power
    return out

# If your code builds a `bonus_words` regex for str.contains, make it NON-capturing:
# Old (warns): "(standard deviation|standard error|...|MDE|power)"
# New (no warning):
try:
    bonus_words = r"(?:standard deviation|standard error|\bvariance\b|\bICC\b|minimum detectable|MDE|\bpower\b)"
except Exception:
    pass
# ====================== PATCH END ===========================================
