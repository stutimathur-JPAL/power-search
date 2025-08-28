# app.py — “Exact-figure” study search (≤4 results), bulletproof edition
# - Loads .pkl or .pkl.zip embeddings
# - Strict Sector & Region filters (AND)
# - Extracts SD / SE / Variance / ICC / MDE / Power exactly (robust)
# - Study-level results (≤4), clean cards with big number + citation + link
# - Embedding model/dimension sanity checks, safe fallback if no key
# - Filters out references/appendix/empty tables, boosts tables for power queries

import os, re, zipfile, pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============================ CONFIG ============================

DATA_DIR = "data"
PKL_NAME = "embeddings_dataframe.pkl"
ZIP_NAME = "embeddings_dataframe.pkl.zip"

# Set these in Streamlit → Manage app → Settings → Secrets
EMBED_MODEL = st.secrets.get("EMBED_MODEL") or "text-embedding-3-small"  # override in Secrets (you: "...3-large")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")) or None

MAX_RESULTS = 4
CHUNKS_PER_STUDY = 3
TABLE_BOOST = 0.25
NUMERIC_BONUS = 0.06
SHOW_EXCERPTS = False  # you said no paragraphs

# ============================ OPENAI ============================

try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# ============================ LOADERS ============================

@st.cache_data(show_spinner=False)
def _read_pickle_bytes_from_zip(zip_path: str) -> bytes:
    with zipfile.ZipFile(zip_path, "r") as zf:
        inner = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
        if not inner:
            raise FileNotFoundError(f"No .pkl inside {zip_path}. Found: {zf.namelist()}")
        with zf.open(inner[0]) as f:
            return f.read()

@st.cache_data(show_spinner=False)
def load_embeddings_df() -> Tuple[pd.DataFrame, int]:
    """Load DataFrame from data/embeddings_dataframe.pkl or .pkl.zip; validate columns & dims."""
    pkl_path = os.path.join(DATA_DIR, PKL_NAME)
    zip_path = os.path.join(DATA_DIR, ZIP_NAME)

    if os.path.exists(pkl_path):
        raw = open(pkl_path, "rb").read()
    elif os.path.exists(zip_path):
        raw = _read_pickle_bytes_from_zip(zip_path)
    else:
        raise FileNotFoundError(f"Missing {PKL_NAME} or {ZIP_NAME} in /{DATA_DIR}")

    try:
        df = pickle.loads(raw)
    except Exception as e:
        raise RuntimeError(
            "Could not unpickle the embeddings file. Keep numpy==2.0.1 in requirements, "
            "or re-export embeddings as plain lists in Colab.\n\n"
            f"Loader error: {e}"
        )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Embeddings file did not contain a pandas DataFrame.")

    # Normalize required columns
    if "text" not in df.columns:
        for alt in ["chunk_text", "content", "body"]:
            if alt in df.columns: df = df.rename(columns={alt: "text"}); break
    if "embedding" not in df.columns:
        for alt in ["embeddings", "vector", "vec"]:
            if alt in df.columns: df = df.rename(columns={alt: "embedding"}); break
    if "text" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"DataFrame needs 'text' and 'embedding'. Found: {list(df.columns)}")

    # Coerce embeddings to lists and drop bad rows
    def to_list(v):
        if isinstance(v, (list, tuple, np.ndarray)): return list(v)
        if isinstance(v, str) and v.strip().startswith("["):
            import ast
            try:
                x = ast.literal_eval(v)
                if isinstance(x, (list, tuple, np.ndarray)): return list(x)
            except Exception: pass
        return None

    df = df.dropna(subset=["text", "embedding"]).copy()
    df["embedding"] = df["embedding"].apply(to_list)
    df = df.dropna(subset=["embedding"]).reset_index(drop=True)

    dims = {len(e) for e in df["embedding"]}
    if not dims or len(dims) != 1:
        raise ValueError(f"Inconsistent embedding lengths in file: {sorted(list(dims)) if dims else 'none'}")
    emb_dim = list(dims)[0]

    if "source_type" not in df.columns: df["source_type"] = "text"
    if "pub_id" not in df.columns:      df["pub_id"] = np.arange(len(df)).astype(str)
    if "page" not in df.columns:        df["page"] = np.nan

    return df, emb_dim

# ============================ EMBEDDINGS & SCORING ============================

def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v);  return v / (n + 1e-12)

def embed_query(q: str) -> Optional[np.ndarray]:
    if not client: return None
    try:
        r = client.embeddings.create(model=EMBED_MODEL, input=q.strip())
        return np.array(r.data[0].embedding, dtype=np.float32)
    except Exception as e:
        st.warning(f"Embedding service error: {e}")
        return None

# ============================ PARAM EXTRACTOR (SAFE) ============================

NUM = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"

def looks_like_power_query(q: str) -> bool:
    q = (q or "").lower()
    return any(k in q for k in [
        "sd","standard deviation","std dev","std. dev",
        "se","standard error",
        "variance","icc","mde","mbe","minimum detectable","power"
    ])

def extract_power_inputs(text: str) -> Dict[str, float]:
    """Crash-proof extraction of SD/SE/variance/ICC/MDE/power (parameter phrasing only)."""
    t = " " + str(text).lower() + " "

    def pick(regexes, post_filter=None):
        for rgx in regexes:
            m = re.search(rgx, t, flags=re.I)
            if not m: continue
            # SAFE: use captured group if present, else whole match (prevents group(1) crash)
            span = m.group(1) if m.lastindex else m.group(0)
            n = re.search(NUM, span)
            if not n: continue
            try:
                x = float(n.group(0))
            except Exception:
                continue
            if post_filter and not post_filter(x, m): continue
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

    variance = pick([ rf"\bvariance\s*(?:=|:)?\s*({NUM})\b" ], post_filter=not_year)

    icc = pick([ rf"\b(?:icc|intracluster|intra[-\s]?cluster\s+correlation(?:\s+coefficient)?)\s*(?:=|:)?\s*({NUM})\b" ],
               post_filter=icc_0_1)

    mde = pick([ rf"\b(?:mde|mbe|minimum\s+detectable\s+effect(?:\s+size)?)\s*(?:=|:)?\s*({NUM})\s*%?\b",
                 rf"\b({NUM})\s*%?\s*(?:mde|mbe|minimum\s+detectable\s+effect(?:\s+size)?)\b" ],
               post_filter=pct_0_200)

    power = pick([ rf"\bpower\s*(?:=|:)?\s*({NUM})\s*%?\b",
                   rf"\b({NUM})\s*%?\s*power\b" ], post_filter=pct_0_200)

    out = {}
    if sd is not None: out["sd"] = sd
    if se is not None: out["se"] = se
    if variance is not None: out["variance"] = variance
    if icc is not None: out["icc"] = icc
    if mde is not None: out["mde"] = mde
    if power is not None: out["power"] = power
    return out

# ============================ TITLES / LINKS ============================

def guess_title_from_first_chunk(group: pd.DataFrame) -> str:
    g = group.sort_values("page", na_position="last")
    txt = str(g.iloc[0]["text"])
    lines = txt.splitlines()[:15]
    for line in lines:
        s = line.strip()
        if 20 <= len(s) <= 140 and s[0].isupper() and not re.match(r"^(Table|Figure|Appendix)\b", s, flags=re.I):
            return s
    m = re.split(r"(?<=[.!?])\s+", txt.strip())
    return (m[0][:120] + "…") if m else f"Study {group.iloc[0]['pub_id']}"

def fmt_link(row: pd.Series) -> Optional[str]:
    url = row.get("url") or row.get("link")
    doi = row.get("doi") or row.get("DOI")
    if isinstance(url, str) and url.strip(): return url.strip()
    if isinstance(doi, str) and doi.strip():
        d = doi.strip()
        return d if d.lower().startswith("http") else f"https://doi.org/{d}"
    return None

def format_citation(row: pd.Series, titles: Dict[str,str]) -> str:
    pub = str(row.get("pub_id","")).strip()
    title = row.get("title")
    if not isinstance(title, str) or not title.strip():
        title = titles.get(pub) or f"Study {pub}"
    authors = row.get("authors") or row.get("author") or ""
    year = row.get("year") or row.get("date") or ""
    parts = [title]
    if str(authors).strip(): parts.append(str(authors).strip())
    if str(year).strip():    parts.append(str(year).strip())
    return " · ".join(parts)

# ============================ UI ============================

st.set_page_config(page_title="Power Inputs Search (Embeddings)", layout="wide")
st.title("Power Inputs Search (Embeddings)")
st.success("✅ APP READY — type your query, set Sector & Country/Region, pick a figure, then Search.")

# Load data
try:
    df, emb_dim = load_embeddings_df()
except Exception as e:
    st.error(str(e)); st.stop()

with st.expander("Diagnostics (setup)", expanded=False):
    import sys
    st.write("Python:", sys.version.split()[0])
    st.write("Streamlit:", st.__version__)
    st.write("Rows:", len(df))
    st.write("Columns:", list(df.columns))
    st.write("Embedding dimension (from file):", emb_dim)
    st.write("Embedding model (secrets):", EMBED_MODEL)
    st.caption("1536 ↔ text-embedding-3-small, 3072 ↔ text-embedding-3-large")

# Sidebar — strict filters & focus
with st.sidebar:
    st.header("Search controls")
    query = st.text_input("What do you want?", placeholder="e.g., sd of education studies in India").strip()
    sectors = st.text_input("Sector (comma-separated, exact)", value="").strip()
    regions = st.text_input("Country/Region (comma-separated, exact)", value="").strip()
    focus = st.selectbox("Figure to extract (exact)", ["auto","sd","se","variance","icc","mde","power"], index=0)
    topk = st.slider("How many studies (max 4)", 1, 4, 3, 1)
    use_embeddings = st.toggle("Use embeddings (needs OPENAI_API_KEY)", value=bool(client))
    power_mode = st.toggle("Prefer numeric/table chunks", value=True)

# Titles for nicer display
titles = {}
try:
    titles = {pid: guess_title_from_first_chunk(g) for pid, g in df.groupby("pub_id")}
except Exception:
    pass

# ============================ HELPERS ============================

def parse_list_field(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def enforce_filters(row: pd.Series, countries: List[str], sectors: List[str]) -> bool:
    """
    AND logic. If explicit 'country'/'sector' columns exist, we use them; else substring in text.
    Also drops refs/appendix and empty tables.
    """
    t = str(row.get("text","")).lower()

    # Country
    if countries:
        if "country" in row.index and isinstance(row["country"], str) and row["country"].strip():
            if row["country"].strip().lower() not in countries:
                return False
        elif not any(c in t for c in countries):
            return False

    # Sector
    if sectors:
        if "sector" in row.index and isinstance(row["sector"], str) and row["sector"].strip():
            if row["sector"].strip().lower() not in sectors:
                return False
        elif not any(s in t for s in sectors):
            return False

    # Drop obvious noise unless the query explicitly asks power params
    if re.search(r"\b(references|bibliography|appendix)\b", t) and not looks_like_power_query(query):
        return False

    # Drop empty table stubs
    if str(row.get("source_type","")).lower() == "table" and re.search(r"columns:\s*0\b", t):
        return False

    return True

def keyword_overlap(q: str, txt: str) -> float:
    qw = set([w for w in re.findall(r"[A-Za-z]+", q.lower()) if len(w) > 2])
    tw = set([w for w in re.findall(r"[A-Za-z]+", str(txt).lower()) if len(w) > 2])
    return len(qw & tw) / (len(qw) or 1)

def detect_focus(q: str, manual: str) -> str:
    if manual != "auto": return manual
    ql = (q or "").lower()
    for k, keys in {
        "sd": [" sd", "standard deviation", "std dev", "std. dev"],
        "se": [" se", "standard error"],
        "variance": [" variance"],
        "icc": [" icc", "intra cluster", "intracluster", "intra-cluster"],
        "mde": [" mde", "mbe", "minimum detectable"],
        "power": [" power", "1-beta", "1 − beta", "1-β", "1 − β"]
    }.items():
        if any(s in ql for s in keys): return k
    return "sd"  # sensible default

def choose_metric_from_group(group: pd.DataFrame, metric: str) -> Optional[float]:
    """
    Pick the focused metric for a study:
    1) Prefer table chunks
    2) Else other chunks
    """
    g = group.copy()
    # Sort by score desc, with tables ahead
    g["_table"] = (g["source_type"].str.lower() == "table").astype(int)
    if "score" in g.columns:
        g = g.sort_values(["_table", "score"], ascending=[False, False])
    else:
        g = g.sort_values(["_table", "page"], ascending=[False, True])

    for _, r in g.iterrows():
        vals = r.get("power_inputs") or {}
        if metric in vals and vals[metric] is not None:
            return float(vals[metric])
    return None

def format_metric_value(metric: str, val: Optional[float]) -> str:
    if val is None: return "—"
    if metric == "power":
        if 0 <= val <= 1:   return f"{val*100:.0f}%  ({val:g})"
        if 0 < val <= 100:  return f"{val:g}%"
    return f"{val:g}"

# ============================ SEARCH ============================

if st.button("Search") and query:
    countries = parse_list_field(regions)
    sectors_  = parse_list_field(sectors)
    focus_metric = detect_focus(query, focus)

    work = df.copy()
    work = work[work.apply(lambda r: enforce_filters(r, countries, sectors_), axis=1)]
    if work.empty:
        st.warning("No chunks matched your (strict) Sector/Region filters. Try adjusting them.")
        st.stop()

    # Score chunks
    if use_embeddings and client is not None:
        try:
            E = np.vstack(work["embedding"].apply(lambda e: np.asarray(e, dtype=np.float32)).to_numpy())
        except Exception as e:
            st.error(f"Embeddings matrix build failed: {e}"); st.stop()

        qvec = embed_query(query)
        if qvec is None:
            st.warning("Embedding call failed; falling back to keyword overlap.")
            work["score"] = work["text"].apply(lambda t: keyword_overlap(query, t))
        else:
            # dimension sanity
            if E.shape[1] != qvec.size:
                st.error(
                    f"Model/file dimension mismatch. Query dim={qvec.size}, file dim={E.shape[1]}.\n"
                    "Set EMBED_MODEL in Secrets to the exact model used to build the file."
                ); st.stop()
            qv = normalize_vec(qvec)
            sims = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)) @ qv
            work["score"] = sims
    else:
        work["score"] = work["text"].apply(lambda t: keyword_overlap(query, t))

    # Power boosts
    if power_mode and looks_like_power_query(query):
        work["score"] = work["score"] + np.where(work["source_type"].str.lower()=="table", TABLE_BOOST, 0.0)
        work["score"] = work["score"] + work["text"].apply(
            lambda s: NUMERIC_BONUS if re.search(NUM, str(s).lower()) and any(
                k in str(s).lower() for k in ["standard deviation","std dev","sd","se","variance","icc","mde","power"]
            ) else 0.0
        )

    # Extract parameters per chunk
    work["power_inputs"] = work["text"].apply(extract_power_inputs)

    # Aggregate to study level (≤ CHUNKS_PER_STUDY per study considered)
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
                power_inputs_list=("power_inputs", list)
            )
            .reset_index()
            .sort_values("study_score", ascending=False)
    )

    # Build rows with exact figure (and filter out those lacking it if user chose a specific focus)
    rows = []
    for _, r in studies.iterrows():
        group = best[best["pub_id"] == r["pub_id"]].copy()
        exact_val = choose_metric_from_group(group, focus_metric)
        # merge other metrics (first-seen)
        merged = {}
        for d in r["power_inputs_list"]:
            for k, v in (d or {}).items():
                merged.setdefault(k, v)
        rows.append({
            "pub_id": r["pub_id"],
            "study_score": float(r["study_score"]),
            "title": r.get("title"),
            "authors": r.get("authors"),
            "year": r.get("year"),
            "url": r.get("url"),
            "doi": r.get("doi"),
            "any_text": r.get("any_text"),
            "exact_metric_value": exact_val,
            "all_metrics": merged
        })

    # If user asked a specific figure (not Auto), require it to be present
    if focus_metric != "auto":
        rows = [r for r in rows if r["exact_metric_value"] is not None]

    # Cap to MAX_RESULTS / slider
    rows = rows[: min(int(topk), MAX_RESULTS)]

    if not rows:
        st.warning("No studies contained that exact figure. Try another figure or adjust filters.")
        st.stop()

    # Display cards (clean; no paragraphs unless SHOW_EXCERPTS=True)
    st.success(f"Top {len(rows)} study/studies (focus: {focus_metric.upper() if focus_metric!='auto' else 'AUTO'})")
    for row in rows:
        pub = row["pub_id"]
        fake = pd.Series({
            "pub_id": pub,
            "title": row["title"],
            "authors": row["authors"],
            "year": row["year"],
            "url": row["url"],
            "doi": row["doi"]
        })
        cit = format_citation(fake, titles)
        link = fmt_link(fake)
        top_value = format_metric_value(focus_metric if focus_metric!="auto" else "sd", row["exact_metric_value"])

        with st.container(border=True):
            c1, c2 = st.columns([1, 5], gap="large")
            with c1:
                st.metric((focus_metric if focus_metric!="auto" else "sd").upper(), top_value)
            with c2:
                if link:
                    st.markdown(f"### [{cit}]({link})  \n*{pub}* · Score **{row['study_score']:.2f}**")
                else:
                    st.markdown(f"### {cit}  \n*{pub}* · Score **{row['study_score']:.2f}**")

            # Other parameters
            others = row["all_metrics"].copy()
            focus_key = focus_metric if focus_metric!="auto" else "sd"
            if focus_key in others: others.pop(focus_key, None)
            if others:
                order = ["sd","se","variance","icc","mde","power"]
                ordered = [(k, others[k]) for k in order if k in others] + [(k,v) for k,v in others.items() if k not in order]
                cols = st.columns(min(len(ordered), 6))
                for (k, v), c in zip(ordered, cols):
                    try:
                        c.metric(k.upper(), f"{float(v):g}")
                    except Exception:
                        c.metric(k.upper(), str(v))
            else:
                st.caption("Other parameters: —")

            if SHOW_EXCERPTS:
                ex = str(row.get("any_text",""))
                ex = ex[:900] + ("…" if len(ex) > 900 else "")
                st.write(ex)

else:
    st.info("Type your query, set **Sector** and **Country/Region** (exact), choose a figure (or leave Auto), then click **Search**.")
