import os
import ast
import re
import json
import numpy as np
import pandas as pd
import streamlit as st

# --------- CONFIG ---------
DATA_PATH = "data/embeddings_dataframe.pkl"   # put your .pkl here
EMBEDDING_MODEL = "text-embedding-3-large"    # must match how the .pkl was created
TOPK_DEFAULT = 10
MAX_EXCERPT_CHARS = 1200

# --------- OPTIONAL OPENAI (for embeddings of the query; required for semantic search) ----------
# We only create the client if an API key is present.
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")) or None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as _e:
        _openai_client = None
else:
    _openai_client = None

# =====================
# Utilities
# =====================

def _as_float(x):
    try:
        return float(x)
    except:
        return None

def _is_year(x):
    try:
        v = int(float(x))
        return 1800 <= v <= 2100
    except:
        return False

_NUM = r"(-?\d+(?:\.\d+)?)"

def extract_power_inputs_clean(text: str) -> dict:
    """
    Conservative extractor for key power inputs.
    Returns dict like {"sd": 1.23, "variance": 0.5, "icc": 0.12, "N": 1234, "alpha": 0.05, "power": 0.8, "mde": 0.2}
    """
    t = " ".join((text or "").split())
    out = {}

    # SD
    m = re.search(rf"(?i)\b(sd|standard\s*deviation|std\.?|σ)\b[^0-9]{{0,20}}{_NUM}", t)
    if m:
        val = _as_float(m.group(2))
        if val is not None and not _is_year(val) and 0 < val < 1000:
            out["sd"] = val

    # Variance
    m = re.search(rf"(?i)\b(var(?:iance)?)\b[^0-9]{{0,20}}{_NUM}", t)
    if m:
        val = _as_float(m.group(2))
        if val is not None and not _is_year(val) and 0 <= val < 1e6:
            out["variance"] = val

    # ICC
    m = re.search(rf"(?i)\bicc\b[^0-9]{{0,12}}{_NUM}", t)
    if m:
        val = _as_float(m.group(1))
        if val is not None:
            if 0 <= val <= 1:
                out["icc"] = val
            elif 1 < val <= 100:
                out["icc"] = round(val/100.0, 4)

    # Sample size N
    m = re.search(rf"(?i)\bN\s*=\s*{_NUM}\b", t)
    if m:
        val = _as_float(m.group(1))
        if val is not None and val >= 10:
            out["N"] = int(val)

    # Alpha
    m = re.search(rf"(?i)\b(alpha|significance)\b[^0-9]{{0,12}}{_NUM}", t)
    if m:
        val = _as_float(m.group(2))
        if val is not None and 0 < val < 1:
            out["alpha"] = val

    # Power
    m = re.search(rf"(?i)\bpower\b[^0-9]{{0,12}}{_NUM}\s*%?", t)
    if m:
        val = _as_float(m.group(1))
        if val is not None:
            if 0 < val <= 1:
                out["power"] = val
            elif 1 < val <= 100:
                out["power"] = round(val/100.0, 4)

    # MDE
    m = re.search(rf"(?i)\b(mde|min(?:imum)?\s*detect(?:able)?\s*effect)\b[^0-9]{{0,12}}{_NUM}", t)
    if m:
        val = _as_float(m.group(3) if m.lastindex and m.lastindex >= 3 else m.group(m.lastindex or 1))
        if val is not None and 0 < val < 1000:
            out["mde"] = val

    return out

def human_power_dict(d: dict) -> str:
    if not d:
        return "—"
    order = ["sd", "se", "variance", "icc", "N", "alpha", "power", "mde"]
    parts = []
    for k in order:
        if k in d:
            v = d[k]
            if k in ("alpha", "power", "icc") and 0 < v <= 1:
                parts.append(f"{k}: {v:.3f}")
            else:
                try:
                    if float(v).is_integer():
                        parts.append(f"{k}: {int(v)}")
                    else:
                        parts.append(f"{k}: {float(v):.3g}")
                except:
                    parts.append(f"{k}: {v}")
    return " · ".join(parts) if parts else "—"

@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    # Normalize common column names
    if "text" not in df.columns:
        # Some dumps might call it 'chunk' or something else
        txt_col = None
        for c in df.columns:
            if str(c).lower() in ("chunk", "chunk_text", "content", "page_text"):
                txt_col = c
                break
        if txt_col:
            df["text"] = df[txt_col]
        else:
            raise RuntimeError("Could not find text column in embeddings dataframe.")
    # Ensure 'embedding' is a list of floats
    if isinstance(df["embedding"].iloc[0], str):
        # sometimes saved as stringified list; convert
        df["embedding"] = df["embedding"].apply(lambda s: ast.literal_eval(s))
    return df

@st.cache_resource(show_spinner=False)
def build_matrix(df: pd.DataFrame):
    """
    Returns (matrix, norms) for cosine similarity.
    """
    E = np.array(df["embedding"].to_list(), dtype=np.float32)  # shape (n, d)
    # Cosine similarity => normalize rows
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E_norm = E / norms
    return E_norm

def get_query_embedding(query: str) -> np.ndarray:
    """
    Get embedding for the query using OpenAI. Fails with a nice message if no key.
    """
    if not _openai_client:
        st.error("Semantic search needs an OpenAI key. Click ⋮ (top-right) → **Settings** → **Secrets** → add `OPENAI_API_KEY`.\n"
                 "Then retry the search.")
        st.stop()
    try:
        r = _openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        v = np.array(r.data[0].embedding, dtype=np.float32)
        # normalize
        n = np.linalg.norm(v)
        if n == 0:
            return v
        return v / n
    except Exception as e:
        st.error(f"Embedding call failed: {e}")
        st.stop()

def contains_any(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)

def pretty_title_from_row(row: pd.Series) -> str:
    """Make something like PUB-XXXX • text(table) • Page 12 • etc."""
    parts = []
    parts.append(str(row.get("pub_id", "Unknown")))
    stype = row.get("source_type") or "text"
    parts.append("table" if stype == "table" else "text")
    if stype == "text" and pd.notnull(row.get("page")):
        parts.append(f"Page {int(row.get('page'))}")
    if stype == "table" and pd.notnull(row.get("table_index")):
        parts.append(f"Table {int(row.get('table_index'))}")
    return " • ".join(parts)

def clip_excerpt(s: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit(" ", 1)[0] + "…"

def badge(text):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            background:#eef2ff;
            color:#1f3a8a;
            border:1px solid #c7d2fe;
            padding:2px 8px;
            border-radius:9999px;
            font-size:12px;
            margin-right:6px;
        ">{text}</span>
        """,
        unsafe_allow_html=True
    )

# =====================
# UI
# =====================

st.set_page_config(page_title="Power Inputs Search", layout="wide")
st.title("Power Inputs Search (Embeddings)")

# Load data
try:
    df = load_df(DATA_PATH)
except Exception as e:
    st.error(f"Could not load `data/embeddings_dataframe.pkl`.\n\n"
             f"Please upload it to your repo at `data/embeddings_dataframe.pkl`.\n\nError: {e}")
    st.stop()

E_norm = build_matrix(df)

with st.sidebar:
    st.subheader("Filters")
    sector_kw = st.text_input("Sector keywords (optional)", placeholder="education; health; governance")
    region_kw = st.text_input("Region/country keywords (optional)", placeholder="India; Pakistan; Kenya")
    include_tables = st.checkbox("Include tables", value=True)
    power_only = st.checkbox("Power-only (show only rows where SD/Var/ICC/N/α/Power/MDE detected)", value=False)
    min_sim = st.slider("Min similarity", 0.0, 1.0, 0.0, 0.01)
    topk = st.slider("Max results", 5, 100, TOPK_DEFAULT, 1)

# Main search box
query = st.text_input("Search (e.g., “variance in student test scores India”)", "")
go = st.button("Search")

# Hints
with st.expander("Examples", expanded=False):
    st.markdown("- `standard deviation test scores India`\n"
                "- `ICC school cluster trials`\n"
                "- `minimum detectable effect vaccination`\n"
                "- `variance student outcomes Kenya`")

if go:
    if not query.strip():
        st.warning("Type something to search.")
        st.stop()

    # 1) Query embedding
    qv = get_query_embedding(query.strip())

    # 2) Cosine similarities
    sims = (E_norm @ qv).astype(np.float32)  # (n,)

    # 3) Attach to df
    work = df.copy()
    work["similarity"] = sims

    # 4) Filter by tables, sector/region keywords
    if not include_tables:
        work = work[(work["source_type"].fillna("text") != "table")]

    sector_list = [w.strip() for w in (sector_kw or "").split(";") if w.strip()]
    region_list = [w.strip() for w in (region_kw or "").split(";") if w.strip()]

    def _row_passes_keywords(row):
        hay = " ".join([
            str(row.get("text", "")),
            str(row.get("caption", "")),
            str(row.get("pub_id", "")),
        ])
        return contains_any(hay, sector_list) and contains_any(hay, region_list)

    if sector_list or region_list:
        work = work[work.apply(_row_passes_keywords, axis=1)]

    # 5) Drop low similarities
    work = work[work["similarity"] >= float(min_sim)]

    # 6) Extract power inputs on the filtered slice
    work["power_fields"] = work["text"].fillna("").apply(extract_power_inputs_clean)
    work["has_power"] = work["power_fields"].apply(lambda d: bool(d))

    # 7) If power-only, filter again
    if power_only:
        work = work[work["has_power"]]

    # 8) Rank
    work = work.sort_values("similarity", ascending=False).head(topk)

    # 9) Display
    st.subheader("Results")
    st.caption(f"{len(work)} shown")

    if len(work) == 0:
        st.info("No matches with current filters. Try removing filters or lowering the min similarity.")
        st.stop()

    # Download CSV
    _dl_cols = ["pub_id", "source_type", "page", "table_index", "caption", "similarity", "text"]
    _csv = work[_dl_cols + ["power_fields"]].copy()
    _csv["power_fields"] = _csv["power_fields"].apply(lambda d: json.dumps(d, ensure_ascii=False))
    st.download_button("Download results (CSV)", _csv.to_csv(index=False).encode("utf-8"), file_name="results.csv", mime="text/csv")

    for i, row in work.reset_index(drop=True).iterrows():
        with st.container(border=True):
            left, right = st.columns([0.75, 0.25])
            with left:
                st.markdown(f"**{i+1}. {pretty_title_from_row(row)}**")
                st.write(clip_excerpt(row.get("text", "")))
                if row.get("source_type") == "table" and pd.notnull(row.get("caption")):
                    st.caption(f"Table caption: {row['caption']}")
            with right:
                badge(f"Score {row['similarity']*100:.1f}")
                if row.get("has_power"):
                    badge("Power inputs found")
                st.markdown("**Power inputs (detected):**")
                st.write(human_power_dict(row.get("power_fields", {})))
