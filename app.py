# app.py  — robust filters + safer power-input extraction + titles

import os, re, ast, json, zipfile, io
import numpy as np
import pandas as pd
import streamlit as st

# ---------- CONFIG ----------
DATA_DIR = "data"
DATA_FILE = "embeddings_dataframe.pkl"           # preferred (unzipped)
ZIP_FILE  = "embeddings_dataframe.pkl.zip"       # allowed (zipped)
TOPK_DEFAULT = 10
MAX_EXCERPT_CHARS = 900

# ---------- OPENAI (optional for semantic search) ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")) or None
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        oai_client = None
except Exception:
    oai_client = None

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def _load_df() -> pd.DataFrame:
    """Load embeddings dataframe; accept .pkl or .pkl.zip in /data."""
    pkl_path = os.path.join(DATA_DIR, DATA_FILE)
    zip_path = os.path.join(DATA_DIR, ZIP_FILE)

    if not os.path.exists(pkl_path):
        # Try to unzip if a zip exists
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find inner file (allow any name as long as it ends with .pkl)
                inner = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
                if not inner:
                    raise FileNotFoundError("Zip found but no .pkl inside.")
                # Extract to memory then write to /data/embeddings_dataframe.pkl
                with zf.open(inner[0]) as f:
                    bytes_ = f.read()
                with open(pkl_path, "wb") as out:
                    out.write(bytes_)
        else:
            raise FileNotFoundError(
                f"Could not find {DATA_FILE} or {ZIP_FILE} in /{DATA_DIR}. "
                "Upload your file to data/ (exact name)."
            )

    df = pd.read_pickle(pkl_path)

    # Ensure expected columns exist
    for col in ["pub_id", "text", "source_type"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # If embeddings exist, keep them; if not, we do keyword fallback
    if "embedding" in df.columns:
        df = df.dropna(subset=["embedding"])
    return df

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _contains(text: str, word_list) -> bool:
    t = text.lower()
    return any(w.lower() in t for w in word_list if w)

def _is_country_hit(text: str, countries):
    if not countries: 
        return True
    return _contains(text, countries)

def _is_sector_hit(text: str, sectors):
    if not sectors:
        return True
    return _contains(text, sectors)

def _title_from_chunk(txt: str) -> str:
    """
    Guess a title-ish line from the top of a paper’s first page/abstract chunk.
    """
    t = txt.strip().splitlines()
    # Look for a longish capitalized line before 'Abstract'
    for line in t[:15]:
        clean = line.strip()
        if 20 <= len(clean) <= 140 and clean[0].isupper():
            # Avoid lines that are obviously not a title
            if not re.search(r"^\s*(Table|Figure|Appendix)\b", clean, flags=re.I):
                return clean
    # fallback: first sentence
    m = re.split(r"(?<=[.!?])\s+", txt.strip())
    return m[0][:120] + ("…" if len(m[0]) > 120 else "")

@st.cache_data(show_spinner=False)
def build_title_index(df: pd.DataFrame) -> dict:
    """
    Build a map pub_id -> best-guess title from earliest page chunk.
    """
    # Prefer text chunks, earliest page
    work = df[df["source_type"] == "text"].copy()
    if "page" in work.columns:
        work["page"] = work["page"].fillna(1e9)
        work = work.sort_values(["pub_id", "page"])
    else:
        work = work.sort_values(["pub_id"])
    titles = {}
    for pub_id, group in work.groupby("pub_id"):
        txt = str(group.iloc[0]["text"])
        titles[pub_id] = _title_from_chunk(txt)
    return titles

# ---------- Embeddings / scoring ----------
def embed_query(q: str) -> np.ndarray | None:
    if not oai_client:
        return None
    try:
        r = oai_client.embeddings.create(
            model="text-embedding-3-large",
            input=q.strip()
        )
        return np.array(r.data[0].embedding, dtype=np.float32)
    except Exception:
        return None

def dot(a, b): 
    return float(np.dot(a, b)) if a is not None and b is not None else 0.0

def rough_keyword_score(q: str, txt: str) -> float:
    """Fallback scorer if we don’t have an embedding for query/chunks."""
    qw = set([w for w in re.findall(r"[A-Za-z]+", q.lower()) if len(w) > 2])
    tw = set([w for w in re.findall(r"[A-Za-z]+", txt.lower()) if len(w) > 2])
    if not qw: 
        return 0.0
    return len(qw & tw) / len(qw)

# ---------- Safer power-input extraction ----------
NUM = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
def extract_power_inputs(text: str) -> dict:
    """
    Try to pull sd, se, variance, icc, mde, power only when the local wording
    suggests *parameters*, not 'effects in SD units'.
    """
    t = " " + text.lower() + " "

    def pick(regexes, post_filter=None):
        for rgx in regexes:
            m = re.search(rgx, t, flags=re.I)
            if m:
                val = m.group(1)
                try:
                    x = float(val)
                except Exception:
                    continue
                if post_filter and not post_filter(x, m):
                    continue
                return x
        return None

    # heuristics to avoid grabbing years / effect sizes in SD-units:
    not_year = lambda x, m: not (x >= 1900 and x <= 2100)
    not_pct_200 = lambda x, m: 0 <= x <= 200

    # standard deviation: require "standard deviation" or "(sd" or " sd:" near number
    sd = pick([
        rf"(?:standard\s+deviation|sd)\s*(?:=|:|\(|of)?\s*({NUM})\b",
        rf"\b({NUM})\s*(?:standard\s+deviation|sd)\b(?!\s*units)"
    ], post_filter=not_year)

    se = pick([
        rf"(?:standard\s+error|se)\s*(?:=|:|\()?{NUM}\)?",
        rf"\b({NUM})\s*(?:standard\s+error|se)\b"
    ], post_filter=not_year)

    variance = pick([
        rf"\bvariance\s*(?:=|:)?\s*({NUM})\b"
    ], post_filter=not_year)

    # ICC usually between 0 and 1 (sometimes up to ~0.5 in practice)
    icc = pick([
        rf"\b(?:icc|intracluster|intra[-\s]?cluster\s+correlation(?:\s+coefficient)?)\s*(?:=|:)?\s*({NUM})\b"
    ], post_filter=lambda x, m: 0 <= x <= 1)

    mde = pick([
        rf"\b(?:mde|minimum\s+detectable\s+effect(?:\s+size)?)\s*(?:=|:)?\s*({NUM})\b%?",
        rf"\b({NUM})\s*%?\s*(?:mde|minimum\s+detectable\s+effect(?:\s+size)?)\b"
    ], post_filter=not_pct_200)

    power = pick([
        rf"\bpower\s*(?:=|:)?\s*({NUM})\s*%?\b",
        rf"\b({NUM})\s*%?\s*power\b"
    ], post_filter=not_pct_200)

    out = {}
    if sd is not None: out["sd"] = sd
    if se is not None: out["se"] = se
    if variance is not None: out["variance"] = variance
    if icc is not None: out["icc"] = icc
    if mde is not None: out["mde"] = mde
    if power is not None: out["power"] = power
    return out

def looks_like_power_query(q: str) -> bool:
    return bool(re.search(r"\b(sd|se|variance|icc|power|mde|minimum detectable|standard deviation|standard error)\b", q, flags=re.I))

# ---------- UI ----------
st.set_page_config(page_title="Power Inputs Search (Embeddings)", layout="wide")
st.title("Power Inputs Search (Embeddings)")

with st.sidebar:
    st.header("Search")
    query = st.text_input("Query", value="standard deviation of education studies in India")
    sectors = st.text_input("Sector filter (e.g., education; comma-separated)", value="education")
    regions = st.text_input("Region/Country filter (e.g., India; comma-separated)", value="India")
    topk = st.number_input("Results", min_value=1, max_value=50, value=TOPK_DEFAULT, step=1)
    power_mode = st.toggle("Power mode (prefer sd/se/variance/icc/mde/power)", value=True)
    use_embeddings = st.toggle("Use embeddings (if key set)", value=True if oai_client else False)
    run = st.button("Search")

# Try load DF (show nice message if missing)
try:
    df = _load_df()
except Exception as e:
    st.error(f"Could not load `data/{DATA_FILE}`. Upload `{DATA_FILE}` or `{ZIP_FILE}` to `/data`.\n\nError: {e}")
    st.stop()

titles = build_title_index(df)

if run:
    q_norm = _norm(query)
    # If query mentions India or a sector, fold them into filters
    query_countries = [s.strip() for s in regions.split(",") if s.strip()]
    query_sectors = [s.strip() for s in sectors.split(",") if s.strip()]

    # 1) Score
    if use_embeddings and oai_client and "embedding" in df.columns:
        qvec = embed_query(q_norm)
        if qvec is not None:
            df["score"] = df["embedding"].apply(lambda e: dot(np.array(e, dtype=np.float32), qvec))
        else:
            df["score"] = df["text"].apply(lambda t: rough_keyword_score(q_norm, str(t)))
    else:
        df["score"] = df["text"].apply(lambda t: rough_keyword_score(q_norm, str(t)))

    # 2) Hard filters for sector/region
    def passes_filters(row):
        t = str(row["text"])
        ok_country = _is_country_hit(t, query_countries)
        ok_sector  = _is_sector_hit(t, query_sectors)
        return ok_country and ok_sector

    filt = df[df.apply(passes_filters, axis=1)].copy()

    # If power_mode AND the question looks like a power question,
    # slightly prefer table-ish chunks and add a small bonus for phrases we care about.
    if power_mode and looks_like_power_query(q_norm):
        # bonus for table chunks
        filt["score"] = filt["score"] + np.where(filt["source_type"].str.lower() == "table", 0.10, 0.0)
        # bonus if text explicitly mentions our keys
        bonus_words = r"(standard deviation|standard error|\bvariance\b|\bICC\b|minimum detectable|MDE|\bpower\b)"
        filt["score"] = filt["score"] + filt["text"].str.contains(bonus_words, case=False, regex=True).astype(float)*0.05

    # 3) Rank & display
    out = filt.sort_values("score", ascending=False).head(int(topk))

    if out.empty:
        st.warning("No results matched your filters. Try fewer filters or different words.")
    else:
        for _, row in out.iterrows():
            pub = row["pub_id"]
            typ = row.get("source_type", "text")
            score = float(row["score"])
            txt = _norm(str(row["text"]))
            title = titles.get(pub, f"Study {pub}")
            power_vals = extract_power_inputs(txt) if power_mode else {}

            with st.container(border=True):
                st.markdown(f"### {title}  \n*{pub} • {typ}*  \nScore: **{score:.2f}**")

                if power_vals:
                    cols = st.columns(len(power_vals))
                    for (k,v), c in zip(power_vals.items(), cols):
                        c.metric(k.upper(), f"{v:g}")
                else:
                    st.caption("Power inputs: —")

                # excerpt
                ex = txt[:MAX_EXCERPT_CHARS] + ("…" if len(txt) > MAX_EXCERPT_CHARS else "")
                st.write(ex)
