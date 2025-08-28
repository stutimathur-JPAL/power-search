# ---------- Power Inputs Search (Embeddings) ----------
# Clean UI + numeric extraction (sd, se, var, icc, power, mde, n)
# Works with your prebuilt embeddings_dataframe.pkl

import os, re, json, zipfile, math, io, requests
import numpy as np
import pandas as pd
import streamlit as st

# Optional fuzzy fallback if no OpenAI key
try:
    from rapidfuzz import process, fuzz
    HAS_FUZZ = True
except Exception:
    HAS_FUZZ = False

# Optional OpenAI for query embeddings
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---- CONFIG ----
DATA_PATH = "data/embeddings_dataframe.pkl"  # your file
ZIP_CANDIDATES = [
    "data/embeddings_dataframe.pkl.zip",
    "data/embeddings_dataframe.zip",
]
EMBEDDING_MODEL = "text-embedding-3-large"   # must match how .pkl was created
TOPK_DEFAULT = 10
MAX_EXCERPT_CHARS = 1200

# ---- OPTIONAL: remote download (if you later store the file elsewhere) ----
REMOTE_URL = st.secrets.get("EMBEDDINGS_URL", "")  # leave blank unless you add it in Streamlit Secrets

# ---- OpenAI client (only if you add OPENAI_API_KEY in Streamlit Secrets) ----
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")) or None
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None


# ---------- Helpers ----------
def ensure_embeddings_file() -> str:
    """Return path to the .pkl. If missing, try zip; if provided, try REMOTE_URL."""
    if os.path.exists(DATA_PATH):
        return DATA_PATH

    # Try zip in repo
    for z in ZIP_CANDIDATES:
        if os.path.exists(z):
            try:
                os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
                with zipfile.ZipFile(z, "r") as zf:
                    member = next((m for m in zf.namelist() if m.lower().endswith(".pkl")), None)
                    if not member:
                        raise RuntimeError("Zip has no .pkl inside.")
                    with zf.open(member) as src, open(DATA_PATH, "wb") as dst:
                        dst.write(src.read())
                return DATA_PATH
            except Exception as e:
                st.error(f"Found {z} but failed to extract it: {e}")
                st.stop()

    # Try remote URL (only if you added EMBEDDINGS_URL in Secrets)
    if REMOTE_URL:
        try:
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            r = requests.get(REMOTE_URL, timeout=60)
            r.raise_for_status()
            with open(DATA_PATH, "wb") as f:
                f.write(r.content)
            return DATA_PATH
        except Exception as e:
            st.error(f"Failed to download embeddings from EMBEDDINGS_URL: {e}")
            st.stop()

    st.error(f"Could not load {DATA_PATH}. Please upload it to your repo at data/embeddings_dataframe.pkl.")
    st.stop()


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def get_query_embedding(query: str) -> np.ndarray | None:
    """Use OpenAI if key is present, else None (fallback to fuzzy)."""
    if client is None:
        return None
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        return np.array(resp.data[0].embedding, dtype=float)
    except Exception as e:
        st.warning(f"Embedding service unavailable, using fallback search. ({e})")
        return None


NUM_PAT = r"[-+]?\d+(?:\.\d+)?"
def _find(pattern, text):
    m = re.search(pattern, text, flags=re.I)
    return float(m.group(1)) if m else None

def extract_power_inputs(text: str) -> dict:
    """
    Pulls key stats commonly used for power inputs.
    Heuristic but robust to minor formatting variations.
    """
    clean = " ".join(text.split())  # collapse whitespace

    # sd / standard deviation
    sd = _find(r"(?:\bsd\b|std\.?\s*dev(?:iation)?|standard\s*deviation)\s*[:=]?\s*(" + NUM_PAT + ")", clean)

    # se / standard error
    se = _find(r"(?:\bse\b|standard\s*error)\s*[:=]?\s*(" + NUM_PAT + ")", clean)

    # variance / var / σ²
    var = _find(r"(?:\bvariance\b|\bvar\b|σ\^?2|sigma\^?2)\s*[:=]?\s*(" + NUM_PAT + ")", clean)

    # ICC / intracluster corr
    icc = _find(r"(?:\bicc\b|intra[-\s]*cluster\s*corr(?:elation)?)\s*[:=]?\s*(" + NUM_PAT + ")", clean)

    # MDE / detectable effect
    mde = _find(r"(?:\bmde\b|minimum\s*detectable\s*effect)\s*[:=]?\s*(" + NUM_PAT + ")", clean)

    # power (as a probability 0–1 or %). Avoid picking up “power grid”, etc.
    power = _find(r"(?:statistical\s*power|power\s*\(1-β\)|\bpower\b)\s*[:=]?\s*(" + NUM_PAT + ")", clean)

    # sample size n / N / cluster size
    n = _find(r"(?:\bn\s*=\s*|\bN\s*=\s*)(\d+)", clean)

    out = {"sd": sd, "se": se, "variance": var, "icc": icc, "mde": mde, "power": power, "n": n}
    return {k: v for k, v in out.items() if v is not None}


def score_numeric_density(text: str) -> int:
    """Used to boost 'Power Mode' results: count how many target tokens appear."""
    keys = ["sd", "standard deviation", "se", "standard error", "variance", "var", "icc",
            "intra cluster", "power", "mde", "minimum detectable", "sample size", " n = "]
    t = text.lower()
    return sum(1 for k in keys if k in t)


def format_power_inputs(d: dict) -> str:
    if not d:
        return "—"
    nice = []
    for k in ["sd", "se", "variance", "icc", "mde", "power", "n"]:
        if k in d:
            v = d[k]
            nice.append(f"**{k}**: {v:g}" if isinstance(v, (int, float)) else f"**{k}**: {v}")
    return " • ".join(nice)


def search(df: pd.DataFrame, query: str, top_k: int, power_mode: bool,
           sector_filter: str, region_filter: str, use_embeddings: bool):
    work = df.copy()

    # Optional filters by substring (case-insensitive) on text/caption
    if sector_filter:
        work = work[work["text"].str.contains(sector_filter, case=False, na=False) |
                    work.get("caption", pd.Series([""]*len(work))).str.contains(sector_filter, case=False, na=False)]

    if region_filter:
        work = work[work["text"].str.contains(region_filter, case=False, na=False) |
                    work.get("caption", pd.Series([""]*len(work))).str.contains(region_filter, case=False, na=False)]

    if work.empty:
        return work

    # Compute score
    if use_embeddings:
        q = get_query_embedding(query)
        if q is not None:
            q = normalize(q)
            def dot_sim(vec):
                v = normalize(np.array(vec, dtype=float))
                return float(np.dot(v, q))
            work["score"] = work["embedding"].apply(dot_sim)
        else:
            use_embeddings = False  # fallback below

    if not use_embeddings:
        # Fallback: fuzzy + keyword boosts
        if not HAS_FUZZ:
            # simple keyword count
            key = query.lower()
            work["score"] = work["text"].str.lower().str.count(re.escape(key))
        else:
            def fuzz_score(row):
                s1 = process.extractOne(query, [row["text"]], scorer=fuzz.token_set_ratio)[1]
                s2 = process.extractOne(query, [row.get("caption", "")], scorer=fuzz.token_set_ratio)[1]
                return max(s1, s2) / 100.0
            work["score"] = work.apply(fuzz_score, axis=1)

    # Power mode: boost rows that look numeric/statistical
    if power_mode:
        bonus = work["text"].apply(score_numeric_density)
        work["score"] = work["score"] + 0.05 * bonus.clip(0, 10)

    # Extract numbers for display
    work["power_inputs"] = work["text"].apply(extract_power_inputs)

    # Order and return top_k
    return work.sort_values("score", ascending=False).head(top_k)


# ---------- UI ----------
st.set_page_config(page_title="Power Inputs Search (Embeddings)", layout="wide")
st.title("Power Inputs Search (Embeddings)")

# Make sure the data exists, then load
pkl_path = ensure_embeddings_file()
try:
    df = pd.read_pickle(pkl_path)
except Exception as e:
    st.error(f"Could not load {pkl_path}. Error: {e}")
    st.stop()

# Safety: ensure required columns exist
for c in ["pub_id", "source_type", "text", "embedding"]:
    if c not in df.columns:
        st.error(f"Missing required column `{c}` in embeddings dataframe.")
        st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Search options")
    query = st.text_input("Query", placeholder="e.g., sd or icc for health in India")
    power_mode = st.toggle("Power mode (prioritize numeric/statistical chunks)", value=True,
                           help="Boosts chunks that contain sd/se/variance/icc/power/mde/n")
    sector_filter = st.text_input("Sector filter (optional)", placeholder="education, health, labor, climate…")
    region_filter = st.text_input("Region/Country filter (optional)", placeholder="India, Ghana, Haryana…")
    top_k = st.slider("Results", 5, 50, TOPK_DEFAULT, 1)
    use_emb = st.toggle("Use embeddings (requires OpenAI key in Secrets)", value=client is not None)

    st.caption("Tip: add **OPENAI_API_KEY** in Streamlit → Settings → Secrets to enable high-quality semantic search.")

# Run search
if st.button("Search") and query.strip():
    results = search(
        df=df,
        query=query.strip(),
        top_k=top_k,
        power_mode=power_mode,
        sector_filter=sector_filter.strip(),
        region_filter=region_filter.strip(),
        use_embeddings=use_emb,
    )

    if results.empty:
        st.warning("No results. Try removing filters or different keywords.")
    else:
        for _, row in results.iterrows():
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.markdown(f"**Study:** {row.get('pub_id', '—')}")
                    st.markdown(f"**Type:** {row.get('source_type', '—')}")
                with col2:
                    st.markdown(f"**Score:** {row.get('score', 0):.2f}")
                with col3:
                    st.markdown("**Power inputs:**")
                    st.markdown(format_power_inputs(row.get("power_inputs", {})))
                with col4:
                    cap = row.get("caption", "")
                    if isinstance(cap, str) and cap.strip():
                        st.markdown("**Caption:**")
                        st.markdown(cap[:120] + ("…" if len(cap) > 120 else ""))

                # Excerpt
                text = row.get("text", "")
                snippet = (text[:MAX_EXCERPT_CHARS] + "…") if len(text) > MAX_EXCERPT_CHARS else text
                st.markdown("**Excerpt:**")
                st.write(snippet)

else:
    st.info("Enter a query on the left and click **Search**.")
