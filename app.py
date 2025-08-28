# --- Streamlit app: nicer cards + Power mode extraction ---
import io, pickle, re
import numpy as np
import pandas as pd
import streamlit as st

EMBED_MODEL = "text-embedding-3-large"

# Get key from Secrets or once-per-session paste
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")

st.set_page_config(page_title="Paper Search", layout="wide")
st.title("Paper Search")

# Optional: paste key if Secrets not set
if not OPENAI_KEY:
    with st.expander("🔐 Add your OpenAI key (or set it in Settings → Secrets)"):
        pasted = st.text_input("OpenAI API key", type="password")
        if pasted:
            st.session_state["OPENAI_API_KEY"] = pasted
            OPENAI_KEY = pasted

st.subheader("Data source")
emb_file = st.file_uploader("Upload embeddings file (.pkl)", type=["pkl"])
meta_file = st.file_uploader("Optional metadata CSV (columns: pub_id,title,url or doi)", type=["csv"])

# ---------- Loaders ----------
@st.cache_data(show_spinner=True, ttl=3600)
def load_embeddings(b: bytes) -> pd.DataFrame:
    return pickle.load(io.BytesIO(b))  # expects a DataFrame with an 'embedding' column

@st.cache_data(show_spinner=False)
def load_meta(b: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(b))
    cols = {c.lower(): c for c in df.columns}
    if "pub_id" not in cols:
        raise ValueError("Metadata CSV must have a 'pub_id' column.")
    out = pd.DataFrame()
    out["pub_id"] = df[cols["pub_id"]].astype(str)
    if "title" in cols: out["title"] = df[cols["title"]].astype(str)
    if "url" in cols:   out["url"]   = df[cols["url"]].astype(str)
    if "doi" in cols:   out["doi"]   = df[cols["doi"]].astype(str)
    return out

def normalize_matrix(M: np.ndarray) -> np.ndarray:
    M = M.astype("float32")
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n

def embed_query(text: str) -> np.ndarray:
    if not OPENAI_KEY:
        st.warning("Add OPENAI_API_KEY in Settings → Secrets (or paste it above)."); st.stop()
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    v = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    v = np.asarray(v, dtype="float32")
    v /= max(np.linalg.norm(v), 1e-8)
    return v

SECTION_KEYS = [
    "abstract","introduction","background","methods","materials and methods",
    "data","results","findings","discussion","conclusion","limitations","appendix","supplementary"
]
def guess_section(text: str, row: pd.Series) -> str:
    if str(row.get("source_type")) == "table":
        ti = row.get("table_index")
        return f"Table {int(ti)}" if pd.notna(ti) else "Table"
    low = (text or "").lower()
    for key in SECTION_KEYS:
        if re.search(rf"\b{re.escape(key)}\b", low):
            return key.title()
    p = row.get("page")
    return f"Page {int(p)}" if pd.notna(p) else "Text"

# ---------- Power extractor (simple rules) ----------
NUM = r"[-+]?\d+(?:\.\d+)?"
PCT = r"[-+]?\d+(?:\.\d+)?\s*%"

def find_one(pattern: str, s: str):
    m = re.search(pattern, s, flags=re.I)
    return m.group(1).strip() if m else None

def extract_power_bits(s: str) -> dict:
    s = s or ""
    out = {}
    out["power"]   = find_one(r"\bpower(?:ed|)\s*(?:at|=|of|to)?\s*(%s)" % PCT, s) or \
                     find_one(r"\bpower(?:ed|)\s*(?:at|=|of|to)?\s*(%s)" % NUM, s)
    out["alpha"]   = find_one(r"\balpha\s*(?:=|at|:)\s*(%s)" % NUM, s)
    out["mde"]     = find_one(r"(?:minimum detectable effect|mde|mdes)\D*(%s)" % NUM, s)
    out["effect"]  = find_one(r"(?:effect size|cohen.?s d)\D*(%s)" % NUM, s)
    out["sd"]      = find_one(r"(?:standard deviation|sd|σ)\D*(%s)" % NUM, s)
    out["var"]     = find_one(r"(?:variance|var)\D*(%s)" % NUM, s)
    out["icc"]     = find_one(r"(?:intra[-\s]?cluster correlation|icc)\D*(%s)" % NUM, s)
    out["n_total"] = find_one(r"(?:sample size|total\s+N|N\s*=\s*)(%s)" % NUM, s)
    # clean percents like "80 %" -> "80%"
    if out["power"] and " %" in out["power"]: out["power"] = out["power"].replace(" %", "%")
    return {k:v for k,v in out.items() if v}

# ---------- Load data ----------
df = None; M = None; meta = None
if emb_file is not None:
    try:
        df = load_embeddings(emb_file.read())
        if "embedding" not in df.columns:
            st.error("No 'embedding' column in the uploaded embeddings file."); st.stop()
        M = np.vstack(df["embedding"].to_numpy()).astype("float32")
        M = normalize_matrix(M)
    except Exception as e:
        st.error(f"Could not read embeddings file: {e}")

if meta_file is not None:
    try:
        meta = load_meta(meta_file.read())
    except Exception as e:
        st.error(f"Could not read metadata CSV: {e}")

if df is not None and meta is not None and "pub_id" in df.columns:
    df["pub_id"] = df["pub_id"].astype(str)
    df = df.merge(meta, how="left", on="pub_id", suffixes=("","_meta"))

with st.expander("Show data preview"):
    if df is not None:
        cols = [c for c in ["pub_id","title","source_type","page","table_index","caption","text","url","doi"] if c in df.columns]
        st.write(f"Rows: {len(df):,}")
        st.dataframe(df.head(10)[cols] if cols else df.head(10), use_container_width=True)
    else:
        st.info("Upload your embeddings `.pkl` (and optionally metadata CSV) to enable search.")

# ---------- Query UI ----------
st.subheader("Search")
colA, colB, colC = st.columns([3,1,1])
with colA:
    q = st.text_input("Query (e.g., 'variance in student test scores')", "")
with colB:
    topk = st.number_input("Results", 1, 50, 10)
with colC:
    power_mode = st.checkbox("Power mode", value=True, help="Try to extract power inputs (power, alpha, MDE, SD, ICC, N).")

go = st.button("Search") or bool(q)

if go:
    if df is None or M is None:
        st.warning("Please upload your embeddings `.pkl` first."); st.stop()
    if not OPENAI_KEY:
        st.warning("Add OPENAI_API_KEY in Settings → Secrets (or paste it above)."); st.stop()

    qv = embed_query(q)
    sims = M @ qv
    out = df.copy()
    out["score"] = (sims * 100).round(1)
    out = out.sort_values("score", ascending=False).head(int(topk))

    st.subheader("Results")
    for _, row in out.iterrows():
        title = row.get("title")
        name  = title if isinstance(title, str) and title.strip() else str(row.get("pub_id"))
        section = guess_section(str(row.get("text") or ""), row)
        link = None
        if isinstance(row.get("url"), str) and row["url"].startswith(("http://","https://")):
            link = row["url"]
        elif isinstance(row.get("doi"), str) and row["doi"]:
            doi = row["doi"].strip()
            link = doi if doi.startswith("http") else f"https://doi.org/{doi}"

        # Card header
        st.markdown(f"### {name}")
        badgeline = f"`{section}` • `{row.get('source_type','text')}` • **Score: {row['score']:.1f}**"
        if link: badgeline += f" • [Open link]({link})"
        st.markdown(badgeline)

        # Two columns: power chips (right) + clean excerpt (left)
        c1, c2 = st.columns([2,1])
        with c1:
            txt = str(row.get("text") or "")
            st.markdown("**Excerpt:**")
            st.write(txt[:1200] + ("…" if len(txt) > 1200 else ""))

            cap = row.get("caption")
            if isinstance(cap, str) and cap.strip():
                st.caption(f"Caption: {cap}")

        with c2:
            if power_mode:
                bits = extract_power_bits(row.get("text") or "")
                st.markdown("**Power inputs (detected):**")
                if bits:
                    for k,v in bits.items():
                        st.markdown(f"- **{k}**: {v}")
                else:
                    st.write("— none found in this chunk —")

        st.divider()

    st.caption("Scores are cosine similarity × 100 (higher = more relevant).")
