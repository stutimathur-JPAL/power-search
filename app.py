# --- Streamlit app: upload embeddings + optional metadata; show title & section ---
import io, pickle, re
import numpy as np
import pandas as pd
import streamlit as st

EMBED_MODEL = "text-embedding-3-large"

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")

st.set_page_config(page_title="Paper Search", layout="wide")
st.title("Paper Search")

# Allow pasting a key if Secrets aren't set
if not OPENAI_KEY:
    with st.expander("🔐 Add your OpenAI key (or set it in Settings → Secrets)"):
        pasted = st.text_input("OpenAI API key", type="password")
        if pasted:
            st.session_state["OPENAI_API_KEY"] = pasted
            OPENAI_KEY = pasted

st.subheader("Data source")
emb_file = st.file_uploader("Upload embeddings file (.pkl)", type=["pkl"])
meta_file = st.file_uploader("Optional: upload metadata CSV (columns: pub_id,title,url or doi)", type=["csv"])

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

if df is not None and meta is not None:
    if "pub_id" in df.columns:
        df["pub_id"] = df["pub_id"].astype(str)
        df = df.merge(meta, how="left", on="pub_id", suffixes=("","_meta"))
    else:
        st.warning("Embeddings file has no 'pub_id' column; titles can’t be matched.")

with st.expander("Show data preview"):
    if df is not None:
        cols = [c for c in ["pub_id","title","source_type","page","table_index","caption","text","url","doi"] if c in df.columns]
        st.write(f"Rows: {len(df):,}")
        st.dataframe(df.head(10)[cols] if cols else df.head(10), use_container_width=True)
    else:
        st.info("Upload your embeddings `.pkl` (and optionally a metadata CSV) to enable search.")

st.subheader("Search")
q = st.text_input("Ask (e.g., 'variance in student test scores')", "")
topk = st.number_input("Results", 1, 50, 10)
go = st.button("Search") or bool(q)

if go:
    if df is None or M is None:
        st.warning("Please upload your embeddings `.pkl` above first."); st.stop()
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
        name = title if isinstance(title, str) and title.strip() else str(row.get("pub_id"))
        section = guess_section(str(row.get("text") or ""), row)
        header = f"{row['score']:>5.1f} | {name} | {section}"

        with st.expander(header):
            link = None
            if isinstance(row.get("url"), str) and row["url"].startswith(("http://","https://")):
                link = row["url"]
            elif isinstance(row.get("doi"), str) and row["doi"]:
                doi = row["doi"].strip()
                link = doi if doi.startswith("http") else f"https://doi.org/{doi}"
            if link:
                st.markdown(f"**Link:** {link}")

            cap = row.get("caption")
            if isinstance(cap, str) and cap.strip():
                st.markdown(f"**Caption:** {cap}")

            txt = str(row.get("text") or "")
            st.markdown("**Excerpt:**")
            st.write(txt[:1500] + ("…" if len(txt) > 1500 else ""))

    st.caption("Scores are cosine similarity × 100 (higher = more relevant).")
