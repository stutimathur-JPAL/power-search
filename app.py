# --- Minimal Streamlit app with an Upload box for your embeddings file ---
import io, pickle, requests, numpy as np, pandas as pd, streamlit as st

# Your file was built with text-embedding-3-large (3072-dim). Keep this to match.
EMBED_MODEL = "text-embedding-3-large"

# Read your OpenAI key from Streamlit Secrets (add it once in Manage app → Settings → Secrets)
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")

st.set_page_config(page_title="Paper Search", layout="wide")
st.title("Paper Search")

# --- Allow pasting a key if Secrets aren't set ---
if not OPENAI_KEY:
    with st.expander("🔐 Add your OpenAI key (or set it in Settings → Secrets)"):
        pasted = st.text_input("OpenAI API key", type="password")
        if pasted:
            st.session_state["OPENAI_API_KEY"] = pasted
            OPENAI_KEY = pasted

# --- Upload UI ---
st.subheader("Data source")
uploaded = st.file_uploader("Upload your embeddings file (.pkl)", type=["pkl"])

@st.cache_data(show_spinner=True, ttl=3600)
def load_df_from_bytes(b: bytes) -> pd.DataFrame:
    return pickle.load(io.BytesIO(b))

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

df = None
M = None
if uploaded is not None:
    try:
        df = load_df_from_bytes(uploaded.read())
        if "embedding" not in df.columns:
            st.error("No 'embedding' column found in the uploaded file."); st.stop()
        M = np.vstack(df["embedding"].to_numpy()).astype("float32")
        M = normalize_matrix(M)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")

with st.expander("Show data preview"):
    if df is not None:
        cols = [c for c in ["pub_id","source_type","page","table_index","caption","text"] if c in df.columns]
        st.write(f"Rows: {len(df):,}")
        st.dataframe(df.head(10)[cols] if cols else df.head(10), use_container_width=True)
    else:
        st.info("Upload your `.pkl` to enable search.")

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

    keep = ["score","pub_id","source_type","page","table_index","caption","text"]
    for c in keep:
        if c not in out.columns: out[c] = None
    out = out.sort_values("score", ascending=False).head(int(topk))

    st.subheader("Results")
    for _, row in out.iterrows():
        header = f"{row['score']:>5.1f} | {row['pub_id']} | {row['source_type']}"
        if pd.notna(row.get("page")): header += f" | page {int(row['page'])}"
        if pd.notna(row.get("table_index")): header += f" | table {int(row['table_index'])}"
        with st.expander(header):
            cap = row.get("caption")
            if isinstance(cap, str) and cap.strip():
                st.markdown(f"**Caption:** {cap}")
            txt = str(row.get("text") or "")
            st.markdown("**Excerpt:**")
            st.write(txt[:1500] + ("…" if len(txt) > 1500 else ""))
    st.caption("Scores are cosine similarity × 100 (higher = more relevant).")
