import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

# === YOUR LIVE SHEET CSV (use this exact link) ===
CSV_URL = "https://docs.google.com/spreadsheets/d/1qUQ-Hvpu5hCTLcRNMISHXs4CpD2y_LKeVUoR-YLHTug/export?format=csv&gid=0"

st.set_page_config(page_title="Power Inputs Search", layout="wide")
st.title("Power Inputs Search (Live Google Sheet)")

EXPECTED = [
    "study_id","title","authors","year","sector","intervention_type","country",
    "outcome_variable","effect_size_type","effect_size_value","variance","sd",
    "se","icc","sample_size_total","doi_of_study","qc_status","qc_issues"
]

def read_sheet():
    df = pd.read_csv(CSV_URL)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        st.error(f"Missing columns in sheet: {missing}")
        st.stop()
    return df

# ---- quick preview to verify the link works ----
if st.checkbox("Show data preview"):
    try:
        st.dataframe(read_sheet().head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read the CSV link: {e}")

def make_blob(row):
    parts = [
        row.get("title",""), row.get("sector",""), row.get("intervention_type",""),
        row.get("country",""), row.get("outcome_variable",""), row.get("effect_size_type","")
    ]
    return " | ".join([str(p) for p in parts if pd.notna(p) and str(p).strip()])

q = st.text_input("Search (e.g., 'malnourished children in Haryana')", "")
verified_only = st.toggle("Verified-only", value=True)
topk = st.number_input("Results", 1, 50, 10)

if st.button("Search") or q:
    df = read_sheet()
    if verified_only:
        df = df[df["qc_status"].astype(str).str.lower() == "verified"].copy()

    if df.empty:
        st.info("No rows available. Check your sheet or QC status.")
    elif not q.strip():
        st.info("Type something to search.")
    else:
        df["__blob"] = df.apply(make_blob, axis=1)
        df["score"]  = df["__blob"].apply(lambda t: fuzz.token_set_ratio(q, t) / 100.0)
        out = df.sort_values("score", ascending=False).head(int(topk)).copy()

        def doi_link(v):
            v = str(v or "").strip()
            return v if v.startswith("http") else (f"https://doi.org/{v}" if v else "")

        out["doi_link"] = out["doi_of_study"].apply(doi_link)

        cols = [
            "study_id","title","country","sector","intervention_type","outcome_variable",
            "effect_size_type","effect_size_value","sd","variance","icc","sample_size_total",
            "qc_status","score","doi_link"
        ]
        for c in cols:
            if c not in out.columns: out[c] = ""
        st.dataframe(out[cols], use_container_width=True)
