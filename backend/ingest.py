"""
Loads the ParcelPilot data pack: extracts + chunks PDF text (tagging each
chunk with source-reliability metadata), and loads the xlsx sheets into
pandas DataFrames.

This runs once at server startup and the results are cached in memory —
fine for a demo, not for production scale.
"""
import os
import pdfplumber
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Filename -> reliability metadata. Adjust here if your filenames differ.
# authority: higher number = should be trusted more when sources conflict.
# scope: "general" applies to all customers; "account:<name>" overrides
#        general policy for that specific customer only.
DOCUMENT_FILES = [
    {
        "file": "01_Support_Policy_v3_CURRENT.pdf",
        "doc_type": "policy",
        "status": "current",
        "authority": 3,
        "scope": "general",
    },
    {
        "file": "02_Support_Policy_v2_DEPRECATED.pdf",
        "doc_type": "policy",
        "status": "deprecated",
        "authority": 0,  # never prefer this over the current policy
        "scope": "general",
    },
    {
        "file": "03_Cancellation_and_Service_Credit_SOP_v4.pdf",
        "doc_type": "sop",
        "status": "current",
        "authority": 3,
        "scope": "general",
    },
    {
        "file": "04_Product_Operations_Guide_and_Known_Issues.pdf",
        "doc_type": "product_ops_guide",
        "status": "current",
        "authority": 2,
        "scope": "general",
    },
    {
        "file": "05_Northstar_Logistics_Enterprise_Agreement.pdf",
        "doc_type": "customer_agreement",
        "status": "current",
        "authority": 4,  # customer agreement overrides general policy
        "scope": "account:northstar",
    },
    {
        "file": "06_LumenWorks_Service_Agreement.pdf",
        "doc_type": "customer_agreement",
        "status": "current",
        "authority": 4,
        "scope": "account:lumenworks",
    },
]

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40


def _extract_pdf_text(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts)


def _chunk_text(text: str, source_meta: dict):
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i : i + CHUNK_WORDS]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append(
                {
                    "chunk_id": f"{source_meta['file']}::{idx}",
                    "text": chunk_text,
                    **{k: v for k, v in source_meta.items() if k != "file"},
                    "source_file": source_meta["file"],
                }
            )
            idx += 1
        i += CHUNK_WORDS - CHUNK_OVERLAP
    return chunks


def load_documents():
    """Returns a flat list of chunk dicts across all documents found."""
    all_chunks = []
    for meta in DOCUMENT_FILES:
        path = os.path.join(DATA_DIR, meta["file"])
        if not os.path.exists(path):
            print(f"[ingest] WARNING: missing {meta['file']}, skipping.")
            continue
        text = _extract_pdf_text(path)
        all_chunks.extend(_chunk_text(text, meta))
    print(f"[ingest] Loaded {len(all_chunks)} chunks from {len(DOCUMENT_FILES)} documents.")
    return all_chunks


def _find_sheet(xls: pd.ExcelFile, candidates):
    """Fuzzy-match a sheet name from a list of likely candidates."""
    lower_map = {s.lower(): s for s in xls.sheet_names}
    for cand in candidates:
        for lower_name, actual in lower_map.items():
            if cand in lower_name:
                return actual
    return None


def load_structured_data():
    """Loads Accounts / Orders / Tickets sheets into DataFrames."""
    path = os.path.join(DATA_DIR, "ParcelPilot_Assessment_Data.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {path} — put the assessment xlsx in the data/ folder."
        )
    xls = pd.ExcelFile(path)

    accounts_sheet = _find_sheet(xls, ["account"])
    orders_sheet = _find_sheet(xls, ["order"])
    tickets_sheet = _find_sheet(xls, ["ticket"])
    readme_sheet = _find_sheet(xls, ["readme", "read_me", "read me"])

    accounts_df = pd.read_excel(xls, accounts_sheet) if accounts_sheet else pd.DataFrame()
    orders_df = pd.read_excel(xls, orders_sheet) if orders_sheet else pd.DataFrame()
    tickets_df = pd.read_excel(xls, tickets_sheet) if tickets_sheet else pd.DataFrame()
    readme_df = pd.read_excel(xls, readme_sheet) if readme_sheet else pd.DataFrame()

    # normalize column names: lowercase, strip, spaces -> underscores
    for df in (accounts_df, orders_df, tickets_df):
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    print(
        f"[ingest] Loaded sheets -> accounts:{accounts_sheet} "
        f"orders:{orders_sheet} tickets:{tickets_sheet} readme:{readme_sheet}"
    )

    return {
        "accounts": accounts_df,
        "orders": orders_df,
        "tickets": tickets_df,
        "readme": readme_df,
    }
