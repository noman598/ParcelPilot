"""
The three required tools, plus the access-control enforcement.

IMPORTANT (access control design):
Every function that touches structured customer data takes the caller's
`account_id` as a normal Python argument supplied by the SERVER (bound
from the logged-in session), never as a field the model fills in. The
tool *schemas* exposed to Claude do not include account_id at all for
this reason — the model literally cannot ask to see another account's
data through this interface, regardless of what it's told or tricked
into asking.
"""
import json
import os
import uuid
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import ingest

ESCALATIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "escalations.json")

# If your xlsx columns differ from these, edit the right-hand side to match
# your actual column names (after lowercasing, spaces->underscores).
COLUMN_MAP = {
    "accounts": {
        "account_id": "account_id",
        "account_name": "account_name",
    },
    "orders": {
        "order_id": "order_id",
        "account_id": "account_id",
    },
    "tickets": {
        "ticket_id": "ticket_id",
        "account_id": "account_id",
        "order_id": "order_id",
    },
}

import math

def _clean_nans(obj):
    """Recursively replace NaN/NaT with None so JSON serialization doesn't crash."""
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nans(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj

class ParcelPilotData:
    """Loads documents + structured data once and exposes search/query methods."""

    def __init__(self):
        self.chunks = ingest.load_documents()
        self.structured = ingest.load_structured_data()

        self._vectorizer = None
        self._doc_matrix = None
        if self.chunks:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._doc_matrix = self._vectorizer.fit_transform(
                [c["text"] for c in self.chunks]
            )

    # ---------- Tool 1: document search ----------
    def document_search(self, query: str, k: int = 5, account_scope: str | None = None):
        if not self.chunks:
            return {"results": [], "note": "No documents loaded."}

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._doc_matrix)[0]
        ranked_idx = sims.argsort()[::-1]

        results = []
        for i in ranked_idx:
            if sims[i] <= 0:
                continue
            chunk = self.chunks[i]
            # Only include general-scope docs, or docs scoped to this account
            if chunk["scope"] != "general" and chunk["scope"] != account_scope:
                continue
            results.append(
                {
                    "source_file": chunk["source_file"],
                    "doc_type": chunk["doc_type"],
                    "status": chunk["status"],
                    "authority": chunk["authority"],
                    "scope": chunk["scope"],
                    "excerpt": chunk["text"][:800],
                    "relevance": round(float(sims[i]), 3),
                }
            )
            if len(results) >= k:
                break
        return {"results": results}

    # ---------- Tool 2: structured data lookup / calculation ----------
    def structured_data_lookup(self, account_id: str, query_type: str, order_id: str = None, ticket_id: str = None):
        """
        query_type one of: "account_summary", "order_lookup", "ticket_lookup",
        "list_orders", "list_tickets".
        account_id is ALWAYS the session's authenticated account_id — the
        caller (agent.py) enforces this is never overridden by the model.
        """
        acc_col = COLUMN_MAP["accounts"]["account_id"]
        ord_acc_col = COLUMN_MAP["orders"]["account_id"]
        tick_acc_col = COLUMN_MAP["tickets"]["account_id"]
        ord_id_col = COLUMN_MAP["orders"]["order_id"]
        tick_id_col = COLUMN_MAP["tickets"]["ticket_id"]

        accounts = self.structured["accounts"]
        orders = self.structured["orders"]
        tickets = self.structured["tickets"]

        # scope everything to this account_id no matter what
        my_orders = orders[orders.get(ord_acc_col) == account_id] if ord_acc_col in orders.columns else pd.DataFrame()
        my_tickets = tickets[tickets.get(tick_acc_col) == account_id] if tick_acc_col in tickets.columns else pd.DataFrame()
        my_account = accounts[accounts.get(acc_col) == account_id] if acc_col in accounts.columns else pd.DataFrame()

        if query_type == "account_summary":
            # return {"account": my_account.to_dict(orient="records")}
            return _clean_nans({"account": my_account.to_dict(orient="records")})

        if query_type == "order_lookup":
            if order_id is None:
                return {"error": "order_id required for order_lookup"}
            row = my_orders[my_orders.get(ord_id_col) == order_id]
            if row.empty:
                return {"error": f"Order {order_id} not found for this account."}
            # return {"order": row.to_dict(orient="records")[0]}
            return _clean_nans({"order": row.to_dict(orient="records")[0]})

        if query_type == "ticket_lookup":
            if ticket_id is None:
                return {"error": "ticket_id required for ticket_lookup"}
            row = my_tickets[my_tickets.get(tick_id_col) == ticket_id]
            if row.empty:
                return {"error": f"Ticket {ticket_id} not found for this account."}
            # return {"ticket": row.to_dict(orient="records")[0]}
            return _clean_nans({"ticket": row.to_dict(orient="records")[0]})

        if query_type == "list_orders":
            # return {"orders": my_orders.to_dict(orient="records")}
            return _clean_nans({"orders": my_orders.to_dict(orient="records")})

        if query_type == "list_tickets":
            # return {"tickets": my_tickets.to_dict(orient="records")}
            return _clean_nans({"tickets": my_tickets.to_dict(orient="records")})

        return {"error": f"Unknown query_type: {query_type}"}

    # ---------- Tool 3: state-changing action (mocked, needs confirmation) ----------
    def create_escalation(self, account_id: str, summary: str, reason: str, related_order_id: str = None, related_ticket_id: str = None):
        """Actually writes the escalation. Only ever called AFTER user confirmation
        (see agent.py's confirmation gate) — never directly from a raw model tool call."""
        record = {
            "escalation_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
            "account_id": account_id,
            "summary": summary,
            "reason": reason,
            "related_order_id": related_order_id,
            "related_ticket_id": related_ticket_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "status": "open",
        }
        existing = []
        if os.path.exists(ESCALATIONS_PATH):
            with open(ESCALATIONS_PATH) as f:
                existing = json.load(f)
        existing.append(record)
        with open(ESCALATIONS_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        return record


def account_scope_key(account_name: str) -> str:
    """Maps an account_name to the scope key used in DOCUMENT_FILES, e.g.
    'Northstar Logistics' -> 'account:northstar'. Extend this if you add
    more customer-specific agreements."""
    name = (account_name or "").lower()
    if "northstar" in name:
        return "account:northstar"
    if "lumenworks" in name or "lumen works" in name:
        return "account:lumenworks"
    return "account:none"
