import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .tools import ParcelPilotData, account_scope_key, COLUMN_MAP

app = FastAPI(title="ParcelPilot Support Chatbot (Demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup.
data_store = ParcelPilotData()

# In-memory session store: session_id -> {account_id, account_name, history, pending_action}
SESSIONS: dict = {}


@app.get("/accounts")
def list_accounts():
    """Powers the mock login dropdown."""
    acc_col = COLUMN_MAP["accounts"]["account_id"]
    name_col = COLUMN_MAP["accounts"]["account_name"]
    df = data_store.structured["accounts"]
    if df.empty:
        return {"accounts": []}
    cols = [c for c in (acc_col, name_col) if c in df.columns]
    return {"accounts": df[cols].drop_duplicates().to_dict(orient="records")}


class LoginRequest(BaseModel):
    account_id: str
    account_name: str = ""


@app.post("/login")
def login(req: LoginRequest):
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "account_id": req.account_id,
        "account_scope": account_scope_key(req.account_name),
        "history": [],
        "pending_action": None,
    }
    return {"session_id": session_id}


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    from .agent import run_agent_turn  # local import to avoid circular import at module load

    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session. Please log in again.")

    session["history"].append({"role": "user", "content": req.message})

    history, tool_trace, pending_action, final_text = run_agent_turn(
        data_store=data_store,
        account_id=session["account_id"],
        account_scope=session["account_scope"],
        history=session["history"],
    )

    session["history"] = history
    session["pending_action"] = pending_action

    return {
        "reply": final_text,
        "tool_trace": tool_trace,
        "pending_action": pending_action,
    }


class ConfirmRequest(BaseModel):
    session_id: str
    confirm: bool


@app.post("/confirm_action")
def confirm_action(req: ConfirmRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session. Please log in again.")

    action = session.get("pending_action")
    if not action:
        raise HTTPException(status_code=400, detail="No pending action for this session.")

    if not req.confirm:
        session["pending_action"] = None
        session["history"].append(
            {"role": "user", "content": "(The customer declined to confirm this action.)"}
        )
        return {"status": "cancelled"}

    # Execute the actual mocked action now — and only now.
    result = data_store.create_escalation(
        account_id=action["account_id"],
        summary=action["summary"],
        reason=action["reason"],
        related_order_id=action.get("related_order_id"),
        related_ticket_id=action.get("related_ticket_id"),
    )
    session["pending_action"] = None
    session["history"].append(
        {
            "role": "user",
            "content": f"(The customer confirmed. Escalation {result['escalation_id']} was created.)",
        }
    )
    return {"status": "created", "escalation": result}




# ---------- Serve the frontend (single container for Hugging Face Spaces) ----------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))