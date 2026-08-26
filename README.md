# ParcelPilot Customer Support Chatbot

## Architecture Note

The chatbot is designed around four main principles:

- **Secure customer data access**
- **Reliable source selection**
- **Human confirmation for state-changing actions**
- **Escalation instead of guessing when information is uncertain**

---

## Customer Data Access

### How did you make sure customers can only see their own account's data?

The `account_id` is bound to the **session at login time on the server**, not passed by the model.

It is never a field the LLM can set in its tool call - so even if the model "hallucinates" a different `account_id`, the underlying query is still filtered to the **logged-in customer's own data**.

> **Key idea:** The LLM is not responsible for access control. The server is.

---

## Source Reliability

### How does the agent decide which source to trust when documents conflict?

Every document chunk is tagged with **status** and **authority** at ingestion time.

The system prompt instructs the model to:

1. Prefer **current** documents over deprecated documents.
2. Prefer **customer-specific agreements** over general policy whenever they conflict.
3. Treat historical ticket resolutions as **context only**, not as policy.
4. Escalate when conflicting information cannot be confidently resolved.

> **Key idea:** The chatbot does not blindly trust the first document it finds.

---

## Escalation Handling

### What happens when an escalation scenario comes?

The graph pauses execution the moment the model calls the escalation tool and only resumes to actually write the record after the user clicks **Confirm** in the UI.

The escalation is **not created immediately**.

The flow is:

**Customer Request → Escalation Proposed → Customer Confirmation → Escalation Created**

If the customer declines, nothing is created.

> **Key idea:** Nothing is created on the model's word alone.

---

## Handling Uncertain Questions

### What happens when the chatbot can't answer confidently from the data?

It's instructed to **say so and propose an escalation rather than guess**, so uncertain or out-of-policy questions get routed to a human instead of a fabricated answer.

> **Key idea:** When reliable information is not available, the chatbot prefers escalation over hallucination.

---

# System Architecture

<img width="700" height="700" alt="calib" src="https://github.com/user-attachments/assets/dea19840-88c0-469e-bbdb-9d28479c3a64" />

---

# SetUp

## 1. Put your data files in `data/`
### About the xlsx schema
 
`backend/tools.py` expects sheets roughly named `Accounts`, `Orders`,
and `Tickets` (case-insensitive, fuzzy-matched) with columns such as
`account_id`, `order_id`, `ticket_id`.

## 2. Install dependencies
 
```bash
cd parcelpilot-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
 
## 3. Set your API key
 
```bash
cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY
```
 
## 4. Run
 
```bash
uvicorn backend.app:app --reload --port 8000
```
 
Then open `frontend/index.html` directly in your browser (or serve it
with `python -m http.server 5500` from the `frontend/` folder and visit
`http://localhost:5500`).

# Tool Design

The agent uses three focused tools:

### `document_search`

Searches **policies, SOPs, operational guides, and customer-specific agreements.**

### `structured_data_lookup`

Retrieves the **authenticated customer's account, order, and ticket information.**

### `create_escalation`

Proposes a **support escalation rather than immediately creating one.**

The tools are intentionally kept focused so that the agent has clear boundaries around what each tool can access and what each tool is responsible for.

> **Key idea:** Each tool has a specific responsibility, while access control remains enforced by the backend.

---

# Document and Structured-Data Handling

## Document Handling

Documents are loaded at **application startup** and converted into searchable chunks.

A **TF-IDF vectorizer with cosine similarity** is used for lightweight document retrieval.

Each document chunk contains metadata such as:

- Source
- Document type
- Status
- Authority
- Scope

This metadata is used by the agent when deciding which information to trust.

## Structured-Data Handling

Customer account, order, and ticket information is handled separately from document retrieval.

The structured data is loaded into **Pandas DataFrames** for the demo.

Structured-data queries are always filtered using the **authenticated customer's `account_id`**.

> **Key idea:** Documents answer policy-related questions, while structured data answers customer-specific questions.

---

# Major Technical Trade-offs, why choose over this?

The implementation intentionally favors **simplicity and speed of development** because this is a working demonstration rather than a production system.

## 1. TF-IDF instead of a Vector Database

**Why choose it for the demo?**

Reduces infrastructure and setup complexity, but provides less semantic retrieval quality than modern embedding-based vector search.

**Better for production:**  
Use an **embedding model with a vector database** such as **pgvector** or **Pinecone** for better semantic retrieval and scalable document search.

> **Trade-off:** Simpler setup now vs. better semantic retrieval and scalability in production.

---

## 2. In-Memory Sessions

**Why choose it for the demo?**

Simple and fast for a demo.

**Better for production:**  
Use **Redis** or a persistent database for session/state storage so sessions survive restarts and can be shared across multiple application instances.

> **Trade-off:** Simplicity for the demo vs. persistence and scalability in production.

---

## 3. LLM Tool Calling

**Why choose it?**

Provides **flexible reasoning and tool selection**, but introduces additional model calls and token consumption.

**Better for production:**  
Use **LangGraph**, with strict limits on tool calls, retries, and execution steps to prevent unnecessary token usage and loops.

> **Trade-off:** Flexible agent behavior vs. tighter control over execution and token usage.

---

## 4. Pandas for Structured Data

**Why choose it for the demo?**

Makes the demo easy to implement and inspect but not good for high volume data.

**Better for production:**  
Use a **relational database such as PostgreSQL** with proper indexing.

> **Trade-off:** Easy data handling for the demo vs. scalable and persistent data storage in production.

---

## Why these trade-offs?

The goal of this project was to demonstrate the **core agent architecture, tool usage, data isolation, source reliability, and human-in-the-loop safety** without introducing unnecessary infrastructure.

The production alternatives can be introduced later without fundamentally changing the core design.
