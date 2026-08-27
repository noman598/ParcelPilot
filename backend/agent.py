"""
The agent loop.

Groq is given three tools:
- document_search
- structured_data_lookup
- create_escalation

The first two execute immediately.

create_escalation is intercepted — instead of actually executing it,
we return a pending_action to the frontend and only call
ParcelPilotData.create_escalation once the user clicks Confirm.
"""

import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

MODEL = "qwen/qwen3.6-27b"
# MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are ParcelPilot's customer support assistant.

You only answer using information returned by your tools — never invent
policy, order, or account details.

You are talking to an authenticated customer. You can only ever see and
discuss THIS customer's own account, orders, and tickets.

The tools are already scoped to their account. You never need to and
cannot ask for or supply an account_id yourself.

SOURCE RELIABILITY RULES:

- Each document result has a "status" (current/deprecated) and "authority"
  score. NEVER rely on a deprecated document if a current one covers the
  same topic.

- A customer-specific agreement (scope starting with "account:")
  OVERRIDES general policy for that customer wherever they conflict.

- Always check for a customer agreement first on questions about fees,
  cancellation terms, or SLAs.

- Ticket data / past resolutions are historical CONTEXT ONLY.
  They may contain incorrect guidance and must never be treated as policy.

- Do not present a past ticket's resolution as the current correct answer.
  Verify against policy/SOP/agreement documents instead.

- If sources conflict and you can't confidently resolve it, or the question
  requires human judgment / an exception outside documented policy, say so
  plainly and escalate rather than guessing.

ESCALATION:

- If a request needs a human, propose creating an escalation and explain why.

- Never create an escalation silently.

- The system will always ask the customer to confirm before the escalation
  is actually created.

- If the user declines, do not create it.

STYLE:

- Be direct and concise.

- When you give an answer that depends on a specific policy/SOP/agreement,
  briefly cite which document it came from.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "document_search",
            "description": (
                "Search ParcelPilot's policies, SOPs, product/operations "
                "guide, and (if applicable) this customer's specific "
                "agreement. Returns ranked excerpts tagged with source, "
                "status (current/deprecated), and authority level."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return.",
                        "default": 5
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "structured_data_lookup",
            "description": (
                "Look up or list this customer's own account, order, or "
                "ticket records. Always automatically scoped to the "
                "logged-in customer's account — never provide account_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "account_summary",
                            "order_lookup",
                            "ticket_lookup",
                            "list_orders",
                            "list_tickets",
                        ],
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Required for order_lookup."
                    },
                    "ticket_id": {
                        "type": "string",
                        "description": "Required for ticket_lookup."
                    },
                },
                "required": ["query_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_escalation",
            "description": (
                "Propose creating a support escalation for this customer. "
                "This does NOT immediately create anything. It will be "
                "shown to the customer as a proposed action requiring "
                "explicit confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the escalation."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this needs human attention."
                    },
                    "related_order_id": {
                        "type": "string"
                    },
                    "related_ticket_id": {
                        "type": "string"
                    },
                },
                "required": ["summary", "reason"],
            },
        },
    },
]


# def run_agent_turn(
#     data_store,
#     account_id: str,
#     account_scope: str,
#     history: list,
# ):
#     """
#     Runs one full turn, potentially involving several tool calls.

#     Returns:
#         history
#         tool_trace
#         pending_action
#         final_text
#     """

#     tool_trace = []
#     pending_action = None
#     called_tools = set()

#     MAX_ITERATIONS = 2

#     # while True:
#     for iteration in range(MAX_ITERATIONS):
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": SYSTEM_PROMPT,
#                 },
#                 ,
#             ]+history,
#             tools=TOOLS,
#             # tool_choice="auto",
#             # max_tokens=1500,
#         )

#         message = response.choices[0].message
#         history.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})

#         # ---------------------------------------------------------
#         # No tool call -> final answer
#         # ---------------------------------------------------------

#         if not message.tool_calls:

#             # final_text = message.content or ""

#             # history.append(
#             #     {
#             #         "role": "assistant",
#             #         "content": final_text,
#             #     }
#             # )

#             return (
#                 history,
#                 tool_trace,
#                 pending_action,
#                 final_text,
#             )

#         # ---------------------------------------------------------
#         # Store assistant's tool-call message
#         # ---------------------------------------------------------

#         assistant_message = {
#             "role": "assistant",
#             "content": message.content or "",
#             "tool_calls": [],
#         }

#         for tool_call in message.tool_calls:

#             assistant_message["tool_calls"].append(
#                 {
#                     "id": tool_call.id,
#                     "type": "function",
#                     "function": {
#                         "name": tool_call.function.name,
#                         "arguments": tool_call.function.arguments,
#                     },
#                 }
#             )

#         history.append(assistant_message)

#         # ---------------------------------------------------------
#         # Execute tools
#         # ---------------------------------------------------------

#         for tool_call in message.tool_calls:

#             tool_name = tool_call.function.name

#             try:
#                 tool_input = json.loads(
#                     tool_call.function.arguments
#                 )
#             except json.JSONDecodeError:
#                 tool_input = {}

#             # -----------------------------------------------------
#             # document_search
#             # -----------------------------------------------------

#             if tool_name == "document_search":

#                 output = data_store.document_search(
#                     query=tool_input["query"],
#                     k=tool_input.get("k", 5),
#                     account_scope=account_scope,
#                 )

#             # -----------------------------------------------------
#             # structured_data_lookup
#             # -----------------------------------------------------

#             elif tool_name == "structured_data_lookup":

#                 # IMPORTANT:
#                 # account_id comes from the authenticated session,
#                 # NOT from the LLM.

#                 output = data_store.structured_data_lookup(
#                     account_id=account_id,
#                     query_type=tool_input["query_type"],
#                     order_id=tool_input.get("order_id"),
#                     ticket_id=tool_input.get("ticket_id"),
#                 )

#             # -----------------------------------------------------
#             # create_escalation
#             # -----------------------------------------------------

#             elif tool_name == "create_escalation":

#                 # DO NOT ACTUALLY CREATE THE ESCALATION.

#                 pending_action = {
#                     "tool": "create_escalation",
#                     "account_id": account_id,
#                     "summary": tool_input.get("summary"),
#                     "reason": tool_input.get("reason"),
#                     "related_order_id": tool_input.get(
#                         "related_order_id"
#                     ),
#                     "related_ticket_id": tool_input.get(
#                         "related_ticket_id"
#                     ),
#                 }

#                 output = {
#                     "status": "pending_confirmation",
#                     "note": (
#                         "Not created yet. Waiting for explicit "
#                         "customer confirmation."
#                     ),
#                 }

#             # -----------------------------------------------------
#             # Unknown tool
#             # -----------------------------------------------------

#             else:

#                 output = {
#                     "error": f"Unknown tool {tool_name}"
#                 }

#             # -----------------------------------------------------
#             # Store trace
#             # -----------------------------------------------------

#             tool_trace.append(
#                 {
#                     "tool": tool_name,
#                     "input": tool_input,
#                     "output": output,
#                 }
#             )

#             # -----------------------------------------------------
#             # Add tool result to conversation
#             # -----------------------------------------------------

#             history.append(
#                 {
#                     "role": "tool",
#                     "tool_call_id": tool_call.id,
#                     "content": json.dumps(
#                         output,
#                         default=str,
#                     ),
#                 }
#             )

#             # -----------------------------------------------------
#             # Stop immediately if escalation is pending
#             # -----------------------------------------------------

#             if pending_action is not None:

#                 final_text = (
#                     message.content
#                     or "I've prepared this action — please confirm "
#                        "below before I create it."
#                 )

#                 return (
#                     history,
#                     tool_trace,
#                     pending_action,
#                     final_text,
#                 )
def run_agent_turn(data_store, account_id: str, account_scope: str, history: list):
    """
    Runs one full turn (which may involve several tool calls) and returns:
      - updated `history`
      - `tool_trace`: list of {"tool": name, "input": ..., "output": ...} for the UI
      - `pending_action`: dict if create_escalation was requested, else None
      - `final_text`: the assistant's final natural-language reply
    """
    tool_trace = []
    pending_action = None
    called_tools = set()

    MAX_ITERATIONS = 3

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
        )

        message = response.choices[0].message

        # No tool call -> build the assistant message WITHOUT a tool_calls
        # key at all (Groq rejects tool_calls: null explicitly).
        # if not message.tool_calls:
        #     final_text = message.content or ""
        #     history.append({"role": "assistant", "content": final_text})
        #     print(f"[DEBUG] returning final_text={final_text!r}, pending_action={pending_action}")
        #     return history, tool_trace, pending_action, final_text

        if not message.tool_calls:
            final_text = message.content
            if not final_text:
                # Model returned no answer — likely because this needs human judgment.
                pending_action = {
                    "tool": "create_escalation",
                    "account_id": account_id,
                    "summary": "Customer request needs review — no direct answer available.",
                    "reason": "The model could not produce a definitive answer from available tools/documents.",
                    "related_order_id": None,
                    "related_ticket_id": None,
                }
                final_text = "I'm not confident I can resolve this from our policies — I've prepared an escalation for you to confirm."
            history.append({"role": "assistant", "content": final_text})
            return history, tool_trace, pending_action, final_text
        # Has tool calls -> store assistant message with tool_calls included.
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        }
        history.append(assistant_message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_input = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            # Block repeated use of the same tool this turn.
            if tool_name in called_tools:
                output = {
                    "error": f"{tool_name} was already used this turn. "
                             "Use a different tool or give your final answer."
                }
                tool_trace.append({"tool": tool_name, "input": tool_input, "output": output})
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(output, default=str),
                })
                continue

            called_tools.add(tool_name)

            if tool_name == "document_search":
                output = data_store.document_search(
                    query=tool_input["query"],
                    k=tool_input.get("k", 5),
                    account_scope=account_scope,
                )

            elif tool_name == "structured_data_lookup":
                output = data_store.structured_data_lookup(
                    account_id=account_id,
                    query_type=tool_input["query_type"],
                    order_id=tool_input.get("order_id"),
                    ticket_id=tool_input.get("ticket_id"),
                )

            elif tool_name == "create_escalation":
                pending_action = {
                    "tool": "create_escalation",
                    "account_id": account_id,
                    "summary": tool_input.get("summary"),
                    "reason": tool_input.get("reason"),
                    "related_order_id": tool_input.get("related_order_id"),
                    "related_ticket_id": tool_input.get("related_ticket_id"),
                }
                output = {
                    "status": "pending_confirmation",
                    "note": "Not created yet. Waiting for explicit customer confirmation.",
                }

            else:
                output = {"error": f"Unknown tool {tool_name}"}

            tool_trace.append({"tool": tool_name, "input": tool_input, "output": output})
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(output, default=str),
            })

            if pending_action is not None:
                final_text = message.content or "I've prepared this action — please confirm below before I create it."
                print(f"[DEBUG] returning final_text={final_text!r}, pending_action={pending_action}")
                return history, tool_trace, pending_action, final_text

    # Hit the iteration cap without resolving — escalate instead of looping.
    return history, tool_trace, {
        "tool": "create_escalation",
        "account_id": account_id,
        "summary": "Customer request could not be resolved automatically.",
        "reason": "The assistant could not confidently answer within 2 tool-call attempts — needs human review.",
        "related_order_id": None,
        "related_ticket_id": None,
    }, "This needs a closer look from our support team — I've prepared an escalation for you to confirm."
