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


def run_agent_turn(
    data_store,
    account_id: str,
    account_scope: str,
    history: list,
):
    """
    Runs one full turn, potentially involving several tool calls.

    Returns:
        history
        tool_trace
        pending_action
        final_text
    """

    tool_trace = []
    pending_action = None

    while True:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                *history,
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1500,
        )

        message = response.choices[0].message

        # ---------------------------------------------------------
        # No tool call -> final answer
        # ---------------------------------------------------------

        if not message.tool_calls:

            final_text = message.content or ""

            history.append(
                {
                    "role": "assistant",
                    "content": final_text,
                }
            )

            return (
                history,
                tool_trace,
                pending_action,
                final_text,
            )

        # ---------------------------------------------------------
        # Store assistant's tool-call message
        # ---------------------------------------------------------

        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [],
        }

        for tool_call in message.tool_calls:

            assistant_message["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )

        history.append(assistant_message)

        # ---------------------------------------------------------
        # Execute tools
        # ---------------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            try:
                tool_input = json.loads(
                    tool_call.function.arguments
                )
            except json.JSONDecodeError:
                tool_input = {}

            # -----------------------------------------------------
            # document_search
            # -----------------------------------------------------

            if tool_name == "document_search":

                output = data_store.document_search(
                    query=tool_input["query"],
                    k=tool_input.get("k", 5),
                    account_scope=account_scope,
                )

            # -----------------------------------------------------
            # structured_data_lookup
            # -----------------------------------------------------

            elif tool_name == "structured_data_lookup":

                # IMPORTANT:
                # account_id comes from the authenticated session,
                # NOT from the LLM.

                output = data_store.structured_data_lookup(
                    account_id=account_id,
                    query_type=tool_input["query_type"],
                    order_id=tool_input.get("order_id"),
                    ticket_id=tool_input.get("ticket_id"),
                )

            # -----------------------------------------------------
            # create_escalation
            # -----------------------------------------------------

            elif tool_name == "create_escalation":

                # DO NOT ACTUALLY CREATE THE ESCALATION.

                pending_action = {
                    "tool": "create_escalation",
                    "account_id": account_id,
                    "summary": tool_input.get("summary"),
                    "reason": tool_input.get("reason"),
                    "related_order_id": tool_input.get(
                        "related_order_id"
                    ),
                    "related_ticket_id": tool_input.get(
                        "related_ticket_id"
                    ),
                }

                output = {
                    "status": "pending_confirmation",
                    "note": (
                        "Not created yet. Waiting for explicit "
                        "customer confirmation."
                    ),
                }

            # -----------------------------------------------------
            # Unknown tool
            # -----------------------------------------------------

            else:

                output = {
                    "error": f"Unknown tool {tool_name}"
                }

            # -----------------------------------------------------
            # Store trace
            # -----------------------------------------------------

            tool_trace.append(
                {
                    "tool": tool_name,
                    "input": tool_input,
                    "output": output,
                }
            )

            # -----------------------------------------------------
            # Add tool result to conversation
            # -----------------------------------------------------

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        output,
                        default=str,
                    ),
                }
            )

            # -----------------------------------------------------
            # Stop immediately if escalation is pending
            # -----------------------------------------------------

            if pending_action is not None:

                final_text = (
                    message.content
                    or "I've prepared this action — please confirm "
                       "below before I create it."
                )

                return (
                    history,
                    tool_trace,
                    pending_action,
                    final_text,
                )