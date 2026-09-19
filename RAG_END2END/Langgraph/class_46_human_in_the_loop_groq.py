"""Human-in-the-loop customer-support workflow using Groq and LangGraph."""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel


load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file before running this script."
    )


MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
model = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    reasoning_format="hidden",
)


class ProposedAction(BaseModel):
    """Action proposed by the model for human approval."""

    action: Literal["refund", "reject", "manual_review"]
    customer: str
    amount: float
    reason: str


# JSON mode avoids Groq tool-call failures when the model returns structured
# fields as text instead of emitting a tool call.
structured_model = model.with_structured_output(
    ProposedAction,
    method="json_mode",
)


class AgentState(TypedDict):
    user_request: str
    action: str
    customer: str
    amount: float
    reason: str
    approved: bool
    final_message: str


def analyze_request(state: AgentState) -> dict[str, object]:
    """Ask Groq to propose an action without executing it."""
    prompt = f"""
You are a customer support AI agent.

Analyze the following customer request:

{state["user_request"]}

Decide the appropriate action.

Possible actions:
- refund
- reject
- manual_review

Return exactly one valid JSON object with these keys:
{{
    "action": "refund" | "reject" | "manual_review",
    "customer": "customer ID",
    "amount": 0,
    "reason": "short explanation"
}}

Use an amount of 0 when no refund amount is requested. Do not use Markdown
code fences or add any text outside the JSON object.
Do not execute anything. Only propose an action for human approval.
"""

    result = structured_model.invoke(prompt)
    return {
        "action": result.action,
        "customer": result.customer,
        "amount": result.amount,
        "reason": result.reason,
    }


def human_approval(state: AgentState) -> dict[str, bool]:
    """Pause the graph until a human supplies a Boolean decision."""
    decision = interrupt(
        {
            "message": "Please review the AI proposed action.",
            "action": state["action"],
            "customer": state["customer"],
            "amount": state["amount"],
            "reason": state["reason"],
        }
    )
    return {"approved": decision}


def approval_router(state: AgentState) -> str:
    """Route the paused workflow according to the human decision."""
    return "execute" if state["approved"] else "reject"


def execute_action(state: AgentState) -> dict[str, str]:
    """Execute the approved action and return a user-facing result."""
    if state["action"] == "refund":
        message = (
            f"Refund of INR {state['amount']} processed for "
            f"{state['customer']}."
        )
    elif state["action"] == "reject":
        message = f"Request rejected for {state['customer']}."
    else:
        message = (
            f"Request for {state['customer']} sent for manual review."
        )

    return {"final_message": message}


def reject_action(state: AgentState) -> dict[str, str]:
    """Record that the human reviewer rejected the proposal."""
    return {"final_message": "Action rejected by human reviewer."}


def build_graph():
    """Build the checkpointed human-in-the-loop graph."""
    builder = StateGraph(AgentState)
    builder.add_node("analyze", analyze_request)
    builder.add_node("human_approval", human_approval)
    builder.add_node("execute", execute_action)
    builder.add_node("reject", reject_action)

    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "human_approval")
    builder.add_conditional_edges(
        "human_approval",
        approval_router,
        {"execute": "execute", "reject": "reject"},
    )
    builder.add_edge("execute", END)
    builder.add_edge("reject", END)

    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    """Run the proposal, review, and conditional execution workflow."""
    user_request = input("User Request: ").strip()
    if not user_request:
        raise ValueError("User Request cannot be empty.")

    graph = build_graph()
    config = {"configurable": {"thread_id": "refund_thread_1"}}
    initial_state: AgentState = {
        "user_request": user_request,
        "action": "",
        "customer": "",
        "amount": 0.0,
        "reason": "",
        "approved": False,
        "final_message": "",
    }

    paused_result = graph.invoke(initial_state, config=config)
    print("\nGraph paused for human approval.")
    print("\nInterrupt Information:")
    print(paused_result["__interrupt__"])

    human_input = input("\nApprove action? (yes/no): ").strip().lower()
    approved = human_input == "yes"
    final_result = graph.invoke(Command(resume=approved), config=config)

    print("\nFinal Result:")
    print(final_result["final_message"])


if __name__ == "__main__":
    main()