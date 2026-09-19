"""LangGraph backend for the human-in-the-loop support workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, SecretStr


class ProposedAction(BaseModel):
    """Structured action proposed by the model for human review."""

    action: Literal["refund", "reject", "manual_review"]
    customer: str = Field(description="Customer name or customer ID")
    amount: float = Field(default=0, ge=0, description="Refund amount, or 0 when not applicable")
    reason: str


class AgentState(TypedDict):
    user_request: str
    action: str
    customer: str
    amount: float
    reason: str
    approved: bool
    final_message: str


class HITLWorkflow:
    """Owns the model, checkpointer, and compiled pause/resume graph."""

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        api_key: str | None = None,
    ) -> None:
        model_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": 0,
        }
        if api_key:
            model_kwargs["api_key"] = SecretStr(api_key)

        self.model = ChatOpenAI(**model_kwargs)
        self.structured_model = self.model.with_structured_output(ProposedAction)
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()

    def _analyze_request(self, state: AgentState) -> dict[str, Any]:
        prompt = f"""
You are a customer support AI agent.

Analyze the customer request below and propose exactly one action.

Customer request:
{state["user_request"]}

Possible actions:
- refund
- reject
- manual_review

Extract the customer name or ID and the requested amount. Use 0 when no
monetary amount applies. Give a concise reason. Do not execute the action;
only propose it for a human reviewer.
"""
        result = self.structured_model.invoke(prompt)

        if isinstance(result, dict):
            proposal = ProposedAction.model_validate(result)
        else:
            proposal = ProposedAction.model_validate(result)

        return proposal.model_dump()

    @staticmethod
    def _human_approval(state: AgentState) -> dict[str, bool]:
        decision = interrupt(
            {
                "message": "Please review the AI proposed action.",
                "action": state["action"],
                "customer": state["customer"],
                "amount": state["amount"],
                "reason": state["reason"],
            }
        )
        return {"approved": bool(decision)}

    @staticmethod
    def _approval_router(state: AgentState) -> Literal["execute", "reject"]:
        return "execute" if state["approved"] else "reject"

    @staticmethod
    def _execute_action(state: AgentState) -> dict[str, str]:
        if state["action"] == "refund":
            message = (
                f"Refund of ₹{state['amount']:,.2f} processed for "
                f"{state['customer']}."
            )
        elif state["action"] == "reject":
            message = f"Request rejected for {state['customer']}."
        else:
            message = (
                f"Request for {state['customer']} sent for manual review."
            )
        return {"final_message": message}

    @staticmethod
    def _reject_action(_: AgentState) -> dict[str, str]:
        return {"final_message": "Action rejected by the human reviewer."}

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("analyze", self._analyze_request)
        builder.add_node("human_approval", self._human_approval)
        builder.add_node("execute", self._execute_action)
        builder.add_node("reject", self._reject_action)

        builder.add_edge(START, "analyze")
        builder.add_edge("analyze", "human_approval")
        builder.add_conditional_edges(
            "human_approval",
            self._approval_router,
            {"execute": "execute", "reject": "reject"},
        )
        builder.add_edge("execute", END)
        builder.add_edge("reject", END)

        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    def start_review(self, user_request: str, thread_id: str) -> dict[str, Any]:
        """Run through analysis and return the payload at the interrupt."""
        request = user_request.strip()
        if not request:
            raise ValueError("Please enter a customer request.")

        initial_state: AgentState = {
            "user_request": request,
            "action": "",
            "customer": "",
            "amount": 0.0,
            "reason": "",
            "approved": False,
            "final_message": "",
        }
        result = self.graph.invoke(initial_state, config=self._config(thread_id))
        interrupts = result.get("__interrupt__", ())
        if not interrupts:
            raise RuntimeError("The graph did not pause for human approval.")

        first_interrupt = interrupts[0]
        payload = getattr(first_interrupt, "value", first_interrupt)
        if not isinstance(payload, dict):
            raise RuntimeError("The approval payload has an unexpected format.")
        return payload

    def resume_review(self, thread_id: str, approved: bool) -> dict[str, Any]:
        """Resume the paused graph with the human reviewer's decision."""
        result = self.graph.invoke(
            Command(resume=approved),
            config=self._config(thread_id),
        )
        if not result.get("final_message"):
            raise RuntimeError("The graph resumed without producing a final message.")
        return result


__all__ = ["AgentState", "HITLWorkflow", "ProposedAction"]
