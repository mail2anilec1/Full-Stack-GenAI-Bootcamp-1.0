#!/usr/bin/env python
# coding: utf-8

# # Handoff Multi-Agent
#
# Run this script by itself to avoid invoking the other multi-agent demos.

from typing import (
    TypedDict,
    Literal,
    Annotated,
)

import operator
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from langchain_core.tools import tool
from langchain.agents import create_agent

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState,
)
from langgraph.types import (
    Command,
    Send,
)
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY not found. Add it to the project .env file before running."
    )

model = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
)


def print_title(title: str):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def show_graph(compiled_graph):
    """Print the generated graph as Mermaid text."""
    print(compiled_graph.get_graph().draw_mermaid())

# ## 4. Handoff Multi-Agent
# 
# **How it works:** The triage agent selects a specialist and uses `Command(goto=...)` to transfer graph control. The selected specialist completes the request.
# 
# **Real-world example:** A reception desk transfers a 401 API error to technical support rather than trying to solve it itself.
# 
# ```text
# User → Triage ─┬→ Billing → END
#                └→ Technical → END
# ```
# 

# ### State and handoff decision schema
# 

# In[ ]:


class HandoffDecision(BaseModel):

    target: Literal[
        "billing_agent",
        "technical_agent",
    ]

    reason: str


handoff_router_model = (
    model.with_structured_output(
        HandoffDecision
    )
)


class HandoffState(MessagesState):

    active_agent: str


# ### Triage and specialist nodes
# 

# In[ ]:


def triage_handoff_agent(
    state: HandoffState,
) -> Command[
    Literal[
        "billing_agent",
        "technical_agent",
    ]
]:

    latest_user_message = (
        state["messages"][-1].content
    )

    decision = (
        handoff_router_model.invoke(
            f"""
You are a triage agent.

Choose which specialist should handle this user request.

billing_agent:
Payments, invoices, refunds, subscriptions.

technical_agent:
Software errors, login problems, API issues, technical troubleshooting.

User request:
{latest_user_message}
"""
        )
    )

    return Command(
        goto=decision.target,

        update={
            "active_agent":
            decision.target,

            "messages": [
                AIMessage(
                    content=(
                        "Handoff reason: "
                        + decision.reason
                    )
                )
            ],
        },
    )


def handoff_billing_agent(
    state: HandoffState,
):

    response = model.invoke(
        [
            {
                "role": "system",
                "content":
                "You are a billing support specialist.",
            },

            *state["messages"],
        ]
    )

    return {
        "messages": [
            response
        ],

        "active_agent":
        "billing_agent",
    }


def handoff_technical_agent(
    state: HandoffState,
):

    response = model.invoke(
        [
            {
                "role": "system",
                "content":
                "You are a technical support specialist.",
            },

            *state["messages"],
        ]
    )

    return {
        "messages": [
            response
        ],

        "active_agent":
        "technical_agent",
    }


# ### Build the graph
# 

# In[ ]:


def build_handoff_graph():

    builder = StateGraph(
        HandoffState
    )

    builder.add_node(
        "triage",
        triage_handoff_agent,
    )

    builder.add_node(
        "billing_agent",
        handoff_billing_agent,
    )

    builder.add_node(
        "technical_agent",
        handoff_technical_agent,
    )

    builder.add_edge(
        START,
        "triage",
    )

    builder.add_edge(
        "billing_agent",
        END,
    )

    builder.add_edge(
        "technical_agent",
        END,
    )

    return builder.compile()


# ### Generated graph structure
# 
# This cell builds the graph and prints its Mermaid definition without invoking the model.
# 

# In[ ]:


handoff_graph = build_handoff_graph()
show_graph(handoff_graph)


# ### Test
# 
# **Expected behavior:** The 401 authentication problem is handed to `technical_agent`, which writes the final response.
# 
# **Model calls:** 2.
# 

# In[ ]:


def run_handoffs():

    print_title(
        "4. HANDOFF MULTI-AGENT"
    )

    graph = build_handoff_graph()

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "My API keeps returning "
                        "a 401 authentication error."
                    )
                )
            ],

            "active_agent":
            "triage",
        }
    )

    print(
        "Active Agent:",
        result["active_agent"],
    )

    print(
        "\nFinal Response:\n",
        result["messages"][-1].content,
    )

    return result


# In[ ]:


if __name__ == "__main__":
    handoff_result = run_handoffs()