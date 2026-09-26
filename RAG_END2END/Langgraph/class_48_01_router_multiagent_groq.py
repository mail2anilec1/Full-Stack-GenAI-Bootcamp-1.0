#!/usr/bin/env python
# coding: utf-8

# # Router Multi-Agent
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

# ## 1. Router Multi-Agent
# 
# **How it works:** A router reads the request and selects exactly one specialist. Only that specialist answers.
# 
# **Real-world example:** A customer-support desk sends a duplicate payment issue to billing, a login issue to technical support, and a general question to general support.
# 
# ```text
# User → Router → One Specialist → Response
# ```
# 

# ### State and routing schema
# 

# In[ ]:


class RouterDecision(BaseModel):

    route: Literal[
        "technical",
        "billing",
        "general",
    ]


router_model = model.with_structured_output(
    RouterDecision
)


class RouterState(TypedDict):

    query: str
    route: str
    response: str


# ### Router and specialist nodes
# 

# In[ ]:


def router_node(
    state: RouterState,
):

    decision = router_model.invoke(
        f"""
Route the following user query to exactly one specialist.

Available specialists:
- technical
- billing
- general

User query:
{state["query"]}
"""
    )

    return {
        "route": decision.route
    }


def technical_agent(
    state: RouterState,
):

    response = model.invoke(
        f"""
You are a technical support specialist.

Answer this query:
{state["query"]}
"""
    )

    return {
        "response": response.content
    }


def billing_agent(
    state: RouterState,
):

    response = model.invoke(
        f"""
You are a billing support specialist.

Answer this query:
{state["query"]}
"""
    )

    return {
        "response": response.content
    }


def general_agent(
    state: RouterState,
):

    response = model.invoke(
        f"""
You are a general support specialist.

Answer this query:
{state["query"]}
"""
    )

    return {
        "response": response.content
    }


def route_to_specialist(
    state: RouterState,
):

    return state["route"]


# ### Build the graph
# 

# In[ ]:


def build_router_multi_agent():

    builder = StateGraph(
        RouterState
    )

    builder.add_node(
        "router",
        router_node,
    )

    builder.add_node(
        "technical",
        technical_agent,
    )

    builder.add_node(
        "billing",
        billing_agent,
    )

    builder.add_node(
        "general",
        general_agent,
    )

    builder.add_edge(
        START,
        "router",
    )

    builder.add_conditional_edges(
        "router",
        route_to_specialist,
        {
            "technical": "technical",
            "billing": "billing",
            "general": "general",
        },
    )

    builder.add_edge(
        "technical",
        END,
    )

    builder.add_edge(
        "billing",
        END,
    )

    builder.add_edge(
        "general",
        END,
    )

    return builder.compile()


# ### Generated graph structure
# 
# This cell builds the graph and prints its Mermaid definition without invoking the model.
# 

# In[ ]:


router_graph = build_router_multi_agent()
show_graph(router_graph)


# ### Test
# 
# **Expected behavior:** The duplicate-charge query is routed to `billing`, and the billing specialist produces the response.
# 
# **Model calls:** 2.
# 

# In[ ]:


def run_router_multi_agent():

    print_title(
        "1. ROUTER MULTI-AGENT"
    )

    graph = (
        build_router_multi_agent()
    )

    result = graph.invoke(
        {
            "query":
            "My subscription payment was charged twice.",
            "route": "",
            "response": "",
        }
    )

    print(
        "Selected Agent:",
        result["route"],
    )

    print(
        "\nResponse:\n",
        result["response"],
    )

    return result


# In[ ]:


if __name__ == "__main__":
    router_result = run_router_multi_agent()