#!/usr/bin/env python
# coding: utf-8

# # Coordinator-Guided Peer Network
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

# ## 8. Coordinator-Guided Peer Network
# 
# **How it works:** Research, analysis, and writing peers share notes. A lightweight coordinator selects the next peer after each step. This is coordinator-guided, not a pure peer-to-peer network. A six-turn limit prevents an endless loop.
# 
# **Real-world example:** A round-table facilitator decides whether the researcher, analyst, or writer should speak next until the group has a complete explanation.
# 
# ```text
# Coordinator → Peer → Shared Notes → Coordinator → Next Peer → Finish
# ```
# 

# ### Decision and network state
# 

# In[ ]:


class PeerDecision(BaseModel):

    next_agent: Literal[
        "research_peer",
        "analysis_peer",
        "writer_peer",
        "FINISH",
    ]

    message: str


peer_router = model.with_structured_output(
    PeerDecision
)


class NetworkState(TypedDict):

    task: str

    notes: Annotated[
        list[str],
        operator.add,
    ]

    next_agent: str

    final_answer: str

    network_turns: int


# ### Peer and coordinator nodes
# 

# In[ ]:


def network_research_peer(
    state: NetworkState,
):

    previous = "\n\n".join(
        state.get(
            "notes",
            [],
        )
    )

    response = model.invoke(
        f"""
You are the research peer.

Task:
{state["task"]}

Previous peer notes:
{previous}

Add useful research information.
"""
    )

    return {
        "notes": [
            "RESEARCH:\n"
            + response.content
        ]
    }


def network_analysis_peer(
    state: NetworkState,
):

    previous = "\n\n".join(
        state.get(
            "notes",
            [],
        )
    )

    response = model.invoke(
        f"""
You are the analysis peer.

Task:
{state["task"]}

Current team notes:
{previous}

Analyze the information and identify useful conclusions.
"""
    )

    return {
        "notes": [
            "ANALYSIS:\n"
            + response.content
        ]
    }


def network_writer_peer(
    state: NetworkState,
):

    previous = "\n\n".join(
        state.get(
            "notes",
            [],
        )
    )

    response = model.invoke(
        f"""
You are the writing peer.

Task:
{state["task"]}

Current team notes:
{previous}

Create a polished draft.
"""
    )

    return {
        "notes": [
            "WRITER:\n"
            + response.content
        ]
    }


def peer_coordinator(
    state: NetworkState,
):

    network_turns = state.get(
        "network_turns",
        0,
    ) + 1

    notes = "\n\n".join(
        state.get(
            "notes",
            [],
        )
    )

    if network_turns >= 6 and notes:
        return {
            "next_agent": "FINISH",
            "network_turns": network_turns,
        }

    decision = peer_router.invoke(
        f"""
You coordinate a guided peer network.

Task:
{state["task"]}

Current notes:
{notes or "No work completed yet."}

Available peers:
- research_peer
- analysis_peer
- writer_peer
- FINISH

Choose the most useful next peer.
When the task has enough research, analysis,
and a polished answer, choose FINISH.

Return:
- next_agent
- short reason/message
"""
    )

    return {
        "next_agent":
        decision.next_agent,

        "network_turns":
        network_turns,
    }


def network_finish(
    state: NetworkState,
):

    notes = "\n\n".join(
        state["notes"]
    )

    response = model.invoke(
        f"""
Create the final answer.

Original task:
{state["task"]}

Peer-agent work:
{notes}
"""
    )

    return {
        "final_answer":
        response.content
    }


def network_route(
    state: NetworkState,
):

    return state["next_agent"]


# ### Build the graph
# 

# In[ ]:


def build_coordinator_guided_network():

    builder = StateGraph(
        NetworkState
    )

    builder.add_node(
        "peer_coordinator",
        peer_coordinator,
    )

    builder.add_node(
        "research_peer",
        network_research_peer,
    )

    builder.add_node(
        "analysis_peer",
        network_analysis_peer,
    )

    builder.add_node(
        "writer_peer",
        network_writer_peer,
    )

    builder.add_node(
        "finish",
        network_finish,
    )

    builder.add_edge(
        START,
        "peer_coordinator",
    )

    builder.add_conditional_edges(
        "peer_coordinator",
        network_route,
        {
            "research_peer":
            "research_peer",

            "analysis_peer":
            "analysis_peer",

            "writer_peer":
            "writer_peer",

            "FINISH":
            "finish",
        },
    )

    builder.add_edge(
        "research_peer",
        "peer_coordinator",
    )

    builder.add_edge(
        "analysis_peer",
        "peer_coordinator",
    )

    builder.add_edge(
        "writer_peer",
        "peer_coordinator",
    )

    builder.add_edge(
        "finish",
        END,
    )

    return builder.compile()


# ### Generated graph structure
# 
# This cell builds the graph and prints its Mermaid definition without invoking the model.
# 

# In[ ]:


network_graph = build_coordinator_guided_network()
show_graph(network_graph)


# ### Test
# 
# **Expected behavior:** The coordinator selects useful peers, notes accumulate, and the finish node creates the final beginner-friendly explanation.
# 
# **Model calls:** Variable and bounded by the network-turn limit.
# 

# In[ ]:


def run_coordinator_guided_network():

    print_title(
        "8. COORDINATOR-GUIDED PEER NETWORK"
    )

    graph = (
        build_coordinator_guided_network()
    )

    result = graph.invoke(
        {
            "task": (
                "Create a beginner-friendly explanation "
                "of multi-agent AI systems."
            ),

            "notes": [],
            "next_agent": "",
            "final_answer": "",
            "network_turns": 0,
        },

        config={
            "recursion_limit": 20
        },
    )

    print(
        result["final_answer"]
    )

    return result


# In[ ]:


if __name__ == "__main__":
    network_result = run_coordinator_guided_network()