#!/usr/bin/env python
# coding: utf-8

# # Supervisor Multi-Agent
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

# ## 3. Supervisor Multi-Agent
# 
# **How it works:** A central supervisor repeatedly selects the next worker, reviews accumulated work, and finishes when enough work is available. A six-turn limit prevents an endless loop.
# 
# **Real-world example:** A project manager assigns research, calculation, and writing tasks one at a time, then creates the final training note.
# 
# ```text
# Supervisor → Worker → Supervisor → Another Worker → Finish
# ```
# 

# ### State and supervisor decision schema
# 

# In[ ]:


class SupervisorDecision(BaseModel):

    next_agent: Literal[
        "research_agent",
        "math_agent",
        "writer_agent",
        "FINISH",
    ]

    instruction: str


class SupervisorState(TypedDict):

    task: str

    next_agent: str

    instruction: str

    worker_outputs: Annotated[
        list[str],
        operator.add,
    ]

    final_answer: str

    supervisor_turns: int


supervisor_router = (
    model.with_structured_output(
        SupervisorDecision,
        method="json_mode",
    )
)


# ### Supervisor and worker nodes
# 

# In[ ]:


def supervisor(
    state: SupervisorState,
):

    supervisor_turns = state.get(
        "supervisor_turns",
        0,
    ) + 1

    completed_work = "\n\n".join(
        state.get(
            "worker_outputs",
            [],
        )
    )

    if supervisor_turns >= 6 and completed_work:
        return {
            "next_agent": "FINISH",
            "instruction": (
                "Maximum supervisor turns reached. "
                "Create the final answer."
            ),
            "supervisor_turns": supervisor_turns,
        }

    decision = supervisor_router.invoke(
        f"""
You are a supervisor managing three workers:

- research_agent
- math_agent
- writer_agent

Original task:
{state["task"]}

Work completed so far:
{completed_work or "No worker has completed any work yet."}

Choose the next worker.

Rules:
- Use research_agent for research/explanation.
- Use math_agent for calculations.
- Use writer_agent when a polished final presentation is useful.
- If enough work is complete, choose FINISH.
- Do not repeatedly call the same worker unless required.

Return:
Return only a valid JSON object with these fields:
- next_agent: one of research_agent, math_agent, writer_agent, or FINISH
- instruction: a string describing the next action, or the finalization instruction
"""
    )

    return {
        "next_agent":
        decision.next_agent,

        "instruction":
        decision.instruction,

        "supervisor_turns":
        supervisor_turns,
    }


def supervisor_research_worker(
    state: SupervisorState,
):

    response = model.invoke(
        f"""
You are the research worker.

Original task:
{state["task"]}

Supervisor instruction:
{state["instruction"]}

Perform only the research/explanation part.
"""
    )

    return {
        "worker_outputs": [
            "RESEARCH WORKER:\n"
            + response.content
        ]
    }


def supervisor_math_worker(
    state: SupervisorState,
):

    response = model.invoke(
        f"""
You are the math worker.

Original task:
{state["task"]}

Supervisor instruction:
{state["instruction"]}

Perform only the calculation/reasoning part.
"""
    )

    return {
        "worker_outputs": [
            "MATH WORKER:\n"
            + response.content
        ]
    }


def supervisor_writer_worker(
    state: SupervisorState,
):

    previous_work = "\n\n".join(
        state.get(
            "worker_outputs",
            [],
        )
    )

    response = model.invoke(
        f"""
You are the writing worker.

Original task:
{state["task"]}

Supervisor instruction:
{state["instruction"]}

Available previous work:
{previous_work}

Create a polished response or writing contribution.
"""
    )

    return {
        "worker_outputs": [
            "WRITER WORKER:\n"
            + response.content
        ]
    }


def supervisor_finish(
    state: SupervisorState,
):

    completed_work = "\n\n".join(
        state["worker_outputs"]
    )

    response = model.invoke(
        f"""
You are the supervisor.

Create the final answer for the user.

Original task:
{state["task"]}

Worker outputs:
{completed_work}
"""
    )

    return {
        "final_answer":
        response.content
    }


def supervisor_route(
    state: SupervisorState,
):

    return state["next_agent"]


# ### Build the graph
# 

# In[ ]:


def build_supervisor_graph():

    builder = StateGraph(
        SupervisorState
    )

    builder.add_node(
        "supervisor",
        supervisor,
    )

    builder.add_node(
        "research_agent",
        supervisor_research_worker,
    )

    builder.add_node(
        "math_agent",
        supervisor_math_worker,
    )

    builder.add_node(
        "writer_agent",
        supervisor_writer_worker,
    )

    builder.add_node(
        "finish",
        supervisor_finish,
    )

    builder.add_edge(
        START,
        "supervisor",
    )

    builder.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "research_agent":
            "research_agent",

            "math_agent":
            "math_agent",

            "writer_agent":
            "writer_agent",

            "FINISH":
            "finish",
        },
    )

    builder.add_edge(
        "research_agent",
        "supervisor",
    )

    builder.add_edge(
        "math_agent",
        "supervisor",
    )

    builder.add_edge(
        "writer_agent",
        "supervisor",
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


supervisor_graph = build_supervisor_graph()
show_graph(supervisor_graph)


# ### Test
# 
# **Expected behavior:** The supervisor chooses useful workers and eventually returns a final training note. `supervisor_turns` cannot exceed the safety limit.
# 
# **Model calls:** Variable and bounded by the supervisor-turn limit.
# 

# In[ ]:


def run_supervisor():

    print_title(
        "3. SUPERVISOR MULTI-AGENT"
    )

    graph = build_supervisor_graph()

    result = graph.invoke(
        {
            "task": (
                "Explain Retrieval-Augmented Generation, "
                "calculate 48 * 25, and present everything "
                "as a short training note."
            ),

            "next_agent": "",
            "instruction": "",
            "worker_outputs": [],
            "final_answer": "",
            "supervisor_turns": 0,
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
    supervisor_result = run_supervisor()