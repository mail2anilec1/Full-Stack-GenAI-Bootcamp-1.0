#!/usr/bin/env python
# coding: utf-8

# # Parallel Multi-Agent Collaboration with `Send`
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

# ## 7. Parallel Multi-Agent Collaboration with `Send`
# 
# **How it works:** A coordinator creates a structured task plan. `Send` starts one specialist worker per task, the reducer collects all outputs, and the aggregator creates the final answer.
# 
# **Real-world example:** A financial company asks research, security, and business specialists to evaluate an AI-agent proposal at the same time.
# 
# ```text
# Coordinator → Dynamic Workers → Reducer → Aggregator → END
# ```
# 

# ### Plan and state schemas
# 

# In[ ]:


class AgentTask(BaseModel):

    role: Literal[
        "research",
        "security",
        "business",
    ]

    task: str


class AgentPlan(BaseModel):

    tasks: list[AgentTask]


multi_agent_planner = (
    model.with_structured_output(
        AgentPlan,
        method="json_mode",
    )
)


class CollaborationState(TypedDict):

    user_task: str

    tasks: list[AgentTask]

    outputs: Annotated[
        list[str],
        operator.add,
    ]

    final_answer: str


class CollaborationWorkerState(TypedDict):

    role: str
    task: str

    outputs: Annotated[
        list[str],
        operator.add,
    ]


# ### Coordinator, workers, and aggregator
# 

# In[ ]:


def collaboration_coordinator(
    state: CollaborationState,
):

    plan = multi_agent_planner.invoke(
        f"""
Break this task into useful specialist tasks.

Available specialist roles:
- research
- security
- business

Create 2-5 concise tasks, and include only useful tasks.
Return only a valid JSON object in this shape:
{{"tasks": [{{"role": "research", "task": "..."}}]}}
Each role must be research, security, or business.

User task:
{state["user_task"]}
"""
    )

    return {
        "tasks":
        plan.tasks
    }


def send_collaboration_workers(
    state: CollaborationState,
):

    return [
        Send(
            "specialist_worker",
            {
                "role":
                item.role,

                "task":
                item.task,
            },
        )

        for item
        in state["tasks"]
    ]


def specialist_worker(
    state: CollaborationWorkerState,
):

    role = state["role"]

    role_prompts = {

        "research":
        "You are a technical research specialist.",

        "security":
        "You are an AI security specialist.",

        "business":
        "You are an enterprise business specialist.",
    }

    response = model.invoke(
        f"""
{role_prompts[role]}

Complete this task:

{state["task"]}
"""
    )

    return {
        "outputs": [
            f"{role.upper()} AGENT:\n"
            f"{response.content}"
        ]
    }


def collaboration_aggregator(
    state: CollaborationState,
):

    all_outputs = "\n\n".join(
        state["outputs"]
    )

    response = model.invoke(
        f"""
You are the final coordinator.

Original task:
{state["user_task"]}

Specialist outputs:
{all_outputs}

Combine them into one coherent final answer.
"""
    )

    return {
        "final_answer":
        response.content
    }


# ### Build the graph
# 

# In[ ]:


def build_parallel_collaboration():

    builder = StateGraph(
        CollaborationState
    )

    builder.add_node(
        "coordinator",
        collaboration_coordinator,
    )

    builder.add_node(
        "specialist_worker",
        specialist_worker,
    )

    builder.add_node(
        "aggregator",
        collaboration_aggregator,
    )

    builder.add_edge(
        START,
        "coordinator",
    )

    builder.add_conditional_edges(
        "coordinator",
        send_collaboration_workers,
        [
            "specialist_worker"
        ],
    )

    builder.add_edge(
        "specialist_worker",
        "aggregator",
    )

    builder.add_edge(
        "aggregator",
        END,
    )

    return builder.compile()


# ### Generated graph structure
# 
# This cell builds the graph and prints its Mermaid definition without invoking the model.
# 

# In[ ]:


collaboration_graph = build_parallel_collaboration()
show_graph(collaboration_graph)


# ### Test
# 
# **Expected behavior:** The planner creates useful specialist tasks, workers run dynamically, and one combined enterprise assessment is returned.
# 
# **Model calls:** `N + 2`: planner, `N` workers, and aggregator.
# 

# In[ ]:


def run_parallel_collaboration():

    print_title(
        "7. PARALLEL MULTI-AGENT COLLABORATION WITH Send()"
    )

    graph = (
        build_parallel_collaboration()
    )

    result = graph.invoke(
        {
            "user_task": (
                "Evaluate the use of AI agents "
                "inside a large financial enterprise."
            ),

            "tasks": [],
            "outputs": [],
            "final_answer": "",
        }
    )

    print(
        result["final_answer"]
    )

    return result


# In[ ]:


if __name__ == "__main__":
    collaboration_result = run_parallel_collaboration()