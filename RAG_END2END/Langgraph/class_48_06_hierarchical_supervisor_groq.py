#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Supervisor
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

# ## 6. Hierarchical Supervisor
# 
# **How it works:** A top supervisor selects one team subgraph. The selected team performs specialist work and then rewrites it as a concise report.
# 
# **Real-world example:** A company director sends a task to either the research department or the business department instead of managing every employee directly.
# 
# ```text
# Top Supervisor ─┬→ Research Team Subgraph → END
#                 └→ Business Team Subgraph → END
# ```
# 

# ### Build the team subgraphs
# 

# In[ ]:


class TeamState(TypedDict):

    task: str
    team_output: str


def research_team_node(
    state: TeamState,
):

    research = model.invoke(
        f"""
You are the research specialist.

Research/explain:
{state["task"]}
"""
    )

    writer = model.invoke(
        f"""
You are the research-team writer.

Convert this research into a concise report:

{research.content}
"""
    )

    return {
        "team_output":
        writer.content
    }


def business_team_node(
    state: TeamState,
):

    analysis = model.invoke(
        f"""
You are a business analyst.

Analyze this task from a business perspective:

{state["task"]}
"""
    )

    writer = model.invoke(
        f"""
You are the business-team writer.

Convert this analysis into a concise business report:

{analysis.content}
"""
    )

    return {
        "team_output":
        writer.content
    }


research_team_builder = StateGraph(
    TeamState
)

research_team_builder.add_node(
    "research_team",
    research_team_node,
)

research_team_builder.add_edge(
    START,
    "research_team",
)

research_team_builder.add_edge(
    "research_team",
    END,
)

research_team_graph = (
    research_team_builder.compile()
)


business_team_builder = StateGraph(
    TeamState
)

business_team_builder.add_node(
    "business_team",
    business_team_node,
)

business_team_builder.add_edge(
    START,
    "business_team",
)

business_team_builder.add_edge(
    "business_team",
    END,
)

business_team_graph = (
    business_team_builder.compile()
)


# ### Top-supervisor state and nodes
# 

# In[ ]:


class TopSupervisorDecision(BaseModel):

    team: Literal[
        "research_team",
        "business_team",
    ]


top_supervisor_model = (
    model.with_structured_output(
        TopSupervisorDecision
    )
)


class HierarchicalState(TypedDict):

    task: str
    selected_team: str
    final_answer: str


def top_supervisor(
    state: HierarchicalState,
):

    decision = (
        top_supervisor_model.invoke(
            f"""
Choose the best team for this task.

research_team:
Technical research, explanations, educational content.

business_team:
Business analysis, strategy, commercial implications.

Task:
{state["task"]}
"""
        )
    )

    return {
        "selected_team":
        decision.team
    }


def call_research_team(
    state: HierarchicalState,
):

    result = (
        research_team_graph.invoke(
            {
                "task":
                state["task"],

                "team_output":
                "",
            }
        )
    )

    return {
        "final_answer":
        result["team_output"]
    }


def call_business_team(
    state: HierarchicalState,
):

    result = (
        business_team_graph.invoke(
            {
                "task":
                state["task"],

                "team_output":
                "",
            }
        )
    )

    return {
        "final_answer":
        result["team_output"]
    }


def hierarchical_route(
    state: HierarchicalState,
):

    return state["selected_team"]


# ### Build the parent graph
# 

# In[ ]:


def build_hierarchical_supervisor():

    builder = StateGraph(
        HierarchicalState
    )

    builder.add_node(
        "top_supervisor",
        top_supervisor,
    )

    builder.add_node(
        "research_team",
        call_research_team,
    )

    builder.add_node(
        "business_team",
        call_business_team,
    )

    builder.add_edge(
        START,
        "top_supervisor",
    )

    builder.add_conditional_edges(
        "top_supervisor",
        hierarchical_route,
        {
            "research_team":
            "research_team",

            "business_team":
            "business_team",
        },
    )

    builder.add_edge(
        "research_team",
        END,
    )

    builder.add_edge(
        "business_team",
        END,
    )

    return builder.compile()


# ### Generated graph structure
# 
# This cell builds the graph and prints its Mermaid definition without invoking the model.
# 

# In[ ]:


print("RESEARCH TEAM")
show_graph(research_team_graph)

print("BUSINESS TEAM")
show_graph(business_team_graph)

print("TOP-LEVEL GRAPH")
hierarchical_graph = build_hierarchical_supervisor()
show_graph(hierarchical_graph)


# ### Test
# 
# **Expected behavior:** The enterprise-value question is normally sent to `business_team`, and that team returns a concise report.
# 
# **Model calls:** 3: one team-selection call and two calls inside the selected team.
# 

# In[ ]:


def run_hierarchical_supervisor():

    print_title(
        "6. HIERARCHICAL SUPERVISOR"
    )

    graph = (
        build_hierarchical_supervisor()
    )

    result = graph.invoke(
        {
            "task": (
                "Explain the business value "
                "of using Agentic RAG in an enterprise."
            ),

            "selected_team": "",
            "final_answer": "",
        }
    )

    print(
        "Selected Team:",
        result["selected_team"],
    )

    print(
        "\nFinal Answer:\n",
        result["final_answer"],
    )

    return result


# In[ ]:


if __name__ == "__main__":
    hierarchical_result = run_hierarchical_supervisor()