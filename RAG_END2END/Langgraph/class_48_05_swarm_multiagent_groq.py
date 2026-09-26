#!/usr/bin/env python
# coding: utf-8

# # Swarm Multi-Agent
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

# ## 5. Swarm Multi-Agent
# 
# **How it works:** Agents can hand control directly to another agent. `InMemorySaver` preserves the active-agent state for the thread.
# 
# **Real-world example:** A math expert calculates a result and hands it to a writer for professional presentation.
# 
# ```text
# Math Agent ⇄ Writer Agent
# ```
# 

# ### Build the Swarm
# 

# In[ ]:


def build_swarm_graph():

    # Optional dependency imported here
    # so the rest of the file can still
    # be studied without langgraph-swarm.

    from langgraph_swarm import (
        create_handoff_tool,
        create_swarm,
    )

    math_agent = create_agent(
        model=model,

        tools=[
            create_handoff_tool(
                agent_name="writer_agent",
                description=(
                    "Transfer to the writer "
                    "when polished writing is needed."
                ),
            )
        ],

        system_prompt=(
            "You are a math specialist. "
            "Solve calculations. "
            "Transfer to writer_agent when "
            "the result should be professionally presented."
        ),

        name="math_agent",
    )

    writer_agent = create_agent(
        model=model,

        tools=[
            create_handoff_tool(
                agent_name="math_agent",
                description=(
                    "Transfer to the math specialist "
                    "when calculations are needed."
                ),
            )
        ],

        system_prompt=(
            "You are a professional writer. "
            "Present information clearly. "
            "Transfer to math_agent when calculations are needed."
        ),

        name="writer_agent",
    )

    workflow = create_swarm(
        [
            math_agent,
            writer_agent,
        ],

        default_active_agent=
        "math_agent",
    )

    checkpointer = (
        InMemorySaver()
    )

    return workflow.compile(
        checkpointer=checkpointer
    )


# ### Generated graph structure
# 
# This cell builds the graph and prints its Mermaid definition without invoking the model.
# 

# In[ ]:


try:
    swarm_graph = build_swarm_graph()
    show_graph(swarm_graph)
except ImportError as error:
    print(error)


# ### Test
# 
# **Expected behavior:** The calculation begins with the math agent and can be handed to the writer agent. The final response is stored under one thread ID.
# 
# **Model calls:** Variable because agents decide whether to hand off.
# 

# In[ ]:


def run_swarm():

    print_title(
        "5. SWARM MULTI-AGENT"
    )

    try:

        graph = build_swarm_graph()

    except ImportError:

        print(
            "langgraph-swarm is not installed.\n"
            "Install it using:\n"
            "pip install -U langgraph-swarm"
        )

        return

    config = {
        "configurable": {
            "thread_id":
            "swarm_thread_1"
        }
    }

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Calculate 125 * 32 "
                        "and present the result "
                        "as a professional one-line answer."
                    ),
                }
            ]
        },

        config=config,
    )

    print(
        result["messages"][-1].content
    )

    return result


# In[ ]:


if __name__ == "__main__":
    swarm_result = run_swarm()