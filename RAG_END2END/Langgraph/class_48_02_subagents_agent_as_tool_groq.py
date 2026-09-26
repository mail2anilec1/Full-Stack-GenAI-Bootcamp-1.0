#!/usr/bin/env python
# coding: utf-8

# # Subagents / Agent-as-Tool
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

# ## 2. Subagents / Agent-as-Tool
# 
# **How it works:** A main coordinator remains in control and calls specialist agents as tools. Each specialist has a focused role.
# 
# **Real-world example:** A consultant asks a researcher for facts, a math expert for a calculation, and a writer to present the combined answer clearly.
# 
# ```text
# User → Main Agent → Specialist Tools → Main Agent → Final Answer
# ```
# 

# ### Create the specialist subagents
# 

# In[ ]:


research_subagent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a research specialist. "
        "Provide concise factual research."
    ),
    name="research_subagent",
)


math_subagent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a math specialist. "
        "Solve calculations accurately."
    ),
    name="math_subagent",
)


writing_subagent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a professional writing specialist. "
        "Rewrite information clearly and concisely."
    ),
    name="writing_subagent",
)


# ### Wrap subagents as tools and create the coordinator
# 

# In[ ]:


def get_last_message_content(
    result,
):

    return (
        result["messages"][-1].content
    )


@tool
def ask_research_agent(
    task: str,
) -> str:
    """
    Delegate a research task to the research specialist.
    """

    result = research_subagent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
    )

    return get_last_message_content(
        result
    )


@tool
def ask_math_agent(
    task: str,
) -> str:
    """
    Delegate a mathematical task to the math specialist.
    """

    result = math_subagent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
    )

    return get_last_message_content(
        result
    )


@tool
def ask_writing_agent(
    task: str,
) -> str:
    """
    Delegate a writing task to the writing specialist.
    """

    result = writing_subagent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
    )

    return get_last_message_content(
        result
    )


main_delegation_agent = create_agent(
    model=model,
    tools=[
        ask_research_agent,
        ask_math_agent,
        ask_writing_agent,
    ],
    system_prompt="""
You are the main coordinator.

You have three specialist subagents:
1. Research specialist
2. Math specialist
3. Writing specialist

Delegate work to the appropriate specialist tools.
You may call multiple specialists when necessary.
Finally provide one clear answer to the user.
""",
    name="main_coordinator",
)


# ### Generated graph structure
# 
# This cell builds the graph and prints its Mermaid definition without invoking the model.
# 

# In[ ]:


show_graph(main_delegation_agent)


# ### Test
# 
# **Expected behavior:** The coordinator may delegate research, multiplication, and writing before returning one answer.
# 
# **Model calls:** Variable because the main agent decides which tools to call.
# 

# In[ ]:


def run_subagents():

    print_title(
        "2. SUBAGENTS / AGENT-AS-TOOL"
    )

    result = (
        main_delegation_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Explain RAG briefly, calculate "
                            "25 * 16, and present the answer "
                            "in a clean professional format."
                        ),
                    }
                ]
            }
        )
    )

    print(
        result["messages"][-1].content
    )

    return result


# In[ ]:


if __name__ == "__main__":
    subagents_result = run_subagents()