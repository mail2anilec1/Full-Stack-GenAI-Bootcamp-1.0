"""Runnable, debugger-friendly companion to ``langchain_latest_version.ipynb``.

The notebook introduces LangChain concepts through small, separate examples.
This script keeps those examples in named functions so they can be run and
debugged one at a time:

    python langchain_latest_version_groq.py --demo prompts
    python langchain_latest_version_groq.py --demo chat
    python langchain_latest_version_groq.py --demo all

Most demos make requests to Groq and therefore need ``GROQ_API_KEY`` in the
repository's root ``.env`` file. The ``document``, ``messages`` and ``prompts``
demos run locally without an API key. Install the core dependencies with
``pip install langchain langchain-core langchain-groq python-dotenv pydantic``.
The optional ``deep-agent`` demo additionally needs ``pip install deepagents``.

Groq is used for chat completions; it is not an OpenAI embeddings replacement.
The notebook's OpenAI embedding model is only a model-name example, so this
script leaves embeddings out. For a RAG application, pair Groq chat with a
separate embeddings provider such as local Hugging Face embeddings.

The notebook's image example uses a placeholder URL, and its final extensive
Deep Agents configuration refers to objects that are not defined in the
notebook. Those examples are not presented as runnable demos here.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def configure_environment() -> None:
    """Load environment variables from the repository-level ``.env`` file."""
    load_dotenv(REPOSITORY_ROOT / ".env")


def build_chat_model():
    """Create the Groq chat model used by the live model and agent examples.

    Set ``GROQ_MODEL`` to select a different model available to your Groq
    account. The API key is read by LangChain from ``GROQ_API_KEY``.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to the repository-root .env file "
            "or export it before running a demo that calls Groq."
        )

    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=0,
    )


def demo_document() -> None:
    """Show a LangChain ``Document`` and its retrieval-related metadata."""
    from langchain_core.documents import Document

    document = Document(
        id="doc-001",
        page_content="LangChain provides composable building blocks.",
        metadata={"source": "course_notes", "chapter": 1},
    )
    print("Document content:", document.page_content)
    print("Document metadata:", document.metadata)
    print("Document ID:", document.id)


def demo_chat() -> None:
    """Compare one-shot, batched, and streamed calls to the Groq chat model."""
    model = build_chat_model()

    print("Single request (invoke):")
    response = model.invoke("Explain semantic search in two sentences.")
    print(response.content)

    print("\nIndependent requests (batch):")
    responses = model.batch(
        [
            "Define embeddings.",
            "Define vector store.",
            "Define retriever.",
        ]
    )
    for response in responses:
        print(response.content)

    print("\nIncremental response (stream):")
    for chunk in model.stream("Explain LCEL briefly."):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


def demo_messages() -> None:
    """Show message roles and trim an example conversation to recent turns."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.messages import trim_messages

    messages = [
        SystemMessage(content="You are a concise teacher."),
        HumanMessage(content="What is a vector store?"),
    ]
    print("Message roles:")
    for message in messages:
        print(f"{type(message).__name__}: {message.content}")

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi! How can I help you?"),
        HumanMessage(content="What is LangChain?"),
        AIMessage(content="LangChain builds applications powered by LLMs."),
        HumanMessage(content="What is RAG?"),
        AIMessage(content="RAG combines retrieval with generation."),
        HumanMessage(content="Teach me RAG with an example."),
    ]
    recent_history = trim_messages(
        history,
        max_tokens=4,
        strategy="last",
        token_counter=len,
    )

    print("\nRecent conversation (each message counts as one token):")
    for message in recent_history:
        print(f"{type(message).__name__}: {message.content}")


def demo_prompts() -> None:
    """Format text and chat prompts, including history and a partial prompt."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
    )

    text_prompt = PromptTemplate.from_template(
        "Explain {topic} to a {level} learner."
    )
    print(
        "Formatted text prompt:",
        text_prompt.format(topic="embeddings", level="beginner"),
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI teacher. Explain at {level} level."),
            ("human", "{question}"),
        ]
    )
    print(
        "\nFormatted chat messages:",
        chat_prompt.invoke(
            {"level": "advanced", "question": "What is LCEL?"}
        ).to_messages(),
    )

    history_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a tutor."),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    print(
        "\nChat prompt with history:",
        history_prompt.invoke(
            {
                "history": [
                    HumanMessage(content="I am learning LangChain."),
                    AIMessage(content="Great."),
                ],
                "question": "Teach retrievers.",
            }
        ).to_messages(),
    )

    partial_prompt = chat_prompt.partial(level="expert")
    print(
        "\nPartial prompt with the level filled in:",
        partial_prompt.invoke({"question": "Explain MMR."}).to_messages(),
    )


def demo_parsers() -> None:
    """Compare string, JSON, Pydantic, and native structured model outputs."""
    from pydantic import BaseModel, Field

    from langchain_core.output_parsers import (
        JsonOutputParser,
        PydanticOutputParser,
        StrOutputParser,
    )
    from langchain_core.prompts import ChatPromptTemplate

    model = build_chat_model()

    string_chain = model | StrOutputParser()
    print("String output:")
    print(string_chain.invoke("What is RAG?"))

    json_parser = JsonOutputParser()
    json_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Return the answer only as JSON.\n{format_instructions}",
            ),
            ("human", "Extract name and skills from: {text}"),
        ]
    ).partial(format_instructions=json_parser.get_format_instructions())
    json_chain = json_prompt | model | json_parser
    print("\nJSON parser output:")
    print(
        json_chain.invoke(
            {"text": "Sunny knows Python, LangChain and RAG."}
        )
    )

    class Person(BaseModel):
        """The fields that the Pydantic parser should extract."""

        name: str = Field(description="Name of the person")
        age: int = Field(description="Age of the person")
        skills: list[str] = Field(description="Technical skills of the person")

    pydantic_parser = PydanticOutputParser(pydantic_object=Person)
    pydantic_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract the information.\n{format_instructions}",
            ),
            ("human", "{text}"),
        ]
    ).partial(format_instructions=pydantic_parser.get_format_instructions())
    pydantic_chain = pydantic_prompt | model | pydantic_parser
    print("\nPydantic parser output:")
    print(
        pydantic_chain.invoke(
            {"text": "Sunny is 31 and knows Python, LangChain and RAG."}
        )
    )

    class CityDetails(BaseModel):
        """A schema returned directly by the chat model."""

        city: str
        country: str
        population_note: str | None = None

    print("\nNative structured output:")
    print(
        model.with_structured_output(CityDetails).invoke(
            "Give details for Bengaluru."
        )
    )


def demo_tools() -> None:
    """Bind a calculator tool and let an agent call it to answer a question."""
    from langchain.agents import create_agent
    from langchain.tools import tool

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search a small placeholder knowledge base."""
        return f"Dummy result for query: {query}"

    print("Direct tool invocation:", multiply.invoke({"a": 6, "b": 7}))

    model = build_chat_model()
    tool_bound_model = model.bind_tools([multiply])
    tool_request = tool_bound_model.invoke("What is 12 * 8?")
    print("\nTool calls proposed by the model:", tool_request.tool_calls)

    agent = create_agent(
        model=model,
        tools=[multiply, search_knowledge_base],
        system_prompt="You are helpful. Use tools when needed.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 9 * 9?"}]}
    )
    print("\nAgent answer:", result["messages"][-1].content)


def demo_agents() -> None:
    """Return a typed structured answer from an agent that has a search tool."""
    from pydantic import BaseModel

    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain.tools import tool

    class ResearchAnswer(BaseModel):
        """Expected shape for the agent's final response."""

        answer: str
        confidence: str
        sources_used: list[str]

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search a small placeholder knowledge base for the given query."""
        return (
            f"Course notes for '{query}': RAG retrieves relevant content "
            "before asking a language model to generate a grounded answer."
        )

    agent = create_agent(
        model=build_chat_model(),
        tools=[search_knowledge_base],
        response_format=ToolStrategy(ResearchAnswer),
        system_prompt="Answer from tool results and be clear about uncertainty.",
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is RAG? Use the knowledge base.",
                }
            ]
        }
    )
    print("Structured agent response:", result.get("structured_response"))


def demo_middleware() -> None:
    """Add before/after hooks to show where middleware wraps model calls."""
    from langchain.agents import AgentState, create_agent
    from langchain.agents.middleware import after_model, before_model
    from langgraph.runtime import Runtime

    @before_model
    def log_before(state: AgentState, runtime: Runtime) -> None:
        """Run immediately before the agent calls its model."""
        del runtime
        print(
            "Middleware: model is about to run "
            f"({len(state['messages'])} message(s))"
        )

    @after_model
    def log_after(state: AgentState, runtime: Runtime) -> None:
        """Run immediately after the agent receives the model response."""
        del runtime
        print(f"Middleware: model finished ({len(state['messages'])} message(s))")

    agent = create_agent(
        model=build_chat_model(),
        tools=[],
        middleware=[log_before, log_after],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is RAG?"}]}
    )
    print("Agent answer:", result["messages"][-1].content)


def demo_deep_agent() -> None:
    """Run the notebook's small Deep Agents example with Groq.

    This optional demo needs the separate ``deepagents`` package. Its search
    tool intentionally returns a clearly labeled placeholder result.
    """
    from langchain.tools import tool

    try:
        create_deep_agent = import_module("deepagents").create_deep_agent
    except ModuleNotFoundError as error:
        if error.name != "deepagents":
            raise
        raise RuntimeError(
            "The deep-agent demo requires the optional 'deepagents' package. "
            "Install it with: python -m pip install -U deepagents"
        ) from error

    @tool
    def search_web(query: str) -> str:
        """Return a placeholder search result for the requested query."""
        return f"Dummy search result for: {query}"

    agent = create_deep_agent(
        model=build_chat_model(),
        tools=[search_web],
        system_prompt=(
            "You are a research assistant. Research the topic, organize "
            "the information, and provide a clear final report."
        ),
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Research Agentic AI and create a short report.",
                }
            ]
        }
    )
    print("Deep Agent report:", result["messages"][-1].content)


DEMO_FUNCTIONS: dict[str, Callable[[], None]] = {
    "document": demo_document,
    "chat": demo_chat,
    "messages": demo_messages,
    "prompts": demo_prompts,
    "parsers": demo_parsers,
    "tools": demo_tools,
    "agents": demo_agents,
    "middleware": demo_middleware,
    "deep-agent": demo_deep_agent,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line options and run the selected notebook demo."""
    parser = argparse.ArgumentParser(
        description="Run individual LangChain tutorial examples using Groq."
    )
    parser.add_argument(
        "--demo",
        choices=[*DEMO_FUNCTIONS, "all"],
        default="chat",
        help=(
            "Example to run. 'all' runs every core example except the optional "
            "Deep Agents example. Default: chat."
        ),
    )
    args = parser.parse_args(argv)
    configure_environment()

    demos = (
        [
            name
            for name in DEMO_FUNCTIONS
            if name != "deep-agent"
        ]
        if args.demo == "all"
        else [args.demo]
    )
    for name in demos:
        print(f"\n{'=' * 12} {name.upper()} {'=' * 12}")
        DEMO_FUNCTIONS[name]()


if __name__ == "__main__":
    main()
