"""Runnable, debugger-friendly companion to the Class 51 notebook
``Class-51-04-oct-2026-langchain-new-version-recap/langchain_latest_version.ipynb``.

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

The notebook's image example uses a placeholder URL; this script constructs a
multimodal message for inspection but does not send that placeholder to Groq.
The notebook's extensive Deep Agents setup is implemented as an optional,
full-configuration demo. Run ``--demo list`` to see the lessons included in
this companion script.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents.middleware import ModelRequest
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


@dataclass
class UserContext:
    """Trusted, per-invocation user context passed to runtime-aware tools."""

    user_id: str
    role: str


@tool
def show_profile(runtime: ToolRuntime[UserContext]) -> str:
    """Show the currently authenticated user's ID and role.

    The runtime argument is injected by LangChain when the tool is called from
    an agent. The model does not supply the user ID or role; the application
    passes those trusted values as ``context=UserContext(...)`` to ``invoke``.
    """
    return (
        f"User ID: {runtime.context.user_id}, "
        f"Role: {runtime.context.role}"
    )


def configure_environment() -> None:
    """Load environment variables from the repository-level ``.env`` file."""
    load_dotenv(REPOSITORY_ROOT / ".env")


def build_chat_model(model_name: str | None = None):
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
        model=model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=0,
    )


def demo_architecture() -> None:
    """Explain the LangChain package roles from the notebook.

    This local lesson introduces the abstraction boundaries before running
    code: Core contains shared interfaces and LCEL; integrations connect
    providers; community contains additional integrations; LangGraph handles
    stateful control flow; Deep Agents adds higher-level agent workflows; and
    LangSmith provides tracing and evaluation.
    """
    print("langchain-core     -> messages, prompts, Runnables, output parsers")
    print("langchain          -> high-level agents and application components")
    print("provider packages  -> model integrations (this script uses langchain-groq)")
    print("langchain-community-> community document loaders and integrations")
    print("langgraph          -> explicit stateful graphs and agent runtime")
    print("deepagents         -> optional higher-level long-running agent harness")
    print("langsmith          -> optional tracing, testing, and evaluation")


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
    print("Model type:", type(model).__name__)
    print("Model profile:", getattr(model, "profile", None))

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
    """Show system, human, AI, tool, multimodal, and trimmed messages.

    LangChain chat history is a sequence of typed messages. A tool response
    uses ``ToolMessage`` and must carry the matching ``tool_call_id``. Newer
    multimodal messages can use content blocks; the image URL below is only a
    construction example and is not sent to a model.

    Debug steps: inspect each message's class, ``content``, and metadata.
    Verify that every tool result's ``tool_call_id`` matches an AI tool call
    before passing a message history back to a model.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
        trim_messages,
    )

    messages = [
        SystemMessage(content="You are a concise teacher."),
        HumanMessage(content="What is a vector store?"),
    ]
    print("Message roles:")
    for message in messages:
        print(f"{type(message).__name__}: {message.content}")

    tool_message = ToolMessage(
        content="42",
        tool_call_id="multiply-call-001",
        name="multiply",
    )
    print(
        "Tool result message:",
        type(tool_message).__name__,
        tool_message.content,
        tool_message.tool_call_id,
    )

    multimodal_message = HumanMessage(
        content_blocks=[
            {"type": "text", "text": "Describe this image."},
            {"type": "image", "url": "https://example.com/image.jpg"},
        ]
    )
    print("Multimodal content blocks:", multimodal_message.content_blocks)

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
    """Show direct tool use, manual tool-call execution, and agent use.

    ``@tool`` uses the function name, docstring, and type annotations to create
    a model-facing schema. ``tool.invoke`` executes a tool directly.
    ``bind_tools`` only asks the model to propose a tool call: in the manual
    flow below, application code executes the named tool and replies with a
    ``ToolMessage`` that carries the exact ``tool_call_id``. ``create_agent``
    performs this loop automatically.

    Debug steps: inspect ``tool_request.tool_calls`` before executing anything.
    Verify the requested name and arguments; step through the local dispatch;
    then inspect the ToolMessage and final response. Never execute an
    unvalidated model-proposed operation in a production application.
    """
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage, ToolMessage
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
    initial_message = HumanMessage(content="What is 12 * 8?")
    tool_request = tool_bound_model.invoke([initial_message])
    print("\nTool calls proposed by the model:", tool_request.tool_calls)

    tool_results = []
    for call in tool_request.tool_calls:
        if call["name"] != multiply.name:
            raise ValueError(f"Unexpected tool requested: {call['name']}")
        tool_result = multiply.invoke(call["args"])
        tool_results.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=call["id"],
                name=call["name"],
            )
        )

    if tool_results:
        final_message = model.invoke([initial_message, tool_request, *tool_results])
        print("Manual tool result messages:", tool_results)
        print("Manual tool-call round-trip answer:", final_message.content)
    else:
        print("The model did not request the calculator for this prompt.")

    agent = create_agent(
        model=model,
        tools=[multiply, search_knowledge_base],
        system_prompt="You are helpful. Use tools when needed.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 9 * 9?"}]}
    )
    print("\nAgent answer:", result["messages"][-1].content)


def demo_show_profile() -> None:
    """Teach ToolRuntime and ``show_profile`` with trusted invocation context.

    ``context_schema`` declares the per-run context type. The application,
    not the model, supplies the authenticated user's ID and role in
    ``agent.invoke(..., context=UserContext(...))``. LangChain injects a
    ``ToolRuntime`` object into ``show_profile`` when the agent calls it.
    This is the safe pattern for identity or permission data: do not ask an
    LLM to guess it from the user prompt.

    Debug steps: set breakpoints on the application context construction and
    the ``show_profile`` return. Confirm the runtime context matches the
    authenticated values and that the model never controls those fields.
    """
    agent = create_profile_agent(build_chat_model())
    logged_in_user = {"id": "101", "role": "admin"}
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Show my profile."}
            ]
        },
        context=UserContext(
            user_id=logged_in_user["id"],
            role=logged_in_user["role"],
        ),
    )
    print("Agent response:", response["messages"][-1].content)


def create_profile_agent(model: BaseChatModel):
    """Create the context-aware profile agent, injectable for offline tests."""
    from langchain.agents import create_agent

    return create_agent(
        model=model,
        tools=[show_profile],
        system_prompt="Use show_profile to answer questions about the current user.",
        context_schema=UserContext,
    )


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


def demo_structured_agents() -> None:
    """Compare automatic, provider-native, and tool-based agent output modes.

    ``response_format=ResearchAnswer`` lets LangChain choose a strategy from
    model capabilities. ``ProviderStrategy`` asks the provider's native
    structured-output API to enforce a schema. ``ToolStrategy`` represents
    the schema as a tool call and is the path invoked in this Groq demo,
    avoiding the JSON-mode/tool-calling conflict some Groq models reject.

    Debug steps: inspect the schema fields, then inspect each agent's
    ``response_format`` before invocation. The live call uses the tool
    strategy; inspect ``result["structured_response"]`` and check each field's
    type and content.
    """
    from pydantic import BaseModel

    from langchain.agents import create_agent
    from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
    from langchain.tools import tool

    class ResearchAnswer(BaseModel):
        """Structured contract for the research answer."""

        answer: str
        confidence: str
        sources_used: list[str]

    @tool
    def search_knowledge_base(query: str) -> str:
        """Return a sample course-note result for the supplied query."""
        return (
            f"Course note for {query!r}: RAG retrieves relevant source "
            "documents and uses them as context for generation."
        )

    model = build_chat_model()
    automatic_agent = create_agent(
        model=model,
        tools=[],
        response_format=ResearchAnswer,
    )
    provider_agent = create_agent(
        model=model,
        tools=[],
        response_format=ProviderStrategy(ResearchAnswer),
    )
    tool_agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        response_format=ToolStrategy(ResearchAnswer),
        system_prompt="Use the knowledge-base result as evidence.",
    )
    print(
        "Constructed structured agents:",
        {
            "automatic": type(automatic_agent).__name__,
            "provider": type(provider_agent).__name__,
            "tool": type(tool_agent).__name__,
        },
    )
    result = tool_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is RAG? Use the knowledge base.",
                }
            ]
        }
    )
    print("Tool-strategy result:", result.get("structured_response"))


def demo_middleware() -> None:
    """Add before/after hooks and observe their agent lifecycle ordering.

    ``before_model`` runs immediately before each model call; ``after_model``
    runs after a model response is recorded. Hooks may return state updates,
    but this logging example returns ``None`` and changes no state.

    Debug steps: place breakpoints in both callbacks, inspect
    ``state["messages"]``, and compare the final message list with the message
    counts printed by each hook.
    """
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


def demo_dynamic_prompt() -> None:
    """Select a system prompt from trusted runtime context.

    ``@dynamic_prompt`` is middleware that returns a system instruction for
    the current invocation. Here a beginner gets a plain-language prompt; an
    advanced user gets a technical prompt. ``context_schema`` validates the
    context shape and the role is passed by the application.

    Debug steps: break in ``role_prompt`` and inspect
    ``request.runtime.context.role``. Invoke with each role and compare the
    system message in the resulting ``messages`` list.
    """
    from dataclasses import dataclass

    from langchain.agents import create_agent
    from langchain.agents.middleware import dynamic_prompt

    @dataclass
    class TeachingContext:
        role: str

    @dynamic_prompt
    def role_prompt(request: ModelRequest) -> str:
        """Return role-specific instructions before the model is called."""
        if request.runtime.context.role == "beginner":
            return "Explain concepts in simple English with one short example."
        return "Explain concepts precisely using technical terminology."

    agent = create_agent(
        model=build_chat_model(),
        tools=[],
        middleware=[role_prompt],
        context_schema=TeachingContext,
    )
    for role in ("beginner", "expert"):
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "What is RAG?"}]},
            context=TeachingContext(role=role),
        )
        print(f"{role} answer:", response["messages"][-1].content)


def demo_model_router() -> None:
    """Route agent model calls based on conversation length.

    ``@wrap_model_call`` runs around a model call. The handler continues the
    call with ``request.override(model=...)``. This demo chooses the regular
    Groq model for short histories and a stronger Groq model for longer ones.
    Set ``GROQ_STRONG_MODEL`` to a model enabled for your Groq account.

    Debug steps: inspect ``len(request.state["messages"])`` and ``selected``.
    The two calls below deliberately exercise both sides of the threshold.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call

    primary_model = build_chat_model()
    strong_model = build_chat_model(
        os.getenv("GROQ_STRONG_MODEL", os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    )

    @wrap_model_call
    def choose_model(request, handler):
        """Use the stronger configured model when the history exceeds 10."""
        message_count = len(request.state["messages"])
        selected = strong_model if message_count > 10 else primary_model
        print(f"Model router: {message_count} messages -> {selected.model}")
        return handler(request.override(model=selected))

    agent = create_agent(
        model=primary_model,
        tools=[],
        middleware=[choose_model],
    )
    for messages in (
        [{"role": "user", "content": "Explain RAG in simple English."}],
        [
            {"role": "user", "content": f"Previous message {index}"}
            for index in range(11)
        ],
    ):
        response = agent.invoke({"messages": messages})
        print("Routed answer:", response["messages"][-1].content)


def demo_prebuilt_middleware() -> None:
    """Configure every prebuilt middleware family shown in the notebook.

    This is a configuration lesson: it constructs but does not execute
    middleware. Summarization compacts older history; human-in-the-loop pauses
    selected tool calls; call limits bound loops; fallback selects another
    model after failure; PII middleware detects/redacts sensitive values;
    TodoList supports task planning; tool selection narrows a large tool list;
    retry middleware retries transient model/tool failures; and context editing
    removes older tool results.

    Debug steps: step over each constructor and inspect its configuration.
    To run one in an application, pass the instance in ``create_agent``'s
    ``middleware=[...]`` argument and test its interrupt/error behavior in an
    isolated environment before production.
    """
    from langchain.agents.middleware import (
        ClearToolUsesEdit,
        ContextEditingMiddleware,
        HumanInTheLoopMiddleware,
        LLMToolSelectorMiddleware,
        ModelCallLimitMiddleware,
        ModelFallbackMiddleware,
        ModelRetryMiddleware,
        PIIMiddleware,
        SummarizationMiddleware,
        TodoListMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
    )

    model_name = f"groq:{os.getenv('GROQ_MODEL', DEFAULT_GROQ_MODEL)}"
    fallback_name = f"groq:{os.getenv('GROQ_FALLBACK_MODEL', DEFAULT_GROQ_MODEL)}"
    middleware = {
        "summarization": SummarizationMiddleware(
            model=model_name,
            trigger=("messages", 40),
            keep=("messages", 10),
        ),
        "human_approval": HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_record": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            }
        ),
        "model_call_limit": ModelCallLimitMiddleware(
            run_limit=5,
            exit_behavior="end",
        ),
        "tool_call_limit": ToolCallLimitMiddleware(
            tool_name="multiply",
            run_limit=3,
        ),
        "model_fallback": ModelFallbackMiddleware(model_name, fallback_name),
        "pii_redaction": PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        "todo_list": TodoListMiddleware(),
        "tool_selector": LLMToolSelectorMiddleware(
            model=model_name,
            max_tools=2,
        ),
        "tool_retry": ToolRetryMiddleware(max_retries=2),
        "model_retry": ModelRetryMiddleware(max_retries=2),
        "context_editing": ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=2000, keep=3)]
        ),
    }
    for name, configured in middleware.items():
        print(f"{name}: {type(configured).__name__}")


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


def demo_deep_agent_full(*, invoke_agent: bool) -> None:
    """Set up every advanced Deep Agents feature from the notebook.

    The complete setup includes dummy document/database/search tools, dummy
    side-effect tools, specialist subagents, temporary skill and memory files,
    a filesystem backend, write permissions that request approval, structured
    output, trusted user context, an in-memory checkpointer, and a long-term
    store. Temporary files are created only inside a managed temporary
    directory and removed when this function returns.

    This optional example needs ``deepagents`` and its filesystem middleware
    dependencies. Configuration is built without an LLM request by default.
    Pass ``--invoke-deep-agent`` to send a request to Groq. The email/delete
    tools are simulated and do not contact external services or delete files.

    Debug steps: inspect the temporary workspace, skills, memory, backend,
    permissions, subagent tool lists, and final ``create_deep_agent`` kwargs.
    If invoking, inspect messages for tool calls and interrupts before
    inspecting the structured report.
    """
    import tempfile

    from pydantic import BaseModel, Field

    try:
        deepagents = import_module("deepagents")
        filesystem_backend_module = import_module("deepagents.backends.filesystem")
        filesystem_permission_module = import_module(
            "deepagents.middleware.filesystem"
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The full Deep Agents demo requires deepagents and its filesystem "
            "middleware. Install with: python -m pip install -U deepagents"
        ) from error

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    FilesystemBackend = filesystem_backend_module.FilesystemBackend
    FilesystemPermission = filesystem_permission_module.FilesystemPermission
    create_deep_agent = deepagents.create_deep_agent

    @tool
    def search_docs(query: str) -> str:
        """Search internal course documents (placeholder implementation)."""
        return (
            f"Document result for {query!r}: LangChain builds LLM applications; "
            "LangGraph provides stateful orchestration; Deep Agents adds "
            "long-running knowledge-work workflows."
        )

    @tool
    def search_database(query: str) -> str:
        """Search a demonstration company database without external access."""
        return f"Database result for {query!r}: found 3 dummy records."

    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Simulate sending an email; this demo never sends real email."""
        del body
        return f"Dummy email prepared for {to!r} with subject {subject!r}."

    @tool
    def delete_file(path: str) -> str:
        """Simulate a delete request; this tool does not delete any file."""
        return f"Dummy delete request for {path!r}; no file was removed."

    class ResearchReport(BaseModel):
        """Structured format for the full agent's final research report."""

        summary: str = Field(description="Short research summary")
        key_findings: list[str] = Field(description="Important findings")
        sources: list[str] = Field(description="Tools or sources used")
        confidence: str = Field(description="Confidence: low, medium, or high")

    with tempfile.TemporaryDirectory(prefix="langchain-deep-agent-") as directory:
        workspace = Path(directory)
        research_skill = workspace / "skills" / "research" / "SKILL.md"
        finance_skill = workspace / "skills" / "finance" / "SKILL.md"
        memory_file = workspace / "memory" / "AGENTS.md"
        for file_path in (research_skill, finance_skill, memory_file):
            file_path.parent.mkdir(parents=True, exist_ok=True)
        research_skill.write_text(
            "# Research skill\n1. Clarify the question.\n"
            "2. Search documents and database.\n3. Compare evidence.\n",
            encoding="utf-8",
        )
        finance_skill.write_text(
            "# Finance skill\nCheck revenue, expenses, trends, and risks.\n",
            encoding="utf-8",
        )
        memory_file.write_text(
            "# Agent memory\nPrefer concise technical answers with examples.\n",
            encoding="utf-8",
        )

        researcher = {
            "name": "researcher",
            "description": "Research technical topics in course documents and database.",
            "system_prompt": (
                "Research the topic carefully. Use search tools when useful. "
                "Return concise factual findings."
            ),
            "tools": [search_docs, search_database],
            "model": build_chat_model(),
        }
        analyst = {
            "name": "analyst",
            "description": "Analyze research findings and draw evidence-based conclusions.",
            "system_prompt": (
                "Compare findings, identify key conclusions, and state uncertainty."
            ),
            "tools": [search_database],
            "model": build_chat_model(),
        }
        backend = FilesystemBackend(
            root_dir=workspace,
            virtual_mode=True,
        )
        permissions = [
            FilesystemPermission(
                operations=["write"],
                paths=["/memory/**"],
                mode="interrupt",
            )
        ]
        checkpointer = InMemorySaver()
        store = InMemoryStore()

        agent = create_deep_agent(
            model=build_chat_model(),
            tools=[search_docs, search_database, send_email, delete_file],
            system_prompt=(
                "You are a careful research analyst. Gather evidence, use "
                "specialists where useful, and produce a factual report. "
                "Do not invoke side-effect tools unless explicitly requested."
            ),
            subagents=[researcher, analyst],
            skills=["/skills/"],
            memory=["/memory/AGENTS.md"],
            backend=backend,
            permissions=permissions,
            interrupt_on={"send_email": True, "delete_file": True},
            response_format=ResearchReport,
            context_schema=UserContext,
            checkpointer=checkpointer,
            store=store,
            name="research_agent",
        )
        print(
            "Configured Deep Agent:",
            {
                "workspace": str(workspace),
                "subagents": ["researcher", "analyst"],
                "skills": ["/skills/"],
                "memory": ["/memory/AGENTS.md"],
                "permissions": ["write /memory/** requires approval"],
                "interrupt_on": ["send_email", "delete_file"],
                "response_format": ResearchReport.__name__,
                "name": "research_agent",
            },
        )

        if not invoke_agent:
            print(
                "Agent invocation skipped. Add --invoke-deep-agent to send a "
                "Groq request while this temporary workspace is active."
            )
            return

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Research the roles of LangChain, LangGraph, and "
                            "Deep Agents. Use available research tools and "
                            "return a short report."
                        ),
                    }
                ]
            },
            context=UserContext(user_id="student-001", role="student"),
            config={"configurable": {"thread_id": "deep-agent-demo"}},
        )
        print("Deep Agent messages:", result.get("messages"))
        print("Structured report:", result.get("structured_response"))


def demo_langsmith() -> None:
    """Explain LangSmith tracing without exposing credentials.

    Set ``LANGSMITH_TRACING=true``, ``LANGSMITH_API_KEY`` and optionally
    ``LANGSMITH_PROJECT`` in the repository-root ``.env`` file to enable
    tracing. This demo reports whether values are configured but never prints
    the key. LangChain integrations can then send traces for supported
    invocations to the selected LangSmith project.
    """
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").casefold() in {
        "true",
        "1",
        "yes",
    }
    key_configured = bool(os.getenv("LANGSMITH_API_KEY"))
    project = os.getenv("LANGSMITH_PROJECT", "default project")
    print("Tracing enabled:", tracing_enabled)
    print("API key configured:", key_configured)
    print("Project:", project)
    if not tracing_enabled or not key_configured:
        print(
            "To enable tracing, configure LANGSMITH_TRACING=true and "
            "LANGSMITH_API_KEY in the repository-root .env file."
        )


DEMO_FUNCTIONS: dict[str, Callable[..., None]] = {
    "architecture": demo_architecture,
    "document": demo_document,
    "chat": demo_chat,
    "messages": demo_messages,
    "prompts": demo_prompts,
    "parsers": demo_parsers,
    "tools": demo_tools,
    "show-profile": demo_show_profile,
    "agents": demo_agents,
    "structured-agents": demo_structured_agents,
    "middleware": demo_middleware,
    "dynamic-prompt": demo_dynamic_prompt,
    "model-router": demo_model_router,
    "prebuilt-middleware": demo_prebuilt_middleware,
    "deep-agent": demo_deep_agent,
    "deep-agent-full": demo_deep_agent_full,
    "langsmith": demo_langsmith,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line options and run the selected notebook demo."""
    parser = argparse.ArgumentParser(
        description="Run individual LangChain tutorial examples using Groq."
    )
    parser.add_argument(
        "--demo",
        choices=[*DEMO_FUNCTIONS, "all", "list"],
        default="chat",
        help=(
            "Example to run. 'all' runs core examples and excludes optional "
            "Deep Agents. 'list' prints every lesson. Default: chat."
        ),
    )
    parser.add_argument(
        "--invoke-deep-agent",
        action="store_true",
        help="Invoke the full optional Deep Agent (requires credentials).",
    )
    args = parser.parse_args(argv)
    configure_environment()

    if args.invoke_deep_agent and args.demo != "deep-agent-full":
        parser.error("--invoke-deep-agent can only be used with --demo deep-agent-full")

    if args.demo == "list":
        for name in DEMO_FUNCTIONS:
            print(name)
        return

    demos = (
        [
            name
            for name in DEMO_FUNCTIONS
            if name not in {"deep-agent", "deep-agent-full"}
        ]
        if args.demo == "all"
        else [args.demo]
    )
    for name in demos:
        print(f"\n{'=' * 12} {name.upper()} {'=' * 12}")
        if name == "deep-agent-full":
            demo_deep_agent_full(invoke_agent=args.invoke_deep_agent)
        else:
            DEMO_FUNCTIONS[name]()



if __name__ == "__main__":
    main()
