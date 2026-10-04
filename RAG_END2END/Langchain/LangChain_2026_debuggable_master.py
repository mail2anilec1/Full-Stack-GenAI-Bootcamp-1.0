"""Debuggable Python companion to ``LangChain_2026_COMPLETE_Master_Notebook_v2.ipynb``.

This file turns the notebook's main themes into small, named, repeatable
demos. Read each function's docstring before stepping through its body. The
script is intentionally safe to run without credentials: API-backed model
calls are disabled unless ``--live`` is passed.

Run from the repository root:

    python RAG_END2END/Langchain/LangChain_2026_debuggable_master.py --help
    python RAG_END2END/Langchain/LangChain_2026_debuggable_master.py --demo fundamentals
    python RAG_END2END/Langchain/LangChain_2026_debuggable_master.py --demo all

The notebook's ``CHAT_MODEL`` and ``EMBED_MODEL`` settings are retained:

    LANGCHAIN_MODEL=openai:gpt-5.4-mini
    LANGCHAIN_EMBED_MODEL=openai:text-embedding-3-small

Change either setting to a provider/model identifier supported by your
installed LangChain integration. Add that provider's API key to the
repository-root ``.env`` file. The default offline embedding implementation
is a deterministic, hashed bag-of-words representation for teaching vector
store interfaces only; it is not a substitute for a production semantic
embedding model. Pass ``--provider-embeddings`` to use the configured provider
embedding model instead.

VS Code debugging quick start:

1. Open the repository root and select the Python environment with LangChain
   installed.
2. Open this file and set a breakpoint inside the demo you want to understand.
3. Press F5 and start with ``--demo fundamentals``. Use F10 to step over,
   F11 to step into, Shift+F11 to step out, and inspect Variables/Call Stack.
4. For external model calls, add ``--live`` only after configuring credentials.
5. To inspect LangChain internals, set ``justMyCode`` to false in launch.json.

Every ``demo_*`` function is a standalone lesson. ``main`` loads environment
variables, chooses the requested lesson, and controls optional live calls.
The notebook's MCP servers, actual web search, production database, and
Deep Agents are environment-specific integrations; they are explained in the
printed roadmap but are not launched by this safe local tutorial.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain.tools import ToolRuntime
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CHAT_MODEL = "openai:gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "openai:text-embedding-3-small"
HASH_DIMENSIONS = 256


class LocalHashEmbeddings(Embeddings):
    """Produce deterministic hashed bag-of-words vectors without an API key.

    Each token contributes to one signed dimension chosen by a stable hash.
    This makes examples repeatable and allows an in-memory vector store to run
    offline. Hash collisions and lack of semantic understanding mean this
    implementation is for illustrating APIs only, not for real RAG quality.
    """

    def __init__(self, dimensions: int = HASH_DIMENSIONS) -> None:
        """Create embeddings with ``dimensions`` numeric features per text."""
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        """Hash normalized word tokens into a normalized fixed-size vector."""
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"\b[\w'-]+\b", text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            feature = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[feature] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each document body; vector order matches input order."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query in the same vector space as the documents."""
        return self._embed(text)


class NormalizeState(TypedDict):
    """State keys used by the simple LangGraph normalization demo."""

    text: str
    normalized: str


class NumberState(TypedDict):
    """State used by the conditional even/odd LangGraph demo."""

    number: int
    result: str


class MemoryState(TypedDict, total=False):
    """State fields used to demonstrate checkpointed conversational history."""

    history: list[str]
    new_message: str


@dataclass
class UserContext:
    """Trusted caller information supplied to tools through agent runtime."""

    user_id: str
    role: str = "student"


def print_heading(title: str) -> None:
    """Print a visible divider so separate demos are easy to follow."""
    print(f"\n{'=' * 16} {title} {'=' * 16}")


def create_chat_model():
    """Construct the notebook-configured chat model on demand.

    ``init_chat_model`` selects the provider from the ``provider:model`` name
    in ``LANGCHAIN_MODEL``. Creating the object does not normally send a
    request; ``invoke``, ``batch`` or ``stream`` performs the API call.
    """
    from langchain.chat_models import init_chat_model

    model_name = os.getenv("LANGCHAIN_MODEL", DEFAULT_CHAT_MODEL)
    return init_chat_model(model_name)


def create_embeddings(*, provider_embeddings: bool) -> Embeddings:
    """Return local teaching embeddings or the configured provider embeddings.

    Provider-backed embedding models make API requests when text is embedded
    and require the selected provider integration and credentials. Local
    embeddings are deterministic and useful for following the vector-store
    code, but do not have the semantic quality of a trained model.
    """
    if not provider_embeddings:
        return LocalHashEmbeddings()

    from langchain.embeddings import init_embeddings

    embedding_name = os.getenv(
        "LANGCHAIN_EMBED_MODEL",
        DEFAULT_EMBEDDING_MODEL,
    )
    return init_embeddings(embedding_name)


def demo_fundamentals(*, live: bool) -> None:
    """Explore Documents, messages, prompts, parsers, and chat-model methods.

    ``Document`` groups text with metadata and an optional ID. A prompt
    template separates stable instructions from per-request variables.
    ``invoke`` handles one input, ``batch`` handles independent inputs, and
    ``stream`` yields partial results. Output parsers convert model messages
    into strings or schema-validated values. Without ``--live``, this demo
    formats prompts and parsers locally but does not contact a model.

    Debug steps: inspect ``doc`` and its metadata; step over
    ``chat_prompt.invoke`` and inspect ``to_messages()``; then, with ``--live``,
    set breakpoints on each model call and inspect message content/chunks.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        trim_messages,
    )
    from langchain_core.output_parsers import (
        JsonOutputParser,
        PydanticOutputParser,
        StrOutputParser,
    )
    from langchain_core.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
    )
    from pydantic import BaseModel, Field

    print_heading("DOCUMENT AND MESSAGES")
    doc = Document(
        id="doc-001",
        page_content="LangChain provides composable building blocks.",
        metadata={"source": "course_notes", "chapter": 1},
    )
    print("Document text:", doc.page_content)
    print("Document metadata:", doc.metadata)
    print("Document id:", doc.id)

    messages = [
        SystemMessage(content="You are a concise teacher."),
        HumanMessage(content="What is a vector store?"),
    ]
    print("Messages are typed by role:", [type(message).__name__ for message in messages])

    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi"),
        HumanMessage(content="Teach RAG"),
        AIMessage(content="RAG retrieves context before generation."),
        HumanMessage(content="Give a short example."),
    ]
    recent_history = trim_messages(
        history,
        max_tokens=3,
        strategy="last",
        token_counter=len,
    )
    print("Trimmed history:", [message.content for message in recent_history])

    print_heading("PROMPT TEMPLATES")
    text_prompt = PromptTemplate.from_template(
        "Explain {topic} to a {level} learner."
    )
    print(
        "PromptTemplate.format:",
        text_prompt.format(topic="embeddings", level="beginner"),
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a teacher. Explain at {level} level."),
            ("human", "{question}"),
        ]
    )
    formatted_chat = chat_prompt.invoke(
        {"level": "advanced", "question": "What is LCEL?"}
    )
    print("ChatPromptTemplate.invoke:", formatted_chat.to_messages())

    history_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a tutor."),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    with_history = history_prompt.invoke(
        {
            "history": [
                HumanMessage(content="I am learning LangChain."),
                AIMessage(content="Great."),
            ],
            "question": "What is a retriever?",
        }
    )
    print("MessagesPlaceholder:", with_history.to_messages())
    print(
        "partial() fixes level once:",
        chat_prompt.partial(level="expert")
        .invoke({"question": "Explain MMR."})
        .to_messages(),
    )

    print_heading("OUTPUT PARSERS")

    class TopicSummary(BaseModel):
        """Fields required when validating a topic summary."""

        topic: str = Field(description="The concept being explained")
        explanation: str = Field(description="A concise explanation")
        difficulty: str = Field(description="Beginner, intermediate, or advanced")

    string_parser = StrOutputParser()
    json_parser = JsonOutputParser()
    pydantic_parser = PydanticOutputParser(pydantic_object=TopicSummary)
    print("StrOutputParser parses AI messages into text.")
    print("JsonOutputParser formatting instructions:", json_parser.get_format_instructions())
    print(
        "PydanticOutputParser formatting instructions:",
        pydantic_parser.get_format_instructions(),
    )

    if not live:
        print("Model requests skipped. Re-run with --live to use invoke/batch/stream.")
        return

    print_heading("CHAT MODEL: invoke, batch, stream")
    model = create_chat_model()
    single_response = model.invoke("Explain semantic search in two sentences.")
    print("invoke() content:", single_response.content)

    batch_responses = model.batch(
        [
            "Define embeddings.",
            "Define a vector store.",
            "Define a retriever.",
        ]
    )
    print("batch() contents:", [response.content for response in batch_responses])

    print("stream() chunks:")
    for chunk in model.stream("Explain LCEL briefly."):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

    parsed_chain = chat_prompt | model | string_parser
    parsed_answer = parsed_chain.invoke(
        {"level": "beginner", "question": "What is a Runnable?"}
    )
    print("Prompt | model | parser result:", parsed_answer)

    class CityDetails(BaseModel):
        """Example schema for model-native structured output."""

        city: str
        country: str
        population_note: str | None = None

    structured_model = model.with_structured_output(CityDetails)
    print(
        "with_structured_output():",
        structured_model.invoke("Give a brief description of Bengaluru."),
    )


def demo_runnables() -> None:
    """Debug the Runnable protocol and common LCEL composition patterns.

    Runnables share methods such as ``invoke``, ``batch``, and ``stream``.
    ``|`` composes sequential steps; ``RunnableParallel`` and dictionary
    literals produce named parallel branches; ``RunnablePassthrough.assign``
    adds derived fields without discarding input; and ``RunnableBranch``
    selects a path using predicates.

    Debug steps: break before each ``invoke``; use F10 and watch the input and
    output shape. Check that keys entering a parallel mapping are the keys
    consumed by the next Runnable. A chain stage receives the previous stage's
    output, not the original input unless it was explicitly passed through.
    """
    from langchain_core.runnables import (
        RunnableBranch,
        RunnableLambda,
        RunnableParallel,
        RunnablePassthrough,
    )

    print_heading("SEQUENTIAL COMPOSITION")
    add_one = RunnableLambda(lambda value: value + 1)
    double = RunnableLambda(lambda value: value * 2)
    sequence = add_one | double
    print("10 -> add one -> double:", sequence.invoke(10))

    print_heading("PARALLEL BRANCHES")
    parallel = RunnableParallel(
        doubled=RunnableLambda(lambda value: value * 2),
        squared=RunnableLambda(lambda value: value**2),
    )
    print("parallel.invoke(5):", parallel.invoke(5))

    dictionary_branches = (
        RunnableLambda(lambda value: value + 1)
        | {
            "times_2": RunnableLambda(lambda value: value * 2),
            "times_5": RunnableLambda(lambda value: value * 5),
        }
    )
    print("Dictionary parallel syntax:", dictionary_branches.invoke(1))

    print_heading("PASSTHROUGH AND ROUTING")
    add_length = RunnablePassthrough.assign(
        length=lambda values: len(values["text"])
    )
    print("assign() keeps original fields:", add_length.invoke({"text": "LangChain"}))

    number_route = RunnableBranch(
        (lambda value: value < 0, RunnableLambda(lambda _: "negative")),
        (lambda value: value == 0, RunnableLambda(lambda _: "zero")),
        RunnableLambda(lambda _: "positive"),
    )
    print(
        "Conditional outputs:",
        [number_route.invoke(value) for value in [-2, 0, 3]],
    )

    print_heading("RETRY AND FALLBACK CONFIGURATION")

    def fail_first_attempt(value: str) -> str:
        """Illustrate where a transient error might be introduced for retry."""
        raise RuntimeError(f"Demonstration failure for input {value!r}")

    retriable = RunnableLambda(fail_first_attempt).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )
    print(
        "with_retry() configured for three attempts; not invoked to avoid "
        "intentionally failing this demo."
    )
    primary = RunnableLambda(lambda value: f"primary:{value}")
    fallback = RunnableLambda(lambda value: f"fallback:{value}")
    resilient = primary.with_fallbacks([fallback])
    print("with_fallbacks() result when primary succeeds:", resilient.invoke("ok"))
    print("Runnable methods available: invoke(input), batch(inputs), stream(input).")
    del retriable


def _sample_documents() -> list[Document]:
    """Build small course-note Documents with traceable source metadata."""
    return [
        Document(
            id="lcel",
            page_content="LCEL composes Runnable components into clear sequences.",
            metadata={"source": "course-notes", "topic": "lcel"},
        ),
        Document(
            id="retrieval",
            page_content="A retriever searches for relevant documents and returns context.",
            metadata={"source": "course-notes", "topic": "retrieval"},
        ),
        Document(
            id="agents",
            page_content="Agent middleware intercepts model and tool lifecycle operations.",
            metadata={"source": "course-notes", "topic": "agents"},
        ),
        Document(
            id="rag",
            page_content="Two-step RAG retrieves evidence before generating an answer.",
            metadata={"source": "course-notes", "topic": "rag"},
        ),
    ]


def load_documents_from_path(path: Path) -> list[Document]:
    """Load a text, PDF, or CSV file into LangChain Documents.

    The loader preserves source metadata where supported. Text, PDF and CSV
    loaders require their corresponding LangChain community integration and
    file dependencies. Unsupported suffixes fail explicitly rather than
    silently returning empty documents.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Document file does not exist: {path}")

    suffix = path.suffix.casefold()
    if suffix in {".txt", ".md"}:
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()
    if suffix == ".csv":
        from langchain_community.document_loaders import CSVLoader

        return CSVLoader(str(path), encoding="utf-8").load()
    raise ValueError(
        f"Unsupported document extension {suffix!r}; use .txt, .md, .pdf, or .csv."
    )


def format_documents(documents: Sequence[Document]) -> str:
    """Format retrieved text with a source label for prompt grounding.

    This formats the actual retrieved records rather than just joining raw
    text. Keeping labels visible makes it easier to debug missing or incorrect
    citations in a RAG answer.
    """
    return "\n\n".join(
        "[source="
        + str(document.metadata.get("source", document.metadata.get("topic", "unknown")))
        + "]\n"
        + document.page_content
        for document in documents
    )


def demo_retrieval(
    *,
    provider_embeddings: bool,
    live: bool,
    document_path: Path | None,
) -> None:
    """Load, split, embed, index, retrieve, and optionally generate an answer.

    The stages are deliberately kept visible:

    1. A loader (or sample data) creates ``Document`` objects.
    2. ``RecursiveCharacterTextSplitter`` creates overlapping chunks while
       preserving each source document's metadata.
    3. An embedding model maps text to fixed-size numeric vectors.
    4. ``InMemoryVectorStore`` indexes the vectors and original Documents.
    5. ``similarity_search`` and a retriever return likely relevant chunks.
    6. A two-step RAG chain passes those chunks as prompt context to a model.

    Local hash embeddings are used by default, so retrieval can be stepped
    through without a provider key. Pass ``--provider-embeddings`` to use
    ``init_embeddings`` and the configured ``LANGCHAIN_EMBED_MODEL`` instead.
    ``--live`` separately enables the final chat generation call.

    Debug steps: inspect ``documents`` before splitting; compare chunk
    contents and metadata; inspect vector dimensions from
    ``embed_query``; then inspect ``retrieved`` before constructing the prompt.
    If results are poor, check the query, chunk boundaries/overlap, embedding
    model, and top-k setting independently.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.vectorstores import InMemoryVectorStore

    print_heading("LOAD DOCUMENTS")
    documents = (
        load_documents_from_path(document_path)
        if document_path is not None
        else _sample_documents()
    )
    print(f"Loaded {len(documents)} document(s).")
    for index, document in enumerate(documents, start=1):
        print(
            f"  {index}. {document.metadata} "
            f"{document.page_content[:100]!r}"
        )

    print_heading("SPLIT DOCUMENTS")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=180,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(documents)} document(s) -> {len(chunks)} chunk(s)")
    for index, chunk in enumerate(chunks, start=1):
        print(f"  chunk {index}: {chunk.metadata} {chunk.page_content!r}")

    print_heading("EMBED AND STORE")
    embeddings = create_embeddings(provider_embeddings=provider_embeddings)
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    query = "How does retrieval find useful context?"
    query_vector = embeddings.embed_query(query)
    print(f"Embedding dimension: {len(query_vector)}")
    print(f"Indexed {len(chunks)} chunks in an in-memory vector store.")

    print_heading("RETRIEVE")
    retrieved = vector_store.similarity_search(query, k=3)
    scored = vector_store.similarity_search_with_score(query, k=2)
    print("similarity_search() results:")
    for document in retrieved:
        print(f"  {document.metadata}: {document.page_content}")
    print(
        "similarity_search_with_score() scores:",
        [(document.metadata.get("topic"), round(score, 4)) for document, score in scored],
    )

    diverse = vector_store.max_marginal_relevance_search(
        query,
        k=2,
        fetch_k=min(4, len(chunks)),
        lambda_mult=0.5,
    )
    print("max_marginal_relevance_search() topics:",
          [document.metadata.get("topic") for document in diverse])

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    retrieved_by_runnable = retriever.invoke(query)
    print("Retriever.invoke(query) returned:", len(retrieved_by_runnable), "documents")

    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": min(4, len(chunks)), "lambda_mult": 0.5},
    )
    print("MMR retriever topics:",
          [document.metadata.get("topic") for document in mmr_retriever.invoke(query)])

    print_heading("TWO-STEP RAG")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only from the supplied context. If it is insufficient, "
                "say that you do not know.",
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}"),
        ]
    )
    rag_prompt_chain = (
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
        | prompt
    )
    prompt_value = rag_prompt_chain.invoke(query)
    print("Formatted prompt messages:", prompt_value.to_messages())
    print(
        "Grounded context preview:",
        format_documents(retrieved_by_runnable)[:350],
    )

    if not live:
        print("Generation skipped. Add --live to send the retrieved context to a model.")
    if live:
        live_rag_chain = rag_prompt_chain | create_chat_model() | StrOutputParser()
        print(
            "RAG chain result:",
            live_rag_chain.invoke(query, config={"tags": ["class-51-rag"]}),
        )


def demo_agents(*, live: bool) -> None:
    """Contrast direct tool execution, tool binding, and a tool-using agent.

    A decorated ``@tool`` function becomes a schema-described action. Calling
    ``tool.invoke`` executes it directly. ``model.bind_tools`` asks a model to
    produce structured tool-call requests but does not itself run arbitrary
    side effects. ``create_agent`` coordinates model/tool cycles. Runtime
    context (``ToolRuntime``) can pass trusted application data to a tool
    without asking the model to invent it.

    Debug steps: first step through direct ``multiply.invoke``; with ``--live``,
    inspect ``AIMessage.tool_calls`` after model binding, then inspect the
    agent's messages for AI tool requests, ToolMessages, and its final answer.
    Never wire destructive production actions without approval middleware.
    """
    from langchain.tools import tool

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and return the exact product."""
        return a * b

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the class's sample notes for a short retrieval result."""
        matches = [
            document
            for document in _sample_documents()
            if any(
                token in document.page_content.casefold()
                for token in query.casefold().split()
            )
        ]
        return format_documents(matches[:2]) if matches else "No matching course note."

    @tool
    def current_user(runtime: ToolRuntime[UserContext]) -> str:
        """Return the caller identity from trusted runtime context."""
        return f"{runtime.context.user_id}:{runtime.context.role}"

    print_heading("DIRECT TOOL INVOCATION")
    print("multiply.invoke({'a': 6, 'b': 7}):", multiply.invoke({"a": 6, "b": 7}))
    print("ToolRuntime context is injected when the tool runs inside an agent.")

    if not live:
        print("Model binding and agent.invoke skipped; add --live to call the model.")
        print(
            "The knowledge-base tool here uses sample notes. Replace its body "
            "with the vector retriever from the retrieval demo for RAG."
        )
        return

    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from pydantic import BaseModel

    model = create_chat_model()
    tool_bound_model = model.bind_tools([multiply])
    proposed_call = tool_bound_model.invoke("What is 12 times 8?")
    print("bind_tools() proposed calls:", proposed_call.tool_calls)

    class ResearchAnswer(BaseModel):
        """Structured agent response returned under ``structured_response``."""

        answer: str
        confidence: str
        sources_used: list[str]

    agent = create_agent(
        model=model,
        tools=[multiply, search_knowledge_base, current_user],
        system_prompt=(
            "Use tools for arithmetic and course knowledge. "
            "Do not claim unsupported sources."
        ),
        response_format=ToolStrategy(ResearchAnswer),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is RAG? Use the course notes."}]},
        context=UserContext(user_id="student-001", role="student"),
        config={"tags": ["class-51-agent"], "metadata": {"lesson": "tool-calling"}},
    )
    print("Agent message types:",
          [type(message).__name__ for message in result["messages"]])
    print("Structured response:", result.get("structured_response"))


def demo_middleware(*, live: bool) -> None:
    """Demonstrate agent middleware hooks around model calls.

    ``before_model`` and ``after_model`` are decorator-based hooks for
    lifecycle logging. ``dynamic_prompt`` derives system instructions from
    trusted runtime context. Prebuilt middleware can add retries, PII
    redaction, call limits, human approval, summarization, or context editing.
    Middleware is useful for cross-cutting policy; it should not replace
    validating inputs or authorizing sensitive actions in application code.

    Debug steps: place a breakpoint in each hook and inspect the state
    message list. Check that the hook's returned state update is intentional.
    This demo is offline unless ``--live`` is passed.
    """
    from langchain.agents import AgentState, create_agent
    from langchain.agents.middleware import after_model, before_model
    from langgraph.runtime import Runtime

    @before_model
    def trace_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log message count immediately before each model call."""
        del runtime
        print(f"before_model: {len(state['messages'])} message(s)")
        return None

    @after_model
    def trace_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log message count after the model response enters agent state."""
        del runtime
        print(f"after_model: {len(state['messages'])} message(s)")
        return None

    if not live:
        print_heading("MIDDLEWARE EXPLANATION")
        print("Configured hooks: before_model -> model call -> after_model.")
        print("Agent not invoked; add --live to execute these hooks.")
        return

    agent = create_agent(
        model=create_chat_model(),
        tools=[],
        middleware=[trace_before, trace_after],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Explain middleware briefly."}]}
    )
    print("Final answer:", result["messages"][-1].content)


def demo_graph() -> None:
    """Build, compile, and invoke a stateful LangGraph workflow.

    A ``StateGraph`` declares the state shape, nodes transform state, edges
    define execution order, and ``compile`` validates and creates the runnable
    graph. ``invoke`` supplies initial state and returns the final state.
    Conditional edges route execution based on state. A checkpointer can
    persist thread state, while ``interrupt``/``Command(resume=...)`` can
    pause and continue a human review workflow (not enabled in this basic
    deterministic example).

    Debug steps: break inside ``normalize`` or a route node, inspect the
    current state dictionary, then step to ``graph.invoke``'s returned state.
    For a larger graph, inspect node names, edge routing, and checkpoint
    thread IDs to determine where execution paused or resumed.
    """
    from langgraph.graph import END, START, StateGraph

    def normalize(state: NormalizeState) -> dict[str, str]:
        """Write a lowercase/trimmed value to state without mutating input."""
        return {"normalized": state["text"].strip().lower()}

    graph_builder = StateGraph(NormalizeState)
    graph_builder.add_node("normalize", normalize)
    graph_builder.add_edge(START, "normalize")
    graph_builder.add_edge("normalize", END)
    graph = graph_builder.compile()
    result = graph.invoke({"text": "  LANGCHAIN  ", "normalized": ""})
    print_heading("LINEAR GRAPH")
    print("Compiled graph result:", result)

    def route(state: NumberState) -> str:
        """Select a named conditional edge from the current number."""
        return "even" if state["number"] % 2 == 0 else "odd"

    def label_even(_: NumberState) -> dict[str, str]:
        """Write the even branch's result."""
        return {"result": "even"}

    def label_odd(_: NumberState) -> dict[str, str]:
        """Write the odd branch's result."""
        return {"result": "odd"}

    conditional_builder = StateGraph(NumberState)
    conditional_builder.add_node("route", lambda _state: {})
    conditional_builder.add_node("even", label_even)
    conditional_builder.add_node("odd", label_odd)
    conditional_builder.add_edge(START, "route")
    conditional_builder.add_conditional_edges(
        "route",
        route,
        {"even": "even", "odd": "odd"},
    )
    conditional_builder.add_edge("even", END)
    conditional_builder.add_edge("odd", END)
    conditional_graph = conditional_builder.compile()
    print(
        "Conditional graph results:",
        [
            conditional_graph.invoke({"number": number, "result": ""})
            for number in (4, 5)
        ],
    )


def demo_memory() -> None:
    """Contrast short-term checkpointed state with a long-term in-memory store.

    A checkpointer saves graph state by ``thread_id`` so a later invocation
    for that thread can continue from prior state. ``InMemorySaver`` is useful
    for local experiments but is not durable across process restarts.
    ``InMemoryStore`` holds namespaced records that can be read by other
    threads during this process; production applications should choose a
    persistent store and define tenant/access boundaries.

    Debug steps: inspect the ``configurable.thread_id`` in ``thread_config``;
    step through ``append_message`` twice and compare the returned history.
    Then inspect the store's namespace, key, and stored ``value``. A different
    thread ID should not be assumed to share checkpointed conversation state.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.store.memory import InMemoryStore

    def append_message(state: MemoryState) -> dict[str, list[str]]:
        """Append the current input to this thread's persisted message list."""
        history = state.get("history", [])
        return {"history": [*history, state["new_message"]]}

    builder = StateGraph(MemoryState)
    builder.add_node("append_message", append_message)
    builder.add_edge(START, "append_message")
    builder.add_edge("append_message", END)
    memory_graph = builder.compile(checkpointer=InMemorySaver())

    thread_config = {"configurable": {"thread_id": "student-001"}}
    first_turn = memory_graph.invoke(
        {"history": [], "new_message": "I am learning LangChain."},
        config=thread_config,
    )
    second_turn = memory_graph.invoke(
        {"new_message": "Explain retrievers."},
        config=thread_config,
    )
    print_heading("SHORT-TERM THREAD MEMORY")
    print("First invocation:", first_turn)
    print("Second invocation in the same thread:", second_turn)

    store = InMemoryStore()
    namespace = ("users", "student-001")
    store.put(namespace, "preferences", {"style": "technical but simple"})
    preferences = store.get(namespace, "preferences")
    print_heading("LONG-TERM STORE")
    print("Namespace:", namespace)
    print("Stored preferences:", preferences.value if preferences else None)


def demo_streaming() -> None:
    """Inspect incremental graph updates without making a model request.

    ``stream`` yields outputs as a graph executes; the selected
    ``stream_mode`` controls what is yielded. ``updates`` reports node-level
    state changes, which helps locate where an unexpected value was produced.
    Chat-model token streaming uses ``model.stream`` (shown in the
    fundamentals demo), while asynchronous event streams use
    ``astream_events`` and must be consumed from an async context.

    Debug steps: set a breakpoint in ``normalize``, then step through the
    iterator and compare each node's update with the final result. Streaming
    exposes intermediate results; it does not by itself make an operation
    asynchronous or cancel a running request.
    """
    from langgraph.graph import END, START, StateGraph

    def normalize(state: NormalizeState) -> dict[str, str]:
        """Return a node update that can be inspected as a stream event."""
        return {"normalized": state["text"].strip().lower()}

    builder = StateGraph(NormalizeState)
    builder.add_node("normalize", normalize)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", END)
    graph = builder.compile()

    print_heading("GRAPH STREAM: NODE UPDATES")
    for update in graph.stream(
        {"text": "  STREAM ME  ", "normalized": ""},
        stream_mode="updates",
    ):
        print(update)


def demo_abstraction_roadmap() -> None:
    """Explain where notebook topics belong and which layer to debug first.

    Use a Runnable for deterministic linear composition, a retriever and
    vector store for search, an agent when the model must choose tools, and
    LangGraph when explicit stateful control flow, cycles, persistence or
    interrupts are central. Deep Agents add planning/filesystem/subagent
    workflows; MCP adapts external tools; LangSmith provides tracing and
    evaluation. These products are separate integrations, not required to run
    the local examples in this script.
    """
    print_heading("CHOOSING AN ABSTRACTION")
    options = [
        ("Runnable / LCEL", "Compose predictable steps: prompt -> model -> parser."),
        ("Retriever / Vector store", "Search documents and inspect evidence."),
        ("Two-step RAG", "Retrieve first on every question, then generate."),
        ("Agent", "Let a model choose among tools for a request."),
        ("LangGraph", "Model explicit state, routing, cycles, persistence, or interrupts."),
        ("Middleware", "Apply reusable policy around agent lifecycle steps."),
        ("MCP", "Connect an application to externally hosted tools."),
        ("LangSmith", "Trace, evaluate, and monitor model/application runs."),
        ("Deep Agents", "Use higher-level planning and long-running agent workflows."),
    ]
    for abstraction, purpose in options:
        print(f"{abstraction:24} {purpose}")


DEMO_FUNCTIONS: dict[str, Callable[..., None]] = {
    "fundamentals": demo_fundamentals,
    "runnables": demo_runnables,
    "retrieval": demo_retrieval,
    "agents": demo_agents,
    "middleware": demo_middleware,
    "graph": demo_graph,
    "memory": demo_memory,
    "streaming": demo_streaming,
    "roadmap": demo_abstraction_roadmap,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Parse demo options, load ``.env``, and run selected teaching sections.

    ``--live`` enables chat model calls but is off by default. Provider
    embeddings are selected separately with ``--provider-embeddings``.
    ``--document`` accepts a text/Markdown, PDF, or CSV input for the retrieval
    demo. These separate switches make it easier to identify which dependency
    or network request caused an error while debugging.
    """
    parser = argparse.ArgumentParser(
        description="Step through LangChain 2026 notebook concepts in Python."
    )
    parser.add_argument(
        "--demo",
        choices=[*DEMO_FUNCTIONS, "all"],
        default="fundamentals",
        help="Choose a lesson. 'all' runs each lesson in a safe sequence.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow demos to call the configured chat model.",
    )
    parser.add_argument(
        "--provider-embeddings",
        action="store_true",
        help="Use LANGCHAIN_EMBED_MODEL instead of offline teaching embeddings.",
    )
    parser.add_argument(
        "--document",
        type=Path,
        help="Optional .txt, .md, .pdf, or .csv file for --demo retrieval.",
    )
    args = parser.parse_args(argv)
    load_dotenv(REPOSITORY_ROOT / ".env")

    if args.document is not None and args.demo not in {"retrieval", "all"}:
        parser.error("--document can only be used with --demo retrieval or --demo all")

    selected_demos = (
        list(DEMO_FUNCTIONS)
        if args.demo == "all"
        else [args.demo]
    )
    for name in selected_demos:
        demo = DEMO_FUNCTIONS[name]
        print_heading(name.upper())
        if name == "fundamentals":
            demo(live=args.live)
        elif name == "retrieval":
            demo(
                provider_embeddings=args.provider_embeddings,
                live=args.live,
                document_path=args.document,
            )
        elif name in {"agents", "middleware"}:
            demo(live=args.live)
        else:
            demo()


if __name__ == "__main__":
    main()
