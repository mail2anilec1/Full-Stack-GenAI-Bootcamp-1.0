#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("all ok")


# In[1]:


def hello_node(input:str):
    return {"message": f"Hello, {input}!"}


# In[2]:


hello_node("sunny")


# In[3]:


hello_node("rajesh")


# In[4]:


hello_node("langgraph")


# In[5]:


from typing_extensions import TypedDict


# In[7]:


class State(TypedDict):
    message: str


# In[14]:


class State:
    message: str


# In[15]:


State()


# In[13]:

# # failstep
# State({"message": "hello python"})


# In[10]:


state: State = {"message": "hello python"}


# In[11]:


State


# In[16]:


def hello_node(state: State):
    return {"message": f"Hello, {state['message']}!"}


# In[17]:


from langgraph.graph import StateGraph, START, END


# In[27]:


builder = StateGraph(State)


# In[28]:


builder.add_node("first_hello_node",hello_node)


# In[29]:


builder.add_edge(START,"first_hello_node")


# In[30]:


builder.add_edge("first_hello_node", END)


# In[31]:


graph = builder.compile()


# In[32]:


from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))


# In[33]:


graph.invoke({"message": "from INDIA"})


# In[34]:


result = graph.invoke({"message": "from INDIA"})


# In[35]:


result["message"]


# In[36]:


class Sunny(TypedDict):
    text: str



# In[ ]:


state: Sunny = {"text": "hello"}


# In[39]:


def function1(state: Sunny):
    print(f"state from first function : {state}")
    return {
        "text": state["text"] + " from first function"
    }


# In[40]:


def function2(state: Sunny):
    print(f"state from second function : {state}")
    return {
        "text": state["text"] + " savita from second function"
    }


# In[42]:


workflow = StateGraph(Sunny)


# In[43]:


workflow.add_node("fun1", function1)
workflow.add_node("fun2", function2)


# In[44]:


workflow.add_edge(START, "fun1")
workflow.add_edge("fun1", "fun2")
workflow.add_edge("fun2", END)


# In[45]:


app = workflow.compile()


# In[46]:


from IPython.display import Image, display

display(
    Image(
        app.get_graph().draw_mermaid_png()
    )
)


# In[ ]:


#  return {
#         "text": state["text"] + " from first function"
#     }


# In[48]:


result = app.invoke({"text": "hi this is sunny"})


# In[49]:


result


# state from first function before return(this is input to the first function) : {'text': 'hi this is sunny'}
# state from second function before return statement(this is output from the first function) : {'text': 'hi this is sunny from first function'}
# final state after the section function output: {'text': 'hi this is sunny from first function savita from second function'}

# In[50]:


result["text"]


# In[51]:


from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# In[52]:


from dotenv import load_dotenv
import os

load_dotenv()


# In[53]:


if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )


# In[54]:


MODEL_NAME = "openai/gpt-oss-20b"
llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.6,
    reasoning_format="hidden"
)


# In[55]:


response = llm.invoke("Hi")

print(response.content)


# In[57]:


from langgraph.graph import MessagesState


# In[ ]:


# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]


# In[ ]:


# {"messages":["hi","how are are you?","what are you doing?"]}


# In[58]:


def call_model(state: MessagesState):

    print("\n=========================")
    print("MYBOT NODE")
    print("=========================")

    # Get complete conversation history
    messages = state["messages"]

    print(
        "Current Messages:",
        messages
    )

    # Send complete message history to LLM
    response = llm.invoke(messages)

    print("\nModel Response:",response.content)

    # MessagesState internally uses add_messages reducer.
    # Therefore the new AIMessage will be added
    # to the existing list of messages.
    return {
        "messages": [response]
    }


# In[59]:


workflow = StateGraph(MessagesState)


# In[60]:


workflow.add_node("mybot",call_model)


# In[61]:


workflow.add_edge(START,"mybot")
workflow.add_edge("mybot",END)


# In[62]:


app = workflow.compile()


# 

# In[63]:


from IPython.display import (
    Image,
    display,
)


display(
    Image(
        app.get_graph().draw_mermaid_png()
    )
)


# In[64]:


input = {"messages": [HumanMessage(content="Hi hello, how are you?")]}


# In[65]:


result = app.invoke(input)


# State Schema
# │
# ├── TypedDict
# │
# │   class State(TypedDict):
# │       text: str
# │
# ├── Dataclass
# │
# │   @dataclass
# │   class State:
# │       text: str
# │
# ├── Pydantic
# │
# │   class State(BaseModel):
# │       text: str
# │
# └── Built-in MessagesState
#     │
#     └── messages + add_messages

# ## 4th workflow

# In[82]:


response =llm.invoke("hi hello how are you?")


# In[83]:


response.content


# In[84]:


response.usage_metadata


# In[69]:


from typing_extensions import TypedDict, NotRequired
from langchain_core.messages import AIMessage
class State(TypedDict):
    # User input
    question: str
    # Generated answer
    answer: NotRequired[str]
    # Complete model response
    response: NotRequired[AIMessage]
    # Token information
    input_tokens: NotRequired[int]
    output_tokens: NotRequired[int]
    total_tokens: NotRequired[int]


# In[ ]:


def llm_node(state: State):
    print("\n========================")
    print("LLM NODE")
    print("========================")

    response = llm.invoke(state["question"])
    print(
        "\nGenerated Answer:\n",
        response.content
    )

    # IMPORTANT:
    # We are NOT counting tokens here.
    # We simply save the complete AIMessage
    # so the next node can inspect its token metadata.
    return { "answer": response.content, "response": response}



# In[ ]:


def token_counter(state: State):
    print("\n========================")
    print("TOKEN COUNTER NODE")
    print("========================")
    response = state["response"]

    usage = response.usage_metadata or {}

    input_tokens = usage.get(
        "input_tokens"
    )

    output_tokens = usage.get(
        "output_tokens"
    )

    total_tokens = usage.get(
        "total_tokens"
    )

    print(
        "Input Tokens:",
        input_tokens
    )

    print(
        "Output Tokens:",
        output_tokens
    )

    print(
        "Total Tokens:",
        total_tokens
    )

    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


# In[87]:


workflow = StateGraph(State)


# In[88]:


workflow.add_node("llm", llm_node)


# In[89]:


workflow.add_node("token_counter", token_counter)


# In[90]:


workflow.add_edge(START, "llm")
workflow.add_edge("llm", "token_counter")
workflow.add_edge("token_counter", END)


# In[91]:


app = workflow.compile()


# In[92]:


from IPython.display import Image, display

display(
    Image(
        app.get_graph().draw_mermaid_png()
    )
)


# In[93]:


result = app.invoke({"question": "Tell me about Tata Group in detail."})


# In[94]:


result


# # 5th and advance workflow

# In[1]:


import os
import json
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
# LangChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
# LangGraph
from langgraph.graph import (
    StateGraph,
    START,
    END,
)


# In[19]:


MODEL_NAME = "openai/gpt-oss-20b"
llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.6,
    reasoning_format="hidden"
)


# In[20]:


llm


# In[3]:


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# In[4]:


embedding_vector = embeddings.embed_query("Hello")
print("Embedding Dimension:",len(embedding_vector))


# In[5]:


from pathlib import Path


Data_Path = (
    Path(__file__).resolve().parents[2]
    / "Class-41-23-Aug-2026-Langgraph-Introduction"
    / "data"
)


# In[6]:


loader = DirectoryLoader(
    Data_Path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    }
)


# In[7]:


docs = loader.load()
print(
    "Total documents loaded:",
    len(docs)
)


# In[8]:


docs


# In[9]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)


# In[10]:


new_docs = text_splitter.split_documents(
    docs
)


print(
    "Total chunks:",
    len(new_docs)
)


# In[11]:


new_docs


# In[12]:


vector_store = Chroma(
    collection_name="usa_economy",
    embedding_function=embeddings
)


# In[13]:


vector_store.add_documents(
    documents=new_docs
)


# In[14]:


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# In[15]:


test_documents = retriever.invoke(
    "What is GDP of USA?"
)

for document in test_documents:

    print(
        document.page_content
    )

    print("-" * 50)


# In[16]:


from langchain_tavily import TavilySearch

web_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    topic="general"
)


# In[17]:


result = web_search.invoke({"query": "What is the latest AI news?"})


# In[18]:


print(result)


# In[21]:


class RouteDecision(BaseModel):
    route: Literal["RAG","LLM","WEB"] = Field(description="Node that should handle the user query")
    reasoning: str = Field(description="Reason for selecting this node")


# In[23]:


RouteDecision.model_json_schema()


# In[27]:


supervisor_model = llm.with_structured_output(schema=RouteDecision.model_json_schema(),method="json_schema")


# In[28]:


supervisor_model.invoke("can you tell me the USA current GDP?")


# In[29]:


class ValidationResult(BaseModel):
    passed: bool = Field(description="True if the generated answer passes validation")
    score: int = Field(ge=0,le=10,description="Quality score from 0 to 10")
    feedback: str = Field(description="Reason for validation result")


# In[30]:


validator_model = llm.with_structured_output(schema=ValidationResult.model_json_schema(),method="json_schema")


# In[ ]:


class AgentState(TypedDict, total=False):
    # Original user query
    question: str
    # Supervisor decision
    route: str
    # Supervisor reasoning
    route_reasoning: str
    # Context from RAG / Web
    context: str
    # Answer generated by RAG / LLM / Web
    draft_answer: str
    # Validation result
    validation_passed: bool
    validation_score: int
    validation_feedback: str
    # Retry information
    attempts: int
    # Keep track of previously used routes
    used_routes: list[str]
    # Answer shown to user only after validation
    final_answer: str


# In[ ]:


def supervisor_node(state: AgentState):
    print("\n==============================")
    print("SUPERVISOR NODE")
    print("==============================")
    question = state["question"]
    previous_feedback = state.get("validation_feedback","")
    used_routes = state.get("used_routes",[])
    attempts = state.get("attempts",0)
    supervisor_prompt = f"""
You are the supervisor of an AI system.

Your responsibility is to decide which node should answer
the user's question.

You have three possible nodes:

1. RAG

Use RAG when:
- the question is related to the USA economy
- GDP of USA
- industrial growth of USA
- US economic sectors
- US debt
- US economic strengths
- USA economic information contained in our internal documents

The RAG knowledge base mainly contains information
about the United States economy.


2. WEB

Use WEB when:
- the user asks for current information
- today
- latest
- realtime
- recent news
- current stock price
- current weather
- latest economic information
- information that may have changed recently
- internal RAG documents may be outdated


3. LLM

Use LLM when:
- the question is general knowledge
- casual conversation
- greeting
- conceptual question
- question unrelated to the USA knowledge base
- information does not require current internet data


User Question:
{question}


Routes already attempted:

{used_routes}


Previous validation feedback:

{previous_feedback}


Important:

If a previous answer failed validation,
use the validation feedback to decide whether another
information source should be selected.

Return the best route.
"""


    decision = supervisor_model.invoke(supervisor_prompt)
    route = decision["route"]
    reasoning = decision["reasoning"]

    print("Selected Route:",route)
    print("Reasoning:",reasoning)
    updated_routes = (used_routes + [route])


    return {

        "route": route,

        "route_reasoning": reasoning,

        "attempts": attempts + 1,

        "used_routes": updated_routes,

        # Clear previous attempt
        "context": "",

        "draft_answer": "",

    }


# In[ ]:


def router(state: AgentState):

    print(
        "\n-> ROUTER ->"
    )


    route = state["route"]


    print(
        "Routing to:",
        route
    )


    return route


# In[ ]:


def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)


# In[ ]:


def rag_node(state: AgentState):
    print("\n==============================")
    print("RAG NODE")
    print("==============================")
    question = state["question"]
    # -----------------------------------
    # Retrieve documents
    # -----------------------------------
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)
    print("Retrieved Documents:",len(retrieved_docs))

    # -----------------------------------
    # Generate answer
    # -----------------------------------

    rag_prompt = f"""
You are a question-answering assistant.

Answer the user's question using ONLY the supplied
retrieved context.

If the context does not contain enough information,
clearly say that the available internal knowledge
is insufficient.

Do not invent facts.

User Question:

{question}


Retrieved Context:

{context}


Generate a clear and concise answer.
"""
    response = llm.invoke(rag_prompt)

    answer = response.content

    print("RAG Answer:", answer)


    return {

        "context": context,

        "draft_answer": answer,

    }


# In[ ]:


def llm_node(state: AgentState):
    print("\n==============================")
    print("LLM NODE")
    print("==============================")
    question = state["question"]
    prompt = f"""
Answer the following question using your general
knowledge.

Give a clear, accurate and useful response.

If the question requires current or real-time information
that you cannot reliably know, explicitly mention that
real-time information should be checked.

Question:

{question}
"""
    response = llm.invoke(prompt)
    answer = response.content
    print("LLM Answer:",answer)

    return {

        "context": "",

        "draft_answer": answer,

    }


# In[ ]:


def web_node(state: AgentState):
    print(
        "\n=============================="
    )

    print(
        "WEB SEARCH NODE"
    )

    print(
        "=============================="
    )


    question = state["question"]


    # -----------------------------------
    # Search realtime internet
    # -----------------------------------

    search_result = web_search.invoke(

        {
            "query": question
        }

    )


    # -----------------------------------
    # Convert returned data to text
    # -----------------------------------

    if isinstance(
        search_result,
        dict
    ):

        results = search_result.get(
            "results",
            []
        )


        if results:

            web_context_parts = []


            for result in results:

                title = result.get(
                    "title",
                    ""
                )

                url = result.get(
                    "url",
                    ""
                )

                content = result.get(
                    "content",
                    ""
                )


                web_context_parts.append(

                    f"""
Title:
{title}

URL:
{url}

Content:
{content}
"""

                )


            web_context = "\n\n".join(
                web_context_parts
            )


        else:

            web_context = json.dumps(

                search_result,

                indent=2,

                ensure_ascii=False,

                default=str

            )


    else:

        web_context = str(
            search_result
        )


    print(
        "Web search completed."
    )


    # -----------------------------------
    # Generate answer using web context
    # -----------------------------------

    web_prompt = f"""
You are answering a user question using current
web-search results.

Use the web information below to answer the question.

Do not fabricate facts.

When useful, mention the source names or URLs
available in the search result.

User Question:

{question}


Web Search Results:

{web_context}


Generate the best answer based on the current
web information.
"""


    response = llm.invoke(
        web_prompt
    )


    answer = response.content


    print(
        "WEB Answer:",
        answer
    )


    return {

        "context": web_context,

        "draft_answer": answer,

    }



# In[54]:


def validator_node(
    state: AgentState
):

    print(
        "\n=============================="
    )

    print(
        "VALIDATOR NODE"
    )

    print(
        "=============================="
    )


    question = state["question"]

    route = state["route"]

    answer = state["draft_answer"]

    context = state.get(
        "context",
        ""
    )


    validation_prompt = f"""
You are the quality-control validator of an AI system.

Evaluate whether the generated answer is acceptable.


USER QUESTION:

{question}


NODE USED:

{route}


SOURCE CONTEXT:

{context}


GENERATED ANSWER:

{answer}


Validate the answer using these criteria:


1. RELEVANCE

Does the answer directly answer the user's question?


2. CORRECTNESS

Does the answer appear logically and factually correct
based on the available evidence?


3. COMPLETENESS

Does it provide enough information to answer the question?


4. HALLUCINATION

Does the answer make unsupported claims?


5. SOURCE GROUNDING

If NODE USED = RAG:

The answer must be supported by the retrieved context.


If NODE USED = WEB:

The answer must be supported by the web search context.


If NODE USED = LLM:

The answer should be suitable for general stable knowledge
and must not pretend to know unavailable real-time facts.


SCORING:

0-4  = poor
5-6  = needs improvement
7    = acceptable but weak
8-10 = good


PASS only if the score is 7 or higher and there is no
major factual or grounding problem.


Return:

passed
score
feedback
"""


    result = validator_model.invoke(
        validation_prompt
    )


    passed = result["passed"]

    score = result["score"]

    feedback = result["feedback"]


    print(
        "Validation Passed:",
        passed
    )

    print(
        "Score:",
        score
    )

    print(
        "Feedback:",
        feedback
    )


    return {

        "validation_passed": passed,

        "validation_score": score,

        "validation_feedback": feedback,

    }


# In[55]:


MAX_ATTEMPTS = 3


#     return {
# 
#         "validation_passed": passed,
# 
#         "validation_score": score,
# 
#         "validation_feedback": feedback,
# 
#     }
# 

# In[ ]:


def validation_router(
    state: AgentState
):

    print(
        "\n-> VALIDATION ROUTER ->"
    )


    # -----------------------------------
    # Validation passed
    # -----------------------------------

    if state["validation_passed"]:

        print(
            "Validation PASSED"
        )

        return "Final"


    # -----------------------------------
    # Too many attempts
    # -----------------------------------

    if state.get("attempts",0) >= MAX_ATTEMPTS:
        print(
            "Maximum attempts reached"
        )

        return "Failed"


    # -----------------------------------
    # Validation failed -> supervisor
    # -----------------------------------

    print(
        "Validation FAILED"
    )

    print(
        "Returning to Supervisor..."
    )


    return "Supervisor"


# In[57]:


def final_node(
    state: AgentState
):

    print(
        "\n=============================="
    )

    print(
        "FINAL ANSWER NODE"
    )

    print(
        "=============================="
    )


    answer = state["draft_answer"]


    return {

        "final_answer": answer

    }


# In[58]:


def failed_node(
    state: AgentState
):

    print(
        "\n=============================="
    )

    print(
        "VALIDATION FAILED AFTER RETRIES"
    )

    print(
        "=============================="
    )


    message = (
        "The system could not generate a sufficiently "
        "validated answer after multiple attempts."
    )


    return {

        "final_answer": message

    }


# In[59]:


workflow = StateGraph(AgentState)


# In[60]:


workflow.add_node("Supervisor",supervisor_node)
workflow.add_node("RAG",rag_node)
workflow.add_node("LLM",llm_node)
workflow.add_node("WEB",web_node)
workflow.add_node("Validator",validator_node)
workflow.add_node("Final",final_node)
workflow.add_node("Failed",failed_node)


#     return {
# 
#         "route": route,
# 
#         "route_reasoning": reasoning,
# 
#         "attempts": attempts + 1,
# 
#         "used_routes": updated_routes,
# 
#         # Clear previous attempt
#         "context": "",
# 
#         "draft_answer": "",
# 
#     }

# In[ ]:


workflow.add_edge(START,"Supervisor")

workflow.add_conditional_edges("Supervisor",
                            router,
    {
        "RAG": "RAG",

        "LLM": "LLM",

        "WEB": "WEB",

    }

)

#fan-in
workflow.add_edge("RAG","Validator")
workflow.add_edge("LLM","Validator")
workflow.add_edge("WEB","Validator")


workflow.add_conditional_edges(
    "Validator",
    validation_router,
    {
        "Final": "Final",
        "Supervisor": "Supervisor",
        "Failed": "Failed",

    }

)
workflow.add_edge("Final",END)
workflow.add_edge("Failed",END)


# In[62]:


app = workflow.compile()
print(
    "\nGraph compiled successfully."
)


# In[63]:


from IPython.display import (
    Image,
    display,
)

display(

    Image(

        app.get_graph().draw_mermaid_png()

    )

)


# In[64]:


def ask_agent(question: str):

    initial_state = {

        "question": question,

        "attempts": 0,

        "used_routes": [],

        "validation_feedback": "",

        "validation_passed": False,

        "validation_score": 0,

        "context": "",

        "draft_answer": "",

        "final_answer": "",

    }


    result = app.invoke(

        initial_state,

        {
            "recursion_limit": 30
        }

    )


    print(
        "\n\n================================"
    )

    print(
        "FINAL RESULT"
    )

    print(
        "================================"
    )


    print(
        "Question:"
    )

    print(
        result["question"]
    )


    print(
        "\nFinal Route:"
    )

    print(
        result.get(
            "route"
        )
    )


    print(
        "\nRoutes Tried:"
    )

    print(
        result.get(
            "used_routes"
        )
    )


    print(
        "\nValidation Score:"
    )

    print(
        result.get(
            "validation_score"
        )
    )


    print(
        "\nValidation Feedback:"
    )

    print(
        result.get(
            "validation_feedback"
        )
    )


    print(
        "\nFinal Answer:"
    )

    print(
        result["final_answer"]
    )


    return result


# In[65]:


result = ask_agent("hi can you help me?")


# In[66]:


result = ask_agent("hcan you tell me the current GDP of china?")


# In[67]:


result = ask_agent("What are the major economic challenges of the USA?")


# In[68]:


result = ask_agent("What does the document say about future growth of the U.S. economy?")


# In[69]:


result = ask_agent("can you give me current update of the usa and iran war?")


# In[70]:


result = ask_agent("hi my name is sunny how you are doing?")


# In[71]:


result = ask_agent("hi what is friend name?")


# In[75]:


result = ask_agent("can you give me salary of sunny?")


# In[78]:


result = ask_agent("inflation rate, unemployment rate, and Federal Reserve interest rate of the")


# In[80]:


result = ask_agent("when is rakhi festivel in 2026?")


#                               USER
#                                │
#                                │
#                      ┌─────────┴─────────┐
#                      │                   │
#                      ↓                   ↓
#                  User ID            New Request
#                      │
#                      │
#         ┌────────────┴────────────────────────────┐
#         │                                         │
#         │                                         │
#         ↓                                         ↓
# 
#    THREAD 1                                   THREAD 2
#    thread_id = 101                            thread_id = 202
#         │                                         │
#         │                                         │
#         ↓                                         ↓
# 
# ┌──────────────────────┐                ┌──────────────────────┐
# │   CURRENT STATE      │                │   CURRENT STATE      │
# │                      │                │                      │
# │ messages             │                │ messages             │
# │ question             │                │ question             │
# │ context              │                │ context              │
# │ tool_result          │                │ tool_result          │
# │ route                │                │ route                │
# │ final_answer         │                │ final_answer         │
# └──────────┬───────────┘                └──────────┬───────────┘
#            │                                       │
#            │ graph executes                       │ graph executes
#            ↓                                       ↓
# 
#       Node A                                  Node A
#            │                                       │
#            ↓                                       ↓
#       State Update                            State Update
#            │                                       │
#            ↓                                       ↓
#       Node B                                  Node B
#            │                                       │
#            ↓                                       ↓
# 
# ┌──────────────────────┐                ┌──────────────────────┐
# │    CHECKPOINTER      │                │    CHECKPOINTER      │
# │                      │                │                      │
# │ saves State          │                │ saves State          │
# │ after supersteps     │                │ after supersteps     │
# └──────────┬───────────┘                └──────────┬───────────┘
#            │                                       │
#            │                                       │
#            ↓                                       ↓
# 
# ┌──────────────────────┐                ┌──────────────────────┐
# │ SHORT-TERM MEMORY    │                │ SHORT-TERM MEMORY    │
# │                      │                │                      │
# │ Thread scoped        │                │ Thread scoped        │
# │                      │                │                      │
# │ previous messages    │                │ previous messages    │
# │ conversation state   │                │ conversation state   │
# │ tool results         │                │ tool results         │
# │ intermediate values  │                │ intermediate values  │
# └──────────┬───────────┘                └──────────┬───────────┘
#            │                                       │
#            │ SAME thread only                      │ SAME thread only
#            │                                       │
#            └────────────────┬──────────────────────┘
#                             │
#                             │
#                             │
#                             ↓
# 
#                   ┌──────────────────────┐
#                   │   LONG-TERM STORE    │
#                   │                      │
#                   │ Cross-thread memory  │
#                   └──────────┬───────────┘
#                              │
#                              ↓
# 
#                   ┌──────────────────────┐
#                   │ User Memory          │
#                   │                      │
#                   │ name                 │
#                   │ preferences          │
#                   │ profile              │
#                   │ important facts      │
#                   │ learned preferences  │
#                   │ past information     │
#                   └──────────────────────┘
# 
# 
#                               │
#               ┌───────────────┼───────────────┐
#               ↓               ↓               ↓
# 
#           Thread 1         Thread 2         Thread 3
#           can access       can access       can access
#               │               │               │
#               └───────────────┼───────────────┘
#                               ↓
#                       LONG-TERM MEMORY

# In[ ]:




