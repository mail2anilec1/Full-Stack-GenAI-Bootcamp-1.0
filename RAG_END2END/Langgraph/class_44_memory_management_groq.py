#!/usr/bin/env python
# coding: utf-8

# ```
# ConversationBufferMemory
# ConversationBufferWindowMemory
# ConversationTokenBufferMemory
# ConversationSummaryMemory
# ConversationSummaryBufferMemory
# ConversationEntityMemory
# VectorStoreRetrieverMemory
# CombinedMemory
# ConversationStringBufferMemory
# ```

# ```
# Memory Management
# │
# ├── Short-Term Memory
# │      │
# │      ├── Checkpointer
# │      ├── Trim
# │      ├── Delete
# │      ├── Summarize
# │      └── Custom filtering
# │
# └── Long-Term Memory
#        │
#        ├── Store.put()
#        ├── Store.get()
#        ├── Store.search()
#        ├── user_id / namespace
#        └── Semantic retrieval
# ```
# 
# ```
# Short-Term Memory
# Checkpointer → current conversation/thread ka state save karta hai so same chat ko continue kar sake.
# Trim → old messages hata kar sirf recent messages LLM ko bhejta hai to reduce tokens/cost.
# Delete → selected messages ko memory/state se permanently remove karta hai.
# Summarize → purani long conversation ko short summary me convert karta hai.
# Custom filtering → apne rule se decide karte ho kaunse messages context me rakhne hain aur kaunse nahi.
# 
# Long-Term Memory
# Store.put() → koi user fact/preference/memory save karta hai.
# Store.get() → exact saved memory ko key/id se retrieve karta hai.
# Store.search() → stored memories me search karke relevant memories nikalta hai.
# user_id / namespace → memories ko different users ya categories ke according separate rakhta hai.
# Semantic retrieval → exact keywords ke instead meaning/similarity ke basis par relevant memory retrieve karta hai.
# ```
# 
# ```
# Short-Term Memory
# Checkpointer → Saves the state of the current conversation/thread so the same chat can be continued.
# Trim → Removes old messages and sends only recent messages to the LLM to reduce tokens/cost.
# Delete → Permanently removes selected messages from memory/state.
# Summarize → Converts an old, lengthy conversation into a short summary.
# Custom filtering → You decide, based on your own rules, which messages should be kept in the context and which should not.
# 
# Long-Term Memory
# Store.put() → Saves a user fact/preference/memory.
# Store.get() → Retrieves an exact saved memory using a key/ID.
# Store.search() → Searches stored memories and retrieves relevant memories.
# user_id / namespace → Separates memories according to different users or categories.
# Semantic retrieval → Retrieves relevant memories based on meaning/similarity instead of exact keywords.
# ```

# What is Memory in AI?
# 
# Memory is the mechanism that allows an AI application to retain and reuse information from previous interactions.
# 
# Memory in AI is the ability of an application to store, retrieve, and reuse information from previous interactions so that the system can maintain context and provide more consistent and personalized responses.
# 
# An LLM is usually stateless by itself. If the previous context is not provided with each new call, the model does not remember the earlier conversation.
# 
# User: My name is Sunny.
# 
# AI: Nice to meet you.
# 
# User: What is my name?
# 
# Without memory:
# 
# Every request → treated like a new conversation
# 
# With memory:
# 
# Previous Context
# 
#       +
# 
# Current Query
# 
#       ↓
#       
# Better Response
# 
# Example:
# 
# User: I am working on Agentic RAG.
# 
# Later: What project am I working on?
# 
# Memory ke through AI answer : Agentic RAG
# 
# Two major types
# 
# Short-Term Memory
# 
# → current conversation/session
# 
# → recent messages
# 
# Long-Term Memory
# 
# → information retained across conversations
# 
# → user preferences, facts, past interactions
# 
# Why do we need Memory Management in AI Applications?
# 
# - Maintain conversation continuity
# - Handle follow-up questions
# - Personalize responses
# - Remember user preferences
# - Support long-running agent workflows
# - Avoid repeated questions
# - Maintain context across sessions
# - Track workflow state
# - Control context-window usage
# - Reduce token usage and cost

# In[1]:


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file before running this script."
    )

print("GROQ_API_KEY configured:", bool(os.getenv("GROQ_API_KEY")))


# In[2]:


# In[3]:


model = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    reasoning_format="hidden",
)


# In[4]:


model.invoke([{"role": "user", "content": "Hello, how are you?"}])


# In[5]:


memory = []


# In[6]:


def chat(user_input:str)-> str:
    memory.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = model.invoke(memory)

    memory.append(
        {
            "role": "assistant",
            "content": response.content
        }
    )
    return response.content


# In[11]:


def run_chat_loop():

    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit","quit",}:
            print("Chat ended.")
            break

        if not user_input:
            continue

        answer = chat(user_input)

        print(
            "\nAI:",
            answer
        )


# In[12]:


run_chat_loop()


# In[13]:


memory


# In[14]:


for message in memory:
        print(
        message["role"],
        "->",
        message["content"]
    )


# ## Problem with Full Memory
# 
# If the conversation keeps growing, sending the entire history on every request increases:
# 
# - input tokens
# - cost
# - latency
# - context-window usage
# 
# So we normally separate **what we store** from **what we send to the LLM**.

# ## Sliding Window Memory
# 
# 
# We still store the complete conversation, but only the most recent messages are sent to the LLM.
# 
# **Stored history can be large; active LLM context should stay controlled.**

# In[16]:


# chatgpt

# 100 question

# 1
# 2
# 3
# 4
# 5





# 100

# 101 question 1 question sometime you will notice this chatgpt is hallucinating and giving wrong answer.

# 90 question
# 95 question

# then you will see that chatgpt is giving somewhat correct answer.


# In[ ]:


full_history = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "M10"
]


# In[18]:


recent_context = full_history[-6:]


# In[19]:


recent_context


# In[26]:


memory = []


# In[27]:


MAX_CONTEXT_MESSAGES = 6


# In[28]:


def chat(user_input: str) -> str:
    memory.append(
        {
            "role": "user",
            "content": user_input
        }
    )
    recent_context = memory[-MAX_CONTEXT_MESSAGES:]

    response = model.invoke(
        recent_context
    )
    memory.append(
        {
            "role": "assistant",
            "content": response.content
        }
    )
    return response.content



# In[29]:


def run_chat_loop():
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input(
            "\nYou: "
        ).strip()

        if user_input.lower() in {
            "exit",
            "quit",
        }:
            print("Chat ended.")
            break

        if not user_input:
            continue

        answer = chat(user_input)

        print(
            "\nAI:",
            answer
        )

        print(
            "\n[Full stored messages]:",
            len(memory)
        )

        print(
            "[Messages used as active context]:",
            min(
                len(memory),
                MAX_CONTEXT_MESSAGES,
            )
        )


# In[30]:


run_chat_loop()


# # Summary + Recent Messages
# 
# Strategy:
# 
# - keep recent messages exactly as they are
# - summarize older messages
# - send `summary + recent messages` to the LLM
# - optionally keep complete history separately for audit/storage
# 
# Compression happens only after a complete user/assistant turn, so we do not intentionally split a turn during summarization.

# In[46]:


memory = []


# In[47]:


conversation_summary = ""


# In[48]:


MAX_RECENT_MESSAGES = 6


# In[49]:


full_history = []


# In[50]:


def format_messages(messages):
    if not messages:
        return "No recent conversation."

    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in messages
    )



# In[35]:


def create_summary(old_summary: str, old_messages: list) -> str:

    conversation_text = format_messages(old_messages)

    prompt = f"""
You maintain conversation memory for an AI assistant.

Existing summary:
{old_summary or "No previous summary."}

Older conversation messages to merge:
{conversation_text}

Create one concise updated summary.

Preserve useful information such as:
- user name
- preferences
- projects
- decisions
- important facts
- important questions and answers

Ignore greetings and unnecessary small talk.
"""

    response = model.invoke(prompt)

    return response.content.strip()


# In[51]:


def compress_memory_into_summary():

    global memory
    global conversation_summary

    if len(memory) <= MAX_RECENT_MESSAGES:
        return

    old_messages = memory[:-MAX_RECENT_MESSAGES]

    memory = memory[-MAX_RECENT_MESSAGES:]

    conversation_summary = create_summary(
        conversation_summary,
        old_messages
    )


# In[52]:


def chat(user_input: str) -> str:

    global memory
    global full_history

    user_message = {"role": "user","content": user_input}

    memory.append(user_message)
    full_history.append(user_message)

    messages_for_llm = [{"role": "system", "content": f"""Previous conversation summary:{conversation_summary or "No previous summary."}"""},
                        *memory
                        ]

    response = model.invoke(messages_for_llm)

    assistant_message = {"role": "assistant","content": response.content}

    memory.append(assistant_message)
    full_history.append(assistant_message)

    # Compress only after a complete user/assistant turn
    compress_memory_into_summary()

    return response.content


# In[ ]:


def chat_loop():

    print("Type 'exit' or 'quit' to stop.")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit","quit"}:
            print("Chat ended.")
            break

        if not user_input:
            continue

        answer = chat(user_input)

        print("\nAI:",answer)

        print(
            "\n[Recent active messages]:",
            len(memory)
        )

        print(
            "[Full-history messages]:",
            len(full_history)
        )

        if conversation_summary:

            print(
                "\n[Current summary]:"
            )

            print(
                conversation_summary
            )


# In[40]:


chat_loop()


# In[ ]:




