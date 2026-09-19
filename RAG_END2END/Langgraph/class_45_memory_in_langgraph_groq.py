#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("all ok")


# In[24]:


import uuid
from dataclasses import dataclass
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END
)
from langgraph.runtime import Runtime


# In[25]:


load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file before running this script."
    )

model = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0
)


# In[27]:


from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()


# In[28]:


from langgraph.store.memory import InMemoryStore
store = InMemoryStore()


# In[29]:


@dataclass
class Context:
    user_id: str


# In[14]:


Context.user_id=str(uuid.uuid4())


# In[15]:


Context.user_id


# In[30]:


def assistant_node(
    state: MessagesState,
    runtime: Runtime[Context]
):
    user_id = runtime.context.user_id
    namespace = ("users", user_id, "memories")
    user_message = state["messages"][-1].content

    if user_message.lower().startswith("remember:"):
        memory = user_message[len("remember:"):].strip()
        runtime.store.put(
            namespace,
            str(uuid.uuid4()),
            {
                "data": memory
            }
        )
        print(
            f"[MEMORY SAVED]: {memory}"
        )

    memories = runtime.store.search(namespace)
    memory_text = "\n".join(
        memory.value["data"]
        for memory in memories
    )
    if not memory_text:
        memory_text = "No saved memories."


    system_prompt = f"""
    You are a helpful assistant.

    Long-term memories about this user:

    {memory_text}
    """


    response = model.invoke(
        [
            {
                "role": "system",
                "content": system_prompt
            },
            *state["messages"]
        ]
    )
    return {
        "messages": [response]
    }


# In[31]:


builder = StateGraph(
    MessagesState,
    context_schema=Context
)


# In[32]:


builder.add_node(
    "assistant",
    assistant_node
)


# In[33]:


builder.add_edge(
    START,
    "assistant"
)

builder.add_edge(
    "assistant",
    END
)


# In[34]:


graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)


# In[35]:


def chat(
    message,
    user_id,
    thread_id
):

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        },

        config={
            "configurable": {
                "thread_id": thread_id
            }
        },

        context=Context(
            user_id=user_id
        )
    )


    answer = result["messages"][-1].content

    print("\nUSER:")
    print(message)

    print("\nASSISTANT:")
    print(answer)

    print("\n" + "=" * 60)


# In[36]:


chat(
    message="My current project is Agentic RAG.",
    user_id="sunny_123",
    thread_id="thread_1"
)


# In[37]:


chat(
    message="What is my current project?",
    user_id="sunny_123",
    thread_id="thread_1"
)


# In[38]:


chat(
    message="Remember: my favorite programming language is Python.",
    user_id="sunny_123",
    thread_id="thread_1"
)


# In[39]:


chat(
    message="What is my favorite programming language?",
    user_id="sunny_123",
    thread_id="thread_2"
)


#                  sunny_123
#                      ↓
#               InMemoryStore
#                      ↓
#       "favorite language = Python"
#                      ↓
#              ┌───────┴───────┐
#              ↓               ↓
#          thread_1         thread_2

# In[40]:


chat(
    message="Remember: my favorite language is Python.",
    user_id="sunny_123",
    thread_id="sunny_thread_1"
)


# In[41]:


chat(
    message="Remember: my favorite language is Java.",
    user_id="rahul_456",
    thread_id="rahul_thread_1"
)


# users
# │
# ├── sunny_123
# │      └── memories
# │           └── favorite language = Python
# │
# └── rahul_456
#        └── memories
#             └── favorite language = Java

# In[42]:


chat(
    message="What is my favorite programming language?",
    user_id="rahul_456",
    thread_id="rahul_thread_2"
)


# In[43]:


chat(
    message="What is my favorite programming language?",
    user_id="sunny_123",
    thread_id="sunny_thread_2"
)


# In[44]:


user_id = "sunny_123"


# In[45]:


namespace = (
    "users",
    user_id,
    "memories"
)


# In[46]:


memories = store.search(namespace)

for memory in memories:
    print(memory.value)


# In[47]:


memories = store.search(
    ("users", "sunny_123", "memories")
)

for memory in memories:
    print(memory.value["data"])


# In[ ]:




