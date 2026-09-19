#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("all ok")


# ```
# Incoming Data
#      ↓
# Pydantic Model
#      ↓
# Check:
# - required fields?
# - correct type?
# - valid range?
# - correct format?
#      ↓
# Valid → continue
# Invalid → error
# ```
# 
# 
# ```
# Pydantic = data validation and schema library.
# 
# Pydantic AI = agent framework for building GenAI/Agentic AI applications, built around Pydantic-style type safety.
# ```
# 
# ##### Pydantic is a Python library used to define the expected structure of data and validate incoming data against that structure.
# 
# ##### Pydantic helps us define what our data should look like and checks whether the received data follows that structure.
# 
# ##### Pydantic is a Python library that lets us define the expected structure of data using Python types and automatically validates the input data against that structure.
# 
# ##### Schema enforcement simply means making sure that the data follows the structure and rules we have already defined.
# 
# ##### For example, if our schema says name should be a string and age should be an integer, then the incoming data should follow these rules. If it doesn’t, Pydantic will raise a validation error.
# 
# ```
# Schema defining = Defining what the data structure should look like.
# 
# Schema validation = Checking whether the input data follows the defined structure and rules.
# 
# Schema enforcement = Ensuring that invalid data is not accepted and the defined structure and rules are followed.
# ```
# 
# ```
# Raw / Untrusted Data
#         ↓
#    Pydantic Schema
#         ↓
#  ┌───────────────────┐
#  │ Type checking     │
#  │ Type conversion   │
#  │ Constraints       │
#  │ Custom validation │
#  └───────────────────┘
#         ↓
#  Valid Python Object
#  ```
# 
# ```
# What is BaseModel?
# 
# You will see this everywhere: 
# 
# from pydantic import BaseModel
# 
# Then:
# 
# class User(BaseModel):
#     name: str
#     age: int
# 
# BaseModel gives your class Pydantic capabilities:
# 
# Validation
# Parsing
# Serialization
# JSON Schema
# Error messages
# 
# Without BaseModel:
# 
# class User:
#     ...
# 
# it's just a normal Python class.
# ```
# 
# !uv pip install pydantic
# 
# ```
# Pydantic V1
#       ↓
# major API changes
#       ↓
# Pydantic V2
# ```
# 
# ```
# dict()              → model_dump()
# json()              → model_dump_json()
# parse_obj()         → model_validate()
# parse_raw()         → model_validate_json()
# schema()            → model_json_schema()
# copy()              → model_copy()
# @validator           → @field_validator
# @root_validator      → @model_validator
# ```
# 

# ```
# What is Field()?
# 
# Example:
# 
# age: int = Field(
#     ge=18,
#     description="Age of the user"
# )
# 
# Field() lets you add:
# 
# Validation rules
# Description
# Default values
# Constraints
# Metadata
# 
# Useful constraints:
# 
# Field(gt=0)
# 
# greater than.
# 
# Field(ge=0)
# 
# greater than or equal.
# 
# Field(lt=100)
# 
# less than.
# 
# Field(le=100)
# 
# less than or equal.
# 
# For strings:
# 
# Field(
#     min_length=3,
#     max_length=50
# )
# ```

# # Without Pydantic vs with Pydantic
# 
# input from USER
# 
# in the input i will ask their NAME and AGE>18

# In[2]:


def create_user(data):
    if "name" not in data:
        raise ValueError("Missing 'name' field")

    if not isinstance(data["name"], str):
        raise ValueError("Name must be a string")

    if "age" not in data:
        raise ValueError("Age is required")

    if not isinstance(data["age"], int):
        raise ValueError("Age must be an integer")

    if data["age"] < 18:
        raise ValueError("Age must be greater than or equal to 18")

    return {
        "message": "User created successfully",
        "user": data
    }


# In[3]:


user_data = {
    "name": "kumar gaurav",
    "age": 35
    }


# In[4]:


create_user(user_data)


# In[5]:


user_data = {
    "name": "Mayank Chugh",
    "age": 49
    }


# In[6]:


create_user(user_data)


# In[7]:


user_data = {
    "name": "Mayank Singh",
    "age": 17
    }


# In[8]:


create_user(user_data)


# In[10]:


user_data = {
    "name": ["Sunny"],
    "age": 30
    }


# In[11]:


create_user(user_data)


# In[22]:


from pydantic import BaseModel, EmailStr, Field, ValidationError


# In[14]:


class User(BaseModel):
    name: str = Field(..., description="The name of the user")
    age: int = Field(..., ge=18, description="The age of the user, must be 18 or older")


# In[15]:


user = User(name="Mayank Chugh", age=49)


# In[16]:


user.name


# In[17]:


user.age


# In[18]:


user = User(name="Sunny", age=17)


# In[19]:


User(name="Ankit Singh", age="49")


# In[20]:


User(name="Ankit Singh", age="forthy-nine")


# ## API Validation

# User Registration
# 
# Signup Form
#  ↓
# Backend API
#  ↓
# Validate user data

# In[23]:


class UserRegistration(BaseModel):
    name: str
    email: EmailStr
    age: int = Field(ge=18)


# In[24]:


user = UserRegistration(
    name="Sunny",
    email="sunny@gmail.com",
    age=30
)


# In[25]:


user.name


# In[26]:


user.email


# In[27]:


user.age


# In[28]:


user = UserRegistration(
    name="Sunny",
    email="sunnysavita",
    age=19
)


# In[29]:


user = UserRegistration(
    name="Sunny",
    email="sunnysavita",
    age=16
)


# In[31]:


try:
    user = UserRegistration(
        name="Sunny",
        email="sunnysavita",
        age=16
    )
except ValidationError as e:
    print(e)


# In[ ]:


raise
handle or log


# Pydantic tells you exactly:
# 
# email → invalid email
# 
# age → must be >= 18
# 
# This is much better than manually writing dozens of if conditions.
# 
# ValidationError tells us what data is invalid, while try/except allows us to handle that validation failure without crashing the application.
# 
# Pydantic
# → detects the problem
# 
# ValidationError
# → represents the problem
# 
# try/except
# → handles the problem
# 
# print(e)
# → displays the problem

#  ## Product / E-commerce Validation

# In[ ]:


# Schema
class Product(BaseModel):
    name: str
    price: float = Field(gt=0)
    quantity: int = Field(ge=0)


# In[33]:


product = Product(
    name="Laptop",
    price=75000,
    quantity=5
)


# In[34]:


product


# In[35]:


Product(
    name="Laptop",
    price=-500,
    quantity=2
)


# # Banking Transaction

# In[40]:


class MoneyTransfer(BaseModel):
    sender_account: str
    receiver_account: str
    amount: float = Field(gt=500)


# In[37]:


transfer = MoneyTransfer(
    sender_account="ACC001",
    receiver_account="ACC002",
    amount=5000000
)


# In[38]:


transfer


# In[39]:


transfer = MoneyTransfer(
    sender_account="ACC001",
    receiver_account="ACC002",
    amount=-5000000
)


# In[42]:


transfer = MoneyTransfer(
    sender_account="ACC001",
    receiver_account="ACC002",
    amount=400
)


# In[44]:


transfer = MoneyTransfer(
    sender_account="ACC001",
    receiver_account="ACC002",
    amount="40L"
)


# # Model Configuration

# In[45]:


class ModelConfig(BaseModel):
    model_name: str
    temperature: float = Field(ge=0, le=2)
    timeout: int = Field(gt=0)


# In[46]:


config = ModelConfig(
    model_name="gpt-model",
    temperature=0.7,
    timeout=30
)


# In[47]:


config


# In[48]:


config = ModelConfig(
    model_name="gpt-model",
    temperature=2.1,
    timeout=30
)


# In[49]:


config = ModelConfig(
    model_name="gpt-model",
    temperature=0.7,
    timeout=-5
)


# In[50]:


from langchain_groq import ChatGroq


# In[51]:


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# In[52]:


llm.invoke("What is the capital of France?")


# In[53]:


class PersonDetails(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="City where the person lives")


# In[ ]:


structured_model =llm.with_structured_output(PersonDetails)


# In[55]:


structured_model.invoke("sunny is 30 years old and lives in delhi")


# In[ ]:


# from typing import Literal
# class RouteDecision(BaseModel):
#     route: Literal[
#         "RAG",
#         "WEB",
#         "LLM"
#     ]
#     reasoning: str


# In[ ]:


# class User(BaseModel):
#     # 1. REQUIRED FIELD
#     # The value must be provided.
#     name: str

#     # 2. DEFAULT FIELD
#     # If no value is provided, "India" will be used automatically.
#     country: str = "India"

#     # 3. REQUIRED BUT NULLABLE
#     # The field must be provided,
#     # but its value can be None.
#     middle_name: str | None

#     # 4. OPTIONAL-TO-PROVIDE + NULLABLE
#     # The field does not have to be provided,
#     # and its value can also be None.
#     nickname: str | None = None


# In[60]:


class User(BaseModel):
    name: str
    country: str = "India"
    middle_name: str | None
    nickname: str | None = None


# In[58]:


user = User(
    name="Sunny",
    country="USA",
    middle_name="Kumar",
    nickname="Sun"
)


# In[59]:


user


# In[61]:


user = User(
    name="Sunny",
    middle_name="Kumar"
)


# In[62]:


user


# In[63]:


user = User(
    name="Sunny",
    middle_name=None
)


# In[64]:


user


# In[65]:


class User(BaseModel):
    name: str
    age: int = Field(ge=18)
    city: str = "Bangalore"


# In[66]:


user = User(
    name="Sunny",
    age=30
)


# In[67]:


user


# In[68]:


user.name


# In[69]:


user.age


# In[70]:


user.city


# In[71]:


type(user)


# In[72]:


user.model_dump()


# In[73]:


type(user.model_dump())


# In[74]:


user.model_dump_json()


# In[75]:


type(user.model_dump_json())


# In[76]:


import json
json.loads(user.model_dump_json())


# In[77]:


type(json.loads(user.model_dump_json()))


# In[79]:


class User(BaseModel):
    name: str
    age: int = Field(ge=18)
    city: str = "Bangalore"


# In[85]:


User.model_json_schema()


# In[82]:


User(name = "sunny", age = 30, city = "Goa")


# In[80]:


data = {
    "name": "Sunny",
    "age": "30",
    "city": "Delhi"
}


# In[81]:


User.model_validate(data)


# In[83]:


data = """
{
    "name": "Sunny",
    "age": 30,
    "city": "Bangalore"
}
"""


# In[84]:


User.model_validate_json(data)


# In[86]:


user1 = User(
    name="Sunny",
    age=30
)


# In[87]:


user2 = user1.model_copy()


# In[88]:


user2


# In[89]:


user3=user1.model_copy(
    update={
        "city": "Mumbai"
    }
)


# In[90]:


user3


# In[91]:


from pydantic import field_validator, model_validator


# In[92]:


class Employee(BaseModel):
    name: str 
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Name must contain at least 2 characters")
        return value.title()


# In[93]:


Employee(
    name="sunny",
    age=30
)


# In[94]:


Employee(
    name="su",
    age=30
)


# In[95]:


class Signup(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def check_passwords(self):
        if (self.password!=self.confirm_password):
            raise ValueError("Passwords do not match")
        return self


# In[96]:


Signup(
    password="hello123",
    confirm_password="hello123"
)


# In[97]:


Signup(
    password="hello123",
    confirm_password="wrong123"
)


# One field → field_validator
# 
# Multiple fields / whole object → model_validator

# | `field_validator`                  | `model_validator`                                 |
# | ---------------------------------- | ------------------------------------------------- |
# | **Validates a single field**       | **Validates multiple fields or the entire model** |
# | Works on an individual field value | Works on the complete object/model                |
# 

# ### Nested Schema

# ```
# User
#  ├── name
#  ├── age
#  ├── email
#  ├── Address
#  │    ├── city
#  │    ├── state
#  │    └── pin_code
#  │
#  └── Company
#       ├── company_name
#       └── department
# ```

# In[99]:


class Address(BaseModel):
    city: str
    state: str
    pin_code: int


# In[100]:


class Company(BaseModel):
    company_name: str
    department: str


# In[101]:


class User(BaseModel):
    name: str
    age: int
    email: EmailStr
    address: Address
    company: Company



# In[104]:


User(
    name="Sunny",
    age=30,
    email="sunny@gmail.com",

    address={
        "city": "Bangalore",
        "state": "Karnataka",
        "pin_code": 560001
    },

    company={
        "company_name": "ABC Technologies",
        "department": "AI Engineering"
    }
)


# In[102]:


user_data = {
    "name": "Sunny",
    "age": 30,
    "email": "sunny@gmail.com",

    "address": {
        "city": "Bangalore",
        "state": "Karnataka",
        "pin_code": 560001
    },

    "company": {
        "company_name": "ABC Technologies",
        "department": "AI Engineering"
    }
}


# In[103]:


User.model_validate(user_data)


# In[ ]:




