#!/usr/bin/env python
# coding: utf-8

# # Complete Pydantic Code Notebook — 0 to 204
# 
# This notebook contains **every numbered code section (0–204)** from the complete Pydantic code collection above.
# 
# **Important:** Some sections intentionally demonstrate validation failures. Some require optional packages/services (`pydantic-settings`, `pydantic-extra-types`, `email-validator`, `fastapi`, `langchain`, `langgraph`) or an initialized LLM variable named `model`.
# 

# ## Table of Contents
# - **0.** Installation
# - **1.** Pydantic V1 → V2 Important API Changes
# - **2.** Basic `BaseModel`
# - **3.** Required Field
# - **4.** Field With Default Value
# - **5.** Required But Nullable Field
# - **6.** Optional-To-Provide + Nullable Field
# - **7.** Type Coercion / Parsing
# - **8.** Invalid Type
# - **9.** Normal Model Constructor
# - **10.** `model_validate()`
# - **11.** `model_validate_json()`
# - **12.** `model_validate_strings()`
# - **13.** `model_construct()` — Skip Validation
# - **14.** `model_rebuild()`
# - **15.** Basic `Field()`
# - **16.** Numeric Constraints
# - **17.** String Constraints
# - **18.** Decimal Constraints
# - **19.** `allow_inf_nan`
# - **20.** `Annotated`
# - **21.** Reusable Annotated Types
# - **22.** Simple Default Value
# - **23.** `default_factory`
# - **24.** Validate Default Value
# - **25.** Model-Level `validate_default`
# - **26.** Field Inspection
# - **27.** Nested Models
# - **28.** List Validation
# - **29.** Set Validation
# - **30.** Tuple Validation
# - **31.** Dictionary Validation
# - **32.** Sequence / Mapping
# - **33.** Boolean
# - **34.** Bytes
# - **35.** Decimal
# - **36.** Date / Time / Datetime / Timedelta
# - **37.** UUID
# - **38.** Path
# - **39.** Callable
# - **40.** `Any`
# - **41.** `Hashable`
# - **42.** Pydantic Strict Types
# - **43.** Positive / Negative Types
# - **44.** Finite Float
# - **45.** `Json`
# - **46.** Secret Types
# - **47.** Payment Card Number
# - **48.** ByteSize
# - **49.** Past / Future Date
# - **50.** Aware / Naive Datetime
# - **51.** Future / Past Datetime
# - **52.** File / Directory Paths
# - **53.** URLs
# - **54.** Email Validation
# - **55.** Database DSNs
# - **56.** IP Address Types
# - **57.** `Literal`
# - **58.** Enum
# - **59.** Simple Union
# - **60.** Left-to-Right Union Mode
# - **61.** Discriminated Union
# - **62.** Nested Discriminated Union
# - **63.** Basic Alias
# - **64.** Validation Alias
# - **65.** Serialization Alias
# - **66.** `AliasPath`
# - **67.** `AliasChoices`
# - **68.** Automatic Alias Generator
# - **69.** Built-In `to_camel`, `to_pascal`, `to_snake`
# - **70.** Field Validator — After Mode
# - **71.** Field Validator — Before Mode
# - **72.** Field Validator — Plain Mode
# - **73.** Field Validator — Wrap Mode
# - **74.** `AfterValidator`
# - **75.** `BeforeValidator`
# - **76.** `PlainValidator`
# - **77.** `WrapValidator`
# - **78.** Model Validator — After Mode
# - **79.** Model Validator — Before Mode
# - **80.** Model Validator — Wrap Mode
# - **81.** `ValidationInfo`
# - **82.** Validation Context
# - **83.** Validator Ordering
# - **84.** Raising Validation Error With `ValueError`
# - **85.** `ValidationError`
# - **86.** Inspect Error Details
# - **87.** Nested Validation Error Location
# - **88.** Serialization — `model_dump()`
# - **89.** Serialization — `model_dump_json()`
# - **90.** Python Mode vs JSON Mode
# - **91.** Include Fields
# - **92.** Exclude Fields
# - **93.** `exclude_none`
# - **94.** `exclude_unset`
# - **95.** `exclude_defaults`
# - **96.** Field-Level Exclusion
# - **97.** `field_serializer`
# - **98.** Wrap Field Serializer
# - **99.** `model_serializer`
# - **100.** Serialization Context
# - **101.** `SerializeAsAny`
# - **102.** Computed Field
# - **103.** Strict Validation at Call Level
# - **104.** Strict Field
# - **105.** Strict Using `Annotated`
# - **106.** Strict Whole Model
# - **107.** Extra Fields — Ignore
# - **108.** Extra Fields — Allow
# - **109.** Extra Fields — Forbid
# - **110.** `ConfigDict`
# - **111.** `validate_assignment`
# - **112.** Frozen Model
# - **113.** Frozen Field
# - **114.** `from_attributes`
# - **115.** Arbitrary Types
# - **116.** `revalidate_instances`
# - **117.** `RootModel`
# - **118.** Generic Models
# - **119.** Dynamic Model With `create_model()`
# - **120.** Private Attributes
# - **121.** `ClassVar`
# - **122.** Abstract Base Model
# - **123.** Structural Pattern Matching
# - **124.** Recursive Model
# - **125.** JSON Parsing
# - **126.** Lower-Level JSON Parsing
# - **127.** Partial JSON Parsing
# - **128.** TypeAdapter JSON Validation
# - **129.** Generate JSON Schema
# - **130.** TypeAdapter JSON Schema
# - **131.** JSON Schema — Validation Mode
# - **132.** JSON Schema — Serialization Mode
# - **133.** JSON Schema Metadata
# - **134.** `json_schema_extra`
# - **135.** `WithJsonSchema`
# - **136.** `SkipJsonSchema`
# - **137.** Custom Core Schema
# - **138.** Custom JSON Schema Hook
# - **139.** Basic `TypeAdapter`
# - **140.** TypeAdapter `dump_python()`
# - **141.** TypeAdapter `dump_json()`
# - **142.** TypeAdapter + TypedDict
# - **143.** Pydantic Dataclass
# - **144.** Standard Dataclass Inside BaseModel
# - **145.** Forward Annotation
# - **146.** String Forward Reference
# - **147.** `@validate_call`
# - **148.** `@validate_call` + Field Constraints
# - **149.** Async `@validate_call`
# - **150.** Pydantic Settings
# - **151.** `.env` Support
# - **152.** Environment Prefix
# - **153.** Case-Sensitive Settings
# - **154.** Nested Settings
# - **155.** Environment Nested Delimiter
# - **156.** Secrets Directory
# - **157.** `SecretStr` in Settings
# - **158.** Settings Source Customization
# - **159.** CLI Settings
# - **160.** Model Copy
# - **161.** Model Copy With Update
# - **162.** Pickling
# - **163.** Model Iteration
# - **164.** Field Ordering
# - **165.** `model_post_init()`
# - **166.** `model_extra`
# - **167.** `model_fields_set`
# - **168.** JSON Schema vs JSON Serialization
# - **169.** Validator vs Serializer
# - **170.** Custom Type With `Annotated`
# - **171.** `FailFast`
# - **172.** `OnErrorOmit`
# - **173.** Performance — Use `model_validate_json()`
# - **174.** Performance — Reuse `TypeAdapter`
# - **175.** Performance — Prefer Concrete Collections
# - **176.** Performance — Use `Any` When Validation Is Not Needed
# - **177.** Experimental Pipeline API
# - **178.** Experimental Partial Validation
# - **179.** Experimental `MISSING`
# - **180.** Pydantic Extra Types Installation
# - **181.** Color
# - **182.** Country
# - **183.** Currency
# - **184.** Phone Number
# - **185.** Coordinate
# - **186.** MAC Address
# - **187.** ISBN
# - **188.** Timezone Name
# - **189.** Semantic Version
# - **190.** ULID
# - **191.** Pydantic + FastAPI Request Model
# - **192.** FastAPI Response Model
# - **193.** Pydantic + LLM Structured Output
# - **194.** Structured Classification
# - **195.** Pydantic + LangChain Tool Input
# - **196.** Pydantic + Agent Routing
# - **197.** Pydantic + LangGraph Conditional Routing
# - **198.** Pydantic as LangGraph State
# - **199.** TypedDict as Lightweight LangGraph State
# - **200.** AI Application Settings
# - **201.** Schema Defining / Validation / Enforcement
# - **202.** Complete Real-World User Model
# - **203.** Full Agentic AI Example — Pydantic + LLM + Routing + Tool Schema
# - **204.** Pydantic Mental Model in Code

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

# Python class
# 
# +
# 
# type annotations
# 
# +
# 
# validation
# 
# +
# 
# serialization
# 
# +
# 
# JSON Schema

# ## 0. Installation

# ```bash
# pip install -U pydantic
# pip install -U pydantic-settings
# pip install "pydantic[email]"
# ```

# ## 1. Pydantic V1 → V2 Important API Changes

# In[ ]:


# Pydantic V1                 Pydantic V2
# .dict()                     .model_dump()
# .json()                     .model_dump_json()
# .parse_obj()                .model_validate()
# .parse_raw()                .model_validate_json()
# .schema()                   .model_json_schema()
# .copy()                     .model_copy()
# @validator                  @field_validator
# @root_validator             @model_validator


# ## 2. Basic `BaseModel`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Sunny", age=30)
print(user)
print(user.name)
print(user.age)


# ## 3. Required Field

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str

user = User(name="Sunny")
print(user)


# ## 4. Field With Default Value

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    country: str = "India"

user = User(name="Sunny")
print(user)


# ## 5. Required But Nullable Field

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    middle_name: str | None

user = User(middle_name=None)
print(user)


# ## 6. Optional-To-Provide + Nullable Field

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    middle_name: str | None = None

user = User()
print(user)


# ## 7. Type Coercion / Parsing

# In[ ]:


from pydantic import BaseModel

class Employee(BaseModel):
    employee_id: int

employee = Employee(employee_id="123")
print(employee)
print(type(employee.employee_id))


# ## 8. Invalid Type

# In[ ]:


from pydantic import BaseModel

class Employee(BaseModel):
    employee_id: int

employee = Employee(employee_id="Sunny")


# ## 9. Normal Model Constructor

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Sunny", age=30)
print(user)


# ## 10. `model_validate()`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

data = {"name": "Sunny", "age": "30"}
user = User.model_validate(data)
print(user)


# ## 11. `model_validate_json()`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

json_data = '{"name":"Sunny","age":30}'
user = User.model_validate_json(json_data)
print(user)


# ## 12. `model_validate_strings()`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

data = {"name": "Sunny", "age": "30"}
user = User.model_validate_strings(data)
print(user)


# ## 13. `model_construct()` — Skip Validation

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User.model_construct(name="Sunny", age="not-an-int")
print(user)


# ## 14. `model_rebuild()`

# In[ ]:


# The notebook placed this future import in a later cell; annotations are
# already supported by the target Python version.
from pydantic import BaseModel

class Employee(BaseModel):
    name: str
    manager: Employee | None = None

Employee.model_rebuild()


# ## 15. Basic `Field()`

# In[ ]:


from pydantic import BaseModel, Field

class Employee(BaseModel):
    name: str
    age: int = Field(ge=18, le=65)


# ## 16. Numeric Constraints

# In[ ]:


from pydantic import BaseModel, Field

class Product(BaseModel):
    price: float = Field(gt=0)
    quantity: int = Field(ge=0)
    discount: float = Field(ge=0, le=100)
    pack_size: int = Field(multiple_of=5)


# ## 17. String Constraints

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    code: str = Field(pattern=r"^[A-Z]{3}\d{3}$")


# ## 18. Decimal Constraints

# In[ ]:


from decimal import Decimal
from pydantic import BaseModel, Field

class Product(BaseModel):
    price: Decimal = Field(max_digits=8, decimal_places=2)

product = Product(price="12345.67")
print(product)


# ## 19. `allow_inf_nan`

# In[ ]:


from pydantic import BaseModel, Field

class Metrics(BaseModel):
    score: float = Field(allow_inf_nan=False)


# ## 20. `Annotated`

# In[ ]:


from typing import Annotated
from pydantic import BaseModel, Field

Age = Annotated[int, Field(ge=18, le=100)]

class User(BaseModel):
    name: str
    age: Age


# ## 21. Reusable Annotated Types

# In[ ]:


from typing import Annotated
from pydantic import BaseModel, Field

PositivePrice = Annotated[float, Field(gt=0)]

class Product(BaseModel):
    name: str
    price: PositivePrice

class Service(BaseModel):
    name: str
    price: PositivePrice


# ## 22. Simple Default Value

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    status: str = "active"

user = User()
print(user)


# ## 23. `default_factory`

# In[ ]:


from datetime import datetime
from pydantic import BaseModel, Field

class Event(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)

event = Event()
print(event)


# ## 24. Validate Default Value

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    age: int = Field(default="30", validate_default=True)

user = User()
print(user)
print(type(user.age))


# ## 25. Model-Level `validate_default`

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(validate_default=True)
    age: int = "30"

print(User())


# ## 26. Field Inspection

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(description="User name")
    age: int

print(User.model_fields)
print(User.model_fields["name"])
print(User.model_fields["name"].description)


# ## 27. Nested Models

# In[ ]:


from pydantic import BaseModel

class Address(BaseModel):
    city: str
    pin: int

class User(BaseModel):
    name: str
    address: Address

user = User(name="Sunny", address={"city":"Bangalore","pin":560001})
print(user)


# ## 28. List Validation

# In[ ]:


from pydantic import BaseModel

class Student(BaseModel):
    marks: list[int]

student = Student(marks=["80",90,95])
print(student)


# ## 29. Set Validation

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    skills: set[str]

user = User(skills=["Python","AI","Python"])
print(user.skills)


# ## 30. Tuple Validation

# In[ ]:


from pydantic import BaseModel

class Coordinate(BaseModel):
    location: tuple[float,float]

coordinate = Coordinate(location=(12.97,77.59))
print(coordinate)


# ## 31. Dictionary Validation

# In[ ]:


from pydantic import BaseModel

class Scores(BaseModel):
    values: dict[str,float]

scores = Scores(values={"math":"95.5","science":90})
print(scores)


# ## 32. Sequence / Mapping

# In[ ]:


from collections.abc import Mapping, Sequence
from pydantic import BaseModel

class Data(BaseModel):
    numbers: Sequence[int]
    mapping: Mapping[str,int]


# ## 33. Boolean

# In[ ]:


from pydantic import BaseModel

class Settings(BaseModel):
    enabled: bool

settings = Settings(enabled=True)


# ## 34. Bytes

# In[ ]:


from pydantic import BaseModel

class FileData(BaseModel):
    content: bytes

obj = FileData(content=b"hello")


# ## 35. Decimal

# In[ ]:


from decimal import Decimal
from pydantic import BaseModel

class Payment(BaseModel):
    amount: Decimal

payment = Payment(amount="1250.50")


# ## 36. Date / Time / Datetime / Timedelta

# In[ ]:


from datetime import date, datetime, time, timedelta
from pydantic import BaseModel

class Schedule(BaseModel):
    event_date: date
    event_time: time
    created_at: datetime
    duration: timedelta


# ## 37. UUID

# In[ ]:


from uuid import UUID
from pydantic import BaseModel

class User(BaseModel):
    id: UUID

user = User(id="12345678-1234-5678-1234-567812345678")
print(user.id)


# ## 38. Path

# In[ ]:


from pathlib import Path
from pydantic import BaseModel

class FileConfig(BaseModel):
    path: Path

config = FileConfig(path="./data/file.txt")
print(config.path)


# ## 39. Callable

# In[ ]:


from collections.abc import Callable
from pydantic import BaseModel

class Processor(BaseModel):
    handler: Callable[[int],int]

def double(value:int)->int:
    return value*2

processor=Processor(handler=double)
print(processor.handler(10))


# ## 40. `Any`

# In[ ]:


from typing import Any
from pydantic import BaseModel

class Payload(BaseModel):
    data: Any

payload=Payload(data={"anything":[1,2,3]})


# ## 41. `Hashable`

# In[ ]:


from collections.abc import Hashable
from pydantic import BaseModel

class KeyModel(BaseModel):
    key: Hashable


# ## 42. Pydantic Strict Types

# In[ ]:


from pydantic import BaseModel, StrictBool, StrictBytes, StrictFloat, StrictInt, StrictStr

class StrictModel(BaseModel):
    age: StrictInt
    score: StrictFloat
    name: StrictStr
    enabled: StrictBool
    content: StrictBytes


# ## 43. Positive / Negative Types

# In[ ]:


from pydantic import BaseModel, NegativeFloat, NegativeInt, NonNegativeFloat, NonNegativeInt, NonPositiveFloat, NonPositiveInt, PositiveFloat, PositiveInt

class Numbers(BaseModel):
    positive_int: PositiveInt
    negative_int: NegativeInt
    non_negative_int: NonNegativeInt
    non_positive_int: NonPositiveInt
    positive_float: PositiveFloat
    negative_float: NegativeFloat
    non_negative_float: NonNegativeFloat
    non_positive_float: NonPositiveFloat


# ## 44. Finite Float

# In[ ]:


from pydantic import BaseModel, FiniteFloat

class Metric(BaseModel):
    score: FiniteFloat


# ## 45. `Json`

# In[ ]:


from pydantic import BaseModel, Json

class Payload(BaseModel):
    data: Json[dict[str,int]]

payload=Payload(data='{"a":10,"b":20}')
print(payload.data)


# ## 46. Secret Types

# In[ ]:


from pydantic import BaseModel, SecretBytes, SecretStr

class Credentials(BaseModel):
    password: SecretStr
    token: SecretBytes

credentials=Credentials(password="mypassword",token=b"secret-token")
print(credentials)
print(credentials.password.get_secret_value())


# ## 47. Payment Card Number

# In[ ]:


from pydantic import BaseModel, PaymentCardNumber

class Card(BaseModel):
    number: PaymentCardNumber


# ## 48. ByteSize

# In[ ]:


from pydantic import BaseModel, ByteSize

class Storage(BaseModel):
    size: ByteSize

storage=Storage(size="10 MB")
print(storage.size)


# ## 49. Past / Future Date

# In[ ]:


from pydantic import BaseModel, FutureDate, PastDate

class Dates(BaseModel):
    birthday: PastDate
    appointment: FutureDate


# ## 50. Aware / Naive Datetime

# In[ ]:


from pydantic import AwareDatetime, BaseModel, NaiveDatetime

class Event(BaseModel):
    server_time: AwareDatetime
    local_time: NaiveDatetime


# ## 51. Future / Past Datetime

# In[ ]:


from pydantic import BaseModel, FutureDatetime, PastDatetime

class Timeline(BaseModel):
    previous_event: PastDatetime
    next_event: FutureDatetime


# ## 52. File / Directory Paths

# In[ ]:


from pydantic import BaseModel, DirectoryPath, FilePath, NewPath

class Paths(BaseModel):
    existing_file: FilePath
    existing_directory: DirectoryPath
    new_file: NewPath


# ## 53. URLs

# In[ ]:


from pydantic import AnyHttpUrl, AnyUrl, BaseModel, HttpUrl

class URLs(BaseModel):
    website: HttpUrl
    service: AnyHttpUrl
    arbitrary: AnyUrl


# ## 54. Email Validation

# In[ ]:


from pydantic import BaseModel, EmailStr, NameEmail

class User(BaseModel):
    email: EmailStr
    contact: NameEmail


# ## 55. Database DSNs

# In[ ]:


from pydantic import AmqpDsn, BaseModel, ClickHouseDsn, KafkaDsn, MariaDBDsn, MongoDsn, MySQLDsn, NatsDsn, PostgresDsn, RedisDsn, SnowflakeDsn

class DatabaseConfig(BaseModel):
    postgres: PostgresDsn | None = None
    mysql: MySQLDsn | None = None
    mariadb: MariaDBDsn | None = None
    mongo: MongoDsn | None = None
    redis: RedisDsn | None = None
    kafka: KafkaDsn | None = None
    amqp: AmqpDsn | None = None
    nats: NatsDsn | None = None
    clickhouse: ClickHouseDsn | None = None
    snowflake: SnowflakeDsn | None = None


# ## 56. IP Address Types

# In[ ]:


from pydantic import BaseModel, IPvAnyAddress, IPvAnyInterface, IPvAnyNetwork

class Network(BaseModel):
    address: IPvAnyAddress
    interface: IPvAnyInterface
    network: IPvAnyNetwork


# ## 57. `Literal`

# In[ ]:


from typing import Literal
from pydantic import BaseModel

class RouteDecision(BaseModel):
    route: Literal["RAG","WEB","LLM"]

decision=RouteDecision(route="RAG")


# ## 58. Enum

# In[ ]:


from enum import Enum
from pydantic import BaseModel

class Role(str, Enum):
    ADMIN="admin"
    USER="user"

class User(BaseModel):
    role: Role

user=User(role="admin")
print(user)


# ## 59. Simple Union

# In[ ]:


from pydantic import BaseModel

class Data(BaseModel):
    value: int | str

print(Data(value=10))
print(Data(value="hello"))


# ## 60. Left-to-Right Union Mode

# In[ ]:


from typing import Union
from pydantic import BaseModel, Field

class Data(BaseModel):
    value: Union[int,str] = Field(union_mode="left_to_right")


# ## 61. Discriminated Union

# In[ ]:


from typing import Literal, Union
from pydantic import BaseModel, Field

class Cat(BaseModel):
    type: Literal["cat"]
    meows: int

class Dog(BaseModel):
    type: Literal["dog"]
    barks: int

class PetOwner(BaseModel):
    pet: Union[Cat,Dog] = Field(discriminator="type")

owner=PetOwner(pet={"type":"dog","barks":5})
print(owner)


# ## 62. Nested Discriminated Union

# In[ ]:


from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

class BlackCat(BaseModel):
    pet_type: Literal["cat"]
    color: Literal["black"]

class WhiteCat(BaseModel):
    pet_type: Literal["cat"]
    color: Literal["white"]

Cat = Annotated[Union[BlackCat,WhiteCat], Field(discriminator="color")]

class Dog(BaseModel):
    pet_type: Literal["dog"]

Pet = Annotated[Union[Cat,Dog], Field(discriminator="pet_type")]


# ## 63. Basic Alias

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    first_name: str = Field(alias="firstName")

user=User(firstName="Sunny")
print(user.first_name)


# ## 64. Validation Alias

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(validation_alias="userName")

user=User(userName="Sunny")
print(user)


# ## 65. Serialization Alias

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    first_name: str = Field(serialization_alias="firstName")

user=User(first_name="Sunny")
print(user.model_dump(by_alias=True))


# ## 66. `AliasPath`

# In[ ]:


from pydantic import AliasPath, BaseModel, Field

class User(BaseModel):
    first_name: str = Field(validation_alias=AliasPath("names",0))

user=User.model_validate({"names":["Sunny","Savita"]})
print(user)


# ## 67. `AliasChoices`

# In[ ]:


from pydantic import AliasChoices, BaseModel, Field

class User(BaseModel):
    first_name: str = Field(validation_alias=AliasChoices("first_name","firstName","fname"))

print(User(fname="Sunny"))


# ## 68. Automatic Alias Generator

# In[ ]:


from pydantic import BaseModel, ConfigDict

def to_camel_case(name:str)->str:
    parts=name.split("_")
    return parts[0]+"".join(word.title() for word in parts[1:])

class User(BaseModel):
    model_config=ConfigDict(alias_generator=to_camel_case,populate_by_name=True)
    first_name: str
    last_name: str

user=User(firstName="Sunny",lastName="Savita")
print(user)


# ## 69. Built-In `to_camel`, `to_pascal`, `to_snake`

# In[ ]:


from pydantic.alias_generators import to_camel, to_pascal, to_snake

print(to_camel("first_name"))
print(to_pascal("first_name"))
print(to_snake("FirstName"))


# ## 70. Field Validator — After Mode

# In[ ]:


from pydantic import BaseModel, field_validator

class User(BaseModel):
    age: int

    @field_validator("age")
    @classmethod
    def validate_age(cls,value):
        if value < 18:
            raise ValueError("Age must be at least 18")
        return value


# ## 71. Field Validator — Before Mode

# In[ ]:


from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str

    @field_validator("name", mode="before")
    @classmethod
    def clean_name(cls,value):
        return value.strip()

user=User(name="   Sunny   ")
print(user)


# ## 72. Field Validator — Plain Mode

# In[ ]:


from pydantic import BaseModel, field_validator

class Model(BaseModel):
    value: int

    @field_validator("value", mode="plain")
    @classmethod
    def validate_value(cls,value):
        return int(value)


# ## 73. Field Validator — Wrap Mode

# In[ ]:


from pydantic import BaseModel, ValidatorFunctionWrapHandler, field_validator

class Model(BaseModel):
    value: int

    @field_validator("value", mode="wrap")
    @classmethod
    def validate_value(cls,value,handler:ValidatorFunctionWrapHandler):
        print("Before validation:",value)
        result=handler(value)
        print("After validation:",result)
        return result


# ## 74. `AfterValidator`

# In[ ]:


from typing import Annotated
from pydantic import AfterValidator, BaseModel

def positive(value:int)->int:
    if value <= 0:
        raise ValueError("Must be positive")
    return value

PositiveInteger=Annotated[int,AfterValidator(positive)]

class Product(BaseModel):
    quantity: PositiveInteger


# ## 75. `BeforeValidator`

# In[ ]:


from typing import Annotated
from pydantic import BaseModel, BeforeValidator

def strip_spaces(value):
    if isinstance(value,str):
        return value.strip()
    return value

CleanString=Annotated[str,BeforeValidator(strip_spaces)]

class User(BaseModel):
    name: CleanString


# ## 76. `PlainValidator`

# In[ ]:


from typing import Annotated
from pydantic import BaseModel, PlainValidator

def convert_value(value):
    return int(value)

CustomInt=Annotated[int,PlainValidator(convert_value)]

class Data(BaseModel):
    number: CustomInt


# ## 77. `WrapValidator`

# In[ ]:


from typing import Annotated
from pydantic import BaseModel, ValidatorFunctionWrapHandler, WrapValidator

def validate_number(value,handler:ValidatorFunctionWrapHandler):
    result=handler(value)
    if result < 0:
        raise ValueError("Number cannot be negative")
    return result

PositiveNumber=Annotated[int,WrapValidator(validate_number)]

class Model(BaseModel):
    number: PositiveNumber


# ## 78. Model Validator — After Mode

# In[ ]:


from typing_extensions import Self
from pydantic import BaseModel, model_validator

class Signup(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def validate_passwords(self)->Self:
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


# ## 79. Model Validator — Before Mode

# In[ ]:


from typing import Any
from pydantic import BaseModel, model_validator

class User(BaseModel):
    name: str

    @model_validator(mode="before")
    @classmethod
    def preprocess(cls,data:Any):
        if isinstance(data,dict):
            data=data.copy()
            if "name" in data:
                data["name"]=data["name"].strip()
        return data


# ## 80. Model Validator — Wrap Mode

# In[ ]:


from typing import Any
from pydantic import BaseModel, ModelWrapValidatorHandler, model_validator

class User(BaseModel):
    name: str

    @model_validator(mode="wrap")
    @classmethod
    def log_validation(cls,data:Any,handler:ModelWrapValidatorHandler):
        print("Input:",data)
        result=handler(data)
        print("Validated:",result)
        return result


# ## 81. `ValidationInfo`

# In[ ]:


from pydantic import BaseModel, ValidationInfo, field_validator

class User(BaseModel):
    password: str
    confirm_password: str

    @field_validator("confirm_password")
    @classmethod
    def validate_confirm_password(cls,value,info:ValidationInfo):
        password=info.data.get("password")
        if password != value:
            raise ValueError("Passwords do not match")
        return value


# ## 82. Validation Context

# In[ ]:


from pydantic import BaseModel, ValidationInfo, field_validator

class Payment(BaseModel):
    amount: float

    @field_validator("amount")
    @classmethod
    def validate_amount(cls,value,info:ValidationInfo):
        limit=(info.context or {}).get("limit",10000)
        if value > limit:
            raise ValueError("Amount exceeds limit")
        return value

payment=Payment.model_validate({"amount":5000},context={"limit":6000})


# ## 83. Validator Ordering

# In[ ]:


from typing import Annotated
from pydantic import AfterValidator, BaseModel, BeforeValidator

def before_one(value):
    print("before_one")
    return value

def before_two(value):
    print("before_two")
    return value

def after_one(value):
    print("after_one")
    return value

CustomInt=Annotated[int,BeforeValidator(before_one),BeforeValidator(before_two),AfterValidator(after_one)]

class Model(BaseModel):
    value: CustomInt

Model(value="10")


# ## 84. Raising Validation Error With `ValueError`

# In[ ]:


from pydantic import BaseModel, field_validator

class User(BaseModel):
    age: int

    @field_validator("age")
    @classmethod
    def validate_age(cls,value):
        if value < 18:
            raise ValueError("Age must be at least 18")
        return value


# ## 85. `ValidationError`

# In[ ]:


from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int

try:
    User(name="Sunny",age="abc")
except ValidationError as error:
    print(error)
    print(error.errors())
    print(error.error_count())
    print(error.json())


# ## 86. Inspect Error Details

# In[ ]:


from pydantic import BaseModel, ValidationError

class User(BaseModel):
    age: int

try:
    User(age="abc")
except ValidationError as error:
    for item in error.errors():
        print("type:",item.get("type"))
        print("loc:",item.get("loc"))
        print("msg:",item.get("msg"))
        print("input:",item.get("input"))
        print("ctx:",item.get("ctx"))
        print("url:",item.get("url"))


# ## 87. Nested Validation Error Location

# In[ ]:


from pydantic import BaseModel, ValidationError

class Address(BaseModel):
    pin: int

class User(BaseModel):
    address: Address

try:
    User(address={"pin":"wrong"})
except ValidationError as error:
    print(error.errors())


# ## 88. Serialization — `model_dump()`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user=User(name="Sunny",age=30)
data=user.model_dump()
print(data)
print(type(data))


# ## 89. Serialization — `model_dump_json()`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user=User(name="Sunny",age=30)
json_data=user.model_dump_json()
print(json_data)
print(type(json_data))


# ## 90. Python Mode vs JSON Mode

# In[ ]:


from datetime import datetime
from pydantic import BaseModel

class Event(BaseModel):
    created_at: datetime

event=Event(created_at=datetime.now())
print(event.model_dump(mode="python"))
print(event.model_dump(mode="json"))


# ## 91. Include Fields

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    password: str

user=User(name="Sunny",email="sunny@example.com",password="secret")
print(user.model_dump(include={"name","email"}))


# ## 92. Exclude Fields

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    password: str

user=User(name="Sunny",email="sunny@example.com",password="secret")
print(user.model_dump(exclude={"password"}))


# ## 93. `exclude_none`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    middle_name: str | None = None

user=User(name="Sunny")
print(user.model_dump(exclude_none=True))


# ## 94. `exclude_unset`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    country: str = "India"

user=User(name="Sunny")
print(user.model_dump(exclude_unset=True))


# ## 95. `exclude_defaults`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    country: str = "India"

user=User(name="Sunny")
print(user.model_dump(exclude_defaults=True))


# ## 96. Field-Level Exclusion

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    password: str = Field(exclude=True)

user=User(name="Sunny",password="secret")
print(user.model_dump())


# ## 97. `field_serializer`

# In[ ]:


from pydantic import BaseModel, field_serializer

class Payment(BaseModel):
    amount: float

    @field_serializer("amount")
    def serialize_amount(self,value):
        return f"₹{value:.2f}"

payment=Payment(amount=1250)
print(payment.model_dump())


# ## 98. Wrap Field Serializer

# In[ ]:


from pydantic import BaseModel, SerializerFunctionWrapHandler, field_serializer

class User(BaseModel):
    name: str

    @field_serializer("name",mode="wrap")
    def serialize_name(self,value,handler:SerializerFunctionWrapHandler):
        result=handler(value)
        return result.upper()


# ## 99. `model_serializer`

# In[ ]:


from pydantic import BaseModel, model_serializer

class User(BaseModel):
    first_name: str
    last_name: str

    @model_serializer
    def serialize_model(self):
        return {"full_name":f"{self.first_name} {self.last_name}"}

user=User(first_name="Sunny",last_name="Savita")
print(user.model_dump())


# ## 100. Serialization Context

# In[ ]:


from pydantic import BaseModel, FieldSerializationInfo, field_serializer

class User(BaseModel):
    email: str

    @field_serializer("email")
    def serialize_email(self,value,info:FieldSerializationInfo):
        role=(info.context or {}).get("role")
        if role == "admin":
            return value
        return "***hidden***"

user=User(email="sunny@example.com")
print(user.model_dump(context={"role":"customer"}))


# ## 101. `SerializeAsAny`

# In[ ]:


from pydantic import BaseModel, SerializeAsAny

class User(BaseModel):
    name: str

class Admin(User):
    password: str

class Wrapper(BaseModel):
    user: SerializeAsAny[User]

wrapper=Wrapper(user=Admin(name="Sunny",password="secret"))
print(wrapper.model_dump())


# ## 102. Computed Field

# In[ ]:


from pydantic import BaseModel, computed_field

class Product(BaseModel):
    price: float
    quantity: int

    @computed_field
    @property
    def total(self)->float:
        return self.price*self.quantity

product=Product(price=100,quantity=5)
print(product.model_dump())


# ## 103. Strict Validation at Call Level

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    age: int

user=User.model_validate({"age":30},strict=True)
print(user)


# ## 104. Strict Field

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    age: int = Field(strict=True)


# ## 105. Strict Using `Annotated`

# In[ ]:


from typing import Annotated
from pydantic import BaseModel, Strict

StrictInteger=Annotated[int,Strict()]

class User(BaseModel):
    age: StrictInteger


# ## 106. Strict Whole Model

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(strict=True)
    age: int


# ## 107. Extra Fields — Ignore

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(extra="ignore")
    name: str

user=User(name="Sunny",unknown="hello")
print(user)


# ## 108. Extra Fields — Allow

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(extra="allow")
    name: str

user=User(name="Sunny",company="OpenAI")
print(user)
print(user.model_extra)


# ## 109. Extra Fields — Forbid

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(extra="forbid")
    name: str

User(name="Sunny",unknown="Not allowed")


# ## 110. `ConfigDict`

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(
        title="User",
        str_strip_whitespace=True,
        str_to_lower=False,
        str_to_upper=False,
        extra="forbid",
        frozen=False,
        populate_by_name=True,
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
        strict=False,
        validate_default=True,
        hide_input_in_errors=True,
        serialize_by_alias=True,
    )
    name: str


# ## 111. `validate_assignment`

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(validate_assignment=True)
    age: int

user=User(age=30)
user.age=35
user.age="invalid"


# ## 112. Frozen Model

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(frozen=True)
    name: str

user=User(name="Sunny")
user.name="John"


# ## 113. Frozen Field

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    id: int = Field(frozen=True)
    name: str


# ## 114. `from_attributes`

# In[ ]:


from pydantic import BaseModel, ConfigDict

class ORMUser:
    def __init__(self,name,age):
        self.name=name
        self.age=age

class UserResponse(BaseModel):
    model_config=ConfigDict(from_attributes=True)
    name: str
    age: int

orm_user=ORMUser("Sunny",30)
user=UserResponse.model_validate(orm_user)
print(user)


# ## 115. Arbitrary Types

# In[ ]:


from pydantic import BaseModel, ConfigDict

class DatabaseClient:
    pass

class App(BaseModel):
    model_config=ConfigDict(arbitrary_types_allowed=True)
    db: DatabaseClient

app=App(db=DatabaseClient())


# ## 116. `revalidate_instances`

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(revalidate_instances="always")
    age: int

class Container(BaseModel):
    user: User


# ## 117. `RootModel`

# In[ ]:


from pydantic import RootModel

class Numbers(RootModel[list[int]]):
    pass

numbers=Numbers([1,2,3])
print(numbers)
print(numbers.root)


# ## 118. Generic Models

# In[ ]:


from typing import Generic, TypeVar
from pydantic import BaseModel

T=TypeVar("T")

class APIResponse(BaseModel,Generic[T]):
    data: T

class User(BaseModel):
    name: str

response=APIResponse[User](data=User(name="Sunny"))
print(response)


# ## 119. Dynamic Model With `create_model()`

# In[ ]:


from pydantic import create_model

DynamicUser=create_model("DynamicUser",name=(str,...),age=(int,18))
user=DynamicUser(name="Sunny")
print(user)


# ## 120. Private Attributes

# In[ ]:


from pydantic import BaseModel, PrivateAttr

class User(BaseModel):
    name: str
    _cache: dict = PrivateAttr(default_factory=dict)

user=User(name="Sunny")
user._cache["token"]="abc"
print(user._cache)


# ## 121. `ClassVar`

# In[ ]:


from typing import ClassVar
from pydantic import BaseModel

class User(BaseModel):
    type_name: ClassVar[str] = "user"
    name: str

print(User.model_fields)


# ## 122. Abstract Base Model

# In[ ]:


from abc import ABC, abstractmethod
from pydantic import BaseModel

class Animal(BaseModel,ABC):
    name: str

    @abstractmethod
    def speak(self)->str:
        pass

class Dog(Animal):
    def speak(self)->str:
        return "Woof"

dog=Dog(name="Bruno")
print(dog.speak())


# ## 123. Structural Pattern Matching

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user=User(name="Sunny",age=30)

match user:
    case User(name="Sunny"):
        print("Sunny found")
    case _:
        print("Other user")


# ## 124. Recursive Model

# In[ ]:


# The notebook placed this future import in a later cell; annotations are
# already supported by the target Python version.
from pydantic import BaseModel, Field

class Tree(BaseModel):
    name: str
    children: list[Tree] = Field(default_factory=list)

tree=Tree(name="root",children=[{"name":"child"}])
print(tree)


# ## 125. JSON Parsing

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

json_data='{"name":"Sunny","age":30}'
user=User.model_validate_json(json_data)
print(user)


# ## 126. Lower-Level JSON Parsing

# In[ ]:


from pydantic_core import from_json

data=from_json(b'{"name":"Sunny","age":30}')
print(data)


# ## 127. Partial JSON Parsing

# In[ ]:


from pydantic_core import from_json

partial_json=b'{"name":"Sunny","age":'
result=from_json(partial_json,allow_partial=True)
print(result)


# ## 128. TypeAdapter JSON Validation

# In[ ]:


from pydantic import TypeAdapter

adapter=TypeAdapter(list[int])
result=adapter.validate_json('["1","2","3"]')
print(result)


# ## 129. Generate JSON Schema

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

schema=User.model_json_schema()
print(schema)


# ## 130. TypeAdapter JSON Schema

# In[ ]:


from pydantic import TypeAdapter

adapter=TypeAdapter(list[int])
print(adapter.json_schema())


# ## 131. JSON Schema — Validation Mode

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    age: int

schema=User.model_json_schema(mode="validation")
print(schema)


# ## 132. JSON Schema — Serialization Mode

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    age: int

schema=User.model_json_schema(mode="serialization")
print(schema)


# ## 133. JSON Schema Metadata

# In[ ]:


from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(title="User Name",description="Full name of the user",examples=["Sunny"])


# ## 134. `json_schema_extra`

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(json_schema_extra={"examples":[{"name":"Sunny"}]})
    name: str


# ## 135. `WithJsonSchema`

# In[ ]:


from typing import Annotated
from pydantic import BaseModel, WithJsonSchema

CustomString=Annotated[str,WithJsonSchema({"type":"string","description":"Custom schema"})]

class Model(BaseModel):
    value: CustomString


# ## 136. `SkipJsonSchema`

# In[ ]:


from pydantic import BaseModel
from pydantic.json_schema import SkipJsonSchema

class Model(BaseModel):
    visible: str
    hidden: SkipJsonSchema[str]


# ## 137. Custom Core Schema

# In[ ]:


from typing import Any
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

class Username(str):
    @classmethod
    def __get_pydantic_core_schema__(cls,source_type:Any,handler:GetCoreSchemaHandler)->CoreSchema:
        return core_schema.no_info_after_validator_function(cls,core_schema.str_schema())

class User(BaseModel):
    username: Username


# ## 138. Custom JSON Schema Hook

# In[ ]:


from typing import Any
from pydantic import BaseModel, GetJsonSchemaHandler
from pydantic_core import CoreSchema

class CustomType(str):
    @classmethod
    def __get_pydantic_json_schema__(cls,core_schema:CoreSchema,handler:GetJsonSchemaHandler)->dict[str,Any]:
        schema=handler(core_schema)
        schema["description"]="My custom type"
        return schema

class Model(BaseModel):
    value: CustomType


# ## 139. Basic `TypeAdapter`

# In[ ]:


from pydantic import TypeAdapter

adapter=TypeAdapter(list[int])
result=adapter.validate_python(["1","2","3"])
print(result)


# ## 140. TypeAdapter `dump_python()`

# In[ ]:


from pydantic import TypeAdapter

adapter=TypeAdapter(list[int])
data=adapter.dump_python([1,2,3])
print(data)


# ## 141. TypeAdapter `dump_json()`

# In[ ]:


from pydantic import TypeAdapter

adapter=TypeAdapter(list[int])
data=adapter.dump_json([1,2,3])
print(data)


# ## 142. TypeAdapter + TypedDict

# In[ ]:


from typing_extensions import TypedDict
from pydantic import TypeAdapter

class Employee(TypedDict):
    name: str
    age: int

adapter=TypeAdapter(list[Employee])
result=adapter.validate_python([{"name":"Sunny","age":"30"}])
print(result)


# ## 143. Pydantic Dataclass

# In[ ]:


from pydantic.dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int

user=User(name="Sunny",age="30")
print(user)
print(type(user.age))


# ## 144. Standard Dataclass Inside BaseModel

# In[ ]:


from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class Address:
    city: str

class User(BaseModel):
    name: str
    address: Address

user=User(name="Sunny",address={"city":"Bangalore"})
print(user)


# ## 145. Forward Annotation

# In[ ]:


# The notebook placed this future import in a later cell; annotations are
# already supported by the target Python version.
from pydantic import BaseModel

class Employee(BaseModel):
    name: str
    manager: Employee | None = None


# ## 146. String Forward Reference

# In[ ]:


from pydantic import BaseModel

class Employee(BaseModel):
    name: str
    manager: "Employee | None" = None

Employee.model_rebuild()


# ## 147. `@validate_call`

# In[ ]:


from pydantic import validate_call

@validate_call
def create_user(name:str,age:int):
    return {"name":name,"age":age}

result=create_user("Sunny","30")
print(result)


# ## 148. `@validate_call` + Field Constraints

# In[ ]:


from typing import Annotated
from pydantic import Field, validate_call

@validate_call
def transfer_money(amount:Annotated[float,Field(gt=0)]):
    return amount

print(transfer_money(5000))


# ## 149. Async `@validate_call`

# In[ ]:


import asyncio
from pydantic import validate_call

@validate_call
async def fetch_user(user_id:int):
    await asyncio.sleep(0.1)
    return {"user_id":user_id}

result=asyncio.run(fetch_user("123"))
print(result)


# ## 150. Pydantic Settings

# In[ ]:


from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    api_key: str
    debug: bool = False

settings=Settings()
print(settings)


# ## 151. `.env` Support

# In[ ]:


from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env")
    openai_api_key: str
    database_url: str

settings=Settings()
print(settings)


# ## 152. Environment Prefix

# In[ ]:


from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_prefix="MYAPP_")
    database_url: str
    api_key: str


# Environment example:
# ```bash
# export MYAPP_DATABASE_URL="postgresql://localhost/db"
# export MYAPP_API_KEY="secret"
# ```

# ## 153. Case-Sensitive Settings

# In[ ]:


from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config=SettingsConfigDict(case_sensitive=True)
    API_KEY: str


# ## 154. Nested Settings

# In[ ]:


from pydantic import BaseModel
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseModel):
    host: str
    port: int

class Settings(BaseSettings):
    database: DatabaseSettings


# ## 155. Environment Nested Delimiter

# In[ ]:


from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseModel):
    host: str
    port: int

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_nested_delimiter="__")
    database: DatabaseSettings


# Environment example:
# ```bash
# export DATABASE__HOST="localhost"
# export DATABASE__PORT="5432"
# ```

# ## 156. Secrets Directory

# In[ ]:


from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config=SettingsConfigDict(secrets_dir="/run/secrets")
    database_password: str


# ## 157. `SecretStr` in Settings

# In[ ]:


from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: SecretStr

settings=Settings()
print(settings.openai_api_key)
print(settings.openai_api_key.get_secret_value())


# ## 158. Settings Source Customization

# In[ ]:


from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

class Settings(BaseSettings):
    api_key: str

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource,...]:
        return (init_settings,env_settings,dotenv_settings,file_secret_settings)


# ## 159. CLI Settings

# In[ ]:


from pydantic_settings import BaseSettings, CliApp

class Settings(BaseSettings):
    host: str = "localhost"
    port: int = 8000

CliApp.run(Settings)


# ## 160. Model Copy

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user=User(name="Sunny",age=30)
copy_user=user.model_copy()
print(copy_user)


# ## 161. Model Copy With Update

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user=User(name="Sunny",age=30)
updated=user.model_copy(update={"age":31})
print(updated)


# ## 162. Pickling

# In[ ]:


import pickle
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user=User(name="Sunny",age=30)
serialized=pickle.dumps(user)
restored=pickle.loads(serialized)
print(restored)


# ## 163. Model Iteration

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user=User(name="Sunny",age=30)

for key,value in user:
    print(key,value)


# ## 164. Field Ordering

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    age: int

print(list(User.model_fields))


# ## 165. `model_post_init()`

# In[ ]:


from typing import Any
from pydantic import BaseModel

class User(BaseModel):
    first_name: str
    last_name: str
    full_name: str = ""

    def model_post_init(self,context:Any)->None:
        self.full_name=f"{self.first_name} {self.last_name}"

user=User(first_name="Sunny",last_name="Savita")
print(user.full_name)


# ## 166. `model_extra`

# In[ ]:


from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config=ConfigDict(extra="allow")
    name: str

user=User(name="Sunny",company="ABC",city="Bangalore")
print(user.model_extra)


# ## 167. `model_fields_set`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    country: str = "India"
    active: bool = True

user=User(name="Sunny")
print(user.model_fields_set)


# ## 168. JSON Schema vs JSON Serialization

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

print(User.model_json_schema())
user=User(name="Sunny",age=30)
print(user.model_dump_json())


# ## 169. Validator vs Serializer

# In[ ]:


from pydantic import BaseModel, field_serializer, field_validator

class User(BaseModel):
    name: str

    @field_validator("name",mode="before")
    @classmethod
    def clean_input(cls,value):
        return value.strip().title()

    @field_serializer("name")
    def change_output(self,value):
        return value.upper()

user=User(name="   sunny   ")
print(user)
print(user.model_dump())


# ## 170. Custom Type With `Annotated`

# In[ ]:


from typing import Annotated
from pydantic import AfterValidator, BaseModel

def validate_even(value:int)->int:
    if value % 2 != 0:
        raise ValueError("Value must be even")
    return value

EvenNumber=Annotated[int,AfterValidator(validate_even)]

class Data(BaseModel):
    value: EvenNumber


# ## 171. `FailFast`

# In[ ]:


from typing import Annotated
from pydantic import FailFast, TypeAdapter

FastList=Annotated[list[int],FailFast()]
adapter=TypeAdapter(FastList)
adapter.validate_python([1,"bad","also bad"])


# ## 172. `OnErrorOmit`

# In[ ]:


from pydantic import OnErrorOmit, TypeAdapter

adapter=TypeAdapter(list[OnErrorOmit[int]])
result=adapter.validate_python([1,"bad",2,"wrong",3])
print(result)


# ## 173. Performance — Use `model_validate_json()`

# In[ ]:


from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

json_data='{"name":"Sunny","age":30}'
user=User.model_validate_json(json_data)
print(user)


# ## 174. Performance — Reuse `TypeAdapter`

# In[ ]:


from pydantic import TypeAdapter

adapter=TypeAdapter(list[int])

for data in [["1","2"],["3","4"],["5","6"]]:
    print(adapter.validate_python(data))


# ## 175. Performance — Prefer Concrete Collections

# In[ ]:


from pydantic import BaseModel

class Model(BaseModel):
    values: list[int]
    mapping: dict[str,int]


# ## 176. Performance — Use `Any` When Validation Is Not Needed

# In[ ]:


from typing import Any
from pydantic import BaseModel

class RawPayload(BaseModel):
    raw_data: Any


# ## 177. Experimental Pipeline API

# In[ ]:


from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

class Model(BaseModel):
    value: Annotated[int,validate_as(int).ge(0).le(100)]

print(Model(value=50))


# ## 178. Experimental Partial Validation

# In[ ]:


from pydantic import TypeAdapter

adapter=TypeAdapter(list[int])
result=adapter.validate_json('["1","2",',experimental_allow_partial=True)
print(result)


# ## 179. Experimental `MISSING`

# In[ ]:


from pydantic.experimental.missing_sentinel import MISSING
from pydantic import BaseModel

class Configuration(BaseModel):
    timeout: int | None | MISSING = MISSING


# ## 180. Pydantic Extra Types Installation

# ```bash
# pip install pydantic-extra-types
# ```

# ## 181. Color

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.color import Color

class Theme(BaseModel):
    primary_color: Color

theme=Theme(primary_color="#ff0000")
print(theme)


# ## 182. Country

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.country import CountryAlpha2

class Address(BaseModel):
    country: CountryAlpha2

address=Address(country="IN")
print(address)


# ## 183. Currency

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.currency_code import ISO4217

class Payment(BaseModel):
    currency: ISO4217

payment=Payment(currency="INR")


# ## 184. Phone Number

# Optional dependency:
# ```bash
# pip install phonenumbers
# ```

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.phone_numbers import PhoneNumber

class Contact(BaseModel):
    phone: PhoneNumber


# ## 185. Coordinate

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.coordinate import Coordinate

class Location(BaseModel):
    coordinate: Coordinate


# ## 186. MAC Address

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.mac_address import MacAddress

class Device(BaseModel):
    mac: MacAddress


# ## 187. ISBN

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.isbn import ISBN

class Book(BaseModel):
    isbn: ISBN


# ## 188. Timezone Name

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.timezone_name import TimeZoneName

class User(BaseModel):
    timezone: TimeZoneName


# ## 189. Semantic Version

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.semantic_version import SemanticVersion

class Package(BaseModel):
    version: SemanticVersion


# ## 190. ULID

# In[ ]:


from pydantic import BaseModel
from pydantic_extra_types.ulid import ULID

class Event(BaseModel):
    id: ULID


# ## 191. Pydantic + FastAPI Request Model

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

app=FastAPI()

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    age: int

@app.post("/users")
def create_user(user:UserCreate):
    return {"message":"User created","data":user}


# ## 192. FastAPI Response Model

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class UserResponse(BaseModel):
    id: int
    name: str

@app.get("/users/{user_id}",response_model=UserResponse)
def get_user(user_id:int):
    return {"id":user_id,"name":"Sunny"}


# ## 193. Pydantic + LLM Structured Output

# In[ ]:


from pydantic import BaseModel, Field

class CityDetails(BaseModel):
    city: str
    country: str
    population: int
    description: str

structured_model=model.with_structured_output(CityDetails)
result=structured_model.invoke("Give details about Bangalore")
print(result)
print(result.city)


# ## 194. Structured Classification

# In[ ]:


from typing import Literal
from pydantic import BaseModel

class Classification(BaseModel):
    category: Literal["billing","technical","sales"]
    confidence: float

classifier=model.with_structured_output(Classification)
result=classifier.invoke("My payment was deducted twice.")
print(result)


# ## 195. Pydantic + LangChain Tool Input

# In[ ]:


from pydantic import BaseModel, Field
from langchain_core.tools import tool

class SearchInput(BaseModel):
    query: str = Field(min_length=3,description="Search query")
    max_results: int = Field(default=5,ge=1,le=10)

@tool(args_schema=SearchInput)
def search_documents(query:str,max_results:int=5):
    return f"Searching for {query}, max_results={max_results}"

print(search_documents.invoke({"query":"LangGraph","max_results":3}))


# ## 196. Pydantic + Agent Routing

# In[ ]:


from typing import Literal
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    route: Literal["RAG","WEB","DATABASE","FINAL"]
    reason: str = Field(description="Reason for selecting route")

router_model=model.with_structured_output(RouteDecision)
decision=router_model.invoke("What is today's weather?")
print(decision.route)
print(decision.reason)


# ## 197. Pydantic + LangGraph Conditional Routing

# In[ ]:


from typing import Literal
from pydantic import BaseModel
from typing_extensions import TypedDict

class RouteDecision(BaseModel):
    route: Literal["RAG","WEB","LLM"]

class AgentState(TypedDict):
    question: str
    route: str

router_llm=model.with_structured_output(RouteDecision)

def supervisor(state:AgentState):
    decision=router_llm.invoke(state["question"])
    return {"route":decision.route}

def route_next(state:AgentState):
    return state["route"]


# ## 198. Pydantic as LangGraph State

# In[ ]:


from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

class AgentState(BaseModel):
    question: str
    answer: str | None = None
    attempts: int = Field(default=0,ge=0)

def process(state:AgentState):
    return {"answer":"Processed","attempts":state.attempts+1}

builder=StateGraph(AgentState)
builder.add_node("process",process)
builder.add_edge(START,"process")
builder.add_edge("process",END)
app=builder.compile()


# ## 199. TypedDict as Lightweight LangGraph State

# In[ ]:


from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

class AgentState(TypedDict):
    question: str
    answer: str

def process(state:AgentState):
    return {"answer":"Processed"}

builder=StateGraph(AgentState)
builder.add_node("process",process)
builder.add_edge(START,"process")
builder.add_edge("process",END)
app=builder.compile()


# ## 200. AI Application Settings

# In[ ]:


from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class AISettings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env")
    groq_api_key: SecretStr
    model_name: str = "openai/gpt-oss-20b"
    temperature: float = Field(default=0,ge=0,le=2)
    qdrant_url: HttpUrl
    qdrant_api_key: SecretStr


# ## 201. Schema Defining / Validation / Enforcement

# In[ ]:


from pydantic import BaseModel, Field

class Employee(BaseModel):
    # Schema definition
    name: str
    age: int = Field(ge=18)

# Validation + enforcement
employee=Employee(name="Sunny",age=30)
print(employee)

# Invalid
Employee(name="Sunny",age=10)


# ## 202. Complete Real-World User Model

# In[ ]:


from datetime import datetime
from enum import Enum
from typing import Annotated
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, computed_field, field_validator, model_validator

class Role(str,Enum):
    ADMIN="admin"
    USER="user"

UserAge=Annotated[int,Field(ge=18,le=100)]

class User(BaseModel):
    model_config=ConfigDict(extra="forbid",validate_assignment=True,str_strip_whitespace=True)
    id: int
    name: str = Field(min_length=2,max_length=50)
    email: EmailStr
    age: UserAge
    role: Role = Role.USER
    password: SecretStr
    created_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def normalize_name(cls,value:str)->str:
        return value.title()

    @model_validator(mode="after")
    def validate_admin_age(self):
        if self.role == Role.ADMIN and self.age < 21:
            raise ValueError("Admin must be at least 21")
        return self

    @computed_field
    @property
    def profile_label(self)->str:
        return f"{self.name} ({self.role.value})"

user=User(
    id=1,
    name=" sunny savita ",
    email="sunny@example.com",
    age=30,
    password="secret",
    tags=["Python","GenAI"]
)

print(user)
print(user.model_dump())
print(user.model_dump_json())
print(User.model_json_schema())


# ## 203. Full Agentic AI Example — Pydantic + LLM + Routing + Tool Schema

# In[ ]:


from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# 1. ROUTING SCHEMA
class RouteDecision(BaseModel):
    route: Literal["RAG","WEB","CALCULATOR"]
    reason: str

# 2. TOOL INPUT SCHEMA
class CalculatorInput(BaseModel):
    a: float
    b: float
    operation: Literal["add","subtract","multiply","divide"]

# 3. TOOL
@tool(args_schema=CalculatorInput)
def calculator(a:float,b:float,operation:str):
    if operation == "add":
        return a+b
    if operation == "subtract":
        return a-b
    if operation == "multiply":
        return a*b
    if operation == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        return a/b

# 4. STRUCTURED ROUTER
router_llm=model.with_structured_output(RouteDecision)

# 5. DECISION
decision=router_llm.invoke("""
User question:
What is 20 multiplied by 10?
""")
print(decision)

# 6. EXECUTION
if decision.route == "CALCULATOR":
    result=calculator.invoke({"a":20,"b":10,"operation":"multiply"})
    print(result)


# ## 204. Pydantic Mental Model in Code

# In[ ]:


from pydantic import BaseModel, Field, ValidationError

class User(BaseModel):
    name: str
    age: int = Field(ge=18)

raw_data={"name":"Sunny","age":"30"}

try:
    # RAW DATA
    #     ↓
    # PYDANTIC SCHEMA
    #     ↓
    # TYPE CONVERSION
    #     ↓
    # VALIDATION
    #     ↓
    # PYTHON OBJECT
    user=User.model_validate(raw_data)
    print(user)
    print(user.model_dump())
    print(user.model_dump_json())
except ValidationError as error:
    print(error.errors())

