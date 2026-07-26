# AGENTS.md

This file describes the project's stack, conventions, and code standards for AI agents working in this codebase.

---

## Stack

- **API framework:** FastAPI + Dishka
- **ORM / Database:** TortoiseORM + PostgreSQL
- **AI layer:** Pydantic AI

---

## Prerequisites

- Python **3.14.x** is required. Do not assume an older version.

---

## Programming Style

The codebase follows a **blend of Object-Oriented and Functional Programming**. When writing code:

- Use classes to model domain entities, services, and components with shared state or behaviour.
- Use pure functions for logic that transforms data without side effects.
- Do not default to one paradigm exclusively — apply whichever fits the context.

---

## Code Standards

These rules are **mandatory**. Apply them to every file you create or modify.

### Type Annotations

Always annotate types wherever possible — function signatures, return types, and local variables.

```python
# Good
def get_user(user_id: int) -> User:
    result: User | None = repository.find(user_id)
    ...

# Bad
def get_user(user_id):
    result = repository.find(user_id)
    ...
```

### Docstrings

Every function and class must have a docstring written in **reST style**. Include:

- A short summary on the first line.
- `:param <n>:` for each parameter.
- `:returns:` describing the return value (if any).
- `:raises <ExceptionType>:` for each exception the function may raise (if any).

**Endpoint functions** (those decorated with `@router.get`, `@router.post`, `@router.put`, `@router.delete`, etc.) must have a docstring with **only the summary line** — omit `:param`, `:returns`, and `:raises`.

```python
def create_order(user_id: int, items: list[Item]) -> Order:
    """
    Create a new order for the given user.
    :param user_id: The ID of the user placing the order.
    :param items: The list of items to include in the order.
    :returns: The newly created Order instance.
    :raises ValueError: If the items list is empty.
    """
    ...
```

### Quotes

Prefer **single quotes** (`'`) for strings over double quotes (`"`).

```python
# Good
name: str = 'Alice'

# Bad
name: str = "Alice"
```

### FastAPI — Request & Response Models

Every endpoint must have its own dedicated Pydantic models for the request body and the response. Do not reuse ORM models or generic dicts as endpoint contracts.

```python
class CreateUserRequest(BaseModel):
    username: Annotated[str, Field(min_length=3)]

class CreateUserResponse(BaseModel):
    id: Annotated[int, Field()]
    username: Annotated[str, Field()]

@router.post('/users', response_model=CreateUserResponse)
async def create_user(body: CreateUserRequest) -> CreateUserResponse:
    ...
```

### Pydantic Models

Prefer Pydantic models over dataclasses. Annotate all fields using the `Annotated[T, Field()]` pattern for models that you either create or edit.

```python
# Good
class Product(BaseModel):
    id: Annotated[int, Field()]
    name: Annotated[str, Field(min_length=1)]
    price: Annotated[float, Field(gt=0)]

# Bad
@dataclass
class Product:
    id: int
    name: str
    price: float
```

### Enums for Literal Types

When a field or variable can take **3 or more** named literal values, define an `Enum` type rather than using inline string literals.

```python
# Good
class OrderStatus(str, Enum):
    pending = 'pending'
    confirmed = 'confirmed'
    cancelled = 'cancelled'

class Order(BaseModel):
    status: Annotated[OrderStatus, Field()]

# Bad
class Order(BaseModel):
    status: Annotated[Literal['pending', 'confirmed', 'cancelled'], Field()]
```

### Guard Clauses

Prefer early returns to reduce nesting and clarify intent.

```python
# Good
def process(order: Order) -> Result:
    if order.status == OrderStatus.cancelled:
        raise ValueError('Cannot process a cancelled order.')
    if not order.items:
        raise ValueError('Order has no items.')
    return execute(order)

# Bad
def process(order: Order) -> Result:
    if order.status != OrderStatus.cancelled:
        if order.items:
            return execute(order)
```

### No Side Effects

Functions should return values rather than mutating external or internal state silently. Avoid functions that set state and return `None` as their primary purpose.

```python
# Good
def with_status(order: Order, status: OrderStatus) -> Order:
    return order.model_copy(update={'status': status})

# Bad
def set_status(order: Order, status: OrderStatus) -> None:
    order.status = status
```

### Circular Dependencies

If you ever come across a circular dependency, use `from __future__ import annotations` to define annotations, along with a `TYPE_CHECKING` check, and the imports at the end of the file with a call to `model_rebuild()`.

```python
# Good
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Annotated, TYPE_CHECKING

if TYPE_CHECKING:
    from database.enums import UserPermission


class AdminUser(BaseModel):
    name: Annotated[str, Field()]
    permissions: Annotated[UserPermission, Field()]


from database.enums import UserPermission
AdminUser.model_rebuild()

# Bad
from pydantic import BaseModel, Field
from typing import Annotated, TYPE_CHECKING

if TYPE_CHECKING:
    from database.enums import UserPermission


class AdminUser(BaseModel):
    name: Annotated[str, Field()]
    permissions: Annotated['UserPermission', Field()]
```

### Prefer PEP 585-compliant over PEP 484-compliant annotations

PEP 484-compliant type annotations are being dropped on Python 3.17, so let's keep the project clear of them for a future smoother transition.

If no alternative is available, import from the `collections.abc` module.

```py
# Good
names: list[str] = []

# Acceptable if there is no other alternative
from collections.abc import Sequence

names: Sequence[str] = []

# Bad
from typing import List

names: List[str] = []
```

### Preferred Type References

Prefer direct imports to dotted references for **type annotations** and **object instantiation**. Module-level **function calls** (e.g., `logging.info()`, `os.path.join()`) may keep the dotted form.

```python
# Good — direct import for type annotation and instantiation
from asyncio import Lock
from httpx import AsyncClient, HTTPStatusError

lock: Lock = Lock()
client: AsyncClient = AsyncClient()

# Good — dotted form is acceptable for module-level function calls
import logging
import os.path

logging.info('Starting...')
path = os.path.join('a', 'b')

# Bad — dotted form used for type annotation or instantiation
import asyncio
import httpx

lock: asyncio.Lock = asyncio.Lock()
client: httpx.AsyncClient = httpx.AsyncClient()
```

### Null Checking

Prefer `is` and `is not` to check for null, as it is more expressive.

```python
text: str | None = maybe_get_text()

# Good
if text is not None:
    ...

# Bad
if text:
    ...
```

### Post-Task

Once you're done writing your changes, execute the following command:

```bash
mypy . && basedpyright .
```

Fix any error returned by the linters, even if they are not related to the task at hand.

## Common Pitfalls to Avoid

- Do not skip type annotations, even for simple local variables.
- Do not write functions that return `None` as a way to communicate results.
- Do not share request/response models across multiple endpoints if their shapes diverge.


## General Agent Guidance

- Do not introduce new dependencies without being asked.
- Do not change the database schema without being asked.
- When adding a new endpoint, always create accompanying request/response Pydantic models.
- When in doubt about naming or structure, follow the patterns already established in the codebase.
