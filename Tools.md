# Tools

## What a tool is

A tool is an application function made available to an agent through a schema.
The model can request a call with arguments; the framework validates and routes
the request to the Python function, then returns the function's result to the
model. A tool should do bounded work and return a clear result.

## Defining a LangChain tool

The repository examples use the `@tool` decorator:

```python
from langchain.tools import tool

@tool
def add(first_number: float, second_number: float) -> float:
    """Add two numbers."""
    return first_number + second_number
```

The function name, type annotations, and docstring help form the tool schema
that the model sees. Keep the docstring explicit about the operation and
constraints. Validate inputs inside the function even when a schema exists.

## Register tools with an agent

```python
agent = create_agent(
    model=model,
    tools=[add],
    system_prompt="Use the add tool for arithmetic.",
)
```

The model can choose whether a tool is needed. For data that must be exact,
such as prices, calculations, or booking status, the tool result should be the
source of truth; instruct the model not to invent an answer.

## Tool execution in LangGraph

In the ReAct graph example, `ToolNode` executes tool calls and
`tools_condition` routes the model response either back to the tools or to the
end of the graph:

```text
START -> assistant --tool call--> tools -> assistant
                      \--no tool call------> END
```

This makes the agent loop explicit and allows the application to customize
state, routing, validation, and interruption behavior.

## MCP tools

MCP tools expose operations across an MCP client/server boundary. In
`langchain-poc\langchain-poc-mcp.py`, the same arithmetic helpers back both
LangChain tools and MCP tools named `add`, `subtract`, `multiply`, and `divide`.
Sharing the implementation keeps validation and results consistent across
interfaces.

## Tool safety

- Validate ranges, formats, and domain rules in the function.
- Reject unsafe or unsupported operations explicitly.
- Keep side effects narrow and observable.
- Require human approval for actions that send, delete, purchase, or publish.
- Avoid exposing secrets in tool arguments, logs, or outputs.
- Use idempotency and confirmation patterns for operations that may be retried.
- Do not trust model-generated arguments simply because they passed schema
  validation.
