# Model Context Protocol (MCP)

## What MCP is

The Model Context Protocol is a standard way for an AI application (the client)
to discover and use capabilities provided by another process or service (the
server). It separates the application that wants to use a capability from the
program that implements it.

MCP commonly exposes three kinds of capabilities:

- **Tools** are callable operations, such as adding two numbers.
- **Resources** are readable context addressed by a URI, such as a calculator
  guide at `calculator://guide`.
- **Prompts** are reusable message templates, optionally parameterized by
  arguments such as a calculation question.

Tools, resources, and prompts are distinct protocol features. A resource read
does not execute a tool, and fetching a prompt renders a template but does not
automatically send it to a model.

## Calculator server in this repository

`langchain-poc\langchain-poc-mcp.py` registers:

| Capability | Name or URI | Purpose |
| --- | --- | --- |
| Tool | `add` | Add two finite numbers |
| Tool | `subtract` | Subtract the second number from the first |
| Tool | `multiply` | Multiply two finite numbers |
| Tool | `divide` | Divide the first by the second; zero division is rejected |
| Resource | `calculator://guide` | Read-only Markdown documentation |
| Prompt | `calculator_request` | Template that takes a required `question` |

The calculator functions validate input values and share the same arithmetic
helpers across the MCP and LangChain tool interfaces.

## Transports

- **stdio**: An MCP client starts the server as a child process and exchanges
  protocol messages through standard input and output. This is the default
  transport and is useful for local desktop clients.
- **Streamable HTTP**: The server listens for HTTP MCP connections. This sample
  binds to `127.0.0.1`, so it is intended for local access only.

## Run and connect

From the repository root in PowerShell:

```powershell
# Run a stdio server directly (typically an MCP client launches this).
uv run python langchain-poc\langchain-poc-mcp.py

# Or run the server on local Streamable HTTP.
uv run python langchain-poc\langchain-poc-mcp.py --transport streamable-http
```

The companion client, `langchain-poc\langchain-poc-mcpclient.py`, uses stdio
and launches the server automatically by default. To connect to an HTTP server,
leave the server running in one terminal and run the client in another:

```powershell
uv run python langchain-poc\langchain-poc-mcpclient.py --transport streamable-http
```

## Access resources and prompts

The client exposes the protocol operations directly:

```powershell
uv run python langchain-poc\langchain-poc-mcpclient.py --list-resources
uv run python langchain-poc\langchain-poc-mcpclient.py --read-resource calculator://guide
uv run python langchain-poc\langchain-poc-mcpclient.py --list-prompts
uv run python langchain-poc\langchain-poc-mcpclient.py --get-prompt calculator_request --prompt-arg "question=What is 12 divided by 3?"
```

At the MCP API level the matching client calls are:

```python
resources = await session.list_resources()
contents = await session.read_resource("calculator://guide")

prompts = await session.list_prompts()
rendered = await session.get_prompt(
    "calculator_request",
    {"question": "What is 12 divided by 3?"},
)
```

`get_prompt` returns messages for the caller to use. The client or host
application decides whether and how to pass those messages to an LLM.

## MCP is not a browser UI

The Streamable HTTP endpoint is a protocol endpoint, not a human-facing
calculator page. Use an MCP client or an MCP inspector to list and call its
capabilities. Opening `/mcp` in a normal browser does not create a web UI.

## Practical considerations

- Keep stdio output reserved for protocol messages; send diagnostics to stderr.
- Validate tool input in the server; model-generated arguments are not trusted.
- Do not expose a local development server to an untrusted network without
  appropriate authentication and transport security.
- Pin and test MCP SDK versions because SDK APIs and supported transports can
  change across major releases.
