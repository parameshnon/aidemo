# Agents

## What an agent does

An agent combines a language model with instructions and optional tools. The
model interprets the request and can choose to call a tool; the application
executes the selected tool, returns its result to the model, and then produces a
response. Tool selection is model-driven, so the model should not be treated as
the authority for factual calculations or external actions.

## LangChain agent sample

`langchain-poc\langchain-agent-poc.py` uses LangChain's `create_agent` with a
Groq chat model and two tools:

```python
agent = create_agent(
    model=ChatGroq(model="openai/gpt-oss-120b", temperature=0),
    tools=[calculate, get_city_temperature],
    checkpointer=InMemorySaver(),
    system_prompt="Use calculator and sample-temperature tools for those facts.",
)
```

The loop invokes the agent with a user message and prints its final message.
`GROQ_API_KEY` is required by model-backed samples. The temperatures in this
example come from a fixed local dictionary; they are not live weather data.

## LangChain agent versus a custom LangGraph workflow

Use a LangChain agent when the model should decide which tools to use and in
what order for an open-ended task.

Use a custom LangGraph workflow when the application needs explicit steps,
typed state, branching, or deterministic routing. For example,
`langraph\langgraph-poc-multiagent.py` uses a structured request model to route
concierge requests to support, booking, refund, or clarification nodes.

These patterns can be combined: a graph can orchestrate specialist agents,
while each specialist uses its own tools.

## Core pieces

- **Model**: generates responses and, when supported, tool-call requests.
- **System prompt**: sets the agent's role, tool-use rules, and boundaries.
- **Tools**: typed application functions the model may request.
- **State**: messages and any other values the workflow needs to carry.
- **Checkpointer**: saves graph state between invocations for a conversation
  thread or pause/resume flow.
- **Runtime configuration**: commonly includes a thread ID and, where needed,
  caller context.

## Threaded invocation

The samples pass the conversation as messages and provide a stable thread ID:

```python
config = {"configurable": {"thread_id": "cli-session"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 12 times 7?"}]},
    config=config,
)
```

When a checkpointer is configured, a stable ID lets later invocations continue
that thread. Use a different ID for an independent conversation. Do not treat a
thread ID as an authorization credential.

## Agent design guidelines

- Give tools clear names, typed arguments, and specific docstrings.
- Keep authoritative calculations, catalog lookups, and validation in code.
- Tell the model when data is demo/static so it does not present it as live.
- Handle missing information with clarification rather than invented values.
- Bound tool side effects and make consequential actions require explicit
  approval.
- Check the result's shape and errors before displaying or acting on it.
