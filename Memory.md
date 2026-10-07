# Memory

## Meaning of memory in agents

Agent memory means information an application keeps available across model
calls. In LangGraph, the conversation and workflow values live in graph state.
A checkpointer can save snapshots of that state so later invocations with the
same thread can continue the conversation.

Memory is not the same as the model remembering earlier requests on its own.
The application must pass state or retrieve stored information.

## Short-term, thread-scoped memory

The examples use `InMemorySaver`:

```python
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "customer-session-123"}}
result = agent.invoke(input_state, config=config)
```

The thread ID selects a conversation/checkpoint stream. Reuse it for follow-up
turns in the same conversation; choose another ID for a new conversation.
Examples include `langchain-poc\langchain-agent-poc.py`,
`langchain-poc\langchain-poc-langsmith.py`,
`langraph\langgraph-poc-multiagent.py`, and `langchain-poc-HINL.py`.

`InMemorySaver` is for demonstrations and process-local use. Its contents are
not durable across process restarts and are not a shared persistent database.
Use an appropriate durable checkpointer and retention strategy for production.

## Graph state versus conversation history

Graph state can contain more than messages. The concierge workflow defines
fields such as `query`, `intent`, `location`, `preference`, and `response`.
Nodes return partial updates, and the graph passes the resulting state through
the workflow. The `add_messages` reducer controls how message updates are
combined in that example.

Keep state minimal and typed. Avoid persisting credentials or unnecessary
personal data in conversation history.

## Long-term memory

Long-term memory means information intentionally retained across separate
threads or sessions, typically through a database or a store keyed by user or
domain. It is different from the in-memory checkpointer used in these examples.
Before adding long-term memory, decide what may be stored, how it is scoped,
how it can be corrected/deleted, and how it is protected from cross-user access.

## Memory is not model truth

Memory may be stale, incomplete, or attached to the wrong user/thread. Verify
important facts against authoritative data sources and avoid letting old
messages override current user instructions or application policy.
