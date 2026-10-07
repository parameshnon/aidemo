# Observation and Observability

“Observation” in agent workflows usually means inspecting what happened during
execution. **Observability** is the broader practice of understanding,
debugging, and evaluating an application from its outputs, logs, traces, and
metrics. LangChain and LangGraph applications benefit from both application
logging and end-to-end traces.

## What to observe

Useful information can include:

- User request and final response, subject to privacy and retention rules.
- Model calls, latency, errors, and token usage.
- Tool name, validated inputs, result status, and duration.
- Graph node transitions, state updates, and interrupt/resume events.
- Agent and subagent handoffs.
- Evaluation results and regressions across test cases.

Avoid recording secrets, access tokens, or unnecessary personal data. A trace
can contain prompt and tool data, so treat it as sensitive.

## LangSmith tracing in the repository

`langchain-poc\langchain-poc-langsmith.py` sets:

```python
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "langchain-groq-product-calculator")
```

It decorates calculator and product lookup functions with `@traceable`, naming
them as tool runs. With a configured LangSmith API key and tracing enabled, the
framework can send run traces to the configured project. Without the API key,
that sample prints a notice; local application logs and tests remain useful.

Environment values such as `LANGSMITH_API_KEY` should be kept out of source
control and logs.

## Application logs

The middleware examples use Python's `logging` module for validation and
completion events. They deliberately avoid logging prompt content and access
tokens. Logs are useful for simple operational signals but usually do not show
the full causal path through a multi-step graph.

## LangGraph state inspection

The LangGraph samples call `invoke` and inspect values in the returned state,
such as the final `response` or the last message. During development, this
provides a direct way to confirm state transitions and outputs. For more
detailed observation, applications can use graph streaming and checkpoint
inspection supported by their installed LangGraph version.

The ReAct graph illustrates the reasoning/action/observation cycle at a
workflow level:

```text
assistant requests a tool -> ToolNode executes it
                           -> tool result returns to assistant
assistant returns final answer when no more tool is needed
```

The tool result is the observation available to the model; it is not a
guarantee that the final response uses that result correctly. Test the whole
cycle and verify final answers against the tool outputs.

## Testing and evaluation

- Unit-test tools without an LLM for deterministic behavior and error cases.
- Use mocked model responses for routing, retries, and interruption paths.
- Test graph outputs and state transitions, not only that `invoke` completed.
- Evaluate representative prompts and edge cases after prompt/model changes.
- Keep latency and cost budgets for loops and subagent delegation.
- Separate test/development traces from production projects.

Observability helps explain system behavior; it does not itself enforce
authorization, correctness, or privacy. Those controls must be designed into
the application.
