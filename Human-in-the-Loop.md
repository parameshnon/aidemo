# Human in the Loop

## Why add a human review step

Human-in-the-loop (HITL) pauses an agent before or during an action so a person
can inspect it, approve it, edit it, or reject it. It is useful for external or
otherwise consequential effects where the model should not act autonomously.

HITL does not replace validation or authorization. The tool must still enforce
its input rules, and the application must still decide who is allowed to
approve.

## Repository example

`langchain-poc-HINL.py` configures `HumanInTheLoopMiddleware` for the
`send_message` tool and permits `approve`, `edit`, and `reject` decisions. The
agent uses `InMemorySaver` so LangGraph can persist the paused state while
waiting for a reviewer.

The sample tool only creates a demo delivery ID and explicitly does not deliver
a real message.

## Pause, review, resume

The basic flow is:

1. Invoke the agent with a request and a thread ID.
2. If the result contains an interrupt, display the proposed tool name and
   arguments to the reviewer.
3. Collect a valid decision.
4. Resume the same graph thread with `Command(resume=...)`.
5. Report the tool's result, or report that the action was rejected.

Example decision values used by the sample:

```python
# Approve the proposed tool call.
{"decisions": [{"type": "approve"}]}

# Reject the action.
{"decisions": [{"type": "reject", "message": "Reviewer rejected the action."}]}

# Edit the arguments before approval.
{
    "decisions": [{
        "type": "edit",
        "edited_action": {
            "name": "send_message",
            "args": {"recipient": "person@example.com", "message": "Approved text"},
        },
    }]
}
```

Resume the interrupted execution using the same configuration:

```python
result = agent.invoke(
    Command(resume=decision_payload),
    config={"configurable": {"thread_id": "human-review-demo"}},
)
```

The checkpointer and thread ID connect the resume operation to the paused
conversation.

## Production checklist

- Show reviewers exactly what will happen, including all significant
  parameters.
- Bind approval to an authenticated reviewer and the specific pending action.
- Revalidate edited arguments before execution.
- Handle timeouts, duplicate submissions, rejection, and stale approvals.
- Use a durable checkpointer if the process may restart while waiting.
- Keep audit records of the decision without logging unnecessary sensitive
  content.
- Never interpret a missing or invalid decision as approval.
