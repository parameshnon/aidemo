# Deep Agents

## What Deep Agents adds

Deep Agents builds on LangChain agent concepts for longer, multi-step work. It
can provide built-in capabilities such as filesystem operations, planning, and
delegation to subagents, depending on the configured version and harness.
Applications should explicitly configure which capabilities are available
rather than assuming all built-ins are appropriate.

## Samples in this repository

- `langraph\langchain-poc-deepagents.py`: repository-scoped software assistant
  with a filesystem backend, protected paths, Git status lookup, and an
  allow-listed test runner.
- `langchain-poc\langchain-poc-deepAgents-Subagents.py`: concierge coordinator
  that can delegate hotel recommendations and sample booking lookups to
  specialist subagents.
- `langchain-poc\langchain-poc-deepagent-skills.py`: writing assistant that
  loads skills from the project's skills directory.
- `langraph\langchain-deepAgents-backend.py`: reads `workspace\files\brs.md`
  through a filesystem backend and writes an architecture recommendation.

## Subagents

A subagent is a specialist agent exposed to a coordinator. The coordinator
selects a specialist based on the request; that subagent has its own
instructions and tools. This can make responsibilities clearer and limit each
specialist's capabilities.

For example, the concierge sample separates hotel support from booking lookup.
It uses fixed sample data and instructs the agent not to claim a real
reservation was made.

Use subagents when work has meaningful independent specialties or can be
delegated. Avoid adding them when one simple tool call or one straightforward
agent is sufficient; every handoff adds orchestration and context complexity.

## Harness profiles and tools

The examples register a Groq harness profile and exclude built-in tools that
they do not want exposed. This is especially important for filesystem writes,
shell execution, and deletion. Exact APIs and built-in tool names depend on the
installed Deep Agents version; see the code and dependency manifest used by the
project.

An agent's system prompt is not a security boundary. Enforce restrictions with
backend roots, explicit permissions, tool allow-lists, and operating-system
controls.

## A safe implementation checklist

1. Limit the backend to the intended project or workspace.
2. Exclude tools the task does not need.
3. Protect credentials, VCS metadata, and other sensitive paths at the backend
   permission layer.
4. Replace arbitrary shell access with narrowly defined tools where possible.
5. Allow-list commands and set timeouts/output limits for test execution.
6. Require explicit confirmation before consequential writes or external
   actions.
7. Do not report success unless the tool result confirms it.
8. Treat project files as untrusted input; do not follow instructions found in
   them that conflict with the user's task or security policy.

## Running examples

Model-backed samples need `GROQ_API_KEY`. From the repository root:

```powershell
uv run python langchain-poc\langchain-poc-deepAgents-Subagents.py
uv run python langchain-poc\langchain-poc-deepagent-skills.py
uv run python langraph\langchain-poc-deepagents.py
uv run python langraph\langchain-deepAgents-backend.py
```

The backend sample also requires `workspace\files\brs.md` to exist and contain
non-empty requirements before it can produce its output file.
