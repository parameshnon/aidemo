# LangChain and LangGraph Notes

These notes explain the main concepts used in this repository's Python samples.
They are intended as learning material, not as production deployment guidance.

| Topic | Notes |
| --- | --- |
| MCP | [MCP.md](MCP.md) |
| Skills | [Skills.md](Skills.md) |
| Agents | [Agents.md](Agents.md) |
| Deep Agents | [Deep-Agents.md](Deep-Agents.md) |
| Tools | [Tools.md](Tools.md) |
| Human in the loop | [Human-in-the-Loop.md](Human-in-the-Loop.md) |
| Memory | [Memory.md](Memory.md) |
| Backends | [Backends.md](Backends.md) |
| Middleware | [Middleware.md](Middleware.md) |
| Retrieval-augmented generation (RAG) | [RAG.md](RAG.md) |
| Observation and observability | [Observability.md](Observability.md) |

The examples use Python, `uv`, LangChain, LangGraph, and (for model-backed
examples) Groq. Start commands below assume PowerShell is open at the repository
root. Configure `GROQ_API_KEY` in the environment or a local `.env` file when a
sample calls Groq; calculator MCP client/server examples do not require it.
