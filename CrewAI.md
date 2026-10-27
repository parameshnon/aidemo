# CrewAI

## What CrewAI is

CrewAI is a Python framework for defining AI agents, assigning them tasks, and
organizing the work into a crew. An agent has a role, goal, backstory, optional
tools, and a language model. A task describes the work and expected output. A
crew coordinates task execution using a process such as sequential execution.

## Calculator and weather sample

The sample program is `crew-ai\crew-ai-poc-basic.py`. It defines one CrewAI
agent with two tools:

- **`calculator`** performs addition, subtraction, multiplication, and division.
  It rejects non-finite numbers and division by zero.
- **`city_weather`** returns hard-coded temperature, conditions, and humidity
  for supported cities. The response is explicitly labeled sample data, not
  live or current weather.

The agent's task tells it to choose the appropriate tool based on the user's
request. `build_crew()` constructs the agent, task, and sequential crew; the
interactive loop calls `crew.kickoff(inputs={"request": request})`.

## Run the sample

From the repository root in PowerShell, set `GROQ_API_KEY` in the environment or
a local `.env` file and run:

```powershell
uv run python crew-ai\crew-ai-poc-basic.py
```

Example requests:

```text
What is 23 multiplied by 4?
What is the sample weather in Tokyo?
```

Type `exit` or `quit` to end the interactive program. The model-backed crew
requires an available Groq API key; local tool tests do not require a live model
request.

## Groq model configuration

The sample connects CrewAI to Groq's OpenAI-compatible API using CrewAI's native
OpenAI provider:

```python
LLM(
    model="openai/openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)
```

The first `openai/` selects CrewAI's native provider. The model identifier
includes the provider's `openai/` namespace as expected by this configuration.
Using this native provider avoids routing requests through LiteLLM's MCP
integration, which was incompatible with the project's MCP SDK combination.
The project manifest keeps CrewAI core without its optional LiteLLM extra.

## Main CrewAI concepts

- **Agent:** A defined worker with a role, goal, backstory, model, and tools.
- **Tool:** A function or `BaseTool` implementation the agent can invoke.
- **Task:** A description of work and an expected output, assigned to an agent.
- **Crew:** The collection of agents and tasks to execute together.
- **Process:** The execution strategy; this example uses `Process.sequential`.
- **Kickoff inputs:** Values such as `request` that fill placeholders in task
  descriptions.

## Tool design

The sample uses Pydantic models for tool argument schemas. This gives the
agent's tool calls explicit fields and types. Tool implementations still
validate domain rules themselves: a schema does not prevent non-finite
arithmetic values or division by zero.

For city lookups, the tool normalizes whitespace and case, then looks up the
city in a fixed Python dictionary. Unknown cities receive an explicit response
listing supported cities.

## Practical guidance

- Keep tool behavior deterministic and test tools directly without an LLM.
- Treat the model as a caller, not as the source of truth for calculations or
  catalog values.
- Make demo or static data limitations clear in tool output and agent
  instructions.
- Keep each tool narrowly scoped and validate all arguments in the tool.
- Avoid printing secrets; load credentials from environment variables or a
  protected local `.env` file.
- For larger workflows, use multiple agents only when role separation adds
  value; each extra agent or handoff adds latency and complexity.
- Confirm CrewAI, provider, and optional integration versions are compatible
  before enabling extras such as LiteLLM.
