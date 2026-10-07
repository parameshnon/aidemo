import os
from pathlib import Path

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_groq import ChatGroq


MODEL_NAME = "openai/gpt-oss-120b"
PROJECT_DIRECTORY = Path(__file__).resolve().parent
SKILLS_DIRECTORY = PROJECT_DIRECTORY / "skills"
BUILTIN_TOOLS = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "task",
    }
)


def build_agent(model: ChatGroq):
    """Build a Deep Agent that loads skills from the project's skills directory."""
    register_harness_profile(
        "groq",
        HarnessProfile(
            excluded_tools=BUILTIN_TOOLS,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    return create_deep_agent(
        model=model,
        tools=[],
        backend=FilesystemBackend(
            root_dir=PROJECT_DIRECTORY,
            virtual_mode=True,
        ),
        skills=["/skills/"],
        system_prompt=(
            "You are a helpful writing assistant. Check the available skills and "
            "follow a skill when it applies to the user's request. Ask a concise "
            "clarifying question if the task is ambiguous. Do not claim you saved "
            "or changed a file unless a tool actually did so."
        ),
    )


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file."
        )
    if not SKILLS_DIRECTORY.is_dir():
        raise FileNotFoundError(f"Skills directory does not exist: {SKILLS_DIRECTORY}")

    agent = build_agent(ChatGroq(model=MODEL_NAME, temperature=0))
    config = {"configurable": {"thread_id": "skills-sample-session"}}
    print("Deep Agent skills sample. Ask for writing help; type 'exit' to quit.")
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.casefold() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
        )
        print(f"Assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()