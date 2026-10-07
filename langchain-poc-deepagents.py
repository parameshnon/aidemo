"""Deep Agent that analyzes and updates a local GitHub project checkout."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver


MODEL_NAME = "openai/gpt-oss-120b"
TIMEOUT_SECONDS = 180
MAX_TEST_OUTPUT_CHARS = 16_000
SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
DEEP_AGENT_UNAVAILABLE_TOOLS = frozenset({"execute", "task", "delete"})


def _redact_environment_secrets(output: str) -> str:
    for name, value in os.environ.items():
        if (
            any(marker in name.upper() for marker in SENSITIVE_ENV_MARKERS)
            and len(value) >= 6
        ):
            output = output.replace(value, "[REDACTED]")
    return output


def _test_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in SENSITIVE_ENV_MARKERS)
    }


def _format_command_result(
    command: list[str],
    result: subprocess.CompletedProcess[str],
) -> str:
    output = (
        f"Command: {' '.join(command)}\n"
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout or '(empty)'}\n"
        f"STDERR:\n{result.stderr or '(empty)'}"
    )
    output = _redact_environment_secrets(output)
    if len(output) > MAX_TEST_OUTPUT_CHARS:
        output = output[:MAX_TEST_OUTPUT_CHARS] + "\n[Output truncated]"
    return output


def _project_tools(project_root: Path):
    project_python = next(
        (
            candidate
            for candidate in (
                project_root / ".venv" / "Scripts" / "python.exe",
                project_root / ".venv" / "bin" / "python",
                project_root / "venv" / "Scripts" / "python.exe",
                project_root / "venv" / "bin" / "python",
            )
            if candidate.is_file()
        ),
        Path(sys.executable),
    )

    @tool
    def get_git_status() -> str:
        """Read the current Git branch and working-tree status without changing files."""
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            shell=False,
        )
        if result.returncode:
            detail = _redact_environment_secrets(result.stderr.strip())
            raise RuntimeError(f"git status failed: {detail}")
        return result.stdout.strip() or "Git working tree is clean."

    @tool
    def run_project_tests(
        test_suite: Literal[
            "pytest",
            "unittest",
            "npm",
            "pnpm",
            "yarn",
            "go",
            "cargo",
            "dotnet",
            "maven",
            "gradle",
        ],
    ) -> str:
        """Run a supported project test suite using a fixed command in the repository."""
        command_map = {
        "pytest": [str(project_python), "-m", "pytest"],
        "unittest": [str(project_python), "-m", "unittest", "discover"],
            "npm": ["npm", "test"],
            "pnpm": ["pnpm", "test"],
            "yarn": ["yarn", "test"],
            "go": ["go", "test", "./..."],
            "cargo": ["cargo", "test"],
            "dotnet": ["dotnet", "test"],
            "maven": ["mvn", "test"],
            "gradle": ["gradle", "test"],
        }
        command = command_map[test_suite]
        executable = command[0]
        if executable != str(project_python):
            resolved_executable = shutil.which(executable)
            if resolved_executable is None:
                return (
                    f"Cannot run {test_suite} tests: {executable!r} is not installed "
                    "or not available on PATH."
                )
            command[0] = resolved_executable
            if os.name == "nt" and Path(resolved_executable).suffix.casefold() in {
                ".bat",
                ".cmd",
            }:
                command = [
                    os.environ.get("COMSPEC", "cmd.exe"),
                    "/d",
                    "/s",
                    "/c",
                    subprocess.list2cmdline(command),
                ]

        try:
            result = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
                shell=False,
                env=_test_environment(),
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            partial_output = _redact_environment_secrets(f"{stdout}\n{stderr}")
            return (
                f"{test_suite} tests exceeded the {TIMEOUT_SECONDS}-second limit.\n"
                f"Partial output:\n{partial_output[:MAX_TEST_OUTPUT_CHARS]}"
            )
        except OSError as error:
            raise RuntimeError(f"Could not start the {test_suite} test runner.") from error
        return _format_command_result(command, result)

    return [get_git_status, run_project_tests]


def build_project_agent(project_root: Path, model: ChatGroq):
    """Create a repository-scoped Deep Agent with safe test and Git tools."""
    project_root = project_root.resolve(strict=True)
    if not project_root.is_dir():
        raise ValueError(f"Project path is not a directory: {project_root}")
    if not (project_root / ".git").exists():
        raise ValueError(f"Project path is not a Git checkout: {project_root}")

    register_harness_profile(
        "groq",
        HarnessProfile(
            excluded_tools=DEEP_AGENT_UNAVAILABLE_TOOLS,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    protected_paths = [
        "/.git/**",
        "/**/.git/**",
        "/.env",
        "/**/.env",
        "/.env.local",
        "/**/.env.local",
        "/.env.development",
        "/**/.env.development",
        "/.env.production",
        "/**/.env.production",
        "/.env.test",
        "/**/.env.test",
        "/.env.staging",
        "/**/.env.staging",
        "/.env.development.local",
        "/**/.env.development.local",
        "/.env.production.local",
        "/**/.env.production.local",
        "/.env.test.local",
        "/**/.env.test.local",
        "/.npmrc",
        "/**/.npmrc",
        "/.pypirc",
        "/**/.pypirc",
        "/.netrc",
        "/**/.netrc",
        "/.ssh/**",
        "/**/.ssh/**",
        "/.aws/**",
        "/**/.aws/**",
        "/.docker/config.json",
        "/**/.docker/config.json",
        "/**/id_rsa*",
        "/**/id_ed25519*",
        "/**/*.pem",
        "/**/*.key",
        "/**/*.p12",
        "/**/*.pfx",
        "/**/*.keystore",
    ]
    return create_deep_agent(
        model=model,
        tools=_project_tools(project_root),
        backend=FilesystemBackend(root_dir=project_root, virtual_mode=True),
        permissions=[
            FilesystemPermission(
                operations=["read", "write"],
                paths=protected_paths,
                mode="deny",
            )
        ],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You are a careful software engineering agent working only in the "
            "user-selected local Git checkout. Complete this workflow in order: "
            "1) inspect the project structure, README, manifests, source, tests, and "
            "Git status; 2) summarize the architecture and identify the framework; "
            "3) implement authentication using the project's established patterns "
            "and sensible secure defaults; if a consequential requirement such as "
            "identity provider, account lifecycle, or session model cannot be safely "
            "inferred, ask the user before choosing; 4) add or update focused tests; "
            "5) inspect Git status again and avoid overwriting unrelated pre-existing "
            "changes; 6) run the matching existing test suite using "
            "run_project_tests, fix failures caused by your changes, and rerun; "
            "7) update the existing README with authentication setup and usage. "
            "Do not read or modify .git, .env, private keys, or credential files. "
            "Do not install packages, execute arbitrary shell commands, delete files, "
            "commit changes, or claim tests passed unless the test tool confirms it. "
            "Do not expose secrets in your response. If a test runner is unavailable "
            "or a task is blocked, explain that clearly. Finish with a concise report "
            "of the architecture, changes, tests run, and remaining limitations."
        ),
    )


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it in the environment or the current "
            "project's .env file before starting this agent."
        )

    raw_project_path = input("Path to the local GitHub project checkout: ").strip()
    if not raw_project_path:
        raise ValueError("A project path is required.")
    project_root = Path(raw_project_path).expanduser().resolve(strict=True)
    if not project_root.is_dir() or not (project_root / ".git").exists():
        raise ValueError(f"Not a Git checkout directory: {project_root}")

    print(
        f"\nThe Deep Agent can read and edit project files under:\n{project_root}\n"
        "It will not access .git, common .env files, or private-key files. "
        "It can run only the allow-listed test commands. Test code runs locally "
        "with your user permissions, so continue only with a trusted checkout. "
        "Project source is sent to the configured Groq model. Changes are written "
        "directly to this checkout."
    )
    confirmation = input("Continue? Type 'yes' to authorize: ").strip()
    if confirmation.casefold() != "yes":
        print("Cancelled; no agent was started.")
        return

    agent = build_project_agent(
        project_root,
        ChatGroq(model=MODEL_NAME, temperature=0),
    )
    config = {"configurable": {"thread_id": f"project-{project_root.name}"}}
    task = (
        "Analyze this GitHub project, explain its architecture, add authentication "
        "using appropriate project conventions, update authentication tests, run "
        "the tests and fix failures caused by the changes, then update the README "
        "with authentication setup and usage. Begin by inspecting the repository "
        "and its current Git status."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config=config,
    )
    print(f"\nAgent:\n{result['messages'][-1].content}")

    while True:
        follow_up = input("\nFollow-up or answer to the agent (or 'exit'): ").strip()
        if follow_up.casefold() in {"exit", "quit"}:
            break
        if not follow_up:
            continue
        result = agent.invoke(
            {"messages": [{"role": "user", "content": follow_up}]},
            config=config,
        )
        print(f"\nAgent:\n{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
