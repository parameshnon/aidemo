"""LangChain calculator tools exposed through a local MCP server.

Run the MCP server with ``uv run python langchain-poc\\langchain-poc-mcp.py``.
Run the optional Groq-backed LangChain chat with ``--chat``.
"""

import argparse
import math
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP


def _validate_numbers(first_number: float, second_number: float) -> None:
    if not math.isfinite(first_number) or not math.isfinite(second_number):
        raise ValueError("Both numbers must be finite.")


def _validate_result(result: float) -> float:
    if not math.isfinite(result):
        raise ValueError("The result is not finite.")
    return result


def _add(first_number: float, second_number: float) -> float:
    _validate_numbers(first_number, second_number)
    return _validate_result(first_number + second_number)


def _subtract(first_number: float, second_number: float) -> float:
    _validate_numbers(first_number, second_number)
    return _validate_result(first_number - second_number)


def _multiply(first_number: float, second_number: float) -> float:
    _validate_numbers(first_number, second_number)
    return _validate_result(first_number * second_number)


def _divide(first_number: float, second_number: float) -> float:
    _validate_numbers(first_number, second_number)
    if second_number == 0:
        raise ValueError("Cannot divide by zero.")
    return _validate_result(first_number / second_number)


@tool
def add(first_number: float, second_number: float) -> float:
    """Add two numbers."""
    return _add(first_number, second_number)


@tool
def subtract(first_number: float, second_number: float) -> float:
    """Subtract the second number from the first number."""
    return _subtract(first_number, second_number)


@tool
def multiply(first_number: float, second_number: float) -> float:
    """Multiply two numbers."""
    return _multiply(first_number, second_number)


@tool
def divide(first_number: float, second_number: float) -> float:
    """Divide the first number by the second number."""
    return _divide(first_number, second_number)


mcp = FastMCP(
    "langchain-calculator",
    instructions=(
        "A calculator that adds, subtracts, multiplies, and divides. "
        "Use the calculator://guide resource for tool details."
    ),
    host="127.0.0.1",
)


@mcp.resource(
    "calculator://guide",
    name="Calculator guide",
    description="Supported calculator operations, argument names, and limitations.",
    mime_type="text/markdown",
)
def calculator_guide() -> str:
    """Describe the calculator tools available from this MCP server."""
    return """# Calculator tools

Use one of these tools for arithmetic:

- `add(first_number, second_number)`: add two numbers.
- `subtract(first_number, second_number)`: subtract the second number from the first.
- `multiply(first_number, second_number)`: multiply two numbers.
- `divide(first_number, second_number)`: divide the first number by the second.

Inputs and results must be finite numbers. Division by zero is not allowed.
"""


@mcp.prompt(
    name="calculator_request",
    description="Create a prompt that asks an assistant to solve a calculation using calculator tools.",
)
def calculator_request(question: str) -> str:
    """Build a reusable calculation request for an MCP client."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    return (
        "Solve the user's calculation using the available calculator tools. "
        "Do not calculate mentally; call the appropriate tool and explain the "
        f"result clearly.\n\nUser's calculation: {question.strip()}"
    )


@mcp.tool(name="add")
def mcp_add(first_number: float, second_number: float) -> float:
    """Add two numbers."""
    return _add(first_number, second_number)


@mcp.tool(name="subtract")
def mcp_subtract(first_number: float, second_number: float) -> float:
    """Subtract the second number from the first number."""
    return _subtract(first_number, second_number)


@mcp.tool(name="multiply")
def mcp_multiply(first_number: float, second_number: float) -> float:
    """Multiply two numbers."""
    return _multiply(first_number, second_number)


@mcp.tool(name="divide")
def mcp_divide(first_number: float, second_number: float) -> float:
    """Divide the first number by the second number."""
    return _divide(first_number, second_number)


def run_chat() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    agent = create_agent(
        model=ChatGroq(model="openai/gpt-oss-120b", temperature=0),
        tools=[add, subtract, multiply, divide],
        system_prompt=(
            "You are a calculator assistant. Always use the appropriate "
            "calculator tool for arithmetic and report its result accurately."
        ),
    )
    print("Ask for a calculation. Type 'exit' to quit.")
    while True:
        prompt = input("You: ").strip()
        if prompt.casefold() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        print(f"Assistant: {result['messages'][-1].content}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chat",
        action="store_true",
        help="run the optional Groq-backed LangChain calculator instead of MCP",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport (streamable HTTP listens only on localhost)",
    )
    args = parser.parse_args()

    if args.chat:
        run_chat()
        return
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()