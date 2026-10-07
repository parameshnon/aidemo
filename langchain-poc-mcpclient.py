"""Client for the local LangChain calculator MCP server.

Start this client from the repository root with:
``uv run python langchain-poc\\langchain-poc-mcpclient.py``

Use --list-resources, --read-resource, --list-prompts, or --get-prompt
to access server resources and prompts directly.
"""

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client


def _format_content(content: object) -> str:
    text = getattr(content, "text", None)
    if text is not None:
        return text
    dump_json = getattr(content, "model_dump_json", None)
    if dump_json is not None:
        return dump_json()
    return str(content)


async def list_resources(session: ClientSession) -> None:
    result = await session.list_resources()
    if not result.resources:
        print("The server has no resources.")
        return
    for resource in result.resources:
        print(f"{resource.name or resource.uri}")
        print(f"  URI: {resource.uri}")
        if resource.description:
            print(f"  Description: {resource.description}")


async def read_resource(session: ClientSession, uri: str) -> None:
    result = await session.read_resource(uri)
    for content in result.contents:
        print(_format_content(content))


async def list_prompts(session: ClientSession) -> None:
    result = await session.list_prompts()
    if not result.prompts:
        print("The server has no prompts.")
        return
    for prompt in result.prompts:
        print(f"{prompt.name}")
        if prompt.description:
            print(f"  Description: {prompt.description}")
        for argument in prompt.arguments or []:
            requirement = "required" if argument.required else "optional"
            description = f" - {argument.description}" if argument.description else ""
            print(f"  Argument: {argument.name} ({requirement}){description}")


async def get_prompt(
    session: ClientSession, name: str, arguments: dict[str, str]
) -> None:
    result = await session.get_prompt(name, arguments)
    for message in result.messages:
        print(f"{message.role}: {_format_content(message.content)}")


@asynccontextmanager
async def connect_to_server(
    transport: str, url: str
) -> AsyncIterator[ClientSession]:
    if transport == "stdio":
        server_script = Path(__file__).with_name("langchain-poc-mcp.py")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script)],
            cwd=str(server_script.parent.parent),
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
        return

    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def run_client(
    transport: str,
    url: str,
    action: str | None = None,
    resource_uri: str | None = None,
    prompt_name: str | None = None,
    prompt_arguments: dict[str, str] | None = None,
) -> None:
    async with connect_to_server(transport, url) as session:
        if action == "list-resources":
            await list_resources(session)
            return
        if action == "read-resource":
            if resource_uri is None:
                raise ValueError("A resource URI is required.")
            await read_resource(session, resource_uri)
            return
        if action == "list-prompts":
            await list_prompts(session)
            return
        if action == "get-prompt":
            if prompt_name is None:
                raise ValueError("A prompt name is required.")
            await get_prompt(session, prompt_name, prompt_arguments or {})
            return

        available_tools = await session.list_tools()
        tool_names = [item.name for item in available_tools.tools]

        print(f"Connected to calculator MCP server ({transport}).")
        print(f"Available tools: {', '.join(tool_names)}")
        print("Enter a tool name, 'resources', 'resource', 'prompts', 'prompt', or 'exit'.")

        while True:
            selected_tool = input("Tool: ").strip().casefold()
            if selected_tool in {"exit", "quit"}:
                break
            if selected_tool == "resources":
                await list_resources(session)
                continue
            if selected_tool == "resource":
                uri = input("Resource URI: ").strip()
                if uri:
                    await read_resource(session, uri)
                else:
                    print("Resource URI cannot be empty.")
                continue
            if selected_tool == "prompts":
                await list_prompts(session)
                continue
            if selected_tool == "prompt":
                name = input("Prompt name: ").strip()
                if not name:
                    print("Prompt name cannot be empty.")
                    continue
                prompt_argument = input("question: ").strip()
                await get_prompt(session, name, {"question": prompt_argument})
                continue
            if selected_tool not in tool_names:
                options = [
                    *tool_names,
                    "resources",
                    "resource",
                    "prompts",
                    "prompt",
                    "exit",
                ]
                print(f"Choose one of: {', '.join(options)}")
                continue

            try:
                first_number = float(input("First number: ").strip())
                second_number = float(input("Second number: ").strip())
            except ValueError:
                print("Please enter valid numbers.")
                continue

            result = await session.call_tool(
                selected_tool,
                {
                    "first_number": first_number,
                    "second_number": second_number,
                },
            )

            if result.isError:
                details = "; ".join(
                    item.text for item in result.content if hasattr(item, "text")
                )
                print(f"Calculator error: {details or 'Unknown tool error.'}")
                continue

            for item in result.content:
                if hasattr(item, "text"):
                    print(f"Result: {item.text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples: --list-resources; --read-resource calculator://guide; "
            "--list-prompts; --get-prompt calculator_request "
            "--prompt-arg question='What is 12 divided by 3?'"
        ),
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio launches the local server; streamable-http connects to a running server",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/mcp",
        help="MCP URL used with --transport streamable-http",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--list-resources",
        dest="action",
        action="store_const",
        const="list-resources",
        help="list resources exposed by the server",
    )
    actions.add_argument(
        "--read-resource",
        dest="resource_uri",
        metavar="URI",
        help="read a resource by its URI",
    )
    actions.add_argument(
        "--list-prompts",
        dest="action",
        action="store_const",
        const="list-prompts",
        help="list prompts exposed by the server",
    )
    actions.add_argument(
        "--get-prompt",
        dest="prompt_name",
        metavar="NAME",
        help="render a server prompt by name",
    )
    parser.add_argument(
        "--prompt-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="prompt argument; may be supplied multiple times with --get-prompt",
    )
    args = parser.parse_args()

    if args.prompt_arg and not args.prompt_name:
        parser.error("--prompt-arg can only be used with --get-prompt.")
    prompt_arguments: dict[str, str] = {}
    for item in args.prompt_arg:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            parser.error(f"Invalid --prompt-arg {item!r}; expected KEY=VALUE.")
        prompt_arguments[key.strip()] = value

    action = args.action
    if args.resource_uri:
        action = "read-resource"
    elif args.prompt_name:
        action = "get-prompt"

    try:
        asyncio.run(
            run_client(
                args.transport,
                args.url,
                action=action,
                resource_uri=args.resource_uri,
                prompt_name=args.prompt_name,
                prompt_arguments=prompt_arguments,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(
            f"Could not connect to the calculator MCP server: {error}"
        ) from error


if __name__ == "__main__":
    main()