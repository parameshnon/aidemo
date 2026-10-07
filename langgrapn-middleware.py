import logging
import os
import secrets
from getpass import getpass
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model, before_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field


MAX_INPUT_LENGTH = 4_000
logger = logging.getLogger("middleware_agent")


class AgentContext(TypedDict):
    access_token: str


class AssistantResponse(BaseModel):
    answer: str = Field(description="A concise answer to the user's request.")
    category: str = Field(description="A short category for the response.")


@tool
def count_words(text: str) -> int:
    """Count the words in the provided text."""
    return len(text.split())


@before_model
def validate_user_input(state: AgentState, runtime: Runtime[AgentContext]) -> None:
    """Reject empty or excessively long user requests before calling the model."""
    latest_user_message = next(
        (
            message
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    if latest_user_message is None or not isinstance(latest_user_message.content, str):
        raise ValueError("A text user message is required.")

    request = latest_user_message.content.strip()
    if not request:
        raise ValueError("The request cannot be empty.")
    if len(request) > MAX_INPUT_LENGTH:
        raise ValueError(f"The request must be at most {MAX_INPUT_LENGTH} characters.")


@before_model
def authorize_request(state: AgentState, runtime: Runtime[AgentContext]) -> None:
    """Check the caller's token against the server-configured token."""
    expected_token = os.getenv("CONCIERGE_ACCESS_TOKEN")
    if not expected_token:
        raise RuntimeError(
            "CONCIERGE_ACCESS_TOKEN is not configured; refusing unauthorized access."
        )

    context = runtime.context
    supplied_token = context.get("access_token", "") if context else ""
    if not supplied_token or not secrets.compare_digest(
        supplied_token, expected_token
    ):
        logger.warning("Rejected an unauthorized agent request.")
        raise PermissionError("Unauthorized. The access token is missing or invalid.")


@before_model
def log_request(state: AgentState, runtime: Runtime[AgentContext]) -> None:
    """Log request metadata without recording prompt contents or credentials."""
    logger.info("Processing agent request with %d message(s).", len(state["messages"]))


@after_model
def log_response(state: AgentState, runtime: Runtime[AgentContext]) -> None:
    """Log model completion without recording generated content."""
    logger.info("Model response generated.")


def build_agent(model: ChatGroq):
    return create_agent(
        model=model,
        tools=[count_words],
        system_prompt=(
            "You are a helpful assistant. Use the count_words tool when asked to "
            "count words. Return a concise answer and classify the response."
        ),
        middleware=[
            validate_user_input,
            authorize_request,
            log_request,
            log_response,
        ],
        context_schema=AgentContext,
        response_format=AssistantResponse,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY must be configured in the environment or .env.")
    if not os.getenv("CONCIERGE_ACCESS_TOKEN"):
        raise RuntimeError(
            "CONCIERGE_ACCESS_TOKEN must be configured in the environment or .env."
        )

    agent = build_agent(ChatGroq(model="openai/gpt-oss-120b", temperature=0))
    access_token = getpass("Concierge access token: ")
    prompt = input("What would you like help with? ")

    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        context={"access_token": access_token},
    )
    structured_response = result.get("structured_response")
    if not isinstance(structured_response, AssistantResponse):
        raise RuntimeError("The agent did not return the expected structured response.")
    print(structured_response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()