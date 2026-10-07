import logging
import os

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model, before_model
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
MAX_PROMPT_LENGTH = 4_000


class ContentResponse(BaseModel):
    answer: str = Field(description="A clear answer to the user's request.")
    category: str = Field(
        description="A short category describing the type of request."
    )
    key_points: list[str] = Field(
        description="The main points in the answer."
    )


@tool
def count_words(text: str) -> int:
    """Count the words in a text string."""
    return len(text.split())


@before_model
def validate_and_log_request(state: AgentState, runtime: Runtime) -> None:
    """Validate the latest user request before sending it to the model."""
    user_messages = [
        message for message in state["messages"] if isinstance(message, HumanMessage)
    ]
    if not user_messages:
        raise ValueError("The conversation must contain a user message.")
    if len(user_messages[-1].content) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Please keep the request under {MAX_PROMPT_LENGTH} characters."
        )
    logger.info("Validated request with %d conversation message(s).", len(state["messages"]))


@after_model
def log_model_response(state: AgentState, runtime: Runtime) -> None:
    """Log when the model has returned a response."""
    if state["messages"]:
        logger.info("Model returned a response.")


def build_agent(model: ChatGroq):
    return create_agent(
        model=model,
        tools=[count_words],
        system_prompt=(
            "You are a helpful assistant. Use the count_words tool when the user "
            "asks for a word count. Provide concise, accurate answers."
        ),
        middleware=[validate_and_log_request, log_model_response],
        response_format=ToolStrategy(ContentResponse),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    agent = build_agent(ChatGroq(model="openai/gpt-oss-120b", temperature=0))
    prompt = input("What would you like help with? ").strip()
    if not prompt:
        raise ValueError("Please enter a request.")

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    structured_response = result.get("structured_response")
    if not isinstance(structured_response, ContentResponse):
        raise RuntimeError("The model did not return the expected structured response.")
    print(structured_response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()