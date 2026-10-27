"""CrewAI sample with calculator and hard-coded city weather tools.

Run from the repository root after setting ``GROQ_API_KEY`` in the environment
or a local .env file:

    uv run python crew-ai\\crew-ai-poc-basic.py
"""

import math
import os
from typing import Literal

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field


MODEL_NAME = "openai/gpt-oss-120b"

# These are illustrative sample values, not live weather observations.
CITY_WEATHER = {
    "london": {
        "temperature_c": 15,
        "condition": "Cloudy",
        "humidity_percent": 72,
    },
    "mumbai": {
        "temperature_c": 32,
        "condition": "Sunny",
        "humidity_percent": 68,
    },
    "new york": {
        "temperature_c": 18,
        "condition": "Partly cloudy",
        "humidity_percent": 60,
    },
    "paris": {
        "temperature_c": 17,
        "condition": "Light rain",
        "humidity_percent": 78,
    },
    "tokyo": {
        "temperature_c": 22,
        "condition": "Clear",
        "humidity_percent": 55,
    },
    "chennai": {
        "temperature_c": 30,
        "condition": "Hot and humid",
        "humidity_percent": 76,
    },
    "sydney": {
        "temperature_c": 21,
        "condition": "Partly cloudy",
        "humidity_percent": 64,
    },
}


class CalculatorInput(BaseModel):
    """Validated arguments for a basic arithmetic operation."""

    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="Arithmetic operation to perform."
    )
    first_number: float = Field(description="First operand.")
    second_number: float = Field(description="Second operand.")


class CalculatorTool(BaseTool):
    """Calculate using addition, subtraction, multiplication, or division."""

    name: str = "calculator"
    description: str = (
        "Perform one arithmetic operation on two numbers. Use for calculations. "
        "Supported operations: add, subtract, multiply, divide."
    )
    args_schema: type[BaseModel] = CalculatorInput

    def _run(
        self,
        operation: Literal["add", "subtract", "multiply", "divide"],
        first_number: float,
        second_number: float,
    ) -> str:
        if not math.isfinite(first_number) or not math.isfinite(second_number):
            raise ValueError("Both numbers must be finite.")

        if operation == "add":
            result = first_number + second_number
        elif operation == "subtract":
            result = first_number - second_number
        elif operation == "multiply":
            result = first_number * second_number
        elif operation == "divide":
            if second_number == 0:
                raise ValueError("Cannot divide by zero.")
            result = first_number / second_number
        else:
            raise ValueError(f"Unsupported operation: {operation!r}.")

        if not math.isfinite(result):
            raise ValueError("The result is not finite.")
        return str(result)


class CityWeatherInput(BaseModel):
    """Input schema for the local city weather lookup."""

    city: str = Field(description="City name to look up.")


class CityWeatherTool(BaseTool):
    """Look up sample weather details for a supported city."""

    name: str = "city_weather"
    description: str = (
        "Look up hard-coded sample temperature, conditions, and humidity for a "
        "city. This is sample data, not live or current weather."
    )
    args_schema: type[BaseModel] = CityWeatherInput

    def _run(self, city: str) -> str:
        normalized_city = " ".join(city.casefold().split())
        if not normalized_city:
            raise ValueError("City name cannot be empty.")

        weather = CITY_WEATHER.get(normalized_city)
        if weather is None:
            available_cities = ", ".join(
                name.title() for name in CITY_WEATHER
            )
            return (
                f"No sample weather is available for {city!r}. "
                f"Available cities: {available_cities}."
            )

        return (
            f"Sample weather for {normalized_city.title()}: "
            f"{weather['temperature_c']} degrees C, {weather['condition']}, "
            f"humidity {weather['humidity_percent']}%. "
            "This is hard-coded sample data, not live weather."
        )


def build_crew() -> Crew:
    """Create one CrewAI agent that chooses the relevant tool for each request."""
    tools = [CalculatorTool(), CityWeatherTool()]
    agent = Agent(
        role="Calculator and City Weather Assistant",
        goal=(
            "Answer arithmetic questions with the calculator tool and city "
            "weather questions with the city weather tool."
        ),
        backstory=(
            "You are a precise assistant. You use the calculator for all "
            "arithmetic and the city weather tool for every weather lookup. "
            "You never invent calculation results or weather data."
        ),
        tools=tools,
        llm=LLM(
            # Use CrewAI's native OpenAI-compatible provider with Groq's
            # endpoint. This avoids LiteLLM's MCP integration, which is not
            # compatible with this project's MCP SDK version.
            model=f"openai/{MODEL_NAME}",
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        ),
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=(
            "Answer the user's request: {request}\n"
            "Choose the appropriate available tool. For arithmetic, call the "
            "calculator with the requested operation and operands. For a city "
            "weather request, call the city_weather tool. Clearly identify "
            "weather results as hard-coded samples, not current conditions. "
            "If the request is unrelated or missing necessary details, ask a "
            "short clarifying question instead of guessing."
        ),
        expected_output=(
            "A concise answer based on the selected tool's result, or a concise "
            "clarifying question when needed."
        ),
        agent=agent,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    crew = build_crew()
    print(
        "Calculator and sample city weather assistant ready. "
        "Ask a calculation or weather question; type 'exit' to quit."
    )
    while True:
        try:
            request = input("\nYou: ").strip()
        except EOFError:
            print()
            break
        if request.casefold() in {"exit", "quit"}:
            break
        if not request:
            continue

        result = crew.kickoff(inputs={"request": request})
        print(f"Assistant: {result}")


if __name__ == "__main__":
    main()