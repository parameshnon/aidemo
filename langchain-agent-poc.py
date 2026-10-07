import math
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver


CITY_TEMPERATURES = {
    "london": 15,
    "mumbai": 32,
    "new york": 18,
    "paris": 17,
    "tokyo": 22,
    "chennai" :"100",
    "sydney" :"78"
}


@tool
def calculate(operation: str, first_number: float, second_number: float) -> str:
    """Perform one basic addition, subtraction, multiplication, or division."""
    if not math.isfinite(first_number) or not math.isfinite(second_number):
        raise ValueError("Both numbers must be finite.")

    normalized_operation = operation.strip().casefold()
    operations = {
        "add": lambda: first_number + second_number,
        "addition": lambda: first_number + second_number,
        "+": lambda: first_number + second_number,
        "subtract": lambda: first_number - second_number,
        "subtraction": lambda: first_number - second_number,
        "-": lambda: first_number - second_number,
        "multiply": lambda: first_number * second_number,
        "multiplication": lambda: first_number * second_number,
        "*": lambda: first_number * second_number,
        "×": lambda: first_number * second_number,
        "divide": lambda: first_number / second_number,
        "division": lambda: first_number / second_number,
        "/": lambda: first_number / second_number,
        "÷": lambda: first_number / second_number,
    }
    if normalized_operation not in operations:
        raise ValueError("Operation must be add, subtract, multiply, or divide.")
    if normalized_operation in {"divide", "division", "/", "÷"} and second_number == 0:
        raise ValueError("Cannot divide by zero.")

    result = operations[normalized_operation]()
    if not math.isfinite(result):
        raise ValueError("The result is not finite.")
    return str(result)


@tool
def get_city_temperature(city: str) -> str:
    """Look up a sample temperature in Celsius from the built-in city dictionary."""
    normalized_city = city.strip().casefold()
    if not normalized_city:
        raise ValueError("City name cannot be empty.")

    temperature = CITY_TEMPERATURES.get(normalized_city)
    if temperature is None:
        available_cities = ", ".join(city.title() for city in CITY_TEMPERATURES)
        return f"No temperature data for {city!r}. Available cities: {available_cities}."

    return f"The sample temperature in {city.strip().title()} is {temperature} degrees C."


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    agent = create_agent(
        model=ChatGroq(model="openai/gpt-oss-120b", temperature=0),
        tools=[calculate, get_city_temperature],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You are a helpful assistant. Use the calculate tool for basic "
            "addition, subtraction, multiplication, and division. Pass exactly "
            "one operation and two numbers to the tool. Also use the "
            "get_city_temperature tool for city temperatures. "
            "The temperature tool returns hard-coded sample data, not live weather. "
            "Do not claim that these temperatures are current."
        ),
    )

    config = {"configurable": {"thread_id": "cli-session"}}
    print("Chat with the agent. Type 'exit' to quit.")

    while True:
        prompt = input("You: ").strip()
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