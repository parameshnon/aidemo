"""A small LangGraph ReAct (reason, act, observe) agent using Groq tools."""

import operator
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


CITY_TEMPERATURES_C = {
    "london": 15,
    "mumbai": 32,
    "new york": 18,
    "paris": 17,
    "tokyo": 22,
}

ARITHMETIC_OPERATIONS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


@tool
def calculate(operation: str, first_number: float, second_number: float) -> str:
    """Calculate two numbers using add, subtract, multiply, or divide."""
    operation = operation.strip().casefold()
    if operation not in ARITHMETIC_OPERATIONS:
        raise ValueError("Choose add, subtract, multiply, or divide.")
    if operation == "divide" and second_number == 0:
        raise ValueError("Cannot divide by zero.")
    return str(ARITHMETIC_OPERATIONS[operation](first_number, second_number))


@tool
def get_city_temperature(city: str) -> str:
    """Get the sample temperature in Celsius for a city in the local dictionary."""
    normalized_city = " ".join(city.casefold().split())
    temperature = CITY_TEMPERATURES_C.get(normalized_city)
    if temperature is None:
        available_cities = ", ".join(city.title() for city in CITY_TEMPERATURES_C)
        return f"No sample temperature for {city!r}. Available cities: {available_cities}."
    return f"The sample temperature in {city.title()} is {temperature} degrees C."


TOOLS = [calculate, get_city_temperature]


def build_react_graph(model_name: str = "openai/gpt-oss-120b"):
    """Build a ReAct cycle: model -> optional tools -> model."""
    model = ChatGroq(model=model_name, temperature=0).bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict[str, list]:
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a helpful ReAct assistant. Think about the user's "
                        "request, call an available tool when you need calculation "
                        "or city-temperature data, then use the tool result to answer. "
                        "City temperatures are hard-coded sample values, not live data."
                    )
                ),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "assistant")
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    agent = build_react_graph()
    config = {"configurable": {"thread_id": "react-demo-session"}}
    print("ReAct agent ready. Try a calculation or ask for a sample city temperature.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        query = input("\nYou: ").strip()
        if query.casefold() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )
        print(f"Assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()