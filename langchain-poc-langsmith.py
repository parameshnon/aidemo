import math
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable


PRODUCTS = {
    "laptop": {
        "price": 999.99,
        "currency": "USD",
        "description": "14-inch laptop with 16 GB memory and 512 GB storage",
    },
    "headphones": {
        "price": 149.50,
        "currency": "USD",
        "description": "Wireless noise-cancelling headphones",
    },
    "keyboard": {
        "price": 79.99,
        "currency": "USD",
        "description": "Mechanical keyboard with quiet switches",
    },
    "mouse": {
        "price": 39.95,
        "currency": "USD",
        "description": "Wireless optical mouse",
    },
}


@tool
@traceable(name="calculator", run_type="tool")
def calculate(operation: str, first_number: float, second_number: float) -> str:
    """Calculate using add, subtract, multiply, or divide and two numbers."""
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
        "divide": lambda: first_number / second_number,
        "division": lambda: first_number / second_number,
        "/": lambda: first_number / second_number,
    }
    if normalized_operation not in operations:
        raise ValueError("Operation must be add, subtract, multiply, or divide.")
    if normalized_operation in {"divide", "division", "/"} and second_number == 0:
        raise ValueError("Cannot divide by zero.")

    result = operations[normalized_operation]()
    if not math.isfinite(result):
        raise ValueError("The result is not finite.")
    return str(result)


@tool
@traceable(name="product_lookup", run_type="tool")
def get_product_details(product_name: str) -> str:
    """Find the price, currency, and description of a product in the catalog."""
    normalized_name = " ".join(product_name.casefold().split())
    if not normalized_name:
        raise ValueError("Product name cannot be empty.")

    product = PRODUCTS.get(normalized_name)
    if product is None:
        available_products = ", ".join(sorted(PRODUCTS))
        return (
            f"No product named {product_name!r} was found. "
            f"Available products: {available_products}."
        )

    return (
        f"{normalized_name.title()}: {product['currency']} {product['price']:.2f} - "
        f"{product['description']}."
    )


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "langchain-groq-product-calculator")
    if not os.getenv("LANGSMITH_API_KEY"):
        print(
            "LangSmith tracing is enabled, but traces will not be sent until "
            "LANGSMITH_API_KEY is configured."
        )

    agent = create_agent(
        model=ChatGroq(model="openai/gpt-oss-120b", temperature=0),
        tools=[calculate, get_product_details],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You are a helpful assistant. Use the calculate tool for addition, "
            "subtraction, multiplication, and division. Use the "
            "get_product_details tool for product prices and details. "
            "Product information comes from a fixed catalog; do not invent "
            "prices or claim the catalog is live."
        ),
    )

    config = {"configurable": {"thread_id": "product-calculator-session"}}
    print("Ask for a calculation or product details. Type 'exit' to quit.")
    while True:
        prompt = input("You: ").strip()
        if prompt.casefold() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
        )
        print(f"Assistant: {response['messages'][-1].content}")


if __name__ == "__main__":
    main()