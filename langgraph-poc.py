from typing import NotRequired, TypedDict

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph


class TemperatureState(TypedDict):
    celsius: float
    fahrenheit: NotRequired[float]


def convert_to_fahrenheit(state: TemperatureState) -> dict[str, float]:
    """Convert the Celsius value in the graph state to Fahrenheit."""
    return {"fahrenheit": (state["celsius"] * 9 / 5) + 32}


def build_graph():
    graph = StateGraph(TemperatureState)
    graph.add_node("convert_to_fahrenheit", convert_to_fahrenheit)
    graph.add_edge(START, "convert_to_fahrenheit")
    graph.add_edge("convert_to_fahrenheit", END)
    return graph.compile()


def main() -> None:
    try:
        celsius = float(input("Enter temperature in Celsius: ").strip())
    except ValueError as error:
        raise ValueError("Please enter a valid number for Celsius.") from error

    app = build_graph()
    result = app.invoke({"celsius": celsius})
    print(f"{celsius:g} C = {result['fahrenheit']:.2f} F")

    graph_image = app.get_graph().draw_mermaid_png()
    display(Image(graph_image))


if __name__ == "__main__":
    main()