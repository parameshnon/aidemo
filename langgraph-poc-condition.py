import math
from typing import Literal, NotRequired, TypedDict

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph


USD_TO_INR_RATE = 93.0
USD_TO_EUR_RATE = 0.92


class CurrencyState(TypedDict):
    amount_usd: float
    currency: str
    currency_code: NotRequired[str]
    converted_amount: NotRequired[float]
    exchange_rate: NotRequired[float]


def select_currency(state: CurrencyState) -> dict[str, str]:
    """Normalize and validate the requested target currency."""
    currency = state["currency"].strip().casefold()
    if currency in {"inr", "rupee", "rupees"}:
        return {"currency_code": "INR"}
    if currency in {"eur", "euro", "euros"}:
        return {"currency_code": "EUR"}
    raise ValueError("Unsupported currency. Choose INR or EUR (Euro).")


def route_currency(state: CurrencyState) -> Literal["convert_to_inr", "convert_to_eur"]:
    if state["currency_code"] == "INR":
        return "convert_to_inr"
    if state["currency_code"] == "EUR":
        return "convert_to_eur"
    raise ValueError("Currency was not selected.")


def convert_to_inr(state: CurrencyState) -> dict[str, float]:
    return {
        "converted_amount": state["amount_usd"] * USD_TO_INR_RATE,
        "exchange_rate": USD_TO_INR_RATE,
    }


def convert_to_eur(state: CurrencyState) -> dict[str, float]:
    return {
        "converted_amount": state["amount_usd"] * USD_TO_EUR_RATE,
        "exchange_rate": USD_TO_EUR_RATE,
    }


def build_graph():
    graph = StateGraph(CurrencyState)
    graph.add_node("select_currency", select_currency)
    graph.add_node("convert_to_inr", convert_to_inr)
    graph.add_node("convert_to_eur", convert_to_eur)
    graph.add_edge(START, "select_currency")
    graph.add_conditional_edges(
        "select_currency",
        route_currency,
        {
            "convert_to_inr": "convert_to_inr",
            "convert_to_eur": "convert_to_eur",
        },
    )
    graph.add_edge("convert_to_inr", END)
    graph.add_edge("convert_to_eur", END)
    return graph.compile()


def main() -> None:
    try:
        amount_usd = float(input("Enter amount in USD: ").strip())
    except ValueError as error:
        raise ValueError("Please enter a valid USD amount.") from error
    if not math.isfinite(amount_usd) or amount_usd < 0:
        raise ValueError("USD amount must be a finite, non-negative number.")

    currency = input("Convert to INR or EUR (Euro): ").strip()
    app = build_graph()
    result = app.invoke({"amount_usd": amount_usd, "currency": currency})

    currency_code = result["currency_code"]
    print(
        f"{amount_usd:.2f} USD = {result['converted_amount']:.2f} "
        f"{currency_code} (rate: 1 USD = {result['exchange_rate']} {currency_code})"
    )
    print("Note: Exchange rates are hard-coded example values, not live rates.")

    graph_image = app.get_graph().draw_mermaid_png()
    display(Image(graph_image))


if __name__ == "__main__":
    main()
