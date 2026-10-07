import os
from typing import Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


DESTINATIONS = {
    "bali": {
        "name": "Bali",
        "hotels": [
            {"hotel_id": "HTL-BALI-001", "name": "Palm Garden Resort", "price": 145, "tags": ["beach", "luxury"]},
            {"hotel_id": "HTL-BALI-002", "name": "Ubud Valley Hotel", "price": 110, "tags": ["quiet", "nature", "budget"]},
        ],
        "booking": {"booking_id": "BOOK-BALI-1001", "hotel_id": "HTL-BALI-001"},
    },
    "paris": {
        "name": "Paris",
        "hotels": [
            {"hotel_id": "HTL-PARIS-001", "name": "Left Bank Boutique Hotel", "price": 220, "tags": ["central", "luxury"]},
            {"hotel_id": "HTL-PARIS-002", "name": "Montmartre Stay", "price": 160, "tags": ["quiet", "budget"]},
        ],
        "booking": {"booking_id": "BOOK-PARIS-1001", "hotel_id": "HTL-PARIS-001"},
    },
    "tokyo": {
        "name": "Tokyo",
        "hotels": [
            {"hotel_id": "HTL-TOKYO-001", "name": "Shinjuku City Hotel", "price": 175, "tags": ["central", "nightlife"]},
            {"hotel_id": "HTL-TOKYO-002", "name": "Asakusa Riverside Inn", "price": 125, "tags": ["quiet", "budget"]},
        ],
        "booking": {"booking_id": "BOOK-TOKYO-1001", "hotel_id": "HTL-TOKYO-001"},
    },
    "singapore": {
        "name": "Singapore",
        "hotels": [
            {"hotel_id": "HTL-SG-001", "name": "Marina Bay City Hotel", "price": 210, "tags": ["central", "luxury"]},
            {"hotel_id": "HTL-SG-002", "name": "Garden District Stay", "price": 155, "tags": ["quiet", "nature"]},
        ],
        "booking": {"booking_id": "BOOK-SG-1001", "hotel_id": "HTL-SG-001"},
    },
    "india": {
        "name": "India",
        "hotels": [
            {"hotel_id": "HTL-IND-001", "name": "Jaipur Heritage Hotel", "price": 95, "tags": ["heritage", "budget"]},
            {"hotel_id": "HTL-IND-002", "name": "Goa Coast Resort", "price": 130, "tags": ["beach", "relaxing"]},
        ],
        "booking": {"booking_id": "BOOK-IND-1001", "hotel_id": "HTL-IND-001"},
    },
}

REFUNDS = {
    "BOOK-BALI-1001": "REF-BALI-5001",
    "BOOK-PARIS-1001": "REF-PARIS-5001",
    "BOOK-TOKYO-1001": "REF-TOKYO-5001",
    "BOOK-SG-1001": "REF-SG-5001",
    "BOOK-IND-1001": "REF-IND-5001",
}


class RouterState(TypedDict):
    query: str
    route: NotRequired[str]
    location: NotRequired[str]
    preference: NotRequired[str]
    booking_id: NotRequired[str]
    response: NotRequired[str]


class RouteDecision(BaseModel):
    route: Literal["support", "booking", "refund", "clarify"] = Field(
        description="Best destination route for the user's request."
    )
    location: str = Field(
        default="",
        description="Destination name mentioned by the user, or an empty string.",
    )
    preference: str = Field(
        default="",
        description="Hotel preference such as budget, quiet, or beach.",
    )
    booking_id: str = Field(
        default="",
        description="Booking ID mentioned by the user, or an empty string.",
    )
    clarification: str = Field(
        default="",
        description="Short question to ask if the request is unclear.",
    )


@tool
def recommend_hotels(location: str, preference: str = "") -> str:
    """Recommend catalog hotels for a destination and optional preference."""
    destination = DESTINATIONS.get(location.strip().casefold())
    if destination is None:
        return "Unknown destination. Available destinations: " + ", ".join(
            item["name"] for item in DESTINATIONS.values()
        )
    hotels = destination["hotels"]
    tags = {
        tag
        for tag in ("budget", "luxury", "quiet", "beach", "central", "nature", "heritage")
        if tag in preference.casefold()
    }
    if tags:
        matches = [hotel for hotel in hotels if tags.intersection(hotel["tags"])]
        if matches:
            hotels = matches
    return "\n".join(
        f"{hotel['name']} | hotel ID: {hotel['hotel_id']} | "
        f"USD {hotel['price']}/night | {', '.join(hotel['tags'])}"
        for hotel in hotels
    )


@tool
def get_booking_details(location: str) -> str:
    """Look up the sample hotel and booking IDs for a destination."""
    destination = DESTINATIONS.get(location.strip().casefold())
    if destination is None:
        return "No booking found. Available destinations: " + ", ".join(
            item["name"] for item in DESTINATIONS.values()
        )
    booking = destination["booking"]
    hotel = next(
        hotel
        for hotel in destination["hotels"]
        if hotel["hotel_id"] == booking["hotel_id"]
    )
    return (
        f"Sample booking: {booking['booking_id']}; hotel: {hotel['name']}; "
        f"hotel ID: {hotel['hotel_id']}. This is not a real reservation."
    )


@tool
def get_refund_details(booking_id: str) -> str:
    """Look up the sample refund ID for a booking ID."""
    booking_id = booking_id.strip().upper()
    refund_id = REFUNDS.get(booking_id)
    if refund_id is None:
        return f"No refund record found for {booking_id or 'an empty booking ID'}."
    return f"Sample refund ID for booking {booking_id}: {refund_id} (processed)."


def build_router_graph(model: ChatGroq):
    """Route a user query to a dedicated LangChain specialist agent."""
    router_model = model.with_structured_output(RouteDecision)
    support_agent = create_agent(
        model=model,
        tools=[recommend_hotels],
        system_prompt=(
            "You handle hotel support. Always use recommend_hotels for catalog "
            "recommendations, and do not invent hotel details."
        ),
    )
    booking_agent = create_agent(
        model=model,
        tools=[get_booking_details],
        system_prompt=(
            "You handle booking lookups. Use get_booking_details and clearly state "
            "that catalog bookings are samples, not real reservations."
        ),
    )
    refund_agent = create_agent(
        model=model,
        tools=[get_refund_details],
        system_prompt=(
            "You handle refund lookups. Use get_refund_details and clearly state "
            "that these are sample records, not real refunds."
        ),
    )

    def classify(state: RouterState) -> dict[str, str]:
        decision = router_model.invoke(
            [
                (
                    "system",
                    "Route hotel recommendations to support, booking lookups to "
                    "booking, refund or cancellation requests to refund, and unclear "
                    "requests to clarify. Extract destination, hotel preference, "
                    "and booking ID only when present. Do not guess missing values.",
                ),
                ("human", state["query"]),
            ]
        )
        return {
            "route": decision.route,
            "location": decision.location,
            "preference": decision.preference,
            "booking_id": decision.booking_id,
            "response": decision.clarification,
        }

    def run_support(state: RouterState) -> dict[str, str]:
        result = support_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Request: {state['query']}\n"
                            f"Location: {state.get('location', '')}\n"
                            f"Preference: {state.get('preference', '')}"
                        ),
                    }
                ]
            }
        )
        return {"response": str(result["messages"][-1].content)}

    def run_booking(state: RouterState) -> dict[str, str]:
        result = booking_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Request: {state['query']}\n"
                            f"Location: {state.get('location', '')}"
                        ),
                    }
                ]
            }
        )
        return {"response": str(result["messages"][-1].content)}

    def run_refund(state: RouterState) -> dict[str, str]:
        result = refund_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Request: {state['query']}\n"
                            f"Booking ID: {state.get('booking_id', '')}"
                        ),
                    }
                ]
            }
        )
        return {"response": str(result["messages"][-1].content)}

    def clarify(state: RouterState) -> dict[str, str]:
        return {
            "response": state.get("response")
            or "Would you like hotel recommendations, a booking lookup, or a refund?"
        }

    def route(state: RouterState) -> Literal["support", "booking", "refund", "clarify"]:
        destination = state.get("route")
        if destination in {"support", "booking", "refund", "clarify"}:
            return destination
        raise ValueError(f"Unsupported route: {destination!r}.")

    graph = StateGraph(RouterState)
    graph.add_node("router", classify)
    graph.add_node("support", run_support)
    graph.add_node("booking", run_booking)
    graph.add_node("refund", run_refund)
    graph.add_node("clarify", clarify)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route,
        {
            "support": "support",
            "booking": "booking",
            "refund": "refund",
            "clarify": "clarify",
        },
    )
    for node in ("support", "booking", "refund", "clarify"):
        graph.add_edge(node, END)
    return graph.compile()


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )
    app = build_router_graph(ChatGroq(model="openai/gpt-oss-120b", temperature=0))
    print("Concierge router ready. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ").strip()
        if query.casefold() in {"exit", "quit"}:
            break
        if query:
            result = app.invoke({"query": query})
            print(f"Concierge: {result['response']}")


if __name__ == "__main__":
    main()