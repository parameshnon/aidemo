import os
from typing import Annotated, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


DESTINATIONS = {
    "bali": {
        "name": "Bali",
        "hotels": [
            {
                "hotel_id": "HTL-BALI-001",
                "name": "Palm Garden Resort",
                "nightly_rate_usd": 145,
                "tags": ["beach", "luxury"],
            },
            {
                "hotel_id": "HTL-BALI-002",
                "name": "Ubud Valley Hotel",
                "nightly_rate_usd": 110,
                "tags": ["quiet", "nature", "budget"],
            },
        ],
        "booking": {
            "booking_id": "BOOK-BALI-1001",
            "hotel_id": "HTL-BALI-001",
        },
    },
    "paris": {
        "name": "Paris",
        "hotels": [
            {
                "hotel_id": "HTL-PARIS-001",
                "name": "Left Bank Boutique Hotel",
                "nightly_rate_usd": 220,
                "tags": ["central", "luxury"],
            },
            {
                "hotel_id": "HTL-PARIS-002",
                "name": "Montmartre Stay",
                "nightly_rate_usd": 160,
                "tags": ["quiet", "budget"],
            },
        ],
        "booking": {
            "booking_id": "BOOK-PARIS-1001",
            "hotel_id": "HTL-PARIS-001",
        },
    },
    "tokyo": {
        "name": "Tokyo",
        "hotels": [
            {
                "hotel_id": "HTL-TOKYO-001",
                "name": "Shinjuku City Hotel",
                "nightly_rate_usd": 175,
                "tags": ["central", "nightlife"],
            },
            {
                "hotel_id": "HTL-TOKYO-002",
                "name": "Asakusa Riverside Inn",
                "nightly_rate_usd": 125,
                "tags": ["quiet", "budget"],
            },
        ],
        "booking": {
            "booking_id": "BOOK-TOKYO-1001",
            "hotel_id": "HTL-TOKYO-001",
        },
    },
    "singapore": {
        "name": "Singapore",
        "hotels": [
            {
                "hotel_id": "HTL-SG-001",
                "name": "Marina Bay City Hotel",
                "nightly_rate_usd": 210,
                "tags": ["central", "luxury"],
            },
            {
                "hotel_id": "HTL-SG-002",
                "name": "Garden District Stay",
                "nightly_rate_usd": 155,
                "tags": ["quiet", "nature"],
            },
        ],
        "booking": {
            "booking_id": "BOOK-SG-1001",
            "hotel_id": "HTL-SG-001",
        },
    },
    "india": {
        "name": "India",
        "hotels": [
            {
                "hotel_id": "HTL-IND-001",
                "name": "Jaipur Heritage Hotel",
                "nightly_rate_usd": 95,
                "tags": ["heritage", "budget"],
            },
            {
                "hotel_id": "HTL-IND-002",
                "name": "Goa Coast Resort",
                "nightly_rate_usd": 130,
                "tags": ["beach", "relaxing"],
            },
        ],
        "booking": {
            "booking_id": "BOOK-IND-1001",
            "hotel_id": "HTL-IND-001",
        },
    },
}

REFUNDS = {
    "BOOK-BALI-1001": {"refund_id": "REF-BALI-5001", "status": "processed"},
    "BOOK-PARIS-1001": {"refund_id": "REF-PARIS-5001", "status": "processed"},
    "BOOK-TOKYO-1001": {"refund_id": "REF-TOKYO-5001", "status": "processed"},
    "BOOK-SG-1001": {"refund_id": "REF-SG-5001", "status": "processed"},
    "BOOK-IND-1001": {"refund_id": "REF-IND-5001", "status": "processed"},
}


class ConciergeState(TypedDict):
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    intent: NotRequired[str]
    location: NotRequired[str]
    preference: NotRequired[str]
    booking_id: NotRequired[str]
    clarification: NotRequired[str]
    response: NotRequired[str]


class ConciergeRequest(BaseModel):
    intent: Literal["support", "booking", "refund", "clarify"] = Field(
        description="The single best action for the concierge to take."
    )
    location: str = Field(
        default="",
        description="A supported destination mentioned by the user, otherwise empty.",
    )
    preference: str = Field(
        default="",
        description="Hotel preference such as budget, beach, quiet, or luxury.",
    )
    booking_id: str = Field(
        default="",
        description="Booking ID mentioned by the user, otherwise empty.",
    )
    clarification: str = Field(
        default="",
        description="A short question if the request is unclear, otherwise empty.",
    )


@tool
def get_hotel_recommendations(location: str, preference: str = "") -> str:
    """Find hotels for a location, optionally matching tags like budget, beach, or quiet."""
    key = location.strip().casefold()
    destination = DESTINATIONS.get(key)
    if destination is None:
        available = ", ".join(item["name"] for item in DESTINATIONS.values())
        return f"No hotel list for {location!r}. Available locations: {available}."

    hotels = destination["hotels"]
    preference_tags = {
        tag
        for tag in ("budget", "luxury", "quiet", "beach", "central", "nature", "heritage")
        if tag in preference.casefold()
    }
    if preference_tags:
        matches = [
            hotel
            for hotel in hotels
            if preference_tags.intersection(hotel["tags"])
        ]
        if matches:
            hotels = matches

    return "\n".join(
        f"{hotel['name']} (hotel ID: {hotel['hotel_id']}, "
        f"${hotel['nightly_rate_usd']}/night; tags: {', '.join(hotel['tags'])})"
        for hotel in hotels
    )


@tool
def lookup_booking(location: str) -> str:
    """Look up the sample booking and hotel ID for a destination."""
    key = location.strip().casefold()
    destination = DESTINATIONS.get(key)
    if destination is None:
        available = ", ".join(item["name"] for item in DESTINATIONS.values())
        return f"No sample booking for {location!r}. Available locations: {available}."

    booking = destination["booking"]
    hotel = next(
        hotel
        for hotel in destination["hotels"]
        if hotel["hotel_id"] == booking["hotel_id"]
    )
    return (
        f"Sample booking for {destination['name']}: {hotel['name']}, "
        f"hotel ID {hotel['hotel_id']}, booking ID {booking['booking_id']}, "
        f"${hotel['nightly_rate_usd']} per night. No real reservation was made."
    )


@tool
def lookup_refund(booking_id: str) -> str:
    """Look up a sample refund ID using a booking ID."""
    key = booking_id.strip().upper()
    if not key:
        examples = ", ".join(REFUNDS)
        return f"Provide a booking ID. Sample booking IDs: {examples}."
    refund = REFUNDS.get(key)
    if refund is None:
        return f"No sample refund record was found for booking {key!r}."
    return (
        f"Sample refund for booking {key}: refund ID {refund['refund_id']}, "
        f"status {refund['status']}. No real refund was processed."
    )


def build_concierge_workflow(model: ChatGroq):
    """Build a LangGraph workflow with an LLM orchestrator and specialist agents."""
    route_model = model.with_structured_output(ConciergeRequest)
    support_agent = create_agent(
        model=model,
        tools=[get_hotel_recommendations],
        system_prompt=(
            "You are the hotel support specialist. Use get_hotel_recommendations "
            "for hotel facts and recommendations. Do not invent hotels, IDs, or prices."
        ),
    )
    booking_agent = create_agent(
        model=model,
        tools=[lookup_booking],
        system_prompt=(
            "You are the booking specialist. Use lookup_booking to retrieve sample "
            "hotel and booking IDs. Never claim a real reservation was made."
        ),
    )
    refund_agent = create_agent(
        model=model,
        tools=[lookup_refund],
        system_prompt=(
            "You are the refund specialist. Use lookup_refund for refund status and "
            "IDs. Never claim a real refund was processed."
        ),
    )

    def orchestrator_node(state: ConciergeState) -> dict[str, str]:
        request = route_model.invoke(
            [
                (
                    "system",
                    "Classify the concierge request. Choose support for hotel "
                    "recommendations, booking for booking details, refund for "
                    "refund lookup, or clarify if the user's goal is unclear. "
                    "Extract the destination, hotel preference, and booking ID. "
                    "Use the conversation history to resolve follow-up requests "
                    "such as 'there' or 'that booking'. Reuse previous details only "
                    "when they clearly apply. Do not invent missing values; the "
                    "latest user message is the current request.",
                ),
                *state.get("messages", []),
            ]
        )
        return {
            "intent": request.intent,
            "location": request.location,
            "preference": request.preference,
            "booking_id": request.booking_id,
            "clarification": request.clarification,
        }

    def support_node(state: ConciergeState) -> dict[str, object]:
        result = support_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Customer request: {state['query']}\n"
                            f"Location: {state.get('location', '')}\n"
                            f"Preference: {state.get('preference', '')}"
                        ),
                    }
                ]
            }
        )
        response = str(result["messages"][-1].content)
        return {"response": response, "messages": [AIMessage(content=response)]}

    def booking_node(state: ConciergeState) -> dict[str, object]:
        result = booking_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Customer request: {state['query']}\n"
                            f"Location: {state.get('location', '')}"
                        ),
                    }
                ]
            }
        )
        response = str(result["messages"][-1].content)
        return {"response": response, "messages": [AIMessage(content=response)]}

    def refund_node(state: ConciergeState) -> dict[str, object]:
        result = refund_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Customer request: {state['query']}\n"
                            f"Booking ID: {state.get('booking_id', '')}"
                        ),
                    }
                ]
            }
        )
        response = str(result["messages"][-1].content)
        return {"response": response, "messages": [AIMessage(content=response)]}

    def clarification_node(state: ConciergeState) -> dict[str, object]:
        message = state.get("clarification", "").strip()
        if not message:
            message = (
                "Would you like hotel recommendations, booking details, "
                "or help with a refund?"
            )
        return {"response": message, "messages": [AIMessage(content=message)]}

    def route_request(
        state: ConciergeState,
    ) -> Literal["support_agent", "booking_agent", "refund_agent", "clarify"]:
        intent = state.get("intent")
        if intent == "support":
            return "support_agent"
        if intent == "booking":
            return "booking_agent"
        if intent == "refund":
            return "refund_agent"
        if intent == "clarify":
            return "clarify"
        raise ValueError(f"Unsupported concierge intent: {intent!r}.")

    workflow = StateGraph(ConciergeState)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("support_agent", support_node)
    workflow.add_node("booking_agent", booking_node)
    workflow.add_node("refund_agent", refund_node)
    workflow.add_node("clarify", clarification_node)
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_request,
        {
            "support_agent": "support_agent",
            "booking_agent": "booking_agent",
            "refund_agent": "refund_agent",
            "clarify": "clarify",
        },
    )
    workflow.add_edge("support_agent", END)
    workflow.add_edge("booking_agent", END)
    workflow.add_edge("refund_agent", END)
    workflow.add_edge("clarify", END)
    return workflow.compile(checkpointer=InMemorySaver())


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    concierge = build_concierge_workflow(model)
    config = {"configurable": {"thread_id": "concierge-session"}}
    print(
        "Concierge demo: ask for hotel recommendations, booking details, or a refund. "
        "Type 'exit' to quit."
    )

    while True:
        query = input("\nYou: ").strip()
        if query.casefold() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = concierge.invoke(
            {
                "query": query,
                "messages": [{"role": "user", "content": query}],
            },
            config=config,
        )
        print(f"Concierge: {result['response']}")


if __name__ == "__main__":
    main()
