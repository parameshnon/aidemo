import os

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver


DESTINATIONS = {
    "bali": {
        "name": "Bali",
        "hotels": [
            {
                "hotel_id": "HTL-BALI-001",
                "name": "Palm Garden Resort",
                "price_usd": 145,
                "tags": ["beach", "luxury"],
            },
            {
                "hotel_id": "HTL-BALI-002",
                "name": "Ubud Valley Hotel",
                "price_usd": 110,
                "tags": ["quiet", "nature", "budget"],
            },
        ],
        "booking": {"booking_id": "BOOK-BALI-1001", "hotel_id": "HTL-BALI-001"},
    },
    "paris": {
        "name": "Paris",
        "hotels": [
            {
                "hotel_id": "HTL-PARIS-001",
                "name": "Left Bank Boutique Hotel",
                "price_usd": 220,
                "tags": ["central", "luxury"],
            },
            {
                "hotel_id": "HTL-PARIS-002",
                "name": "Montmartre Stay",
                "price_usd": 160,
                "tags": ["quiet", "budget"],
            },
        ],
        "booking": {"booking_id": "BOOK-PARIS-1001", "hotel_id": "HTL-PARIS-001"},
    },
    "tokyo": {
        "name": "Tokyo",
        "hotels": [
            {
                "hotel_id": "HTL-TOKYO-001",
                "name": "Shinjuku City Hotel",
                "price_usd": 175,
                "tags": ["central", "nightlife"],
            },
            {
                "hotel_id": "HTL-TOKYO-002",
                "name": "Asakusa Riverside Inn",
                "price_usd": 125,
                "tags": ["quiet", "budget"],
            },
        ],
        "booking": {"booking_id": "BOOK-TOKYO-1001", "hotel_id": "HTL-TOKYO-001"},
    },
    "singapore": {
        "name": "Singapore",
        "hotels": [
            {
                "hotel_id": "HTL-SG-001",
                "name": "Marina Bay City Hotel",
                "price_usd": 210,
                "tags": ["central", "luxury"],
            },
            {
                "hotel_id": "HTL-SG-002",
                "name": "Garden District Stay",
                "price_usd": 155,
                "tags": ["quiet", "nature"],
            },
        ],
        "booking": {"booking_id": "BOOK-SG-1001", "hotel_id": "HTL-SG-001"},
    },
    "india": {
        "name": "India",
        "hotels": [
            {
                "hotel_id": "HTL-IND-001",
                "name": "Jaipur Heritage Hotel",
                "price_usd": 95,
                "tags": ["heritage", "budget"],
            },
            {
                "hotel_id": "HTL-IND-002",
                "name": "Goa Coast Resort",
                "price_usd": 130,
                "tags": ["beach", "relaxing"],
            },
        ],
        "booking": {"booking_id": "BOOK-IND-1001", "hotel_id": "HTL-IND-001"},
    },
}

DEEP_AGENT_BUILTIN_TOOLS = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "task",
    }
)


@tool
def recommend_hotels(location: str, preference: str = "") -> str:
    """Recommend catalog hotels at a location, optionally matching budget, beach, quiet, or luxury."""
    destination = DESTINATIONS.get(location.strip().casefold())
    if destination is None:
        available = ", ".join(item["name"] for item in DESTINATIONS.values())
        return f"No hotel catalog for {location!r}. Available destinations: {available}."

    hotels = destination["hotels"]
    preferences = {
        tag
        for tag in (
            "budget",
            "beach",
            "quiet",
            "luxury",
            "central",
            "nature",
            "heritage",
        )
        if tag in preference.casefold()
    }
    if preferences:
        matching = [
            hotel
            for hotel in hotels
            if preferences.intersection(hotel["tags"])
        ]
        if matching:
            hotels = matching

    return "\n".join(
        f"{hotel['name']} (hotel ID {hotel['hotel_id']}, "
        f"${hotel['price_usd']}/night; {', '.join(hotel['tags'])})"
        for hotel in hotels
    )


@tool
def get_booking_details(location: str) -> str:
    """Look up sample booking and hotel details for a destination."""
    destination = DESTINATIONS.get(location.strip().casefold())
    if destination is None:
        available = ", ".join(item["name"] for item in DESTINATIONS.values())
        return f"No sample booking for {location!r}. Available destinations: {available}."

    booking = destination["booking"]
    hotel = next(
        item
        for item in destination["hotels"]
        if item["hotel_id"] == booking["hotel_id"]
    )
    return (
        f"Sample booking ID: {booking['booking_id']}; destination: "
        f"{destination['name']}; hotel: {hotel['name']}; "
        f"hotel ID: {hotel['hotel_id']}; price: ${hotel['price_usd']}/night. "
        "This is a demo record; no real reservation was made."
    )


def build_concierge_agent(model: ChatGroq):
    """Create a Deep Agent that delegates support and booking to specialist subagents."""
    register_harness_profile(
        "groq",
        HarnessProfile(
            excluded_tools=DEEP_AGENT_BUILTIN_TOOLS,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    return create_deep_agent(
        model=model,
        tools=[],
        subagents=[
            {
                "name": "hotel-support",
                "description": (
                    "Handles hotel recommendations, destination options, and "
                    "preferences such as budget, quiet, or beach."
                ),
                "system_prompt": (
                    "You are the concierge hotel support specialist. Use the "
                    "recommend_hotels tool for all hotel names, IDs, prices, and "
                    "recommendations. Never invent catalog details."
                ),
                "tools": [recommend_hotels],
            },
            {
                "name": "hotel-booking",
                "description": (
                    "Looks up sample booking IDs and the associated hotel by destination."
                ),
                "system_prompt": (
                    "You are the concierge booking specialist. Use "
                    "get_booking_details for booking and hotel facts. Clearly state "
                    "that records are samples and do not represent real reservations."
                ),
                "tools": [get_booking_details],
            },
        ],
        system_prompt=(
            "You are a concierge coordinator. Delegate hotel recommendation and "
            "preference requests to hotel-support. Delegate booking lookups to "
            "hotel-booking. Extract the destination and relevant preference from "
            "the user's request; ask a concise follow-up if required information "
            "is missing. Do not invent hotel facts, prices, or IDs. All booking "
            "data is sample data; do not claim to create a real reservation."
        ),
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file."
        )

    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    agent = build_concierge_agent(model)
    config = {"configurable": {"thread_id": "concierge-subagent-session"}}

    print(
        "Concierge Deep Agent ready. Ask for hotel recommendations or booking "
        "details. Type 'exit' to quit."
    )
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
        print(f"Concierge: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()