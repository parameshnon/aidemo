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
    """Recommend hotels from the local catalog by destination and optional preference."""
    destination = DESTINATIONS.get(location.strip().casefold())
    if destination is None:
        choices = ", ".join(item["name"] for item in DESTINATIONS.values())
        return f"No catalog for {location!r}. Available destinations: {choices}."

    hotels = destination["hotels"]
    requested_tags = {
        tag
        for tag in ("budget", "luxury", "quiet", "beach", "central", "nature", "heritage")
        if tag in preference.casefold()
    }
    if requested_tags:
        matches = [
            hotel
            for hotel in hotels
            if requested_tags.intersection(hotel["tags"])
        ]
        if matches:
            hotels = matches

    return "\n".join(
        f"{hotel['name']} | hotel ID {hotel['hotel_id']} | "
        f"${hotel['price']}/night | {', '.join(hotel['tags'])}"
        for hotel in hotels
    )


@tool
def lookup_booking(location: str) -> str:
    """Look up a sample booking ID and hotel ID by destination."""
    destination = DESTINATIONS.get(location.strip().casefold())
    if destination is None:
        choices = ", ".join(item["name"] for item in DESTINATIONS.values())
        return f"No sample booking for {location!r}. Available destinations: {choices}."

    booking = destination["booking"]
    hotel = next(
        hotel
        for hotel in destination["hotels"]
        if hotel["hotel_id"] == booking["hotel_id"]
    )
    return (
        f"Sample booking {booking['booking_id']}: {hotel['name']} "
        f"(hotel ID {hotel['hotel_id']}, ${hotel['price']}/night). "
        "No real reservation was made."
    )


@tool
def lookup_refund(booking_id: str) -> str:
    """Look up a sample refund ID for a booking ID."""
    normalized_id = booking_id.strip().upper()
    refund_id = REFUNDS.get(normalized_id)
    if refund_id is None:
        return f"No sample refund record found for {normalized_id or 'empty booking ID'}."
    return (
        f"Sample refund for booking {normalized_id}: {refund_id} (processed). "
        "No real refund was processed."
    )


def build_concierge_agent(model: ChatGroq):
    register_harness_profile(
        "groq",
        HarnessProfile(
            excluded_tools=DEEP_AGENT_BUILTIN_TOOLS,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    return create_deep_agent(
        model=model,
        tools=[recommend_hotels, lookup_booking, lookup_refund],
        system_prompt=(
            "You are a hotel concierge deep agent. Use recommend_hotels for hotel "
            "suggestions, lookup_booking for booking and hotel IDs, and lookup_refund "
            "for refund IDs. Always get facts from the tools; never invent prices, "
            "hotels, or IDs. All records are samples and no real bookings or refunds "
            "are performed. If a location or booking ID is missing, ask the user."
        ),
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    agent = build_concierge_agent(
        ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    )
    config = {"configurable": {"thread_id": "deepagent-concierge-session"}}
    print(
        "Deep Agent concierge: ask for hotel recommendations, booking details, "
        "or refund status. Type 'exit' to quit."
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