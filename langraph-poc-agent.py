from typing import NotRequired, TypedDict
from uuid import uuid4

from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph


VACATIONS = {
    "bali": {
        "destination": "Bali",
        "duration_days": 7,
        "description": "A week of beaches, temples, and Ubud rice terraces.",
        "activities": ["Uluwatu Temple", "Ubud", "Seminyak Beach"],
    },
    "paris": {
        "destination": "Paris",
        "duration_days": 5,
        "description": "A city break for museums, landmarks, and French cuisine.",
        "activities": ["Eiffel Tower", "Louvre Museum", "Seine cruise"],
    },
    "tokyo": {
        "destination": "Tokyo",
        "duration_days": 6,
        "description": "Explore Tokyo's food, neighborhoods, and cultural sites.",
        "activities": ["Asakusa", "Shibuya", "Meiji Shrine"],
    },
    "singapore": {
        "destination": "Singapore",
        "duration_days": 4,
        "description": "Enjoy a city break with gardens, waterfront views, and local food.",
        "activities": ["Gardens by the Bay", "Marina Bay", "Chinatown"],
    },
    "india": {
        "destination": "India",
        "duration_days": 8,
        "description": "Explore India's heritage, colorful markets, and regional cuisine.",
        "activities": ["Jaipur's historic center", "Amber Fort", "Local markets"],
    },
}

HOTELS = {
    "bali": [
        {"name": "Palm Garden Resort", "nightly_rate_usd": 145},
        {"name": "Ubud Valley Hotel", "nightly_rate_usd": 110},
    ],
    "paris": [
        {"name": "Left Bank Boutique Hotel", "nightly_rate_usd": 220},
        {"name": "Montmartre Stay", "nightly_rate_usd": 160},
    ],
    "tokyo": [
        {"name": "Shinjuku City Hotel", "nightly_rate_usd": 175},
        {"name": "Asakusa Riverside Inn", "nightly_rate_usd": 125},
    ],
    "singapore": [
        {"name": "Marina Bay City Hotel", "nightly_rate_usd": 210},
        {"name": "Garden District Stay", "nightly_rate_usd": 155},
    ],
    "india": [
        {"name": "Jaipur Heritage Hotel", "nightly_rate_usd": 95},
        {"name": "Goa Coast Resort", "nightly_rate_usd": 130},
    ],
}


class VacationState(TypedDict):
    destination: str
    hotel_name: str
    vacation: NotRequired[dict[str, object]]
    available_hotels: NotRequired[list[dict[str, str | int]]]
    selected_hotel: NotRequired[dict[str, str | int]]
    booking_id: NotRequired[str]
    confirmation: NotRequired[str]


def plan_vacation(state: VacationState) -> dict[str, object]:
    destination_key = state["destination"].strip().casefold()
    vacation = VACATIONS.get(destination_key)
    if vacation is None:
        available = ", ".join(item["destination"] for item in VACATIONS.values())
        raise ValueError(f"Unknown destination. Available vacation options: {available}.")
    return {"vacation": vacation}


def find_hotels(state: VacationState) -> dict[str, object]:
    destination_key = state["destination"].strip().casefold()
    available_hotels = HOTELS.get(destination_key, [])
    if not available_hotels:
        raise ValueError(f"No hotels are available for {state['destination']!r}.")

    requested_hotel = " ".join(state["hotel_name"].casefold().split())
    selected_hotel = next(
        (
            hotel
            for hotel in available_hotels
            if " ".join(hotel["name"].casefold().split()) == requested_hotel
        ),
        None,
    )
    if selected_hotel is None:
        choices = ", ".join(hotel["name"] for hotel in available_hotels)
        raise ValueError(
            f"Hotel {state['hotel_name']!r} is not available in "
            f"{state['destination']}. Choose from: {choices}."
        )
    return {
        "available_hotels": available_hotels,
        "selected_hotel": selected_hotel,
    }


def book_hotel(state: VacationState) -> dict[str, str]:
    vacation = state["vacation"]
    hotel = state["selected_hotel"]
    if vacation is None or hotel is None:
        raise ValueError("Vacation and hotel must be selected before booking.")

    booking_id = f"DEMO-{uuid4().hex[:10].upper()}"
    confirmation = (
        f"Demo booking confirmed: {hotel['name']} in {vacation['destination']}, "
        f"{vacation['duration_days']} days at "
        f"${hotel['nightly_rate_usd']} per night."
    )
    return {"booking_id": booking_id, "confirmation": confirmation}


def build_graph():
    graph = StateGraph(VacationState)
    graph.add_node("plan_vacation", plan_vacation)
    graph.add_node("find_hotels", find_hotels)
    graph.add_node("book_hotel", book_hotel)
    graph.add_edge(START, "plan_vacation")
    graph.add_edge("plan_vacation", "find_hotels")
    graph.add_edge("find_hotels", "book_hotel")
    graph.add_edge("book_hotel", END)
    return graph.compile()


def main() -> None:
    destinations = ", ".join(
        item["destination"] for item in VACATIONS.values()
    )
    destination = input(
        f"Choose a destination ({destinations}): "
    ).strip()
    destination_key = destination.casefold()
    options = HOTELS.get(destination_key)
    if options is None:
        raise ValueError(f"Unknown destination. Choose from: {destinations}.")

    print(f"\nHotels available in {destination.title()}:")
    for hotel in options:
        print(f"- {hotel['name']}: ${hotel['nightly_rate_usd']} per night")
    hotel_name = input("\nChoose a hotel by name: ").strip()

    app = build_graph()
    result = app.invoke({"destination": destination, "hotel_name": hotel_name})
    vacation = result["vacation"]
    print(f"\nVacation plan: {vacation['description']}")
    print(f"Suggested activities: {', '.join(vacation['activities'])}")
    print(result["confirmation"])
    print(f"Booking ID: {result['booking_id']}")
    print("This is a simulated booking; no real hotel reservation was made.")

    graph_image = app.get_graph().draw_mermaid_png()
    display(Image(graph_image))


if __name__ == "__main__":
    main()