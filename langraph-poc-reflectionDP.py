import os
from typing import Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph


MAX_REVISIONS = 3


class ReflectionState(TypedDict):
    task: str
    draft: NotRequired[str]
    critique: NotRequired[str]
    revision_count: NotRequired[int]
    final_answer: NotRequired[str]


def build_reflection_graph(model: ChatGroq):
    """Build a draft -> critique -> revise loop with a bounded number of revisions."""

    def generate_draft(state: ReflectionState) -> dict[str, str | int]:
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a careful writer. Create a useful first draft that "
                        "directly addresses the user's task."
                    )
                ),
                HumanMessage(content=state["task"]),
            ]
        )
        return {
            "draft": str(response.content),
            "revision_count": 0,
        }

    def reflect(state: ReflectionState) -> dict[str, str]:
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Review the draft against the user's task. Identify specific "
                        "issues in accuracy, relevance, clarity, completeness, or "
                        "format. If it is already good, say 'No changes needed.' "
                        "Give actionable feedback only; do not rewrite the draft."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Task:\n{state['task']}\n\n"
                        f"Draft to review:\n{state['draft']}"
                    )
                ),
            ]
        )
        return {"critique": str(response.content)}

    def revise(state: ReflectionState) -> dict[str, str | int]:
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Improve the draft by addressing the reflection feedback. "
                        "Preserve correct information, follow the original task, "
                        "and return only the revised answer."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original task:\n{state['task']}\n\n"
                        f"Current draft:\n{state['draft']}\n\n"
                        f"Reflection feedback:\n{state['critique']}"
                    )
                ),
            ]
        )
        return {
            "draft": str(response.content),
            "revision_count": state.get("revision_count", 0) + 1,
        }

    def finish(state: ReflectionState) -> dict[str, str]:
        return {"final_answer": state["draft"]}

    def should_revise(state: ReflectionState) -> Literal["revise", "finish"]:
        critique = state.get("critique", "").casefold()
        if "no changes needed" in critique or "no issues" in critique:
            return "finish"
        if state.get("revision_count", 0) >= MAX_REVISIONS:
            return "finish"
        return "revise"

    graph = StateGraph(ReflectionState)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("reflect", reflect)
    graph.add_node("revise", revise)
    graph.add_node("finish", finish)
    graph.add_edge(START, "generate_draft")
    graph.add_edge("generate_draft", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_revise,
        {"revise": "revise", "finish": "finish"},
    )
    graph.add_edge("revise", "reflect")
    graph.add_edge("finish", END)
    return graph.compile()


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    task = input("What should the writer create or improve? ").strip()
    if not task:
        raise ValueError("Please enter a writing task.")

    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    app = build_reflection_graph(model)
    result = app.invoke({"task": task})
    print(f"\nFinal answer:\n{result['final_answer']}")
    print(f"\nRevisions made: {result.get('revision_count', 0)}")


if __name__ == "__main__":
    main()