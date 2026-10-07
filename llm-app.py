import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


class GeneratedContent(BaseModel):
    title: str = Field(description="A concise title for the generated content.")
    content: str = Field(description="The requested content.")
    summary: str = Field(description="A one- or two-sentence summary.")
    key_points: list[str] = Field(
        description="Important points covered in the generated content.its should be two lines only"
    )


def main() -> None:
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    prompt = input("What would you like the model to write? ").strip()
    if not prompt:
        raise ValueError("Please enter a prompt.")

    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    structured_model = model.with_structured_output(GeneratedContent)
    response = structured_model.invoke(
        f"Generate content for this request and return all required structured fields: {prompt}"
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()