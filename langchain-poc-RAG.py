import csv
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


PRODUCTS_CSV = Path(__file__).with_name("products.csv")
CHROMA_DB_DIRECTORY = Path(__file__).with_name("chroma_db")
CHROMA_COLLECTION_NAME = "products"
RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's question using only the product information in the "
            "provided context. Include the product name, price, review, and "
            "description when asked for product details. Do not invent or alter "
            "product facts. If the context does not contain the requested fact, "
            "say that it is not listed.",
        ),
        ("human", "Product information:\n{context}\n\nQuestion: {question}"),
    ]
)


def load_product_documents(csv_path: Path = PRODUCTS_CSV) -> list[Document]:
    """Load the product catalog as LangChain documents."""
    required_columns = {"name", "review", "description", "price", "currency"}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing_columns = required_columns - columns
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}."
            )

        documents = []
        for row_number, row in enumerate(reader, start=2):
            product = {key: (row[key] or "").strip() for key in required_columns}
            if not all(product.values()):
                raise ValueError(
                    f"{csv_path} has an empty product field on row {row_number}."
                )
            try:
                price = float(product["price"])
            except ValueError as error:
                raise ValueError(
                    f"{csv_path} has an invalid price on row {row_number}."
                ) from error
            if not math.isfinite(price) or price < 0:
                raise ValueError(
                    f"{csv_path} has an invalid price on row {row_number}."
                )

            content = (
                f"Product name: {product['name']}\n"
                f"Review: {product['review']}\n"
                f"Description: {product['description']}\n"
                f"Price: {product['currency']} {price:.2f}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={"name": product["name"], "row": row_number},
                )
            )

    if not documents:
        raise ValueError(f"No products were found in {csv_path}.")
    return documents


def create_product_vector_store(
    csv_path: Path = PRODUCTS_CSV,
    persist_directory: Path = CHROMA_DB_DIRECTORY,
) -> tuple[Chroma, list[str]]:
    """Rebuild and persist the Chroma product collection from the CSV."""
    documents = load_product_documents(csv_path)
    names = [document.metadata["name"] for document in documents]
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )
    existing_ids = vector_store.get()["ids"]
    if existing_ids:
        vector_store.delete(ids=existing_ids)
    vector_store.add_documents(
        documents,
        ids=[f"product-{index}" for index in range(len(documents))],
    )
    return vector_store, names


def find_product_name(query: str, product_names: list[str]) -> str | None:
    """Find a catalog name mentioned in a product-name query."""
    normalized_query = " ".join(query.casefold().split())
    matches = [
        name
        for name in product_names
        if " ".join(name.casefold().split()) in normalized_query
    ]
    return max(matches, key=len) if matches else None


def answer_product_question(
    query: str,
    vector_store: Chroma,
    product_names: list[str],
    model: BaseChatModel,
) -> str:
    """Retrieve a matching CSV product and generate an answer from its details."""
    if not query.strip():
        raise ValueError("Please enter a product name or question.")

    product_name = find_product_name(query, product_names)
    if product_name is None:
        available = ", ".join(product_names)
        return f"No matching product was found. Available products: {available}."

    retrieved_documents = vector_store.similarity_search(
        query=product_name,
        k=1,
        filter={"name": product_name},
    )
    if not retrieved_documents:
        raise RuntimeError(f"Could not retrieve product details for {product_name!r}.")

    prompt = RESPONSE_PROMPT.invoke(
        {
            "context": retrieved_documents[0].page_content,
            "question": query,
        }
    )
    response = model.invoke(prompt)
    return str(response.content)


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    vector_store, product_names = create_product_vector_store()
    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    print(
        "Ask about a product in products.csv. Type 'exit' to quit. "
        "The Chroma database is stored in langchain-poc/chroma_db."
    )
    while True:
        query = input("Product question: ").strip()
        if query.casefold() in {"exit", "quit"}:
            break
        if not query:
            continue
        print(answer_product_question(query, vector_store, product_names, model))


if __name__ == "__main__":
    main()