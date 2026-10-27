"""Index products.csv in a local Chroma database and answer product queries."""

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
CHROMA_DB_DIRECTORY = Path(__file__).with_name("products_chroma_db")
COLLECTION_NAME = "product_catalog"
RESULT_COUNT = 3

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the retrieved product records. "
            "Do not invent or alter product names, prices, reviews, or features. "
            "If the records do not contain the requested information, say so. "
            "Product data is a local sample catalog, not live inventory.",
        ),
        (
            "human",
            "Retrieved product records:\n{context}\n\n"
            "User question: {question}",
        ),
    ]
)


def load_product_documents(csv_path: Path | None = None) -> list[Document]:
    """Validate supported product catalog formats and convert rows to documents."""
    if csv_path is None:
        csv_path = PRODUCTS_CSV
        if not csv_path.is_file():
            alternate_csv = csv_path.parent / "skills" / "concise-writing" / "products.csv"
            if alternate_csv.is_file():
                csv_path = alternate_csv
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Product catalog not found: {csv_path}. Place products.csv beside "
            f"{Path(__file__).name} or under skills\\concise-writing."
        )

    documents: list[Document] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        sample = csv_file.read(4096)
        csv_file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(csv_file, dialect=dialect)
        columns: dict[str, str] = {}
        currency_from_header = ""
        for column in reader.fieldnames or []:
            normalized = column.strip().lstrip("#").strip().casefold()
            if normalized in {"name", "product name"}:
                columns["name"] = column
            elif normalized == "description":
                columns["description"] = column
            elif normalized == "review":
                columns["review"] = column
            elif normalized == "currency":
                columns["currency"] = column
            elif normalized == "price" or normalized.startswith("price ("):
                columns["price"] = column
                if normalized.startswith("price (") and normalized.endswith(")"):
                    currency_from_header = normalized[7:-1].upper()

        required_fields = {"name", "description", "price"}
        missing_fields = required_fields - columns.keys()
        if "currency" not in columns and not currency_from_header:
            missing_fields.add("currency")
        if missing_fields:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing_fields))}."
            )

        seen_names: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            product = {
                field: (row.get(column) or "").strip()
                for field, column in columns.items()
            }
            product["currency"] = (
                product.get("currency", "") or currency_from_header
            )
            if not all(product.get(field) for field in required_fields | {"currency"}):
                raise ValueError(
                    f"{csv_path} has an empty product field on row {row_number}."
                )

            normalized_name = " ".join(product["name"].casefold().split())
            if normalized_name in seen_names:
                raise ValueError(
                    f"{csv_path} contains duplicate product name "
                    f"{product['name']!r} on row {row_number}."
                )
            seen_names.add(normalized_name)

            try:
                price = float(product["price"].replace("$", "").replace(",", ""))
            except ValueError as error:
                raise ValueError(
                    f"{csv_path} has an invalid price on row {row_number}."
                ) from error
            if not math.isfinite(price) or price < 0:
                raise ValueError(
                    f"{csv_path} has an invalid price on row {row_number}."
                )

            content = [f"Product name: {product['name']}"]
            if product.get("review"):
                content.append(f"Review: {product['review']}")
            content.extend(
                [
                    f"Description: {product['description']}",
                    f"Price: {product['currency']} {price:.2f}",
                ]
            )
            documents.append(
                Document(
                    page_content="\n".join(content),
                    metadata={
                        "name": product["name"],
                        "currency": product["currency"],
                        "price": price,
                        "source_row": row_number,
                    },
                )
            )

    if not documents:
        raise ValueError(f"No product records were found in {csv_path}.")
    return documents


def index_products(
    csv_path: Path | None = None,
    persist_directory: Path = CHROMA_DB_DIRECTORY,
) -> Chroma:
    """Rebuild and persist the Chroma collection from the current product CSV."""
    documents = load_product_documents(csv_path)
    persist_directory.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )

    # This dedicated collection mirrors the CSV on every startup, avoiding
    # stale or duplicate entries when the catalog changes.
    existing_ids = vector_store.get()["ids"]
    if existing_ids:
        vector_store.delete(ids=existing_ids)
    vector_store.add_documents(
        documents,
        ids=[f"product-{index}" for index in range(len(documents))],
    )
    return vector_store


def retrieve_products(query: str, vector_store: Chroma) -> list[Document]:
    """Retrieve product documents most relevant to a user's natural-language query."""
    if not query.strip():
        raise ValueError("Please enter a product question.")
    return vector_store.similarity_search(query=query.strip(), k=RESULT_COUNT)


def answer_product_question(
    query: str,
    vector_store: Chroma,
    model: BaseChatModel,
) -> str:
    """Generate an answer grounded in the documents retrieved from Chroma."""
    retrieved_documents = retrieve_products(query, vector_store)
    if not retrieved_documents:
        return "No matching product information was found in the catalog."

    context = "\n\n---\n\n".join(
        document.page_content for document in retrieved_documents
    )
    response = model.invoke(
        ANSWER_PROMPT.invoke({"context": context, "question": query.strip()})
    )
    return str(response.content)


def main() -> None:
    """Build the local index, accept queries, and print grounded answers."""
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    vector_store = index_products()
    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    print(
        f"Indexed {len(load_product_documents())} products from {PRODUCTS_CSV.name} "
        f"into the local Chroma database at {CHROMA_DB_DIRECTORY}."
    )
    print("Ask about the product catalog. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except EOFError:
            print()
            break
        if query.casefold() in {"exit", "quit"}:
            break
        if not query:
            continue

        print(f"Answer: {answer_product_question(query, vector_store, model)}")


if __name__ == "__main__":
    main()