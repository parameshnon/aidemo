# Retrieval-Augmented Generation (RAG)

## What RAG is

Retrieval-augmented generation (RAG) combines information retrieval with
language-model generation. Rather than relying only on what the model learned
during training, an application retrieves relevant source material and
provides it as context for a response.

A typical RAG flow is:

1. Load source data.
2. Clean and validate it.
3. Split or transform it into documents, when appropriate.
4. Create embeddings and store them in a vector store.
5. Retrieve relevant documents for a user query.
6. Pass the retrieved context and question to a language model.
7. Return an answer grounded in the supplied context.

The repository's product example uses one document per CSV product instead of
chunking long documents, because each product row is already a small,
self-contained record.

## Repository example

The implementation is in `langchain-poc\langchain-poc-RAG.py`, and the source
catalog is `langchain-poc\products.csv`.

The sample:

- Validates that the CSV has `name`, `review`, `description`, `price`, and
  `currency` columns.
- Rejects empty fields, invalid prices, negative prices, and non-finite prices.
- Converts each row into a LangChain `Document`.
- Builds a persistent Chroma collection named `products` in
  `langchain-poc\chroma_db`.
- Resolves the product name from the user's question, then retrieves the
  matching product document.
- Sends the product facts and question to a Groq chat model with an instruction
  to answer from context and not invent missing details.

## Run the sample

From the repository root in PowerShell:

```powershell
uv run python langchain-poc\langchain-poc-RAG.py
```

Set `GROQ_API_KEY` in the environment or a local `.env` file before running.
Ask about a product included in `products.csv`, for example:

```text
What is the price and description of the AeroBook 14?
```

The sample recreates the product collection from the CSV at startup. The Chroma
directory is local application data, not a source of truth; the CSV remains the
catalog input for this example.

## Retrieval versus generation

Retrieval chooses source documents; generation produces a natural-language
answer. These are separate stages and should be evaluated separately:

- If the wrong document is retrieved, improve query normalization, metadata
  filters, chunking, or retrieval strategy.
- If the correct document is retrieved but the answer is wrong, improve prompt
  constraints, output validation, or model selection.

The current sample first matches a known product name in the user's query and
then performs a Chroma similarity search filtered to that product's `name`.
This makes product lookup precise for the fixed catalog, but it is not a
general semantic search across arbitrary product questions.

## Embeddings and vector stores

Vector stores index documents by embeddings so semantically related content can
be retrieved. The exact embedding configuration is part of the vector-store
setup and must remain compatible when indexing and querying. This example lets
Chroma use its configured/default embedding behavior; for a production system,
select and configure an embedding model explicitly, record its version, and
rebuild the index when the embedding model or source content changes.

Metadata can support filtering and traceability. In this sample each document
records the product `name` and source CSV row number.

## Grounding and limitations

The system prompt says to answer using only the retrieved product information
and to say when requested facts are not listed. This helps constrain the
response, but a prompt is not a factual guarantee. Test answers against source
records and expose citations or product identifiers when users need to verify
claims.

The source CSV is a small, fixed demo catalog. It is not live inventory,
pricing, or availability data. Do not represent its values as current external
facts.

## Production considerations

- Keep a reproducible source of truth and track index versions.
- Define an update strategy: full rebuild, incremental update, or scheduled
  synchronization.
- Avoid deleting or replacing a live index until a new index is validated.
- Use explicit embedding configuration and monitor embedding/API failures.
- Measure retrieval precision and recall with representative test queries.
- Return “not found” when evidence is absent instead of generating a guess.
- Consider source access controls before retrieving documents for different
  users.
- Log retrieval IDs and outcome metadata carefully; retrieved documents may
  contain sensitive information.
- Treat indexed content as untrusted data. Do not execute instructions found in
  retrieved documents.
