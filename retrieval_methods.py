from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hsnw:space": "cosine"},
)

query = "How much did Microsoft pay to acquire GitHub?"
print(f"Query: {query}")

# ──────────────────────────────────────────────────────────────────
# METHOD 1: Basic Similarity Search
# Returns the top k most similar documents
# ──────────────────────────────────────────────────────────────────

print("=== METHOD 1: Similarity Search (k=3) ===")
retriever = db.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"Document: {i + 1}")
    print(f"Page Content: {doc.page_content}")

print("=" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 2: Similarity with Score Threshold
# Only returns documents above a certain similarity score
# ──────────────────────────────────────────────────────────────────

print(
    "=== METHOD 2: Similarity Search with Score Threshold (k=3 and score_threshold=0.3) ==="
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3},
)

retrieved_docs = retriever.invoke(query)
for i, doc in enumerate(retrieved_docs):
    print(f"Document: {i + 1}")
    print(f"Content: {doc.page_content}")

print("=" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 3: Maximum Marginal Relevance (MMR)
# Balances relevance and diversity - avoids redundant results
# ──────────────────────────────────────────────────────────────────

print("=== METHOD 3: Maximum Marginal Relevance (MMR) ===")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,  # Initial pool to select from
        "lamda_mult": 0.5,  # 0=max diversity, 1=max relevance
    },
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (λ=0.5):\n")

for i, doc in enumerate(docs):
    print(f"Document: {i + 1}")
    print(f"Content: {doc.page_content}")

print("=" * 60)
