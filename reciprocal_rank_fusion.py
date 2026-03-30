from collections import defaultdict
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

load_dotenv()

persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


class QueryVariations(BaseModel):
    queries: List[str]


original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}")

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
Original query: {original_query}
Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations):
    print(f"{i + 1}. {variation}")

retriever = db.as_retriever(search_kwargs={"k": 5})
all_retrieved_results = []

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")

    docs = retriever.invoke(query)
    all_retrieved_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")

    print("-" * 50)

print("\n" + "=" * 60)
print("Multi-Query Retrieval Complete!")


def reciprocal_rank_fusion(chunk_list, k=60, verbose=True):
    if verbose:
        print("\n" + "=" * 60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("=" * 60)
        print(f"\nUsing k={k}")

    rrf_scores = defaultdict(float)  # {chunk_content: rrf_score}
    all_unique_chunks = {}  # {chunk_content: actual_chunk_object}

    # For verbose
    chunk_id_map = {}
    chunk_counter = 1

    for chunk_index, chunks in enumerate(chunk_list, 1):
        if verbose:
            print(f"Processing query {chunk_index} results:")

        for position, chunk in enumerate(chunks, 1):
            chunk_content = chunk.page_content

            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]

            all_unique_chunks[chunk_content] = chunk

            position_score = 1 / (k + position)

            rrf_scores[chunk_content] += position_score

            if verbose:
                print(
                    f"    Position {position}: {chunk_id} + {position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})"
                )
                print(f"    Preview: {chunk_content[:80]}...")

    sorted_chunks = sorted(
        [
            (all_unique_chunks[chunk_content], rrf_score)
            for chunk_content, rrf_score in rrf_scores.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    if verbose:
        print(
            f"✅ RRF completed! Processed {len(sorted_chunks)} unique chunks from {len(chunk_list)} queries."
        )

    return sorted_chunks


chunks_with_rrf_scores = reciprocal_rank_fusion(
    all_retrieved_results, k=60, verbose=True
)

print("\n" + "=" * 60)
print("FINAL RRF RANKING")
print("=" * 60)

print(f"\nTop {min(10, len(chunks_with_rrf_scores))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(chunks_with_rrf_scores, 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(
    f"\n✅ RRF Complete! Fused {len(chunks_with_rrf_scores)} unique documents from {len(query_variations)} query variations."
)
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score")
print("   • Balanced fusion using k=60 for gentle position penalties")
