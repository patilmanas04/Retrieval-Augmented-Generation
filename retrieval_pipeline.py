import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
  persist_directory=persist_directory,
  embedding_function=embedding_model,
  collection_metadata={"hnsw:space": "cosine"}
)

query = "In what year does the Tesla begin it's production of the Roadster?"
# query = "What was NVIDIA's first graphics accelarator called?"
# query = "How much did microsoft paid to aquire Github?"

retriever = db.as_retriever(
  search_type="similarity_score_threshold",
  search_kwargs={
    "k": 5,
    "score_threshold": 0.3 # Only return chunks with cosine similarity >= 0.4
  }
)

relavant_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in relavant_docs])

raw_prompt = f"""
You are a highly capable AI assistant. Answer the user's question using ONLY the provided context below.
If the answer is not in the context, politely say "I don't know based on the provided documents." Do not use outside knowledge.

Context:
{context}

User Query:
{query}
"""

print("--- The Raw Prompt being sent to Gemini ---")
print(raw_prompt)
print("-------------------------------------------\n")

llm = ChatGoogleGenerativeAI(
  model="gemini-2.5-flash-lite",
  temperature=0.2
)

print("Thinking...")

response = llm.invoke(raw_prompt)

print("\n" + "="*50)
print(f"Final Answer: \n{response.content}")
print("="*50)