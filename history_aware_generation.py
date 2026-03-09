from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
  persist_directory=persist_directory,
  embedding_function=embedding_model,
  collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(
  search_type="similarity_score_threshold",
  search_kwargs={
    "k": 5,
    "score_threshold": 0.4
  }
)

llm = ChatGoogleGenerativeAI(
  model="gemini-2.5-flash",
  temperature=0.2
)

chat_history = []

def ask_question(question):
  if chat_history:
    messages = [SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")] + chat_history + [HumanMessage(content=f"New question: {question}")]

    response = llm.invoke(messages)
    search_question = response.content
  else:
    search_question = question

  print(f"---Answering for this question: {search_question}---")
  relevant_docs = retriever.invoke(search_question)
  context = "\n".join([doc.page_content for doc in relevant_docs])

  raw_prompt = f"""
  You are a highly capable AI assistant. Answer the user's question using ONLY the provided context below.
  If the answer is not in the context, politely say "I don't know based on the provided documents." Do not use outside knowledge.

  Context:
  {context}

  User Query:
  {question}
  """

  messages = chat_history + [HumanMessage(content=raw_prompt)]

  ai_response = llm.invoke(messages)
  ai_message = ai_response.content

  chat_history.append(HumanMessage(content=question))
  chat_history.append(AIMessage(content=ai_message))

  print(f"Answer: {ai_message}")
  return ai_message

def start_chat():
  print("---Ask me questions. Type 'quit' or 'q' to exit---")

  while True:
    question = input("Ask question: ")

    if question.lower()=="quit" or question.lower()=="q":
      print("Goodbye!")
      break

    ask_question(question)

if __name__=="__main__":
  start_chat()