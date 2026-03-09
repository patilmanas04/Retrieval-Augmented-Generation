import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="sources"):
  "Loads all the text files from the sources directory"
  print(f"Loading all the text documents from the {docs_path} directory...")

  loader = DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
  )

  documents = loader.load()

  # for i, doc in enumerate(documents):
  #   print(f"\nDocument: {i+1}")
  #   print(f"  Source: {doc.metadata["source"]}")
  #   print(f"  Content length: {len(doc.page_content)} characters")
  #   print(f"  Content preview: {doc.page_content[:100]}...")
  #   print(f"  Metadata: {doc.metadata}")

  return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
	"Split documents into smaller chunks with overlap"
	print("Splitting documents into chunks...")

	text_splitter = CharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap
	)
   
	chunks = text_splitter.split_documents(documents)
   
	# if chunks:
	# 	for i, chunk in enumerate(chunks[:5]):
	# 		print(f"--- Chunk {i+1} ---")
	# 		print(f"Source: {chunk.metadata["source"]}")
	# 		print(f"Length: {len(chunk.page_content)}")
	# 		print(f"Content-->")
	# 		print(chunk.page_content)
	# 		print("-"*50)
               
	# 	if len(chunks)>5:
	# 		print(f"... and {len(chunks)-5} more chunks.")
  
	print(f"Total {len(chunks)} are created.")

	return chunks

def create_vector_db(chunks, persist_directory="db/chroma_db"):
	"Create embeddings and persist vector database"
	print("Creating embeddings and storing in chroma database...")

	embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

	print("---Creating Vector Store---")
	vector_store = Chroma.from_documents(
		documents=chunks,
		embedding=embedding_model,
		persist_directory=persist_directory,
		collection_metadata={"hnsw:space": "cosine"}
	)
	print("---Finished creating vectore store---")

	print(f"Vector store has been successfully created and stored to {persist_directory}")

	return vector_store

def main():
  # 1. Loading all the text documents
  documents = load_documents()

  # Chunking the files
  chunks = split_documents(documents)

	# Creating embeddings and storing to Vector DB
  vector_db = create_vector_db(chunks)

if __name__ == "__main__":
  main()