from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

character_text_splitter = CharacterTextSplitter(
  chunk_size=100,
  # separator=" ", # Default separator: \n\n and Other options include ["\n\n", "\n", ". ", " ", ""],
  chunk_overlap=0
)

chunks = character_text_splitter.split_text(tesla_text)

print("\n" + "=" * 60)
print("1. CHARACTER TEXT SPLITTER")
print("=" * 60)

for i, chunk in enumerate(chunks):
  print(f"Chunk {i+1}: ({len(chunk)} characters)")
  print(f"'{chunk}'")
  print()


print("\n" + "=" * 60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER")
print("=" * 60)

recursive_character_text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=100,
  separators=["\n\n", "\n", ". ", " ", ""], # Multiple Separaters
  chunk_overlap=0
)

chunks2 = recursive_character_text_splitter.split_text(tesla_text)

for i, chunk in enumerate(chunks2):
  print(f"Chunk {i+1}: ({len(chunk)} characters)")
  print(f"'{chunk}'")
  print()