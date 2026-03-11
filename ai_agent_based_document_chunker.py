from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

system_instructions = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

TEXT:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split.
"""

response = llm.invoke(system_instructions)
marked_text = response.content

splitted_text = marked_text.split("<<<SPLIT>>>")

cleaned_chunks = []
for uncleaned_chunk in splitted_text:
  cleaned_chunk = uncleaned_chunk.strip()
  if cleaned_chunk:
    cleaned_chunks.append(cleaned_chunk)

print("===AI AGENT BASED DOCUMENT CHUNKING===")
for i, chunk in enumerate(cleaned_chunks):
  print(f"Chunk {i+1}: ({len(chunk)})")
  print(f"'{chunk}'")
  print()