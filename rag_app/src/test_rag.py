import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import retrieve
from src.llm import generate_answer

query = "Summarize this document in one short sentence"

docs = retrieve(query)

print("\nCHUNKS:\n")
for i, doc in enumerate(docs):
    print(f"\n--- {i+1} ---")
    print(doc.page_content[:200])

answer = generate_answer(query, docs)

print("\nANSWER:\n")
print(answer)
