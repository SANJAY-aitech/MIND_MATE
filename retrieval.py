import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# Pinecone setup
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Retrieval
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},  # Lowered threshold
)

query = "I am so stressed out, I need some help with my mental health. What can I do?"
results = retriever.invoke(query)

# Print results
print(f"\nQuery: {query}")
if not results:
    print("No relevant documents found. Try re-ingesting or lowering threshold.")
else:
    print("\nRESULTS:")
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
