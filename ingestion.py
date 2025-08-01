# INGESTION.PY for MindMate (Local Embeddings with Ollama-compatible model)

import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Setup Pinecone (still using cloud vector DB)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # for MiniLM
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Local embeddings 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load PDF

loader = PyPDFDirectoryLoader("document")
raw_documents = loader.load()

# Debug: print a sample chunk
print(f"Loaded {len(raw_documents)} raw documents")
for doc in raw_documents[:2]:
    print(doc.page_content[:200])


# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.split_documents(raw_documents)

# Assign UUIDs
uuids = [f"id{i}" for i in range(1, len(documents) + 1)]

# Store in vector DB
vector_store.add_documents(documents=documents, ids=uuids)
print("✅ Documents ingested successfully.")
