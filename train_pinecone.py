import os
from getpass import getpass
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from llama_parse import LlamaParse
from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from pinecone import Pinecone, ServerlessSpec
import pinecone

from datasets import load_dataset

# Environment variables for API keys
SUPABASE_URL = "XXXX"
SUPABASE_KEY = "XXXX"

PINECONE_KEY = "XXXX"
PINECONE_ENV = "us-east-1"

OPENAI_API_KEY = "XXXX"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_KEY)

# Create Pinecone index if it doesn't exist
index_name = 'documents'
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimension of text-embedding-ada-002
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENV  # Ensure this is the correct region for your setup
        )
    )

client = create_client(SUPABASE_URL, SUPABASE_KEY)

def parse_and_load_documents():
    instruction = """The provided document is information on BIMWERX's FEA software from their website.
     It contains information on products, product capabilities, cost, and where to find resources.
     Try to be precise while answering the questions, if unsure, do not make up responses, say 'I am not sure, let me connect you with a BIMWERX FEA person'"""

    parser = LlamaParse(
        api_key="XXXX",
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=5000,
    )

    llama_parse_documents = parser.load_data("./data/webcontent.pdf")
    parsed_doc = llama_parse_documents[0]

    document_path = Path("data/parsed_document.md")
    with document_path.open("a") as f:
        f.write(parsed_doc.text)

    loader = UnstructuredMarkdownLoader(document_path)
    loaded_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
    docs = text_splitter.split_documents(loaded_documents)

    return docs

def create_vector_store(docs):
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    # Convert document paths to strings for JSON serialization
    for doc in docs:
        if "source" in doc.metadata and isinstance(doc.metadata["source"], Path):
            doc.metadata["source"] = str(doc.metadata["source"])

    # Enable pgvector extension in Supabase (one-time setup)
    # You can achieve this through Supabase UI or by running the following SQL command:
    # CREATE EXTENSION IF NOT EXISTS vector;

    vector_store = SupabaseVectorStore.from_documents(
        documents=docs,
        embedding=embed,
        client=client,
        table_name="document2",
        query_name="match_documents3"
    )

    return vector_store

def main():
    docs = parse_and_load_documents()
    vector_store = create_vector_store(docs)
    print(f"Vector store created with {len(docs)} documents in Supabase.")

if __name__ == "__main__":
    main()
