import streamlit as st
import os
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from supabase import create_client

# Supabase configuration (replace with your details)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def load_retriever():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # Create the Supabase client
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Initialize the Supabase vector store directly
    supabase_vector_store = SupabaseVectorStore(
        client=client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    
    retriever = supabase_vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

def main():
    retriever = load_retriever()
    
    # Test query to verify the vector store is loaded correctly
    query = "What are the capabilities of GenFEA?"
    results = retriever.get_relevant_documents(query)
    
    # Print the results
    for i, result in enumerate(results):
        print(f"Result {i+1}:\n{result.page_content}\n")

if __name__ == "__main__":
    main()
