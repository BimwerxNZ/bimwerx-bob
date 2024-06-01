import os
import textwrap
from loadsupabase import load_retriever  # Import the load_retriever function from load.py
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from rich.console import Console
from rich.markdown import Markdown

os.environ["GROQ_API_KEY"] = "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"

console = Console()

def print_response(response):
    response_txt = response["result"]
    markdown_response = Markdown(response_txt)
    console.print(markdown_response)

def main():
    retriever = load_retriever()

    # Initialize ChatGroq
    #llama3-70b-8192
    #mixtral-8x7b-32768
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant for BIMWERX. Use the following context to answer the question:
        {context}
        Question: {question}
        """,
    )

    # Setup RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    print("Welcome to the BIMWERX Chatbot! Type 'exit' to quit.")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = qa_chain({"query": query})
        print_response(response)

if __name__ == "__main__":
    main()
