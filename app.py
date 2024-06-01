import streamlit as st
import os
from loadsupabase import load_retriever  # Import the load_retriever function from load.py
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Set the GROQ API key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def main():
    # Initialize session state for conversation history if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    retriever = load_retriever()

    # Initialize ChatGroq
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

    # Streamlit UI
    st.title('BIMWERX Chatbot')
    query = st.text_input('Ask a question:')
    
    # Display conversation history
    for exchange in st.session_state['history']:
        st.text_area("Q:", value=exchange['question'], height=50, disabled=True)
        st.text_area("A:", value=exchange['answer'], height=100, disabled=True)

    if query:
        response = qa_chain({"query": query})
        response_txt = response["result"]
        
        # Update conversation history
        st.session_state['history'].append({"question": query, "answer": response_txt})
        
        # Display the latest response
        st.text_area("A:", value=response_txt, height=100, disabled=True)

if __name__ == "__main__":
    main()
