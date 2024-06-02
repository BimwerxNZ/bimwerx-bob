import streamlit as st
import os
from load import load_retriever  # Import the load_retriever function from load.py
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
        You are an AI assistant for BIMWERX. Avoid referring to 'context' in your responses, instead use 'knowledge', but only when required.
        Respond with bulleted points when listing response content.
        Never make up answers, if unsure, say: 'I am not sure, let me connect you with a BIMWERX person'.
        Only answer questions related to the context, if the question is out of scope, say: 'I am not sure, let me connect you with a BIMWERX person'.
        Use the following context to answer the question:
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
    st.title('BIMWERX Bob')
    st.markdown('**Virtual web assistant**')

    # CSS to fix input textbox at the bottom
    st.markdown(
        """
        <style>
        .input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f9f9f9;
            padding: 10px 0;
            z-index: 1000;
        }
        .chat-container {
            margin-bottom: 100px; /* Space for the fixed input box */
        }
        .icon {
            width: 32px;
            vertical-align: middle;
            margin-right: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display conversation history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, exchange in enumerate(st.session_state['history']):
        st.text_area(f"Q_{i}", value=exchange['question'], height=50, disabled=True)
        st.markdown(f'<img src="https://bimwerxfea.com/AI/Boxlogosmall32.png" class="icon"/>', unsafe_allow_html=True)
        st.text_area(f"A_{i}", value=exchange['answer'], height=100, disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Container for the input box
    with st.container():
        query = st.text_input('Ask a question:', key="query_input")

    # Process the query
    if query and st.session_state.get('last_query') != query:
        response = qa_chain({"query": query})
        response_txt = response["result"]
        
        # Update conversation history
        st.session_state['history'].append({"question": query, "answer": response_txt})
        st.session_state['last_query'] = query

        # Clear the input box after submission
        st.experimental_rerun()

if __name__ == "__main__":
    main()
