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
    if 'last_query' not in st.session_state:
        st.session_state['last_query'] = None

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
    st.markdown(
        '<h1><img src="https://bimwerxfea.com/AI/Boxlogosmall32.png" class="icon"/> BIMWERX Bob</h1>',
        unsafe_allow_html=True,
    )
    st.markdown('**Virtual web assistant**')

    # CSS to style the input textbox and history container
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
            margin-bottom: 120px; /* Space for the fixed input box */
        }
        .history-container {
            max-height: 200px;
            overflow: auto;
            margin-bottom: 20px;
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

    # Display conversation history in an expandable div
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if st.session_state['history']:
        history = "\n\n".join([f"Q: {exchange['question']}\nA: {exchange['answer']}" for exchange in st.session_state['history']])
        with st.expander("History"):
            st.text_area("Chat History", value=history, height=200, disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Container for the input box
    query = st.text_input('Ask a question:', key="query_input")

    # Process the query
    if query and st.button('Submit'):
        if st.session_state['last_query'] != query:
            response = qa_chain({"query": query})
            response_txt = response["result"]
            
            # Update conversation history
            st.session_state['history'].append({"question": query, "answer": response_txt})
            st.session_state['last_query'] = query

            # Clear the input box after submission
            st.experimental_rerun()

    # Display the latest response in a text area to maintain formatting consistency
    if st.session_state['history']:
        latest_response = st.session_state['history'][-1]['answer']
        st.text_area("Latest Response:", value=latest_response, height=200, disabled=True)

if __name__ == "__main__":
    main()
