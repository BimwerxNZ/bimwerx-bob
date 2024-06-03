
# Basic Python + Streamlit-powered AI (Groq) Chatbot on Custom Trained Data

## Overview
This guide provides a quick overview of creating a basic but functional AI chatbot that can be embedded into almost any website as a widget. The chatbot connects to a Supabase database vector store on custom trained content.

## Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BimwerxNZ/bimwerx-bob.git
   ```

2. **Obtain LlamaParse API key from [LlamaParse](https://cloud.llamaindex.ai).**

3. **Obtain Groq API key from [Groq](https://console.groq.com/keys).**

4. **Create a database on Supabase and add the SQL function for `match_documents`.**
   
5. **Obtain Supabase database, API, and Auth keys from [Supabase](https://supabase.com/).**

6. **Prepare training data (PDF) and place it in the data folder.**

7. **Update API keys and training references.**

8. **Run the training script:**
   ```bash
   python train.py
   ```

9. **Deploy on Streamlit (app.py).**

10. **Insert CSS and iframe into the existing HTML page.**

## Supabase SQL Steps

1. **Create an extension:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Create the table:**
   ```sql
   CREATE TABLE document_embeddings (
       id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
       content text,
       metadata jsonb,
       embedding vector(768)
   );
   ```

3. **Create the `match_documents` function:**
   ```sql
   CREATE OR REPLACE FUNCTION match_documents(
       query_embedding vector(768), 
       match_count int = 5
   )
   RETURNS TABLE(
       id uuid,
       content text,
       metadata jsonb,
       similarity float
   )
   LANGUAGE plpgsql
   AS $$
   BEGIN
       RETURN QUERY
       SELECT
           d.id,
           d.content,
           d.metadata,
           (1 - (d.embedding <=> query_embedding)) AS similarity
       FROM
           documents d
       ORDER BY
           d.embedding <=> query_embedding
       LIMIT
           match_count;
   END;
   $$;
   ```

## To Insert into an Existing Webpage

1. **Add CSS Style to the page:**
   ```html
   <style>
       /* Chatbot button styles */
       #chatbot-button {
           position: fixed;
           bottom: 10px;
           left: 10px;
           height: 50px;
           width: 50px;
           background-color: #007bff;
           color: white;
           border: none;
           border-radius: 50%;
           display: flex;
           justify-content: center;
           align-items: center;
           cursor: pointer;
           z-index: 1000;
       }

       /* Chatbot iframe styles */
       #chatbot-iframe-container {
           position: fixed;
           bottom: 70px;
           left: 10px;
           height: 0;
           width: 0;
           overflow: hidden; /* Ensure the content is hidden when minimized */
           transition: height 0.3s ease-in-out, width 0.3s ease-in-out; /* Explicitly define transitions for height and width */
           z-index: 1000;
       }

       #chatbot-iframe {
           width: 100%;
           height: 100%;
           border: none;
           border-radius: 10px;
       }

       /* Expanded iframe styles */
       #chatbot-iframe-container.expanded {
           height: 500px !important;
           width: 500px !important; /* Adjust width if needed */
       }
   </style>
   ```

2. **Add HTML to the body:**
   ```html
   <!-- Chatbot button -->
   <div id="chatbot-button">Chat</div>

   <!-- Chatbot iframe container -->
   <div id="chatbot-iframe-container">
       <iframe
           id="chatbot-iframe"
           src="[YOUR STREAMLIT APP URL HERE]?embed=true"
       ></iframe>
   </div>

   <script>
       // Get the chatbot button and iframe container elements
       const chatbotButton = document.getElementById('chatbot-button');
       const chatbotIframeContainer = document.getElementById('chatbot-iframe-container');

       // Add a click event listener to the chatbot button
       chatbotButton.addEventListener('click', () => {
           // Toggle the expanded class on the iframe container
           console.log('Chatbot button clicked'); // Debugging log
           chatbotIframeContainer.classList.toggle('expanded');
           console.log('Iframe container classes:', chatbotIframeContainer.className); // Debugging log
       });
   </script>
   ```
