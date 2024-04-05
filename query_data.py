import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

from dotenv import load_dotenv
load_dotenv()
import os
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""




def main():
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        /* Background color */
        .stApp {
            background-color: #FFD300;
        }

        /* Chat message styles */
        .chat-message {
            color: black;
            padding: 10px;
            margin: 5px 0;
            border-radius: 25px;
        }

        .user-message {
            background-color: #000000;
            color: #FFD300;
            margin-left: auto;
            margin-right: 0;
        }

        .bot-message {
            background-color: #000000;
            color: #FFD300;
            margin-left: 0;
            margin-right: auto;
        }

        /* Input field and button styling */
        .stTextInput>div>div>input {
            color: black;
        }

        .stButton>button {
            color: black;
            background-color: #FFD300;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Welcome to the HealthAid!", anchor=None)
    st.write("How can I assist you today?", anchor=None)

    # Placeholder for messages
    messages = st.empty()

    query_text = st.text_input("Your question:", "", key="query_input")
    
    # Button to submit the question
    submit_button = st.button('Submit')

    if submit_button and query_text:
        # Integrating model logic
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        print(results)
        if len(results) == 0 or results[0][1] < 0.7:
            bot_response = "Unable to find matching results."
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            
            model = ChatOpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'))
            bot_response = model.predict(prompt)

        # Use a container to display messages
        with messages.container():
            # Displaying user query
            st.markdown(f'<div class="chat-message user-message">{query_text}</div>', unsafe_allow_html=True)
            # Displaying model response
            st.markdown(f'<div class="chat-message bot-message">{bot_response}</div>', unsafe_allow_html=True)

        # st.session_state.query_input = ""  # Clear input field after submission

if __name__ == "__main__":
    main()

