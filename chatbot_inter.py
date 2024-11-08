import streamlit as st
import pandas as pd
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# Load saved model and data
vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = sp.load_npz('tfidf_matrix.npz')
df = pd.read_csv('translated_with_tfidf.csv')

# Function to get chatbot response
def get_chatbot_response(user_query):
    # Vectorize the user query
    user_query_vector = vectorizer.transform([user_query])

    # Calculate cosine similarity between user query and all dataset queries
    cosine_similarities = cosine_similarity(user_query_vector, tfidf_matrix)

    # Get the index of the most similar query (highest cosine similarity)
    most_similar_idx = np.argmax(cosine_similarities)

    # Retrieve the corresponding response from the 'translated_KccAns' column
    response = df.iloc[most_similar_idx]['KccAns']

    return response

# Streamlit App Layout
st.title("Chatbot - Agricultural Query Assistant")
st.write("Welcome to the Agricultural Query Assistant! You can ask questions related to farming, paddy production, etc.")

# Initialize session state for conversation if it doesn't exist
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# User input for chatbot
user_query = st.text_input("Ask your question here:")

if user_query:
    # Get response from the chatbot
    response = get_chatbot_response(user_query)
    
    # Add user input and chatbot response to the conversation history
    st.session_state['conversation'].append({"role": "user", "text": user_query})
    st.session_state['conversation'].append({"role": "bot", "text": response})

# Display the conversation history (show messages in a bottom-to-top order)
for message in reversed(st.session_state['conversation']):
    if message['role'] == 'user':
        st.chat_message("user").markdown(f"**You:** {message['text']}")
    else:
        st.chat_message("bot").markdown(f"**Chatbot:** {message['text']}")



