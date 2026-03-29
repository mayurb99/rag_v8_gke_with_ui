import streamlit as st
import requests

API_URL = "http://api-service/ask"   # Kubernetes service name

st.title("RAG Q&A System")

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        response = requests.post(
            API_URL,
            json={"question": query}
        )

        if response.status_code == 200:
            st.write("### Answer:")
            st.write(response.json()["answer"])
        else:
            st.error("Error calling API")