import streamlit as st
import requests

st.set_page_config(page_title="Bot Response Comparison", page_icon="🤖", layout="wide")

RETRIEVAL_API_URL = "http://127.0.0.1:8000/predict-retrieval"
GENERATIVE_API_URL = "http://127.0.0.1:8000/predict-generative"

st.title("🤖 Bot Response Model Comparison")
st.markdown("Enter a tweet to compare a **safe, retrieval-based bot** with a **creative, generative bot**.")

user_input = st.text_area("Enter a sample tweet:", "This new feature is absolutely amazing, thank you!", height=100)

if st.button("Generate Bot Responses"):
    if user_input:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Retrieval-Based Bot")
            st.info("Classifies sentiment and pulls a pre-written, safe response.")
            with st.spinner("Analyzing sentiment..."):
                try:
                    response = requests.post(RETRIEVAL_API_URL, json={"text": user_input})
                    response.raise_for_status()
                    result = response.json()
                    sentiment, reply = result.get("sentiment", "N/A"), result.get("reply", "N/A")
                    st.write("**Detected Sentiment:**")
                    st.success(f"**{sentiment}**") if sentiment == "Positive" else st.error(f"**{sentiment}**")
                    st.write("**Bot Reply:**")
                    st.success(f"_{reply}_")
                except requests.exceptions.RequestException as e:
                    st.error(f"API Connection Error: {e}")

        with col2:
            st.subheader("Generative Bot")
            st.warning("Generates a new response from scratch. It's more natural but can be unpredictable.")
            with st.spinner("Generating response..."):
                try:
                    response = requests.post(GENERATIVE_API_URL, json={"text": user_input})
                    response.raise_for_status()
                    result = response.json()
                    reply = result.get("reply", "N/A")
                    st.write("**Bot Reply:**")
                    st.warning(f"_{reply}_")
                except requests.exceptions.RequestException as e:
                    st.error(f"API Connection Error: {e}")
    else:
        st.warning("Please enter some text to analyze.")
