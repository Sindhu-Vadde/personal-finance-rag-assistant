import streamlit as st
import requests

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f1b0f 0%, #1a2e1a 50%, #0f1b0f 100%);
    background-image:
        radial-gradient(circle at 20% 30%, rgba(76,175,80,0.08) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(76,175,80,0.08) 0%, transparent 40%);
}
h1 {
    color: #4CAF50;
    font-weight: 700;
}
h1::before {
    content: "💰 ";
}
.stTextInput > div > div > input {
    background-color: #1b2e1b;
    color: #e8f5e9;
    border: 1px solid #2E7D32;
}
.stButton > button {
    background-color: #2E7D32;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
}
.stButton > button:hover {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)


st.title("Personal Finance Assistant")
st.write("Ask a question, and I'll answer based on real personal finance discussions.")

question = st.text_input("Your question:")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": question}
            )
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown(answer.replace("$", "\\$"))
            else:
                st.error("Something went wrong. Please try again.")
    else:
        st.warning("Please enter a question first.")
