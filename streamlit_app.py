import streamlit as st
from app1 import get_answer

st.set_page_config(
    page_title="AI Career Mentor Chatbot",
    page_icon="🎓"
)

st.title("🎓 AI Career Mentor Chatbot")

st.write(
    "Ask questions about AI, Machine Learning, careers, internships, and projects."
)

question = st.chat_input(
    "Ask your question..."
)

if question:

    st.chat_message("user").write(
        question
    )

    answer, confidence = get_answer(
        question
    )

    st.chat_message("assistant").write(
        answer
    )

    st.caption(
        f"Confidence: {confidence:.2%}"
    )