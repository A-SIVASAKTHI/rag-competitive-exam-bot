# app.py

import streamlit as st
from rag_pipeline import process_pdf, load_llm, get_qa_chain
import re

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Competitive Exam RAG Bot", layout="wide")

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown("""
    <style>
    .stApp { background-color: #87CEEB; }
    .main-title { font-size: 38px; font-weight: bold; color: #1f4e79; }
    .answer-box { background-color: #eef5ff; padding: 20px; border-radius: 10px; font-size: 16px; white-space: pre-line; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown('<div class="main-title">📚 Competitive Exam Guidance Bot (RAG)</div>', unsafe_allow_html=True)
st.write("Upload your study PDF and ask questions.")
st.markdown("---")

# -----------------------------
# Two-column layout
# -----------------------------
col1, col2 = st.columns(2)

# LEFT SIDE: PDF Upload
with col1:
    st.subheader("📂 Upload PDF (Max 200MB)")
    uploaded_file = st.file_uploader("Upload your study material", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            retriever = process_pdf(uploaded_file, use_real_embeddings=True)
            st.session_state.retriever = retriever
            st.session_state.pdf_name = uploaded_file.name
            st.success(f"PDF '{uploaded_file.name}' processed successfully!")

# RIGHT SIDE: Ask Question
with col2:
    st.subheader("❓ Ask Your Question")
    question = st.text_input("Enter your question here")
    submit = st.button("Submit")

# -----------------------------
# Helper function: format answers (works even for 1-line answers)
# -----------------------------
def format_answer(question, raw_answer):
    """
    Always returns a structured, exam-ready answer.
    Even if raw_answer is 1 line.
    """
    formatted = f"### {question.capitalize()}\n\n"
    formatted += raw_answer.strip() + "\n\n"
    formatted += "**Note:** This is a concise explanation."
    return formatted

# -----------------------------
# Answer Section
# -----------------------------
if submit:
    if "retriever" not in st.session_state:
        st.warning("Please upload a PDF first!")
    elif not question.strip():
        st.warning("Please enter a question!")
    else:
        llm = load_llm()
        final_answer = ""

        # Split multiple questions using "and" or "?"
        questions = re.split(r'\band\b|\?', question.lower())
        questions = [q.strip() for q in questions if q.strip()]

        for q in questions:
            # Strong prompt with PDF context
            prompt = f"""
            You are a competitive exam assistant. Answer the following question in a structured format:
            Heading, definition/explanation, key points, example if applicable, note.
            Question: {q}
            """

            # Use RAG chain to answer
            qa_chain = get_qa_chain(llm, st.session_state.retriever)
            response = qa_chain.run(prompt)

            # Fallback: if response too short, ask LLM directly
            if not response or len(response.strip().split()) <= 5:
                response = llm(prompt)

            # Format answer
            final_answer += format_answer(q, response) + "\n\n"

        # Display final answer
        st.markdown(f" 📖 Answer from PDF: {st.session_state.pdf_name}")
        st.markdown(f'<div class="answer-box">{final_answer}</div>', unsafe_allow_html=True)
