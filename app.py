# app.py

import streamlit as st
from rag_pipeline import process_pdf, generate_answer
from utils import split_multiple_questions, format_answer_clean

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Competitive Exam RAG Bot",
    layout="wide"
)

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #87CEEB; }

.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #1f4e79;
}

.qa-block {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
    width: 100%;
    max-width: 750px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}

.question-title {
    font-weight: bold;
    font-size: 16px;
    color: #003366;
    margin-bottom: 8px;
}

.answer-content {
    font-size: 14px;
    color: #333333;
    line-height: 1.6;
}

.small-answer-box {
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    '<div class="main-title">📚 Competitive Exam Guidance Bot (RAG)</div>',
    unsafe_allow_html=True
)

st.write("Upload your study PDF and ask questions.")
st.markdown("---")

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

# LEFT SIDE: Upload PDF
with col1:
    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Upload your study material",
        type="pdf"
    )

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            retriever = process_pdf(uploaded_file)
            st.session_state.retriever = retriever
            st.session_state.pdf_name = uploaded_file.name
            st.success(f"PDF '{uploaded_file.name}' processed successfully!")

# RIGHT SIDE: Ask Question
with col2:
    st.subheader("❓ Ask Your Question")
    question = st.text_input("Enter question (you can combine using AND)")
    submit = st.button("Submit")

# -----------------------------
# Answer Section
# -----------------------------
if submit:
    if "retriever" not in st.session_state:
        st.warning("Please upload a PDF first!")
    elif not question.strip():
        st.warning("Please enter a question!")
    else:
        final_answer = ""

        # Split combined questions
        questions = split_multiple_questions(question)

        for q in questions:

            # Retrieve relevant chunks
            retrieved_chunks = st.session_state.retriever.retrieve(q)

            # Generate structured answer
            response = generate_answer(retrieved_chunks, q)

            # Format nicely
            final_answer += format_answer_clean(q, response)

        # Display result
        st.markdown(
            f"### 📖 Answer from PDF: {st.session_state.pdf_name}"
        )

        st.markdown(
            f'<div class="small-answer-box">{final_answer}</div>',
            unsafe_allow_html=True
        )
