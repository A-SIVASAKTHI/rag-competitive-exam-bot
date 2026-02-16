# rag_pipeline.py

import os
import re
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Simple TF-IDF Retriever Class
# -----------------------------
class SimpleRetriever:
    def __init__(self, texts):
        self.texts = texts
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors = self.vectorizer.fit_transform(texts)

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.texts[i] for i in top_indices]


# -----------------------------
# Process Uploaded PDF
# -----------------------------
def process_pdf(uploaded_file):
    """
    Loads PDF, splits into chunks,
    and creates TF-IDF retriever.
    """

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    # Remove temp file
    os.remove(tmp_path)

    return SimpleRetriever(texts)


# -----------------------------
# Generate Clean Short Answer
# -----------------------------
def generate_answer(retrieved_chunks, question):
    """
    Extracts the most relevant sentence from retrieved chunks
    instead of returning the entire chunk.
    """

    if not retrieved_chunks:
        return "Answer not found in the uploaded PDF."

    # Combine top chunks
    combined_text = " ".join(retrieved_chunks[:2])

    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', combined_text)

    # Remove very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return "Relevant answer not clearly found."

    # Match best sentence based on word overlap
    question_words = set(question.lower().split())

    best_sentence = ""
    max_overlap = 0

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(question_words & sentence_words)

        if overlap > max_overlap:
            max_overlap = overlap
            best_sentence = sentence

    if best_sentence:
        return best_sentence.strip()

    # Fallback
    return sentences[0].strip()
