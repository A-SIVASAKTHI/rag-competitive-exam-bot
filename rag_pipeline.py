# rag_pipeline.py

import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# -----------------------------
# Load LLM (text generation)
# -----------------------------
def load_llm():
    """
    Loads HuggingFace text2text-generation pipeline wrapped in LangChain.
    """
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # Can upgrade to flan-t5-large
        max_length=512
    )
    return HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Process PDF and create retriever
# -----------------------------
def process_pdf(uploaded_file, use_real_embeddings=True):
    """
    Converts uploaded PDF to a retriever.
    Args:
        uploaded_file : Uploaded PDF file
        use_real_embeddings : If True, uses real embeddings
    Returns:
        retriever : LangChain retriever object
    """

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text into chunks (with overlap for context)
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Choose embeddings
    if use_real_embeddings:
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        from langchain.embeddings import FakeEmbeddings
        embeddings = FakeEmbeddings(size=768)

    # Create FAISS vector store
    db = FAISS.from_documents(docs, embeddings)

    # Return retriever
    return db.as_retriever(search_kwargs={"k": 3})  # top 3 chunks

# -----------------------------
# Create QA Chain
# -----------------------------
def get_qa_chain(llm, retriever):
    """
    Creates a RetrievalQA chain using provided LLM and retriever.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )
