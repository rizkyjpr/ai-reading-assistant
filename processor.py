from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_document(uploaded_file):
    # --- FILE HANDLING ---
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- DOCUMENT LOADING ---
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # --- TEXT SPLITTING ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=200, add_start_index=True
    )
    chunks = text_splitter.split_documents(pages)

    # --- EMBEDDING MODEL ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- VECTOR DATABASE ---
    vector_db = DocArrayInMemorySearch.from_documents(chunks, embeddings)

    return vector_db
