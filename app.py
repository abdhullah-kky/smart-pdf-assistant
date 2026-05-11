import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain & RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# OCR & Image Processing
import pytesseract
from pdf2image import convert_from_path

# Attempt to fetch paths from system environment variables; fallback to default paths if not set
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r'C:\Program Files\Tesseract-OCR\tesseract.exe')
POPPLER_PATH = os.getenv("POPPLER_PATH", r'C:\poppler\Library\bin')

# Check if the paths exist, otherwise display a warning in the terminal
if not os.path.exists(TESSERACT_PATH):
    print(f"WARNING: Tesseract not found at {TESSERACT_PATH}. OCR might fail.")
if not os.path.exists(POPPLER_PATH):
    print(f"WARNING: Poppler not found at {POPPLER_PATH}. PDF Image extraction might fail.")

# Set Tesseract path for pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_SECRET")

# ==========================================
#  AI & RAG SETUP FUNCTIONS
# ==========================================
@st.cache_resource # Cache to prevent reloading the model on every UI click
def get_llm_and_embeddings():
    llm = ChatOpenAI(
        model_name="meta-llama/llama-3.1-8b-instruct", # openai/gpt-4o-mini
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

def process_pdf_with_ocr(file_path: str) -> list[Document]:
    """Extracts text from PDF and performs OCR on pages as images."""
    documents = []
    
    # 1. Standard text extraction
    loader = PyPDFLoader(file_path)
    standard_docs = loader.load()
    
    # 2. OCR Extraction (converting PDF pages to images)
    try:
        pages_as_images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        for i, img in enumerate(pages_as_images):
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                documents.append(Document(
                    page_content=f"[OCR Extracted Content from Page {i+1}]:\n{ocr_text}",
                    metadata={"source": "uploaded_pdf", "page": i, "type": "ocr"}
                ))
    except Exception as e:
        st.warning(f"OCR Processing failed. Text from images won't be extracted. Error: {e}")

    return standard_docs + documents

def build_vector_db(documents: list[Document], embeddings) -> Chroma:
    """Splits documents and stores them in ChromaDB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectordb

def get_rag_chain(vectordb: Chroma, llm):
    """Sets up the LangChain retrieval pipeline."""
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    system_prompt = (
        "You are a Smart PDF Document Assistant. Use the following retrieved context "
        "to answer the user's question. If you don't know the answer based on the context, "
        "say that you don't know. Provide clear summaries and key insights when asked.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

# ==========================================
#  STREAMLIT WEB UI
# ==========================================
def main():
    st.set_page_config(page_title="Smart PDF Assistant", page_icon="📄", layout="wide")
    
    # Check for API Key
    if not OPENROUTER_API_KEY:
        st.error("⚠️ OPENROUTER_SECRET is missing from the .env file.")
        st.stop()

    llm, embeddings = get_llm_and_embeddings()

    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # Sidebar: File Upload and Processing
    with st.sidebar:
        st.title("📄 Smart PDF Assistant")
        st.markdown("Upload a PDF to extract text and embedded images (via OCR).")
        
        uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
        
        if uploaded_file and st.session_state.rag_chain is None:
            with st.spinner("Processing PDF & Running OCR... This may take a minute."):
                # Save uploaded file temporarily to disk (required for PyPDF and pdf2image)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process Pipeline
                docs = process_pdf_with_ocr(tmp_file_path)
                vectordb = build_vector_db(docs, embeddings)
                st.session_state.rag_chain = get_rag_chain(vectordb, llm)
                
                # Cleanup temp file
                os.remove(tmp_file_path)
                
            st.success("✅ Document processed successfully! You can now chat.")

    # Main Chat Interface
    st.header("💬 Chat with your Document")
    
    # Display warning if no document is uploaded
    if st.session_state.rag_chain is None:
        st.info("👈 Please upload a PDF document from the sidebar to begin.")
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box
    if prompt := st.chat_input("Ask a question about your document..."):
        # 1. Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Check if chain is ready
        if st.session_state.rag_chain is None:
            with st.chat_message("assistant"):
                st.warning("Please upload a document first!")
            return

        # 3. Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()