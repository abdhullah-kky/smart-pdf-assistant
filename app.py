import os
import sys
import tempfile
import io
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# LangChain & RAG Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# OCR & PyMuPDF (Fast PDF Processing)
import pytesseract
import fitz  # PyMuPDF

# Tesseract Path Configuration
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r'C:\Program Files\Tesseract-OCR\tesseract.exe')

if not os.path.exists(TESSERACT_PATH):
    print(f"WARNING: Tesseract not found at {TESSERACT_PATH}. Image OCR might fail.")
else:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_SECRET")

# ==========================================
#  AI & RAG SETUP FUNCTIONS
# ==========================================
@st.cache_resource # Cache to prevent reloading the model on every UI click
def get_llm_and_embeddings():
    # Using the requested Llama 3.1 model
    llm = ChatOpenAI(
        model_name="meta-llama/llama-3.1-8b-instruct", 
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

def process_pdf_with_ocr(file_path: str) -> list[Document]:
    """Extracts text natively and performs OCR ONLY on embedded images (Lightning Fast!)."""
    documents = []
    
    try:
        # Open the PDF using PyMuPDF
        pdf_document = fitz.open(file_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # 1. Standard Text Extraction (Extracting normal text only)
            standard_text = page.get_text()
            if standard_text.strip():
                documents.append(Document(
                    page_content=f"[Page {page_num+1} Text]:\n{standard_text}",
                    metadata={"source": "uploaded_pdf", "page": page_num, "type": "standard_text"}
                ))
                
            # 2. Extract ONLY Images for OCR (Extracting and processing only images)
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0] # Image Reference ID
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert Image Bytes to a PIL Image and send directly to Tesseract
                    image = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(image)
                    
                    if ocr_text.strip():
                        documents.append(Document(
                            page_content=f"[OCR Content from Image {img_index+1} on Page {page_num+1}]:\n{ocr_text}",
                            metadata={"source": "uploaded_pdf", "page": page_num, "type": "ocr_image"}
                        ))
                except Exception as img_e:
                    print(f"Warning: Failed to process image {img_index} on page {page_num}. Error: {img_e}")
                    
    except Exception as e:
        st.warning(f"PDF Processing failed. Error: {e}")

    return documents

def build_vector_db(documents: list[Document], embeddings) -> Chroma:
    """Splits documents and stores them in ChromaDB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectordb

def get_rag_chain(vectordb: Chroma, llm):
    """Sets up the LangChain retrieval pipeline with Strict Safety Rules."""
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    # Strict Safety Guardrails for the AI
    system_prompt = (
        "You are a smart, helpful, and respectful PDF assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "\n\n"
        "CRITICAL SAFETY RULE: You must NEVER generate harmful, abusive, discriminatory, "
        "sexually explicit, or unsafe content intentionally. If the user asks a question "
        "that violates these rules, you must strictly refuse to answer and state that it goes against your safety guidelines."
        "\n\n"
        "Context: {context}"
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
            with st.spinner("Processing PDF (Hybrid Text & OCR Extraction)..."):
                # Save uploaded file temporarily to disk (required for PyMuPDF)
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