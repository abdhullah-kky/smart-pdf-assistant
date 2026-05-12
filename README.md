# Smart PDF Document Assistant with Embedded Image Understanding

An AI-powered chatbot that allows users to upload PDF documents and analyze both text and embedded images using RAG (Retrieval-Augmented Generation) and OCR.

## 🚀 Features
- **PDF Text Extraction**: Reads standard text from PDF pages.
- **OCR (Optical Character Recognition)**: Extracts text from images and diagrams embedded inside PDFs using Tesseract OCR.
- **RAG Pipeline**: Uses LangChain and ChromaDB to provide context-aware answers.
- **Modern UI**: A clean, interactive web interface built with Streamlit.
- **Powered by OpenRouter**: Uses advanced models (like Llama) for accurate, free responses.

## 🛠️ Technology Stack
- **Backend**: Python
- **LLM Framework**: LangChain
- **UI**: Streamlit
- **Vector Database**: ChromaDB
- **OCR Engine**: Tesseract OCR

## 📋 Prerequisites
Before running the project, ensure you have the following installed on your Windows system:

1. **Python**: Version 3.10 or higher.
2. **Tesseract OCR**: [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki). Install it to `C:\Program Files\Tesseract-OCR`.

## ⚙️ Installation & Setup

1. **Install uv (Package Manager)**:
   ```bash
   pip install uv
   ```

3. **Clone the Repository**:
   ```bash
   git clone https://github.com/abdhullah-kky/smart-pdf-assistant
   cd smart-pdf-assistant
   ```

5. **Install Dependencies**:
   ```bash
   uv sync
   ```

7. **Environment Variables**:
   Create a `.env` file in the root directory and add your OpenRouter API Key:
   ```bash
   OPENROUTER_SECRET="your_openrouter_api_key_here"
   ```

9. **Path Configuration**:
   Open `app.py` and verify the paths for Tesseract and Poppler if you installed them in different directories:\
   `TESSERACT_PATH = os.getenv("TESSERACT_PATH", r'C:\Program Files\Tesseract-OCR\tesseract.exe')`

## 🏃 How to Run
To start the web application, run the following command in your terminal:
```bash
uv run streamlit run app.py
```
The application will automatically open in your default browser at http://localhost:8501.

## 👤 Author
- **Mohamed Ramees Abdhullah**
- Project for American Corner, Batticaloa.