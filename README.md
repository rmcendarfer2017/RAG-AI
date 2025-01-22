# RAG AI Document Assistant

This is a Retrieval-Augmented Generation (RAG) AI assistant that allows users to upload documents and ask questions about their content. The application provides answers based on the uploaded documents and includes source references for transparency.

## Features

- Upload multiple documents (PDF, TXT, DOCX)
- Ask questions about your documents
- Get AI-generated answers with source references
- Interactive web interface
- Document source tracking

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

4. Run the application:
```bash
streamlit run app.py
```

## Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Add your OpenAI API key in the secrets management:
   - Go to "Advanced Settings"
   - Add your OPENAI_API_KEY
7. Deploy!

### Option 2: Manual Deployment

Run the app locally:
```bash
streamlit run app.py
```

## Usage

1. Launch the application using the command above
2. Upload one or more documents using the file uploader
3. Wait for the documents to be processed
4. Start asking questions in the text input field
5. View the AI's answers and their source references

## Supported File Types

- PDF (.pdf)
- Text files (.txt)
- Word documents (.docx)

## Technical Details

- Uses LangChain for document processing and RAG implementation
- ChromaDB as the vector store
- OpenAI's embeddings and chat models
- Streamlit for the web interface
