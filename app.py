import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import DocArrayInMemorySearch
import tempfile
import shutil
from pathlib import Path

# Create a documents directory in the project root
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploaded_docs')
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    # Register cleanup on session end
    def cleanup():
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
    atexit.register(cleanup)

# Load environment variables
load_dotenv()

# Set OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    st.error("OpenAI API key not found. Please set it in your .env file.")
    st.stop()

# Proxy Configuration
proxy_config = {}
if os.getenv("HTTP_PROXY"):
    proxy_config["http"] = os.getenv("HTTP_PROXY")
if os.getenv("HTTPS_PROXY"):
    proxy_config["https"] = os.getenv("HTTPS_PROXY")

# Create OpenAI client with proxy settings if proxy config not empty
if proxy_config:
    client = OpenAI(http_client=openai.http_client.ProxyManager(proxies=proxy_config))
    embeddings = OpenAIEmbeddings(client=client)
else:
    embeddings = OpenAIEmbeddings()

# Set up the page
st.set_page_config(page_title="Document Q&A Assistant", layout="wide")
st.title("📚 Document Q&A Assistant")

def process_document(uploaded_file):
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Choose the appropriate loader based on file type
        if uploaded_file.name.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.lower().endswith('.txt'):
            loader = TextLoader(tmp_file_path)
        else:
            raise ValueError("Unsupported file format")

        # Load the document
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)

        return splits, documents

    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)

def save_uploaded_file(uploaded_file):
    """Save an uploaded file and return its path"""
    # Create a unique filename to avoid conflicts
    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
    base, ext = os.path.splitext(file_path)
    counter = 1
    while os.path.exists(file_path):
        file_path = f"{base}_{counter}{ext}"
        counter += 1
        
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

def create_source_link(source_path, page=None):
    """Create a clickable link to the source document"""
    if not source_path:
        return ""
    
    filename = os.path.basename(source_path)
    if page is not None:
        # For PDFs, create a link with page number
        if filename.lower().endswith('.pdf'):
            display_text = f"{filename} (Page {page})"
            return f'<a href="file://{source_path}#page={page}" target="_blank">{display_text}</a>'
    else:
        display_text = filename
        return f'<a href="file://{source_path}" target="_blank">{display_text}</a>'
    
    return display_text

def format_source_documents(source_docs):
    if not source_docs:
        return ""
    
    sources = []
    for i, doc in enumerate(source_docs, 1):
        metadata = doc.metadata
        source_text = f"[{i}] "
        
        if 'source' in metadata:
            source_path = metadata['source']
            page = metadata.get('page')
            source_text += create_source_link(source_path, page)
        
        # Add a snippet of the relevant text
        snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        source_text += f"\nRelevant text: {snippet}\n"
        
        sources.append(source_text)
    
    return "\n".join(sources)

def initialize_qa_chain(vector_store, all_documents):
    """Initialize the QA chain with the vector store"""
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        verbose=True,
        chain_type="stuff"
    )
    
    return qa_chain

def initialize_conversation(vector_store):
    """Initialize the conversation chain"""
    llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview")
    
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        verbose=True,
        chain_type="stuff"
    )
    
    return conversation

# Create two columns for layout
left_column, right_column = st.columns([1, 4])

# Right column - Chat Interface
with right_column:
    st.header("Chat Interface")
    
    # Chat interface
    with st.form(key='question_form'):
        user_question = st.text_input("Ask a question about your documents:", value=st.session_state.question, key="question_input")
        submit_button = st.form_submit_button("Ask")

    if submit_button and user_question:
        if not st.session_state.vector_store:
            st.warning("Please upload some documents first!")
        else:
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain(
                        {"question": user_question, "chat_history": st.session_state.chat_history}
                    )

                    # Format the answer with sources
                    answer = result["answer"]
                    sources = format_source_documents(result.get("source_documents", []))
                    
                    st.session_state.chat_history.append((user_question, answer, sources))
                    # Clear the question in session state
                    st.session_state.question = ""

                    # Display chat history with citations in reverse order (newest first)
                    for question, answer, sources in reversed(st.session_state.chat_history):
                        with st.container():
                            st.write("### Question:")
                            st.write(question)
                            
                            st.write("### Answer:")
                            st.write(answer)
                            
                            # Display sources with clickable links
                            if sources:
                                with st.expander("📚 View Sources"):
                                    st.markdown("### Sources:")
                                    st.markdown(sources, unsafe_allow_html=True)
                            
                            st.markdown("---")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Left column - Document Management
with left_column:
    st.header("Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your documents:",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Process only newly uploaded files
        new_files = [f for f in uploaded_files if f not in st.session_state.uploaded_files]
        if new_files:
            with st.spinner("Processing new documents..."):
                try:
                    # Process new documents
                    all_splits = []
                    all_documents = []
                    
                    for file in new_files:
                        try:
                            # Save the file and get its path
                            file_path = save_uploaded_file(file)
                            
                            if file.name.endswith(".pdf"):
                                loader = PyPDFLoader(file_path)
                            else:
                                loader = TextLoader(file_path)
                            
                            documents = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200
                            )
                            splits = text_splitter.split_documents(documents)
                            
                            # Update the source paths in metadata
                            for split in splits:
                                split.metadata['source'] = file_path
                            for doc in documents:
                                doc.metadata['source'] = file_path
                            
                            all_splits.extend(splits)
                            all_documents.extend(documents)
                            
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            continue
                    
                    if all_splits:
                        try:
                            # Create or update the vector store
                            vector_store = DocArrayInMemorySearch.from_documents(
                                documents=all_splits,
                                embedding=embeddings
                            )
                            st.session_state.vector_store = vector_store
                            st.session_state.all_documents.extend(all_documents)
                            st.session_state.qa_chain = initialize_qa_chain(vector_store, st.session_state.all_documents)
                            st.session_state.conversation = initialize_conversation(vector_store)
                            st.session_state.uploaded_files.extend(new_files)
                            st.success("Documents processed successfully!")
                        except Exception as e:
                            st.error(f"Error creating embeddings: {str(e)}")
                            # Clean up files if embedding fails
                            for doc in all_documents:
                                if os.path.exists(doc.metadata['source']):
                                    os.remove(doc.metadata['source'])
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display uploaded documents with links
    if st.session_state.uploaded_files:
        st.subheader("Uploaded Documents")
        for file in st.session_state.uploaded_files:
            file_path = os.path.join(DOCS_DIR, file.name)
            if os.path.exists(file_path):
                st.markdown(f"📄 {create_source_link(file_path)}", unsafe_allow_html=True)