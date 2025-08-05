import streamlit as st
import os
import tempfile
import shutil
import logging
import time
from datetime import datetime
from utils.pdf_parser import extract_text_from_pdf
from utils.index_builder import convert_to_documents, build_index, load_index
from utils.query_engine import load_query_engine, ask_question
from utils.metadata_extractor import extract_pdf_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_vector_store():
    """Clear the vector store to start fresh."""
    start_time = time.time()
    index_store_path = "index_store"
    if os.path.exists(index_store_path):
        try:
            shutil.rmtree(index_store_path)
            os.makedirs(index_store_path, exist_ok=True)
            clear_time = time.time() - start_time
            logger.info(f"Vector store cleared in {clear_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            st.error(f"Error clearing vector store: {e}")
            return False
    logger.info("Vector store directory does not exist, nothing to clear")
    return True

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Smart PDF Query System", layout="wide")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'question_input' not in st.session_state:
    st.session_state.question_input = ""

if 'chat_cleared' not in st.session_state:
    st.session_state.chat_cleared = False

st.title("📄 Smart PDF Query System")
st.markdown("Upload one or more PDF documents and ask natural language questions.")

# Upload PDFs
uploaded_files = st.file_uploader("📤 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    app_start_time = time.time()
    logger.info(f"Starting PDF processing for {len(uploaded_files)} files")
    
    with st.spinner("Processing PDFs..."):
        processing_start = time.time()
        parsed_results = []
        metadata_list = []
        all_docs = []

        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")

        for i, file in enumerate(uploaded_files, 1):
            file_start_time = time.time()
            logger.info(f"Processing file {i}/{len(uploaded_files)}: {file.name}")
            
            # Save to temp dir
            save_start = time.time()
            filepath = os.path.join(temp_dir, file.name)
            with open(filepath, "wb") as f:
                f.write(file.read())
            save_time = time.time() - save_start
            logger.info(f"   - File saved in {save_time:.2f} seconds")

            # Extract metadata + text
            metadata_start = time.time()
            metadata = extract_pdf_metadata(filepath)
            metadata_time = time.time() - metadata_start
            logger.info(f"   - Metadata extracted in {metadata_time:.2f} seconds")
            logger.info(f"   - Pages: {metadata['num_pages']}, Author: {metadata['author']}")

            text_start = time.time()
            parsed = extract_text_from_pdf(filepath)
            text_time = time.time() - text_start
            logger.info(f"   - Text extracted in {text_time:.2f} seconds")

            doc_start = time.time()
            docs = convert_to_documents(parsed)
            doc_time = time.time() - doc_start
            logger.info(f"   - Documents converted in {doc_time:.2f} seconds")
            logger.info(f"   - Generated {len(docs)} document chunks")

            metadata_list.append(metadata)
            parsed_results.append(parsed)
            all_docs.extend(docs)
            
            file_time = time.time() - file_start_time
            logger.info(f"   - Total file processing time: {file_time:.2f} seconds")

        # Show metadata summary
        st.subheader("Document Metadata")
        for meta in metadata_list:
            with st.expander(meta['filename']):
                st.json(meta)

        # Build index from all docs
        index_start = time.time()
        logger.info("Building index from all documents")
        index = build_index(all_docs)
        index_time = time.time() - index_start
        logger.info(f"Index built in {index_time:.2f} seconds")

        processing_time = time.time() - processing_start
        logger.info(f"Total PDF processing time: {processing_time:.2f} seconds")
        logger.info(f"Total document chunks: {len(all_docs)}")

    st.success("PDFs processed and indexed!")

    # Clear chat and vector store
    if st.button("New Chat"):
        logger.info("New Chat button clicked - starting cleanup process")
        clear_start = time.time()
        with st.spinner("Clearing conversation and vector store..."):
            # Clear conversation history
            old_history_length = len(st.session_state.chat_history)
            st.session_state.chat_history = []
            st.session_state.chat_cleared = True
            logger.info(f"Cleared conversation history: {old_history_length} entries removed")
            
            # Clear vector store
            if clear_vector_store():
                st.success("Conversation history and vector store cleared!")
                logger.info("New Chat cleanup completed successfully")
            else:
                st.error("Failed to clear vector store")
                logger.error("New Chat cleanup failed - vector store clearing failed")
        clear_time = time.time() - clear_start
        logger.info(f"Chat clearing completed in {clear_time:.2f} seconds")
        st.rerun()

    # Conversation history
    if st.session_state.chat_history and not st.session_state.chat_cleared:
        st.subheader("Conversation")

        for entry in st.session_state.chat_history:
            question, answer, timestamp = entry["question"], entry["answer"], entry["timestamp"]

            with st.container():
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.markdown("👤")
                with col2:
                    st.markdown(f"**You:** {question}  \n*🕒 {timestamp}*")

            with st.container():
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.markdown("🤖")
                with col2:
                    st.markdown(f"**AI:** {answer}")

            st.divider()
    
    # Reset chat_cleared flag after displaying
    if st.session_state.chat_cleared:
        st.session_state.chat_cleared = False

    # Question input form
    st.subheader("Ask Questions")
    with st.form(key="question_form"):
        question = st.text_input("Type your question here...", key="question_input")
        submit_button = st.form_submit_button("Send")

        if submit_button and question:
            question_start = time.time()
            logger.info(f"Processing question: {question[:50]}...")
            
            with st.spinner("Thinking..."):
                engine_start = time.time()
                query_engine = load_query_engine()
                engine_time = time.time() - engine_start
                logger.info(f"Query engine loaded in {engine_time:.2f} seconds")
                
                # Convert chat_history to the format expected by ask_question
                chat_history_list = [(entry["question"], entry["answer"]) for entry in st.session_state.chat_history]
                answer_start = time.time()
                raw_answer = ask_question(query_engine, question, chat_history_list)
                answer_time = time.time() - answer_start
                logger.info(f"Answer generated in {answer_time:.2f} seconds")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": raw_answer,
                    "timestamp": timestamp
                })

            question_time = time.time() - question_start
            logger.info(f"Total question processing time: {question_time:.2f} seconds")
            logger.info(f"Question added to chat history with timestamp: {timestamp}")

            st.rerun()

    app_time = time.time() - app_start_time
    logger.info(f"Total app session time: {app_time:.2f} seconds")

else:
    st.info("Upload PDF(s) above to get started.")
