import streamlit as st
import os
import tempfile
import shutil
import logging
import time
from datetime import datetime
from utils.pdf_parser import extract_text_from_pdf
from utils.index_builder import convert_to_documents, build_index, load_index, get_index_status
from utils.query_engine import load_query_engine, ask_question
from utils.metadata_extractor import extract_pdf_metadata
from utils.query_analyzer import QueryAnalyzer

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

def clear_cache_only():
    """Clear only the cache information, keeping the index."""
    cache_file = os.path.join("index_store", "cache_info.pkl")
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
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

if 'query_analyzer' not in st.session_state:
    st.session_state.query_analyzer = QueryAnalyzer()

if 'auto_parameters' not in st.session_state:
    st.session_state.auto_parameters = True

st.title("📄 Smart PDF Query System")
st.markdown("Upload one or more PDF documents and ask natural language questions.")

# Example queries for auto-parameter demonstration
if st.session_state.auto_parameters:
    with st.expander("💡 Example Queries to Try", expanded=False):
        st.write("**Try these different types of queries to see automatic parameter optimization:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Factual Questions:**")
            st.code("What is the main topic of this document?")
            st.code("Who is the author?")
            st.code("When was this published?")
            
            st.write("**Numerical Questions:**")
            st.code("How many pages are there?")
            st.code("What percentage of...")
            st.code("How much does it cost?")
        
        with col2:
            st.write("**Analytical Questions:**")
            st.code("Analyze the advantages and disadvantages")
            st.code("What are the key benefits?")
            st.code("Evaluate the impact of...")
            
            st.write("**Comparative Questions:**")
            st.code("Compare X versus Y")
            st.code("What are the differences between...")
            st.code("Which is better, A or B?")
            
            st.write("**Summarization:**")
            st.code("Summarize the main points")
            st.code("Give me a brief overview")
            st.code("What are the key takeaways?")

# Parameter Configuration
st.subheader("⚙️ Parameter Configuration")

# Auto vs Manual parameter selection
col1, col2 = st.columns([1, 3])
with col1:
    auto_params = st.checkbox("🤖 Auto-Optimize Parameters", 
                             value=st.session_state.auto_parameters,
                             help="Automatically optimize parameters based on your query type")
    st.session_state.auto_parameters = auto_params

with col2:
    if auto_params:
        st.info("🎯 Parameters will be automatically optimized based on your query characteristics")
    else:
        st.info("⚙️ Manual parameter configuration enabled")

# Manual parameter configuration (only shown when auto is disabled)
if not auto_params:
    with st.expander("📝 Manual Parameter Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Text Chunking")
            chunk_size = st.slider("Chunk Size", min_value=256, max_value=1024, value=512, step=64, 
                                  help="Size of text chunks in characters. Larger chunks preserve more context but may be less precise.")
            chunk_overlap = st.slider("Chunk Overlap", min_value=32, max_value=128, value=64, step=16,
                                     help="Overlap between consecutive chunks to maintain context continuity.")
        
        with col2:
            st.subheader("🔍 Retrieval Settings")
            top_k = st.slider("Top-K Retrieval", min_value=3, max_value=10, value=5, step=1,
                             help="Number of most relevant chunks to retrieve for each question.")
            dense_weight = st.slider("Dense Weight", min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                                    help="Weight for semantic (dense) retrieval vs keyword (sparse) retrieval.")
        
        st.subheader("🤖 LLM Settings")
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.6, step=0.1,
                               help="Controls randomness in responses. Lower = more focused, Higher = more creative.")
else:
    # Set default values for auto mode
    chunk_size = 512
    chunk_overlap = 64
    top_k = 5
    dense_weight = 0.5
    temperature = 0.6
    
    # Real-time parameter display
    st.subheader("📊 Current Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunk Size", f"{chunk_size} chars")
        st.metric("Chunk Overlap", f"{chunk_overlap} chars")
    with col2:
        st.metric("Top-K", f"{top_k} chunks")
        st.metric("Dense Weight", f"{dense_weight:.1f}")
    with col3:
        st.metric("Temperature", f"{temperature:.1f}")
        if temperature < 0.3:
            st.info("🎯 Focused responses")
        elif temperature < 0.7:
            st.info("⚖️ Balanced responses")
        else:
            st.info("🎨 Creative responses")
    
    # Real-time parameter effects explanation
    st.subheader("🔍 Parameter Effects")
    with st.expander("How these settings affect your results"):
        st.write(f"""
        **Chunk Size ({chunk_size} chars)**: {'Larger chunks preserve more context but may be less precise' if chunk_size > 512 else 'Smaller chunks are more precise but may miss context'}
        
        **Chunk Overlap ({chunk_overlap} chars)**: {'Higher overlap ensures continuity between chunks' if chunk_overlap > 64 else 'Lower overlap reduces redundancy'}
        
        **Top-K ({top_k} chunks)**: {'More chunks provide broader context' if top_k > 5 else 'Fewer chunks focus on most relevant content'}
        
        **Dense Weight ({dense_weight:.1f})**: {'Favors semantic similarity' if dense_weight > 0.5 else 'Favors keyword matching'}
        
        **Temperature ({temperature:.1f})**: {'More creative and varied responses' if temperature > 0.6 else 'More focused and consistent responses'}
        """)

# Store parameters in session state
if 'config' not in st.session_state:
    st.session_state.config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'top_k': top_k,
        'dense_weight': dense_weight,
        'temperature': temperature
    }
else:
    # Update config with current slider values
    st.session_state.config.update({
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'top_k': top_k,
        'dense_weight': dense_weight,
        'temperature': temperature
    })

# Index Status Display
index_status = get_index_status()
if index_status["exists"] and index_status["cache_valid"]:
    with st.expander("📊 Current Index Status", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed Files", len(index_status["files"]))
            st.metric("Total Chunks", index_status["total_chunks"])
        with col2:
            if index_status["build_time"]:
                build_date = datetime.fromtimestamp(index_status["build_time"]).strftime("%Y-%m-%d %H:%M")
                st.metric("Last Built", build_date)
            st.metric("Status", "✅ Valid")
        
        if index_status["files"]:
            st.write("**Indexed Files:**")
            for filename in index_status["files"]:
                st.write(f"• {filename}")

# Upload PDFs
uploaded_files = st.file_uploader("📤 Upload PDF(s)", type="pdf", accept_multiple_files=True)

# File size validation
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit
if uploaded_files:
    for file in uploaded_files:
        if file.size > MAX_FILE_SIZE:
            st.error(f"File {file.name} is too large ({file.size / (1024*1024):.1f}MB). Maximum size is 50MB.")
            st.stop()

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

        try:
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
                try:
                    metadata = extract_pdf_metadata(filepath)
                    metadata_time = time.time() - metadata_start
                    logger.info(f"   - Metadata extracted in {metadata_time:.2f} seconds")
                    logger.info(f"   - Pages: {metadata['num_pages']}, Author: {metadata['author']}")
                except Exception as e:
                    logger.error(f"   - Failed to extract metadata from {file.name}: {e}")
                    st.error(f"Failed to extract metadata from {file.name}: {e}")
                    continue

                text_start = time.time()
                try:
                    parsed = extract_text_from_pdf(filepath)
                    text_time = time.time() - text_start
                    logger.info(f"   - Text extracted in {text_time:.2f} seconds")
                    
                    # Check if any text was extracted
                    total_text = sum(len(page['text']) for page in parsed['pages'])
                    if total_text == 0:
                        st.warning(f"Warning: No text could be extracted from {file.name}. This might be a scanned document or image-only PDF.")
                except Exception as e:
                    logger.error(f"   - Failed to extract text from {file.name}: {e}")
                    st.error(f"Failed to extract text from {file.name}: {e}")
                    continue

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
            
            # Check if user wants to force rebuild
            force_rebuild = st.checkbox("🔄 Force rebuild index (ignore cache)", 
                                      help="Check this if you want to rebuild the index even if no changes are detected")
            
            if force_rebuild:
                logger.info("Force rebuild requested by user")
                # Clear existing cache to force rebuild
                cache_file = os.path.join("index_store", "cache_info.pkl")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logger.info("Cleared cache to force rebuild")
            
            index = build_index(all_docs, chunk_size=st.session_state.config['chunk_size'], 
                               chunk_overlap=st.session_state.config['chunk_overlap'])
            index_time = time.time() - index_start
            logger.info(f"Index built in {index_time:.2f} seconds")

            processing_time = time.time() - processing_start
            logger.info(f"Total PDF processing time: {processing_time:.2f} seconds")
            logger.info(f"Total document chunks: {len(all_docs)}")
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

    st.success("PDFs processed and indexed!")

    # Cache and Chat Management
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ New Chat"):
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
    
    with col2:
        if st.button("🔄 Clear Cache"):
            if clear_cache_only():
                st.success("Cache cleared! Index will be rebuilt on next upload.")
                logger.info("Cache cleared by user")
            else:
                st.error("Failed to clear cache")
            st.rerun()
    
    with col3:
        if st.button("📊 Refresh Status"):
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

                    # Show retrieved chunks if available
                    sources = entry.get("sources") or []
                    if sources:
                        with st.expander("Show retrieved chunks"):
                            for i, src in enumerate(sources, start=1):
                                filename = src.get("filename") or "Unknown file"
                                page_num = src.get("page_num")
                                header = f"{i}. {filename}"
                                if page_num is not None:
                                    header += f" — Page {page_num}"
                                st.markdown(f"**{header}**")
                                st.code(src.get("text") or "", language="markdown")

            st.divider()
    
    # Reset chat_cleared flag after displaying
    if st.session_state.chat_cleared:
        st.session_state.chat_cleared = False

    # Question input form
    st.subheader("Ask Questions")
    with st.form(key="question_form"):
        question = st.text_input("Type your question here...", key="question_input")
        
        # Show parameter preview for auto mode
        if st.session_state.auto_parameters and question:
            try:
                preview_profile = st.session_state.query_analyzer.analyze_query(question)
                with st.expander("🎯 Parameter Preview", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunk Size", f"{preview_profile.chunk_size} chars")
                        st.metric("Chunk Overlap", f"{preview_profile.chunk_overlap} chars")
                    with col2:
                        st.metric("Top-K", f"{preview_profile.top_k} chunks")
                        st.metric("Dense Weight", f"{preview_profile.dense_weight:.1f}")
                    with col3:
                        st.metric("Temperature", f"{preview_profile.temperature:.1f}")
                        st.metric("Query Type", preview_profile.query_type.title())
                    
                    explanation = st.session_state.query_analyzer.get_parameter_explanation(preview_profile)
                    st.info(f"**Strategy:** {explanation}")
            except Exception as e:
                st.warning(f"Could not preview parameters: {e}")
        
        submit_button = st.form_submit_button("Send")

        if submit_button and question:
            question_start = time.time()
            logger.info(f"Processing question: {question[:50]}...")
            
            # Analyze query and get optimal parameters if auto-optimization is enabled
            if st.session_state.auto_parameters:
                analysis_start = time.time()
                query_profile = st.session_state.query_analyzer.analyze_query(question)
                analysis_time = time.time() - analysis_start
                logger.info(f"Query analysis completed in {analysis_time:.2f} seconds")
                
                # Update config with analyzed parameters
                st.session_state.config.update({
                    'chunk_size': query_profile.chunk_size,
                    'chunk_overlap': query_profile.chunk_overlap,
                    'top_k': query_profile.top_k,
                    'dense_weight': query_profile.dense_weight,
                    'temperature': query_profile.temperature
                })
                
                # Show query analysis results
                with st.expander("🔍 Query Analysis", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Query Type", query_profile.query_type.title())
                        st.metric("Complexity", query_profile.complexity.title())
                        st.metric("Confidence", f"{query_profile.confidence:.2f}")
                    with col2:
                        st.metric("Keywords", len(query_profile.keywords))
                        st.metric("Entities", len(query_profile.entities))
                        st.metric("Intent", query_profile.intent.title())
                    
                    if query_profile.keywords:
                        st.write("**Keywords:**", ", ".join(query_profile.keywords[:5]))
                    if query_profile.entities:
                        st.write("**Entities:**", ", ".join(query_profile.entities[:5]))
                    
                    explanation = st.session_state.query_analyzer.get_parameter_explanation(query_profile)
                    st.info(f"**Parameter Strategy:** {explanation}")
            
            with st.spinner("Thinking..."):
                engine_start = time.time()
                query_engine = load_query_engine(top_k=st.session_state.config['top_k'],
                                               dense_weight=st.session_state.config['dense_weight'],
                                               temperature=st.session_state.config['temperature'])
                engine_time = time.time() - engine_start
                logger.info(f"Query engine loaded in {engine_time:.2f} seconds")
                
                # Convert chat_history to the format expected by ask_question
                chat_history_list = [(entry["question"], entry["answer"]) for entry in st.session_state.chat_history]
                answer_start = time.time()
                raw_answer, retrieved_sources = ask_question(query_engine, question, chat_history_list)
                answer_time = time.time() - answer_start
                logger.info(f"Answer generated in {answer_time:.2f} seconds")
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": raw_answer,
                    "sources": retrieved_sources,
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
