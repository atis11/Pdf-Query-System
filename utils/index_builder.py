import logging
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict
import os
import json
import hashlib
import pickle

# Configure logging
logger = logging.getLogger(__name__)

# Initialize embedding model
EMBED_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_file_hash(filepath: str) -> str:
    """Generate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_config_hash(chunk_size: int, chunk_overlap: int) -> str:
    """Generate hash of configuration parameters."""
    config_str = f"chunk_size:{chunk_size},chunk_overlap:{chunk_overlap}"
    return hashlib.sha256(config_str.encode()).hexdigest()

def get_cache_info(index_path: str = "index_store") -> Dict:
    """Get cached index information."""
    cache_file = os.path.join(index_path, "cache_info.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache info: {e}")
    return {}

def save_cache_info(cache_info: Dict, index_path: str = "index_store"):
    """Save cache information."""
    cache_file = os.path.join(index_path, "cache_info.pkl")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache_info, f)
        logger.info("Cache info saved successfully")
    except Exception as e:
        logger.error(f"Failed to save cache info: {e}")

def should_rebuild_index(documents: List[Document], chunk_size: int, chunk_overlap: int, 
                        index_path: str = "index_store") -> bool:
    """Check if index needs to be rebuilt based on file changes and config changes."""
    
    # Check if index directory exists
    if not os.path.exists(index_path):
        logger.info("Index directory does not exist, rebuild required")
        return True
    
    # Check if Chroma files exist
    chroma_files = ["chroma.sqlite3", "index_metadata.pickle"]
    for file in chroma_files:
        if not os.path.exists(os.path.join(index_path, file)):
            logger.info(f"Chroma file {file} missing, rebuild required")
            return True
    
    # Load cache info
    cache_info = get_cache_info(index_path)
    if not cache_info:
        logger.info("No cache info found, rebuild required")
        return True
    
    # Check config changes
    current_config_hash = get_config_hash(chunk_size, chunk_overlap)
    if cache_info.get("config_hash") != current_config_hash:
        logger.info("Configuration changed, rebuild required")
        return True
    
    # Check file changes
    current_files = {}
    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")
        if filename not in current_files:
            current_files[filename] = []
        current_files[filename].append(doc.metadata.get("page_num", 0))
    
    cached_files = cache_info.get("files", {})
    
    # Check if file list changed
    if set(current_files.keys()) != set(cached_files.keys()):
        logger.info("File list changed, rebuild required")
        return True
    
    # Check if page counts changed
    for filename, pages in current_files.items():
        if filename not in cached_files or len(pages) != len(cached_files[filename]):
            logger.info(f"Page count changed for {filename}, rebuild required")
            return True
    
    logger.info("Index is up to date, no rebuild required")
    return False


def convert_to_documents(parsed_output: Dict) -> List[Document]:
    """Convert parsed PDF dict to LangChain Document objects."""
    start_time = time.time()
    docs = []
    
    logger.info(f"Converting {len(parsed_output['pages'])} pages to documents")
    
    for page in parsed_output["pages"]:
        metadata = {
            "filename": parsed_output["filename"],
            "page_num": page["page_num"]
        }
        docs.append(Document(page_content=page["text"], metadata=metadata))
    
    conversion_time = time.time() - start_time
    logger.info(f"Converted {len(docs)} documents in {conversion_time:.2f} seconds")
    
    return docs


def build_index(documents: List[Document], index_path: str = "index_store", 
                chunk_size: int = 512, chunk_overlap: int = 64) -> Chroma:
    """Build a Chroma vector store from documents and persist to disk."""
    start_time = time.time()
    logger.info(f"Building index from {len(documents)} documents")
    logger.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # Check if rebuild is needed
    if not should_rebuild_index(documents, chunk_size, chunk_overlap, index_path):
        logger.info("Using existing index - no rebuild needed")
        return load_index(index_path)
    
    logger.info("Rebuilding index - changes detected")
    
    # Text splitting
    splitter_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split documents into chunks
    texts = []
    metadatas = []
    
    for i, doc in enumerate(documents, 1):
        doc_start = time.time()
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append(doc.metadata)
        doc_time = time.time() - doc_start
        if i % 10 == 0 or i == len(documents):  # Log every 10th document or last one
            logger.info(f"   - Processed document {i}/{len(documents)} in {doc_time:.2f} seconds")
    
    splitter_time = time.time() - splitter_start
    logger.info(f"Text splitting completed in {splitter_time:.2f} seconds")
    logger.info(f"Generated {len(texts)} text chunks")

    # Ensure index directory exists
    os.makedirs(index_path, exist_ok=True)

    # Persist chunked corpus for BM25 (hybrid search)
    corpus_path = os.path.join(index_path, "bm25_corpus.jsonl")
    try:
        persist_corpus_start = time.time()
        with open(corpus_path, "w", encoding="utf-8") as f:
            for text, metadata in zip(texts, metadatas):
                f.write(json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False) + "\n")
        persist_corpus_time = time.time() - persist_corpus_start
        logger.info(f"Persisted BM25 corpus with {len(texts)} entries to {corpus_path} in {persist_corpus_time:.2f} seconds")
    except Exception as e:
        logger.exception(f"Failed to persist BM25 corpus: {e}")
    
    # Create vector store
    vector_start = time.time()
    logger.info("Creating vector store with embeddings")
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=EMBED_MODEL,
        persist_directory=index_path
    )
    vector_time = time.time() - vector_start
    logger.info(f"Vector store created in {vector_time:.2f} seconds")
    
    # Persist to disk
    persist_start = time.time()
    vectorstore.persist()
    persist_time = time.time() - persist_start
    logger.info(f"Index persisted to disk in {persist_time:.2f} seconds")
    
    # Save cache information
    cache_info = {
        "config_hash": get_config_hash(chunk_size, chunk_overlap),
        "files": {},
        "total_chunks": len(texts),
        "build_time": time.time()
    }
    
    # Record file information
    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")
        if filename not in cache_info["files"]:
            cache_info["files"][filename] = []
        cache_info["files"][filename].append(doc.metadata.get("page_num", 0))
    
    save_cache_info(cache_info, index_path)
    
    total_time = time.time() - start_time
    logger.info(f"Total index building time: {total_time:.2f} seconds")
    logger.info(f"Index saved to: {index_path}")
    
    return vectorstore


def load_index(index_path: str = "index_store") -> Chroma:
    """Load existing index from disk."""
    start_time = time.time()
    logger.info(f"Loading index from: {index_path}")
    
    vectorstore = Chroma(
        persist_directory=index_path,
        embedding_function=EMBED_MODEL
    )
    
    load_time = time.time() - start_time
    logger.info(f"Index loaded in {load_time:.2f} seconds")
    
    return vectorstore

def get_index_status(index_path: str = "index_store") -> Dict:
    """Get current index status information."""
    status = {
        "exists": False,
        "files": [],
        "total_chunks": 0,
        "config_hash": None,
        "build_time": None,
        "cache_valid": False
    }
    
    if not os.path.exists(index_path):
        return status
    
    status["exists"] = True
    
    # Check if Chroma files exist
    chroma_files = ["chroma.sqlite3", "index_metadata.pickle"]
    missing_files = []
    for file in chroma_files:
        if not os.path.exists(os.path.join(index_path, file)):
            missing_files.append(file)
    
    if missing_files:
        status["error"] = f"Missing files: {', '.join(missing_files)}"
        return status
    
    # Load cache info
    cache_info = get_cache_info(index_path)
    if cache_info:
        status["files"] = list(cache_info.get("files", {}).keys())
        status["total_chunks"] = cache_info.get("total_chunks", 0)
        status["config_hash"] = cache_info.get("config_hash")
        status["build_time"] = cache_info.get("build_time")
        status["cache_valid"] = True
    
    return status
