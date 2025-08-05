import logging
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict
import os
import json

# Configure logging
logger = logging.getLogger(__name__)

# Initialize embedding model
EMBED_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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


def build_index(documents: List[Document], index_path: str = "index_store") -> Chroma:
    """Build a Chroma vector store from documents and persist to disk."""
    start_time = time.time()
    logger.info(f"Building index from {len(documents)} documents")
    
    # Text splitting
    splitter_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=384,
        chunk_overlap=64,
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
