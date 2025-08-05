import fitz  # PyMuPDF
import os
import logging
import time
from typing import Dict
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def extract_pdf_metadata(filepath: str) -> Dict:
    """Extracts metadata from a PDF file (title, author, etc.)"""
    start_time = time.time()
    filename = os.path.basename(filepath)
    logger.info(f"Extracting metadata from: {filename}")
    
    # Open PDF
    open_start = time.time()
    doc = fitz.open(filepath)
    open_time = time.time() - open_start
    logger.info(f"   - PDF opened in {open_time:.2f} seconds")
    
    # Extract metadata
    meta_start = time.time()
    meta = doc.metadata or {}
    file_stat = os.stat(filepath)

    metadata = {
        "filename": filename,
        "file_size_kb": round(file_stat.st_size / 1024, 2),
        "num_pages": len(doc),
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "keywords": meta.get("keywords", ""),
        "created": _convert_pdf_date(meta.get("creationDate")),
        "modified": _convert_pdf_date(meta.get("modDate")),
    }
    meta_time = time.time() - meta_start
    
    # Close document
    doc.close()
    
    total_time = time.time() - start_time
    logger.info(f"   - Metadata extracted in {meta_time:.2f} seconds")
    logger.info(f"   - File size: {metadata['file_size_kb']} KB, Pages: {metadata['num_pages']}")
    logger.info(f"Total metadata extraction time: {total_time:.2f} seconds")

    return metadata


def _convert_pdf_date(date_str):
    """Convert PDF date string to readable format."""
    if not date_str:
        return ""
    try:
        # PDF dates look like: D:20230101120000Z
        date_str = date_str.replace("D:", "").split("Z")[0]
        return datetime.strptime(date_str, "%Y%m%d%H%M%S").isoformat()
    except Exception:
        return date_str  # fallback

