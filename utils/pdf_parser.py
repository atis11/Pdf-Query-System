import fitz  # PyMuPDF
import easyocr
import os
import logging
import time
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=True)

def extract_text_from_pdf(filepath: str) -> Dict:
    """
    Extracts text from a PDF using PyMuPDF. Falls back to OCR if text is missing.
    
    Returns:
        {
            'filename': str,
            'pages': List[Dict{'page_num': int, 'text': str}]
        }
    """
    start_time = time.time()
    filename = os.path.basename(filepath)
    logger.info(f"Processing PDF: {filename}")
    
    # Open PDF
    open_start = time.time()
    doc = fitz.open(filepath)
    open_time = time.time() - open_start
    logger.info(f"   - PDF opened in {open_time:.2f} seconds")
    logger.info(f"   - Total pages: {len(doc)}")
    
    results = []
    ocr_pages = 0

    for page_num in range(len(doc)):
        page_start = time.time()
        page = doc[page_num]
        
        # Extract text
        text_start = time.time()
        text = page.get_text().strip()
        text_time = time.time() - text_start

        if not text:  # Fallback to OCR if text is empty
            ocr_start = time.time()
            logger.info(f"   - Page {page_num + 1}: No text found, using OCR")
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")

            # OCR from image
            ocr_result = reader.readtext(img_bytes, detail=0, paragraph=True)
            text = "\n".join(ocr_result)
            ocr_time = time.time() - ocr_start
            ocr_pages += 1
            logger.info(f"   - Page {page_num + 1}: OCR completed in {ocr_time:.2f} seconds")
        else:
            logger.info(f"   - Page {page_num + 1}: Text extracted in {text_time:.2f} seconds")

        results.append({
            'page_num': page_num + 1,
            'text': text
        })
        
        page_time = time.time() - page_start
        if page_num % 5 == 0 or page_num == len(doc) - 1:  # Log every 5th page or last page
            logger.info(f"   - Page {page_num + 1} processed in {page_time:.2f} seconds")

    # Close document
    doc.close()
    
    total_time = time.time() - start_time
    logger.info(f"PDF processing completed in {total_time:.2f} seconds")
    logger.info(f"Pages processed: {len(results)}, OCR pages: {ocr_pages}")

    return {
        'filename': filename,
        'pages': results
    }
