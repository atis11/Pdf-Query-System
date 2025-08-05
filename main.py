import os
import logging
import time
from glob import glob
from utils.pdf_parser import extract_text_from_pdf
from utils.index_builder import convert_to_documents, build_index, load_index
from utils.metadata_extractor import extract_pdf_metadata
from utils.query_engine import load_query_engine, ask_question

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cli.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_pdf_folder(folder_path: str = "data/uploads") -> list:
    """
    Process all PDF files in a folder and return list of Document objects.
    """
    start_time = time.time()
    all_docs = []
    logger.info(f"Scanning folder: {folder_path}")

    pdf_files = glob(os.path.join(folder_path, "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")

    for i, filepath in enumerate(pdf_files, 1):
        file_start_time = time.time()
        filename = os.path.basename(filepath)
        logger.info(f"Processing file {i}/{len(pdf_files)}: {filename}")

        # Extract metadata
        metadata_start = time.time()
        metadata = extract_pdf_metadata(filepath)
        metadata_time = time.time() - metadata_start
        logger.info(f"   - Pages: {metadata['num_pages']}")
        logger.info(f"   - Author: {metadata['author']}")
        logger.info(f"   - Created: {metadata['created']}")
        logger.info(f"   - Metadata extraction time: {metadata_time:.2f} seconds")

        # Extract text
        text_start = time.time()
        parsed = extract_text_from_pdf(filepath)
        text_time = time.time() - text_start
        logger.info(f"   - Text extraction time: {text_time:.2f} seconds")

        # Convert to documents
        doc_start = time.time()
        docs = convert_to_documents(parsed)
        doc_time = time.time() - doc_start
        logger.info(f"   - Document conversion time: {doc_time:.2f} seconds")

        all_docs.extend(docs)
        file_time = time.time() - file_start_time
        logger.info(f"   - Total file processing time: {file_time:.2f} seconds")
        logger.info(f"   - Generated {len(docs)} document chunks")

    total_time = time.time() - start_time
    logger.info(f"All PDFs processed in {total_time:.2f} seconds")
    logger.info(f"Total document chunks generated: {len(all_docs)}")

    return all_docs


def main():
    start_time = time.time()
    logger.info("Starting Smart PDF Query System (CLI) - LangChain Version")
    logger.info("=" * 50)

    # Step 1: Process PDFs
    logger.info("Step 1: Processing PDFs")
    docs = process_pdf_folder()

    # Step 2: Build and save index
    logger.info("Step 2: Building and saving index")
    index_start = time.time()
    index = build_index(docs)
    index_time = time.time() - index_start
    logger.info(f"Index built and saved in {index_time:.2f} seconds")

    # Step 3: Load query engine
    logger.info("Step 3: Loading query engine")
    engine_start = time.time()
    qa_chain = load_query_engine()
    engine_time = time.time() - engine_start
    logger.info(f"Query engine loaded successfully in {engine_time:.2f} seconds!")

    # Step 4: Interactive Q&A with conversation history
    chat_history = []
    logger.info("Step 4: Starting interactive Q&A session")
    logger.info("Ask a question (type 'exit' to quit, 'new' to start a new chat):")
    
    session_start = time.time()
    question_count = 0
    
    while True:
        question = input(">> ")
        if question.strip().lower() in ("exit", "quit"):
            session_time = time.time() - session_start
            logger.info(f"Session ended. Total time: {session_time:.2f} seconds, Questions asked: {question_count}")
            break
        if question.strip().lower() == "new":
            chat_history = []
            logger.info("Started a new chat. Conversation history cleared.")
            continue

        question_start = time.time()
        answer = ask_question(qa_chain, question, chat_history)
        question_time = time.time() - question_start
        question_count += 1
        
        logger.info(f"Question {question_count} processed in {question_time:.2f} seconds")
        print(f"\nAnswer:\n{answer}\n")

    total_time = time.time() - start_time
    logger.info(f"Total application runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
