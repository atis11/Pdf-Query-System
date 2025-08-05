import os
import logging
import time
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load Groq API Key
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# Custom prompt template for better responses
CUSTOM_PROMPT_TEMPLATE = """You are an expert AI assistant specialized in analyzing and answering questions about PDF documents. You have access to context from uploaded PDF files.

Context from the PDF documents:
{context}

Question: {question}

Guidelines for your response:
1. **Answer based ONLY on the provided context** - Do not use external knowledge unless the context explicitly mentions it
3. **Cite sources** - When referencing specific information, mention the document name and page number if available
4. **Be comprehensive** - Provide complete, structured, and well-organized responses, using multiple paragraphs or bullet points if necessary
5. **Use professional language** - Maintain a scholarly tone appropriate for document analysis
6. **Write with clarity** - Explain complex ideas clearly, especially for technical or conceptual questions

If the question asks for:
- **Definitions**: Provide clear, concise definitions from the context
- **Comparisons**: Highlight similarities and differences mentioned in the documents
- **Summaries**: Provide key points in a structured format
- **Technical details**: Include specific technical information from the context
- **Examples**: Reference specific examples mentioned in the documents

Answer:"""


# Create the prompt template
PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Configure LLM from Groq
if groq_key:
    llm = ChatGroq(
        model_name="llama3-70b-8192",  # Updated to Llama3-70b-8192 model
        api_key=groq_key,
        temperature=0.6,
        max_tokens=1024
    )
else:
    logger.warning("GROQ_API_KEY not found in environment variables.")
    logger.warning("Please create a .env file with your GROQ_API_KEY")
    logger.warning("The application will continue but LLM features will be limited.")
    llm = None

def load_query_engine(index_path: str = "index_store") -> ConversationalRetrievalChain:
    """Load saved index and prepare a Groq-powered QueryEngine with conversation support."""
    start_time = time.time()
    
    logger.info("Loading embedding model...")
    embed_start = time.time()
    # Use the same embedding model that was used to build the index
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embed_time = time.time() - embed_start
    logger.info(f"Embedding model loaded in {embed_time:.2f} seconds")
    
    logger.info("Loading vector store...")
    vector_start = time.time()
    vectorstore = Chroma(
        persist_directory=index_path,
        embedding_function=embed_model
    )
    vector_time = time.time() - vector_start
    logger.info(f"Vector store loaded in {vector_time:.2f} seconds")
    
    logger.info("Loading index from storage...")
    index_start = time.time()
    
    if llm:
        # Use ConversationalRetrievalChain for conversation history support
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
    else:
        # Fallback to regular RetrievalQA if no LLM
        from langchain.chains import RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(
            llm=None,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    index_time = time.time() - index_start
    total_time = time.time() - start_time
    
    logger.info(f"Index loaded in {index_time:.2f} seconds")
    logger.info(f"Total query engine loading time: {total_time:.2f} seconds")
    
    return qa_chain


def ask_question(qa_chain, question: str, chat_history: list = None) -> str:
    """Ask a natural language question to the PDF knowledge base."""

    if chat_history is None:
        chat_history = []

    start_time = time.time()
    logger.info(f"Processing question: {question[:50]}...")

    # --- Use ConversationalRetrievalChain which supports chat_history ---
    processing_start = time.time()
    response = qa_chain({"question": question, "chat_history": chat_history})
    processing_time = time.time() - processing_start
    
    # Update chat history
    chat_history.append((question, response['answer']))
    
    # Log answer and sources
    logger.info(f"Generated answer: {response['answer'][:100]}...")
    logger.info(f"Question processing time: {processing_time:.2f} seconds")

    if 'source_documents' in response:
        logger.info("Source documents found:")
        for doc in response['source_documents']:
            meta = doc.metadata
            logger.info(f"- {meta.get('filename')} | Page {meta.get('page_num')}")

    total_time = time.time() - start_time
    logger.info(f"Total question time: {total_time:.2f} seconds")
    
    return response['answer']

def get_specialized_prompt(question_type: str = "general") -> PromptTemplate:
    """Get specialized prompt template based on question type."""
    
    if question_type == "summary":
        template = """You are an expert document analyst. Summarize the key information from the provided context.

Context from the PDF documents:
{context}

Question: {question}

Guidelines:
1. Provide a comprehensive summary of the main points
2. Organize information logically with clear sections
3. Include important details, dates, names, and technical concepts
4. Use bullet points for better readability
5. Maintain the original meaning and context

Summary:"""
    
    elif question_type == "technical":
        template = """You are a technical expert analyzing PDF documents. Provide detailed technical explanations.

Context from the PDF documents:
{context}

Question: {question}

Guidelines:
1. Focus on technical details, methodologies, and procedures
2. Explain complex concepts in clear terms
3. Include specific technical parameters, formulas, or algorithms if mentioned
4. Highlight technical advantages, limitations, or considerations
5. Use precise technical language while remaining accessible

Technical Analysis:"""
    
    elif question_type == "comparison":
        template = """You are an expert analyst comparing information from PDF documents. Provide a detailed comparison.

Context from the PDF documents:
{context}

Question: {question}

Guidelines:
1. Identify similarities and differences clearly
2. Use structured comparison (e.g., "Similarities:" and "Differences:")
3. Include specific examples from the documents
4. Highlight key distinctions and their implications
5. Provide balanced analysis of both sides

Comparison:"""
    
    else:  # general
        template = CUSTOM_PROMPT_TEMPLATE
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def detect_question_type(question: str) -> str:
    """Detect the type of question to use appropriate prompt."""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["summarize", "summary", "overview", "main points"]):
        return "summary"
    elif any(word in question_lower for word in ["compare", "difference", "similar", "versus", "vs"]):
        return "comparison"
    elif any(word in question_lower for word in ["how", "method", "process", "technique", "algorithm", "implementation"]):
        return "technical"
    else:
        return "general"
