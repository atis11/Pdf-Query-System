"""
Intelligent PDF Reader - Utilities Package (LangChain Version)

This package contains utility modules for PDF processing, indexing, and querying using LangChain.
"""

from .pdf_parser import extract_text_from_pdf
from .index_builder import convert_to_documents, build_index, load_index
from .query_engine import load_query_engine, ask_question
from .metadata_extractor import extract_pdf_metadata

__all__ = [
    'extract_text_from_pdf',
    'convert_to_documents',
    'build_index', 
    'load_index',
    'load_query_engine',
    'ask_question',
    'extract_pdf_metadata'
]

__version__ = "2.0.0" 