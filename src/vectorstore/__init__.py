# Vector-store package for PECS
from src.vectorstore.config import get_chroma_db_path
from src.vectorstore.init import COLLECTION_NAME, get_or_create_collection
from src.vectorstore.document_format import serialize_document, deserialize_document
from src.vectorstore.metadata_schema import build_metadata, validate_metadata
from src.vectorstore.ingest import ingest_record
from src.vectorstore.client import get_client
from src.vectorstore.upsert import upsert_record
from src.vectorstore.retrieve import retrieve
from src.vectorstore.filtered_retrieve import filtered_retrieve
from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

__all__ = [
    "get_chroma_db_path",
    "COLLECTION_NAME",
    "get_or_create_collection",
    "serialize_document",
    "deserialize_document",
    "build_metadata",
    "validate_metadata",
    "ingest_record",
    "get_client",
    "upsert_record",
    "retrieve",
    "filtered_retrieve",
    "bulk_ingest_from_serialized",
]
