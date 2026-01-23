"""
Ingest data into DuckDB database and ChromaDB vector database.

This module provides functions to:
1. Parse and chunk PDF documents using unstructured
2. Store document chunks in ChromaDB with Gemini embeddings
3. Load CSV data into DuckDB tables
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from uuid import uuid4

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from app.database.chroma_utils import (
    add_documents_to_collection,
    get_chroma_client,
    get_or_create_collection,
)
from app.database.duckdb_utils import get_duckdb_engine
from app.models import (
    DocumentLocation,
    FileType,
    ProofCoordinates,
    SourceMetadata,
    UniversalEvidenceObject,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_pdf_with_unstructured(pdf_path: str) -> list[dict]:
    """
    Parse a PDF file using unstructured library and return elements with metadata.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing text content and metadata (page number, etc.)
    """
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(pdf_path)

    parsed_elements = []
    for idx, element in enumerate(tqdm(elements, desc="Parsing elements")):
        parsed_elements.append(
            {
                "text": str(element),
                "element_type": type(element).__name__,
                "page_number": (
                    element.metadata.page_number
                    if hasattr(element.metadata, "page_number")
                    else None
                ),
                "element_id": str(uuid4()),
                "paragraph_index": idx,
            }
        )

    return parsed_elements


def chunk_elements(
    elements: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Combine small elements into larger chunks using sentence-aware splitting.

    Uses RecursiveCharacterTextSplitter to respect semantic boundaries
    (paragraphs, sentences, words) when splitting text.

    Args:
        elements: List of parsed elements from PDF
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of chunked elements with combined text and metadata
    """

    # Build page-aware text segments
    page_segments: list[dict] = []
    current_text = ""
    current_page = None
    start_paragraph = 0

    for idx, element in enumerate(tqdm(elements, desc="Building segments")):
        element_text = element.get("text", "").strip()
        if not element_text:
            continue

        page = element.get("page_number")

        # If page changes, save current segment and start new one
        if current_page is not None and page != current_page:
            if current_text.strip():
                page_segments.append(
                    {
                        "text": current_text.strip(),
                        "page_number": current_page,
                        "paragraph_index": start_paragraph,
                    }
                )
            current_text = ""
            start_paragraph = idx

        if current_page is None:
            start_paragraph = idx

        current_page = page
        current_text += " " + element_text

    # Add final segment
    if current_text.strip():
        page_segments.append(
            {
                "text": current_text.strip(),
                "page_number": current_page,
                "paragraph_index": start_paragraph,
            }
        )

    # Create sentence-aware splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )

    # Split each page segment and preserve metadata
    chunks: list[dict] = []
    for segment in tqdm(page_segments, desc="Splitting segments"):
        split_texts = text_splitter.split_text(segment["text"])
        for split_text in split_texts:
            chunks.append(
                {
                    "text": split_text,
                    "page_number": segment["page_number"],
                    "paragraph_index": segment["paragraph_index"],
                    "chunk_id": str(uuid4()),
                }
            )

    return chunks


def ingest_pdf_to_chromadb(
    pdf_path: str,
    collection_name: str = "documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[UniversalEvidenceObject]:
    """
    Ingest a PDF file into ChromaDB collection.

    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the ChromaDB collection
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of UniversalEvidenceObject representing ingested chunks
    """
    pdf_file_path = Path(pdf_path)
    file_name = pdf_file_path.name
    file_id = pdf_file_path.stem

    # Parse PDF
    print(f"Parsing PDF: {pdf_file_path}")
    elements = parse_pdf_with_unstructured(str(pdf_file_path))
    print(f"Found {len(elements)} elements")

    # Chunk elements
    chunks = chunk_elements(elements, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks")

    # Prepare data for ChromaDB
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    evidence_objects: list[UniversalEvidenceObject] = []

    for chunk in tqdm(chunks, desc="Preparing chunks"):
        chunk_id = chunk["chunk_id"]

        source_metadata = SourceMetadata(
            file_id=file_id,
            file_name=file_name,
            file_type=FileType.DOCUMENT,
        )

        document_location = DocumentLocation(
            page_number=chunk.get("page_number"),
            paragraph_index=chunk.get("paragraph_index"),
            original_text_snippet=(
                chunk["text"][:200] + "..."
                if len(chunk["text"]) > 200
                else chunk["text"]
            ),
        )

        evidence = UniversalEvidenceObject(
            evidence_id=chunk_id,
            source_metadata=source_metadata,
            extracted_content=chunk["text"],
            proof_coordinates=ProofCoordinates(document_location=document_location),
        )
        evidence_objects.append(evidence)

        documents.append(chunk["text"])
        metadatas.append(
            {
                "evidence_id": chunk_id,
                "file_id": file_id,
                "file_name": file_name,
                "file_type": FileType.DOCUMENT.value,
                "page_number": chunk.get("page_number"),
                "paragraph_index": chunk.get("paragraph_index"),
            }
        )
        ids.append(chunk_id)

    # Add to ChromaDB
    print(
        f"Adding {len(documents)} documents to ChromaDB collection '{collection_name}'"
    )
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)

    # Add in batches to avoid rate limits
    batch_size = 50
    for i in tqdm(
        range(0, len(documents), batch_size), desc="Adding documents to ChromaDB"
    ):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        add_documents_to_collection(collection, batch_docs, batch_meta, batch_ids)

    print(f"Successfully ingested {len(documents)} chunks from {file_name}")
    return evidence_objects


def ingest_csv_to_duckdb(
    csv_path: str,
    database_path: str,
    table_name: str,
    schema_name: str = "main",
    index_column: Optional[str] = None,
    rename_index_to: str = "index",
) -> None:
    """
    Ingest a CSV file into DuckDB table.

    Args:
        csv_path: Path to the CSV file
        database_path: Path to the DuckDB database file
        table_name: Name of the table to create
        schema_name: Schema name (default: main)
        index_column: Original index column name to rename
        rename_index_to: New name for the index column
    """
    csv_file_path = Path(csv_path)
    print(f"Loading CSV: {csv_file_path}")

    # Load CSV with pandas, treating first column as index if it's unnamed
    # This handles CSVs that have a row index saved as the first column
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Handle index column:
    # 1. If first column is "Unnamed: 0", treat it as the row index
    # 2. If a specific index_column is provided, use that
    # 3. Otherwise, create a new index column
    if "Unnamed: 0" in df.columns:
        # First column is an unnamed index - reset and rename
        df = df.rename(columns={"Unnamed: 0": rename_index_to})
        print(f"Renamed 'Unnamed: 0' to '{rename_index_to}'")
    elif index_column and index_column in df.columns:
        df = df.rename(columns={index_column: rename_index_to})
        print(f"Renamed column '{index_column}' to '{rename_index_to}'")
    elif rename_index_to not in df.columns:
        # No index column exists, create one from DataFrame index
        df = df.reset_index(names=[rename_index_to])
        print(f"Created new '{rename_index_to}' column from row numbers")

    print(f"Final schema: {len(df.columns)} columns")

    # Get DuckDB engine
    engine = get_duckdb_engine(database_path)

    # Create table and insert data
    full_table_name = f"{schema_name}.{table_name}"
    print(f"Creating table: {full_table_name}")

    with engine.connect() as conn:
        # Drop table if exists
        conn.execute(text(f"DROP TABLE IF EXISTS {full_table_name}"))
        conn.commit()

    # Use pandas to_sql for table creation
    df.to_sql(
        table_name,
        engine,
        schema=schema_name,
        if_exists="replace",
        index=False,
    )

    print(f"Successfully ingested {len(df)} rows into {full_table_name}")

    # Verify
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {full_table_name}"))
        count = result.scalar()
        print(f"Verified: {count} rows in table")


def run_full_ingestion(
    pdf_path: str = "data/Bhatla.pdf",
    csv_path: str = "data/fraudTrain.csv",
    database_path: str = "data/fraud.duckdb",
    collection_name: str = "documents",
    table_name: str = "fraud_train",
) -> None:
    """
    Run the full ingestion pipeline for both PDF and CSV data.

    Args:
        pdf_path: Path to the PDF file
        csv_path: Path to the CSV file
        database_path: Path to the DuckDB database
        collection_name: Name of the ChromaDB collection
        table_name: Name of the DuckDB table
    """
    print("=" * 60)
    print("Starting Full Ingestion Pipeline")
    print("=" * 60)

    # Ingest PDF to ChromaDB
    print("\n[1/2] Ingesting PDF to ChromaDB...")
    print("-" * 40)
    evidence_objects = ingest_pdf_to_chromadb(
        pdf_path=pdf_path,
        collection_name=collection_name,
    )
    print(f"Created {len(evidence_objects)} evidence objects from PDF")

    # Ingest CSV to DuckDB
    print("\n[2/2] Ingesting CSV to DuckDB...")
    print("-" * 40)
    ingest_csv_to_duckdb(
        csv_path=csv_path,
        database_path=database_path,
        table_name=table_name,
        schema_name="main",
        index_column="Unnamed: 0",  # The CSV has an unnamed index column
        rename_index_to="index",
    )

    print("\n" + "=" * 60)
    print("Full Ingestion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    ingest_csv_to_duckdb(
        csv_path="data/fraudTrain.csv",
        database_path="data/fraud.duckdb",
        table_name="fraud_train",
        schema_name="main",
        index_column="Unnamed: 0",  # The CSV has an unnamed index column
        rename_index_to="index",
    )
