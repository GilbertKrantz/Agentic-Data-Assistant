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


# Known sections from the PDF Table of Contents
KNOWN_SECTIONS = [
    "INTRODUCTION",
    "FRAUD TECHNIQUES",
    "IMPACT OF CREDIT CARD FRAUDS",
    "FRAUD PREVENTION AND MANAGEMENT",
    "CONCLUSION",
]

# Known subsections from the PDF
KNOWN_SUBSECTIONS = [
    "Card Related Frauds",
    "Merchant Related Frauds",
    "Internet Related Frauds",
    "Application Fraud",
    "Lost/Stolen Cards",
    "Account Takeover",
    "Skimming",
    "Site Cloning",
    "False Merchant Sites",
    "Credit Card Generators",
]

# Taxonomy mapping: CSV categories to PDF sections
# This bridges structured data (fraudTrain.csv) with unstructured data (PDF)
CATEGORY_TO_SECTION_MAPPING = {
    # Internet-related categories -> Internet Related Frauds
    "shopping_net": "Internet Related Frauds",
    "misc_net": "Internet Related Frauds",
    "entertainment": "Internet Related Frauds",
    # POS (Point of Sale) categories -> Card Related Frauds
    "grocery_pos": "Card Related Frauds",
    "gas_transport": "Card Related Frauds",
    "shopping_pos": "Card Related Frauds",
    "food_dining": "Card Related Frauds",
    "health_fitness": "Card Related Frauds",
    "travel": "Card Related Frauds",
    "kids_pets": "Card Related Frauds",
    "home": "Card Related Frauds",
    "personal_care": "Card Related Frauds",
    # Miscellaneous POS -> Merchant Related Frauds (potential collusion)
    "misc_pos": "Merchant Related Frauds",
}


def get_section_for_category(category: str) -> str:
    """
    Get the PDF section that corresponds to a CSV category.

    Args:
        category: The category from fraudTrain.csv

    Returns:
        The corresponding PDF subsection name, or "General" if no mapping exists
    """
    return CATEGORY_TO_SECTION_MAPPING.get(category.lower(), "General")


def parse_pdf_with_unstructured(pdf_path: str) -> list[dict]:
    """
    Parse a PDF file using unstructured library and return elements with metadata.
    DEPRECATED: Use parse_pdf_with_sections for section-aware parsing.

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


def parse_pdf_with_sections(pdf_path: str, document_year: int = 2003) -> list[dict]:
    """
    Parse a PDF file with section-aware metadata extraction.

    Detects major sections and subsections from the document hierarchy
    and attaches them as metadata to each element. This enables more
    accurate RAG retrieval for section-specific queries.

    Args:
        pdf_path: Path to the PDF file
        document_year: The publication year of the document (for temporal context)

    Returns:
        List of dictionaries containing text content with section metadata
    """
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(pdf_path)
    parsed_elements = []

    current_section = "General"
    current_subsection = None

    for idx, element in enumerate(
        tqdm(elements, desc="Parsing elements with sections")
    ):
        text = str(element).strip()
        element_type = type(element).__name__

        # Get element category for table detection
        element_category = getattr(element, "category", element_type)
        is_table = element_category == "Table" or element_type == "Table"

        # 1. Detect Major Sections (All Caps in the PDF)
        text_upper = text.upper()
        if text_upper in KNOWN_SECTIONS:
            current_section = text_upper
            current_subsection = None  # Reset subsection when entering new section

        # 2. Detect Subsections (Title Case in PDF)
        elif text in KNOWN_SUBSECTIONS:
            current_subsection = text
        # Also check for partial matches (sometimes OCR adds extra characters)
        else:
            for subsection in KNOWN_SUBSECTIONS:
                if (
                    subsection.lower() in text.lower()
                    and len(text) < len(subsection) + 20
                ):
                    current_subsection = subsection
                    break

        # 3. Build the parsed element with rich metadata
        parsed_elements.append(
            {
                "text": text,
                "element_type": element_type,
                "element_category": element_category,
                "is_table": is_table,
                "page_number": (
                    element.metadata.page_number
                    if hasattr(element.metadata, "page_number")
                    else None
                ),
                "element_id": str(uuid4()),
                "paragraph_index": idx,
                "section": current_section,
                "subsection": current_subsection,
                "year": document_year,
                "source_type": "academic_paper",
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

    IMPORTANT: Tables are preserved as whole documents and not chunked to
    maintain their structural integrity (row/column relationships).

    Uses RecursiveCharacterTextSplitter to respect semantic boundaries
    (paragraphs, sentences, words) when splitting text.

    Args:
        elements: List of parsed elements from PDF (with section metadata)
        chunk_size: Target size for each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of chunked elements with combined text and metadata
    """
    chunks: list[dict] = []

    # First, extract tables as whole documents (don't chunk them)
    table_elements = [e for e in elements if e.get("is_table", False)]
    non_table_elements = [e for e in elements if not e.get("is_table", False)]

    # Add tables as individual chunks (preserve whole structure)
    for table_elem in tqdm(table_elements, desc="Processing tables"):
        table_text = table_elem.get("text", "").strip()
        if table_text:
            chunks.append(
                {
                    "text": table_text,
                    "page_number": table_elem.get("page_number"),
                    "paragraph_index": table_elem.get("paragraph_index"),
                    "chunk_id": str(uuid4()),
                    "section": table_elem.get("section", "General"),
                    "subsection": table_elem.get("subsection"),
                    "year": table_elem.get("year"),
                    "source_type": table_elem.get("source_type"),
                    "is_table": True,
                }
            )

    print(f"Preserved {len(table_elements)} tables as whole documents")

    # Build section-aware text segments (group by page AND section)
    page_section_segments: list[dict] = []
    current_text = ""
    current_page = None
    current_section = None
    current_subsection = None
    current_year = None
    current_source_type = None
    start_paragraph = 0

    for idx, element in enumerate(tqdm(non_table_elements, desc="Building segments")):
        element_text = element.get("text", "").strip()
        if not element_text:
            continue

        page = element.get("page_number")
        section = element.get("section", "General")
        subsection = element.get("subsection")
        year = element.get("year")
        source_type = element.get("source_type")

        # If page or section changes, save current segment and start new one
        section_changed = current_section is not None and section != current_section
        page_changed = current_page is not None and page != current_page

        if section_changed or page_changed:
            if current_text.strip():
                page_section_segments.append(
                    {
                        "text": current_text.strip(),
                        "page_number": current_page,
                        "paragraph_index": start_paragraph,
                        "section": current_section,
                        "subsection": current_subsection,
                        "year": current_year,
                        "source_type": current_source_type,
                    }
                )
            current_text = ""
            start_paragraph = idx

        if current_page is None:
            start_paragraph = idx

        current_page = page
        current_section = section
        current_subsection = subsection
        current_year = year
        current_source_type = source_type
        current_text += " " + element_text

    # Add final segment
    if current_text.strip():
        page_section_segments.append(
            {
                "text": current_text.strip(),
                "page_number": current_page,
                "paragraph_index": start_paragraph,
                "section": current_section,
                "subsection": current_subsection,
                "year": current_year,
                "source_type": current_source_type,
            }
        )

    # Create sentence-aware splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
    )

    # Split each segment and preserve all metadata including section info
    for segment in tqdm(page_section_segments, desc="Splitting segments"):
        split_texts = text_splitter.split_text(segment["text"])
        for split_text in split_texts:
            chunks.append(
                {
                    "text": split_text,
                    "page_number": segment["page_number"],
                    "paragraph_index": segment["paragraph_index"],
                    "chunk_id": str(uuid4()),
                    "section": segment.get("section", "General"),
                    "subsection": segment.get("subsection"),
                    "year": segment.get("year"),
                    "source_type": segment.get("source_type"),
                    "is_table": False,
                }
            )

    return chunks


def ingest_pdf_to_chromadb(
    pdf_path: str,
    collection_name: str = "documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    document_year: int = 2003,
) -> list[UniversalEvidenceObject]:
    """
    Ingest a PDF file into ChromaDB collection with section-aware chunking.

    Uses section-aware parsing to detect document hierarchy (sections/subsections)
    and preserves tables as whole documents. Includes temporal context (year)
    to help the LLM distinguish historical vs current data.

    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the ChromaDB collection
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between chunks
        document_year: Publication year of the document (for temporal context)

    Returns:
        List of UniversalEvidenceObject representing ingested chunks
    """
    pdf_file_path = Path(pdf_path)
    file_name = pdf_file_path.name
    file_id = pdf_file_path.stem

    # Parse PDF with section-aware extraction
    print(f"Parsing PDF with section awareness: {pdf_file_path}")
    elements = parse_pdf_with_sections(str(pdf_file_path), document_year=document_year)
    print(f"Found {len(elements)} elements")

    # Log detected sections for verification
    sections_found = set(e.get("section") for e in elements if e.get("section"))
    subsections_found = set(
        e.get("subsection") for e in elements if e.get("subsection")
    )
    tables_found = sum(1 for e in elements if e.get("is_table", False))
    print(f"Detected sections: {sections_found}")
    print(f"Detected subsections: {subsections_found}")
    print(f"Detected tables: {tables_found}")

    # Chunk elements (tables are preserved whole)
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

        # Include section metadata in DocumentLocation
        document_location = DocumentLocation(
            page_number=chunk.get("page_number"),
            paragraph_index=chunk.get("paragraph_index"),
            original_text_snippet=(
                chunk["text"][:200] + "..."
                if len(chunk["text"]) > 200
                else chunk["text"]
            ),
            section=chunk.get("section"),
            subsection=chunk.get("subsection"),
            year=chunk.get("year"),
            source_type=chunk.get("source_type"),
            is_table=chunk.get("is_table", False),
        )

        evidence = UniversalEvidenceObject(
            evidence_id=chunk_id,
            source_metadata=source_metadata,
            extracted_content=chunk["text"],
            proof_coordinates=ProofCoordinates(document_location=document_location),
        )
        evidence_objects.append(evidence)

        documents.append(chunk["text"])
        # Include all metadata for ChromaDB filtering
        metadatas.append(
            {
                "evidence_id": chunk_id,
                "file_id": file_id,
                "file_name": file_name,
                "file_type": FileType.DOCUMENT.value,
                "page_number": chunk.get("page_number"),
                "paragraph_index": chunk.get("paragraph_index"),
                # Section-aware metadata for targeted retrieval
                "section": chunk.get("section", "General"),
                "subsection": chunk.get("subsection")
                or "",  # ChromaDB doesn't like None
                # Temporal context for historical data awareness
                "year": chunk.get("year") or document_year,
                "source_type": chunk.get("source_type", "academic_paper"),
                # Table flag for special handling
                "is_table": chunk.get("is_table", False),
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
) -> None:
    """
    Ingest a CSV file into DuckDB using native efficient loading.
    """
    csv_file_path = Path(csv_path)
    print(f"Loading CSV directly into DuckDB: {csv_file_path}")

    # Get DuckDB engine
    engine = get_duckdb_engine(database_path)
    full_table_name = f"{schema_name}.{table_name}"

    with engine.connect() as conn:
        # Create schema if not exists
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))

        # 1. Create table directly from CSV (Zero-copy ingestion)
        # We use read_csv with auto_detect=True.
        # The 'union_by_name=True' helps if columns perform schema evolution later.
        print(f"Executing Native DuckDB COPY into {full_table_name}...")

        query = text(
            f"""
            CREATE OR REPLACE TABLE {full_table_name} AS 
            SELECT * FROM read_csv(
                :path, 
                header=True, 
                auto_detect=True,
                filename=True  -- Optional: adds a column with source filename
            )
        """
        )

        conn.execute(query, {"path": str(csv_file_path)})

        # 2. Handle the "Unnamed: 0" index column if it exists
        # It's faster to rename it via SQL than Pandas
        # Check if column exists
        columns_result = conn.execute(text(f"DESCRIBE {full_table_name}")).fetchall()
        print(columns_result)
        print(full_table_name)
        column_names = [row[0] for row in columns_result]

        if "Unnamed: 0" in column_names:
            print("Renaming index column...")
            conn.execute(
                text(
                    f"""
                ALTER TABLE {full_table_name} 
                RENAME COLUMN "Unnamed: 0" TO "index_id"
            """
                )
            )

        # Verify count
        count = conn.execute(text(f"SELECT COUNT(*) FROM {full_table_name}")).scalar()
        conn.commit()
        print(f"Successfully ingested {count} rows.")


def run_full_ingestion(
    pdf_path: str = "data/Bhatla.pdf",
    csv_path: str = "data/fraudTrain.csv",
    database_path: str = "data/fraud.db",
    collection_name: str = "documents",
    table_name: str = "fraud_train",
    document_year: int = 2003,
) -> None:
    """
    Run the full ingestion pipeline for both PDF and CSV data.

    Uses section-aware PDF parsing to capture document hierarchy and
    preserves tables as whole documents. Includes temporal context
    for historical data awareness.

    Args:
        pdf_path: Path to the PDF file
        csv_path: Path to the CSV file
        database_path: Path to the DuckDB database
        collection_name: Name of the ChromaDB collection
        table_name: Name of the DuckDB table
        document_year: Publication year of the PDF document (for temporal context)
    """
    print("=" * 60)
    print("Starting Full Ingestion Pipeline")
    print("=" * 60)

    # Ingest PDF to ChromaDB with section-aware parsing
    print("\n[1/2] Ingesting PDF to ChromaDB (section-aware)...")
    print("-" * 40)
    evidence_objects = ingest_pdf_to_chromadb(
        pdf_path=pdf_path,
        collection_name=collection_name,
        document_year=document_year,
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
    )

    print("\n" + "=" * 60)
    print("Full Ingestion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    ingest_csv_to_duckdb(
        csv_path="data/fraudTrain.csv",
        database_path="data/fraud.db",
        table_name="fraud_train",
        schema_name="main",
    )
