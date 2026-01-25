"""
Shared retrieval tools for all agents.
Provides unified interface for document and database queries.
Eliminates duplication between Retriever and Data Scientist agents.
"""

from typing import List, Optional, Dict, Any
from app.database.data_utils import QueryEngine
from app.models import UniversalEvidenceObject
import json


# Global query engine instance (singleton pattern)
_query_engine: Optional[QueryEngine] = None


def get_query_engine() -> QueryEngine:
    """Get or create the query engine singleton instance."""
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine(
            collection_name="documents_new",
            database_path="data/fraud.db",
        )
    return _query_engine


def search_documents_from_chromadb(
    query: str, n_results: int = 5
) -> List[UniversalEvidenceObject]:
    """Search for documents in ChromaDB vector database.

    Args:
        query: The search query for finding relevant documents.
        n_results: Maximum number of documents to retrieve.

    Returns:
        List of UniversalEvidenceObject containing the search results.
    """
    query_engine = get_query_engine()
    results: List[UniversalEvidenceObject] = query_engine.query_data(
        query_str=[query],
        n_results=n_results,
    )
    return results if results else []


def query_duckdb_sql(
    sql_query: str, schema_name: str = "main", table_name: str = "fraud_train"
) -> List[UniversalEvidenceObject]:
    """Execute SQL query against DuckDB database.

    Args:
        sql_query: The SQL query to execute.
        schema_name: The schema name (default: "main").
        table_name: The table name for metadata purposes.

    Returns:
        List of UniversalEvidenceObject containing the query results.
    """
    query_engine = get_query_engine()
    try:
        results: List[UniversalEvidenceObject] = query_engine.query_data(
            sql_query=sql_query,
            schema_name=schema_name,
            table_name=table_name,
        )
        return results if results else []
    except Exception as e:
        # Return empty list on error; caller can handle
        print(f"SQL Query Error: {str(e)}")
        return []


def get_database_schema_info() -> str:
    """Get the database schema information for all tables.

    Returns:
        Formatted string describing available tables and columns.
    """
    schema_info = """
Available Tables in DuckDB (main schema):

1. fraud_train - Training data for fraud detection

Both tables have the same columns:
- trans_date_trans_time (VARCHAR): Transaction timestamp
- cc_num (BIGINT): Credit card number  
- merchant (VARCHAR): Merchant name
- category (VARCHAR): Transaction category (e.g., 'grocery_pos', 'shopping_net')
- amt (DOUBLE): Transaction amount in USD
- first (VARCHAR): Customer first name
- last (VARCHAR): Customer last name
- gender (VARCHAR): Customer gender (M/F)
- street (VARCHAR): Customer street address
- city (VARCHAR): Customer city
- state (VARCHAR): Customer state
- zip (INTEGER): Customer ZIP code
- lat (DOUBLE): Customer latitude
- long (DOUBLE): Customer longitude
- city_pop (INTEGER): City population
- job (VARCHAR): Customer occupation
- dob (DATE): Customer date of birth
- trans_num (VARCHAR): Unique transaction ID
- unix_time (BIGINT): Unix timestamp
- merch_lat (DOUBLE): Merchant latitude
- merch_long (DOUBLE): Merchant longitude
- is_fraud (INTEGER): Fraud indicator (0 = legitimate, 1 = fraud)

Example queries:
- SELECT COUNT(*) FROM main.fraud_train WHERE is_fraud = 1;
- SELECT category, AVG(amt) FROM main.fraud_train GROUP BY category;
- SELECT * FROM main.fraud_train LIMIT 10;
"""
    return schema_info


def format_evidence_results(
    evidence_list: List[UniversalEvidenceObject], result_type: str = "evidence"
) -> str:
    """Format evidence results as a readable string.

    Args:
        evidence_list: List of UniversalEvidenceObject to format.
        result_type: Type of results ("documents", "sql_results", etc).

    Returns:
        Formatted string representation of results.
    """
    if not evidence_list:
        return f"No {result_type} found."

    output_parts = []
    for i, evidence in enumerate(evidence_list, 1):
        output_parts.append(f"--- Result {i} ---")
        output_parts.append(f"Evidence ID: {evidence.evidence_id}")
        output_parts.append(f"Content: {evidence.extracted_content}")
        output_parts.append(f"Source: {evidence.source_metadata.file_name}")
        if evidence.proof_coordinates.document_location:
            loc = evidence.proof_coordinates.document_location
            if loc.page_number:
                output_parts.append(f"Page: {loc.page_number}")
            if loc.section:
                output_parts.append(f"Section: {loc.section}")
        output_parts.append("")

    return "\n".join(output_parts)


def convert_evidence_to_json(
    evidence_list: List[UniversalEvidenceObject],
) -> Dict[str, Any]:
    """Convert evidence objects to JSON-serializable format.

    Args:
        evidence_list: List of UniversalEvidenceObject.

    Returns:
        Dictionary with evidence data suitable for JSON serialization.
    """
    documents = []
    for evidence in evidence_list:
        documents.append(
            {
                "id": evidence.evidence_id,
                "content": evidence.extracted_content,
                "source": evidence.source_metadata.file_name,
            }
        )
    return {"documents": documents, "count": len(documents)}
