"""
Tools for the Retriever Agent - handles both ChromaDB vector search and DuckDB SQL queries.
"""

from typing import List, Optional
from langchain.tools import tool
from app.database.data_utils import QueryEngine
from app.models import UniversalEvidenceObject


# Global query engine instance (initialized lazily)
_query_engine: Optional[QueryEngine] = None


def get_query_engine() -> QueryEngine:
    """Get or create the query engine instance."""
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine(
            collection_name="documents_new",
            database_path="data/fraud.db",
        )
    return _query_engine


@tool
def search_documents(query: str, n_results: int = 5) -> str:
    """Search for relevant documents in the ChromaDB vector database.

    Use this tool when you need to find information from documents, PDFs,
    or unstructured text data. It performs semantic search to find the
    most relevant content.

    Args:
        query: The search query describing what information you're looking for.
        n_results: Maximum number of results to return (default: 5).

    Returns:
        A formatted string containing the search results with evidence details.
    """
    query_engine = get_query_engine()
    results: List[UniversalEvidenceObject] = query_engine.query_data(
        query_str=[query],
        n_results=n_results,
    )

    if not results:
        return "No documents found matching the query."

    output_parts = []
    for i, evidence in enumerate(results, 1):
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


@tool
def query_sql_database(sql_query: str, table_name: str = "fraud_train") -> str:
    """Execute a SQL query against the DuckDB database.

    Use this tool when you need to query structured data from the fraud
    detection database. The main tables available are 'fraud_train' and
    'fraud_test' in the 'main' schema.

    Available columns in fraud tables:
    - trans_date_trans_time: Transaction timestamp
    - cc_num: Credit card number
    - merchant: Merchant name
    - category: Transaction category
    - amt: Transaction amount
    - first, last: Customer name
    - gender: Customer gender
    - street, city, state, zip: Address info
    - lat, long: Coordinates
    - city_pop: City population
    - job: Customer job
    - dob: Date of birth
    - trans_num: Transaction number
    - unix_time: Unix timestamp
    - merch_lat, merch_long: Merchant coordinates
    - is_fraud: Fraud label (0 or 1)

    Args:
        sql_query: The SQL query to execute.
        table_name: The table name for metadata purposes.

    Returns:
        A formatted string containing the query results.
    """
    query_engine = get_query_engine()

    try:
        results: List[UniversalEvidenceObject] = query_engine.query_data(
            sql_query=sql_query,
            schema_name="main",
            table_name=table_name,
        )

        if not results:
            return "Query executed successfully but returned no results."

        output_parts = [f"Query returned {len(results)} rows:\n"]
        for i, evidence in enumerate(results, 1):
            output_parts.append(f"Row {i}: {evidence.extracted_content}")

        return "\n".join(output_parts)

    except Exception as e:
        return f"SQL Error: {str(e)}"


@tool
def get_database_schema() -> str:
    """Get the schema information for the fraud detection database.

    Use this tool to understand the structure of available tables
    before writing SQL queries.

    Returns:
        A description of the available tables and their columns.
    """
    schema_info = """
Available Tables in DuckDB (main schema):

1. fraud_train - Training data for fraud detection
2. fraud_test - Test data for fraud detection

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
