"""
Tools for the Retriever Agent - handles both ChromaDB vector search and DuckDB SQL queries.
Uses shared retrieval utilities to avoid code duplication.
"""

from langchain.tools import tool
from app.database.retrieval_tools import (
    search_documents_from_chromadb,
    query_duckdb_sql,
    get_database_schema_info,
    format_evidence_results,
)


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
    results = search_documents_from_chromadb(query, n_results)
    return format_evidence_results(results, "documents")


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
    results = query_duckdb_sql(sql_query, "main", table_name)

    if not results:
        return "Query executed successfully but returned no results."

    output_parts = [f"Query returned {len(results)} rows:\n"]
    for i, evidence in enumerate(results, 1):
        output_parts.append(f"Row {i}: {evidence.extracted_content}")

    return "\n".join(output_parts)


@tool
def get_database_schema() -> str:
    """Get the schema information for the fraud detection database.

    Use this tool to understand the structure of available tables
    before writing SQL queries.

    Returns:
        A description of the available tables and their columns.
    """
    return get_database_schema_info()
