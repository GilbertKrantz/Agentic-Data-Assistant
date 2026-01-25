"""
Tools for the Data Scientist Agent - SQL and document retrieval tools.
Code execution is handled by Gemini's integrated code execution capability.
"""

from langchain.tools import tool


@tool
def retrieve_data_for_analysis(
    sql_query: str, description: str = "Data retrieval for analysis"
) -> str:
    """Retrieve data from the database for analysis.

    This tool queries the DuckDB database and returns data that can be
    used for further analysis with execute_python_code.

    Args:
        sql_query: SQL query to execute. Use main.fraud_train or main.fraud_test tables.
        description: Brief description of what data is being retrieved.

    Returns:
        JSON string containing the query results that can be parsed in Python.

    Available columns:
    - trans_date_trans_time: Transaction timestamp
    - cc_num: Credit card number
    - merchant: Merchant name
    - category: Transaction category
    - amt: Transaction amount
    - first, last: Customer name
    - gender: Customer gender
    - city, state, zip: Address info
    - lat, long: Coordinates
    - city_pop: City population
    - job: Customer job
    - dob: Date of birth
    - trans_num: Transaction number
    - is_fraud: Fraud label (0 or 1)

    Example queries:
    - "SELECT * FROM main.fraud_train LIMIT 100"
    - "SELECT category, AVG(amt) as avg_amt, COUNT(*) as cnt FROM main.fraud_train GROUP BY category"
    """
    from app.database.data_utils import QueryEngine
    import json

    try:
        query_engine = QueryEngine(
            collection_name="documents_new",
            database_path="data/fraud.db",
        )

        results = query_engine.query_data(
            sql_query=sql_query,
            schema_name="main",
            table_name="fraud_data",
        )

        if not results:
            return json.dumps(
                {
                    "status": "success",
                    "data": [],
                    "message": "Query returned no results",
                }
            )

        # Convert evidence objects to dict format for JSON serialization
        data_list = []
        for evidence in results:
            try:
                import ast

                row_data = ast.literal_eval(evidence.extracted_content)
                data_list.append(row_data)
            except (ValueError, SyntaxError):
                data_list.append({"raw": evidence.extracted_content})

        return json.dumps(
            {
                "status": "success",
                "data": data_list,
                "row_count": len(data_list),
                "description": description,
            }
        )

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "query": sql_query})


@tool
def retrieve_documents_for_analysis(query: str, n_results: int = 5) -> str:
    """Retrieve documents from ChromaDB for analysis context.

    Use this tool to get text data from documents that can inform your analysis.

    Args:
        query: The search query for finding relevant documents.
        n_results: Maximum number of documents to retrieve.

    Returns:
        JSON string containing the retrieved documents.
    """
    from app.database.data_utils import QueryEngine
    import json

    try:
        query_engine = QueryEngine(
            collection_name="documents_new",
            database_path="data/fraud.db",
        )

        results = query_engine.query_data(
            query_str=[query],
            n_results=n_results,
        )

        if not results:
            return json.dumps(
                {"status": "success", "documents": [], "message": "No documents found"}
            )

        documents = []
        for evidence in results:
            documents.append(
                {
                    "id": evidence.evidence_id,
                    "content": evidence.extracted_content,
                    "source": evidence.source_metadata.file_name,
                }
            )

        return json.dumps(
            {"status": "success", "documents": documents, "count": len(documents)}
        )

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool
def get_schema_info() -> str:
    """Get the database schema information.

    Use this to understand available tables and columns before writing queries.

    Returns:
        Schema description for the fraud detection database.
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
