"""
Tools for the Data Scientist Agent - Analysis and pattern recognition.
Data retrieval is handled by the Retriever Agent (separation of concerns).
Code execution uses Gemini's integrated code execution capability.
"""

from langchain.tools import tool
from app.database.retrieval_tools import (
    query_duckdb_sql,
    get_database_schema_info,
)
from typing import Optional


@tool
def analyze_data_patterns(
    analysis_description: str, sql_query: Optional[str] = None
) -> str:
    """Perform data pattern analysis using SQL queries and statistical interpretation.

    This tool is for ANALYSIS and INTERPRETATION of fraud data patterns,
    not for simple data retrieval. Use the Retriever Agent for raw data fetching.

    Args:
        analysis_description: What patterns or insights you're looking for.
        sql_query: Optional SQL query to fetch specific data for analysis.

    Returns:
        JSON string with analysis results and insights.
    """
    import json

    try:
        analysis_results = {"analysis": analysis_description, "status": "success"}

        if sql_query:
            # Execute the query to get data
            results = query_duckdb_sql(sql_query, "main", "fraud_train")

            if not results:
                analysis_results["data_status"] = "no_results"
                analysis_results["message"] = "Query returned no results for analysis"
            else:
                # Parse results for analysis
                data_list = []
                for evidence in results:
                    try:
                        import ast

                        row_data = ast.literal_eval(evidence.extracted_content)
                        data_list.append(row_data)
                    except (ValueError, SyntaxError):
                        data_list.append({"raw": evidence.extracted_content})

                analysis_results["data"] = str(data_list)
                analysis_results["row_count"] = str(len(data_list))

        return json.dumps(analysis_results)

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error_message": str(e),
                "analysis": analysis_description,
            }
        )


@tool
def get_fraud_schema() -> str:
    """Get database schema for fraud detection data.

    Use this to understand table structure before requesting SQL analysis.

    Returns:
        Schema description.
    """
    return get_database_schema_info()
