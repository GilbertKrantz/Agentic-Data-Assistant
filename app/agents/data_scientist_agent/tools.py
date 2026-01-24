"""
Tools for the Data Scientist Agent - handles code execution for data analysis.
All data queries go through retrieval tools.
"""

import sys
import traceback
from io import StringIO
from langchain.tools import tool


@tool
def execute_python_code(code: str) -> str:
    """Execute Python code for data analysis and return the output.

    Use this tool to run Python code for:
    - Statistical analysis on retrieved data
    - Data visualization
    - Machine learning model training/inference
    - Data manipulation with pandas
    - Numerical computations

    IMPORTANT: Use retrieve_data_for_analysis first to get data, then pass
    that data to this tool for analysis.

    The following libraries are pre-imported in the execution environment:
    - pandas as pd
    - numpy as np
    - matplotlib.pyplot as plt
    - seaborn as sns
    - sklearn (scikit-learn)
    - scipy.stats
    - json, ast for parsing retrieved data

    Args:
        code: The Python code to execute. Should be complete and runnable.

    Returns:
        The stdout output from the code execution, or error message if failed.

    Example:
        code = '''
        import pandas as pd
        import json
        # Parse data retrieved from retrieve_data_for_analysis
        data = json.loads(retrieved_json)
        df = pd.DataFrame(data['data'])
        print(df.describe())
        '''
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    # Pre-import common data science libraries
    exec_globals = {}
    setup_code = """
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
import ast

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except ImportError:
    sklearn = None
"""

    try:
        # Execute setup code to make libraries available
        exec(setup_code, exec_globals)

        # Execute user code
        exec(code, exec_globals)

        # Get output
        output = captured_output.getvalue()

        # Check if any plots were created and save them
        if exec_globals.get("plt") is not None:
            try:
                fig = exec_globals["plt"].gcf()
                if fig.get_axes():
                    output += "\n[Plot generated - saved to output]"
                    exec_globals["plt"].close("all")
            except Exception:
                pass

        if not output:
            output = "Code executed successfully (no output printed)"

        return output

    except Exception as e:
        error_msg = f"Execution Error: {type(e).__name__}: {str(e)}\n"
        error_msg += traceback.format_exc()
        return error_msg

    finally:
        sys.stdout = old_stdout


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
