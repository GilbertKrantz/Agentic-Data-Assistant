from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.data_utils import QueryEngine


def main() -> None:
    # Data Query Testing
    query_engine = QueryEngine(
        collection_name="documents_new",
        database_path="data/fraud.db",
    )

    # result = query_engine.query_data(
    #     ["fraud detection techniques"],
    #     n_results=3,
    # )

    # for evidence in result:
    #     print("Evidence ID:", evidence.evidence_id)
    #     print("Content:", evidence.extracted_content)
    #     print("Metadata:", evidence.source_metadata)
    #     print("-----")

    result_sql = query_engine.query_data(
        sql_query="SELECT * FROM main.fraud_train LIMIT 5;",
    )

    for evidence in result_sql:
        print("Evidence ID:", evidence.evidence_id)
        print("Content:", evidence.extracted_content)
        print("Metadata:", evidence.source_metadata)
        print("=====")


if __name__ == "__main__":
    main()
