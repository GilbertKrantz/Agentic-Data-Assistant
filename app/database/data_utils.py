from chromadb import Collection
from chromadb.api import ClientAPI
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Optional, List
from uuid import uuid4

from app.models import (
    DocumentLocation,
    FileType,
    ProofCoordinates,
    SourceMetadata,
    StructuredLocation,
    UniversalEvidenceObject,
)


class QueryEngine:
    def __init__(self, collection_name: str, database_path: str) -> None:
        self.chromadb_client: Optional[ClientAPI] = None
        self.collection: Optional[Collection] = None
        self.duckdb_client: Optional[Engine] = None

        self._initialize_chromadb(collection_name)
        self._initialize_duckdb(database_path)

    def _initialize_chromadb(self, collection_name: str) -> None:
        from app.database.chroma_utils import (
            get_chroma_client,
            get_or_create_collection,
        )

        self.chromadb_client = get_chroma_client()
        self.collection = get_or_create_collection(
            self.chromadb_client, collection_name
        )

    def _initialize_duckdb(self, database_path: str) -> None:
        from app.database.duckdb_utils import get_duckdb_engine

        self.duckdb_client = get_duckdb_engine(database_path)

    def _query_texts(
        self, query_strs: List[str], n_results: int = 5
    ) -> List[UniversalEvidenceObject]:
        from app.database.chroma_utils import query_collection_by_text

        if self.collection is not None:
            results = query_collection_by_text(
                self.collection,
                query_strs,
                n_results,
            )
            evidence_list: List[UniversalEvidenceObject] = []

            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]

            for idx, evidence_id in enumerate(ids):
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                content = documents[idx] if idx < len(documents) else ""

                source_metadata = SourceMetadata(
                    file_id=str(metadata.get("file_id", "unknown")),
                    file_name=str(metadata.get("file_name", "unknown")),
                    file_type=FileType.DOCUMENT,
                )

                document_location = DocumentLocation(
                    page_number=metadata.get("page_number"),
                    paragraph_index=metadata.get("paragraph_index"),
                    original_text_snippet=content,
                )

                evidence_list.append(
                    UniversalEvidenceObject(
                        evidence_id=str(evidence_id),
                        source_metadata=source_metadata,
                        extracted_content=content,
                        proof_coordinates=ProofCoordinates(
                            document_location=document_location
                        ),
                    )
                )

            return evidence_list
        return []

    def _query_duckdb(
        self,
        sql_query: str,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        sheet_name: Optional[str] = None,
        index_column: Optional[str] = None,
    ) -> List[UniversalEvidenceObject]:
        if self.duckdb_client is not None:
            with self.duckdb_client.connect() as connection:
                result = connection.execute(text(sql_query))
                column_names = list(result.keys())
                rows = result.fetchall()

                evidence_list: List[UniversalEvidenceObject] = []
                for row in rows:
                    row_dict = dict(zip(column_names, row))
                    content = str(row_dict)
                    row_index_value = (
                        row_dict.get(index_column)
                        if index_column and index_column in row_dict
                        else None
                    )
                    row_indices = (
                        [row_index_value] if row_index_value is not None else None
                    )

                    source_metadata = SourceMetadata(
                        file_id=str(schema_name or "unknown"),
                        file_name=str(table_name or "structured_query"),
                        file_type=FileType.STRUCTURED_DATA,
                    )

                    structured_location = StructuredLocation(
                        column_names=column_names,
                        row_indices=row_indices,
                        filter_logic=sql_query,
                        sheet_name=sheet_name,
                    )

                    evidence_list.append(
                        UniversalEvidenceObject(
                            evidence_id=str(uuid4()),
                            source_metadata=source_metadata,
                            extracted_content=content,
                            proof_coordinates=ProofCoordinates(
                                structured_location=structured_location
                            ),
                        )
                    )

                    print(evidence_list)

                return evidence_list
        return []

    def query_data(
        self,
        query_str: Optional[List[str]] = None,
        sql_query: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        sheet_name: Optional[str] = None,
        index_column: Optional[str] = None,
        n_results: int = 5,
    ) -> List[UniversalEvidenceObject]:
        if query_str is not None:
            return self._query_texts(query_str, n_results)
        elif sql_query is not None:
            return self._query_duckdb(
                sql_query,
                schema_name=schema_name,
                table_name=table_name,
                sheet_name=sheet_name,
                index_column=index_column,
            )
        else:
            return []
