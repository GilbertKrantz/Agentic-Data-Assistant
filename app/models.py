from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FileType(str, Enum):
    DOCUMENT = "document"
    STRUCTURED_DATA = "structured_data"


class SourceMetadata(BaseModel):
    file_id: str
    file_name: str
    file_type: Optional[FileType] = None


class DocumentLocation(BaseModel):
    page_number: Optional[int] = None
    paragraph_index: Optional[int] = None
    original_text_snippet: Optional[str] = None


class StructuredLocation(BaseModel):
    sheet_name: Optional[str] = None
    column_names: Optional[List[str]] = None
    row_indices: Optional[List[int]] = None
    filter_logic: Optional[str] = None


class ProofCoordinates(BaseModel):
    document_location: Optional[DocumentLocation] = None
    structured_location: Optional[StructuredLocation] = None


class UniversalEvidenceObject(BaseModel):
    evidence_id: Optional[str] = Field(
        None, description="Unique UUID for this specific piece of evidence."
    )
    source_metadata: SourceMetadata
    extracted_content: str = Field(
        ...,
        description="The actual text snippet or raw value extracted (e.g., '$50,000' or 'Employees must wear badges').",
    )
    proof_coordinates: ProofCoordinates = Field(
        ..., description="Polymorphic location data depending on source type."
    )


class AgentId(str, Enum):
    RETRIEVAL_AGENT = "retrieval_agent"
    DATASCIENCE_AGENT = "datascience_agent"
    INFO_CHECKER = "info_checker"
    ORCHESTRATOR = "orchestrator"


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CLARIFICATION_NEEDED = "clarification_needed"


class StandardAgentResponse(BaseModel):
    agent_id: AgentId
    status: ResponseStatus
    thought_process: Optional[str] = Field(
        None, description="Internal reasoning (CoT) for debugging. Not shown to user."
    )
    final_answer: str = Field(
        ...,
        description="The natural language answer to be displayed or passed to the next agent.",
    )
    supporting_evidence: List[UniversalEvidenceObject] = Field(
        ..., description="List of Evidence Objects that prove the final_answer."
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Self-evaluated confidence. Low confidence triggers Info Checker.",
    )
