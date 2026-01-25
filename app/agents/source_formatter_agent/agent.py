"""
Source Formatter Agent - Handles formatting of evidence and source citations.
"""

import json
from typing import List, Optional
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.models import (
    AgentId,
    ResponseStatus,
    StandardAgentResponse,
    UniversalEvidenceObject,
    SourceMetadata,
    FileType,
    ProofCoordinates,
    DocumentLocation,
)
from app.agents.source_formatter_agent.tools import (
    format_evidence_as_citations,
    format_evidence_as_footnotes,
    format_evidence_as_inline_references,
    create_evidence_summary_table,
    validate_evidence_completeness,
)
from app.agents.source_formatter_agent.prompts import SOURCE_FORMATTER_SYSTEM_PROMPT


class SourceFormatterSkillMiddleware(AgentMiddleware):
    """Middleware that provides source formatting skills to the agent."""

    tools = [
        format_evidence_as_citations,
        format_evidence_as_footnotes,
        format_evidence_as_inline_references,
        create_evidence_summary_table,
        validate_evidence_completeness,
    ]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """Inject skill context into the system prompt."""
        skills_addendum = """

## Source Formatting Skills Available

### Citation Skill
Use `format_evidence_as_citations` for:
- Academic-style reference lists
- Numbered citations with source details
- Professional documentation

### Footnote Skill
Use `format_evidence_as_footnotes` for:
- Document footnotes with content excerpts
- Report-style source attribution
- Detailed source documentation

### Inline Reference Skill
Use `format_evidence_as_inline_references` for:
- Adding reference markers to answer text
- Connecting claims to evidence
- Creating annotated responses

### Summary Table Skill
Use `create_evidence_summary_table` for:
- Quick evidence overview
- Markdown table of all sources
- Comparative source display

### Validation Skill
Use `validate_evidence_completeness` for:
- Checking evidence quality
- Identifying missing fields
- Quality assurance
"""
        # Get current system message content as string
        if request.system_message is not None:
            current_content = request.system_message.content
            if isinstance(current_content, str):
                new_content = current_content + skills_addendum
            else:
                new_content = str(current_content) + skills_addendum
        else:
            new_content = skills_addendum

        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


class SourceFormatterAgent:
    """Agent that formats evidence and creates proper citations."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """Initialize the Source Formatter Agent.

        Args:
            model_name: The Gemini model to use.
        """
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.gemini_api_key,
        )
        self.agent = create_agent(
            model=self.model,
            system_prompt=SOURCE_FORMATTER_SYSTEM_PROMPT,
            middleware=[SourceFormatterSkillMiddleware()],
            response_format=ToolStrategy(StandardAgentResponse),
        )

    def invoke(
        self,
        query: str,
        evidence: Optional[List[UniversalEvidenceObject]] = None,
        answer_text: Optional[str] = None,
        format_type: str = "citations",
    ) -> StandardAgentResponse:
        """Execute the source formatter agent.

        Args:
            query: The formatting request.
            evidence: List of evidence objects to format.
            answer_text: Optional answer text to add references to.
            format_type: Type of formatting (citations, footnotes, inline, table).

        Returns:
            StandardAgentResponse with formatted output.
        """
        try:
            # Prepare evidence JSON for tools
            evidence_json = "[]"
            if evidence:
                evidence_dicts = [e.model_dump() for e in evidence]
                evidence_json = json.dumps(evidence_dicts)

            # Build the request message
            request_parts = [query]
            request_parts.append(f"\nEvidence to format (JSON): {evidence_json}")
            if answer_text:
                request_parts.append(f"\nAnswer text: {answer_text}")
            request_parts.append(f"\nRequested format: {format_type}")

            full_query = "\n".join(request_parts)

            result = self.agent.invoke({"messages": [HumanMessage(content=full_query)]})

            # Use structured response if available
            if "structured_response" in result and result["structured_response"]:
                return result["structured_response"]

            # Fallback: Extract from messages if structured output failed
            final_message = result["messages"][-1]
            content = (
                final_message.content
                if hasattr(final_message, "content")
                else str(final_message)
            )

            # The formatted output is the main evidence
            formatted_evidence = self._create_format_evidence(content, format_type)

            return StandardAgentResponse(
                agent_id=AgentId.SOURCE_FORMATTER_AGENT,
                status=ResponseStatus.SUCCESS,
                thought_process=f"Applied {format_type} formatting to evidence",
                final_answer=content,
                supporting_evidence=[formatted_evidence] if evidence else [],
                confidence_score=0.9,  # High confidence for formatting tasks
            )

        except Exception as e:
            return StandardAgentResponse(
                agent_id=AgentId.SOURCE_FORMATTER_AGENT,
                status=ResponseStatus.FAILED,
                thought_process=f"Error occurred: {str(e)}",
                final_answer=f"Failed to format sources: {str(e)}",
                supporting_evidence=[],
                confidence_score=0.0,
            )

    def format_response(
        self,
        response: StandardAgentResponse,
        format_type: str = "citations",
    ) -> StandardAgentResponse:
        """Format the evidence in an existing StandardAgentResponse.

        Args:
            response: The response containing evidence to format.
            format_type: Type of formatting to apply.

        Returns:
            New StandardAgentResponse with formatted evidence.
        """
        return self.invoke(
            query=f"Format the following evidence as {format_type}",
            evidence=response.supporting_evidence,
            answer_text=response.final_answer,
            format_type=format_type,
        )

    def _create_format_evidence(
        self, formatted_content: str, format_type: str
    ) -> UniversalEvidenceObject:
        """Create an evidence object for the formatted output."""
        from uuid import uuid4

        return UniversalEvidenceObject(
            evidence_id=str(uuid4()),
            source_metadata=SourceMetadata(
                file_id="formatted_output",
                file_name=f"{format_type}_format",
                file_type=FileType.DOCUMENT,
            ),
            extracted_content=formatted_content,
            proof_coordinates=ProofCoordinates(
                document_location=DocumentLocation(
                    original_text_snippet=formatted_content[:200]
                )
            ),
        )
