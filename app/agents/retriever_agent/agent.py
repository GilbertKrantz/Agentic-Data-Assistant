"""
Retriever Agent - Handles retrieval from both ChromaDB and DuckDB.
"""

from typing import Any, Dict, List, Optional
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.models import (
    AgentId,
    ResponseStatus,
    StandardAgentResponse,
    UniversalEvidenceObject,
)
from app.agents.retriever_agent.tools import (
    search_documents,
    query_sql_database,
    get_database_schema,
)
from app.agents.retriever_agent.prompts import RETRIEVER_SYSTEM_PROMPT


class RetrieverSkillMiddleware(AgentMiddleware):
    """Middleware that provides retrieval skills to the agent."""

    tools = [search_documents, query_sql_database, get_database_schema]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """Inject skill context into the system prompt."""
        skills_addendum = """

## Retrieval Skills Available

### Document Retrieval Skill
Use `search_documents` for:
- Finding information from PDF documents
- Semantic search across text content
- Retrieving fraud detection techniques and methodologies

### SQL Retrieval Skill  
Use `query_sql_database` for:
- Analyzing transaction data
- Finding fraud patterns in structured data
- Statistical queries on fraud_train/fraud_test tables

### Schema Discovery Skill
Use `get_database_schema` to:
- Understand available tables and columns
- Plan SQL queries effectively
"""
        # Get current system message content as string
        if request.system_message is not None:
            current_content = request.system_message.content
            if isinstance(current_content, str):
                new_content = current_content + skills_addendum
            else:
                # Handle list of content blocks
                new_content = str(current_content) + skills_addendum
        else:
            new_content = skills_addendum

        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


class RetrieverAgent:
    """Agent that retrieves information from ChromaDB and DuckDB."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """Initialize the Retriever Agent.

        Args:
            model_name: The Gemini model to use.
        """
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.gemini_api_key,
        )
        self.agent = create_agent(
            model=self.model,
            system_prompt=RETRIEVER_SYSTEM_PROMPT,
            middleware=[RetrieverSkillMiddleware()],
            response_format=ToolStrategy(StandardAgentResponse),
        )

    def invoke(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> StandardAgentResponse:
        """Execute the retrieval agent on a query.

        Args:
            query: The user's query to process.
            context: Optional additional context.

        Returns:
            StandardAgentResponse with retrieval results.
        """
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )

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

            evidence_list = self._extract_evidence_from_messages(result["messages"])

            return StandardAgentResponse(
                agent_id=AgentId.RETRIEVAL_AGENT,
                status=ResponseStatus.SUCCESS,
                thought_process=self._extract_thought_process(result["messages"]),
                final_answer=content,
                supporting_evidence=evidence_list,
                confidence_score=self._calculate_confidence(evidence_list),
            )

        except Exception as e:
            return StandardAgentResponse(
                agent_id=AgentId.RETRIEVAL_AGENT,
                status=ResponseStatus.FAILED,
                thought_process=f"Error occurred: {str(e)}",
                final_answer=f"Failed to retrieve information: {str(e)}",
                supporting_evidence=[],
                confidence_score=0.0,
            )

    def _extract_evidence_from_messages(
        self, messages: List[Any]
    ) -> List[UniversalEvidenceObject]:
        """Extract evidence objects from tool call results."""
        from app.models import (
            FileType,
            SourceMetadata,
            ProofCoordinates,
            DocumentLocation,
        )
        from uuid import uuid4

        evidence_list = []

        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                # Create evidence from tool response
                evidence = UniversalEvidenceObject(
                    evidence_id=str(uuid4()),
                    source_metadata=SourceMetadata(
                        file_id="retrieval_result",
                        file_name=getattr(msg, "name", "unknown_tool"),
                        file_type=FileType.DOCUMENT,
                    ),
                    extracted_content=str(msg.content)[:500],  # Truncate for evidence
                    proof_coordinates=ProofCoordinates(
                        document_location=DocumentLocation(
                            original_text_snippet=str(msg.content)[:200]
                        )
                    ),
                )
                evidence_list.append(evidence)

        return evidence_list

    def _extract_thought_process(self, messages: List[Any]) -> str:
        """Extract the agent's reasoning from messages."""
        thoughts = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    thoughts.append(f"Called tool: {tc.get('name', 'unknown')}")
        return " -> ".join(thoughts) if thoughts else "Direct response without tools"

    def _calculate_confidence(self, evidence: List[UniversalEvidenceObject]) -> float:
        """Calculate confidence score based on evidence quality."""
        if not evidence:
            return 0.3  # Low confidence if no evidence
        elif len(evidence) == 1:
            return 0.6  # Medium confidence with single source
        elif len(evidence) >= 2:
            return 0.8  # Higher confidence with multiple sources
        return 0.5
