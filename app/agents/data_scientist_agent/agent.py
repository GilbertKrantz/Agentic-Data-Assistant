"""
Data Scientist Agent - Handles code execution for data analysis and insights.
All data queries go through retrieval tools.
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
    SourceMetadata,
    FileType,
    ProofCoordinates,
    StructuredLocation,
)
from app.agents.data_scientist_agent.tools import (
    execute_python_code,
    retrieve_data_for_analysis,
    retrieve_documents_for_analysis,
    get_schema_info,
)
from app.agents.data_scientist_agent.prompts import DATA_SCIENTIST_SYSTEM_PROMPT


class DataScienceSkillMiddleware(AgentMiddleware):
    """Middleware that provides data science skills to the agent."""

    tools = [
        execute_python_code,
        retrieve_data_for_analysis,
        retrieve_documents_for_analysis,
        get_schema_info,
    ]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """Inject skill context into the system prompt."""
        skills_addendum = """

## Data Science Skills Available

### Code Execution Skill
Use `execute_python_code` for:
- Custom statistical analysis on retrieved data
- Machine learning model training
- Complex data transformations
- Visualization generation
- Any Python-based computation

### Data Retrieval Skill
Use `retrieve_data_for_analysis` for:
- Fetching data from DuckDB via SQL queries
- Getting structured data for analysis
- ALWAYS use this before code execution that needs data

### Document Retrieval Skill
Use `retrieve_documents_for_analysis` for:
- Getting context from documents
- Finding fraud detection methodologies
- Research background for analysis

### Schema Info Skill
Use `get_schema_info` to:
- Understand available tables and columns
- Plan SQL queries effectively
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


class DataScientistAgent:
    """Agent that performs data analysis through code execution."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """Initialize the Data Scientist Agent.

        Args:
            model_name: The Gemini model to use.
        """
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.gemini_api_key,
        )
        self.agent = create_agent(
            model=self.model,
            system_prompt=DATA_SCIENTIST_SYSTEM_PROMPT,
            middleware=[DataScienceSkillMiddleware()],
            response_format=ToolStrategy(StandardAgentResponse),
        )

    def invoke(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> StandardAgentResponse:
        """Execute the data scientist agent on a query.

        Args:
            query: The analysis request to process.
            context: Optional additional context or data.

        Returns:
            StandardAgentResponse with analysis results.
        """
        try:
            # Add context to query if provided
            full_query = query
            if context:
                context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
                full_query = f"{query}\n\nAdditional Context:\n{context_str}"

            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": full_query}]}
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
                agent_id=AgentId.DATASCIENCE_AGENT,
                status=ResponseStatus.SUCCESS,
                thought_process=self._extract_thought_process(result["messages"]),
                final_answer=content,
                supporting_evidence=evidence_list,
                confidence_score=self._calculate_confidence(evidence_list, content),
            )

        except Exception as e:
            return StandardAgentResponse(
                agent_id=AgentId.DATASCIENCE_AGENT,
                status=ResponseStatus.FAILED,
                thought_process=f"Error occurred: {str(e)}",
                final_answer=f"Failed to complete analysis: {str(e)}",
                supporting_evidence=[],
                confidence_score=0.0,
            )

    def _extract_evidence_from_messages(
        self, messages: List[Any]
    ) -> List[UniversalEvidenceObject]:
        """Extract evidence objects from code execution results."""
        from uuid import uuid4

        evidence_list = []

        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                tool_name = getattr(msg, "name", "code_execution")
                content = str(msg.content)

                # Create structured evidence for code execution
                evidence = UniversalEvidenceObject(
                    evidence_id=str(uuid4()),
                    source_metadata=SourceMetadata(
                        file_id="analysis_result",
                        file_name=tool_name,
                        file_type=FileType.STRUCTURED_DATA,
                    ),
                    extracted_content=content[:1000],  # Truncate for evidence
                    proof_coordinates=ProofCoordinates(
                        structured_location=StructuredLocation(
                            filter_logic=f"Tool: {tool_name}",
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
                    tool_name = tc.get("name", "unknown")
                    # Include code snippet if it's execute_python_code
                    if tool_name == "execute_python_code":
                        args = tc.get("args", {})
                        code = args.get("code", "")[:100]
                        thoughts.append(f"Executed code: {code}...")
                    else:
                        thoughts.append(f"Called: {tool_name}")
        return " -> ".join(thoughts) if thoughts else "Direct response"

    def _calculate_confidence(
        self, evidence: List[UniversalEvidenceObject], content: str
    ) -> float:
        """Calculate confidence score based on analysis quality."""
        confidence = 0.5  # Base confidence

        # More evidence = higher confidence
        if evidence:
            confidence += min(len(evidence) * 0.1, 0.3)

        # Check for quantitative results in content
        quantitative_indicators = ["%", "average", "mean", "count", "sum", "std"]
        for indicator in quantitative_indicators:
            if indicator.lower() in content.lower():
                confidence += 0.05

        # Cap at 0.95
        return min(confidence, 0.95)
