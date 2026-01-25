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
    retrieve_data_for_analysis,
    retrieve_documents_for_analysis,
    get_schema_info,
)
from app.agents.data_scientist_agent.prompts import DATA_SCIENTIST_SYSTEM_PROMPT


class DataScienceSkillMiddleware(AgentMiddleware):
    """Middleware that provides data science skills to the agent."""

    tools = [
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

### ⚡ Integrated Code Execution (BUILT-IN)
You have INTEGRATED Python code execution capability!
- Just write Python code directly in your response
- The code will be automatically executed by Gemini
- Available libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, tensorflow
- Use this for statistical analysis, ML models, visualizations

### Data Retrieval Skill
Use `retrieve_data_for_analysis` for:
- Fetching data from DuckDB via SQL queries
- Returns JSON format that can be parsed and analyzed
- Supports all SQL operations (SELECT, JOIN, GROUP BY, etc.)

### Document Retrieval Skill
Use `retrieve_documents_for_analysis` for:
- Getting context from documents
- Finding fraud detection methodologies
- Research background for analysis

### Schema Info Skill
Use `get_schema_info` to:
- Understand available tables and columns
- Plan SQL queries effectively

## Recommended Workflow
1. Use `get_schema_info` to understand data structure
2. Use `retrieve_data_for_analysis` to get data via SQL
3. Write Python code directly for complex analysis (integrated execution)
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
    """Agent that performs data analysis through integrated code execution."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """Initialize the Data Scientist Agent.

        Args:
            model_name: The Gemini model to use.
        """
        # Initialize base model for the agent
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.gemini_api_key,
        )

        # Create a separate model instance with code execution bound
        # This will be used for direct code execution calls
        self.model_with_code_execution = self.model.bind_tools([{"code_execution": {}}])

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

        Uses Gemini's built-in code execution for Python analysis.

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

            # Use the model with code execution capability directly
            # This bypasses the agent framework but enables Gemini's native code execution
            from langchain.messages import (
                HumanMessage,
                SystemMessage as LCSystemMessage,
            )

            messages = [
                LCSystemMessage(content=DATA_SCIENTIST_SYSTEM_PROMPT),
                HumanMessage(content=full_query),
            ]

            result = self.model_with_code_execution.invoke(messages)

            # Extract content from the response
            content = result.text if hasattr(result, "text") else str(result.content)

            # Extract code execution evidence from content_blocks if available
            evidence_list = self._extract_evidence_from_response(result)

            return StandardAgentResponse(
                agent_id=AgentId.DATASCIENCE_AGENT,
                status=ResponseStatus.SUCCESS,
                thought_process=self._extract_thought_from_response(result),
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

    def _extract_evidence_from_response(
        self, response: Any
    ) -> List[UniversalEvidenceObject]:
        """Extract evidence from code execution response."""
        from uuid import uuid4

        evidence_list = []

        # Check for content_blocks (Gemini's structured response format)
        content_blocks = getattr(response, "content_blocks", None)
        if content_blocks:
            for block in content_blocks:
                if isinstance(block, dict):
                    block_type = block.get("type", "")

                    # Handle code execution blocks
                    if (
                        block_type == "server_tool_call"
                        and block.get("name") == "code_interpreter"
                    ):
                        code = block.get("args", {}).get("code", "")
                        evidence = UniversalEvidenceObject(
                            evidence_id=str(uuid4()),
                            source_metadata=SourceMetadata(
                                file_id="code_execution",
                                file_name="python_code",
                                file_type=FileType.STRUCTURED_DATA,
                            ),
                            extracted_content=f"Executed code:\n{code[:500]}",
                            proof_coordinates=ProofCoordinates(
                                structured_location=StructuredLocation(
                                    filter_logic="Code execution via Gemini",
                                )
                            ),
                        )
                        evidence_list.append(evidence)

                    # Handle code execution results
                    elif block_type == "server_tool_result":
                        output = block.get("output", "")
                        evidence = UniversalEvidenceObject(
                            evidence_id=str(uuid4()),
                            source_metadata=SourceMetadata(
                                file_id="code_result",
                                file_name="execution_output",
                                file_type=FileType.STRUCTURED_DATA,
                            ),
                            extracted_content=f"Output:\n{output[:500]}",
                            proof_coordinates=ProofCoordinates(
                                structured_location=StructuredLocation(
                                    filter_logic="Code execution result",
                                )
                            ),
                        )
                        evidence_list.append(evidence)

        return evidence_list

    def _extract_thought_from_response(self, response: Any) -> str:
        """Extract thought process from response."""
        thoughts = []

        content_blocks = getattr(response, "content_blocks", None)
        if content_blocks:
            for block in content_blocks:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "thinking":
                        thoughts.append(block.get("thinking", "")[:200])
                    elif block_type == "server_tool_call":
                        name = block.get("name", "unknown")
                        thoughts.append(f"Called: {name}")

        return " -> ".join(thoughts) if thoughts else "Direct analysis"

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
