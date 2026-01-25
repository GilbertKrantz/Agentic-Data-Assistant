"""
Data Scientist Agent - Handles data analysis through SQL queries.
Simplified version without code execution (which was not working correctly).
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4
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
    StructuredLocation,
)
from app.agents.data_scientist_agent.tools import (
    analyze_data_patterns,
    get_fraud_schema,
)
from app.agents.data_scientist_agent.prompts import DATA_SCIENTIST_SYSTEM_PROMPT


class DataScienceSkillMiddleware(AgentMiddleware):
    """Middleware that provides data science skills to the agent."""

    tools = [
        analyze_data_patterns,
        get_fraud_schema,
    ]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """Inject skill context into the system prompt."""
        skills_addendum = """

## Data Science Analysis Tools

### 1. analyze_data_patterns(analysis_description, sql_query)
Use this to:
- Execute SQL queries to fetch data for analysis
- Analyze patterns, trends, and anomalies
- Generate statistics and insights
- Supports all SQL operations (SELECT, JOIN, GROUP BY, aggregations)

### 2. get_fraud_schema()
Use this to:
- Understand available tables and columns before analysis
- Plan what metrics and patterns to calculate

## Workflow
1. Use `get_fraud_schema()` to understand data structure
2. Use `analyze_data_patterns()` with SQL to gather and analyze data
3. Use `interpret_query_results()` for deeper pattern interpretation
4. Leverage Gemini code execution for statistical and ML analysis
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
    """Agent that performs data analysis through SQL queries and reasoning."""

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

        Uses SQL queries for data analysis and LLM reasoning for insights.

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

            # Use the agent with tools for analysis
            result = self.agent.invoke({"messages": [HumanMessage(content=full_query)]})

            # Handle structured response
            if "structured_response" in result and result["structured_response"]:
                return result["structured_response"]

            # Fallback: Extract from messages
            final_message = result["messages"][-1]
            content = (
                final_message.content
                if hasattr(final_message, "content")
                else str(final_message)
            )

            return StandardAgentResponse(
                agent_id=AgentId.DATASCIENCE_AGENT,
                status=ResponseStatus.SUCCESS,
                thought_process=self._extract_thought_process(result["messages"]),
                final_answer=content,
                supporting_evidence=self._extract_evidence_from_messages(
                    result["messages"]
                ),
                confidence_score=0.7,
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

    def _extract_thought_process(self, messages: List[Any]) -> str:
        """Extract thought process from agent messages."""
        thoughts = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    thoughts.append(f"Called: {tool_name}")
        return " -> ".join(thoughts) if thoughts else "Direct analysis"

    def _extract_evidence_from_messages(
        self, messages: List[Any]
    ) -> List[UniversalEvidenceObject]:
        """Extract evidence from tool call results."""
        import json

        evidence_list = []

        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                try:
                    tool_result = json.loads(msg.content)

                    # Extract data from SQL results
                    if "data" in tool_result:
                        data = tool_result.get("data", [])
                        if data:
                            evidence = UniversalEvidenceObject(
                                evidence_id=str(uuid4()),
                                source_metadata=SourceMetadata(
                                    file_id="sql_query",
                                    file_name="fraud_database",
                                    file_type=FileType.STRUCTURED_DATA,
                                ),
                                extracted_content=str(data),  # All rows
                                proof_coordinates=ProofCoordinates(
                                    structured_location=StructuredLocation(
                                        filter_logic=tool_result.get(
                                            "description", "SQL query result"
                                        ),
                                    )
                                ),
                            )
                            evidence_list.append(evidence)

                    # Extract documents
                    elif "documents" in tool_result:
                        for doc in tool_result.get("documents", [])[:5]:
                            evidence = UniversalEvidenceObject(
                                evidence_id=str(doc.get("id", uuid4())),
                                source_metadata=SourceMetadata(
                                    file_id=str(doc.get("id", "unknown")),
                                    file_name=doc.get("source", "document"),
                                    file_type=FileType.DOCUMENT,
                                ),
                                extracted_content=doc.get("content", ""),
                                proof_coordinates=ProofCoordinates(),
                            )
                            evidence_list.append(evidence)

                except json.JSONDecodeError:
                    pass

        return evidence_list
