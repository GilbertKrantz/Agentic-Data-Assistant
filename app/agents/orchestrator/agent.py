"""
Orchestrator Agent - Main agent that coordinates all other agents.
"""

import json
import sys
from typing import Any, Callable, Dict, List, Optional
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.config import RunnableConfig

from app.config import settings
import langsmith as ls
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
from app.agents.orchestrator.tools import (
    call_retriever_agent,
    call_data_scientist_agent,
    call_parallel_retrieval_and_analysis,
    call_validator_agent,
    reset_evidence_pool,
    get_evidence_pool,
)
from app.agents.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT


# Global progress callback for real-time feedback
_progress_callback: Optional[Callable[[str], None]] = None


def set_progress_callback(callback: Optional[Callable[[str], None]]):
    """Set a callback function for progress updates.

    Args:
        callback: A function that takes a progress message string.
    """
    global _progress_callback
    _progress_callback = callback


def _report_progress(message: str, detail: str = ""):
    """Report progress via callback and print to stdout."""
    full_message = f"🧠 [Orchestrator] {message}"
    if detail:
        full_message += f" | {detail}"

    # Print to stdout
    print(f"\n{full_message}")
    if detail and not full_message.endswith(detail):
        print(f"   └─ {detail}")
    sys.stdout.flush()

    # Call progress callback if set
    if _progress_callback:
        _progress_callback(full_message)


class OrchestratorSkillMiddleware(AgentMiddleware):
    """Middleware that provides orchestration skills to the agent."""

    tools = [
        call_retriever_agent,
        call_data_scientist_agent,
        call_parallel_retrieval_and_analysis,
        call_validator_agent,
    ]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """Inject skill context into the system prompt."""
        skills_addendum = """

## Agent Coordination Skills

### ⚡ Parallel Execution Skill (RECOMMENDED for speed)
Use `call_parallel_retrieval_and_analysis` when you need to:
- Get both document/database information AND perform analysis
- Speed up complex queries that need both retriever and data scientist
- Run independent retrieval and analysis tasks simultaneously
This is MUCH FASTER than calling agents sequentially.

### Retrieval Skill
Use `call_retriever_agent` when you need to:
- Find information from documents or databases
- Get data before analysis
- Look up specific facts

### Analysis Skill  
Use `call_data_scientist_agent` when you need to:
- Perform statistical analysis
- Get insights from data
- Run computations or ML models
- Query SQL database directly (has direct SQL access)

### Validation Skill
Use `call_validator_agent` when you need to:
- Verify accuracy of responses
- Fact-check important claims
- Validate numerical data

## Workflow Patterns

### ⚡ FAST: Parallel Query (PREFERRED for complex questions)
User asks complex question → Call parallel_retrieval_and_analysis → Synthesize response

### Simple Query
User asks a question → Call appropriate single agent → Return response

### Analysis Query (when parallel not suitable)
1. Call Retriever to get data context
2. Call Data Scientist with the context
3. (Optional) Call Validator for important results
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


class OrchestratorAgent:
    """Main agent that coordinates all specialized agents."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """Initialize the Orchestrator Agent.

        Args:
            model_name: The Gemini model to use.
        """
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.gemini_api_key,
        )
        self.agent = create_agent(
            model=self.model,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            middleware=[OrchestratorSkillMiddleware()],
            response_format=ToolStrategy(StandardAgentResponse),
        )

    def invoke(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> StandardAgentResponse:
        """Execute the orchestrator agent on a user query.

        This is the main entry point for user queries. The orchestrator
        will understand the query and route it to appropriate specialized
        agents.

        Args:
            query: The user's query or request.
            context: Optional additional context.
            progress_callback: Optional callback function to receive progress updates.
                              Function signature: (message: str) -> None

        Returns:
            StandardAgentResponse with the final answer.
        """
        try:
            # Set progress callback for this invocation
            set_progress_callback(progress_callback)

            # Reset evidence pool for new query
            reset_evidence_pool()

            _report_progress(
                "Analyzing query...",
                f"Query: {query[:80]}{'...' if len(query) > 80 else ''}",
            )

            # Add context to query if provided
            full_query = query
            if context:
                context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
                full_query = f"{query}\n\nAdditional Context:\n{context_str}"
                _report_progress("Adding additional context...")

            _report_progress("Determining which agents to use...")

            _report_progress("Calling specialized agents...")

            invoke_config = RunnableConfig(
                run_name="OrchestratorAgent", tags=["orchestrator", "multi-agent"]
            )

            client = ls.Client(
                api_key=settings.langsmith_api_key,
                api_url=settings.langsmith_endpoint,
            )

            with ls.tracing_context(
                client=client,
                project_name=settings.langsmith_project_name,
                enabled=settings.langsmith_tracing,
            ):
                result = self.agent.invoke(
                    {"messages": [HumanMessage(content=full_query)]},
                    config=invoke_config,
                )

            _report_progress("Synthesizing final response...")

            # Get all accumulated evidence from the session
            accumulated_evidence = get_evidence_pool()
            evidence_objects = [
                UniversalEvidenceObject.model_validate(e) for e in accumulated_evidence
            ]

            _report_progress(
                f"✓ Response ready",
                f"Status: success, Total evidence: {len(evidence_objects)}",
            )

            # Use structured response if available, but add accumulated evidence
            if "structured_response" in result and result["structured_response"]:
                structured_response: StandardAgentResponse = result[
                    "structured_response"
                ]
                # Merge accumulated evidence with any evidence in the structured response
                all_evidence = self._merge_evidence(
                    evidence_objects, structured_response.supporting_evidence
                )
                return StandardAgentResponse(
                    agent_id=structured_response.agent_id,
                    status=structured_response.status,
                    thought_process=structured_response.thought_process,
                    final_answer=structured_response.final_answer,
                    supporting_evidence=all_evidence,
                    confidence_score=structured_response.confidence_score,
                )

            # Fallback: Extract from messages if structured output failed
            final_message = result["messages"][-1]
            content = (
                final_message.content
                if hasattr(final_message, "content")
                else str(final_message)
            )

            return StandardAgentResponse(
                agent_id=AgentId.ORCHESTRATOR,
                status=ResponseStatus.SUCCESS,
                thought_process=self._extract_thought_process(result["messages"]),
                final_answer=content,
                supporting_evidence=evidence_objects,
                confidence_score=self._calculate_confidence(result["messages"]),
            )

        except Exception as e:
            _report_progress("✗ Error occurred", str(e))
            # Clear progress callback
            set_progress_callback(None)
            return StandardAgentResponse(
                agent_id=AgentId.ORCHESTRATOR,
                status=ResponseStatus.FAILED,
                thought_process=f"Error occurred: {str(e)}",
                final_answer=f"I encountered an error while processing your request: {str(e)}",
                supporting_evidence=[],
                confidence_score=0.0,
            )

    def _merge_evidence(
        self,
        accumulated: List[UniversalEvidenceObject],
        new_evidence: List[UniversalEvidenceObject],
    ) -> List[UniversalEvidenceObject]:
        """Merge evidence lists, removing duplicates by evidence_id."""
        seen_ids = set()
        merged = []

        for evidence in accumulated + new_evidence:
            if evidence.evidence_id not in seen_ids:
                merged.append(evidence)
                seen_ids.add(evidence.evidence_id)

        return merged

    def chat(
        self, query: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Simple chat interface that returns just the answer.

        Args:
            query: The user's question or request.
            progress_callback: Optional callback function to receive progress updates.

        Returns:
            The final answer as a string.
        """
        response = self.invoke(query, progress_callback=progress_callback)
        # Clear progress callback
        set_progress_callback(None)
        return response.final_answer

    def _extract_evidence_from_messages(
        self, messages: List[Any]
    ) -> List[UniversalEvidenceObject]:
        """Extract evidence from agent tool call results."""
        from uuid import uuid4

        evidence_list = []

        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                try:
                    # Parse tool response JSON
                    tool_result = json.loads(msg.content)
                    agent_id = tool_result.get("agent_id", "unknown")

                    evidence = UniversalEvidenceObject(
                        evidence_id=str(uuid4()),
                        source_metadata=SourceMetadata(
                            file_id="agent_response",
                            file_name=agent_id,
                            file_type=FileType.DOCUMENT,
                        ),
                        extracted_content=tool_result.get(
                            "final_answer", str(msg.content)
                        )[:500],
                        proof_coordinates=ProofCoordinates(
                            document_location=DocumentLocation(
                                original_text_snippet=tool_result.get(
                                    "final_answer", ""
                                )[:200]
                            )
                        ),
                    )
                    evidence_list.append(evidence)
                except json.JSONDecodeError:
                    # Non-JSON response
                    evidence = UniversalEvidenceObject(
                        evidence_id=str(uuid4()),
                        source_metadata=SourceMetadata(
                            file_id="tool_response",
                            file_name=getattr(msg, "name", "unknown_tool"),
                            file_type=FileType.DOCUMENT,
                        ),
                        extracted_content=str(msg.content),
                        proof_coordinates=ProofCoordinates(
                            document_location=DocumentLocation(
                                original_text_snippet=str(msg.content)[:200]
                            )
                        ),
                    )
                    evidence_list.append(evidence)

        return evidence_list

    def _extract_thought_process(self, messages: List[Any]) -> str:
        """Extract the orchestrator's reasoning from messages."""
        thoughts = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    # Map tool names to readable descriptions
                    tool_descriptions = {
                        "call_retriever_agent": "Retrieval",
                        "call_data_scientist_agent": "Analysis",
                        "call_validator_agent": "Validation",
                    }
                    desc = tool_descriptions.get(tool_name, tool_name)
                    thoughts.append(f"Called {desc}")
        return " → ".join(thoughts) if thoughts else "Direct response"

    def _calculate_confidence(self, messages: List[Any]) -> float:
        """Calculate overall confidence based on agent responses."""
        confidences = []

        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                try:
                    tool_result = json.loads(msg.content)
                    conf = tool_result.get("confidence_score")
                    if conf is not None:
                        confidences.append(float(conf))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        if not confidences:
            return 0.7  # Default confidence

        # Return average confidence, weighted toward lower scores for safety
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)

        # Blend average with minimum to be conservative
        return (avg_conf * 0.7) + (min_conf * 0.3)
