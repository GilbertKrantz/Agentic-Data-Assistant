"""
Tools for the Orchestrator Agent - calls other specialized agents.
Implements cumulative evidence flow between agents.
"""

import sys
from typing import Optional, List, Dict, Any
from langchain.tools import tool

from app.models import StandardAgentResponse, UniversalEvidenceObject


# Global evidence accumulator for the current orchestration session
_accumulated_evidence: List[dict] = []


def reset_evidence_pool():
    """Reset the evidence pool for a new query session."""
    global _accumulated_evidence
    _accumulated_evidence = []


def get_evidence_pool() -> List[dict]:
    """Get the current accumulated evidence pool."""
    global _accumulated_evidence
    return _accumulated_evidence.copy()


def add_to_evidence_pool(evidence_list: List[UniversalEvidenceObject]):
    """Add new evidence to the pool, filtering duplicates."""
    global _accumulated_evidence

    existing_ids = {e.get("evidence_id") for e in _accumulated_evidence}

    for evidence in evidence_list:
        evidence_dict = evidence.model_dump()
        if evidence_dict.get("evidence_id") not in existing_ids:
            _accumulated_evidence.append(evidence_dict)
            existing_ids.add(evidence_dict.get("evidence_id"))


def _print_thinking(agent_name: str, action: str, query: str = ""):
    """Print thinking feedback to the user."""
    icons = {
        "Retriever": "🔍",
        "Data Scientist": "📊",
        "Source Formatter": "📝",
        "Validator": "✅",
    }
    icon = icons.get(agent_name, "🤖")

    print(f"\n{icon} [{agent_name} Agent] {action}")
    if query:
        print(f"   └─ Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    sys.stdout.flush()


def _print_result(
    agent_name: str, status: str, confidence: Optional[float], evidence_count: int = 0
):
    """Print result feedback to the user."""
    status_icon = "✓" if status == "success" else "✗"
    conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
    print(
        f"   └─ Result: {status_icon} {status} (confidence: {conf_str}, evidence: {evidence_count})"
    )
    sys.stdout.flush()


@tool
def call_retriever_agent(query: str) -> str:
    """Call the Retriever Agent to find information from documents and databases.

    Use this tool when you need to:
    - Search for information in documents
    - Query the fraud detection database
    - Look up specific facts or data points
    - Get context before performing analysis

    IMPORTANT: This tool returns evidence that will be automatically accumulated.
    The evidence is available for subsequent agent calls.

    Args:
        query: The search query or question to answer.

    Returns:
        JSON string containing the agent's response with full evidence.
    """
    from app.agents.retriever_agent import RetrieverAgent
    import json

    _print_thinking("Retriever", "Searching documents and database...", query)

    agent = RetrieverAgent(model_name="gemini-3-flash-preview")
    response: StandardAgentResponse = agent.invoke(query)

    # Add evidence to the pool
    add_to_evidence_pool(response.supporting_evidence)

    _print_result(
        "Retriever",
        response.status.value,
        response.confidence_score,
        len(response.supporting_evidence),
    )

    # Return full evidence for transparency
    evidence_dicts = [e.model_dump() for e in response.supporting_evidence]

    return json.dumps(
        {
            "agent_id": response.agent_id.value,
            "status": response.status.value,
            "final_answer": response.final_answer,
            "evidence": evidence_dicts,
            "evidence_count": len(response.supporting_evidence),
            "total_accumulated_evidence": len(get_evidence_pool()),
            "confidence_score": response.confidence_score,
            "thought_process": response.thought_process,
        }
    )


@tool
def call_data_scientist_agent(query: str, context: Optional[str] = None) -> str:
    """Call the Data Scientist Agent to perform analysis and get insights.

    Use this tool when you need to:
    - Analyze fraud patterns or statistics
    - Perform calculations on data
    - Train or evaluate ML models
    - Get data-driven insights

    IMPORTANT: This agent has access to accumulated evidence from previous calls.
    New evidence from analysis will be added to the pool.

    Args:
        query: The analysis request or question.
        context: Optional additional context (e.g., data from retriever).

    Returns:
        JSON string containing the agent's analysis results with evidence.
    """
    from app.agents.data_scientist_agent import DataScientistAgent
    import json

    _print_thinking("Data Scientist", "Analyzing data and computing insights...", query)

    agent = DataScientistAgent(model_name="gemini-3-flash-preview")

    # Pass accumulated evidence as context
    context_dict: Dict[str, Any] = {"accumulated_evidence": get_evidence_pool()}
    if context:
        context_dict["additional_context"] = context
        print(f"   └─ With context: {context[:80]}{'...' if len(context) > 80 else ''}")
        sys.stdout.flush()

    print(f"   └─ With {len(get_evidence_pool())} accumulated evidence items")
    sys.stdout.flush()

    response: StandardAgentResponse = agent.invoke(query, context=context_dict)

    # Add new evidence to the pool
    add_to_evidence_pool(response.supporting_evidence)

    _print_result(
        "Data Scientist",
        response.status.value,
        response.confidence_score,
        len(response.supporting_evidence),
    )

    evidence_dicts = [e.model_dump() for e in response.supporting_evidence]

    return json.dumps(
        {
            "agent_id": response.agent_id.value,
            "status": response.status.value,
            "final_answer": response.final_answer,
            "evidence": evidence_dicts,
            "evidence_count": len(response.supporting_evidence),
            "total_accumulated_evidence": len(get_evidence_pool()),
            "confidence_score": response.confidence_score,
            "thought_process": response.thought_process,
        }
    )


@tool
def call_source_formatter_agent(query: str, format_type: str = "citations") -> str:
    """Call the Source Formatter Agent to format evidence and citations.

    Use this tool when you need to:
    - Format sources as citations
    - Create footnotes for a response
    - Build an evidence summary table
    - Ensure proper source attribution

    IMPORTANT: This tool automatically uses ALL accumulated evidence from previous
    agent calls. You don't need to pass evidence manually.

    Args:
        query: The formatting request.
        format_type: Type of formatting (citations, footnotes, inline, table).

    Returns:
        JSON string containing the formatted output.
    """
    from app.agents.source_formatter_agent import SourceFormatterAgent
    from app.models import UniversalEvidenceObject
    import json

    _print_thinking(
        "Source Formatter", f"Formatting evidence as {format_type}...", query
    )

    agent = SourceFormatterAgent(model_name="gemini-3-flash-preview")

    # Use accumulated evidence automatically
    accumulated = get_evidence_pool()
    evidence_list = []
    if accumulated:
        try:
            evidence_list = [
                UniversalEvidenceObject.model_validate(e) for e in accumulated
            ]
            print(f"   └─ Processing {len(evidence_list)} accumulated evidence items")
            sys.stdout.flush()
        except Exception as e:
            print(f"   └─ Warning: Could not parse accumulated evidence: {e}")
            sys.stdout.flush()

    response: StandardAgentResponse = agent.invoke(
        query=query,
        evidence=evidence_list if evidence_list else None,
        format_type=format_type,
    )

    _print_result(
        "Source Formatter",
        response.status.value,
        response.confidence_score,
        len(evidence_list),
    )

    return json.dumps(
        {
            "agent_id": response.agent_id.value,
            "status": response.status.value,
            "final_answer": response.final_answer,
            "evidence_count": len(evidence_list),
            "confidence_score": response.confidence_score,
        }
    )


@tool
def call_validator_agent(
    query: str,
    answer_to_validate: str,
    confidence_score: Optional[float] = None,
) -> str:
    """Call the Validator Agent to verify and fact-check responses.

    Use this tool when you need to:
    - Verify the accuracy of an answer
    - Check if claims are supported by evidence
    - Validate numerical data
    - Assess if a confidence score is appropriate

    CRITICAL: This tool REQUIRES accumulated evidence to validate against.
    If no evidence has been accumulated from previous agent calls, the
    validation will fail. Always call Retriever or Data Scientist first.

    IMPORTANT: This tool automatically uses ALL accumulated evidence from previous
    agent calls. You don't need to pass evidence manually.

    Args:
        query: The validation request describing what to check.
        answer_to_validate: The answer text to validate (REQUIRED).
        confidence_score: The claimed confidence score to assess.

    Returns:
        JSON string containing the validation results.
    """
    from app.agents.validator_agent import ValidatorAgent
    from app.models import (
        StandardAgentResponse,
        AgentId,
        ResponseStatus,
        UniversalEvidenceObject,
    )
    import json

    _print_thinking("Validator", "Validating response and checking facts...", query)

    agent = ValidatorAgent(model_name="gemini-3-flash-preview")

    # Use accumulated evidence automatically
    accumulated = get_evidence_pool()
    evidence_list = []

    if accumulated:
        try:
            evidence_list = [
                UniversalEvidenceObject.model_validate(e) for e in accumulated
            ]
        except Exception as e:
            print(f"   └─ Warning: Could not parse accumulated evidence: {e}")
            sys.stdout.flush()

    # Warn if no evidence available
    if not evidence_list:
        print(f"   └─ ⚠️ WARNING: No accumulated evidence! Validation may fail.")
        print(f"   └─ Tip: Call Retriever or Data Scientist first to gather evidence.")
        sys.stdout.flush()
    else:
        print(f"   └─ Using {len(evidence_list)} accumulated evidence items")
        sys.stdout.flush()

    # Build response to validate
    response_to_validate = StandardAgentResponse(
        agent_id=AgentId.ORCHESTRATOR,
        status=ResponseStatus.SUCCESS,
        thought_process="Response to be validated",
        final_answer=answer_to_validate,
        supporting_evidence=evidence_list,
        confidence_score=confidence_score or 0.5,
    )

    response: StandardAgentResponse = agent.invoke(
        query=query,
        response_to_validate=response_to_validate,
    )

    _print_result(
        "Validator",
        response.status.value,
        response.confidence_score,
        len(evidence_list),
    )

    return json.dumps(
        {
            "agent_id": response.agent_id.value,
            "status": response.status.value,
            "final_answer": response.final_answer,
            "validation_passed": response.status == ResponseStatus.SUCCESS,
            "evidence_used": len(evidence_list),
            "confidence_score": response.confidence_score,
        }
    )
