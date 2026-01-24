"""
Tools for the Orchestrator Agent - calls other specialized agents.
"""

import sys
from typing import Optional
from langchain.tools import tool

from app.models import StandardAgentResponse


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


def _print_result(agent_name: str, status: str, confidence: Optional[float]):
    """Print result feedback to the user."""
    status_icon = "✓" if status == "success" else "✗"
    conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
    print(f"   └─ Result: {status_icon} {status} (confidence: {conf_str})")
    sys.stdout.flush()


@tool
def call_retriever_agent(query: str) -> str:
    """Call the Retriever Agent to find information from documents and databases.

    Use this tool when you need to:
    - Search for information in documents
    - Query the fraud detection database
    - Look up specific facts or data points
    - Get context before performing analysis

    Args:
        query: The search query or question to answer.

    Returns:
        JSON string containing the agent's response with evidence.
    """
    from app.agents.retriever_agent import RetrieverAgent
    import json

    _print_thinking("Retriever", "Searching documents and database...", query)

    agent = RetrieverAgent(model_name="gemini-3-flash-preview")
    response: StandardAgentResponse = agent.invoke(query)

    _print_result("Retriever", response.status.value, response.confidence_score)

    return json.dumps(
        {
            "agent_id": response.agent_id.value,
            "status": response.status.value,
            "final_answer": response.final_answer,
            "evidence_count": len(response.supporting_evidence),
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

    Args:
        query: The analysis request or question.
        context: Optional additional context (e.g., data from retriever).

    Returns:
        JSON string containing the agent's analysis results.
    """
    from app.agents.data_scientist_agent import DataScientistAgent
    import json

    _print_thinking("Data Scientist", "Analyzing data and computing insights...", query)

    agent = DataScientistAgent(model_name="gemini-3-flash-preview")

    context_dict = None
    if context:
        context_dict = {"additional_context": context}
        print(f"   └─ With context: {context[:80]}{'...' if len(context) > 80 else ''}")
        sys.stdout.flush()

    response: StandardAgentResponse = agent.invoke(query, context=context_dict)

    _print_result("Data Scientist", response.status.value, response.confidence_score)

    return json.dumps(
        {
            "agent_id": response.agent_id.value,
            "status": response.status.value,
            "final_answer": response.final_answer,
            "evidence_count": len(response.supporting_evidence),
            "confidence_score": response.confidence_score,
            "thought_process": response.thought_process,
        }
    )


@tool
def call_source_formatter_agent(
    query: str, evidence_json: Optional[str] = None, format_type: str = "citations"
) -> str:
    """Call the Source Formatter Agent to format evidence and citations.

    Use this tool when you need to:
    - Format sources as citations
    - Create footnotes for a response
    - Build an evidence summary table
    - Ensure proper source attribution

    Args:
        query: The formatting request.
        evidence_json: Optional JSON string of evidence to format.
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

    evidence_list = None
    if evidence_json:
        try:
            evidence_dicts = json.loads(evidence_json)
            evidence_list = [
                UniversalEvidenceObject.model_validate(e) for e in evidence_dicts
            ]
            print(f"   └─ Processing {len(evidence_list)} evidence items")
            sys.stdout.flush()
        except (json.JSONDecodeError, Exception):
            evidence_list = None

    response: StandardAgentResponse = agent.invoke(
        query=query,
        evidence=evidence_list,
        format_type=format_type,
    )

    _print_result("Source Formatter", response.status.value, response.confidence_score)

    return json.dumps(
        {
            "agent_id": response.agent_id.value,
            "status": response.status.value,
            "final_answer": response.final_answer,
            "confidence_score": response.confidence_score,
        }
    )


@tool
def call_validator_agent(
    query: str,
    answer_to_validate: Optional[str] = None,
    evidence_json: Optional[str] = None,
    confidence_score: Optional[float] = None,
) -> str:
    """Call the Validator Agent to verify and fact-check responses.

    Use this tool when you need to:
    - Verify the accuracy of an answer
    - Check if claims are supported by evidence
    - Validate numerical data
    - Assess if a confidence score is appropriate

    Args:
        query: The validation request.
        answer_to_validate: The answer text to validate.
        evidence_json: JSON string of supporting evidence.
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

    # Build a mock response to validate if components provided
    response_to_validate = None
    if answer_to_validate:
        evidence_list = []
        if evidence_json:
            try:
                evidence_dicts = json.loads(evidence_json)
                evidence_list = [
                    UniversalEvidenceObject.model_validate(e) for e in evidence_dicts
                ]
            except (json.JSONDecodeError, Exception):
                pass

        response_to_validate = StandardAgentResponse(
            agent_id=AgentId.ORCHESTRATOR,
            status=ResponseStatus.SUCCESS,
            thought_process="Response to be validated",
            final_answer=answer_to_validate,
            supporting_evidence=evidence_list,
            confidence_score=confidence_score or 0.5,
        )
        print(f"   └─ Validating answer with {len(evidence_list)} evidence items")
        sys.stdout.flush()

    response: StandardAgentResponse = agent.invoke(
        query=query,
        response_to_validate=response_to_validate,
    )

    _print_result("Validator", response.status.value, response.confidence_score)

    return json.dumps(
        {
            "agent_id": response.agent_id.value,
            "status": response.status.value,
            "final_answer": response.final_answer,
            "validation_passed": response.status == ResponseStatus.SUCCESS,
            "confidence_score": response.confidence_score,
        }
    )
