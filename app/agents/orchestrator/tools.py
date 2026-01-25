"""
Tools for the Orchestrator Agent - calls other specialized agents.
Implements cumulative evidence flow between agents.
Supports parallel execution of retriever and data_scientist agents.

Optimizations:
- Agent instance pool (singleton pattern) to avoid redundant initialization
- Dict-based evidence pool for O(1) lookups
- Auto-detection of parallelizable queries
"""

import concurrent.futures
import re
import sys
from typing import Optional, List, Dict, Any, Tuple
from langchain.tools import tool

from app.models import StandardAgentResponse, UniversalEvidenceObject


_agent_pool: Dict[str, Any] = {}


def _get_retriever_agent():
    """Get or create RetrieverAgent singleton."""
    global _agent_pool
    if "retriever" not in _agent_pool:
        from app.agents.retriever_agent import RetrieverAgent

        _agent_pool["retriever"] = RetrieverAgent(model_name="gemini-3-flash-preview")
    return _agent_pool["retriever"]


def _get_data_scientist_agent():
    """Get or create DataScientistAgent singleton."""
    global _agent_pool
    if "data_scientist" not in _agent_pool:
        from app.agents.data_scientist_agent import DataScientistAgent

        _agent_pool["data_scientist"] = DataScientistAgent(
            model_name="gemini-3-flash-preview"
        )
    return _agent_pool["data_scientist"]


def _get_source_formatter_agent():
    """Get or create SourceFormatterAgent singleton."""
    global _agent_pool
    if "source_formatter" not in _agent_pool:
        from app.agents.source_formatter_agent import SourceFormatterAgent

        _agent_pool["source_formatter"] = SourceFormatterAgent(
            model_name="gemini-3-flash-preview"
        )
    return _agent_pool["source_formatter"]


def _get_validator_agent():
    """Get or create ValidatorAgent singleton."""
    global _agent_pool
    if "validator" not in _agent_pool:
        from app.agents.validator_agent import ValidatorAgent

        _agent_pool["validator"] = ValidatorAgent(model_name="gemini-3-flash-preview")
    return _agent_pool["validator"]


def reset_agent_pool():
    """Reset the agent pool (useful for testing or reconfiguration)."""
    global _agent_pool
    _agent_pool = {}


_accumulated_evidence: Dict[str, dict] = {}


def reset_evidence_pool():
    """Reset the evidence pool for a new query session."""
    global _accumulated_evidence
    _accumulated_evidence = {}


def get_evidence_pool() -> List[dict]:
    """Get the current accumulated evidence pool as a list."""
    global _accumulated_evidence
    return list(_accumulated_evidence.values())


def add_to_evidence_pool(evidence_list: List[UniversalEvidenceObject]):
    """Add new evidence to the pool, filtering duplicates (O(1) per item)."""
    from uuid import uuid4

    global _accumulated_evidence
    for evidence in evidence_list:
        evidence_dict = evidence.model_dump()
        evidence_id = evidence_dict.get("evidence_id")
        # Generate UUID if evidence_id is missing or empty
        if not evidence_id:
            evidence_id = str(uuid4())
            evidence_dict["evidence_id"] = evidence_id
        if evidence_id not in _accumulated_evidence:
            _accumulated_evidence[evidence_id] = evidence_dict


def _run_agent_task(
    agent_type: str, query: str, context: Optional[str] = None
) -> Tuple[str, StandardAgentResponse]:
    """Run a single agent task. Used for parallel execution.

    Uses agent pool (singleton pattern) for efficiency.
    """
    if agent_type == "retriever":
        agent = _get_retriever_agent()
        response = agent.invoke(query)
    elif agent_type == "data_scientist":
        agent = _get_data_scientist_agent()
        context_dict: Dict[str, Any] = {"accumulated_evidence": get_evidence_pool()}
        if context:
            context_dict["additional_context"] = context
        response = agent.invoke(query, context=context_dict)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agent_type, response


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
    import json

    _print_thinking("Retriever", "Searching documents and database...", query)

    agent = _get_retriever_agent()  # Use agent pool
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
    - Get data-driven insights via SQL queries

    IMPORTANT: This agent has access to accumulated evidence from previous calls.
    New evidence from analysis will be added to the pool.

    Args:
        query: The analysis request or question.
        context: Optional additional context (e.g., data from retriever).

    Returns:
        JSON string containing the agent's analysis results with evidence.
    """
    import json

    _print_thinking("Data Scientist", "Analyzing data and computing insights...", query)

    agent = _get_data_scientist_agent()  # Use agent pool

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
def call_parallel_retrieval_and_analysis(
    retriever_query: str,
    analysis_query: str,
    analysis_context: Optional[str] = None,
) -> str:
    """Execute Retriever and Data Scientist agents IN PARALLEL for faster results.

    Use this tool when you need BOTH retrieval AND analysis, and they can run
    independently. This is significantly faster than calling them sequentially.

    WHEN TO USE:
    - User asks for data insights that require both searching and analysis
    - Complex queries that need both document context and SQL analysis
    - When retriever and data_scientist tasks are independent

    WHEN NOT TO USE:
    - When analysis depends on retriever results (use sequential calls)
    - Simple queries that only need one agent

    Args:
        retriever_query: The search/retrieval query for finding information.
        analysis_query: The analysis query for the data scientist.
        analysis_context: Optional additional context for analysis.

    Returns:
        JSON string containing combined results from both agents.
    """
    import json

    print(
        f"\n⚡ [Parallel Execution] Running Retriever and Data Scientist in parallel..."
    )
    print(
        f"   ├─ Retriever: {retriever_query[:60]}{'...' if len(retriever_query) > 60 else ''}"
    )
    print(
        f"   └─ Data Scientist: {analysis_query[:60]}{'...' if len(analysis_query) > 60 else ''}"
    )
    sys.stdout.flush()

    results: Dict[str, StandardAgentResponse] = {}
    errors: Dict[str, str] = {}

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_agent_task, "retriever", retriever_query): "retriever",
            executor.submit(
                _run_agent_task, "data_scientist", analysis_query, analysis_context
            ): "data_scientist",
        }

        for future in concurrent.futures.as_completed(futures):
            agent_type = futures[future]
            try:
                returned_type, response = future.result()
                results[returned_type] = response
                # Add evidence to the pool
                add_to_evidence_pool(response.supporting_evidence)
                _print_result(
                    "Retriever" if returned_type == "retriever" else "Data Scientist",
                    response.status.value,
                    response.confidence_score,
                    len(response.supporting_evidence),
                )
            except Exception as e:
                errors[agent_type] = str(e)
                print(f"   └─ ✗ {agent_type} failed: {e}")
                sys.stdout.flush()

    print(
        f"\n⚡ [Parallel Execution] Complete. Total evidence: {len(get_evidence_pool())}"
    )
    sys.stdout.flush()

    # Build combined response
    combined_response = {
        "parallel_execution": True,
        "agents_called": list(results.keys()),
        "total_accumulated_evidence": len(get_evidence_pool()),
    }

    if "retriever" in results:
        r = results["retriever"]
        combined_response["retriever"] = {
            "status": r.status.value,
            "final_answer": r.final_answer,
            "evidence_count": len(r.supporting_evidence),
            "confidence_score": r.confidence_score,
        }

    if "data_scientist" in results:
        r = results["data_scientist"]
        combined_response["data_scientist"] = {
            "status": r.status.value,
            "final_answer": r.final_answer,
            "evidence_count": len(r.supporting_evidence),
            "confidence_score": r.confidence_score,
        }

    if errors:
        combined_response["errors"] = errors

    return json.dumps(combined_response)


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
    from app.models import UniversalEvidenceObject
    import json

    _print_thinking(
        "Source Formatter", f"Formatting evidence as {format_type}...", query
    )

    agent = _get_source_formatter_agent()  # Use agent pool

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
    from app.models import (
        StandardAgentResponse,
        AgentId,
        ResponseStatus,
        UniversalEvidenceObject,
    )
    import json

    _print_thinking("Validator", "Validating response and checking facts...", query)

    agent = _get_validator_agent()  # Use agent pool

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
