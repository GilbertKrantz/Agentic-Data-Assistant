"""
Tools for the Source Formatter Agent - handles formatting of evidence and sources.
"""

from langchain.tools import tool


@tool
def format_evidence_as_citations(evidence_json: str) -> str:
    """Format evidence objects into proper academic-style citations.

    Args:
        evidence_json: JSON string containing list of evidence objects.

    Returns:
        Formatted citations as a string.
    """
    import json

    try:
        evidence_list = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input"

    citations = []
    for i, evidence in enumerate(evidence_list, 1):
        source = evidence.get("source_metadata", {})
        file_name = source.get("file_name", "Unknown Source")
        file_type = source.get("file_type", "unknown")

        coords = evidence.get("proof_coordinates", {})

        if coords.get("document_location"):
            loc = coords["document_location"]
            page = loc.get("page_number", "")
            section = loc.get("section", "")
            citation = f"[{i}] {file_name}"
            if page:
                citation += f", p. {page}"
            if section:
                citation += f", Section: {section}"
        elif coords.get("structured_location"):
            loc = coords["structured_location"]
            table = source.get("file_name", "table")
            columns = loc.get("column_names", [])
            citation = f"[{i}] {file_name} (Data Source)"
            if columns:
                citation += f", Columns: {', '.join(columns[:3])}"
        else:
            citation = f"[{i}] {file_name}"

        citations.append(citation)

    return "\n".join(citations) if citations else "No citations to format"


@tool
def format_evidence_as_footnotes(evidence_json: str) -> str:
    """Format evidence objects as footnotes for document integration.

    Args:
        evidence_json: JSON string containing list of evidence objects.

    Returns:
        Formatted footnotes as a string.
    """
    import json

    try:
        evidence_list = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input"

    footnotes = []
    for i, evidence in enumerate(evidence_list, 1):
        source = evidence.get("source_metadata", {})
        content = evidence.get("extracted_content", "")[:150]
        file_name = source.get("file_name", "Unknown")

        footnote = f'^{i} Source: {file_name}. "{content}..."'
        footnotes.append(footnote)

    return "\n\n".join(footnotes) if footnotes else "No footnotes to format"


@tool
def format_evidence_as_inline_references(evidence_json: str, answer_text: str) -> str:
    """Format evidence as inline references within an answer text.

    This adds superscript-style reference markers to the answer where
    evidence is used.

    Args:
        evidence_json: JSON string containing list of evidence objects.
        answer_text: The answer text to add references to.

    Returns:
        The answer with inline references and a reference list.
    """
    import json

    try:
        evidence_list = json.loads(evidence_json)
    except json.JSONDecodeError:
        return f"{answer_text}\n\n[Error: Could not parse evidence]"

    # Build reference list
    references = ["\n\n---\n**References:**"]
    for i, evidence in enumerate(evidence_list, 1):
        source = evidence.get("source_metadata", {})
        file_name = source.get("file_name", "Unknown")
        content = evidence.get("extracted_content", "")[:100]
        references.append(f'{i}. {file_name}: "{content}..."')

    # Add note about references (in real implementation, would parse answer for claims)
    result = f"{answer_text}\n\n[Based on {len(evidence_list)} source(s)]"
    result += "\n".join(references)

    return result


@tool
def create_evidence_summary_table(evidence_json: str) -> str:
    """Create a markdown table summarizing all evidence.

    Args:
        evidence_json: JSON string containing list of evidence objects.

    Returns:
        Markdown table summarizing the evidence.
    """
    import json

    try:
        evidence_list = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input"

    if not evidence_list:
        return "No evidence to summarize"

    # Create markdown table
    table_lines = [
        "| # | Source | Type | Content Preview |",
        "|---|--------|------|-----------------|",
    ]

    for i, evidence in enumerate(evidence_list, 1):
        source = evidence.get("source_metadata", {})
        file_name = source.get("file_name", "Unknown")[:20]
        file_type = source.get("file_type", "unknown")
        content = evidence.get("extracted_content", "")[:50].replace("|", "\\|")

        table_lines.append(f"| {i} | {file_name} | {file_type} | {content}... |")

    return "\n".join(table_lines)


@tool
def validate_evidence_completeness(evidence_json: str) -> str:
    """Check if evidence objects have all required fields populated.

    Args:
        evidence_json: JSON string containing list of evidence objects.

    Returns:
        Validation report indicating completeness of evidence.
    """
    import json

    try:
        evidence_list = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input"

    issues = []
    for i, evidence in enumerate(evidence_list, 1):
        evidence_issues = []

        # Check required fields
        if not evidence.get("evidence_id"):
            evidence_issues.append("missing evidence_id")

        source = evidence.get("source_metadata", {})
        if not source.get("file_name"):
            evidence_issues.append("missing file_name")

        if not evidence.get("extracted_content"):
            evidence_issues.append("missing extracted_content")

        coords = evidence.get("proof_coordinates", {})
        if not coords.get("document_location") and not coords.get(
            "structured_location"
        ):
            evidence_issues.append("missing location coordinates")

        if evidence_issues:
            issues.append(f"Evidence {i}: {', '.join(evidence_issues)}")

    if not issues:
        return f"✓ All {len(evidence_list)} evidence objects are complete"
    else:
        return "Issues found:\n" + "\n".join(issues)
