"""
Tools for the Validator Agent - handles validation and fact-checking.
"""

from langchain.tools import tool


@tool
def validate_answer_against_evidence(answer: str, evidence_json: str) -> str:
    """Validate that an answer is supported by the provided evidence.

    Checks if the claims in the answer can be traced back to the evidence.

    Args:
        answer: The answer text to validate.
        evidence_json: JSON string containing list of evidence objects.

    Returns:
        Validation report with supported/unsupported claims.
    """
    import json

    try:
        evidence_list = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input for evidence"

    # Extract all evidence content
    evidence_content = []
    for e in evidence_list:
        content = e.get("extracted_content", "")
        evidence_content.append(content.lower())

    all_evidence_text = " ".join(evidence_content)

    # Simple heuristic validation - check for key terms
    report = ["=== Answer Validation Report ===\n"]

    # Split answer into sentences
    sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if s.strip()]

    supported_count = 0
    unsupported_sentences = []

    for sentence in sentences[:10]:  # Limit to first 10 sentences
        # Check if key words from sentence appear in evidence
        words = set(sentence.lower().split())
        # Remove common words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "to",
            "of",
            "and",
            "in",
            "that",
            "it",
            "for",
            "on",
            "with",
        }
        key_words = words - stop_words

        matches = sum(1 for w in key_words if w in all_evidence_text)
        coverage = matches / len(key_words) if key_words else 0

        if coverage > 0.3:
            supported_count += 1
        else:
            unsupported_sentences.append(sentence[:100])

    total = len(sentences[:10])
    support_rate = (supported_count / total * 100) if total > 0 else 0

    report.append(f"Total sentences analyzed: {total}")
    report.append(f"Supported by evidence: {supported_count} ({support_rate:.1f}%)")
    report.append(f"Evidence sources used: {len(evidence_list)}")

    if unsupported_sentences:
        report.append(
            f"\nPotentially unsupported claims ({len(unsupported_sentences)}):"
        )
        for s in unsupported_sentences[:3]:
            report.append(f'  - "{s}..."')

    if support_rate >= 70:
        report.append("\n✓ VALIDATION PASSED: Answer is well-supported by evidence")
    elif support_rate >= 40:
        report.append(
            "\n⚠ VALIDATION WARNING: Some claims may need additional evidence"
        )
    else:
        report.append("\n✗ VALIDATION FAILED: Answer lacks sufficient evidence support")

    return "\n".join(report)


@tool
def check_numerical_accuracy(answer: str, evidence_json: str) -> str:
    """Check if numerical claims in the answer match the evidence.

    Args:
        answer: The answer text containing numerical claims.
        evidence_json: JSON string containing evidence with numerical data.

    Returns:
        Report on numerical accuracy.
    """
    import json
    import re

    try:
        evidence_list = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input"

    # Extract numbers from answer
    answer_numbers = set(re.findall(r"\d+\.?\d*", answer))

    # Extract numbers from evidence
    evidence_numbers = set()
    for e in evidence_list:
        content = e.get("extracted_content", "")
        nums = re.findall(r"\d+\.?\d*", content)
        evidence_numbers.update(nums)

    report = ["=== Numerical Accuracy Check ===\n"]
    report.append(f"Numbers found in answer: {len(answer_numbers)}")
    report.append(f"Numbers found in evidence: {len(evidence_numbers)}")

    # Check matches
    matched = answer_numbers & evidence_numbers
    unmatched = answer_numbers - evidence_numbers

    if matched:
        report.append(f"\n✓ Verified numbers: {', '.join(list(matched)[:5])}")
    if unmatched:
        report.append(f"\n⚠ Unverified numbers: {', '.join(list(unmatched)[:5])}")
        report.append("  (These may be calculations or need manual verification)")

    return "\n".join(report)


@tool
def assess_confidence_score(
    answer: str, evidence_json: str, claimed_confidence: float
) -> str:
    """Assess whether the claimed confidence score is appropriate.

    Args:
        answer: The answer provided.
        evidence_json: The supporting evidence.
        claimed_confidence: The confidence score claimed by the agent.

    Returns:
        Assessment of confidence appropriateness.
    """
    import json

    try:
        evidence_list = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input"

    report = ["=== Confidence Assessment ===\n"]
    report.append(f"Claimed confidence: {claimed_confidence:.2f}")

    # Factors affecting confidence
    factors = []
    suggested_confidence = 0.5  # Start at neutral

    # Evidence count
    evidence_count = len(evidence_list)
    if evidence_count == 0:
        factors.append("No evidence provided (-0.3)")
        suggested_confidence -= 0.3
    elif evidence_count == 1:
        factors.append("Single source (+0.1)")
        suggested_confidence += 0.1
    elif evidence_count >= 2:
        factors.append(f"Multiple sources ({evidence_count}) (+0.2)")
        suggested_confidence += 0.2

    # Answer length/detail
    answer_length = len(answer.split())
    if answer_length < 20:
        factors.append("Brief answer (+0.0)")
    elif answer_length > 100:
        factors.append("Detailed answer (+0.1)")
        suggested_confidence += 0.1

    # Evidence quality (check for structured data)
    has_structured = any(
        e.get("source_metadata", {}).get("file_type") == "structured_data"
        for e in evidence_list
    )
    if has_structured:
        factors.append("Contains structured data (+0.1)")
        suggested_confidence += 0.1

    suggested_confidence = max(0.0, min(1.0, suggested_confidence))

    report.append("\nFactors considered:")
    for f in factors:
        report.append(f"  • {f}")

    report.append(f"\nSuggested confidence: {suggested_confidence:.2f}")

    diff = abs(claimed_confidence - suggested_confidence)
    if diff < 0.15:
        report.append("✓ Claimed confidence is appropriate")
    elif claimed_confidence > suggested_confidence:
        report.append("⚠ Claimed confidence may be too high")
    else:
        report.append("⚠ Claimed confidence may be too low")

    return "\n".join(report)


@tool
def check_logical_consistency(answer: str) -> str:
    """Check the answer for logical consistency and contradictions.

    Args:
        answer: The answer text to check.

    Returns:
        Report on logical consistency.
    """
    report = ["=== Logical Consistency Check ===\n"]

    # Simple checks for contradictory phrases
    contradiction_pairs = [
        ("always", "never"),
        ("all", "none"),
        ("increase", "decrease"),
        ("higher", "lower"),
        ("more", "less"),
    ]

    answer_lower = answer.lower()
    found_contradictions = []

    for word1, word2 in contradiction_pairs:
        if word1 in answer_lower and word2 in answer_lower:
            found_contradictions.append(f"'{word1}' and '{word2}'")

    # Check for hedging language
    hedging_words = ["might", "maybe", "possibly", "perhaps", "could be", "uncertain"]
    hedging_count = sum(1 for w in hedging_words if w in answer_lower)

    # Check for definitive claims
    definitive_words = ["always", "never", "definitely", "certainly", "absolutely"]
    definitive_count = sum(1 for w in definitive_words if w in answer_lower)

    report.append(f"Hedging language used: {hedging_count} instances")
    report.append(f"Definitive claims: {definitive_count} instances")

    if found_contradictions:
        report.append("\n⚠ Potential contradictions found:")
        for c in found_contradictions:
            report.append(f"  - Contains both {c}")
    else:
        report.append("\n✓ No obvious contradictions detected")

    if definitive_count > 2 and hedging_count == 0:
        report.append(
            "\n⚠ Note: Answer contains many definitive claims - verify evidence support"
        )

    return "\n".join(report)


@tool
def generate_validation_summary(
    answer: str, evidence_json: str, validation_results: str
) -> str:
    """Generate a final validation summary with recommendations.

    Args:
        answer: The original answer.
        evidence_json: The supporting evidence.
        validation_results: Results from other validation checks.

    Returns:
        Final validation summary with pass/fail status.
    """
    import json

    report = ["=" * 50]
    report.append("VALIDATION SUMMARY")
    report.append("=" * 50 + "\n")

    try:
        evidence_list = json.loads(evidence_json)
        evidence_count = len(evidence_list)
    except json.JSONDecodeError:
        evidence_count = 0

    # Parse validation results for key indicators
    is_passed = "PASSED" in validation_results.upper()
    has_warnings = "WARNING" in validation_results.upper() or "⚠" in validation_results
    has_failures = "FAILED" in validation_results.upper() or "✗" in validation_results

    report.append(f"Evidence sources: {evidence_count}")
    report.append(f"Answer length: {len(answer.split())} words")
    report.append("")

    if has_failures:
        report.append("🔴 OVERALL STATUS: NEEDS REVISION")
        report.append("   The answer has significant issues that should be addressed.")
    elif has_warnings:
        report.append("🟡 OVERALL STATUS: ACCEPTABLE WITH CAVEATS")
        report.append("   The answer is usable but has some concerns.")
    else:
        report.append("🟢 OVERALL STATUS: VALIDATED")
        report.append("   The answer meets quality standards.")

    report.append("\n" + "-" * 50)
    report.append("Validation Details:")
    report.append(validation_results)

    return "\n".join(report)
