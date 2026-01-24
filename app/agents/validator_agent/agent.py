"""
Validator Agent - Handles validation and fact-checking of agent responses.
"""

import json
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
    DocumentLocation,
)
from app.agents.validator_agent.tools import (
    validate_answer_against_evidence,
    check_numerical_accuracy,
    assess_confidence_score,
    check_logical_consistency,
    generate_validation_summary,
)
from app.agents.validator_agent.prompts import VALIDATOR_SYSTEM_PROMPT


class ValidatorSkillMiddleware(AgentMiddleware):
    """Middleware that provides validation skills to the agent."""

    tools = [
        validate_answer_against_evidence,
        check_numerical_accuracy,
        assess_confidence_score,
        check_logical_consistency,
        generate_validation_summary,
    ]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        """Inject skill context into the system prompt."""
        skills_addendum = """

## Validation Skills Available

### Evidence Validation Skill
Use `validate_answer_against_evidence` for:
- Checking claim-evidence alignment
- Identifying unsupported statements
- Calculating support coverage

### Numerical Accuracy Skill
Use `check_numerical_accuracy` for:
- Verifying numbers in answer match evidence
- Identifying unverified numerical claims
- Flagging potential calculation errors

### Confidence Assessment Skill
Use `assess_confidence_score` for:
- Evaluating if confidence is justified
- Suggesting appropriate confidence levels
- Identifying over/under-confident responses

### Logical Consistency Skill
Use `check_logical_consistency` for:
- Finding contradictions in answers
- Checking hedging vs definitive language
- Ensuring logical coherence

### Summary Skill
Use `generate_validation_summary` for:
- Creating final validation report
- Providing pass/fail status
- Summarizing all validation results
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


class ValidatorAgent:
    """Agent that validates and fact-checks responses from other agents."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """Initialize the Validator Agent.

        Args:
            model_name: The Gemini model to use.
        """
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=settings.gemini_api_key,
        )
        self.agent = create_agent(
            model=self.model,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            middleware=[ValidatorSkillMiddleware()],
            response_format=ToolStrategy(StandardAgentResponse),
        )

    def invoke(
        self,
        query: str,
        response_to_validate: Optional[StandardAgentResponse] = None,
    ) -> StandardAgentResponse:
        """Execute the validator agent.

        Args:
            query: The validation request.
            response_to_validate: Optional StandardAgentResponse to validate.

        Returns:
            StandardAgentResponse with validation results.
        """
        try:
            # Prepare validation context
            if response_to_validate:
                evidence_dicts = [
                    e.model_dump() for e in response_to_validate.supporting_evidence
                ]
                evidence_json = json.dumps(evidence_dicts)

                validation_request = f"""
{query}

Response to Validate:
- Agent: {response_to_validate.agent_id.value}
- Status: {response_to_validate.status.value}
- Confidence: {response_to_validate.confidence_score}

Answer:
{response_to_validate.final_answer}

Evidence (JSON):
{evidence_json}

Please perform a comprehensive validation:
1. Validate answer against evidence
2. Check numerical accuracy
3. Assess if confidence score is appropriate
4. Check for logical consistency
5. Generate a validation summary
"""
            else:
                validation_request = query

            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": validation_request}]}
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

            # Determine validation status
            validation_passed = self._determine_validation_status(content)
            status = (
                ResponseStatus.SUCCESS if validation_passed else ResponseStatus.FAILED
            )

            # Create validation evidence
            validation_evidence = self._create_validation_evidence(
                content, response_to_validate
            )

            return StandardAgentResponse(
                agent_id=AgentId.VALIDATOR_AGENT,
                status=status,
                thought_process=self._extract_thought_process(result["messages"]),
                final_answer=content,
                supporting_evidence=[validation_evidence],
                confidence_score=0.85 if validation_passed else 0.6,
            )

        except Exception as e:
            return StandardAgentResponse(
                agent_id=AgentId.VALIDATOR_AGENT,
                status=ResponseStatus.FAILED,
                thought_process=f"Error occurred: {str(e)}",
                final_answer=f"Failed to validate response: {str(e)}",
                supporting_evidence=[],
                confidence_score=0.0,
            )

    def validate_response(
        self, response: StandardAgentResponse
    ) -> StandardAgentResponse:
        """Validate a StandardAgentResponse from another agent.

        This is a convenience method that wraps invoke() with appropriate
        validation query.

        Args:
            response: The StandardAgentResponse to validate.

        Returns:
            StandardAgentResponse containing validation results.
        """
        return self.invoke(
            query="Perform comprehensive validation of the following response.",
            response_to_validate=response,
        )

    def quick_validate(
        self,
        answer: str,
        evidence: List[UniversalEvidenceObject],
        confidence: float,
    ) -> Dict[str, Any]:
        """Perform quick validation without full agent invocation.

        Args:
            answer: The answer to validate.
            evidence: List of evidence objects.
            confidence: The claimed confidence score.

        Returns:
            Dictionary with validation results.
        """
        evidence_dicts = [e.model_dump() for e in evidence]
        evidence_json = json.dumps(evidence_dicts)

        # Run validation tools directly
        from app.agents.validator_agent.tools import (
            validate_answer_against_evidence,
            check_numerical_accuracy,
            assess_confidence_score,
        )

        evidence_check = validate_answer_against_evidence.invoke(
            {"answer": answer, "evidence_json": evidence_json}
        )
        numerical_check = check_numerical_accuracy.invoke(
            {"answer": answer, "evidence_json": evidence_json}
        )
        confidence_check = assess_confidence_score.invoke(
            {
                "answer": answer,
                "evidence_json": evidence_json,
                "claimed_confidence": confidence,
            }
        )

        # Determine overall status
        passed = "PASSED" in evidence_check or "✓" in evidence_check
        warnings = "WARNING" in evidence_check or "⚠" in evidence_check

        return {
            "status": "passed" if passed else ("warning" if warnings else "failed"),
            "evidence_validation": evidence_check,
            "numerical_accuracy": numerical_check,
            "confidence_assessment": confidence_check,
        }

    def _determine_validation_status(self, content: str) -> bool:
        """Determine if validation passed based on content."""
        content_upper = content.upper()

        # Check for explicit pass/fail indicators
        if "VALIDATED" in content_upper or "PASSED" in content_upper:
            if "FAILED" not in content_upper:
                return True

        if "FAILED" in content_upper or "NEEDS REVISION" in content_upper:
            return False

        # Default to passed if no explicit failure
        return "✓" in content or "🟢" in content

    def _extract_thought_process(self, messages: List[Any]) -> str:
        """Extract the agent's reasoning from messages."""
        thoughts = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    thoughts.append(f"Ran: {tc.get('name', 'unknown')}")
        return " -> ".join(thoughts) if thoughts else "Direct validation"

    def _create_validation_evidence(
        self,
        validation_content: str,
        original_response: Optional[StandardAgentResponse],
    ) -> UniversalEvidenceObject:
        """Create an evidence object for the validation."""
        from uuid import uuid4

        source_info = "validation"
        if original_response:
            source_info = f"validation_of_{original_response.agent_id.value}"

        return UniversalEvidenceObject(
            evidence_id=str(uuid4()),
            source_metadata=SourceMetadata(
                file_id="validation_result",
                file_name=source_info,
                file_type=FileType.DOCUMENT,
            ),
            extracted_content=validation_content[:500],
            proof_coordinates=ProofCoordinates(
                document_location=DocumentLocation(
                    original_text_snippet=validation_content[:200]
                )
            ),
        )
