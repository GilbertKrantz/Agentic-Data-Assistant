from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


from app.agents import OrchestratorAgent


# Test queries for the multi-agent system
TEST_QUERIES = [
    # "How does the daily or monthly fraud rate fluctuate over the two-year period?",
    # "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?",
    # "What are the primary methods by which credit card fraud is committed?",
    "What are the core components of an effective fraud detection system, according to the authors?",
    # "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
    # "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
]


def main() -> None:
    """Test the Orchestrator Agent with various fraud-related queries."""
    print("=" * 80)
    print("MULTI-AGENT FRAUD DETECTION SYSTEM TEST")
    print("=" * 80)
    print()

    # Initialize the orchestrator agent
    agent = OrchestratorAgent()

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print("=" * 80)

        try:
            # Invoke the orchestrator
            response = agent.invoke(query)

            print(f"\n📊 Status: {response.status.value}")
            print(f"🧠 Thought Process: {response.thought_process}")
            print(f"📈 Confidence Score: {response.confidence_score:.2f}")
            print(f"\n📝 Answer:\n{response.final_answer}")

            if response.supporting_evidence:
                print(
                    f"\n📚 Supporting Evidence ({len(response.supporting_evidence)} sources):"
                )
                for j, evidence in enumerate(response.supporting_evidence, 1):
                    print(
                        f"  [{j}] {evidence.source_metadata.file_name}: {evidence.extracted_content}"
                    )

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

        print("\n" + "-" * 80)

    print("\n✅ All queries processed.")


if __name__ == "__main__":
    main()
