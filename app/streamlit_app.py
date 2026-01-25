import streamlit as st
import sys
import traceback
from pathlib import Path

# Add the project root to sys.path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents import OrchestratorAgent
from app.agents.orchestrator.tools import reset_evidence_pool

# Page Configuration
st.set_page_config(page_title="Fraud Detection Assistant", page_icon="🕵️")

# Initialize Agent via Session State (Singleton-like behavior)
if "agent" not in st.session_state:
    try:
        st.session_state.agent = OrchestratorAgent()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.session_state.agent = None


def main():
    st.title("🕵️ Multi-Agent Fraud Detection System")
    st.markdown("Ask questions about fraud patterns, data analysis, or documentation.")

    # Sidebar for Status/Metrics (Optional, but looks professional in Streamlit)
    with st.sidebar:
        st.header("System Status")
        if st.session_state.agent:
            st.success("Agent Ready")
        else:
            st.error("Agent Offline")

        if st.button("Clear Conversation/Cache"):
            reset_evidence_pool()
            st.rerun()

    # User Input
    query = st.text_input(
        "Query",
        placeholder="e.g., What are the core components of an effective fraud detection system?",
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("Analyze", type="primary")

    if analyze_clicked or query:
        if not query:
            st.warning("Please enter a query.")
            return

        if not st.session_state.agent:
            st.error("Agent not initialized. Please check logs.")
            return

        # Create a progress container
        progress_container = st.container()
        progress_status = st.empty()

        # Reset evidence pool for new session
        reset_evidence_pool()

        # Define progress callback function
        progress_messages = []

        def progress_callback(message: str):
            """Callback to receive progress updates."""
            progress_messages.append(message)
            # Update the progress status with latest messages (last 5)
            with progress_status.container():
                st.markdown("### 📊 Query Progress")
                for msg in progress_messages[-5:]:
                    st.write(f"✓ {msg}")

        try:
            # Call the agent with progress callback
            with st.spinner("Analyzing findings and gathering evidence..."):
                response = st.session_state.agent.invoke(
                    query, progress_callback=progress_callback
                )

            # --- Render Output ---

            # Final Answer Section
            st.markdown(f"# 📝 Final Answer")
            st.write(response.final_answer)

            st.divider()

            # Metrics Columns
            col1, col2 = st.columns(2)
            status_icon = "✅" if response.status.value == "success" else "⚠️"
            col1.metric("Status", f"{status_icon} {response.status.value.title()}")

            if response.confidence_score is not None:
                col2.metric("Confidence Score", f"{response.confidence_score:.2f}")

            # Thought Process (Expandable is usually cleaner for logs)
            if response.thought_process:
                with st.expander("🧠 View Thought Process", expanded=False):
                    st.markdown(response.thought_process)

            # Supporting Evidence
            if response.supporting_evidence:
                st.header("📚 Supporting Evidence")

                for i, evidence in enumerate(response.supporting_evidence, 1):
                    with st.container():
                        st.subheader(
                            f"Evidence {i}: {evidence.source_metadata.file_name}"
                        )

                        # Build Metadata string
                        meta_info = []
                        if evidence.source_metadata.file_type:
                            meta_info.append(evidence.source_metadata.file_type.value)

                        if evidence.proof_coordinates:
                            if evidence.proof_coordinates.document_location:
                                loc = evidence.proof_coordinates.document_location
                                if loc.page_number:
                                    meta_info.append(f"Page {loc.page_number}")

                            if evidence.proof_coordinates.structured_location:
                                loc = evidence.proof_coordinates.structured_location
                                if loc.row_indices:
                                    meta_info.append(f"Rows {loc.row_indices}")

                        if meta_info:
                            st.caption(" | ".join(meta_info))

                        st.code(evidence.extracted_content, language="text")
                        st.divider()

        except Exception as e:
            st.error("## ❌ Error")
            st.error(f"An error occurred: {str(e)}")
            with st.expander("View Traceback"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
