import streamlit as st
import asyncio
import tempfile
import os
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import nest_asyncio

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

nest_asyncio.apply()

from app.graph import AssistantGraph
from app.utils.ingest import PDFIngester
from app.interfaces.qdrant import QdrantInterface
from app.config import config


class StreamlitApp:
    def __init__(self):
        self.assistant_graph = AssistantGraph()
        self.pdf_ingester = PDFIngester()
        self.qdrant_interface = QdrantInterface()

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

    def setup_page(self):
        st.set_page_config(
            page_title="AI Assistant - Weather & RAG",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🤖 AI Assistant")
        st.markdown("**Weather Information & Document Q&A with RAG**")

    def render_sidebar(self):
        with st.sidebar:
            st.header("📁 Document Upload")

            uploaded_file = st.file_uploader(
                "Upload a PDF document",
                type=["pdf"],
                help="Upload a PDF to ask questions about its content",
            )

            if uploaded_file is not None:
                self.handle_file_upload(uploaded_file)

            if st.session_state.uploaded_files:
                st.subheader("📚 Uploaded Documents")
                for file_info in st.session_state.uploaded_files:
                    st.write(f"• {file_info['name']} ({file_info['chunks']} chunks)")

                if st.button("🗑️ Clear All Documents"):
                    self.clear_all_documents()

            st.subheader("📊 Database Statistics")
            try:
                stats = self.qdrant_interface.get_collection_info()
                st.write(f"Total Documents: {stats.get('points_count', 'N/A')}")
                st.write(f"Collection: {stats.get('name', 'N/A')}")
                st.write(f"Status: {stats.get('status', 'N/A')}")
            except Exception as e:
                st.warning("⚠️ Unable to fetch database statistics")
                st.info(
                    "This might be due to API compatibility. The app will still function normally."
                )
                st.write("Collection: ai_assistant_docs")
                st.write("Status: Available")

            st.subheader("⚙️ Configuration")
            st.write(f"Model: {config.google_ai.model}")
            st.write(f"Temperature: {config.google_ai.temperature}")
            st.write(f"Chunk Size: {config.chunk_size}")

    def handle_file_upload(self, uploaded_file):
        try:
            file_name = uploaded_file.name
            existing_files = [f["name"] for f in st.session_state.uploaded_files]

            if file_name in existing_files:
                st.warning(f"File '{file_name}' already uploaded and processed.")
                return

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            with st.spinner(f"Processing {file_name}..."):
                try:
                    result = asyncio.run(
                        self.pdf_ingester.process_pdf(
                            pdf_path=tmp_file_path, source_name=file_name
                        )
                    )

                    st.session_state.uploaded_files.append(
                        {
                            "name": file_name,
                            "chunks": result["total_chunks"],
                            "characters": result["total_characters"],
                        }
                    )

                    st.success(
                        f"✅ Processed {file_name}: {result['total_chunks']} chunks"
                    )

                finally:
                    os.unlink(tmp_file_path)

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

    def clear_all_documents(self):
        try:
            sources = [f["name"] for f in st.session_state.uploaded_files]

            for source in sources:
                self.qdrant_interface.delete_documents_by_source(source)

            st.session_state.uploaded_files = []
            st.success("🗑️ All documents cleared!")

        except Exception as e:
            st.error(f"❌ Error clearing documents: {str(e)}")

    def render_chat_interface(self):
        st.subheader("💬 Chat Interface")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("📊 Processing Details"):
                        metadata = message["metadata"]
                        st.write(
                            f"**Query Type:** {metadata.get('query_type', 'unknown')}"
                        )
                        st.write(
                            f"**Processing Time:** {metadata.get('processing_time', 0):.2f}s"
                        )

                        if "weather_data" in metadata:
                            weather = metadata["weather_data"]
                            st.write(
                                f"**Weather:** {weather['city']} - {weather['temperature']}°C"
                            )

                        if "rag_data" in metadata:
                            rag = metadata["rag_data"]
                            st.write(f"**Sources:** {', '.join(rag['sources'])}")
                            st.write(f"**Chunks Retrieved:** {rag['chunks_retrieved']}")

        if prompt := st.chat_input("Ask about weather or your uploaded documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        final_state = asyncio.run(
                            self.assistant_graph.process_query(
                                query=prompt, session_id=st.session_state.session_id
                            )
                        )

                        response = self.assistant_graph.get_response_text(final_state)
                        metadata = self.assistant_graph.get_processing_metadata(
                            final_state
                        )

                        st.write(response)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "metadata": metadata,
                            }
                        )

                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")

    def render_examples(self):
        st.subheader("💡 Example Queries")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🌤️ Weather Queries:**")
            st.write("• What's the weather in London?")
            st.write("• Temperature in New York")
            st.write("• Weather forecast for Tokyo")
            st.write("• How's the weather in Paris?")

        with col2:
            st.markdown("**📚 Document Queries:**")
            st.write("• What are the main topics?")
            st.write("• Summarize the key points")
            st.write("• What does the document say about...?")
            st.write("• Explain the methodology")

    def run(self):
        self.setup_page()
        self.render_sidebar()
        self.render_chat_interface()
        self.render_examples()


def main():
    try:
        config.validate_config()

        app = StreamlitApp()
        app.run()

    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
