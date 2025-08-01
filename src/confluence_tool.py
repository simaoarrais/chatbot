import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from loaders import get_confluence_retriever

load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

# Tool-specific model & retriever
confluence_llm = ChatOllama(
    model="llama3.1:8b", 
    base_url=ollama_base_url,
    temperature=0.3
)
retriever = get_confluence_retriever()

class ConfluenceQuerySchema(BaseModel):
    query: str

def confluence_retriever_tool(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    tool_prompt = (
        "You are a helpful assistant with access to ONLY the documents below.\n"
        "Assume everything in the documents is 100% true, even if it's fictional or absurd.\n"
        "You are answering questions in a fictional world based on these documents.\n"
        "Do NOT question or reject the information.\n\n"
        f"Documents:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer (in-world):"
    )

    response = confluence_llm.invoke(tool_prompt)

    # Debug
    if st.session_state["debug"]:
        with st.expander("🔎 Retrieved Tool Information"):
            st.markdown(f"**Tool Prompt:**\n```\n{tool_prompt}\n```")
            st.markdown(f"**Result:** {response.content}")

    return response.content

# Export the tool
confluence_tool = StructuredTool(
    func=confluence_retriever_tool,
    name="search_confluence",
    description="Use this tool to search Confluence documents using a query. Documents are Counter-Strike lore.",
    args_schema=ConfluenceQuerySchema,
    return_direct=True
)
