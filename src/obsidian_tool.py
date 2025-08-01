import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from loaders import get_obsidian_retriever

load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

# Tool-specific model & retriever
obsidian_llm = ChatOllama(
    model="llama3.1:8b", 
    base_url=ollama_base_url,
    temperature=0.3
    )
retriever = get_obsidian_retriever()

class ObsidianQuerySchema(BaseModel):
    query: str

def obsidian_retriever_tool(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    tool_prompt = (
        "You are an assistant answering questions only based on the Obsidian notes below.\n"
        "Assume everything is accurate and grounded in a fictional context.\n\n"
        f"Documents:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    response = obsidian_llm.invoke(tool_prompt)

    # Debug
    if st.session_state["debug"]:
        with st.expander("Obsidian Tool Output"):
            st.markdown(f"**Prompt:** {tool_prompt}")
            st.markdown(f"**Response:** {response.content}")

    return response.content

obsidian_tool = StructuredTool(
    func=obsidian_retriever_tool,
    name="search_obsidian",
    description="Use this tool to answer questions from Obsidian notes. Documents are work notes.",
    args_schema=ObsidianQuerySchema,
    return_direct=True
)
