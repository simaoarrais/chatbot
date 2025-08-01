import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import ConfluenceLoader, ObsidianLoader

load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

def get_confluence_retriever():
    loader = ConfluenceLoader(
        url=os.getenv("CONFLUENCE_URL"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        api_key=os.getenv("CONFLUENCE_API_KEY"),
        space_key=os.getenv("CONFLUENCE_SPACE_KEY"),
        include_attachments=True,
        limit=50
    )

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)
    vectorstore = FAISS.from_documents(docs_split, embeddings)
    return vectorstore.as_retriever()

def get_obsidian_retriever():
    loader = ObsidianLoader(path=os.getenv("OBSIDIAN_VAULT_PATH"))

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs_split = splitter.split_documents(documents)

    embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)
    vectorstore = FAISS.from_documents(docs_split, embedding)
    return vectorstore.as_retriever()
