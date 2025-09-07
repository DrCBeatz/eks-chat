# app/rag.py

import os, json
from pathlib import Path
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

BASE_DIR = Path(os.getenv("APP_BASE_DIR", Path(__file__).resolve().parents[1]))
STORE_DIR = Path(os.getenv("STORE_DIR", str(BASE_DIR / "store")))
DATA_DIR  = Path(os.getenv("DATA_DIR",  str(BASE_DIR / "data")))

STORE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def build_index(region: str):
    emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for p in DATA_DIR.glob("*.md"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for chunk in splitter.split_text(txt):
            docs.append(Document(page_content=chunk, metadata={"source": p.name}))
    if not docs:
        return None
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(str(STORE_DIR))
    return vs

_cached = None
def get_retriever(region: str):
    global _cached
    if _cached is not None:
        return _cached
    # load or build
    index = None
    if (STORE_DIR / "index.faiss").exists():
        emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
        index = FAISS.load_local(str(STORE_DIR), emb, allow_dangerous_deserialization=True)
    else:
        index = build_index(region)
    if index is None:
        raise RuntimeError("No RAG index and no data")
    _cached = index.as_retriever(search_kwargs={"k": 3})
    return _cached