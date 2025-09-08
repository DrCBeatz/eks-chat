# app/rag.py
import os, json, time
from pathlib import Path
from typing import List, Optional, Tuple
import boto3
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from threading import Lock

# Paths
BASE_DIR = Path(os.getenv("APP_BASE_DIR", Path(__file__).resolve().parents[1]))
STORE_DIR = Path(os.getenv("STORE_DIR", str(BASE_DIR / "store")))
DATA_DIR  = Path(os.getenv("DATA_DIR",  str(BASE_DIR / "data")))
META_PATH = STORE_DIR / "meta.json"

# Optional S3 config via env
RAG_S3_BUCKET = os.getenv("RAG_S3_BUCKET")
RAG_S3_PREFIX = os.getenv("RAG_S3_PREFIX", "docs/")

STORE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

_lock = Lock()
_cached_retriever = None
_cached_region = None

def _save_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, indent=2))

def _load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}

def _split_docs(texts: List[Tuple[str, str]]) -> List[Document]:
    """texts: list of (content, source) -> list[Document]."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs: List[Document] = []
    for content, source in texts:
        for chunk in splitter.split_text(content):
            docs.append(Document(page_content=chunk, metadata={"source": source}))
    return docs

def _embed_and_save(docs: List[Document], region: str) -> FAISS:
    emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(str(STORE_DIR))
    return vs

def build_index_from_local(region: str) -> Optional[FAISS]:
    texts: List[Tuple[str, str]] = []
    for p in DATA_DIR.glob("*.md"):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            texts.append((content, p.name))
        except Exception:
            continue
    if not texts:
        return None
    docs = _split_docs(texts)
    vs = _embed_and_save(docs, region)
    _save_meta({
        "source": "local",
        "files": len(texts),
        "chunks": len(docs),
        "time": int(time.time())
    })
    return vs

def build_index_from_s3(region: str, bucket: str, prefix: str) -> Optional[FAISS]:
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    texts: List[Tuple[str, str]] = []
    file_count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            low = key.lower()
            if not (low.endswith(".md") or low.endswith(".txt")):
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "ignore")
            texts.append((body, f"s3://{bucket}/{key}"))
            file_count += 1

    if not texts:
        return None

    docs = _split_docs(texts)
    vs = _embed_and_save(docs, region)
    _save_meta({
        "source": "s3",
        "bucket": bucket,
        "prefix": prefix,
        "files": file_count,
        "chunks": len(docs),
        "time": int(time.time())
    })
    return vs

def _load_or_build(region: str) -> Optional[FAISS]:
    # Prefer existing index on disk
    if (STORE_DIR / "index.faiss").exists():
        emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
        return FAISS.load_local(str(STORE_DIR), emb, allow_dangerous_deserialization=True)

    # Else try S3 if configured
    if RAG_S3_BUCKET:
        return build_index_from_s3(region, RAG_S3_BUCKET, RAG_S3_PREFIX)

    # Else fallback to local ./data
    return build_index_from_local(region)

def get_retriever(region: str):
    global _cached_retriever, _cached_region
    with _lock:
        if _cached_retriever is not None and _cached_region == region:
            return _cached_retriever
        index = _load_or_build(region)
        if index is None:
            raise RuntimeError("No RAG index available (no local data and/or empty S3 prefix).")
        _cached_retriever = index.as_retriever(search_kwargs={"k": 3})
        _cached_region = region
        return _cached_retriever

def rebuild_from_s3(region: str, bucket: str, prefix: str):
    global _cached_retriever, _cached_region
    with _lock:
        vs = build_index_from_s3(region, bucket, prefix)
        if vs is None:
            raise RuntimeError("No documents found in S3 for the given bucket/prefix.")
        _cached_retriever = vs.as_retriever(search_kwargs={"k": 3})
        _cached_region = region
        return _cached_retriever

def get_status() -> dict:
    meta = _load_meta()
    has_index = (STORE_DIR / "index.faiss").exists()
    meta["has_index"] = bool(has_index)
    meta["store_dir"] = str(STORE_DIR)
    return meta

def search(region: str, query: str, k: int = 3):
    """Return top-k documents (no scores) for debugging."""
    ret = get_retriever(region)
    docs = ret.get_relevant_documents(query)[:k]
    out = []
    for d in docs:
        out.append({
            "source": d.metadata.get("source"),
            "text": d.page_content[:400],
        })
    return out
