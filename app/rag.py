# app/rag.py
import os
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import boto3
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ----- Paths & env -----
BASE_DIR   = Path(os.getenv("APP_BASE_DIR", Path(__file__).resolve().parents[1]))
STORE_DIR  = Path(os.getenv("STORE_DIR", str(BASE_DIR / "store")))
DATA_DIR   = Path(os.getenv("DATA_DIR",  str(BASE_DIR / "data")))

# S3 source (optional)
RAG_S3_BUCKET = os.getenv("RAG_S3_BUCKET", "").strip()
RAG_S3_PREFIX = os.getenv("RAG_S3_PREFIX", "").strip()

# Tuning
ALLOWED_SUFFIXES = [s.strip().lower() for s in os.getenv("RAG_SUFFIXES", ".md,.txt,.mdx").split(",") if s.strip()]
MAX_OBJECT_MB    = float(os.getenv("RAG_MAX_OBJECT_MB", "5"))
CHUNK_SIZE       = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP    = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

STORE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

_meta_file = STORE_DIR / "meta.json"
_lock = threading.Lock()
_cached_retriever = None   # type: ignore
_cached_vs = None          # type: ignore

def _write_meta(meta: dict):
    _meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def _read_meta() -> dict:
    if _meta_file.exists():
        try:
            return json.loads(_meta_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def _iter_local_docs() -> List[Document]:
    docs: List[Document] = []
    for p in DATA_DIR.glob("*"):
        if not p.is_file():
            continue
        if not any(str(p).lower().endswith(suf) for suf in ALLOWED_SUFFIXES):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        docs.append(Document(page_content=txt, metadata={"source": p.name}))
    return docs

def _iter_s3_docs(region: str, bucket: str, prefix: str) -> List[Document]:
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    docs: List[Document] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size_mb = obj["Size"] / (1024 * 1024)
            if size_mb > MAX_OBJECT_MB:
                continue
            if not any(key.lower().endswith(suf) for suf in ALLOWED_SUFFIXES):
                continue
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "ignore")
            except Exception:
                continue
            docs.append(Document(page_content=body, metadata={"source": f"s3://{bucket}/{key}", "bytes": obj["Size"]}))
    return docs

def _save_faiss(vs: FAISS):
    vs.save_local(str(STORE_DIR))

def _load_faiss(emb: BedrockEmbeddings) -> FAISS:
    return FAISS.load_local(str(STORE_DIR), emb, allow_dangerous_deserialization=True)

def _build_vs_from_docs(region: str, docs_raw: List[Document]) -> Tuple[FAISS, dict]:
    if not docs_raw:
        raise RuntimeError("No documents to index")
    docs = _split_docs(docs_raw)
    emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
    vs = FAISS.from_documents(docs, emb)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "docs_raw": len(docs_raw),
        "chunks": len(docs),
        "source": "s3" if RAG_S3_BUCKET else "local",
        "bucket": RAG_S3_BUCKET or None,
        "prefix": RAG_S3_PREFIX or None,
        "model": "amazon.titan-embed-text-v2:0",
    }
    _save_faiss(vs)
    _write_meta(meta)
    return vs, meta

def build_index_local(region: str) -> dict:
    docs_raw = _iter_local_docs()
    vs, meta = _build_vs_from_docs(region, docs_raw)
    return meta

def build_index_s3(region: str) -> dict:
    if not RAG_S3_BUCKET:
        raise RuntimeError("RAG_S3_BUCKET is not set")
    docs_raw = _iter_s3_docs(region, RAG_S3_BUCKET, RAG_S3_PREFIX)
    vs, meta = _build_vs_from_docs(region, docs_raw)
    return meta

def status() -> dict:
    meta = _read_meta()
    exists = (STORE_DIR / "index.faiss").exists()
    meta["exists"] = exists
    return meta

def get_retriever(region: str):
    """Return a cached retriever; load FAISS if present, else build from configured source."""
    global _cached_retriever, _cached_vs
    with _lock:
        if _cached_retriever is not None:
            return _cached_retriever
        emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
        if (STORE_DIR / "index.faiss").exists():
            _cached_vs = _load_faiss(emb)
        else:
            # Build from S3 if configured, else from /data
            if RAG_S3_BUCKET:
                build_index_s3(region)
            else:
                build_index_local(region)
            _cached_vs = _load_faiss(emb)
        _cached_retriever = _cached_vs.as_retriever(search_kwargs={"k": 3})
        return _cached_retriever

def reindex(region: str) -> dict:
    """Force rebuild from configured source and refresh cache."""
    global _cached_retriever, _cached_vs
    with _lock:
        if RAG_S3_BUCKET:
            meta = build_index_s3(region)
        else:
            meta = build_index_local(region)
        emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name=region)
        _cached_vs = _load_faiss(emb)
        _cached_retriever = _cached_vs.as_retriever(search_kwargs={"k": 3})
        return meta

def debug_search(region: str, query: str, k: int = 3) -> List[dict]:
    ret = get_retriever(region)
    docs = ret.get_relevant_documents(query)[:k]
    return [{"content": d.page_content, "meta": d.metadata} for d in docs]