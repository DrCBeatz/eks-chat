# app/main.py
import os, json, re
from typing import List, Optional, Generator, Tuple

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from botocore.exceptions import ClientError
import boto3

# ----------- Config -----------
USE_LANGCHAIN = os.getenv("USE_LANGCHAIN", "true").lower() == "true"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID   = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

# RAG config
RAG_S3_BUCKET = os.getenv("RAG_S3_BUCKET")            # e.g. eks-chat-rag-...-us-east-1
RAG_S3_PREFIX = os.getenv("RAG_S3_PREFIX", "docs/")   # e.g. "docs/"
RAG_TOKEN     = os.getenv("RAG_TOKEN", "").strip()    # shared secret for /rag/reindex

app = FastAPI(title="Bedrock Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ----------- Models -----------
class ChatTurn(BaseModel):
    role: str   # "user" | "assistant" | "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatTurn]
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.4
    max_tokens: Optional[int] = 512
    rag: Optional[bool] = False
    k: Optional[int] = 3
    strict: Optional[bool] = False

class ChatStreamRequest(ChatRequest):
    system: Optional[str] = None  # additional system prompt

# ----------- Prompts -----------
STRICT_TMPL = (
    "Use ONLY the following policy context to answer. "
    "If it does not directly address the question, reply exactly: Not in policy.\n\n"
    "<CONTEXT>\n{context}\n</CONTEXT>\n"
)

NONSTRICT_TMPL = (
    "Use the following context if helpful. If not helpful, answer normally.\n\n"
    "<CONTEXT>\n{context}\n</CONTEXT>\n"
)

def _format_context_blocks(docs: List["Document"]) -> str:
    return "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])

# ----------- Retrieval helpers -----------
def _expand_query_for_med(q: str) -> str:
    """Light synonym expansion to improve recall for common policy terms."""
    ql = q.lower()
    extra = []
    if "pa" in ql or "prior auth" in ql or "authorization" in ql:
        extra += ["prior authorization", "preauthorization", "re-authorization", "renewal"]
    if "cgm" in ql:
        extra += ["continuous glucose monitor", "continuous glucose monitoring"]
    if "appeal" in ql or "griev" in ql:
        extra += ["appeals", "grievance"]
    if "glp" in ql or "weight" in ql or "semag" in ql:
        extra += ["GLP-1", "weight management", "semaglutide", "liraglutide"]
    if not extra:
        return q
    return q + " " + " ".join(sorted(set(extra)))

_STOP = {
    "the","a","an","and","or","of","to","for","in","on","with","by","at","from",
    "is","are","be","as","that","this","it","its","about","please","list","show"
}

def _normalize_tokens(s: str):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2]

def _keyword_overlap_score(question: str, texts: list[str]) -> float:
    """Tiny lexical safety net if the judge fails/parses poorly."""
    q = set(_normalize_tokens(question))
    if not q: return 0.0
    t = set()
    for x in texts: t.update(_normalize_tokens(x))
    return len(q & t) / max(1, len(q))

def _build_sources(docs) -> list[dict]:
    out = []
    for d in docs or []:
        out.append({
            "source": d.metadata.get("source", "unknown"),
            "section": d.metadata.get("section") or d.metadata.get("title") or d.metadata.get("heading") or "Document",
            "preview": (d.page_content or "")[:240]
        })
    return out

def _collect_docs_and_context(region: str, question: str, k: int = 6) -> Tuple[list, str]:
    """Get top-k docs and build a single context string with light section headers."""
    try:
        from app.rag import get_retriever
        ret = get_retriever(region)
        docs = ret.get_relevant_documents(_expand_query_for_med(question))[:k]
    except Exception:
        docs = []
    if not docs:
        return [], ""
    parts = []
    for i, d in enumerate(docs, 1):
        sec = d.metadata.get("section") or d.metadata.get("title") or d.metadata.get("heading") or d.metadata.get("source") or f"Section {i}"
        parts.append(f"[{sec}]\n{d.page_content.strip()}")
    return docs, "\n\n".join(parts)

# ----------- Judge (JSON-only) -----------
def _judge_supported(question: str, context_blocks: str) -> bool:
    """
    Returns True if the question can be answered *directly* from the provided context
    (based on a small JSON decision by the model). If anything fails, bias to True.
    """
    if not context_blocks.strip():
        return False

    judge_system = (
        "You are a strict policy support judge.\n"
        "Decide if the USER QUESTION can be answered *directly and explicitly* from the CONTEXT excerpts.\n"
        "Only consider information present in the CONTEXT; ignore general world knowledge.\n"
        "Reply with strictly valid JSON: {\"supported\": true|false, \"why\": \"<short reason>\"}.\n"
        "Examples:\n"
        "USER: \"List covered CGM codes.\" CONTEXT contains explicit list of codes -> {\"supported\": true, \"why\": \"codes listed\"}\n"
        "USER: \"What is the capital of France?\" CONTEXT about CGM policy -> {\"supported\": false, \"why\": \"topic not in policy\"}\n"
    )
    user_text = f"USER QUESTION:\n{question}\n\nCONTEXT EXCERPTS:\n{context_blocks}"

    try:
        # Use a fresh Bedrock client here to decouple from streaming client init order.
        bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        resp = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            system=[{"text": judge_system}],
            inferenceConfig={"maxTokens": 128, "temperature": 0.0, "topP": 0.9},
            # If the model supports it, you can add: responseFormat={"type": "json"}
        )
        txt = "".join([p.get("text","") for p in resp["output"]["message"]["content"]]).strip()
        data = json.loads(txt)
        return bool(data.get("supported") is True)
    except Exception:
        # If we can't parse JSON or call judge, err on the side of answering (less false negatives)
        return True

# ----------- Common helpers -----------
def _require_valid_turns(turns: List["ChatTurn"]):
    if not turns:
        raise HTTPException(status_code=400, detail={"error": "BadRequest", "message": "messages[] is required"})
    has_user = any(t.role == "user" and (t.content or "").strip() for t in turns)
    if not has_user:
        raise HTTPException(status_code=400, detail={"error": "BadRequest", "message": "at least one non-empty user message is required"})

def _err_dict(exc: Exception):
    if isinstance(exc, ClientError):
        e = exc.response.get("Error", {})
        return {"error": e.get("Code", "ClientError"), "message": e.get("Message", str(exc))}
    return {"error": exc.__class__.__name__, "message": str(exc)}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

def maybe_rag(query: str, k: int = 3) -> Optional[str]:
    try:
        from app.rag import get_retriever
        retriever = get_retriever(AWS_REGION)
        docs = retriever.get_relevant_documents(query)[:k]
        if not docs:
            return None
        return "\n\n".join([d.page_content for d in docs])
    except Exception:
        return None

# ----------- Non-streaming /chat -----------
if USE_LANGCHAIN:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_aws import ChatBedrockConverse
    llm = ChatBedrockConverse(model=MODEL_ID, region_name=AWS_REGION)

    @app.post("/chat")
    def chat(req: ChatRequest):
        _require_valid_turns(req.messages)
        user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

        sources: list[dict] = []
        context_text = ""
        supported = True        # default
        emit_sources = False    # default: don't show unless strict or grounded

        if req.rag and user_last:
            docs, context_text = _collect_docs_and_context(AWS_REGION, user_last, k=req.k or 6)
            sources = _build_sources(docs)

            if req.strict:
                # Strict: require direct support; if unsupported, short-circuit to "Not in policy."
                supported = _judge_supported(user_last, context_text)
                if not supported and _keyword_overlap_score(user_last, [s.get("preview","") for s in sources]) < 0.15:
                    return {"answer": "Not in policy.", "sources": sources}
                emit_sources = True   # show sources in strict mode
            else:
                # Non-strict: show sources only if the context actually supports the Q
                supported = bool(context_text) and _judge_supported(user_last, context_text)
                emit_sources = supported

        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        from langchain_aws import ChatBedrockConverse
        llm = ChatBedrockConverse(model=MODEL_ID, region_name=AWS_REGION)

        lc_msgs = []
        if context_text:
            sys_text = STRICT_TMPL.format(context=context_text) if req.strict else NONSTRICT_TMPL.format(context=context_text)
            lc_msgs.append(SystemMessage(sys_text))
        for m in req.messages:
            if m.role == "user": lc_msgs.append(HumanMessage(m.content))
            elif m.role == "assistant": lc_msgs.append(AIMessage(m.content))
            elif m.role == "system": lc_msgs.append(SystemMessage(m.content))

        try:
            out = llm.invoke(lc_msgs)
            ans = out.content or ""
            return {"answer": ans, "sources": (sources if emit_sources else None)}
        except Exception as e:
            raise HTTPException(status_code=502, detail=_err_dict(e))

else:
    # Optional: non-LangChain boto3 path with the same strict/judge logic
    @app.post("/chat")
    def chat(req: ChatRequest):
        _require_valid_turns(req.messages)
        user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

        bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        sources: list[dict] = []
        context_text = ""
        supported = True
        emit_sources = False

        if req.rag and user_last:
            docs, context_text = _collect_docs_and_context(AWS_REGION, user_last, k=req.k or 6)
            sources = _build_sources(docs)

            if req.strict:
                supported = _judge_supported(user_last, context_text)
                if not supported and _keyword_overlap_score(user_last, [s.get("preview","") for s in sources]) < 0.15:
                    return {"answer": "Not in policy.", "sources": sources}
                emit_sources = True
            else:
                supported = bool(context_text) and _judge_supported(user_last, context_text)
                emit_sources = supported

        messages = []
        if context_text:
            sys_text = STRICT_TMPL.format(context=context_text) if req.strict else NONSTRICT_TMPL.format(context=context_text)
            messages.append({"role": "system", "content": [{"text": sys_text}]})
        messages.append({"role": "user", "content": [{"text": user_last or ''}]})

        try:
            resp = bedrock.converse(
                modelId=MODEL_ID,
                messages=messages,
                inferenceConfig={
                    "maxTokens": req.max_tokens,
                    "temperature": 0.0 if req.strict else req.temperature,
                    "topP": 0.1 if req.strict else req.top_p,
                },
            )
            parts = resp["output"]["message"]["content"]
            ans = "".join([p.get("text","") for p in parts])
            return {"answer": ans, "sources": (sources if emit_sources else None)}
        except Exception as e:
            raise HTTPException(status_code=502, detail=_err_dict(e))

# ----------- Streaming (boto3) -----------
_bedrock_stream = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def _to_bedrock_messages(turns: List[ChatTurn]):
    out = []
    for t in turns:
        if t.role not in ("user","assistant","system"):
            continue
        if t.role == "system":
            continue
        out.append({"role": t.role, "content": [{"text": t.content}]})
    return out

def _gather_system(req: ChatStreamRequest) -> Optional[str]:
    sys_pieces = []
    if req.system:
        sys_pieces.append(req.system)
    sys_pieces += [t.content for t in req.messages if t.role == "system"]
    return "\n\n".join(sys_pieces) if sys_pieces else None

def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

@app.post("/chat/stream")
def chat_stream(req: ChatStreamRequest):
    _require_valid_turns(req.messages)
    user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    sources: list[dict] = []
    context_text = ""
    supported = True
    emit_sources = False

    if req.rag and user_last:
        docs, context_text = _collect_docs_and_context(AWS_REGION, user_last, k=req.k or 6)
        sources = _build_sources(docs)

        if req.strict:
            supported = _judge_supported(user_last, context_text)
            emit_sources = True
            # Early bail if strict and unsupported
            if not supported and _keyword_overlap_score(user_last, [s.get("preview","") for s in sources]) < 0.15:
                def gen_bail():
                    if emit_sources and sources:
                        yield _sse({"sources": sources})
                    yield _sse({"delta": "Not in policy."})
                    yield _sse({"done": True})
                headers = {"Cache-Control": "no-cache","X-Accel-Buffering": "no","Connection": "keep-alive"}
                return StreamingResponse(gen_bail(), media_type="text/event-stream", headers=headers)
        else:
            supported = bool(context_text) and _judge_supported(user_last, context_text)
            emit_sources = supported

    def gen() -> Generator[bytes, None, None]:
        try:
            if emit_sources and sources:
                yield _sse({"sources": sources})

            # STRICT gate: short-circuit when context doesn't directly support
            if req.rag and req.strict:
                if not _judge_supported(user_last, context_text):
                    if _keyword_overlap_score(user_last, [s.get("preview","") for s in sources]) < 0.15:
                        yield _sse({"delta": "Not in policy."})
                        yield _sse({"done": True})
                        return

            kwargs = {
                "modelId": MODEL_ID,
                "messages": [],
                "inferenceConfig": {
                    "maxTokens": req.max_tokens,
                    "temperature": 0.0 if req.strict else req.temperature,
                    "topP": 0.1 if req.strict else req.top_p,
                },
            }
            # Add system with context if any
            if context_text:
                sys_text = STRICT_TMPL.format(context=context_text) if req.strict else NONSTRICT_TMPL.format(context=context_text)
                kwargs["system"] = [{"text": sys_text}]

            # Send only the latest user question
            kwargs["messages"].append({"role": "user", "content": [{"text": user_last}]})

            resp = _bedrock_stream.converse_stream(**kwargs)
            stream = resp.get("stream")
            if stream is None:
                yield _sse({"error": {"type": "NoStream", "message": "Bedrock returned no stream"}})
                return

            for event in stream:
                if "contentBlockDelta" in event:
                    t = event["contentBlockDelta"]["delta"].get("text")
                    if t:
                        yield _sse({"delta": t})
                elif "messageStop" in event:
                    usage = event["messageStop"].get("metadata", {}).get("usage")
                    yield _sse({"done": True, "usage": usage})
                elif "internalServerException" in event:
                    yield _sse({"error": {"type": "InternalServerException", "message": "Model internal error"}}); break
                elif "throttlingException" in event:
                    yield _sse({"error": {"type": "ThrottlingException", "message": "Throttled by service"}}); break
                elif "validationException" in event:
                    yield _sse({"error": {"type": "ValidationException", "message": "Request validation failed"}}); break
                elif "modelStreamErrorException" in event:
                    yield _sse({"error": {"type": "ModelStreamErrorException", "message": "Model stream error"}}); break
        except Exception as e:
            yield _sse(_err_dict(e))

    headers = {"Cache-Control": "no-cache","X-Accel-Buffering": "no","Connection": "keep-alive"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# ----------- RAG admin -----------
@app.get("/rag/status")
def rag_status():
    try:
        from app.rag import get_status
        return {"ok": True, "status": get_status()}
    except Exception as e:
        return {"ok": False, "error": _err_dict(e)}

@app.post("/rag/reindex")
def rag_reindex(
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    x_rag_token: Optional[str] = Header(default=None, alias="X-RAG-Token"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    # If RAG_TOKEN is set, require it via X-RAG-Token or Authorization: Bearer <token>
    if RAG_TOKEN:
        supplied = (x_rag_token or "") or ((authorization or "").replace("Bearer ", "").strip())
        if supplied != RAG_TOKEN:
            raise HTTPException(status_code=403, detail={"error": "Forbidden", "message": "invalid RAG token"})

    b = bucket or RAG_S3_BUCKET
    p = prefix or RAG_S3_PREFIX
    if not b:
        raise HTTPException(status_code=400, detail={"error": "BadRequest", "message": "RAG_S3_BUCKET not configured"})

    try:
        from app.rag import rebuild_from_s3, get_status
        rebuild_from_s3(AWS_REGION, b, p)
        return {"ok": True, "bucket": b, "prefix": p, "status": get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_err_dict(e))

@app.get("/rag/search")
def rag_search(q: str, k: int = 3):
    try:
        from app.rag import search as rag_search_impl
        return {"results": rag_search_impl(AWS_REGION, q, k)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_err_dict(e))

@app.get("/")
def root():
    return {"ok": True, "model": MODEL_ID, "region": AWS_REGION}