# app/main.py
import os, json, re
from typing import List, Optional, Generator
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from botocore.exceptions import ClientError

# ----------- Config -----------
USE_LANGCHAIN = os.getenv("USE_LANGCHAIN", "true").lower() == "true"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID   = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

# RAG config
RAG_S3_BUCKET = os.getenv("RAG_S3_BUCKET")
RAG_S3_PREFIX = os.getenv("RAG_S3_PREFIX", "docs/")
RAG_TOKEN     = os.getenv("RAG_TOKEN", "").strip()

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

STRICT_TMPL = """You are a medical policy assistant.

<CONTEXT>
{context}
</CONTEXT>

RULES (STRICT):
1) Use ONLY facts from CONTEXT. Do not use outside knowledge.
2) If the answer is not directly present, reply exactly: Not in policy.
3) After each sentence, include bracketed citation(s) to the supporting chunk IDs, e.g. [1] or [1][2].
4) Be concise and policy‑style.
"""

NONSTRICT_TMPL = """Use the following context if helpful. If not helpful, answer normally.

<CONTEXT>
{context}
</CONTEXT>
"""

def _format_context_blocks(docs: List["Document"]) -> str:
    return "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])

def _retrieve_with_sources(query: str, k: int, strict: bool):
    # Lazy import to avoid hard dependency on startup
    from app.rag import get_retriever
    ret = get_retriever(AWS_REGION)
    docs = ret.get_relevant_documents(query)[: (k or 3)]
    sources = [{
        "source": d.metadata.get("source", "?"),
        "section": d.metadata.get("section"),
        "preview": d.page_content[:240],
    } for d in docs]

    if strict:
        # very simple lexical overlap gate to avoid OOD answers
        kws = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", query)}
        overlap = any(any(kw in d.page_content.lower() for kw in kws) for d in docs)
        if not overlap:
            return [], sources
    return docs, sources

# ----------- Helpers -----------
def _expand_query_for_med(q: str) -> str:
    """Light synonym expansion to improve recall."""
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

# ----------- RAG helper -----------

def _retrieve_context(query: str, k: int) -> tuple[str, list]:
    """Return (joined_context_text, docs[]) using section-aware retriever."""
    try:
        from app.rag import get_retriever
        q = _expand_query_for_med(query)
        ret = get_retriever(AWS_REGION)
        docs = ret.get_relevant_documents(q)[: (k or 3)]
        ctx = "\n\n".join(d.page_content for d in docs) if docs else ""
        return ctx, docs
    except Exception:
        return "", []
    
# ----------- Non-streaming paths -----------

if USE_LANGCHAIN:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_aws import ChatBedrockConverse
    llm = ChatBedrockConverse(model=MODEL_ID, region_name=AWS_REGION)

    @app.post("/chat")
    def chat(req: ChatRequest):
        _require_valid_turns(req.messages)

        user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        docs, sources = ([], [])
        sys_text = None

        if req.rag and user_last:
            docs, sources = _retrieve_with_sources(user_last, req.k or 3, req.strict)
            if req.strict and not docs:
                return {"answer": "Not in policy.", "sources": sources}

            if docs:
                context = _format_context_blocks(docs)
                sys_text = STRICT_TMPL.format(context=context) if req.strict else NONSTRICT_TMPL.format(context=context)

        if USE_LANGCHAIN:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            from langchain_aws import ChatBedrockConverse

            llm = ChatBedrockConverse(model=MODEL_ID, region_name=AWS_REGION)

            lc_msgs = []
            if sys_text:
                lc_msgs.append(SystemMessage(sys_text))
            for m in req.messages:
                if m.role == "user": lc_msgs.append(HumanMessage(m.content))
                elif m.role == "assistant": lc_msgs.append(AIMessage(m.content))
                elif m.role == "system": lc_msgs.append(SystemMessage(m.content))

            try:
                out = llm.invoke(lc_msgs)
                ans = out.content or ""
                # optional post-check: must include a citation if strict
                if req.strict and docs and "[" not in ans:
                    ans = "Not in policy."
                return {"answer": ans, "sources": sources}
            except Exception as e:
                raise HTTPException(status_code=502, detail=_err_dict(e))
        else:
            import boto3
            bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

            # Collapse history for simplicity
            text = "\n".join([f"{m.role.upper()}: {m.content}" for m in req.messages])
            messages = []
            if sys_text:
                messages.append({"role": "system", "content": [{"text": sys_text}]})
            messages.append({"role": "user", "content": [{"text": text}]})

            try:
                inference = {
                    "maxTokens": req.max_tokens,
                    "temperature": 0.0 if req.strict else req.temperature,
                    "topP": 0.1 if req.strict else req.top_p,
                }
                resp = bedrock.converse(modelId=MODEL_ID, messages=messages, inferenceConfig=inference)
                parts = resp["output"]["message"]["content"]
                ans = "".join([p.get("text","") for p in parts])
                if req.strict and (not docs or "[" not in ans):
                    ans = "Not in policy."
                return {"answer": ans, "sources": sources}
            except Exception as e:
                raise HTTPException(status_code=502, detail=_err_dict(e))
        
# ----------- Streaming (boto3) -----------

import boto3
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
    docs, sources = ([], [])
    sys_text = None

    if req.rag and user_last:
        docs, sources = _retrieve_with_sources(user_last, req.k or 3, req.strict)
        # send sources immediately so UI can render "Sources"
        if sources:
            yield_first = True
        else:
            yield_first = False
        # early bail if strict and nothing relevant
        if req.strict and not docs:
            def gen_bail():
                if yield_first:
                    yield _sse({"sources": sources})
                yield _sse({"delta": "Not in policy."})
                yield _sse({"done": True})
            return StreamingResponse(gen_bail(), media_type="text/event-stream", headers={
                "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive",
            })
        if docs:
            context = _format_context_blocks(docs)
            sys_text = STRICT_TMPL.format(context=context) if req.strict else NONSTRICT_TMPL.format(context=context)

    def gen() -> Generator[bytes, None, None]:
        try:
            if sources:
                yield _sse({"sources": sources})

            kwargs = {
                "modelId": MODEL_ID,
                "messages": [],
                "inferenceConfig": {
                    "maxTokens": req.max_tokens,
                    "temperature": 0.0 if req.strict else req.temperature,
                    "topP": 0.1 if req.strict else req.top_p,
                },
            }
            if sys_text:
                kwargs["system"] = [{"text": sys_text}]
            # collapse history
            text = "\n".join([f"{m.role.upper()}: {m.content}" for m in req.messages])
            kwargs["messages"].append({"role": "user", "content": [{"text": text}]})

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
        from rag import search as rag_search_impl
        return {"results": rag_search_impl(AWS_REGION, q, k)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_err_dict(e))

@app.get("/")
def root():
    return {"ok": True, "model": MODEL_ID, "region": AWS_REGION}