# app/main.py
import os, json
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
RAG_S3_BUCKET = os.getenv("RAG_S3_BUCKET")             # e.g. eks-chat-rag-...-us-east-1
RAG_S3_PREFIX = os.getenv("RAG_S3_PREFIX", "docs/")    # e.g. "docs/"
RAG_TOKEN     = os.getenv("RAG_TOKEN", "").strip()     # shared secret for /rag/reindex (optional; if empty, open)

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

class ChatStreamRequest(ChatRequest):
    system: Optional[str] = None  # additional system prompt

# ----------- Helpers -----------
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

# ----------- Non-streaming -----------
if USE_LANGCHAIN:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_aws import ChatBedrockConverse
    llm = ChatBedrockConverse(model=MODEL_ID, region_name=AWS_REGION)

    @app.post("/chat")
    def chat(req: ChatRequest):
        _require_valid_turns(req.messages)
        lc_msgs = []
        for m in req.messages:
            if m.role == "user": lc_msgs.append(HumanMessage(m.content))
            elif m.role == "assistant": lc_msgs.append(AIMessage(m.content))
            elif m.role == "system": lc_msgs.append(SystemMessage(m.content))

        if req.rag:
            user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            context = maybe_rag(user_last, req.k or 3)
            if context:
                lc_msgs.insert(0, SystemMessage(
                    "Use the following context to answer. If not helpful, answer normally.\n\n" + context
                ))
        try:
            out = llm.invoke(lc_msgs)
            return {"answer": out.content}
        except Exception as e:
            raise HTTPException(status_code=502, detail=_err_dict(e))
else:
    import boto3
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    @app.post("/chat")
    def chat(req: ChatRequest):
        _require_valid_turns(req.messages)
        text = "\n".join([f"{m.role.upper()}: {m.content}" for m in req.messages])

        if req.rag:
            user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            context = maybe_rag(user_last, req.k or 3)
            if context:
                text = f"CONTEXT:\n{context}\n\nCHAT:\n{text}"
        try:
            resp = bedrock.converse(
                modelId=MODEL_ID,
                messages=[{"role":"user","content":[{"text": text}]}],
                inferenceConfig={"maxTokens": req.max_tokens, "temperature": req.temperature, "topP": req.top_p},
            )
            parts = resp["output"]["message"]["content"]
            answer = "".join([p.get("text","") for p in parts])
            return {"answer": answer}
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
        if t.role == "system":  # moved separately
            continue
        out.append({"role": t.role, "content": [{"text": t.content}]})
    return out

def _gather_system(req: ChatStreamRequest) -> Optional[str]:
    sys_pieces = [t.content for t in req.messages if t.role == "system"]
    if req.system:
        sys_pieces.insert(0, req.system)
    if req.rag:
        user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        ctx = maybe_rag(user_last, req.k or 3)
        if ctx:
            sys_pieces.insert(0, "Use the following context to answer. If not helpful, answer normally.\n\n" + ctx)
    if not sys_pieces:
        return None
    return "\n\n".join(sys_pieces)

def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

@app.post("/chat/stream")
def chat_stream(req: ChatStreamRequest):
    _require_valid_turns(req.messages)
    messages = _to_bedrock_messages(req.messages)
    sys_text = _gather_system(req)

    def gen() -> Generator[bytes, None, None]:
        try:
            kwargs = {
                "modelId": MODEL_ID,
                "messages": messages,
                "inferenceConfig": {
                    "maxTokens": req.max_tokens,
                    "temperature": req.temperature,
                    "topP": req.top_p,
                },
            }
            if sys_text:
                kwargs["system"] = [{"text": sys_text}]  # only include when present

            resp = _bedrock_stream.converse_stream(**kwargs)
            stream = resp.get("stream")
            if stream is None:
                yield _sse({"error": {"type": "NoStream", "message": "Bedrock returned no stream"}})
                return

            for event in stream:
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"]["delta"].get("text")
                    if text:
                        yield _sse({"delta": text})
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

            yield _sse({"done": True})
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