# app/main.py

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from fastapi import HTTPException

# Choose either LangChain or raw boto3.
USE_LANGCHAIN = os.getenv("USE_LANGCHAIN", "true").lower() == "true"

app = FastAPI(title="Bedrock Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

class ChatTurn(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatTurn]
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.4
    max_tokens: Optional[int] = 512
    rag: Optional[bool] = False
    k: Optional[int] = 3

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

def maybe_rag(query: str, k: int = 3) -> Optional[str]:
    try:
        from rag import get_retriever  # lazily import so base chat still works
        retriever = get_retriever(AWS_REGION)
        docs = retriever.get_relevant_documents(query)[:k]
        if not docs:
            return None
        return "\n\n".join([d.page_content for d in docs])
    except Exception:
        return None
# ---------------------------------------------------------------------

if USE_LANGCHAIN:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_aws import ChatBedrockConverse

    llm = ChatBedrockConverse(
        model=MODEL_ID,
        region_name=AWS_REGION,
        # You can pass model kwargs here if needed: model_kwargs={"max_tokens": 512}
    )

    @app.post("/chat")
    def chat(req: ChatRequest):
        # Build chat history
        lc_msgs = []
        for m in req.messages:
            if m.role == "user":
                lc_msgs.append(HumanMessage(m.content))
            elif m.role == "assistant":
                lc_msgs.append(AIMessage(m.content))
            elif m.role == "system":
                lc_msgs.append(SystemMessage(m.content))

        # Optional: add retrieved context
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
            raise HTTPException(status_code=500, detail=str(e))

else:
    import boto3

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    @app.post("/chat")
    def chat(req: ChatRequest):
        # Reduce to a single user prompt (for simplicity)
        text = "\n".join([f"{m.role.upper()}: {m.content}" for m in req.messages])

        # Optional: add retrieved context
        if req.rag:
            user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            context = maybe_rag(user_last, req.k or 3)
            if context:
                text = f"CONTEXT:\n{context}\n\nCHAT:\n{text}"

        resp = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": text}]}],
            inferenceConfig={
                "maxTokens": req.max_tokens,
                "temperature": req.temperature,
                "topP": req.top_p,
            },
        )
        parts = resp["output"]["message"]["content"]
        answer = "".join([p.get("text", "") for p in parts])
        return {"answer": answer}
    
@app.get("/")
def root():
    return {"ok": True, "model": MODEL_ID, "region": AWS_REGION}