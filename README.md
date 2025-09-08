# EKS Chat (Bedrock) — Streaming + RAG Demo

A tiny streaming chat app that runs on **AWS EKS**, serves a static **web UI on S3**, calls **Amazon Bedrock** (Claude 3 Haiku), and optionally performs **RAG** using **Titan Embeddings v2** + **FAISS**. It’s designed to be easy to demo and cheap to run.

> **Demo intent only.** Not medical advice. Do not put PHI in the system.

---

## 👩‍⚕️ 2‑Minute Tour (for non‑technical reviewers)

1. **Open** the web app (S3 website URL).
2. In the header chip, confirm the **API host** (ALB URL of the backend). Click **“set API”** if you need to change it.
3. In the toolbar:
   - **Temperature** — lower = more deterministic; higher = more creative.
   - **RAG** — when ON, the bot can use the internal **policy knowledge base**.
   - **Strict** — when ON, answers must come **only** from the policy context; otherwise the bot replies **“Not in policy.”**
4. (Optional) Add a **System prompt** to change tone/role.
5. Ask a question:
   - **Policy example**: “List covered CGM codes.” → expect a grounded answer + **Sources**.
   - **Out‑of‑policy example**: “What is the capital of France?”  
     - **Strict ON** → **Not in policy.**  
     - **Strict OFF** → “Paris.”
6. Watch tokens stream in real‑time. If RAG is on, a **Sources** box shows which chunks supported the answer.

---

## Features

- ⚡ **Streaming responses (SSE)** — tokens arrive as the model generates them.
- 🧠 **Retrieval‑Augmented Generation (RAG)** — Titan Embeddings v2 + FAISS; index can be rebuilt from S3.
- 🧾 **Sources / Evidence** — the UI shows which chunks supported the answer.
- 🧱 **Strict Mode** — when enabled, answers must be grounded in retrieved policy context; otherwise the bot replies **“Not in policy.”**
- 🧩 **System prompt & Temperature** — tweak behavior and creativity (saved in localStorage).
- ☁️ **One‑command deploy** — Makefile + Helm for EKS, ALB, ECR, and S3 website.
- 🔐 **Admin endpoint** — `/rag/reindex` protected by a shared token (`RAG_TOKEN`).

---

## Architecture

The diagram below is also included as `architecture.svg` in the repo.

![Architecture](./architecture.svg)

**High‑level:**

- **Frontend**: S3 static website (`web/`) calling the API
- **Backend**: FastAPI container on EKS (streaming SSE; `/chat` and `/chat/stream`)
- **Models**: Bedrock (Claude 3 Haiku), Titan Embeddings v2
- **RAG**: FAISS index persisted under `/app/store` (PVC), able to rebuild from S3
- **Auth**: `/rag/reindex` requires `X-RAG-Token`

---

## Prerequisites

- AWS account + CLI configured
- Tools: `docker`, `kubectl`, `helm`, `eksctl`, `jq`, `make`
- **Bedrock access** in your region (us‑east‑1) for:
  - `anthropic.claude-3-haiku-20240307-v1:0`
  - `amazon.titan-embed-text-v2:0`

---

## Quick Start (Local)

1. **Install deps**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run API**
   ```bash
   export AWS_REGION=us-east-1
   export MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
   export USE_LANGCHAIN=false     # optional; forces raw boto3 path for /chat
   uvicorn app.main:app --reload --port 8000
   ```

3. **Open UI**
   - Create/edit `web/config.js`:
     ```js
     window.API_BASE_URL = 'http://localhost:8000';
     ```
   - Open `web/index.html` in your browser (or use `?api=http://localhost:8000`).

---

## Deploy to EKS

**Create cluster + bootstrap IAM (one‑time):**
```bash
make cluster-up           # EKS cluster + OIDC + bedrock invoke policy + SA
```

**Build & push** the image to ECR:
```bash
TAG=0.5 make build-push   # or TAG=$(git rev-parse --short HEAD)
```

**Create RAG bucket** and seed demo content:
```bash
make rag-bucket           # creates s3://<cluster>-rag-<acct>-<region>/docs/demo.md
make rag-iam              # creates/attaches S3 read policy to the workload role
```

**Deploy the app** (ALB Service + envs):
```bash
RAG_TOKEN='your-strong-token' TAG=0.5 make deploy
make url   # prints API base URL (ALB hostname)
```

---

## Frontend Website (S3)

Create a public S3 static website (demo‑only approach):
```bash
make frontend-up
# Point the web UI at your API (ALB hostname)
echo "window.API_BASE_URL = 'http://<ALB_HOSTNAME>'" > web/config.js
make frontend-deploy
make frontend-url   # prints the website URL
```
> **Note:** For production, prefer CloudFront + OAI/WAF instead of a public S3 website.

---

## RAG Admin & Data

1. **Upload / update** docs:
   ```bash
   aws s3 sync data/ s3://$RAG_BUCKET/docs/
   ```
2. **Rebuild** the FAISS index in the pod:
   ```bash
   curl -sS -X POST "$API_URL/rag/reindex" -H "X-RAG-Token: $RAG_TOKEN" | jq .
   ```
3. **Verify**:
   ```bash
   curl -sS "$API_URL/rag/status" | jq .
   curl -sS "$API_URL/rag/search?q=CGM%20codes&k=3" | jq .
   ```
4. **Manual tests**:
   - **RAG ON + Strict ON**  
     - “List covered CGM codes” → policy‑grounded answer + **Sources**
     - “What is the capital of France?” → **Not in policy.**
   - **RAG OFF** — chat behaves like a general LLM.

---

## API Examples

### Non‑streaming `/chat`
```bash
curl -sS -X POST "$API_URL/chat" \
  -H 'Content-Type: application/json' \
  -d '{
        "messages": [{"role":"user","content":"List covered CGM codes"}],
        "rag": true,
        "strict": true,
        "temperature": 0.2
      }' | jq
```
**Response** (truncated):
```json
{
  "answer": "According to the policy context provided...",
  "sources": [
    {"source":"s3://.../docs/demo.md","section":"Covered Codes (examples)","preview":"- A4238 ..."}
  ]
}
```

### Streaming `/chat/stream` (SSE)
```bash
curl -N -X POST "$API_URL/chat/stream" \
  -H 'Content-Type: application/json' \
  -d '{
        "messages":[{"role":"user","content":"What is the capital of France?"}],
        "rag": true,
        "strict": true,
        "temperature": 0.2
      }'
```
**SSE frames**:
- Early **`sources`** frame (for the UI)
- Multiple **`delta`** frames as tokens stream
- Final **`done`** frame  
In strict mode, out‑of‑scope questions short‑circuit with **`delta: "Not in policy."`**.

### RAG Admin
```bash
# Reindex from S3
curl -sS -X POST "$API_URL/rag/reindex" -H "X-RAG-Token: $RAG_TOKEN" | jq .

# Status
curl -sS "$API_URL/rag/status" | jq .

# Debug search
curl -sS "$API_URL/rag/search?q=glp-1%20renewal&k=3" | jq .
```

---

## UI Reference

- **RAG** — toggles use of the knowledge base (S3‑backed policy index).
- **Strict** — answers only if supported by retrieved policy context; else **“Not in policy.”**
- **Temperature** — lower (e.g., 0.2) = more deterministic; higher = more creative.
- **System prompt** — optional; sets tone/role (saved locally).
- **Sources** — shows which chunk/section supported the answer.
- **API badge** — shows the backend host; click **“set API”** to change.

---

## Configuration (env)

| Var             | Default                                           | Notes                                                     |
|-----------------|---------------------------------------------------|-----------------------------------------------------------|
| `AWS_REGION`    | `us-east-1`                                       | Must match Bedrock and S3 region                          |
| `MODEL_ID`      | `anthropic.claude-3-haiku-20240307-v1:0`          | Claude 3 Haiku via Bedrock                                |
| `USE_LANGCHAIN` | `true`                                            | `false` to force raw boto3 path for `/chat`               |
| `RAG_S3_BUCKET` | *(none)*                                          | e.g. `eks-chat-rag-<acct>-us-east-1`                      |
| `RAG_S3_PREFIX` | `docs/`                                           | Folder under bucket                                       |
| `RAG_TOKEN`     | *(empty)*                                         | If set, required for `/rag/reindex`                       |

Helm chart sets these with `--set env.*=...`.

---

## Costs & Cleanup

Approx hourly costs (us‑east‑1, estimates):
- EKS control plane: ~**$0.10/h**
- ALB: ~**$0.02+/h**
- 1× t4g.small worker: a few **¢/h**
- S3 website: fractions of a cent for light traffic
- RAG embeddings: **Titan v2** for tiny docs rounds to **$0.00**

**Stop LB but keep cluster**:
```bash
helm uninstall bedrock-chat
```

**Delete everything**:
```bash
make down
```

---

## Troubleshooting

- **`AccessDenied: s3:ListBucket`** — Attach RAG S3 read policy to the workload role:
  ```bash
  make rag-iam
  kubectl rollout restart deploy/bedrock-chat
  ```
- **`ModuleNotFoundError: rag`** — Import via the package path in FastAPI:
  ```python
  from app.rag import get_retriever, rebuild_from_s3, get_status
  ```
- **`404 /rag/search`** — You’re running an older image. Bump the tag, redeploy, and check:
  ```bash
  curl -sS "$API_URL/openapi.json" | jq '.paths | keys'
  ```
- **Streaming differs from `/chat`** — `/chat` can use LangChain while `/chat/stream` uses boto3. To unify behavior:
  ```bash
  --set env.USE_LANGCHAIN="false"
  ```

---

## Security Notes

- Demo enables CORS and a public ALB. Limit exposure if sharing widely (WAF/IP allowlist, CloudFront).
- Keep `RAG_TOKEN` strong; prefer a Kubernetes Secret / AWS Secrets Manager in real deployments.
- Do not ingest PHI; data is stored in your S3 bucket and vector index on the pod PVC.

---

## License

MIT. Demo content only.
