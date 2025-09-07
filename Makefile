# Makefile
# ---------- config ----------
SHELL := /bin/bash
TAG ?= 0.1
REGION ?= us-east-1
CLUSTER ?= eks-chat
ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)
ECR_URI := $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com
REPO := llm-chat
IMAGE := $(ECR_URI)/$(REPO):0.1
MODEL_ID ?= anthropic.claude-3-haiku-20240307-v1:0
RAG_BUCKET ?= $(CLUSTER)-rag-$(ACCOUNT_ID)-$(REGION)
RAG_PREFIX ?= docs/
RAG_TOKEN ?= demo-rag-token

# ---------- frontend ----------
FRONTEND_BUCKET ?= $(CLUSTER)-web-$(ACCOUNT_ID)-$(REGION)

.PHONY: frontend-up
frontend-up:
	@echo "Creating S3 bucket $(FRONTEND_BUCKET) in $(REGION)..."
	@if [ "$(REGION)" = "us-east-1" ]; then \
	  aws s3api create-bucket --bucket $(FRONTEND_BUCKET) >/dev/null; \
	else \
	  aws s3api create-bucket --bucket $(FRONTEND_BUCKET) \
	    --create-bucket-configuration LocationConstraint=$(REGION) >/dev/null; \
	fi
	# Allow public website reads (quick demo approach)
	aws s3api put-public-access-block --bucket $(FRONTEND_BUCKET) \
	  --public-access-block-configuration BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false >/dev/null
	@echo "Applying public read policy..."
	@POLICY=$$(jq -n --arg b "$(FRONTEND_BUCKET)" '{Version:"2012-10-17",Statement:[{Sid:"PublicReadGetObject",Effect:"Allow",Principal:"*",Action:["s3:GetObject"],Resource:["arn:aws:s3:::"+$$b+"/*"]}]}'); \
	aws s3api put-bucket-policy --bucket $(FRONTEND_BUCKET) --policy "$$POLICY"
	# Enable website hosting
	aws s3 website s3://$(FRONTEND_BUCKET)/ --index-document index.html --error-document index.html

.PHONY: frontend-deploy
frontend-deploy:
	@echo "Syncing ./web to s3://$(FRONTEND_BUCKET)/ ..."
	aws s3 sync web/ s3://$(FRONTEND_BUCKET)/ --delete --cache-control max-age=60
	@$(MAKE) frontend-url

.PHONY: frontend-url
frontend-url:
ifeq ($(REGION),us-east-1)
	@echo "http://$(FRONTEND_BUCKET).s3-website-us-east-1.amazonaws.com"
else
	@echo "http://$(FRONTEND_BUCKET).s3-website-$(REGION).amazonaws.com"
endif

.PHONY: frontend-down
frontend-down:
	-aws s3 rm s3://$(FRONTEND_BUCKET)/ --recursive
	-aws s3api delete-bucket-policy --bucket $(FRONTEND_BUCKET)
	-aws s3api delete-bucket --bucket $(FRONTEND_BUCKET)

# ---------- RAG bucket ----------
.PHONY: rag-bucket
rag-bucket:
	aws s3 mb s3://$(RAG_BUCKET) --region $(REGION) || true
	aws s3 cp data/demo.md s3://$(RAG_BUCKET)/$(RAG_PREFIX)demo.md

.PHONY: rag-iam
rag-iam:
	@echo 'Creating S3 read policy for $(RAG_BUCKET)/$(RAG_PREFIX)...'
	@printf '%s\n' \
	  '{' \
	  '  "Version": "2012-10-17",' \
	  '  "Statement": [' \
	  '    {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::$(RAG_BUCKET)", "Condition": {"StringLike": {"s3:prefix": ["$(RAG_PREFIX)*"]}}},' \
	  '    {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "arn:aws:s3:::$(RAG_BUCKET)/$(RAG_PREFIX)*"}' \
	  '  ]' \
	  '}' > rag-s3-read-policy.json
	-aws iam create-policy --policy-name rag-s3-read-$(RAG_BUCKET) --policy-document file://rag-s3-read-policy.json >/dev/null
	# attach both policies to the same SA (override to add)
	eksctl create iamserviceaccount \
	  --cluster $(CLUSTER) --region $(REGION) \
	  --namespace default --name bedrock-sa \
	  --attach-policy-arn arn:aws:iam::$(ACCOUNT_ID):policy/bedrock-invoke-policy \
	  --attach-policy-arn arn:aws:iam::$(ACCOUNT_ID):policy/rag-s3-read-$(RAG_BUCKET) \
	  --override-existing-serviceaccounts --approve

# ---------- cluster ----------
.PHONY: cluster-up
cluster-up:
	eksctl create cluster \
	  --name $(CLUSTER) --region $(REGION) \
	  --nodes 1 --node-type t4g.small --managed \
	  --vpc-nat-mode Disable  # <- avoid NAT $ if OK with public nodes

	aws eks update-kubeconfig --region $(REGION) --name $(CLUSTER)
	eksctl utils associate-iam-oidc-provider --cluster $(CLUSTER) --region $(REGION) --approve

	# Bedrock invoke policy (idempotent)
	echo '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["bedrock:Converse","bedrock:ConverseStream","bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],"Resource":"*"}]}' \
	  > bedrock-invoke-policy.json
	-aws iam create-policy --policy-name bedrock-invoke-policy \
	  --policy-document file://bedrock-invoke-policy.json >/dev/null
	eksctl create iamserviceaccount \
	  --cluster $(CLUSTER) --region $(REGION) \
	  --namespace default --name bedrock-sa \
	  --attach-policy-arn arn:aws:iam::$(ACCOUNT_ID):policy/bedrock-invoke-policy \
	  --approve

.PHONY: cluster-down
cluster-down:
	# Remove app so LB is deleted before cluster
	-helm uninstall bedrock-chat || true
	# Optional: delete ECR repo (removes images)
	-aws ecr delete-repository --repository-name $(REPO) --force --region $(REGION) || true
	# Delete the whole cluster + VPC/LB/etc
	eksctl delete cluster --name $(CLUSTER) --region $(REGION)

# ---------- image & repo ----------
.PHONY: build-push
build-push:
	-aws ecr create-repository --repository-name $(REPO) --region $(REGION) >/dev/null
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR_URI)
	docker buildx build --platform linux/arm64 -t $(ECR_URI)/$(REPO):$(TAG) .
	docker push $(ECR_URI)/$(REPO):$(TAG)

# ---------- deploy ----------
.PHONY: deploy
deploy:
	helm upgrade --install bedrock-chat charts/bedrock-chat \
	  --set image.repository="$(ECR_URI)/$(REPO)" \
	--set image.tag="$(TAG)" \
	  --set image.pullPolicy="Always" \
	  --set env.AWS_REGION="$(REGION)" \
	  --set env.MODEL_ID="$(MODEL_ID)" \
	  --set env.RAG_S3_BUCKET="$(RAG_BUCKET)" \
	  --set env.RAG_S3_PREFIX="$(RAG_PREFIX)" \
	  --set env.RAG_TOKEN="$(RAG_TOKEN)" \
	  --set service.type="LoadBalancer"
	kubectl rollout status deploy/bedrock-chat

.PHONY: url
url:
	@echo http://$$(kubectl get svc bedrock-chat -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# ---------- everything ----------
.PHONY: up
up: cluster-up build-push deploy url

.PHONY: down
down: cluster-down