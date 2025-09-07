# ---------- config ----------
REGION ?= us-east-1
CLUSTER ?= eks-chat
ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)
ECR_URI := $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com
REPO := llm-chat
IMAGE := $(ECR_URI)/$(REPO):0.1
MODEL_ID ?= anthropic.claude-3-haiku-20240307-v1:0

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
	docker buildx build --platform linux/arm64 -t $(IMAGE) .
	docker push $(IMAGE)

# ---------- deploy ----------
.PHONY: deploy
deploy:
	helm upgrade --install bedrock-chat charts/bedrock-chat \
	  --set image.repository="$(ECR_URI)/$(REPO)" \
	  --set image.tag="0.1" \
	  --set env.AWS_REGION="$(REGION)" \
	  --set env.MODEL_ID="$(MODEL_ID)" \
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