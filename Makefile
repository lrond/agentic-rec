PYTHON ?= python3
PYTHONPATH_VALUE ?= src
DEVICE ?= auto
CONFIG ?= configs/default.json
DATA_VARIANT ?= small
NEGATIVES_PER_POSITIVE ?= 4
MAX_TRAIN_BEHAVIORS ?= 0
MAX_DEV_BEHAVIORS ?= 0
RANKER_EPOCHS ?= 10
WORLD_MODEL_EPOCHS ?= 10
EMBEDDING_BACKEND ?= qwen3
EMBEDDING_MODEL ?= Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DEVICE ?= auto
HIDDEN_DIM ?= 256
BATCH_SIZE ?= 256
N_STEPS ?= 3
WORLD_MODEL_N_STEPS ?= 3
TOP_K ?= 5
MAX_DEV_IMPRESSIONS ?= 2000
HF_OFFLINE ?= 0

.PHONY: test data train eval-all eval-ranker eval-wm-multi eval-planner eval

test:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m unittest discover -s tests -p "test_*.py"

data:
	@if [ "$(HF_OFFLINE)" = "1" ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi; \
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/prepare_mind.py \
		--raw-root data/raw/mind \
		--output-root data/processed \
		--config $(CONFIG) \
		--variant $(DATA_VARIANT) \
		--negatives-per-positive $(NEGATIVES_PER_POSITIVE) \
		--max-train-behaviors $(MAX_TRAIN_BEHAVIORS) \
		--max-dev-behaviors $(MAX_DEV_BEHAVIORS) \
		--embedding-backend $(EMBEDDING_BACKEND) \
		--embedding-model $(EMBEDDING_MODEL) \
		--embedding-device $(EMBEDDING_DEVICE) \
		--world-model-n-steps $(N_STEPS)

train: _train-ranker _train-world-model

_train-ranker:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/train_ranker.py \
		--train-data data/processed/ranker_train.jsonl \
		--vectors-path data/processed/news_vectors_train.npz \
		--save-path artifacts/ranker.pt \
		--epochs $(RANKER_EPOCHS) \
		--hidden-dim $(HIDDEN_DIM) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE)

_train-world-model:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/train_world_model.py \
		--train-data data/processed/world_model_train.jsonl \
		--vectors-path data/processed/news_vectors_train.npz \
		--save-path artifacts/world_model.pt \
		--epochs $(WORLD_MODEL_EPOCHS) \
		--model-type neural_ode \
		--ode-hidden 512 \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE) \
		--n-steps $(N_STEPS)

# ── Eval targets ──

# Full comparison: ranker + identity baseline + world_model(drift) + planner
eval-all:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/eval.py \
		--ranker-ckpt artifacts/ranker.pt \
		--ranker-data data/processed/ranker_dev.jsonl \
		--world-model-ckpt artifacts/world_model.pt \
		--world-model-data data/processed/world_model_dev.jsonl \
		--vectors-path data/processed/news_vectors_dev.npz \
		--device $(DEVICE) \
		--planner \
		--identity-baseline \
		--top-k $(TOP_K) \
		--max-dev-impressions $(MAX_DEV_IMPRESSIONS) \
		--world-model-n-steps $(WORLD_MODEL_N_STEPS) \
		--output artifacts/eval_metrics.json

# Ranker only: classification + intent coverage on ranker top-K
eval-ranker:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/eval.py \
		--ranker-ckpt artifacts/ranker.pt \
		--ranker-data data/processed/ranker_dev.jsonl \
		--vectors-path data/processed/news_vectors_dev.npz \
		--device $(DEVICE) \
		--top-k $(TOP_K) \
		--max-dev-impressions $(MAX_DEV_IMPRESSIONS) \
		--output artifacts/eval_metrics.json

# World Model: multi-step with drift metrics
eval-wm-multi:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/eval.py \
		--world-model-ckpt artifacts/world_model.pt \
		--world-model-data data/processed/world_model_dev.jsonl \
		--vectors-path data/processed/news_vectors_dev.npz \
		--device $(DEVICE) \
		--world-model-n-steps $(WORLD_MODEL_N_STEPS) \
		--identity-baseline \
		--output artifacts/eval_metrics.json

# Planner only: beam search + coverage reranker
eval-planner:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/eval.py \
		--ranker-ckpt artifacts/ranker.pt \
		--ranker-data data/processed/ranker_dev.jsonl \
		--world-model-ckpt artifacts/world_model.pt \
		--news-features data/processed/news_features_dev.jsonl \
		--vectors-path data/processed/news_vectors_dev.npz \
		--device $(DEVICE) \
		--planner \
		--top-k $(TOP_K) \
		--max-impressions $(MAX_DEV_IMPRESSIONS) \
		--output artifacts/eval_metrics.json

# Default eval = full comparison
eval: eval-all
