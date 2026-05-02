# Intent-Drift News Planner

> 基于 Neural ODE 的新闻推荐兴趣漂移建模系统

<br>

## 概述

传统推荐系统追求点击率最大化，容易导致"信息茧房"——用户反复看到同类内容。本项目提出 **Intent-Drift News Planner**，将推荐视为动态演化过程：不只预测用户下一秒想点什么，更预测推荐行为对用户兴趣状态的长期影响。

```
用户当前兴趣 u(t)  →  推荐新闻 a(t)  →  Neural ODE 预测漂移  →  未来兴趣 u(t+1)
```

通过多步前瞻搜索，在即时相关性和长期信息覆盖之间找到平衡——既命中偏好，又拓宽视野。

---

### Ranker 与 Planner：分工与协作

想象一个图书管理员和一个编辑的对话。

| 角色 | 组件 | 做什么 | 关心什么 |
|------|------|--------|----------|
| 📚 图书管理员 | **Ranker** | 快速筛出几百本"你可能会喜欢的" | 你和这本书有多匹配？（**当下**） |
| ✍️ 编辑 | **Planner** | 推演你的阅读轨迹，排出一份多元书单 | 读了这本之后你会变成什么样？（**未来**） |

> 编辑用 Neural ODE（兴趣预测引擎）向前模拟几步——像下棋一样推演你的阅读轨迹。最终选出的 5 本书里，既有你确定会喜欢的，也有你没想到但读了会拓宽视野的。

这就是 Coverage@5 从 **66% → 96%** 的含义：管理员单打独斗时，Top-5 覆盖了约三分之二的兴趣领域。编辑加入后，几乎每份书单都兼顾多个方向——且命中率保持在 44.5%，说明没有为了多样性随便塞书。

---

## 系统架构

```
                        behaviors.tsv
                       （用户行为日志）
                             │
                             ▼
               ┌─────────────────────────┐
               │     Qwen3-Embedding      │
               │    frozen · 256 维输出    │
               │   "每篇新闻 → 意义指纹"    │
               └────────────┬────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │       GRU Ranker         │
               │   hidden=256 · 50 条历史  │
               │   "这篇你有多大可能点？"    │
               └────────────┬────────────┘
                            │  几百个候选
                            ▼
          ┌─────────────────────────────────────┐
          │             Planner                  │
          │                                     │
          │   ┌── Beam Search ──────────────┐   │
          │   │  5 条路径并行 · horizon=3    │   │
          │   │  每步展开 8 个候选分支        │   │
          │   └──────────┬──────────────────┘   │
          │              │  "选了这篇，           │
          │              │   用户会怎样？"         │
          │              ▼                      │
          │   ┌── Neural ODE ───────────────┐   │
          │   │  MLP(512) · ReLU · RK4      │   │
          │   │  du/dt = MLP(u, a)          │   │
          │   │  MSE 1.5e-5 · Cos 99.7%     │   │
          │   └──────────┬──────────────────┘   │
          │              │  预测的未来状态        │
          │              ▼                      │
          │   ┌── Coverage Reranker ────────┐   │
          │   │  贪心去重                     │   │
          │   │  "已有体育？优先选科技"         │   │
          │   └─────────────────────────────┘   │
          └─────────────────┬───────────────────┘
                            │
                            ▼
                    ┌─────────────┐
                    │    Top-5    │
                    │  Coverage   │
                    │ 66% → 96%   │
                    └─────────────┘
```

| 组件 | 职责 | 比喻 |
|------|------|------|
| Ranker | 海量候选中**筛选** | 图书管理员 |
| Neural ODE | 预测**状态变化** | 兴趣预测引擎 |
| Beam Search + Coverage | 多步**前瞻** + 类别**去重** | 编辑排版 |

---

### 一次推荐请求的完整旅程

```
① 编码（离线完成，仅一次）
   新闻标题 → Qwen3 → 256 维向量 → .npz

② 用户画像
   你最近的 50 篇 → 查表取向量 → 加权平均 → 你的兴趣状态 u

③ Ranker 海选
   u + 候选 → GRU 打分 → 取前几百

④ Planner 精排（3 轮 × 5 条并行路径）

   第 1 轮                              第 2 轮                              第 3 轮
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │ 8 个第一选择      │    │ 5 条路径 × 8     │    │ 5 条路径 × 8     │
   │ ↓ Neural ODE     │ →  │ ↓ 再次预测       │ →  │ ↓ 最终预测       │
   │ 保留 5 条路径     │    │ 40 → 保留 5       │    │ 选出 1 条最优     │
   └──────────────────┘    └──────────────────┘    └──────────────────┘

⑤ Coverage Reranker
   贪心去重："已有科技和体育 → 优先财经"

⑥ Top-5 ✅
```

| 参数 | 值 |
|------|-----|
| beam_width（并行路径） | 5 |
| branching_factor（每轮展开） | 8 |
| horizon（前瞻轮数） | 3 |
| ODE 调用次数 | ~120 次 |

---

## 数据集

MIND 新闻推荐数据集（PaddleRec 镜像），约 96,000 篇新闻，百万级用户行为日志。

| | 训练集 | 验证集 |
|------|------|------|
| 原始 behavior 数 | 156,965 | 73,025 |
| 实际使用样本 | 500K（随机采样, seed=42） | 5K impressions（完整候选保留） |
| 负采样 | 每条正例最多 4 条负例 | 保留全部负例 |
| 存储 | JSONL（ID）+ .npz（72MB 嵌入矩阵） | 同左 |

> 验证集采用 impression 级采样——保留每个曝光的完整候选池，确保 Recall@5、NDCG@5 等分组指标有效计算。

详见 [`DATA.md`](DATA.md)。

---

## 实验结果

### Ranker 与 Planner

| 指标 | Ranker-only | + Planner | 变化 |
|------|:---------:|:--------:|:----:|
| Accuracy | 94.48% | — | — |
| AUC | 0.701 | — | — |
| Recall@5 | 50.38% | — | — |
| Coverage@5 | 66.35% | **95.83%** | ↑ 44% |
| ILS@5（越低越多样） | 0.578 | 0.527 | ↓ 9% |
| HitRate@5 | — | 44.50% | — |

### World Model

| 指标 | 值 |
|------|------|
| Neural ODE MSE | 1.52 × 10⁻⁵ |
| Cosine Similarity | 99.71% |

---

## 快速开始

```bash
pip install -e '.[train]'

# 数据预处理
make data

# 训练（Ranker + Neural ODE）
make train

# 评测
make eval
```

**环境**：Python ≥ 3.11 · PyTorch ≥ 2.3 · Qwen3-Embedding-0.6B · NVIDIA RTX 5090

---

## 项目结构

```
.
├── configs/default.json        训练配置
├── DATA.md                     数据集详细说明
├── Makefile                    构建入口
├── README.md
│
├── data/
│   ├── raw/mind/               MIND 原始 TSV（PaddleRec 镜像）
│   └── processed/              Qwen3 编码后的 .npz 嵌入矩阵
│
├── artifacts/
│   └── eval_metrics.json       最终评测结果
│
├── report/
│   ├── main.tex                LaTeX 学术报告
│   └── main.pdf
│
├── slides/
│   └── index.html              HTML 演示幻灯片
│
├── scripts/
│   ├── prepare_mind.py         数据预处理
│   ├── train_ranker.py         Ranker 训练
│   ├── train_world_model.py    世界模型训练（Neural ODE）
│   └── eval.py                 评测
│
├── src/agentic_rec/
│   ├── data/                   数据加载
│   ├── trainers/               Ranker & 世界模型训练逻辑
│   ├── world_model/            Neural ODE（MLP + RK4）
│   ├── planner/                Beam Search + Coverage Reranker
│   ├── eval/                   评测指标
│   └── export.py               checkpoint 导出
│
└── tests/                      单元测试
```

---

## 参考文献

1. Wu et al. *MIND: A Large-scale Dataset for News Recommendation.* ACL 2020.
2. Qwen Team. *Qwen3 Embedding: Technical Report.* arXiv:2506.05176, 2025.
3. Wang et al. *Towards Interest Drift-driven User Representation Learning.* SIGIR 2025.
4. Wang et al. *Beyond Item Dissimilarities: Intent-level Diversity in Recommendation.* KDD 2025.
