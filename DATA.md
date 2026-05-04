# 数据集说明

## 来源

MIND（Microsoft News Dataset）新闻推荐数据集，由微软于 2019 年发布（Wu et al., ACL 2020）。

## 获取方式

通过 PaddleRec 仓库下载预处理后的 MIND 数据集：

```bash
git clone https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec/datasets/MIND
```

或直接访问：https://github.com/PaddlePaddle/PaddleRec/tree/master/datasets/MIND

下载后得到 `MINDsmall_train/` 和 `MINDsmall_dev/` 两个目录，各包含 `behaviors.tsv` 和 `news.tsv`。

将这两个目录放到 `data/raw/mind/` 下：

```
data/raw/mind/
├── MINDsmall_train/
│   ├── behaviors.tsv
│   └── news.tsv
└── MINDsmall_dev/
    ├── behaviors.tsv
    └── news.tsv
```

然后运行 `make data` 即可完成预处理。

## 数据结构

### news.tsv（新闻特征表）
每行 5 个字段（Tab 分隔）：
| 字段 | 示例 | 含义 |
|------|------|------|
| 新闻 ID | `N71090` | 唯一标识符 |
| 一级类别 | `lifestyle` | 如 sports, finance, lifestyle |
| 二级类别 | `liferoyals` | 更细粒度的子类别 |
| 标题 | `3463 2605 1216 1835 ...` | 分词后的 token ID 序列 |
| 摘要 | `1230 4479 35 13 0 ...` | 同上 |

### behaviors.tsv（用户行为表）
每行是一次用户曝光（impression），5 个字段（Tab 分隔）：
| 字段 | 示例 | 含义 |
|------|------|------|
| Impression ID | `imp_train_00000000` | 曝光事件 ID |
| User ID | `user_0000` | 匿名用户 |
| 时间戳 | — | 曝光时间 |
| 浏览历史 | `N88753 N12345 N67890 ...` | 本次曝光前用户点击过的新闻 ID（空格分隔） |
| 曝光新闻 | `N71090-1 N11223-0 N33445-0` | 候选新闻，`-1`=点击，`-0`=未点击 |

## 样本生成规则

从 behavior 到训练样本的转换：

- **Ranker 样本**：每个 impression → 全部正例 + 每条正例最多配 4 条随机负例 → (历史 ID 列表, 候选新闻 ID, 点击标签, 新闻类别)
- **世界模型样本（单步）**：每个正例 → (点击前加权状态, 点击新闻嵌入, 点击后加权状态)
- **世界模型样本（多步，N=3）**：同一 impression 内取连续 3 个点击 → (初始状态, 3 篇新闻嵌入序列, 3 步后的加权状态)。仅当 impression 含 ≥3 个正例时才生成样本
- **验证集**：保留全部候选（不进行负采样），确保 Recall@5、NDCG@5 等分组指标计算有效

## 数据量统计

| 数据集 | Ranker | World Model（多步 N=3） | 新闻数 |
|------|------|------|------|
| 训练 | 9,495,212 条 | 310,338 条 | 96,074 |
| 验证 | 7,496,306 条 | 46,792 条 | 96,074 |

> 多步世界模型数据量约为单步的 16%（因需要 impression 含 ≥3 个正例）。验证集评估默认采样 2,000 impressions（约 78K ranker 样本）。

## 评估采样

```bash
# 快速评估（默认 2000 impressions）
make eval-all

# 全量评估（192K impressions，完整数据）
make eval-all MAX_DEV_IMPRESSIONS=0

# 自定义采样量
make eval-all MAX_DEV_IMPRESSIONS=5000
```

## 嵌入文件

预处理后生成两个二进制嵌入矩阵（`data/processed/`）：

| 文件 | 大小 | 内容 |
|------|------|------|
| `news_vectors_train.npz` | ~72 MB | 训练集 96,074 篇新闻的 Qwen3-256 维嵌入 |
| `news_vectors_dev.npz` | ~72 MB | 验证集 96,074 篇新闻的 Qwen3-256 维嵌入 |

训练脚本通过 `--vectors-path` 参数加载，运行时根据新闻 ID 查表获取向量。
