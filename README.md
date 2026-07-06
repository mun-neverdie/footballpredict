# World Cup 2026 RAG Predictor

基于 RAG（检索增强生成）技术的世界杯淘汰赛预测系统。从 Wikipedia 构建知识库，结合 LLM 进行有依据的比赛预测。

## 架构

```
Wikipedia API ──→ build_kb.py ──→ kb_entries.json ──→ build_index.py ──→ ChromaDB
                      │                │                      │
                  BART 摘要       球队+俱乐部+球员         all-MiniLM-L6-v2
                  赛季章节         赛事小组赛/淘汰赛          384维向量
                      │                                    │
                      └────────────────────────────────────┘
                                           │
                                    predict.py ←── 用户输入 "A vs B"
                                           │
                                    四层 RAG 检索
                                           │
                                      LLM 预测
```

## 四层 RAG 检索

| 层级 | 方式 | 内容 |
|------|------|------|
| 球队画像 | 语义检索 | 战术风格、近期状态、世界杯历史 |
| 球员深度 | 精确 ID | 2FW+2MF+2DF+1GK 多因子评分筛选 |
| 俱乐部背景 | 精确 ID | 球员→俱乐部→联赛竞技水平链 |
| 历史交锋 | 语义检索 | 对阵记录、风格克制分析 |

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 下载模型

```bash
python download_model.py          # embedding 模型
python -c "from transformers import BartForConditionalGeneration, BartTokenizer; BartTokenizer.from_pretrained('facebook/bart-large-cnn'); BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')"
```

### 3. 构建知识库

```bash
python get_cv.py                  # Step 0: 爬取阵容 → squads_2026.json
python build_kb.py                # Step 1: Wikipedia + BART → kb_entries.json
python build_index.py             # Step 2: 向量化 → vector_store/
```

### 4. 预测

```bash
# 设置 API Key
set DEEPSEEK_API_KEY=sk-xxx       # Windows
export DEEPSEEK_API_KEY=sk-xxx    # Linux/Mac

# 预测
python predict.py "Argentina" "France"

# 仅查看检索结果（调试）
python predict.py "Argentina" "France" --dry
```
# 启动前端预测（支持手动输入球队名称、勾选球队名称）
streamlit run streamlit_app.py


## 文件说明

| 文件 | 作用 |
|------|------|
| `get_cv.py` | 从 Wikipedia 爬取 48 队阵容 |
| `build_kb.py` | 爬取 Wikipedia + BART 摘要 + 赛季信息 + 赛事数据 |
| `build_index.py` | 向量化所有知识块，存入 ChromaDB |
| `predict.py` | 四层 RAG 检索 + LLM 预测 |
| `player_selector.py` | 多因子评分球员筛选（2FW+2MF+2DF+1GK） |
| `download_model.py` | 下载 embedding 模型到本地 |
| `squads_2026.json` | 48 队 × 26 人结构化数据 |

## 断点续传

`build_kb.py` 每爬完一条立写 `crawl_cache.json`。中断后直接重跑即自动跳过已完成条目。全量完成后可删除缓存文件。
