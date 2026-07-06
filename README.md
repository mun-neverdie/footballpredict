# World Cup 2026 RAG Predictor

# 2026-7-6 比赛：
Brazil 1-2 Norway! 
CORRECT RESULT AND FINAL SCORE! Ours predicted Norway victory and 1-2 results successfully! OUTPERFORMING RAGLESS DEEPSEEK!
<!-- ### 胜负预测
**Norway 胜**。尽管巴西历史战绩辉煌，但本届阵容老化严重，核心球员远离顶级联赛，竞技状态成疑。挪威拥有当今足坛最顶级的终结者哈兰德和创造力核心厄德高，其进攻体系的直接性和高效性足以撕开巴西并不稳固的防线。挪威是少数对巴西保持不败的球队，心理上不落下风，且球员正处于职业生涯巅峰期。

---

### 比分预测
1. 1-2 (理由: 挪威凭借哈兰德的个人能力取得领先，巴西由内马尔或拉菲尼亚扳回一城，但挪威反击再下一城。)
2. 0-2 (理由: 巴西中场失控，无法有效支援锋线，挪威通过厄德高的组织和哈兰德的冲击力完全掌控比赛，零封对手。)
3. 1-1 (理由: 巴西凭借经验在落后时顽强扳平，将比赛拖入加时。加时赛中挪威体能优势显现，最终点球获胜。)
   *注：最可能结果为挪威在常规时间获胜。* -->

England 3-2 Mexico
CORRECT RESULT!
<!-- ### 胜负预测
**England 胜**。英格兰在球员个人能力、俱乐部竞技水平和战术执行力上全面占优，凯恩的终结能力与赖斯的中场控制将主导比赛节奏。墨西哥虽经验丰富但整体实力差距明显，难以抵挡英格兰的持续施压。

---

### 比分预测
1. 2-0 (理由: 英格兰凭借凯恩和拉什福德的进球早早确立优势，墨西哥进攻乏力，奥乔亚虽能扑救但难阻丢球。)
2. 3-1 (理由: 英格兰通过边路突破和中路渗透制造多点开花，墨西哥可能利用定位球或反击由希门尼斯扳回一球。)
3. 1-0 (理由: 墨西哥收缩防守试图拖入加时，但英格兰凭借凯恩的灵光一现打破僵局，最终小胜晋级。) -->

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
