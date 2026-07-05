"""
predict.py — 四层 RAG 增强的世界杯淘汰赛预测

用法:
  python predict.py "Argentina" "France"        # 预测
  python predict.py "Argentina" "France" --dry  # 仅查看检索结果，不调 LLM

======================================================================
RAG 必要性说明
======================================================================

结构化数据（squads_2026.json）能告诉 LLM：
  "Messi, FW, 37岁, Inter Miami, 出场128, 进球79"
但它不能告诉 LLM：
  "37岁的 Messi 在 MLS 的竞技强度远低于欧冠，但他的大赛经验弥补了运动能力的下滑。"

RAG 的四层检索正是为了解决这些"结构化数据之外"的问题：

  Layer 1 — 球队画像（语义检索）
    问题: Argentina 是什么风格的球队？France 的战术弱点是什么？
    检索: "{team} national team tactical style form strengths weaknesses"
    价值: 战术分析、风格克制判断

  Layer 2 — 关键球员深度（精确 ID 检索）
    问题: 这些球员除了数字之外，有什么故事？
    检索: "player:{name}" — 从 BART 摘要中获取球员职业生涯、技术特点
    价值: 球员评价不再是"进球多"而是"为什么进球多"

  Layer 3 — 俱乐部竞技背景（精确 ID 检索）
    问题: 球员在俱乐部面对什么水平的对抗？
    检索: "club:{club}" — 球员 → 俱乐部 → 俱乐部摘要
    价值: Messi 在 MLS vs Mbappé 在欧冠 — 比赛强度差异是关键变量

  Layer 4 — 历史交锋 & 风格碰撞（语义检索）
    问题: 这两支队踢过吗？风格克制吗？
    检索: "{team_a} vs {team_b} head to head history"
    价值: 心理优势、战术克制历史

四层检索结果 + 结构化数据 → LLM → 有依据的预测
"""

import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import json
import sys
import chromadb
from sentence_transformers import SentenceTransformer
from player_selector import select_key_players

# ============================================================
# Config
# ============================================================
SQUADS_FILE = 'squads_2026.json'
VECTOR_DIR = './vector_store'
import os as _os2
LOCAL_MODEL = './models/all-MiniLM-L6-v2'
EMBEDDING_MODEL = LOCAL_MODEL if _os2.path.isdir(LOCAL_MODEL) else 'all-MiniLM-L6-v2'
COLLECTION_NAME = 'worldcup_kb'
MAX_CONTEXT_CHARS = 8000 

# LLM 配置 — DeepSeek
LLM_PROVIDER = 'deepseek'
LLM_MODEL = 'deepseek-chat' 
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '') 
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'


# ============================================================
# LLM 调用
# ============================================================

def call_llm(prompt):
    if LLM_PROVIDER == 'deepseek':
        import openai
        client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一位资深足球分析师，擅长分析世界杯淘汰赛。你的分析基于事实数据、战术理解和历史经验。给出明确、有依据的判断，不模棱两可。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == 'openai':
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一位资深足球分析师，擅长分析世界杯淘汰赛。你的分析基于事实数据、战术理解和历史经验。给出明确、有依据的判断，不模棱两可。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == 'claude':
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    elif LLM_PROVIDER == 'local':
        import requests
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': LLM_MODEL, 'prompt': prompt, 'stream': False}
        )
        return response.json()['response']

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ============================================================
# 初始化
# ============================================================

print("加载向量库和 embedding 模型...")
client = chromadb.PersistentClient(path=VECTOR_DIR)
collection = client.get_collection(COLLECTION_NAME)
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"  Collection: {COLLECTION_NAME} ({collection.count()} 条)")
print(f"  Embedding:  {EMBEDDING_MODEL} ({model.get_sentence_embedding_dimension()}d)\n")


# ============================================================
# 检索工具
# ============================================================

def exact_lookup(doc_ids):
    """精确 ID 检索。不存在的 ID 静默忽略。"""
    if not doc_ids:
        return {}
    try:
        results = collection.get(ids=doc_ids)
        return dict(zip(results['ids'], results['documents']))
    except Exception:
        return {}


def semantic_search(query, k=3):
    """语义检索，返回 [(doc_id, text, distance), ...]"""
    try:
        q_emb = model.encode([query]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=k)
        return list(zip(results['ids'][0], results['documents'][0], results['distances'][0]))
    except Exception:
        return []


# ============================================================
# 核心：四层 RAG 检索
# ============================================================

def retrieve(team_a, team_b, squads):
    """
    四层 RAG 检索。

    返回:
      layers: {layer_name: {'label': ..., 'results': [...], 'explanation': ...}}
      fetched: {doc_id: text}  所有已检索文本的去重汇总
    """
    sq_a = squads[team_a]
    sq_b = squads[team_b]

    # 关键球员（2FW + 2MF + 2DF + 1GK, 多因子评分）
    key_a = select_key_players(sq_a['players'], n_fw=2, n_mf=2, n_df=2, n_gk=1)
    key_b = select_key_players(sq_b['players'], n_fw=2, n_mf=2, n_df=2, n_gk=1)

    layers = {}
    fetched = {}

    # ================================================================
    # Layer 1: 球队画像 — 语义检索
    # 回答: 这支球队的风格、战术、近期状态是什么？
    # 为什么必须 RAG: 结构化数据只有数字，没有战术描述
    # ================================================================
    layer1_results = []
    for team, side in [(team_a, 'A'), (team_b, 'B')]:
        queries = [
            f"{team} national football team tactical style playing formation strengths",
            f"{team} football team recent form performance World Cup history achievements",
        ]
        for q in queries:
            for doc_id, doc_text, dist in semantic_search(q, k=2):
                if doc_id not in fetched:
                    fetched[doc_id] = doc_text
                    layer1_results.append({
                        'query': q,
                        'doc_id': doc_id,
                        'text': doc_text,
                        'distance': dist,
                    })

    layers['team_profiles'] = {
        'label': '球队画像（语义检索）',
        'explanation': '挖掘战术风格、历史地位、近期状态——结构化数据无法提供的信息',
        'results': layer1_results,
    }

    # ================================================================
    # Layer 2: 关键球员深度 — 精确 ID 检索
    # 回答: 核心球员除了出场/进球数字外，职业生涯和技术特点是什么？
    # 为什么必须 RAG: 球员姓名和进球数 ≠ 球员的实际竞技价值
    # ================================================================
    player_ids = [f"player:{p['name']}" for p in key_a + key_b]
    player_texts = exact_lookup(player_ids)

    for doc_id, text in player_texts.items():
        if doc_id not in fetched:
            fetched[doc_id] = text

    layers['key_players'] = {
        'label': '关键球员深度（精确ID检索）',
        'explanation': f'{team_a}: {", ".join(p["name"] for p in key_a)} | '
                       f'{team_b}: {", ".join(p["name"] for p in key_b)}',
        'results': [{'doc_id': k, 'text': v[:200] + '...'} for k, v in player_texts.items()],
    }

    # ================================================================
    # Layer 3: 俱乐部竞技背景 — 精确 ID 检索
    # 回答: 球员平时在什么水平的联赛踢球？这直接影响其国家队表现。
    # 为什么必须 RAG: "Inter Miami" 对 LLM 只是三个字母，
    #   只有读到摘要才知道它是 MLS 球队，竞技水平远低于欧冠
    # ================================================================
    club_ids = set()
    for p in key_a + key_b:
        club_ids.add(f"club:{p['club']}")
    club_texts = exact_lookup(list(club_ids))

    for doc_id, text in club_texts.items():
        if doc_id not in fetched:
            fetched[doc_id] = text

    # 按球员数排序俱乐部（更重要的俱乐部排在前面）
    club_summaries = []
    for p in key_a + key_b:
        doc_id = f"club:{p['club']}"
        if doc_id in club_texts and doc_id not in [c['doc_id'] for c in club_summaries]:
            club_summaries.append({
                'doc_id': doc_id,
                'club': p['club'],
                'player': p['name'],
                'team': p.get('_team', team_a if p in key_a else team_b),
                'text': club_texts[doc_id][:200] + '...',
            })

    layers['club_context'] = {
        'label': '俱乐部竞技背景（精确ID检索）',
        'explanation': '球员 → 俱乐部 → 联赛水平，反映日常比赛强度差异',
        'results': club_summaries,
    }

    # ================================================================
    # Layer 4: 历史交锋 & 风格碰撞 — 语义检索
    # 回答: 两队历史对阵如何？风格上谁克谁？
    # 为什么必须 RAG: 常规赛季对阵数据不在 squad 表中，
    #   风格克制关系需要从球队描述中推理
    # ================================================================
    layer4_queries = [
        f"{team_a} vs {team_b} head to head World Cup match history result",
        f"{team_a} {team_b} football match tactical analysis recent meeting",
        f"World Cup knockout stage {team_a} {team_b} preview prediction",
    ]
    layer4_results = []
    for q in layer4_queries:
        for doc_id, doc_text, dist in semantic_search(q, k=2):
            if doc_id not in fetched:
                fetched[doc_id] = doc_text
                layer4_results.append({
                    'query': q,
                    'doc_id': doc_id,
                    'text': doc_text,
                    'distance': dist,
                })

    layers['head_to_head'] = {
        'label': '历史交锋 & 风格碰撞（语义检索）',
        'explanation': '查找两队交手记录和风格克制关系——最需要"发现"的一层',
        'results': layer4_results,
    }

    return layers, fetched, (sq_a, sq_b, key_a, key_b)


# ============================================================
# 组装 Prompt
# ============================================================

def assemble_prompt(team_a, team_b, sq_a, sq_b, key_a, key_b, fetched, dry_run=False):
    """将四层检索结果 + 结构化数据组装为 LLM prompt"""

    # ---- 结构化数据对比表 ----
    comparison = f"""## 球队结构化数据对比

| 指标 | {team_a} | {team_b} |
|------|----------|----------|
| 球员数 | {sq_a['player_count']} | {sq_b['player_count']} |
| 平均年龄 | {sq_a['avg_age']}岁 | {sq_b['avg_age']}岁 |
| 总出场 | {sq_a['total_caps']}次 | {sq_b['total_caps']}次 |
| 总进球 | {sq_a['total_goals']}个 | {sq_b['total_goals']}个 |

### {team_a} 核心球员
"""
    for p in key_a:
        comparison += f"- {p['name']} ({p['position']}, {p['age']}岁, {p['club']}, 出场{p['caps']}/进球{p['goals']})\n"

    comparison += f"\n### {team_b} 核心球员\n"
    for p in key_b:
        comparison += f"- {p['name']} ({p['position']}, {p['age']}岁, {p['club']}, 出场{p['caps']}/进球{p['goals']})\n"

    # ---- RAG 检索的深度上下文 ----
    rag_sections = []

    # 球队摘要
    for team in [team_a, team_b]:
        doc_id = f"team:{team}"
        if doc_id in fetched:
            rag_sections.append(f"### {team} 球队分析\n{fetched[doc_id][:1000]}")

    # 球员深度信息
    player_chunks = []
    for p in key_a + key_b:
        doc_id = f"player:{p['name']}"
        if doc_id in fetched:
            player_chunks.append(f"- **{p['name']}** ({p['position']}, {p['club']}): {fetched[doc_id][:500]}")
    if player_chunks:
        rag_sections.append("### 球员深度分析\n" + "\n".join(player_chunks))

    # 俱乐部背景
    club_ids_done = set()
    club_chunks = []
    for p in key_a + key_b:
        doc_id = f"club:{p['club']}"
        if doc_id in fetched and doc_id not in club_ids_done:
            club_ids_done.add(doc_id)
            club_chunks.append(f"- **{p['club']}** ({p['name']}效力): {fetched[doc_id][:400]}")
    if club_chunks:
        rag_sections.append("### 俱乐部竞技背景\n" + "\n".join(club_chunks))

    # 历史交锋（从语义检索中提取 team/chunk 类型之外的条目）
    other_chunks = []
    for doc_id, text in fetched.items():
        if not any(doc_id.startswith(p) for p in ['team:', 'player:', 'club:', 'group:']):
            other_chunks.append(text[:500])
    if other_chunks:
        rag_sections.append("### 历史交锋 & 参考信息\n" + "\n---\n".join(other_chunks[:5]))

    # ---- 最终组装 ----
    rag_context = "\n\n".join(rag_sections)
    full_context = comparison + "\n\n## RAG 检索深度上下文\n" + rag_context

    if len(full_context) > MAX_CONTEXT_CHARS:
        full_context = full_context[:MAX_CONTEXT_CHARS] + "\n\n...[上下文已截断]"

    prompt = f"""你是一位资深足球分析师。请基于以下信息，预测 2026 世界杯淘汰赛: **{team_a} vs {team_b}**。

{full_context}

## 分析要求
请严格按以下结构给出分析:

### 1. 双方实力对比
从球员质量、年龄结构、大赛经验、阵容深度、俱乐部竞技水平等维度对比。
注意：**俱乐部竞技水平直接影响球员状态**——在顶级联赛（英超、西甲、欧冠）效力的球员，日常对抗强度远高于低级别联赛。

### 2. 关键对位
指出 2-3 组决定比赛走向的具体球员对位（如某队边锋 vs 某队边后卫），分析谁占优。

### 3. 战术分析
基于球队风格、球员配置和关键对位，分析可能的战术博弈和比赛走向。

### 4. 预测
给出 90 分钟内比分预测，并简要说明理由。如有需要可附上加时/点球的判断。

输出语言: 中文。给出明确判断，不做"取决于临场发挥"的模糊表述。"""

    if dry_run:
        print("\n" + "=" * 70)
        print("  📋 组装后的 Prompt（前 3000 字符）")
        print("=" * 70)
        print(prompt[:3000])
        print(f"\n  [总长度: {len(prompt)} 字符]")
        return None

    return prompt


# ============================================================
# 主预测函数
# ============================================================

def predict(team_a, team_b, dry_run=False):
    """运行完整预测流程"""

    # ---- 加载结构化数据 ----
    with open(SQUADS_FILE, 'r', encoding='utf-8') as f:
        squads = json.load(f)

    for team in [team_a, team_b]:
        if team not in squads:
            print(f"❌ 未找到球队: {team}")
            return

    # ---- 四层 RAG 检索 ----
    print("=" * 70)
    print(f"  🏆 {team_a} vs {team_b}")
    print("=" * 70)

    layers, fetched, (sq_a, sq_b, key_a, key_b) = retrieve(team_a, team_b, squads)

    # 打印检索摘要
    for layer_name, layer_data in layers.items():
        n = len(layer_data['results'])
        print(f"\n  [{layer_data['label']}] — {n} 条结果")
        print(f"  说明: {layer_data['explanation']}")

    print(f"\n  📊 总计检索: {len(fetched)} 条唯一文档")

    # ---- 组装 Prompt ----
    prompt = assemble_prompt(team_a, team_b, sq_a, sq_b, key_a, key_b, fetched, dry_run)

    if dry_run:
        print("\n  🔍 --dry 模式: 不调用 LLM")
        return

    # ---- 调用 LLM ----
    print(f"\n  🤖 调用 {LLM_PROVIDER}:{LLM_MODEL} ...")
    prediction = call_llm(prompt)

    print("\n" + "=" * 70)
    print("  📊 预测结果")
    print("=" * 70)
    print(prediction)

    return prediction


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    dry = '--dry' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if len(args) >= 2:
        team_a, team_b = args[0], args[1]
    else:
        team_a, team_b = 'Argentina', 'France'

    predict(team_a, team_b, dry_run=dry)
