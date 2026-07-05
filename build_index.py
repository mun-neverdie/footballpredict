"""
build_index.py — 将 kb_entries.json 向量化并存入 ChromaDB

前置条件: 运行 build_kb.py 生成 kb_entries.json
依赖: pip install chromadb sentence-transformers
"""

import json
import os

# 代理设置（ChromaDB 不需要，但如果后续需要联网则保留）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ============================================================
# Config
# ============================================================
KB_FILE = 'kb_entries.json'
VECTOR_DIR = './vector_store'
# 优先从本地加载，避免 HuggingFace httpx 代理兼容问题
import os as _os
LOCAL_MODEL = './models/all-MiniLM-L6-v2'
EMBEDDING_MODEL = LOCAL_MODEL if _os.path.isdir(LOCAL_MODEL) else 'all-MiniLM-L6-v2'
COLLECTION_NAME = 'worldcup_kb'


# ============================================================
# Load kb_entries
# ============================================================
print(f"加载知识库: {KB_FILE}")
with open(KB_FILE, 'r', encoding='utf-8') as f:
    entries = json.load(f)
print(f"共 {len(entries)} 个条目")

# ============================================================
# Init embedding model
# ============================================================
print(f"\n加载 embedding 模型: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"  向量维度: {model.get_sentence_embedding_dimension()}")

# ============================================================
# ChromaDB
# ============================================================
print(f"\n初始化 ChromaDB: {VECTOR_DIR}")
client = chromadb.PersistentClient(path=VECTOR_DIR)

# 如果已存在则先删除（重建）
existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing:
    print(f"  删除旧 collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ============================================================
# Embed + Insert (批量)
# ============================================================
BATCH_SIZE = 100
print(f"\n开始向量化并写入...")

for i in range(0, len(entries), BATCH_SIZE):
    batch = entries[i:i + BATCH_SIZE]
    ids = [e['id'] for e in batch]
    texts = [e['text'] for e in batch]
    metadatas = []
    for e in batch:
        clean_meta = {'type': e['type']}  # 把顶层 type 拍进 metadata
        for k, v in e.get('metadata', {}).items():
            if v is None:
                clean_meta[k] = ''
            elif isinstance(v, (list, dict)):
                clean_meta[k] = str(v)
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        metadatas.append(clean_meta)

    # 生成向量
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    # 写入 ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"  [{min(i + BATCH_SIZE, len(entries))}/{len(entries)}] 已写入")

# ============================================================
# 验证
# ============================================================
print(f"\n✅ 入库完成!")
print(f"  Collection: {COLLECTION_NAME}")
print(f"  条目数: {collection.count()}")
print(f"  向量维度: {model.get_sentence_embedding_dimension()}")

# 按类型统计
print("\n  类型分布:")
for etype in ['team_summary', 'club_summary', 'player_bio', 'group_info']:
    count = len([e for e in entries if e['type'] == etype])
    print(f"    {etype}: {count}")

# 测试检索
print("\n" + "=" * 60)
print("测试检索: 'Argentina national team'")
print("=" * 60)
query_embedding = model.encode(["Argentina national football team"]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=3)
for i, (doc_id, dist, doc) in enumerate(zip(
    results['ids'][0], results['distances'][0], results['documents'][0]
)):
    print(f"\n  #{i+1} [{doc_id}] distance={dist:.4f}")
    print(f"  {doc[:200]}...")

# 测试 metadata 过滤检索
print("\n" + "=" * 60)
print("测试 metadata 过滤: type=='team_summary', team=='Argentina'")
print("=" * 60)
try:
    filtered = collection.get(
        where={"$and": [
            {"type": "team_summary"},
            {"team": "Argentina"}
        ]}
    )
    if filtered['ids']:
        print(f"  找到: {filtered['ids'][0]}")
        print(f"  {filtered['documents'][0][:200]}...")
    else:
        print("  未找到")
except Exception as e:
    print(f"  过滤查询出错: {e}")

print("\n✅ build_index.py 完成")
print(f"向量库位置: {os.path.abspath(VECTOR_DIR)}")
