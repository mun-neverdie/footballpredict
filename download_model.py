"""
用 requests 下载 all-MiniLM-L6-v2 模型到本地目录。
requests 走 Clash 代理已验证可行。
"""
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import json
import requests
from pathlib import Path

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
LOCAL_DIR = Path('./models/all-MiniLM-L6-v2')
HF_API = 'https://huggingface.co/api'

session = requests.Session()
session.proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}

# 1. 获取文件列表
print(f"获取 {MODEL_ID} 文件列表...")
r = session.get(f'{HF_API}/models/{MODEL_ID}')
repo_info = r.json()
files = [f['rfilename'] for f in repo_info.get('siblings', [])]
print(f"  共 {len(files)} 个文件")

# 2. 下载所有文件
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

for i, fname in enumerate(files):
    url = f'https://huggingface.co/{MODEL_ID}/resolve/main/{fname}'
    local_path = LOCAL_DIR / fname
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{i+1}/{len(files)}] {fname}...", end=' ', flush=True)
    try:
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(r.content)
            print(f"✓ {len(r.content) / 1024:.0f} KB")
        else:
            print(f"✗ HTTP {r.status_code}")
    except Exception as e:
        print(f"✗ {e}")

print(f"\n✅ 模型已下载到 {LOCAL_DIR.absolute()}")
