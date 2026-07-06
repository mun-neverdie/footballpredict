"""
streamlit_app.py — 世界杯预测前端

用法: streamlit run streamlit_app.py
"""

import streamlit as st
import json
import sys
sys.path.insert(0, '.')

from predict import predict, SQUADS_FILE

# ---- 页面配置 ----
st.set_page_config(
    page_title="2026世界杯 AI 预测",
    page_icon="⚽",
    layout="wide",
)

# ---- 加载球队列表 ----
@st.cache_data
def load_teams():
    with open(SQUADS_FILE, 'r', encoding='utf-8') as f:
        squads = json.load(f)
    return sorted(squads.keys()), squads

TEAM_LIST, SQUADS = load_teams()

# ---- 解析 LLM 输出为三个区域 ----
def parse_prediction(raw_text):
    """按 `---` 分隔符拆分 LLM 输出"""
    parts = [p.strip() for p in raw_text.split('---')]
    result = {'tactics': '', 'winner': '', 'scores': ''}
    labels = ['tactics', 'winner', 'scores']
    for i, label in enumerate(labels):
        if i < len(parts):
            # 去掉可能的 markdown 标题残留
            text = parts[i]
            text = text.replace('### 战术分析', '').replace('### 胜负预测', '').replace('### 比分预测', '')
            result[label] = text.strip()
    return result


# ---- UI ----
st.title("⚽ 2026 世界杯淘汰赛 AI 预测")
st.caption("基于 RAG 检索增强生成 — 知识库来自 Wikipedia 实时数据")

# ---- 球队选择 ----
col1, col_mid, col2 = st.columns([3, 0.5, 3])

with col1:
    mode_a = st.radio("队伍 A 选择方式", ["下拉选择", "手动输入"], key='mode_a', horizontal=True)
    if mode_a == "下拉选择":
        team_a = st.selectbox("队伍 A", TEAM_LIST, key='select_a',
                              index=TEAM_LIST.index('Argentina') if 'Argentina' in TEAM_LIST else 0)
    else:
        team_a = st.text_input("输入队伍 A 名称 (英文)", "Argentina", key='type_a')

with col_mid:
    st.markdown("<h1 style='text-align:center; margin-top:40px;'>VS</h1>", unsafe_allow_html=True)

with col2:
    mode_b = st.radio("队伍 B 选择方式", ["下拉选择", "手动输入"], key='mode_b', horizontal=True)
    if mode_b == "下拉选择":
        team_b = st.selectbox("队伍 B", TEAM_LIST, key='select_b',
                              index=TEAM_LIST.index('France') if 'France' in TEAM_LIST else 0)
    else:
        team_b = st.text_input("输入队伍 B 名称 (英文)", "France", key='type_b')

# ---- 预测按钮 ----
predict_btn = st.button("🔮 开始预测", type="primary", use_container_width=True)

# ---- 验证 & 预测 ----
if predict_btn:
    # 验证
    errors = []
    if team_a not in SQUADS:
        errors.append(f'"{team_a}" 不在 2026 世界杯参赛队伍中')
    if team_b not in SQUADS:
        errors.append(f'"{team_b}" 不在 2026 世界杯参赛队伍中')
    if team_a == team_b:
        errors.append("不能预测同一支队伍！")

    if errors:
        for e in errors:
            st.error(f"❌ {e}")
    else:
        with st.spinner(f"正在分析 {team_a} vs {team_b}...\n\n检索知识库 → 向量搜索 → LLM 推理，约 10-20 秒"):
            raw_result = predict(team_a, team_b)

        if raw_result and not raw_result.startswith("❌"):
            parsed = parse_prediction(raw_result)

            # ---- 三个结果卡片 ----
            st.divider()

            card1, card2, card3 = st.columns(3)

            card_style = ("background:#1a1a2e;padding:20px;border-radius:12px;"
                          "min-height:250px;font-size:14px;line-height:1.7;"
                          "color:#e8e8e8;")

            with card1:
                st.markdown("### 🎯 战术分析")
                st.markdown(
                    f"<div style='{card_style}'>{parsed['tactics']}</div>",
                    unsafe_allow_html=True,
                )

            with card2:
                st.markdown("### ⚡ 胜负预测")
                st.markdown(
                    f"<div style='{card_style}'>{parsed['winner']}</div>",
                    unsafe_allow_html=True,
                )

            with card3:
                st.markdown("### 📊 最可能比分 Top 3")
                st.markdown(
                    f"<div style='{card_style}'>{parsed['scores']}</div>",
                    unsafe_allow_html=True,
                )

            # 展开原文
            with st.expander("查看原始输出"):
                st.text(raw_result)
        else:
            st.error(raw_result or "预测失败，请检查网络和 API Key")

# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("## 📋 参赛队伍")
    st.markdown(f"共 **{len(TEAM_LIST)}** 支")
    st.markdown("---")
    # 按字母顺序显示全部队伍
    cols = st.columns(2)
    for i, team in enumerate(TEAM_LIST):
        cols[i % 2].markdown(f"- {team}")

    st.markdown("---")
    st.caption("RAG 知识库: Wikipedia + BART 摘要")
    st.caption("Embedding: all-MiniLM-L6-v2")
    st.caption("LLM: DeepSeek Chat")
