import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import json
import re
import time
import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer

API_URL = 'http://en.wikipedia.org/w/api.php'
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; WorldCupPredict/1.0)'}

from player_selector import select_key_players  # noqa: E402


# ---- 新工具: Wikipedia 章节抓取 ----

def fetch_wikipedia_sections(title):
    """获取页面所有章节列表 [{index, line, number}, ...]。失败返回 []。"""
    try:
        r = requests.get(API_URL, params={
            'action': 'parse', 'page': title,
            'prop': 'sections', 'format': 'json', 'redirects': 1
        }, headers=HEADERS, timeout=15)
        if not r.text.strip():
            return []
        data = r.json()
        if 'error' in data:
            return []
        return data.get('parse', {}).get('sections', [])
    except Exception:
        return []


def fetch_wikipedia_section_text(title, section_index):
    """抓取指定章节的纯文本。失败返回 None。"""
    try:
        r = requests.get(API_URL, params={
            'action': 'parse', 'page': title,
            'prop': 'text', 'format': 'json', 'redirects': 1,
            'section': section_index
        }, headers=HEADERS, timeout=15)
        if not r.text.strip():
            return None
        data = r.json()
        if 'error' in data:
            return None
        html = data['parse']['text']['*']
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup.find_all(['table', 'sup', 'script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return None


def fetch_player_season_info(wiki_title, max_chars=1500):
    """
    爬取球员近期表现，按优先级回退：
      1) 找含 '2025' 或 '2026' 的章节
      2) 找 'Club career' 的最后一个子章节
      3) 全量取 'Club career' 章节
      4) 全量取 'Career' 章节
      5) 都没有 → 返回 None
    任何步骤出错 → 返回 None（不中断主流程）
    """
    try:
        sections = fetch_wikipedia_sections(wiki_title)
        if not sections:
            return None

        # --- 优先级 1: 2025/2026 赛季章节 ---
        season_secs = [s for s in sections
                       if any(y in s.get('line', '') for y in ['2025', '2026'])]
        if season_secs:
            texts = []
            for s in season_secs[:3]:
                text = fetch_wikipedia_section_text(wiki_title, s['index'])
                if text:
                    texts.append(f"[{s['line']}]: {text}")
            if texts:
                result = '\n'.join(texts)
                return result[:max_chars] if len(result) > max_chars else result

        # --- 优先级 2-3: Club career ---
        club_idx = None
        for s in sections:
            if s.get('line', '').lower().startswith('club career'):
                club_idx = int(s['index'])
                break

        if club_idx is not None:
            # 找 Club career 的所有直接子章节（number 层级更高一级）
            subs = [s for s in sections
                    if s['index'] != str(club_idx)
                    and s['number'].startswith(str(club_idx + 1) + '.')]
            if subs:
                # 优先级 2: 取最后一个子章节
                last_sub = subs[-1]
                text = fetch_wikipedia_section_text(wiki_title, last_sub['index'])
                if text:
                    result = f"[{last_sub['line']}]: {text}"
                    return result[:max_chars] if len(result) > max_chars else result
            else:
                # 优先级 3: 全量取 Club career 章节
                text = fetch_wikipedia_section_text(wiki_title, str(club_idx))
                if text:
                    result = f"[Club career]: {text}"
                    return result[:max_chars] if len(result) > max_chars else result

        # --- 优先级 4: Career ---
        for s in sections:
            if s.get('line', '').lower().startswith('career'):
                text = fetch_wikipedia_section_text(wiki_title, s['index'])
                if text:
                    result = f"[{s['line']}]: {text}"
                    return result[:max_chars] if len(result) > max_chars else result
                break

        # --- 优先级 5: 放弃 ---
        return None

    except Exception:
        return None


# ---- 加载摘要模型（全局一次） ----
MODEL_NAME = 'facebook/bart-large-cnn'
print(f"加载摘要模型 {MODEL_NAME} ...")
_tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
print("摘要模型就绪\n")

# ============================================================
# 工具函数
# ============================================================

def summarize(text, max_len=150):
    """使用 BART 将 Wikipedia 文本压缩为摘要。短文本直接返回。"""
    if not text or len(text) < 200:
        return text
    try:
        inputs = _tokenizer(text[:3000], return_tensors='pt',
                            max_length=1024, truncation=True)
        summary_ids = _model.generate(
            inputs['input_ids'],
            max_length=max_len,
            min_length=50,
            num_beams=4,
            early_stopping=True,
        )
        return _tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception:
        return text[:max_len * 6]

def search_wikipedia(query, retries=3):
    """搜索 Wikipedia，带重试"""
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params={
                'action': 'opensearch', 'search': query,
                'limit': 3, 'namespace': 0, 'format': 'json'
            }, headers=HEADERS, timeout=15)
            if r.status_code == 200 and r.text.strip():
                data = r.json()
                if len(data) > 1 and data[1]:
                    return data[1]
            # 非 200 或空响应 → 重试
            if attempt < retries - 1:
                time.sleep(1.0)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0)
    return []


def fetch_wikipedia_text(title, max_chars=5000, retries=3):
    """抓取 Wikipedia 页面全文（纯文本），用于后续摘要。带重试。"""
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params={
                'action': 'parse', 'page': title,
                'prop': 'text', 'format': 'json', 'redirects': 1,
            }, headers=HEADERS, timeout=15)
            data = r.json()
            if 'error' in data:
                return None
            html = data['parse']['text']['*']
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup.find_all(['table', 'sup', 'script', 'style', '.mw-editsection',
                                       '.hatnote', '.sidebar', '.navbox', '.infobox']):
                tag.decompose()
            text = soup.get_text(separator=' ')
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                # 返回全文供摘要使用
                return text[:max_chars]
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0)
    return None


# ---- 断点续传缓存 ----
CACHE_FILE = 'crawl_cache.json'

def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def batch_crawl(items, search_suffix, max_chars=5000, summary_len=150, retries=2,
                cache_prefix=''):
    """
    批量爬取 + BART 摘要 + 断点续传。
      items: [(id, name), ...]
      search_suffix: 搜索后缀
      cache_prefix: 缓存键前缀 (如 'team', 'club', 'player')，区分不同阶段
      返回 (results, failures)
    """
    cache = _load_cache()
    results = {}
    failures = []
    total = len(items)
    skipped = 0

    for i, (item_id, item_name) in enumerate(items):
        cache_key = f"{cache_prefix}:{item_id}"

        # --- 断点续传：已缓存则跳过 ---
        if cache_key in cache:
            cached = cache[cache_key]
            results[item_id] = cached
            if cached.get('success'):
                skipped += 1
                continue  # 静默跳过，不刷屏
            else:
                # 之前失败的，重新尝试（不跳过）
                pass

        print(f"  [{i+1}/{total}] {item_name}...", end=' ', flush=True)

        titles = None
        search_query = f"{item_name} {search_suffix}".strip()
        for attempt in range(retries):
            titles = search_wikipedia(search_query, retries=2)
            if not titles and search_suffix:
                titles = search_wikipedia(item_name, retries=2)
            if titles:
                break
            if attempt < retries - 1:
                print(f"(重试{attempt+1})...", end=' ', flush=True)
                time.sleep(2.0)

        if titles:
            raw_text = fetch_wikipedia_text(titles[0], max_chars)
            if raw_text:
                summary = summarize(raw_text, max_len=summary_len)
                entry = {
                    'title': titles[0],
                    'text': summary,
                    'success': True,
                }
                results[item_id] = entry
                # 立即写缓存
                cache[cache_key] = entry
                _save_cache(cache)
                print(f"✓ {len(summary)} chars")
            else:
                entry = {'title': titles[0], 'text': None, 'success': False}
                results[item_id] = entry
                cache[cache_key] = entry
                _save_cache(cache)
                print("✗ 文本为空")
                failures.append({
                    'id': item_id, 'name': item_name,
                    'search_query': search_query,
                    'found_title': titles[0],
                    'reason': '页面获取后文本为空',
                })
        else:
            entry = {'title': None, 'text': None, 'success': False}
            results[item_id] = entry
            # 搜索失败的也缓存（避免重复搜索），但下次会重试
            cache[cache_key] = entry
            _save_cache(cache)
            print("✗ 搜索无结果(代理可能超时)")
            failures.append({
                'id': item_id, 'name': item_name,
                'search_query': search_query,
                'found_title': None,
                'reason': 'Wikipedia搜索无结果，代理/网络问题',
            })

        time.sleep(0.3)

    if skipped:
        print(f"  ⏭ 跳过 {skipped} 个已缓存条目")
    return results, failures


# ============================================================
# Phase 1: 提取实体列表
# ============================================================

print("=" * 60)
print("Phase 1: 加载并分析 squads_2026.json")
print("=" * 60)

with open('squads_2026.json', 'r', encoding='utf-8') as f:
    squads = json.load(f)

# 所有国家队
all_teams = sorted(squads.keys())
print(f"国家队: {len(all_teams)} 支")

# 所有唯一俱乐部
all_clubs = set()
player_club_map = {}  # {(team, player_name): club}
for team, data in squads.items():
    for p in data['players']:
        all_clubs.add(p['club'])
        player_club_map[(team, p['name'])] = p['club']
all_clubs = sorted(all_clubs)
print(f"唯一俱乐部: {len(all_clubs)} 个")

# 关键球员 (每队: 2FW + 2MF + 2DF + 1GK = 7 人, 多因子评分)
key_players = {}
for team, data in squads.items():
    selected = select_key_players(data['players'], n_fw=2, n_mf=2, n_df=2, n_gk=1)
    key_players[team] = [p['name'] for p in selected]
unique_key_players = set()
for names in key_players.values():
    unique_key_players.update(names)
print(f"关键球员: {len(unique_key_players)} 人 (每队 2FW+2MF+2DF+1GK)")

# ============================================================
# Phase 2: 批量爬取 Wikipedia
# ============================================================

print("\n" + "=" * 60)
print("Phase 2.1: 爬取国家队页面（摘要 ~200词）")
print("=" * 60)
team_raw, team_failures = batch_crawl(
    [(t, t) for t in all_teams],
    search_suffix="national football team",
    max_chars=5000,
    summary_len=200,
    cache_prefix='team'
)
for f in team_failures:
    f['entity_type'] = 'team'

print("\n" + "=" * 60)
print("Phase 2.2: 爬取俱乐部页面（摘要 ~150词）")
print("=" * 60)
club_items = [(c, c) for c in all_clubs]
club_raw, club_failures = batch_crawl(
    club_items,
    search_suffix="football club",
    max_chars=5000,
    summary_len=150,
    cache_prefix='club'
)
for f in club_failures:
    f['entity_type'] = 'club'

print("\n" + "=" * 60)
print("Phase 2.3: 爬取关键球员页面（摘要 ~120词 + 2025-26 赛季章节）")
print("=" * 60)
player_items = [(p, p) for p in sorted(unique_key_players)]
player_raw, player_failures = batch_crawl(
    player_items,
    search_suffix="footballer",
    max_chars=5000,
    summary_len=120,
    cache_prefix='player'
)
for f in player_failures:
    f['entity_type'] = 'player'

# 附加: 爬取球员 2025-26 赛季章节
print("\n" + "=" * 60)
print("Phase 2.4: 爬取关键球员 2025-26 赛季表现")
print("=" * 60)
season_ok = 0
for i, (player_name, raw) in enumerate(player_raw.items()):
    if not raw.get('success') or not raw.get('title'):
        continue
    print(f"  [{i+1}/{len(player_raw)}] {player_name}...", end=' ', flush=True)
    season_text = fetch_player_season_info(raw['title'], max_chars=1500)
    if season_text:
        summary = summarize(season_text, max_len=80)
        raw['text'] = raw['text'] + '\n\n[2025-26 赛季]: ' + summary
        print(f"✓ {len(summary)} chars")
        season_ok += 1
    else:
        print("(无 2025-26 章节)")
    time.sleep(0.3)
print(f"  赛季信息成功: {season_ok}/{len(player_raw)}")

# 汇总所有失败
all_failures = team_failures + club_failures + player_failures

# ============================================================
# Phase 2.5: 2026 FIFA World Cup 赛事页面（小组赛+淘汰赛+射手榜）
# ============================================================

print("\n" + "=" * 60)
print("Phase 2.5: 爬取 2026 FIFA World Cup 赛事页面")
print("=" * 60)

TOURNAMENT_PAGE = '2026 FIFA World Cup'
print(f"  抓取 '{TOURNAMENT_PAGE}' ...")
tournament_raw = fetch_wikipedia_text(TOURNAMENT_PAGE, max_chars=8000)

if tournament_raw:
    tournament_summary = summarize(tournament_raw, max_len=300)
    tournament_chunks = [{
        'type': 'tournament_overview',
        'id': 'tournament:overview',
        'text': f'2026 FIFA World Cup 赛事概况:\n{tournament_summary}',
    }]
    print(f"  赛事概况: ✓ {len(tournament_summary)} chars")

    # 抓取小组赛章节
    sections = fetch_wikipedia_sections(TOURNAMENT_PAGE)
    group_sections = [s for s in sections if s.get('line', '').startswith('Group ')]
    print(f"  小组章节: {len(group_sections)} 个 ({', '.join(s['line'] for s in group_sections[:4])}...)")

    for gs in group_sections:
        group_text = fetch_wikipedia_section_text(TOURNAMENT_PAGE, gs['index'])
        if group_text:
            gsummary = summarize(group_text, max_len=120)
            tournament_chunks.append({
                'type': 'group_result',
                'id': f"tournament:group:{gs['line']}",
                'text': f"2026世界杯 {gs['line']}:\n{gsummary}",
            })

    # 抓取淘汰赛章节
    ko_sections = [s for s in sections
                   if any(kw in s.get('line', '').lower()
                          for kw in ['knockout', 'bracket', 'round of 16', 'quarter',
                                     'semi', 'final', 'third place'])]
    for ks in ko_sections[:5]:
        ko_text = fetch_wikipedia_section_text(TOURNAMENT_PAGE, ks['index'])
        if ko_text:
            ksummary = summarize(ko_text, max_len=120)
            tournament_chunks.append({
                'type': 'knockout_info',
                'id': f"tournament:knockout:{ks['line'].replace(' ', '_')}",
                'text': f"2026世界杯 {ks['line']}:\n{ksummary}",
            })

    # 抓取射手榜
    scorer_sections = [s for s in sections
                       if any(kw in s.get('line', '').lower()
                              for kw in ['goal', 'scorer', 'top scorer', 'golden boot'])]
    for ss in scorer_sections[:3]:
        stext = fetch_wikipedia_section_text(TOURNAMENT_PAGE, ss['index'])
        if stext:
            ssummary = summarize(stext, max_len=120)
            tournament_chunks.append({
                'type': 'tournament_stats',
                'id': f"tournament:scorers:{ss['line'].replace(' ', '_')}",
                'text': f"2026世界杯 {ss['line']}:\n{ssummary}",
            })

    print(f"  赛事知识块: {len(tournament_chunks)} 条")
else:
    tournament_chunks = []
    print("  ✗ 赛事页面获取失败")

# ============================================================
# Phase 3: 组装知识块
# ============================================================

print("\n" + "=" * 60)
print("Phase 3: 组装知识库条目")
print("=" * 60)

entries = []
entry_id = 0

# --- 3a: 国家队条目 ---
for team in all_teams:
    entry_id += 1
    data = squads[team]
    raw = team_raw.get(team, {})

    # 构建带统计的结构化文本
    key_names = key_players.get(team, [])
    stats_text = (
        f"{team} 国家队 - 2026世界杯阵容:\n"
        f"球员数: {data['player_count']}, "
        f"平均年龄: {data['avg_age']}岁, "
        f"总出场: {data['total_caps']}次, "
        f"总进球: {data['total_goals']}个.\n"
        f"关键球员: {', '.join(key_names[:8])}.\n"
    )

    wiki_text = raw.get('text', '') if raw.get('success') else ''
    full_text = stats_text + "\nWikipedia 摘要:\n" + wiki_text

    entries.append({
        'id': f"team:{team}",
        'type': 'team_summary',
        'team': team,
        'text': full_text,
        'metadata': {
            'team': team,
            'avg_age': data['avg_age'],
            'total_caps': data['total_caps'],
            'total_goals': data['total_goals'],
            'player_count': data['player_count'],
            'key_players': key_names,
            'wiki_title': raw.get('title'),
        }
    })

# --- 3b: 俱乐部条目 ---
club_player_count = {}
for team, data in squads.items():
    for p in data['players']:
        club_player_count[p['club']] = club_player_count.get(p['club'], 0) + 1

for club in all_clubs:
    entry_id += 1
    raw = club_raw.get(club, {})
    player_count = club_player_count.get(club, 0)

    # 找到使用这个俱乐部的球队
    teams_using = set()
    for team, data in squads.items():
        for p in data['players']:
            if p['club'] == club:
                teams_using.add(team)

    stats_text = (
        f"{club} - 足球俱乐部.\n"
        f"2026世界杯参赛球员数: {player_count}人.\n"
        f"代表国家队: {', '.join(sorted(teams_using))}.\n"
    )

    wiki_text = raw.get('text', '') if raw.get('success') else ''
    full_text = stats_text + "\nWikipedia 摘要:\n" + wiki_text

    entries.append({
        'id': f"club:{club}",
        'type': 'club_summary',
        'club': club,
        'text': full_text,
        'metadata': {
            'club': club,
            'player_count': player_count,
            'teams': sorted(teams_using),
            'wiki_title': raw.get('title'),
        }
    })

# --- 3c: 球员条目 ---
for team in all_teams:
    data = squads[team]
    for p in data['players']:
        if p['name'] in unique_key_players:
            entry_id += 1
            raw = player_raw.get(p['name'], {})

            stats_text = (
                f"{p['name']} - {team}国家队球员.\n"
                f"位置: {p['position']}, 年龄: {p['age']}, "
                f"国家队出场: {p['caps']}, 进球: {p['goals']}, "
                f"效力俱乐部: {p['club']}.\n"
            )

            wiki_text = raw.get('text', '') if raw.get('success') else ''
            full_text = stats_text + "\nWikipedia 摘要:\n" + wiki_text

            entries.append({
                'id': f"player:{p['name']}",
                'type': 'player_bio',
                'player': p['name'],
                'team': team,
                'club': p['club'],
                'text': full_text,
                'metadata': {
                    'player': p['name'],
                    'team': team,
                    'club': p['club'],
                    'position': p['position'],
                    'age': p['age'],
                    'caps': p['caps'],
                    'goals': p['goals'],
                    'wiki_title': raw.get('title'),
                }
            })

# --- 3d: 小组信息 (从原始页面解析的 H2 分组结构重建) ---
# 2026 世界杯分组已从页面 H2 解析，这里硬编码确保完整性
groups_2026 = {
    'A': ['Czech Republic', 'Mexico', 'South Africa', 'South Korea'],
    'B': ['Bosnia and Herzegovina', 'Canada', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Haiti', 'Morocco', 'Scotland'],
    'D': ['Australia', 'Paraguay', 'Turkey', 'United States'],
    'E': ['Curaçao', 'Ecuador', 'Germany', 'Ivory Coast'],
    'F': ['Japan', 'Netherlands', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Cape Verde', 'Saudi Arabia', 'Spain', 'Uruguay'],
    'I': ['France', 'Iraq', 'Norway', 'Senegal'],
    'J': ['Algeria', 'Argentina', 'Austria', 'Jordan'],
    'K': ['Colombia', 'DR Congo', 'Portugal', 'Uzbekistan'],
    'L': ['Croatia', 'England', 'Ghana', 'Panama'],
}

for group_name, teams in groups_2026.items():
    entry_id += 1
    team_stats = []
    for t in teams:
        if t in squads:
            d = squads[t]
            team_stats.append(
                f"{t}(avg_age:{d['avg_age']}, caps:{d['total_caps']}, goals:{d['total_goals']})"
            )

    entries.append({
        'id': f"group:{group_name}",
        'type': 'group_info',
        'text': f"2026世界杯 Group {group_name}: " + "; ".join(team_stats),
        'metadata': {
            'group': group_name,
            'teams': teams,
        }
    })

# --- 3e: 世界杯赛事信息（小组赛结果 + 淘汰赛 + 射手榜） ---
for tc in tournament_chunks:
    entry_id += 1
    entries.append({
        'id': tc['id'],
        'type': tc['type'],
        'text': tc['text'],
        'metadata': {},
    })

# ============================================================
# Phase 4: 统计 + 保存
# ============================================================

print(f"\n总条目数: {len(entries)}")
type_counts = {}
success_count = 0
for e in entries:
    t = e['type']
    type_counts[t] = type_counts.get(t, 0) + 1
    if e.get('metadata', {}).get('wiki_title'):
        success_count += 1

print(f"类型分布: {type_counts}")
print(f"Wikipedia 爬取成功: {success_count}/{len(entries)}")

output_file = 'kb_entries.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)
print(f"\n已保存到 {output_file} (大小: {os.path.getsize(output_file) / 1024:.1f} KB)")

# ---- 保存失败记录 ----
print(f"\n爬取失败: {len(all_failures)} 条")
if all_failures:
    fail_file = 'failed_entries.json'
    with open(fail_file, 'w', encoding='utf-8') as f:
        json.dump(all_failures, f, ensure_ascii=False, indent=2)
    print(f"已保存到 {fail_file}")

    # 按类型统计失败
    fail_by_type = {}
    for f in all_failures:
        t = f.get('entity_type', 'unknown')
        fail_by_type[t] = fail_by_type.get(t, 0) + 1
    print(f"失败分布: {fail_by_type}")
    print(f"\n手动修复方式：对 failed_entries.json 中每个条目，手动查 Wikipedia 页面标题,")
    print(f"然后添加到 kb_entries.json 或 failed_fix.json 中")

# 预览
print("\n" + "=" * 60)
print("预览（各类型 1 条）")
print("=" * 60)
shown_types = set()
for e in entries:
    if e['type'] not in shown_types:
        shown_types.add(e['type'])
        print(f"\n[{e['type']}] {e['id']}")
        print(f"  text 长度: {len(e['text'])} chars")
        print(f"  内容前 200 字: {e['text'][:200]}...")
