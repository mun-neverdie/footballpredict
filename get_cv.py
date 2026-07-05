import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import json
import re
import requests
from bs4 import BeautifulSoup

API_URL = 'http://en.wikipedia.org/w/api.php'
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; WorldCupPredict/1.0)'}


def wikipedia_search(query, limit=10):
    r = requests.get(API_URL, params={
        'action': 'opensearch', 'search': query,
        'limit': limit, 'namespace': 0, 'format': 'json'
    }, headers=HEADERS)
    return r.json()[1]


def wikipedia_page(title):
    r = requests.get(API_URL, params={
        'action': 'parse', 'page': title,
        'prop': 'text', 'format': 'json', 'redirects': 1
    }, headers=HEADERS)
    data = r.json()
    if 'error' in data:
        raise Exception(f"Page error: {data['error']['info']}")
    return data['parse']['text']['*'], data['parse']['title']


def parse_age(dob_text):
    """从 '(2000-05-17)17 May 2000 (aged 26)' 中提取年龄"""
    match = re.search(r'aged\s*(\d+)', dob_text)
    if match:
        return int(match.group(1))
    return None


def clean_text(text):
    """清洗文本：去括号标注、去引用脚注、去多余空白"""
    # 移除 (captain), (vice-captain), (injured) 等
    text = re.sub(r'\(captain\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(vice-captain\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(injured\)', '', text, flags=re.IGNORECASE)
    # 移除 Wikipedia 引用脚注 [1], [note 1] 等
    text = re.sub(r'\[\w*\d+\]', '', text)
    text = re.sub(r'\[note\s*\d+\]', '', text, flags=re.IGNORECASE)
    return text.strip()


def parse_position(pos_text):
    """标准化位置代码: 1GK→GK, 2DF→DF, 3MF→MF, 4FW→FW"""
    pos = clean_text(pos_text)
    # 移除数字前缀
    pos = re.sub(r'^\d+', '', pos)
    return pos.strip()


def get_squads_from_worldcup(year=2026):
    title = f"{year} FIFA World Cup squads"
    try:
        html, page_title = wikipedia_page(title)
    except Exception:
        search_results = wikipedia_search(title)
        if search_results:
            html, page_title = wikipedia_page(search_results[0])
        else:
            return {}

    soup = BeautifulSoup(html, 'html.parser')
    squads = {}

    excluded_sections = {
        'age', 'player representation by club', 'player representation by league system',
        'player representation by club confederation', 'average age of squads',
        'coach representation by country', 'statistics', 'notes', 'references',
        'external links', 'contents', 'see also'
    }

    for header in soup.find_all('h3'):
        team_name = header.get_text(strip=True)

        # 跳过统计章节
        if team_name.lower() in excluded_sections:
            continue

        table = header.find_next('table', class_='wikitable')
        if not table:
            continue

        players = []
        rows = table.find_all('tr')

        for row in rows[1:]:  # 跳过表头
            th = row.find('th')
            if not th:
                continue

            tds = row.find_all('td')
            if len(tds) < 6:
                continue

            # 提取各字段
            name = clean_text(th.get_text(strip=True))
            # 再次清洗 name（去掉 (captain) 后的残留）
            name = name.split('(')[0].strip()
            if not name:
                continue

            jersey_num = clean_text(tds[0].get_text(strip=True))
            position = parse_position(tds[1].get_text(strip=True))
            dob_text = clean_text(tds[2].get_text(strip=True))
            age = parse_age(dob_text)
            caps = clean_text(tds[3].get_text(strip=True))
            goals = clean_text(tds[4].get_text(strip=True))
            club = clean_text(tds[5].get_text(strip=True))

            players.append({
                'name': name,
                'jersey': jersey_num,
                'position': position,
                'age': age,
                'caps': int(caps) if caps.isdigit() else 0,
                'goals': int(goals) if goals.isdigit() else 0,
                'club': club,
            })

        if players:
            squads[team_name] = {
                'players': players,
                'player_count': len(players),
                # 计算球队统计
                'avg_age': round(sum(p['age'] for p in players if p['age']) /
                                 max(1, sum(1 for p in players if p['age'])), 1),
                'total_caps': sum(p['caps'] for p in players),
                'total_goals': sum(p['goals'] for p in players),
            }

    return squads


# ==============================
# Step 1: 爬取 + 保存
# ==============================
if __name__ == '__main__':
    print("正在从 Wikipedia 爬取 2026 世界杯球队名单...")
    all_squads = get_squads_from_worldcup(2026)

    print(f"共提取 {len(all_squads)} 支球队")

    # 保存
    output_file = 'squads_2026.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_squads, f, ensure_ascii=False, indent=2)

    print(f"已保存到 {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024:.1f} KB")

    # 预览
    print("\n" + "=" * 60)
    print("预览（前 10 支球队）")
    print("=" * 60)
    for i, (team, data) in enumerate(all_squads.items()):
        if i >= 10:
            break
        print(f"\n{team} ({data['player_count']} 名球员, "
              f"平均年龄 {data['avg_age']} 岁, "
              f"总出场 {data['total_caps']} 次, "
              f"总进球 {data['total_goals']})")
        for p in data['players'][:10]:
            print(f"  #{p['jersey']} {p['name']} | {p['position']} | "
                  f"年龄:{p['age']} | 出场:{p['caps']} | 进球:{p['goals']} | {p['club']}")
        if data['player_count'] > 3:
            print(f"  ... 还有 {data['player_count'] - 10} 名球员")
