import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import requests
from bs4 import BeautifulSoup

API_URL = 'http://en.wikipedia.org/w/api.php'
HEADERS = {'User-Agent': 'Mozilla/5.0'}

r = requests.get(API_URL, params={
    'action': 'parse', 'page': '2026 FIFA World Cup squads',
    'prop': 'text', 'format': 'json', 'redirects': 1
}, headers=HEADERS)
html = r.json()['parse']['text']['*']
soup = BeautifulSoup(html, 'html.parser')

# ============ 诊断 1：所有 H3 标签 ============
print('=' * 60)
print('诊断1：所有 H3 标签（前10个）')
print('=' * 60)
h3s = soup.find_all('h3')
print(f'共找到 {len(h3s)} 个 H3\n')
for h in h3s[:10]:
    text = h.get_text(strip=True)
    table = h.find_next('table', class_='wikitable')
    has_table = '✅有表格' if table else '❌无表格'
    print(f'  {has_table} | {text}')

# ============ 诊断2：第一个球队的详细解析 ============
print('\n' + '=' * 60)
print('诊断2：第一个 H3 的表格详细结构')
print('=' * 60)
h3 = h3s[0]
team_name = h3.get_text(strip=True)
print(f'球队名: {team_name}')

table = h3.find_next('table', class_='wikitable')
rows = table.find_all('tr')
print(f'表格共 {len(rows)} 行（含表头）')

# 表头
header_row = rows[0]
print(f'\n表头列: {[th.get_text(strip=True) for th in header_row.find_all("th")]}')

# 前3条数据行
print('\n前3条数据行:')
for i, row in enumerate(rows[1:4]):
    th = row.find('th')
    tds = row.find_all('td')
    th_text = th.get_text(strip=True) if th else '【NO TH】'
    td_texts = [td.get_text(strip=True)[:30] for td in tds]
    print(f'  Row {i+1}: th=[{th_text}] | tds={td_texts}')

# ============ 诊断3：模拟 get_cv.py 的逻辑 ============
print('\n' + '=' * 60)
print('诊断3：完全模拟 get_cv.py 的解析逻辑')
print('=' * 60)
squads = {}
for header in soup.find_all('h3'):
    team_name = header.get_text(strip=True)

    table = header.find_next('table', class_='wikitable')
    if not table:
        print(f'  ⚠️ {team_name}: 没有找到 wikitable，跳过')
        continue

    players = []
    rows = table.find_all('tr')
    for row in rows[1:]:
        th = row.find('th')
        if not th:
            continue
        name = th.get_text(strip=True)
        name = name.split('(')[0].strip()
        if name:
            players.append(name)

    if players:
        squads[team_name] = players
        print(f'  ✅ {team_name}: {len(players)} 名球员')
    else:
        print(f'  ❌ {team_name}: 表格有 {len(rows)} 行但提取到 0 名球员')

print(f'\n总共提取到 {len(squads)} 支球队')

# ============ 诊断4：查找 Argentina/France/Brazil ============
print('\n' + '=' * 60)
print('诊断4：查找目标球队')
print('=' * 60)
for target in ['Argentina', 'France', 'Brazil']:
    if target in squads:
        players = squads[target]
        print(f'  ✅ {target}: {len(players)} 名球员')
        for p in players[:3]:
            print(f'       - {p}')
    else:
        # 模糊搜索
        matches = [t for t in squads.keys() if target.lower() in t.lower()]
        print(f'  ❌ {target}: 不在 squads 中，模糊匹配: {matches}')

# ============ 诊断5：打印所有球队名 ============
print('\n' + '=' * 60)
print('诊断5：squads 中的所有球队名')
print('=' * 60)
for i, team in enumerate(sorted(squads.keys())):
    print(f'  {i+1}. {team} ({len(squads[team])} players)')
