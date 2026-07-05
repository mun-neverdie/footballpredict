"""
player_selector.py — 多因子评分 + 位置配额球员筛选。
被 build_kb.py 和 predict.py 共用。
"""

# ---- 精英俱乐部 / 强队分级 ----
ELITE_CLUBS = {
    # 五大联赛欧冠常客
    'Real Madrid', 'Barcelona', 'Atlético Madrid', 'Sevilla',
    'Manchester City', 'Arsenal', 'Liverpool', 'Manchester United',
    'Chelsea', 'Tottenham Hotspur', 'Newcastle United', 'Aston Villa',
    'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
    'Paris Saint-Germain', 'AS Monaco', 'Marseille', 'Lyon', 'Lille',
    'Inter Milan', 'AC Milan', 'Juventus', 'Napoli', 'Atalanta', 'Roma', 'Lazio',
    'Benfica', 'Porto', 'Sporting CP',
    'Ajax', 'PSV Eindhoven', 'Feyenoord',
}

STRONG_CLUBS = {
    # 南美豪门、欧洲二级联赛强队、沙特/MLS 大球会
    'Flamengo', 'Palmeiras', 'São Paulo', 'Santos', 'Fluminense',
    'River Plate', 'Boca Juniors', 'Racing Club',
    'Galatasaray', 'Fenerbahçe', 'Beşiktaş',
    'Olympiacos', 'Panathinaikos', 'AEK Athens',
    'Celtic', 'Rangers',
    'Shakhtar Donetsk', 'Dynamo Kyiv',
    'Red Bull Salzburg', 'Club Brugge', 'Anderlecht',
    'Al Hilal', 'Al Nassr', 'Al Ittihad', 'Al Ahli',
    'LA Galaxy', 'Inter Miami', 'Los Angeles FC',
    'Zenit Saint Petersburg', 'CSKA Moscow',
    'Copenhagen', 'Malmö FF',
    'Fenerbahçe', 'Trabzonspor',
    'Basel', 'Young Boys',
    'Olympiakos', 'PAOK',
    'Dinamo Zagreb', 'Red Star Belgrade',
    'Slavia Prague', 'Sparta Prague',
}


def score_player(player):
    """
    多因子球员重要度评分。
    输入: squads_2026.json 中的单个球员 dict
    返回: float 评分（越高越关键）
    """
    score = 0.0
    age = player.get('age', 28)
    caps = player.get('caps', 0)
    goals = player.get('goals', 0)
    pos = player.get('position', 'MF')
    club = player.get('club', '')

    # 1. 年龄红利（巅峰期 26-30，>34 自然衰减）
    if 26 <= age <= 30:
        score += 1.5
    elif 23 <= age <= 25 or age == 31:
        score += 1.0
    elif 32 <= age <= 33:
        score += 0.3
    # <23 或 >34 不加分

    # 2. 俱乐部竞技水平
    if club in ELITE_CLUBS:
        score += 1.5
    elif club in STRONG_CLUBS:
        score += 0.7

    # 3. 位置加权统计
    if pos == 'FW':
        score += goals * 0.15 + min(caps, 80) * 0.04
    elif pos == 'MF':
        score += goals * 0.08 + min(caps, 100) * 0.05
    elif pos == 'DF':
        score += min(caps, 120) * 0.05
    elif pos == 'GK':
        score += min(caps, 100) * 0.06

    return round(score, 2)


def select_key_players(players, n_fw=2, n_mf=2, n_df=2, n_gk=1):
    """
    按位置配额 + 评分选取关键球员。
    返回 7 人列表: 2FW + 2MF + 2DF + 1GK
    """
    by_pos = {'FW': [], 'MF': [], 'DF': [], 'GK': []}
    for p in players:
        pos = p.get('position', 'MF')
        if pos in by_pos:
            by_pos[pos].append(p)

    selected = []
    quotas = [('FW', n_fw), ('MF', n_mf), ('DF', n_df), ('GK', n_gk)]
    for pos, n in quotas:
        pool = sorted(by_pos.get(pos, []), key=score_player, reverse=True)
        selected.extend(pool[:n])

    return selected
