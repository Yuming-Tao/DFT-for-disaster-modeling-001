import requests
import pandas as pd
import time
import logging
import numpy as np
import os
from config import get_amap_key

# 缓存配置
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
ENTERPRISE_CACHE = os.path.join(CACHE_DIR, "enterprises.csv")
RELATIONS_CACHE = os.path.join(CACHE_DIR, "relations.csv")  # 移除社区缓存

# 暴雨灾害易受影响的行业分类及对应高德POI编码
RAIN_DISASTER_INDUSTRIES = {
    "交通运输业": ["010000", "020000", "030000", "040000", "050000"],
    "农业与养殖业": ["100000", "100100", "100200", "100300"],
    "建筑业": ["140000", "140100", "140200"],
    "旅游业": ["110000", "110100", "110200", "110300", "110400", "110500"],
    "零售业": ["060000", "060100", "060200", "060300", "060400", "060500", "060900", "061000"],
    "电力与能源": ["170000", "170100", "170200", "170300", "170400"],
    "制造业": ["150000", "150100", "150200", "150300", "150400", "150500"],
    "医疗与公共卫生": ["090000", "090100", "090200", "090300", "090400", "090500"]
}

# 行业关键词映射（用于文本描述匹配）
INDUSTRY_KEYWORDS = {
    "交通运输业": ["运输", "物流", "快递", "公交", "铁路", "航空", "港口", "货运"],
    "农业与养殖业": ["农业", "养殖", "种植", "农场", "畜牧", "渔业", "农产品"],
    "建筑业": ["建筑", "施工", "建设", "建材", "工程", "地产"],
    "旅游业": ["旅游", "酒店", "景区", "民宿", "旅行社", "餐饮"],
    "零售业": ["零售", "超市", "商场", "商店", "便利店", "销售", "汽配"],
    "电力与能源": ["电力", "能源", "发电", "供电", "燃气", "石油", "煤炭"],
    "制造业": ["制造", "工厂", "生产", "加工", "工业", "电子", "机械"],
    "医疗与公共卫生": ["医院", "医疗", "卫生", "药店", "诊所", "疾控", "健康"]
}

# 供应链四端口推断关键词（生产/制造/运输/消费）
PORT_KEYWORDS = {
    "生产端": ["农业", "养殖", "电力", "能源", "发电", "供水", "燃气", "采矿", "种植"],
    "制造端": ["制造", "工厂", "加工", "建筑", "工程", "建材", "装配", "生产"],
    "运输端": ["物流", "运输", "快递", "货运", "港口", "仓储", "配送", "托运"],
    "消费端": ["酒店", "商场", "超市", "景区", "医院", "零售", "学校", "博物院", "宾馆", "餐厅", "文旅", "住宿", "购物", "医疗", "教育"]
}


def infer_port(row):
    """根据名称和类型描述，推断企业所属供应链端口（确保port列生成）"""
    name = row['name'].lower()
    type_desc = row['type_code'].lower()
    for port, keywords in PORT_KEYWORDS.items():
        if any(kw in name or kw in type_desc for kw in keywords):
            return port
    return "消费端"  # 兜底，确保无遗漏


def classify_industry(type_input):
    """行业分类（兼容编码和文本描述）"""
    if not isinstance(type_input, str):
        return "其他行业"
    # POI编码匹配
    for industry, codes in RAIN_DISASTER_INDUSTRIES.items():
        if any(code in type_input for code in codes):
            return industry
    # 文本关键词匹配
    type_lower = type_input.lower()
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        if any(keyword in type_lower for keyword in keywords):
            return industry
    return "其他行业"


def fetch_enterprise_data(city='郑州', max_total_pages=20, use_cache=True, force_reclassify=False):
    """获取企业数据（确保port列存在）"""
    if use_cache and os.path.exists(ENTERPRISE_CACHE):
        logging.info(f"使用企业缓存数据: {ENTERPRISE_CACHE}")
        df = pd.read_csv(ENTERPRISE_CACHE)
        # 强制重新分类（含端口推断，修复port列缺失问题）
        if force_reclassify or 'port' not in df.columns:
            df['industry'] = df['type_code'].apply(classify_industry)
            df['port'] = df.apply(infer_port, axis=1)  # 确保生成port列
            df.to_csv(ENTERPRISE_CACHE, index=False, encoding='utf-8-sig')
            logging.info(f"已重新分类+推断端口并更新缓存: {ENTERPRISE_CACHE}")
        return df

    # 爬取逻辑
    all_codes = []
    for codes in RAIN_DISASTER_INDUSTRIES.values():
        all_codes.extend(codes)
    batch_size = 10
    base_url = "https://restapi.amap.com/v3/place/text"
    all_pois = []
    task_queue = []
    logging.info(f"重新获取企业数据: city={city}")

    for i in range(0, len(all_codes), batch_size):
        batch_codes = all_codes[i:i + batch_size]
        enterprise_types = "|".join(batch_codes)
        params = {
            'key': get_amap_key(),
            'city': city,
            'types': enterprise_types,
            'offset': 25,
            'page': 1,
            'extensions': 'all',
            'citylimit': True
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            total_pages = min(int(data.get('count', 0)) // 25, max_total_pages)
            for page in range(1, total_pages + 1):
                task_queue.append((batch_codes, page))
        else:
            logging.error(f"请求失败，状态码: {response.status_code}")
        time.sleep(5)

    for batch_codes, page in task_queue:
        enterprise_types = "|".join(batch_codes)
        params = {
            'key': get_amap_key(),
            'city': city,
            'types': enterprise_types,
            'offset': 25,
            'page': page,
            'extensions': 'all',
            'citylimit': True
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            pois = data.get('pois', [])
            all_pois.extend(pois)
        else:
            logging.error(f"请求失败，状态码: {response.status_code}")
        time.sleep(5)

    enterprises = []
    for poi in all_pois:
        location = poi.get('location', '')
        lng, lat = None, None
        if location and ',' in location:
            coords = location.split(',')
            if len(coords) >= 2:
                lng, lat = float(coords[0]), float(coords[1])
        industry = classify_industry(poi.get('type', ''))
        enterprises.append({
            'id': poi['id'],
            'name': poi['name'],
            'address': poi.get('address', ''),
            'type_code': poi.get('type', ''),
            'industry': industry,
            'port': infer_port({'name': poi['name'], 'type_code': poi.get('type', '')}),  # 强制生成port
            'lng': lng,
            'lat': lat,
            'pname': poi.get('pname', ''),
            'cityname': poi.get('cityname', ''),
            'adname': poi.get('adname', '')
        })

    df = pd.DataFrame(enterprises)
    df.to_csv(ENTERPRISE_CACHE, index=False, encoding='utf-8-sig')
    logging.info(f"企业数据（含端口）已保存至缓存: {ENTERPRISE_CACHE}")
    return df


def generate_realistic_relations(enterprises, use_cache=True, min_relations=300, max_relations=800):
    """基于四端口模型生成供应链关系"""
    if use_cache and os.path.exists(RELATIONS_CACHE):
        logging.info(f"使用供应链关系缓存数据: {RELATIONS_CACHE}")
        return pd.read_csv(RELATIONS_CACHE)

    valid_enterprises = enterprises.dropna(subset=['lng', 'lat']).copy()
    if len(valid_enterprises) < 2:
        logging.warning("有效企业不足，无法生成关系")
        return pd.DataFrame(columns=['from', 'to', 'weight', 'type'])

    # 按端口划分角色
    producers = valid_enterprises[valid_enterprises['port'] == '生产端']['id'].tolist()
    manufacturers = valid_enterprises[valid_enterprises['port'] == '制造端']['id'].tolist()
    transporters = valid_enterprises[valid_enterprises['port'] == '运输端']['id'].tolist()
    consumers = valid_enterprises[valid_enterprises['port'] == '消费端']['id'].tolist()

    # 确保每个端口至少有基础节点
    all_ids = valid_enterprises['id'].tolist()
    ROLE_MIN = 5
    for role_list in [producers, manufacturers, transporters]:
        if len(role_list) < ROLE_MIN:
            role_list += list(set(all_ids) - set(role_list))[:ROLE_MIN - len(role_list)]
    if len(consumers) < ROLE_MIN * 2:
        consumers += list(set(all_ids) - set(consumers))[:ROLE_MIN * 2 - len(consumers)]

    logging.info(f"端口分布：生产{len(producers)} | 制造{len(manufacturers)} | 运输{len(transporters)} | 消费{len(consumers)}")
    relations = []

    # 核心链路：生产→制造→运输→消费
    # 1. 生产→制造
    for p in producers:
        num = np.random.randint(2, 5)
        for m in np.random.choice(manufacturers, num, replace=False):
            relations.append({
                'from': p, 'to': m,
                'weight': np.random.uniform(3, 10),
                'type': 'prod_to_manu'
            })

    # 2. 制造→运输
    for m in manufacturers:
        num = np.random.randint(2, 5)
        for t in np.random.choice(transporters, num, replace=False):
            relations.append({
                'from': m, 'to': t,
                'weight': np.random.uniform(4, 12),
                'type': 'manu_to_trans'
            })

    # 3. 运输→消费
    for t in transporters:
        num = np.random.randint(3, 6)
        for c in np.random.choice(consumers, num, replace=False):
            relations.append({
                'from': t, 'to': c,
                'weight': np.random.uniform(2, 8),
                'type': 'trans_to_cons'
            })

    # 跨链路关系
    # 生产→运输
    if producers and transporters:
        for p in np.random.choice(producers, size=len(producers)//2, replace=False):
            num = np.random.randint(1, 3)
            for t in np.random.choice(transporters, num, replace=False):
                relations.append({
                    'from': p, 'to': t,
                    'weight': np.random.uniform(2, 7),
                    'type': 'prod_to_trans'
                })

    # 制造→消费
    if manufacturers and consumers:
        for m in np.random.choice(manufacturers, size=len(manufacturers)//2, replace=False):
            num = np.random.randint(1, 3)
            for c in np.random.choice(consumers, num, replace=False):
                relations.append({
                    'from': m, 'to': c,
                    'weight': np.random.uniform(3, 9),
                    'type': 'manu_to_cons'
                })

    # 生产→消费
    if producers and consumers:
        for p in np.random.choice(producers, size=len(producers)//2, replace=False):
            num = np.random.randint(1, 2)
            for c in np.random.choice(consumers, num, replace=False):
                relations.append({
                    'from': p, 'to': c,
                    'weight': np.random.uniform(1, 5),
                    'type': 'prod_to_cons'
                })

    # 同端口协作
    for port_group in [producers, manufacturers, transporters, consumers]:
        if len(port_group) < 2:
            continue
        for i in range(len(port_group)):
            src = port_group[i]
            num = np.random.randint(1, 3)
            peers = np.random.choice([p for p in port_group if p != src], num, replace=False)
            for peer in peers:
                relations.append({
                    'from': src, 'to': peer,
                    'weight': np.random.uniform(1, 4),
                    'type': 'cooperation'
                })

    # 其他行业兼容
    others = valid_enterprises[valid_enterprises['port'] == '其他']['id'].tolist()
    if others:
        for o in others:
            if np.random.random() < 0.5 and transporters:
                t = np.random.choice(transporters)
                relations.append({
                    'from': o, 'to': t,
                    'weight': np.random.uniform(1, 3),
                    'type': 'other_to_trans'
                })
            elif manufacturers:
                m = np.random.choice(manufacturers)
                relations.append({
                    'from': o, 'to': m,
                    'weight': np.random.uniform(1, 3),
                    'type': 'other_to_manu'
                })

    # 控制关系数量
    relations_df = pd.DataFrame(relations).drop_duplicates(subset=['from', 'to'])
    current_count = len(relations_df)

    if current_count < min_relations:
        need = min(min_relations - current_count, max_relations - current_count)
        all_ids = valid_enterprises['id'].tolist()
        existing_pairs = set((r['from'], r['to']) for _, r in relations_df.iterrows())
        added = 0
        while added < need:
            src = np.random.choice(all_ids)
            tgt = np.random.choice(all_ids)
            if src != tgt and (src, tgt) not in existing_pairs:
                relations_df.loc[len(relations_df)] = {
                    'from': src, 'to': tgt,
                    'weight': np.random.uniform(1, 3),
                    'type': 'supplementary'
                }
                existing_pairs.add((src, tgt))
                added += 1

    if len(relations_df) > max_relations:
        relations_df = relations_df.sample(max_relations, random_state=42)

    relations_df.to_csv(RELATIONS_CACHE, index=False, encoding='utf-8-sig')
    logging.info(f"供应链关系生成完成：{len(relations_df)}条")
    return relations_df


generate_mock_relations = generate_realistic_relations


def test_data_loading():
    """测试函数（验证端口和关系）"""
    logging.basicConfig(level=logging.INFO)
    enterprises = fetch_enterprise_data(use_cache=True, force_reclassify=True)
    print(f"企业数量：{len(enterprises)}")
    print(f"行业分布：\n{enterprises['industry'].value_counts()}")
    print(f"端口分布：\n{enterprises['port'].value_counts()}")
    relations = generate_realistic_relations(enterprises, use_cache=False)
    print(f"生成关系数量：{len(relations)}")
    assert 'port' in enterprises.columns, "端口列未生成"
    assert len(relations) >= 300, "关系数量不足"


if __name__ == "__main__":
    test_data_loading()