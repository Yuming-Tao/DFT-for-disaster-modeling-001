import pandas as pd
import numpy as np
import networkx as nx
import logging
from sklearn.preprocessing import MinMaxScaler
from geo_processor import calculate_distance_matrix

logger = logging.getLogger(__name__)


def build_supply_network(enterprises_df: pd.DataFrame, relations_df: pd.DataFrame) -> nx.DiGraph:
    """
    构建供应链网络（有向加权图）
    节点属性：name, lng, lat
    边属性：weight
    """
    G = nx.DiGraph()

    # 仅使用坐标有效的企业
    valid_enterprises = enterprises_df.dropna(subset=['lng', 'lat'])
    valid_ids = set(valid_enterprises['id'])

    # 加节点
    for _, row in valid_enterprises.iterrows():
        G.add_node(row['id'], name=row.get('name', ''), lng=row['lng'], lat=row['lat'])

    # 加边（过滤无效端点）
    valid_edges = 0
    for _, r in relations_df.iterrows():
        u, v, w = r.get('from'), r.get('to'), r.get('weight', 1.0)
        if u in valid_ids and v in valid_ids and u != v:
            G.add_edge(u, v, weight=float(w))
            valid_edges += 1

    logger.info(f"网络构建完成：{len(G.nodes)}节点，{valid_edges}有效边")
    return G


def calculate_pagerank(G: nx.DiGraph) -> dict:
    """
    计算 PageRank 重要度。如果无边，则为每个节点分配均等重要度。
    """
    if len(G.nodes) == 0:
        return {}
    if len(G.edges) == 0:
        logger.warning("网络无有效边，为节点分配均等重要度")
        return {n: 1.0 / len(G.nodes) for n in G.nodes}

    # 稀疏网络降低 alpha，稠密网络提高 alpha
    alpha = 0.7 if len(G.edges) < len(G.nodes) * 2 else 0.85
    pr = nx.pagerank(G, weight='weight', alpha=alpha)
    return pr


def calculate_vulnerability(
    enterprises_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    phi_e_df: pd.DataFrame = None,
    default_phi_e: float = None,
    params: dict = None
) -> pd.DataFrame:
    """
    计算企业脆弱性并归一化到 [0,1]。
    —— 关键增强点 ——
    1) 距离：指数衰减 -> 幂律衰减 (1 + d/L)^(-beta)
    2) 重要度：PageRank 先 MinMax 再加幂 gamma
    3) 经济/行业因子：未提供时按 port 给缺省权重（生产>运输>制造>消费）
    4) 受体重要度放大：w_ij *= (1 + lambda_import * I_j_norm)

    返回：包含原字段 + I_i, I_norm, phi_E, vulnerability（[0,1]）
    """

    # -------------------------
    # 超参数（可通过 params 覆盖）
    # -------------------------
    _params = {
        "L_km": 10.0,        # 距离尺度（km）
        "beta": 1.5,         # 幂律指数
        "gamma": 0.8,        # PageRank 重要度幂
        "lambda_import": 0.5,  # 受体重要度放大系数
        "normalize": True    # 是否归一化到 [0,1]
    }
    if params:
        _params.update(params)

    L_km = float(_params["L_km"])
    beta = float(_params["beta"])
    gamma = float(_params["gamma"])
    lambda_import = float(_params["lambda_import"])
    do_normalize = bool(_params["normalize"])

    # -------------------------
    # 预处理：有效企业 + 重置索引与坐标序
    # -------------------------
    valid_enterprises = enterprises_df.dropna(subset=['lng', 'lat']).copy()
    valid_enterprises = valid_enterprises.reset_index(drop=True)

    if len(valid_enterprises) < 2:
        logger.warning("有效企业不足（<2家），返回0脆弱性")
        out = valid_enterprises.copy()
        out['I_i'] = 0.0
        out['I_norm'] = 0.0
        out['phi_E'] = 1.0
        out['vulnerability'] = 0.0
        return out

    # -------------------------
    # 网络与 PageRank
    # -------------------------
    G = build_supply_network(valid_enterprises, relations_df)
    pagerank = calculate_pagerank(G)
    valid_enterprises['I_i'] = valid_enterprises['id'].map(pagerank).fillna(0.0)

    # PageRank -> MinMax 到 [0,1]，再加幂 gamma
    I_vals = valid_enterprises['I_i'].values.reshape(-1, 1)
    if np.allclose(I_vals, 0.0):
        logger.warning("所有节点重要度为0，脆弱性将全为0")
        valid_enterprises['I_norm'] = 0.0
    else:
        scaler_I = MinMaxScaler()
        I_norm = scaler_I.fit_transform(I_vals).flatten()
        I_norm = np.power(I_norm, gamma)
        valid_enterprises['I_norm'] = I_norm

    # -------------------------
    # 关系权重：出边标准化 w_ij
    # -------------------------
    valid_relations = relations_df[
        relations_df['from'].isin(valid_enterprises['id'])
        & relations_df['to'].isin(valid_enterprises['id'])
    ].copy()

    if len(valid_relations) == 0 or valid_enterprises['I_norm'].sum() == 0:
        logger.warning("无有效供应链关系或重要度全为0，脆弱性将全为0")
        valid_enterprises['phi_E'] = 1.0
        valid_enterprises['vulnerability'] = 0.0
        return valid_enterprises

    out_degree = valid_relations.groupby('from')['weight'].sum().reset_index()
    out_degree.columns = ['company_id', 'total_out']
    valid_relations = valid_relations.merge(out_degree, left_on='from', right_on='company_id', how='left')
    valid_relations['w_ij'] = np.where(
        valid_relations['total_out'] > 0,
        valid_relations['weight'] / valid_relations['total_out'],
        0.0
    )

    # -------------------------
    # 距离矩阵
    # -------------------------
    coords = list(zip(valid_enterprises['lng'].astype(float), valid_enterprises['lat'].astype(float)))
    dist_matrix = calculate_distance_matrix(coords)
    n = len(valid_enterprises)
    if dist_matrix.shape != (n, n):
        logger.warning(f"距离矩阵形状异常：{dist_matrix.shape}，期望 {(n, n)}，将填充/截断到匹配大小")
        D = np.zeros((n, n), dtype=np.float64)
        r = min(n, dist_matrix.shape[0])
        c = min(n, dist_matrix.shape[1])
        D[:r, :c] = dist_matrix[:r, :c]
        dist_matrix = D

    # -------------------------
    # 经济因子 phi_E
    # -------------------------
    if phi_e_df is not None and 'id' in phi_e_df.columns and 'phi_E' in phi_e_df.columns:
        valid_enterprises = valid_enterprises.merge(
            phi_e_df[['id', 'phi_E']], on='id', how='left'
        )

    if 'phi_E' not in valid_enterprises.columns:
        if default_phi_e is not None:
            valid_enterprises['phi_E'] = float(default_phi_e)
        else:
            # 按供应链端口的缺省权重
            port_weight = {
                '生产端': 1.20,
                '运输端': 1.15,
                '制造端': 1.10,
                '消费端': 1.00
            }
            valid_enterprises['phi_E'] = valid_enterprises.get('port', pd.Series(['消费端'] * len(valid_enterprises))) \
                .map(port_weight).fillna(1.00)

    # -------------------------
    # 计算脆弱性（幂律距离衰减 + 受体重要度放大）
    # -------------------------
    def distance_decay(d_km: float, L: float = L_km, b: float = beta) -> float:
        # (1 + d/L)^(-beta)
        return np.power(1.0 + (d_km / L), -b)

    # 快速索引
    id_to_idx = {row['id']: i for i, row in valid_enterprises.reset_index().iterrows()}
    I_norm_arr = valid_enterprises['I_norm'].to_numpy(dtype=float)
    phi_arr = valid_enterprises['phi_E'].to_numpy(dtype=float)

    vulnerability = np.zeros(n, dtype=np.float64)

    # 为了效率，先把 successors 索引好
    rel_by_from = valid_relations.groupby('from')

    for i in range(n):
        i_id = valid_enterprises.at[i, 'id']
        i_I = I_norm_arr[i]
        if i_I <= 0:
            continue

        if i_id not in rel_by_from.groups:
            continue
        successors = rel_by_from.get_group(i_id)

        total = 0.0
        for _, edge in successors.iterrows():
            j_id = edge['to']
            j_idx = id_to_idx.get(j_id)
            if j_idx is None:
                continue

            j_phi = phi_arr[j_idx]
            j_I = I_norm_arr[j_idx]
            d_ij = float(dist_matrix[i, j_idx])

            decay = distance_decay(d_ij)
            w_ij = float(edge['w_ij']) * (1.0 + lambda_import * j_I)
            total += i_I * w_ij * j_phi * decay

        vulnerability[i] = total

    # -------------------------
    # 归一化
    # -------------------------
    if do_normalize:
        vmin, vmax = vulnerability.min(), vulnerability.max()
        if vmax > vmin:
            vulnerability = MinMaxScaler((0.0, 1.0)).fit_transform(vulnerability.reshape(-1, 1)).flatten()
        else:
            vulnerability = np.zeros_like(vulnerability)

    valid_enterprises['vulnerability'] = vulnerability

    logger.info(
        "脆弱性计算完成（已归一化到[0,1]）：" if do_normalize else "脆弱性计算完成（未归一化）："
        + f"\n原/当前范围 [{float(vulnerability.min()):.6f}, {float(vulnerability.max()):.6f}]"
        + f"\n均值 {float(np.mean(vulnerability)):.4f}，最大值 {float(np.max(vulnerability)):.4f}"
    )

    return valid_enterprises


# -------------------------
# 简单自测
# -------------------------
def test_vulnerability():
    logging.basicConfig(level=logging.INFO)

    enterprises = pd.DataFrame({
        'id': ['C1', 'C2', 'C3', 'C4'],
        'name': ['A厂', 'B厂', 'C物流', 'D商超'],
        'lng': [113.64, 113.65, 113.66, 113.70],
        'lat': [34.75, 34.76, 34.77, 34.79],
        'port': ['生产端', '制造端', '运输端', '消费端']
    })

    relations = pd.DataFrame({
        'from': ['C1', 'C2', 'C2', 'C3'],
        'to':   ['C2', 'C3', 'C4', 'C4'],
        'weight': [3.0, 4.0, 2.0, 1.0]
    })

    result = calculate_vulnerability(enterprises, relations)
    print(result[['id', 'port', 'I_i', 'I_norm', 'phi_E', 'vulnerability']])

    # 验证区间
    assert (result['vulnerability'] >= 0).all() and (result['vulnerability'] <= 1).all(), \
        "vulnerability 超出 [0,1]！"
    print("测试通过")


if __name__ == "__main__":
    test_vulnerability()
