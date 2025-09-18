# -*- coding: utf-8 -*-
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from esda import Moran_Local
from libpysal.weights import Queen, W
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from shapely.geometry import Point

# ================ 全局与告警控制（精准屏蔽三方库残留提示） ================
warnings.filterwarnings("ignore", category=UserWarning, module="libpysal")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="esda")
warnings.filterwarnings("ignore", category=UserWarning, module="pyogrio.raw")

plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# 输出目录
OUT_DIR = "vulnerability_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ================ 读入与清洗出行数据 ================
def load_travel(travel_data_path: str) -> pd.DataFrame:
    print("===== 处理出行数据 =====")
    df = pd.read_csv(travel_data_path)
    print(f"成功加载出行数据，共{len(df)}条记录")

    required = ['all_pop', 'date', 'o_tid', 'd_tid', 'start_hour']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"出行数据缺少必要字段: {missing}")

    # 清洗
    for col in ['o_tid', 'd_tid']:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != '']
    df = df[df['o_tid'] != df['d_tid']]
    df['all_pop'] = pd.to_numeric(df['all_pop'], errors='coerce')
    df = df.dropna(subset=['all_pop'])

    # 时间
    df['date_str'] = df['date'].astype(str).str.slice(0, 8)
    df['timestamp'] = pd.to_datetime(df['date_str'], format="%Y%m%d") + \
                      pd.to_timedelta(df['start_hour'], unit='h')
    print(f"时间范围：{df['timestamp'].min()} 至 {df['timestamp'].max()}")
    return df


# ================ 计算网格-小时流量与时序指标 ================
def compute_time_series_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("\n===== 计算网格时序流量指标 =====")
    inflow = df.groupby(['d_tid', pd.Grouper(key='timestamp', freq='H')])['all_pop'].sum().reset_index()
    inflow.columns = ['grid_id', 'timestamp', 'inflow']

    outflow = df.groupby(['o_tid', pd.Grouper(key='timestamp', freq='H')])['all_pop'].sum().reset_index()
    outflow.columns = ['grid_id', 'timestamp', 'outflow']

    hourly = pd.merge(inflow, outflow, on=['grid_id', 'timestamp'], how='outer').fillna(0)
    hourly['total_flow'] = hourly['inflow'] + hourly['outflow']
    print(f"生成 {len(hourly)} 条网格-小时级流量记录")

    metrics = []
    region_max = hourly['total_flow'].max()

    for gid, g in hourly.groupby('grid_id'):
        g = g.sort_values('timestamp')
        flows = g['total_flow'].to_numpy()
        if flows.size < 24:
            continue

        flow_mean = float(np.mean(flows))
        flow_std = float(np.std(flows))
        max_flow = float(np.max(flows))

        # 负荷率（相对全域峰值）
        load_rate = max_flow / (region_max + 1e-8)

        # 峰/均比（按日）
        g['date'] = g['timestamp'].dt.date
        daily_peak = g.groupby('date')['total_flow'].max()
        daily_mean = g.groupby('date')['total_flow'].mean()
        peak_avg_ratio = float(np.mean(daily_peak / (daily_mean + 1e-8)))

        # 饱和频率（超过自身80分位）
        thr = np.percentile(flows, 80)
        saturation_freq = float(np.mean(flows > thr))

        # 变异系数、异常频次、斜率
        cv = flow_std / (flow_mean + 1e-8) if flow_mean != 0 else 0.0
        z_scores = np.abs(stats.zscore(flows)) if flows.size > 1 else np.zeros_like(flows)
        anomaly_freq = float(np.mean(z_scores > 3))
        hourly_changes = np.abs(np.diff(flows))
        slope_mean = float(np.mean(hourly_changes)) if hourly_changes.size > 0 else 0.0

        metrics.append({
            'grid_id': str(gid),
            'load_rate': load_rate,
            'peak_avg_ratio': peak_avg_ratio,
            'saturation_freq': saturation_freq,
            'cv': cv,
            'anomaly_freq': anomaly_freq,
            'slope_mean': slope_mean,
            'total_flow': float(np.sum(flows))
        })

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        raise ValueError("无有效时序流量数据")
    print(f"时序指标计算完成，有效网格数：{len(metrics_df)}")
    return metrics_df


# ================ 读网格矢量，合并指标，并执行空间分析 ================
def load_grid_and_spatial_analyze(grid_shp_path: str, metrics_df: pd.DataFrame) -> gpd.GeoDataFrame:
    print("\n===== 空间关联分析 =====")
    grid = gpd.read_file(grid_shp_path)
    print(f"加载网格空间数据，共{len(grid)}个网格")

    # 统一ID
    if 'TID' in grid.columns:
        grid = grid.rename(columns={'TID': 'grid_id'})
    grid['grid_id'] = grid['grid_id'].astype(str).str.strip()

    # 合并指标，仅保留有流量的网格
    grid = grid.merge(metrics_df, on='grid_id', how='left').fillna(0)
    grid = grid[grid['total_flow'] > 0].copy()
    print(f"有效网格数：{len(grid)}")

    # —— 在最大连通子图上计算局部莫兰，避免“未完全连通”的告警 —— #
    # 先构造权重（此处即便短暂触发内部检查，也在后续剔除非最大连通分量）
    w_full = Queen.from_dataframe(grid, use_index=True)
    # 找到最大连通分量的索引
    components = w_full.component_labels  # 每个观测所属连通分量标签
    comp_counts = pd.Series(components).value_counts()
    largest_label = comp_counts.idxmax()
    keep_index = [i for i, lbl in enumerate(components) if lbl == largest_label]
    grid_conn = grid.iloc[keep_index].copy()
    grid_conn.reset_index(drop=True, inplace=True)

    # 仅当样本足够且方差>0 时才计算 Local Moran
    vals = grid_conn['total_flow'].to_numpy(dtype=float)
    if len(grid_conn) >= 50 and np.nanstd(vals) > 1e-12:
        w = Queen.from_dataframe(grid_conn, use_index=True)  # 在最大连通子图上重建权重
        # ESDA 计算
        local_moran = Moran_Local(vals, w, permutations=999)
        moran_df = pd.DataFrame({
            'grid_id': grid_conn['grid_id'],
            'cluster_type': local_moran.q,
            'p_value': local_moran.p_sim
        })
        grid = grid.merge(moran_df, on='grid_id', how='left')
    else:
        grid['cluster_type'] = -1
        grid['p_value'] = 1.0

    # 填充
    grid['cluster_type'] = grid['cluster_type'].fillna(-1)
    grid['p_value'] = grid['p_value'].fillna(1.0)

    # 空间风险得分
    cluster_risk = {1: 1.0, 4: 0.8, 2: 0.3, 3: 0.1, -1: 0.5}
    grid['spatial_risk'] = grid.apply(
        lambda x: cluster_risk.get(x['cluster_type'], 0.5) if x['p_value'] < 0.05 else 0.5,
        axis=1
    )
    return grid


# ================ 计算 0–1 脆弱性指数 ================
def compute_vulnerability(grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # 指标 → 标准化列名映射（短名用于 Shapefile）
    indicator_mapping = {
        'load_rate': 'load_sc',        # 负荷率
        'peak_avg_ratio': 'peak_sc',   # 峰/均
        'saturation_freq': 'satur_sc', # 饱和频率
        'cv': 'cv_sc',                 # 变异系数
        'anomaly_freq': 'anom_sc',     # 异常频次
        'slope_mean': 'slope_sc'       # 波动斜率
    }
    ind_cols = list(indicator_mapping.keys())
    sc_cols = list(indicator_mapping.values())

    # 缺失/非数值兜底
    for c in ind_cols:
        grid_gdf[c] = pd.to_numeric(grid_gdf[c], errors='coerce').fillna(0.0)

    # 归一化到[0,1]
    scaler = MinMaxScaler()
    grid_gdf[sc_cols] = scaler.fit_transform(grid_gdf[ind_cols])

    # 权重（和=1）
    weights = {
        'load_sc': 0.2,
        'peak_sc': 0.15,
        'satur_sc': 0.15,
        'cv_sc': 0.15,
        'anom_sc': 0.15,
        'slope_sc': 0.1,
        'spatial_risk': 0.1
    }

    v = (grid_gdf['load_sc'] * weights['load_sc'] +
         grid_gdf['peak_sc'] * weights['peak_sc'] +
         grid_gdf['satur_sc'] * weights['satur_sc'] +
         grid_gdf['cv_sc'] * weights['cv_sc'] +
         grid_gdf['anom_sc'] * weights['anom_sc'] +
         grid_gdf['slope_sc'] * weights['slope_sc'] +
         grid_gdf['spatial_risk'] * weights['spatial_risk'])

    grid_gdf['vulnerability_index'] = np.clip(v, 0.0, 1.0)
    return grid_gdf


# ================ 保存（无告警：列名主动缩短） ================
def export_outputs(grid_gdf: gpd.GeoDataFrame, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # CSV（全字段，含完整列名）
    csv_path = os.path.join(out_dir, "vulnerability_normalized.csv")
    grid_gdf.drop(columns='geometry').to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n脆弱性结果（0-1区间）已保存至 {csv_path}")

    # Shapefile（提前缩短字段名，避免任何字段名>10字符告警）
    shp_path = os.path.join(out_dir, "vulnerability_spatial.shp")
    gdf_shp = grid_gdf.copy()
    rename_short = {
        'peak_avg_ratio': 'peak_avg_r',     # 10
        'saturation_freq': 'saturation',    # 10
        'anomaly_freq': 'anomaly_fr',       # 10
        'cluster_type': 'cluster_ty',       # 10
        'spatial_risk': 'spatial_rk',       # 10
        'vulnerability_index': 'vulnerabil' # 10
    }
    gdf_shp = gdf_shp.rename(columns=rename_short)
    gdf_shp.to_file(shp_path, encoding='utf-8')
    print(f"空间数据已保存至 {shp_path}")

    # GeoJSON（保留完整列名）
    geojson_path = os.path.join(out_dir, "vulnerability_spatial.geojson")
    grid_gdf.to_file(geojson_path, driver="GeoJSON", encoding='utf-8')
    print("GeoJSON 已保存")

    # 地图（英文图例与标题，锁定 0–1 色标）
    fig, ax = plt.subplots(figsize=(14, 10))
    grid_gdf.plot(
        column='vulnerability_index',
        cmap='OrRd',
        legend=True,
        legend_kwds={'label': 'Vulnerability Index (0–1)', 'orientation': 'horizontal'},
        ax=ax,
        edgecolor='0.8',
        linewidth=0.2,
        vmin=0, vmax=1
    )
    ax.set_title('Grid Vulnerability Distribution', fontsize=16)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "vulnerability_map.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("脆弱性分布图已保存")


# ================ 主流程 ================
def calculate_vulnerability_with_time(travel_data_path: str, grid_shp_path: str) -> gpd.GeoDataFrame:
    df = load_travel(travel_data_path)
    metrics_df = compute_time_series_metrics(df)
    grid_gdf = load_grid_and_spatial_analyze(grid_shp_path, metrics_df)
    grid_gdf = compute_vulnerability(grid_gdf)
    export_outputs(grid_gdf, OUT_DIR)
    print("\n前5个网格的脆弱性指数：")
    print(grid_gdf[['grid_id', 'vulnerability_index']].head())
    return grid_gdf


if __name__ == "__main__":
    TRAVEL_DATA_PATH = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\结果数据\人口出行.csv"
    GRID_SHP_PATH = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\郑州网格shp\郑州分析网格.shp"

    _ = calculate_vulnerability_with_time(TRAVEL_DATA_PATH, GRID_SHP_PATH)
