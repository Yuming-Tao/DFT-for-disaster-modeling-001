# -*- coding: utf-8 -*-
"""
Vulnerability of floating population — Map (green) + ECDF vs Model CDFs
- 布局：16:9 画布，四边 10% 页边距；[地图 | 色标 | 曲线] 三列紧贴
- 颜色：保持绿色系 POP_GREEN_CMAP 不变；仅做数值“对比度拉伸”（分位裁剪+γ增强+最浅色保底）
- 曲线：ECDF 与三种模型 CDF（Beta / 截断正态 TN / 截断指数 TExp）
- 字体：Times New Roman，标题 21pt，其余 10.5pt
- 移除：指北针与比例尺
- 缓存：若存在 vulnerability_results/_cache/vuln_cache.gpkg 则直接读取
"""
import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy import optimize as spo
from libpysal.weights import Queen
from esda import Moran_Local

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter

# ---------------------- 显示样式（16:9、10% 页边距、Times New Roman） ----------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 21,
    "axes.unicode_minus": False,
})

# ---------------------- 路径与缓存 ----------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "vulnerability_results")
CACHE_DIR  = os.path.join(OUTPUT_DIR, "_cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

CACHE_DATA = os.path.join(CACHE_DIR, "vuln_cache.gpkg")
CACHE_META = os.path.join(CACHE_DIR, "vuln_cache_meta.json")

# ---------------------- 绿色渐变（保持不变） ----------------------
POP_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "PopGreenLinear",
    ["#EAF7E6", "#CBEFC7", "#A2E29D", "#6BCD6A", "#38B54A", "#1C8E34", "#0F5F22"]
)

# ---------------------- 描述统计 & 拟合 ----------------------
def _summary(v: np.ndarray):
    x = pd.Series(v, dtype=float)
    x = x[np.isfinite(x) & (x >= 0) & (x <= 1)].to_numpy()
    N = int(x.size)
    if N == 0:
        return {"N": 0}
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return dict(
        N=N, mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)) if N>1 else 0.0,
        median=float(q2), Q1=float(q1), Q3=float(q3),
        IQR=float(q3-q1), p01=float(np.quantile(x, 0.01)), p99=float(np.quantile(x, 0.99)),
    )

def _fit_beta(v):
    vv = np.asarray(v)
    vv = vv[(vv>0)&(vv<1)&np.isfinite(vv)]
    if vv.size < 10: return None, None, vv.size
    a, b, _, _ = stats.beta.fit(vv, floc=0, fscale=1)
    return float(a), float(b), int(vv.size)

# ---- 截断正态 TN(μ,σ) on [0,1] ----
def _fit_truncnorm_mle(y):
    y = np.asarray(y); y = y[(y>=0)&(y<=1)&np.isfinite(y)]
    if y.size < 5: return None, None
    mu0, sd0 = float(np.mean(y)), float(np.std(y, ddof=1) if y.size>1 else 0.1)
    sd0 = max(sd0, 1e-3)
    def nll(theta):
        mu, log_s = theta[0], theta[1]
        s = np.exp(log_s)
        z0 = (0 - mu)/s; z1 = (1 - mu)/s
        denom = stats.norm.cdf(z1) - stats.norm.cdf(z0)
        if denom <= 1e-12 or s <= 0: return 1e12
        z = (y - mu)/s
        ll = -0.5*np.sum(z**2) - y.size*np.log(s) - 0.5*y.size*np.log(2*np.pi) - y.size*np.log(denom)
        return -ll
    res = spo.minimize(nll, x0=np.array([mu0, np.log(sd0)]),
                       method="L-BFGS-B", bounds=[(-5, 6), (np.log(1e-4), np.log(2))])
    if not res.success: return None, None
    mu, s = float(res.x[0]), float(np.exp(res.x[1]))
    return mu, s

def _truncnorm_cdf(y, mu, s):
    z0 = (0 - mu)/s; z1 = (1 - mu)/s
    denom = stats.norm.cdf(z1) - stats.norm.cdf(z0)
    Fy = (stats.norm.cdf((y-mu)/s) - stats.norm.cdf(z0)) / max(denom, 1e-15)
    return np.clip(Fy, 0, 1)

# ---- 截断指数 TExp(λ) on [0,1] ----
def _fit_truncexp_mle(y):
    y = np.asarray(y); y = y[(y>=0)&(y<=1)&np.isfinite(y)]
    if y.size < 5: return None
    n, S = y.size, float(np.sum(y))
    def g(lam):
        lam = float(lam)
        if lam <= 0: return np.inf
        return n/lam - S + n*np.exp(-lam)/(1 - np.exp(-lam))
    try:
        lam = spo.brentq(g, 1e-6, 1e6, maxiter=1000)
        return float(lam)
    except Exception:
        return None

def _truncexp_cdf(y, lam):
    return (1 - np.exp(-lam*y)) / (1 - np.exp(-lam))

# KS & CvM for a given theoretical CDF
def _ks_and_cvm(sorted_y, F_theory):
    n = sorted_y.size
    F = F_theory(sorted_y)
    emp = np.arange(1, n+1) / n
    ks = float(np.max(np.abs(emp - F)))
    cvm = float(1.0/(12*n) + np.sum((F - (2*np.arange(1,n+1)-1)/(2*n))**2))
    return ks, cvm

# ---------------------- 对比度拉伸（不改色系，只改映射强度） ----------------------
def green_contrast(series: pd.Series, q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = x.quantile(q_low), x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(hi - lo) or hi <= lo:
            return pd.Series(np.zeros(len(x)), index=x.index)
    y = (x.clip(lo, hi) - lo) / (hi - lo)
    y = np.power(y, gamma)
    y = min_tint + (1.0 - min_tint) * y
    return pd.Series(y).fillna(min_tint).clip(0, 1)

# ---------------------- 主计算（若无缓存） ----------------------
def calculate_vulnerability_with_time(travel_data_path, grid_shp_path):
    print("===== 处理出行数据 =====")
    try:
        travel_df = pd.read_csv(travel_data_path)
        print(f"成功加载出行数据，共{len(travel_df)}条记录")
    except Exception as e:
        print(f"出行数据加载失败：{str(e)}"); return None

    required_cols = ['all_pop', 'date', 'o_tid', 'd_tid', 'start_hour']
    if not set(required_cols).issubset(travel_df.columns):
        print(f"错误：缺少必要字段 {set(required_cols) - set(travel_df.columns)}"); return None

    for col in ['o_tid','d_tid']:
        travel_df[col] = travel_df[col].astype(str).str.strip()
        travel_df = travel_df[travel_df[col] != '']
    travel_df = travel_df[travel_df['o_tid'] != travel_df['d_tid']]
    travel_df['all_pop'] = pd.to_numeric(travel_df['all_pop'], errors='coerce')
    travel_df = travel_df.dropna(subset=['all_pop'])

    travel_df['date_str'] = travel_df['date'].astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")
    travel_df['timestamp'] = pd.to_datetime(travel_df['date_str']) + pd.to_timedelta(travel_df['start_hour'], unit='h')
    print(f"时间范围：{travel_df['timestamp'].min()} 至 {travel_df['timestamp'].max()}")

    print("\n===== 计算网格时序流量指标 =====")
    inflow_hourly = (travel_df.groupby(['d_tid', pd.Grouper(key='timestamp', freq='H')])['all_pop']
                     .sum().reset_index().rename(columns={'d_tid': 'grid_id', 'all_pop': 'inflow'}))
    outflow_hourly = (travel_df.groupby(['o_tid', pd.Grouper(key='timestamp', freq='H')])['all_pop']
                      .sum().reset_index().rename(columns={'o_tid': 'grid_id', 'all_pop': 'outflow'}))
    hourly_flow = pd.merge(inflow_hourly, outflow_hourly, on=['grid_id','timestamp'], how='outer').fillna(0)
    hourly_flow['total_flow'] = hourly_flow['inflow'] + hourly_flow['outflow']
    print(f"生成 {len(hourly_flow)} 条网格-小时级流量记录")

    grid_metrics = []; grids = hourly_flow['grid_id'].unique()
    region_max_flow = hourly_flow['total_flow'].max()
    for gid in grids:
        gh = hourly_flow[hourly_flow['grid_id']==gid].sort_values('timestamp')
        f = gh['total_flow'].values
        if len(f) < 24: continue
        flow_mean, flow_std, max_flow = float(np.mean(f)), float(np.std(f)), float(np.max(f))
        load_rate = max_flow / (region_max_flow + 1e-8)
        gh['date'] = gh['timestamp'].dt.date
        daily_peak = gh.groupby('date')['total_flow'].max()
        daily_mean = gh.groupby('date')['total_flow'].mean()
        peak_avg_ratio = float(np.mean(daily_peak / (daily_mean + 1e-8)))
        grid_flow_80p = float(np.percentile(f, 80))
        saturation_freq = float(np.mean(f > grid_flow_80p))
        cv = (flow_std/(flow_mean+1e-8)) if flow_mean != 0 else 0.0
        z = np.abs(stats.zscore(f)) if len(f)>1 else np.zeros_like(f)
        anomaly_freq = float(np.mean(z>3))
        hc = np.abs(np.diff(f)); slope_mean = float(np.mean(hc)) if hc.size>0 else 0.0
        grid_metrics.append(dict(grid_id=gid, load_rate=load_rate, peak_avg_ratio=peak_avg_ratio,
                                 saturation_freq=saturation_freq, cv=cv, anomaly_freq=anomaly_freq,
                                 slope_mean=slope_mean, total_flow=float(np.sum(f))))
    grid_metrics_df = pd.DataFrame(grid_metrics)
    if grid_metrics_df.empty:
        print("错误：无有效时序流量数据"); return None
    print(f"时序指标计算完成，有效网格数：{len(grid_metrics_df)}")

    print("\n===== 空间关联分析 =====")
    try:
        grid_gdf = gpd.read_file(grid_shp_path)
        print(f"加载网格空间数据，共{len(grid_gdf)}个网格")
        if 'TID' in grid_gdf.columns: grid_gdf = grid_gdf.rename(columns={'TID':'grid_id'})
        grid_gdf['grid_id'] = grid_gdf['grid_id'].astype(str).str.strip()
        grid_gdf = pd.merge(grid_gdf, grid_metrics_df, on='grid_id', how='left').fillna(0)
        grid_gdf = grid_gdf[grid_gdf['total_flow']>0].reset_index(drop=True)
        valid_count = len(grid_gdf); print(f"有效网格数：{valid_count}")

        w_full = Queen.from_dataframe(grid_gdf, use_index=True)
        islands = w_full.islands or []
        if islands: print(f"发现 {len(islands)} 个孤立网格，已特殊处理")

        if valid_count - len(islands) > 50:
            non_island = grid_gdf.drop(index=islands).copy()
            w = Queen.from_dataframe(non_island, use_index=True); w.transform="r"
            y = non_island['total_flow'].to_numpy()
            if np.nanstd(y) > 0:
                lm = Moran_Local(y, w, permutations=999)
                moran_df = pd.DataFrame({'grid_id': non_island['grid_id'].values,
                                         'cluster_type': lm.q, 'p_value': lm.p_sim})
                grid_gdf = grid_gdf.merge(moran_df, on='grid_id', how='left')
            else:
                grid_gdf['cluster_type']=-1; grid_gdf['p_value']=1.0
        else:
            grid_gdf['cluster_type']=-1; grid_gdf['p_value']=1.0

        grid_gdf['cluster_type']=grid_gdf['cluster_type'].fillna(-1)
        grid_gdf['p_value']=grid_gdf['p_value'].fillna(1.0)
        cluster_risk={1:1.0,4:0.8,2:0.3,3:0.1,-1:0.5}
        grid_gdf['spatial_risk']=grid_gdf.apply(
            lambda x: cluster_risk.get(x['cluster_type'],0.5) if x['p_value']<0.05 else 0.5, axis=1
        )
    except Exception as e:
        print(f"空间数据处理失败：{str(e)}"); return None

    # 归一化 & 指数
    indicator_cols = ['load_rate','peak_avg_ratio','saturation_freq','cv','anomaly_freq','slope_mean']
    scaled_cols    = ['load_sc','peak_sc','satur_sc','cv_sc','anom_sc','slope_sc']
    scaler = MinMaxScaler(); grid_gdf[scaled_cols] = scaler.fit_transform(grid_gdf[indicator_cols])
    weights={'load_sc':0.2,'peak_sc':0.15,'satur_sc':0.15,'cv_sc':0.15,'anom_sc':0.15,'slope_sc':0.1,'spatial_risk':0.1}
    grid_gdf['vulnerability_index'] = (
        grid_gdf['load_sc']*weights['load_sc'] + grid_gdf['peak_sc']*weights['peak_sc'] +
        grid_gdf['satur_sc']*weights['satur_sc'] + grid_gdf['cv_sc']*weights['cv_sc'] +
        grid_gdf['anom_sc']*weights['anom_sc'] + grid_gdf['slope_sc']*weights['slope_sc'] +
        grid_gdf['spatial_risk']*weights['spatial_risk']
    ).clip(0,1)

    # 写缓存
    try:
        grid_gdf.to_file(CACHE_DATA, layer="vuln", driver="GPKG")
        meta={"created_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"n_rows":int(len(grid_gdf))}
        with open(CACHE_META,"w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)
        print("✅ 缓存已更新")
    except Exception as e:
        print(f"缓存写入失败：{e}")
    return grid_gdf

# ---------------------- 导出基础文件 ----------------------
def export_outputs(gdf: gpd.GeoDataFrame):
    csv_out = os.path.join(OUTPUT_DIR, "vulnerability_normalized.csv")
    gdf.drop(columns=[c for c in gdf.columns if c=='geometry']).to_csv(csv_out, index=False, encoding='utf-8-sig')
    print(f"\n脆弱性结果已保存：{csv_out}")
    shp_out = os.path.join(OUTPUT_DIR, "vulnerability_spatial.shp"); gdf.to_file(shp_out, encoding='utf-8')
    print(f"Shapefile 已保存：{shp_out}")
    gpkg_out = os.path.join(OUTPUT_DIR, "vulnerability_spatial.gpkg"); gdf.to_file(gpkg_out, layer="vuln", driver="GPKG")
    print(f"GPKG 已保存：{gpkg_out}")

# ---------------------- 可视化（窄色标/窄曲线；地图高度对齐色标） ----------------------
def visualize_map_ecdf_models(
    gdf_result: gpd.GeoDataFrame,
    output_path=os.path.join(OUTPUT_DIR, "vulnerability_map_ecdf_models_green_compact.png"),
    title="Vulnerabilty of floating population",
    dpi=300,
    add_trunc_normal=True,
    add_trunc_exponential=True,
    # 对比度拉伸参数（不改变色系）
    q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06
):
    # 数据
    v_raw = pd.to_numeric(gdf_result['vulnerability_index'], errors='coerce').fillna(0.0).clip(0,1).to_numpy()
    if v_raw.size == 0: raise ValueError("No valid vulnerability_index values.")
    summ = _summary(v_raw)
    a, b, n_eff = _fit_beta(v_raw)

    # 仅用于着色的“可视化强度”（绿色系不变）
    v_vis = green_contrast(gdf_result['vulnerability_index'], q_low, q_high, gamma, min_tint).to_numpy()

    # 画布：三列 GridSpec（把色标轴与曲线图画“更窄”）
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.suptitle(title, fontsize=21, y=0.965)
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        # 地图 | 色标 | 右侧曲线  （比之前更窄：1.1 与 7）
        width_ratios=[44, 1.1, 7],
        left=0.10, right=0.90, bottom=0.10, top=0.90, wspace=0.015
    )
    ax_map   = fig.add_subplot(gs[0, 0])
    cax      = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2], sharey=cax)

    # 左图：地图（保持 POP_GREEN_CMAP）
    gdf = gdf_result.copy(); gdf['vuln_vis'] = v_vis
    gdf.plot(column='vuln_vis', ax=ax_map, cmap=POP_GREEN_CMAP, vmin=0, vmax=1,
             edgecolor='white', linewidth=0.2, alpha=0.95,
             legend=True, legend_kwds={'label':'', 'cax':cax})
    ax_map.set_axis_off()
    ax_map.margins(0, 0)

    # —— 强制三者等高：对齐地图/色标/曲线的上下边界 —— #
    fig.canvas.draw()
    pos_cax   = cax.get_position()
    pos_map   = ax_map.get_position()
    pos_right = ax_right.get_position()
    ax_map.set_position([pos_map.x0, pos_cax.y0, pos_map.width,  pos_cax.height])
    ax_right.set_position([pos_right.x0, pos_cax.y0, pos_right.width, pos_cax.height])

    # 色标（左刻度）
    ticks=[0.00,0.25,0.50,0.75,1.00]
    cax.set_ylim(0,1)
    cax.yaxis.set_major_locator(FixedLocator(ticks))
    cax.yaxis.set_major_formatter(FixedFormatter([f"{t:.2f}" for t in ticks]))
    cax.set_ylabel("Vulnerability (stretched 0–1)", fontsize=10.5, labelpad=8)
    cax.yaxis.set_ticks_position('left'); cax.yaxis.set_label_position('left')
    cax.tick_params(axis='y', labelsize=10.5)

    # 右图：ECDF + 模型CDF
    for sp in ('left','right'): ax_right.spines[sp].set_visible(False)
    ax_right.yaxis.set_visible(False)
    ax_right.set_title("ECDF vs Model CDFs", fontsize=10.5, pad=6)

    s = np.sort(v_raw); n = s.size
    ecdf_x = (np.arange(1,n+1))/n
    h_ecdf = ax_right.plot(ecdf_x, s, drawstyle="steps-post", color="#2ca02c", lw=1.9, label="ECDF")[0]
    handles=[h_ecdf]; labels=["ECDF"]

    models = []
    if (a is not None) and (b is not None):
        models.append(dict(
            name="Beta", color="#0b6e1d", ls="-", lw=2.0,
            label=f"Beta  (a={a:.2f},  b={b:.2f})",
            cdf=lambda yy: stats.beta.cdf(yy, a, b)
        ))
    if add_trunc_normal:
        mu_n, s_n = _fit_truncnorm_mle(v_raw)
        if (mu_n is not None) and (s_n is not None):
            models.append(dict(
                name="TN", color="#ff7f0e", ls="--", lw=1.5,
                label=f"TN   (μ={mu_n:.3f},  σ={s_n:.3f})",
                cdf=lambda yy, mu=mu_n, s_=s_n: _truncnorm_cdf(yy, mu, s_)
            ))
    if add_trunc_exponential:
        lam = _fit_truncexp_mle(v_raw)
        if lam is not None:
            models.append(dict(
                name="TExp", color="#7d3fc1", ls=":", lw=1.5,
                label=f"TExp (λ={lam:.3f})",
                cdf=lambda yy, l=lam: _truncexp_cdf(yy, l)
            ))

    y_grid = np.linspace(0,1,800)
    for m in models:
        ks, cvm = _ks_and_cvm(s, m['cdf'])
        m['ks'], m['cvm'] = ks, cvm
        h = ax_right.plot(m['cdf'](y_grid), y_grid, color=m['color'], lw=m['lw'], ls=m['ls'], label=m['label'])[0]
        handles.append(h); labels.append(m['label'])

    # 图例靠左上，按 CvM 排序（更好者靠前）
    if models:
        order = np.argsort([m['cvm'] for m in models])
        handles = [handles[0]] + [handles[1+i] for i in order]
        labels  = [labels[0]]  + [labels[1+i]  for i in order]
    leg = ax_right.legend(handles, labels, loc="upper left",
                          bbox_to_anchor=(0.00,0.98), bbox_transform=ax_right.transAxes,
                          fontsize=10.5, frameon=False,
                          handlelength=2.0, handletextpad=0.7, labelspacing=0.5)

    # 文本框（与图例左对齐、置于图例下方）
    fig.canvas.draw()
    leg_bbox_axes = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax_right.transAxes.inverted())
    legend_left  = leg_bbox_axes.x0
    legend_bottom= leg_bbox_axes.y0
    lines = [
        f"N = {summ['N']:,}",
        f"mean = {summ['mean']:.4f}",
        f"sd = {summ['sd']:.4f}",
        f"median = {summ['median']:.4f}",
        f"Q1 = {summ['Q1']:.4f}",
        f"Q3 = {summ['Q3']:.4f}",
        f"IQR = {summ['IQR']:.4f}",
        f"p01 = {summ['p01']:.4f}",
        f"p99 = {summ['p99']:.4f}",
    ]
    ax_right.text(legend_left, max(legend_bottom-0.015, 0.02), "\n".join(lines),
                  transform=ax_right.transAxes, ha="left", va="top", fontsize=10.5,
                  bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.75", alpha=0.92))

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")

# ---------------------- 主入口 ----------------------
if __name__ == "__main__":
    TRAVEL_DATA_PATH = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\结果数据\人口出行.csv"
    GRID_SHP_PATH    = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\郑州网格shp\郑州分析网格.shp"

    # 缓存优先
    if os.path.exists(CACHE_DATA):
        try:
            vuln_gdf = gpd.read_file(CACHE_DATA, layer="vuln")
            print(f"⚡ 使用缓存：{CACHE_DATA}（{len(vuln_gdf)} 行）")
        except Exception as e:
            print(f"读取缓存失败，将重新计算：{e}")
            vuln_gdf = calculate_vulnerability_with_time(TRAVEL_DATA_PATH, GRID_SHP_PATH)
    else:
        vuln_gdf = calculate_vulnerability_with_time(TRAVEL_DATA_PATH, GRID_SHP_PATH)

    if isinstance(vuln_gdf, gpd.GeoDataFrame) and not vuln_gdf.empty:
        export_outputs(vuln_gdf)
        visualize_map_ecdf_models(
            vuln_gdf,
            output_path=os.path.join(OUTPUT_DIR, "vulnerability_map_ecdf_models_green_compact.png"),
            title="Vulnerabilty of floating population",
            add_trunc_normal=True,
            add_trunc_exponential=True,
            # 可微调以改变对比度但不改变色系
            q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06
        )
