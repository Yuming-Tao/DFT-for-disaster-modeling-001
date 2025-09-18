# -*- coding: utf-8 -*-
"""
Population Vulnerability (0–1) — Choropleth + Shared-Y Histogram (Beta fit)
- 左侧：地图（保持原 POP_BLUE_CMAP 蓝色系不变；仅做数值“对比度拉伸”）
- 右侧：与色标共用 y 轴的 直方图 + Beta 拟合（可选 KDE）
- 版式：16:9、四边 10% 页边距；主图与色标等高，色标与直方图更贴近
- 字体：Times New Roman；标题 21pt，其余 10.5pt
- 已移除：指北针与比例尺

Outputs (vulnerability_vector_output/):
  - vulnerability_vector.csv / .shp / .geojson
  - population_vulnerability_map.png
  - population_vulnerability_map_hist.png
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import sjoin
from sklearn.neighbors import BallTree

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

# ---------------------- Global style ----------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10.5,          # 其他文本
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 21,     # suptitle 默认
    "axes.unicode_minus": False,
})

# ---------------------- Paths ----------------------
os.makedirs("vulnerability_vector_output", exist_ok=True)
TARGET_CRS = "EPSG:32650"  # Zhengzhou UTM 50N

# TODO: 修改为你的本地路径
OSM_FOLDER = r"C:\Users\Administrator\Desktop\参数文件\河南省OSM数据"
GRID_SHP   = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\郑州网格shp\郑州分析网格.shp"
POP_CSV    = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\结果数据\人口热力.csv"

# ---------------------- Colormap（保持不变） ----------------------
POP_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "PopBlueLinear",
    [
        "#EAF2FF", "#CFE1FF", "#AFCBFF",
        "#7FAEFF", "#4D8EFF", "#2F7DFF", "#0A4EC2"
    ]
)

# ---------------------- Helpers ----------------------
def _as_clean_unit_interval(x):
    v = np.asarray(pd.Series(x).astype(float).to_numpy())
    m = np.isfinite(v) & (v >= 0) & (v <= 1)
    return v[m]

def _robust_summary(v: np.ndarray):
    x = _as_clean_unit_interval(v)
    N = int(x.size)
    if N == 0:
        return dict(N=0)
    med = float(np.median(x))
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = float(q3 - q1)
    try:
        mad = float(stats.median_abs_deviation(x, scale='normal'))
    except Exception:
        mad = float(np.median(np.abs(x - med)) / 0.67448975)
    p01, p99 = np.quantile(x, [0.01, 0.99])
    zeros = float(np.mean(x == 0.0))
    ones  = float(np.mean(x == 1.0))
    mean  = float(np.mean(x))
    sd    = float(np.std(x, ddof=1)) if N > 1 else 0.0
    return dict(N=N, mean=mean, sd=sd, median=med, Q1=float(q1), Q3=float(q3),
                IQR=iqr, MAD=mad, p01=float(p01), p99=float(p99),
                pct_zero=zeros, pct_one=ones)

def _fit_beta_01(v):
    """在 (0,1) 上做 Beta MLE（loc=0, scale=1），返回 (a,b,n_eff)。"""
    x = np.asarray(v)
    x = x[(x > 0.0) & (x < 1.0) & np.isfinite(x)]
    n_eff = x.size
    if n_eff < 10:
        return None, None, n_eff
    a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
    return float(a), float(b), n_eff

# —— 对比度拉伸（保持色系不变，仅调整映射前的强度） ——
def blue_contrast(series: pd.Series, q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06) -> pd.Series:
    """
    1) 分位裁剪到 [q_low, q_high]
    2) 归一化到 0..1
    3) γ 增益（gamma<1 提升低—中值层次）
    4) 最浅色保底（min_tint，避免近白）
    返回 0..1 的“可视化强度”
    """
    x = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = x.quantile(q_low), x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(hi - lo) or hi <= lo:
            return pd.Series(np.zeros(len(x)), index=x.index)
    y = (x.clip(lo, hi) - lo) / (hi - lo)   # 0..1
    y = np.power(y, gamma)                  # γ 增益
    y = min_tint + (1.0 - min_tint) * y     # 最浅色保底
    return pd.Series(y).fillna(min_tint).clip(0, 1)

def _pct_to_frac(x):
    if isinstance(x, str) and x.endswith("%"):
        return float(x[:-1]) / 100.0
    return float(x)

# ---------------------- 1. Load OSM ----------------------
def load_osm_layers(osm_folder):
    layer_files = {'pois': 'gis_osm_pois_free_1.shp', 'roads': 'gis_osm_roads_free_1.shp'}
    layers = {}
    for name, fn in layer_files.items():
        fp = os.path.join(osm_folder, fn)
        if not os.path.exists(fp):
            print(f"⚠️ Missing {name}: {fp}")
            continue
        try:
            gdf = gpd.read_file(fp).to_crs(TARGET_CRS)
            if name == 'pois':
                found = next((f for f in ['amenity', 'fclass'] if f in gdf.columns), None)
                if found: print(f"✅ POI type field: '{found}'")
                else:     print("⚠️ POI lacks amenity/fclass")
            layers[name] = gdf
            print(f"✅ Loaded {name}: {len(gdf)} rows")
        except Exception as e:
            print(f"❌ Load {name} failed: {e}")
    return layers

# ---------------------- 2. Load grid & population ----------------------
def load_and_preprocess_data(grid_path, pop_path, osm_layers):
    gdf_grid = gpd.read_file(grid_path).to_crs(TARGET_CRS)
    for f in ['TID', 'grid_id', 'ID', 'code', 'gridcode']:
        if f in gdf_grid.columns:
            gdf_grid = gdf_grid.rename(columns={f: 'tid'})
            print(f"✅ Grid ID '{f}' → 'tid'"); break
    else:
        raise ValueError("No grid ID column found")
    gdf_grid['tid'] = gdf_grid['tid'].astype(str).str.strip()
    print(f"✅ Loaded grid: {len(gdf_grid)} cells")

    df_pop = pd.read_csv(pop_path)
    if not {'all_pop','tid'}.issubset(df_pop.columns):
        raise ValueError("Population CSV missing columns: {'all_pop','tid'}")
    df_pop['all_pop'] = pd.to_numeric(df_pop['all_pop'], errors='coerce')
    n_invalid = df_pop['all_pop'].isna().sum()
    if n_invalid > 0:
        print(f"⚠️ {n_invalid} invalid all_pop rows removed")
        df_pop = df_pop.dropna(subset=['all_pop'])
    df_pop['all_pop'] = df_pop['all_pop'].astype(float)
    df_pop['tid'] = df_pop['tid'].astype(str).str.strip()
    df_pop = df_pop.groupby('tid', as_index=False)['all_pop'].mean()
    print(f"✅ Population aggregated: {len(df_pop)} rows")

    return gdf_grid, df_pop, osm_layers.get('pois'), osm_layers.get('roads')

# ---------------------- 3. Infer functional type ----------------------
def infer_grid_function_type(gdf_grid, gdf_pois, gdf_roads):
    gdf_grid['func_type'] = 'other'
    if gdf_pois is not None and not gdf_pois.empty:
        poi_field = next((f for f in ['amenity','fclass'] if f in gdf_pois.columns), None)
        if poi_field:
            mapping = {'restaurant':'commercial','cafe':'commercial','shop':'commercial',
                       'school':'public','hospital':'public','library':'public',
                       'residential':'residential','house':'residential',
                       'parking':'transport','bus_stop':'transport','train_station':'transport',
                       'park':'green','garden':'green',
                       'food':'commercial','education':'public','healthcare':'public',
                       'transportation':'transport','recreation':'green'}
            gdf_pois['func_type'] = gdf_pois[poi_field].map(mapping).fillna('other')
            joined = sjoin(gdf_grid[['tid','geometry']], gdf_pois[['geometry','func_type']],
                           how='left', predicate='contains')
            counts = joined.groupby(['tid','func_type']).size().unstack(fill_value=0)
            if not counts.empty:
                dom = counts.idxmax(axis=1).reset_index(name='func_type_poi')
                gdf_grid = gdf_grid.merge(dom, on='tid', how='left')
                gdf_grid['func_type'] = gdf_grid['func_type'].fillna(gdf_grid['func_type_poi'])
                gdf_grid = gdf_grid.drop(columns=['func_type_poi'])
        else:
            print("⚠️ POIs lack amenity/fclass; skip POI inference")
    if gdf_roads is not None and not gdf_roads.empty:
        road_field = 'fclass' if 'fclass' in gdf_roads.columns else 'highway'
        if road_field in gdf_roads.columns:
            main = gdf_roads[gdf_roads[road_field].isin(['primary','secondary','trunk'])]
            def road_density(geom):
                if geom.area == 0: return 0.0
                lines = main[main.intersects(geom)]
                return lines.length.sum() / geom.area
            gdf_grid['road_density'] = gdf_grid['geometry'].apply(road_density)
            gdf_grid.loc[(gdf_grid['func_type']=='other') & (gdf_grid['road_density']>0.01),
                         'func_type'] = 'transport'
    else:
        print("⚠️ Road layer empty; skip road inference")
    print("Function types:", gdf_grid['func_type'].value_counts().to_dict())
    return gdf_grid

# ---------------------- 4. Compute vulnerability (0–1) ----------------------
def calculate_vulnerability_index(gdf_grid, df_pop, gdf_roads):
    gdf_grid['area_km2'] = gdf_grid['geometry'].area / 1e6
    gdf_grid = gdf_grid.merge(df_pop, on='tid', how='left')
    gdf_grid['all_pop'] = gdf_grid['all_pop'].fillna(0.0)
    gdf_grid['exposure'] = (gdf_grid['all_pop'] / gdf_grid['area_km2'].replace(0,1e-8)).clip(lower=0)
    sens_map = {'residential':1.0,'commercial':1.1,'public':1.0,
                'transport':1.0,'green':0.6,'industrial':0.7,'other':1.0}
    gdf_grid['sensitivity'] = gdf_grid['func_type'].map(sens_map).fillna(1.0)
    if gdf_roads is not None and not gdf_roads.empty:
        road_field = 'fclass' if 'fclass' in gdf_roads.columns else 'highway'
        if road_field in gdf_roads.columns:
            main = gdf_roads[gdf_roads[road_field].isin(['primary','secondary','trunk'])]
            centers = [p for p in main['geometry'].centroid if not p.is_empty]
            if centers:
                coords = np.array([(p.x,p.y) for p in centers]); tree = BallTree(coords)
                gdf_grid['centroid'] = gdf_grid['geometry'].centroid
                def nearest(c):
                    if c.is_empty: return np.inf
                    d,_ = tree.query([[c.x,c.y]], k=1); return d[0][0]
                gdf_grid['road_dist'] = gdf_grid['centroid'].apply(nearest)
                gdf_grid['connectivity'] = 1.0/(gdf_grid['road_dist']+1.0)
            else: gdf_grid['connectivity']=0.5
        else: gdf_grid['connectivity']=0.5
    else: gdf_grid['connectivity']=0.5
    cap = {'residential':1000,'commercial':5000,'public':2000,
           'transport':3000,'green':100,'industrial':800,'other':500}
    gdf_grid['carrying_cap'] = gdf_grid['func_type'].map(cap).fillna(1000)
    dens = gdf_grid['all_pop']/gdf_grid['area_km2'].replace(0,1e-8)
    gdf_grid['capacity_stress'] = (dens/gdf_grid['carrying_cap'].replace(0,1e-8)).replace([np.inf,-np.inf],1).clip(0,5)
    gdf_grid['vulnerability_index_raw'] = (gdf_grid['exposure']*gdf_grid['sensitivity'])/ \
                                          (gdf_grid['connectivity']*(1.0+gdf_grid['capacity_stress']))
    vmin = float(gdf_grid['vulnerability_index_raw'].min())
    vmax = float(gdf_grid['vulnerability_index_raw'].max())
    gdf_grid['vulnerability_index'] = (gdf_grid['vulnerability_index_raw']-vmin)/(vmax-vmin) if vmax>vmin else 0.0
    return gdf_grid[['tid','geometry','func_type','all_pop','exposure','sensitivity','vulnerability_index']]

# ---------------------- 5. Export basic outputs ----------------------
def export_results(gdf_result, output_dir="vulnerability_vector_output",
                   q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    gdf_result.drop(columns='geometry').to_csv(os.path.join(output_dir,"vulnerability_vector.csv"),
                                               index=False, encoding='utf-8-sig')
    print("✅ Saved CSV")
    gdf_result.rename(columns={'vulnerability_index':'vulnerabil','sensitivity':'sensitiv'}) \
              .to_file(os.path.join(output_dir,"vulnerability_vector.shp"), encoding='utf-8')
    print("✅ Saved Shapefile")
    gdf_result.to_file(os.path.join(output_dir,"vulnerability_vector.geojson"),
                       driver='GeoJSON', encoding='utf-8')
    print("✅ Saved GeoJSON")

    # 快速静态图：保持蓝色系，仅做对比度拉伸
    gdf_plot = gdf_result.copy()
    gdf_plot['vuln_vis'] = blue_contrast(gdf_plot['vulnerability_index'],
                                         q_low=q_low, q_high=q_high, gamma=gamma, min_tint=min_tint)
    plt.figure(figsize=(16, 9), dpi=300)
    plt.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.90)
    ax = plt.gca()
    gdf_plot.plot(column='vuln_vis', cmap=POP_BLUE_CMAP, vmin=0, vmax=1,
                  legend=True, legend_kwds={'label':'Vulnerability (stretched 0–1)',
                                            'orientation':'vertical'},
                  ax=ax, edgecolor='white', linewidth=0.2, alpha=0.95)
    ax.set_axis_off()
    plt.suptitle('Population Vulnerability Distribution', y=0.965, fontsize=21)
    plt.savefig(os.path.join(output_dir,"population_vulnerability_map.png"), dpi=300)
    plt.close()
    print("✅ Saved map")

# ---------------------- 6. Visualization (Map + Histogram) ----------------------
def visualize_shared_y_map_with_hist_beta(
    gdf_result: gpd.GeoDataFrame,
    output_path="vulnerability_vector_output/population_vulnerability_map_hist.png",
    colorbar_size="2.6%", colorbar_pad=0.06,
    hist_width="18%", gap_between=0.012,
    bins=60, density=True, dpi=300,
    title="Population Vulnerability Index",
    show_kde=True,
    legend_anchor=(0.02, 0.98),   # 图例左上
    stats_gap=0.02,               # 文本与图例间距（轴坐标）
    stats_fontsize=10.5,
    # 对比度拉伸参数（不改色系，仅增强可读性）
    q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06
):
    # 原始值（0–1）用于统计/Beta/KDE
    v_raw = pd.to_numeric(gdf_result['vulnerability_index'], errors='coerce').fillna(0.0).clip(0, 1).to_numpy()
    if v_raw.size == 0:
        raise ValueError("No valid vulnerability_index values.")

    # 用于着色的“可视化强度”，色系不变
    v_vis = blue_contrast(gdf_result['vulnerability_index'], q_low, q_high, gamma, min_tint).to_numpy()

    # 统计与拟合
    summ = _robust_summary(v_raw)
    a, b, n_eff = _fit_beta_01(v_raw)
    y_grid = np.linspace(0.0, 1.0, 600)
    beta_pdf = stats.beta.pdf(y_grid, a, b) if (a and b) else np.zeros_like(y_grid)
    kde_pdf = np.zeros_like(y_grid)
    if show_kde:
        try:
            v_pos = v_raw[v_raw > 0]
            kde = stats.gaussian_kde(v_pos, bw_method='scott') if v_pos.size >= 5 else None
            if kde is not None: kde_pdf = kde(y_grid)
        except Exception:
            kde_pdf = np.zeros_like(y_grid)

    # 画布与分区（16:9、四边 10%）
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.90)
    fig.suptitle(title, fontsize=21, y=0.965)
    ax_map = fig.add_subplot(1, 1, 1)

    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
    hist_pad_eff = colorbar_pad + _pct_to_frac(colorbar_size) + gap_between
    ax_right = divider.append_axes("right", size=hist_width, pad=hist_pad_eff)

    # 地图（保持蓝色系，使用增强后的强度）
    gdf = gdf_result.copy()
    gdf['vuln_vis'] = v_vis
    gdf.plot(column='vuln_vis', ax=ax_map, cmap=POP_BLUE_CMAP, vmin=0, vmax=1,
             edgecolor='white', linewidth=0.2, alpha=0.95,
             legend=True, legend_kwds={'label':'', 'cax':cax})
    ax_map.set_axis_off()

    # 色标（左刻度；与 v_vis 对齐）
    ticks = [0.00, 0.25, 0.50, 0.75, 1.00]
    cax.set_ylim(0, 1)
    cax.yaxis.set_major_locator(FixedLocator(ticks))
    cax.yaxis.set_major_formatter(FixedFormatter([f"{t:.2f}" for t in ticks]))
    cax.set_ylabel("Vulnerability (stretched 0–1)", fontsize=10.5, labelpad=8)
    cax.yaxis.set_ticks_position('left'); cax.yaxis.set_label_position('left')
    cax.tick_params(axis='y', labelsize=10.5, left=True, right=False)

    # 右侧共享 y
    ax_right.sharey(cax); ax_right.set_ylim(0, 1)
    ax_right.yaxis.set_visible(False)
    ax_right.spines['left'].set_visible(False); ax_right.spines['right'].set_visible(False)

    # 直方图（基于原始 v_raw，保持统计含义）
    hist, bin_edges = np.histogram(v_raw, bins=bins, range=(0,1), density=density)
    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    ax_right.barh(bin_centers, hist, height=0.9*(bin_edges[1]-bin_edges[0]),
                  color='#66a9ff', alpha=0.38, edgecolor='none', label="Histogram")

    # Beta & KDE
    ax_right.plot(beta_pdf, y_grid, lw=2.0, color='#ff7f0e',
                  label=(f"Beta PDF (a={a:.2f}, b={b:.2f})" if a and b else "Beta PDF"))
    if show_kde and np.nanmax(kde_pdf) > 0:
        ax_right.plot(kde_pdf, y_grid, lw=1.6, color='#2ca02c', ls='--', label="Empirical KDE")

    # 图例（左上，便于与统计文本对齐）
    leg = ax_right.legend(loc="upper left",
                          bbox_to_anchor=legend_anchor,
                          bbox_transform=ax_right.transAxes,
                          fontsize=10.5, frameon=False,
                          borderaxespad=0.0,
                          handlelength=2.2, handletextpad=0.8, labelspacing=0.6)

    # 计算图例 bbox，用于对齐统计文本
    fig.canvas.draw()
    bbox_disp = leg.get_window_extent(fig.canvas.get_renderer())
    bbox_axes = bbox_disp.transformed(ax_right.transAxes.inverted())
    legend_left = bbox_axes.x0
    legend_bottom = bbox_axes.y0

    # 统计文本紧贴图例下方左对齐
    stats_lines = [
        f"N = {summ['N']:,}",
        f"mean = {summ['mean']:.4f}",
        f"sd = {summ['sd']:.4f}",
        f"median = {summ['median']:.4f}",
        f"Q1 = {summ['Q1']:.4f}",
        f"Q3 = {summ['Q3']:.4f}",
        f"IQR = {summ['IQR']:.4f}",
        f"p01 = {summ['p01']:.4f}",
        f"p99 = {summ['p99']:.4f}",
        f"zeros = {summ['pct_zero']:.2%}",
        f"ones  = {summ['pct_one']:.2%}",
        (f"a = {a:.3g}, b = {b:.3g}" if a and b else "a/b = NA"),
        (f"n_eff = {n_eff}" if a and b else "")
    ]
    stats_text = "\n".join([s for s in stats_lines if s])
    stats_y = max(legend_bottom - stats_gap, 0.02)  # 防越界
    ax_right.text(legend_left, stats_y, stats_text,
                  transform=ax_right.transAxes, ha="left", va="top",
                  fontsize=10.5, color="black",
                  bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.75", alpha=0.92))

    # 收尾
    ax_right.set_xlim(left=0)
    ax_right.set_xlabel("Density" if density else "Frequency", fontsize=10.5)
    ax_right.set_title("Histogram + Beta fit", fontsize=10.5, pad=6)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")

# ---------------------- Main ----------------------
if __name__ == "__main__":
    try:
        osm_layers = load_osm_layers(OSM_FOLDER)
        gdf_grid, df_pop, gdf_pois, gdf_roads = load_and_preprocess_data(GRID_SHP, POP_CSV, osm_layers)
        gdf_grid = infer_grid_function_type(gdf_grid, gdf_pois, gdf_roads)

        gdf_result = calculate_vulnerability_index(gdf_grid, df_pop, gdf_roads)
        export_results(
            gdf_result,
            q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06  # 可按需微调
        )

        visualize_shared_y_map_with_hist_beta(
            gdf_result,
            output_path="vulnerability_vector_output/population_vulnerability_map_hist.png",
            bins=60, density=True, dpi=300,
            title="Population Vulnerability Index",
            show_kde=True,
            legend_anchor=(0.02, 0.98),
            stats_gap=0.02,
            stats_fontsize=10.5,
            # 版式参数
            colorbar_size="2.6%", colorbar_pad=0.06,
            hist_width="18%", gap_between=0.012,
            # 对比度拉伸参数（保持色系不变）
            q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06
        )

        print("\n🎉 Done!")
        print(f"- Columns: {gdf_result.columns.tolist()}")
        print(f"- Vulnerability range: {gdf_result['vulnerability_index'].min():.4f} — {gdf_result['vulnerability_index'].max():.4f}")
        print(f"- Output dir: {os.path.abspath('vulnerability_vector_output')}")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        print("Please check input paths and data integrity, then run again.")
