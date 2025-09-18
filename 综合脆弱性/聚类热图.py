# -*- coding: utf-8 -*-
"""
矩形研究区：热力图 + 行/列树状图（地理顺序相邻约束聚类）
- 严密贴合：地图轴=网格外边界 [0..N]，树轴同度量，长度一致 + 像素级重叠
- 叶端重标到 0.5..N-0.5（网格中心），并按分支深度少量“压入”地图以消缝
- 彩色树状图：cluster（分簇着色）/ depth（按深度渐变）
- 画布宽:高 = 16:9；上下左右边距 = 10%；色标轴更窄
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from shapely.geometry import box
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, dendrogram

# ========= 可调参数 =========
FIGSIZE = (16, 9)           # (width, height) -> 宽:高 = 16:9（横屏）
DPI_EXPORT = 600
CMAP = "viridis"

# 网格描边尽量轻，避免误判为白缝
EDGE_KW = dict(edgecolor="0.88", linewidth=0.06)

USE_GEOGRAPHIC_ORDER = True   # 使用相邻约束聚类（保持地理顺序）
LEAF_EXT_FRAC = 0.015         # 叶端向地图“压入”比例（按树深度），0.01~0.03 合理
DENDRO_LW = 1.1               # 树线宽

# —— 彩色树状图开关与样式 ——
TREE_COLOR = True             # True=彩色；False=单色
TREE_COLOR_MODE = "cluster"   # "cluster" 按分簇着色；"depth" 按高度渐变
TREE_COLOR_THRESHOLD = "auto" # cluster模式阈值："auto"=默认0.7*max_depth；(0,1]=相对阈值；>1=绝对值
TREE_CMAP = "tab20"           # depth模式或其他需要的 colormap

# —— 画面边距（相对 figure）——
MARGIN = 0.10                 # 上下左右均 10%
# 色标列更窄
CBAR_COL_RATIO = 0.35

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 21,
    "axes.unicode_minus": False,
})

# ========= 数据与聚类参数 =========
VALUE_COL = "v_total"
METRIC = "correlation"      # 在相邻约束里用 'cosine' 近似
METHOD = "average"          # 'ward' 只能配 'euclidean'
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*identical low and high ylims.*")

# ---- 路径 ----
try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "融合结果")

for nm in ["vulnerability_integrated.shp",
           "vulnerability_integrated.geojson",
           "vulnerability_integrated.json"]:
    p = os.path.join(DATA_DIR, nm)
    if os.path.exists(p):
        VEC_PATH = p; break
else:
    raise FileNotFoundError("未找到融合结果矢量：融合结果/vulnerability_integrated.*")

gdf = gpd.read_file(VEC_PATH)
if "grid_id" not in gdf.columns:
    for c in ["id","ID","tid","TID","code","gridcode"]:
        if c in gdf.columns:
            gdf = gdf.rename(columns={c:"grid_id"}); break
assert "grid_id" in gdf.columns and VALUE_COL in gdf.columns
gdf["grid_id"] = gdf["grid_id"].astype(str)

# ---- 若无 row/col，用质心推断 ----
def infer_row_col_from_centroid(gdf_in: gpd.GeoDataFrame):
    ctr = gdf_in.geometry.centroid
    xs, ys = ctr.x.values, ctr.y.values
    def grid_res(vals):
        u = np.unique(np.round(vals, 6))
        d = np.diff(np.sort(u)); d = d[d>0]
        return np.median(d) if len(d) else 1.0
    rx, ry = grid_res(xs), grid_res(ys)
    xmin, ymax = xs.min(), ys.max()
    col = np.rint((xs - xmin) / rx).astype(int)   # 左→右
    row = np.rint((ymax - ys) / ry).astype(int)   # 上→下
    return pd.DataFrame({"grid_id": gdf_in["grid_id"].values, "row": row, "col": col})

if not {"row","col"}.issubset(gdf.columns):
    gdf = gdf.merge(infer_row_col_from_centroid(gdf), on="grid_id", how="left")

# ---- 组装矩阵 ----
n_rows = int(gdf["row"].max()) + 1
n_cols = int(gdf["col"].max()) + 1
M = np.full((n_rows, n_cols), np.nan)
rc = gdf[["row","col",VALUE_COL]].dropna()
M[rc["row"].to_numpy(), rc["col"].to_numpy()] = rc[VALUE_COL].to_numpy()

# 缺失插补（仅供聚类）
Mat = pd.DataFrame(M)
if Mat.isna().any().any():
    rmean = Mat.mean(axis=1); Mat = Mat.apply(lambda r: r.fillna(rmean[r.name]), axis=1)
    cmean = Mat.mean(axis=0); Mat = Mat.apply(lambda c: c.fillna(cmean[c.name]))
    Mat = Mat.fillna(Mat.values.mean())
M_imp = Mat.values

# ---- 聚类（无约束/相邻约束） ----
def _linkage_unconstrained(A, metric=METRIC, method=METHOD):
    d = pdist(A, metric=metric)
    if np.isnan(d).any():
        d = pdist(A, metric="cosine")
        if method == "ward": method = "average"
    Z = linkage(d, method=method)
    try: Z = optimal_leaf_ordering(Z, d)
    except Exception: pass
    return Z

def _linkage_constrained_1d(A, metric=METRIC, method=METHOD):
    try:
        from scipy.sparse import diags
        from sklearn.cluster import AgglomerativeClustering
    except Exception as e:
        raise ImportError("需要 scikit-learn：pip install scikit-learn") from e
    n = A.shape[0]
    conn = diags([1,1], offsets=[-1,1], shape=(n,n), format="csr")
    use_metric = "cosine" if metric=="correlation" else metric
    if method == "ward": use_metric = "euclidean"
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.0,
        linkage=method, metric=use_metric,
        connectivity=conn, compute_distances=True
    ).fit(A)
    children = model.children_
    distances = getattr(model, "distances_", None)
    if distances is None:
        distances = np.linspace(0, 1, children.shape[0], endpoint=False)
    counts = np.zeros(children.shape[0])
    for i,(a,b) in enumerate(children):
        counts[i] = (1 if a<n else counts[a-n]) + (1 if b<n else counts[b-n])
    return np.column_stack([children, distances, counts]).astype(float)

def enforce_geo_order(Z, n_leaves):
    Z = Z.copy()
    span_min = np.empty(n_leaves + Z.shape[0], dtype=int)
    span_max = np.empty_like(span_min)
    for i in range(n_leaves): span_min[i]=i; span_max[i]=i
    for i in range(Z.shape[0]):
        a, b = int(Z[i,0]), int(Z[i,1])
        if span_min[a] > span_min[b]:
            Z[i,0], Z[i,1] = Z[i,1], Z[i,0]; a, b = b, a
        new = n_leaves + i
        span_min[new] = min(span_min[a], span_min[b])
        span_max[new] = max(span_max[a], span_max[b])
    return Z

if USE_GEOGRAPHIC_ORDER:
    Z_row = _linkage_constrained_1d(M_imp,  metric=METRIC, method=METHOD)
    Z_col = _linkage_constrained_1d(M_imp.T, metric=METRIC, method=METHOD)
    Z_row = enforce_geo_order(Z_row, n_rows)
    Z_col = enforce_geo_order(Z_col, n_cols)
else:
    Z_row = _linkage_unconstrained(M_imp)
    Z_col = _linkage_unconstrained(M_imp.T)

# ========= 工具：准备 dendrogram（两遍以支持相对阈值着色） =========
def _prepare_dendro(Z, orientation, colorize=True, color_mode="cluster",
                    color_threshold="auto"):
    # 第一次：拿到 max_depth
    d0 = dendrogram(Z, orientation=orientation, no_labels=True,
                    no_plot=True, color_threshold=None)
    max_depth = 0.0
    for dd in d0["dcoord"]:
        max_depth = max(max_depth, max(dd))
    # 计算真正用于着色的阈值
    ct = None
    if colorize and color_mode == "cluster":
        if isinstance(color_threshold, (int, float)):
            if 0 < color_threshold <= 1 and max_depth > 0:
                ct = color_threshold * max_depth
            elif color_threshold > 1:
                ct = float(color_threshold)
        elif isinstance(color_threshold, str) and color_threshold == "auto":
            ct = None  # SciPy 默认：0.7*max_depth
    # 第二次：带阈值真正取出坐标与颜色
    d = dendrogram(Z, orientation=orientation, no_labels=True,
                   no_plot=True, color_threshold=ct)
    color_list = d["color_list"] if (colorize and color_mode == "cluster") else None
    return d, max_depth, color_list

# ========= 树绘制（叶中心重标 + 轴限=边界0..N + 彩色） =========
def _draw_dendro_rescaled(
    Z, ax, n_leaves, orientation="left",
    lw=DENDRO_LW, mono_color="0.25",
    extend_frac=LEAF_EXT_FRAC, antialiased=False,
    colorize=TREE_COLOR, color_mode=TREE_COLOR_MODE,
    color_threshold=TREE_COLOR_THRESHOLD, cmap=TREE_CMAP,
):
    d, max_depth, color_list = _prepare_dendro(
        Z, orientation, colorize=colorize, color_mode=color_mode, color_threshold=color_threshold
    )

    def scale_leaf_coords(vals):
        v = np.asarray(vals, dtype=float)
        return (v - 5.0)/10.0 + 0.5  # 0.5..N-0.5（网格中心）

    segs, cols = [], []
    cmap_obj = mpl.cm.get_cmap(cmap)

    if orientation == "left":
        for i, (x, y) in enumerate(zip(d["dcoord"], d["icoord"])):
            yy = scale_leaf_coords(y)
            segs += [[(x[0], yy[0]), (x[1], yy[1])],
                     [(x[1], yy[1]), (x[2], yy[2])],
                     [(x[2], yy[2]), (x[3], yy[3])]]
            if not colorize:
                c = mono_color
            else:
                if color_mode == "cluster":
                    c = color_list[i]
                else:
                    c = cmap_obj((max(x)/max_depth) if max_depth>0 else 0.0)
            cols += [c, c, c]

        lc = LineCollection(segs, colors=cols, linewidths=lw,
                            capstyle="butt", joinstyle="miter")
        lc.set_antialiased(antialiased)
        ax.add_collection(lc)
        delta = float(extend_frac) * max_depth
        ax.set_xlim(max_depth, -delta)   # 向地图“压入”
        ax.set_ylim(n_leaves, 0)         # 轴限=边界 0..N（上→下）
        ax.set_xmargin(0); ax.set_ymargin(0)
        ax.axis('off')
        return max_depth

    elif orientation == "top":
        for i, (x, y) in enumerate(zip(d["icoord"], d["dcoord"])):
            xx = scale_leaf_coords(x)
            segs += [[(xx[0], y[0]), (xx[1], y[1])],
                     [(xx[1], y[1]), (xx[2], y[2])],
                     [(xx[2], y[2]), (xx[3], y[3])]]
            if not colorize:
                c = mono_color
            else:
                if color_mode == "cluster":
                    c = color_list[i]
                else:
                    c = cmap_obj((max(y)/max_depth) if max_depth>0 else 0.0)
            cols += [c, c, c]

        lc = LineCollection(segs, colors=cols, linewidths=lw,
                            capstyle="butt", joinstyle="miter")
        lc.set_antialiased(antialiased)
        ax.add_collection(lc)
        delta = float(extend_frac) * max_depth
        ax.set_xlim(0, n_leaves)         # 轴限=边界 0..N（左→右）
        ax.set_ylim(-delta, max_depth)   # 向地图“压入”
        ax.set_xmargin(0); ax.set_ymargin(0)
        ax.axis('off')
        return max_depth
    else:
        raise ValueError("orientation 仅支持 'left' 或 'top'")

# ========= 作图 =========
fig = plt.figure(figsize=FIGSIZE)

# 色标更窄：把最后一列比例调小为 CBAR_COL_RATIO
gs = GridSpec(nrows=4, ncols=4,
              width_ratios=[1.4, 0.2, 8, CBAR_COL_RATIO],
              height_ratios=[1.0, 0.2, 8, 0.8],
              wspace=0.05, hspace=0.05)

ax_row = fig.add_subplot(gs[2, 0])   # 左树
ax_col = fig.add_subplot(gs[0, 2])   # 顶树
ax_map = fig.add_subplot(gs[2, 2])   # 地图
ax_cb  = fig.add_subplot(gs[2, 3])   # 色带（更窄）

# ========= 地图（补全矩形网格，强制边到边范围） =========
vmin, vmax = 0.0, 1.0
ctr = gdf.geometry.centroid
xs, ys = ctr.x.values, ctr.y.values

def grid_res(vals):
    u = np.unique(np.round(vals, 6))
    d = np.diff(np.sort(u)); d = d[d>0]
    return np.median(d) if len(d) else 1.0

rx, ry = grid_res(xs), grid_res(ys)
xmin_c, xmax_c = xs.min(), xs.max()
ymin_c, ymax_c = ys.min(), ys.max()
# 外边界（边到边）
xL, xR = xmin_c - rx/2, xmax_c + rx/2
yB, yT = ymin_c - ry/2, ymax_c + ry/2

# 完整矩形网格（背景浅灰 + 值覆盖）
geoms, rows_all, cols_all = [], [], []
for r in range(n_rows):
    for c in range(n_cols):
        x0 = xL + c*rx; x1 = x0 + rx
        y1 = yT - r*ry; y0 = y1 - ry
        geoms.append(box(x0, y0, x1, y1)); rows_all.append(r); cols_all.append(c)

full_grid = gpd.GeoDataFrame({"row": rows_all, "col": cols_all},
                             geometry=geoms, crs=gdf.crs)
vals = gdf[["row","col",VALUE_COL]].drop_duplicates()
gdf_full = full_grid.merge(vals, on=["row","col"], how="left")

cmap = plt.get_cmap(CMAP).copy()
gdf_full.plot(column=VALUE_COL, cmap=cmap, vmin=vmin, vmax=vmax,
              ax=ax_map, rasterized=True, **EDGE_KW,
              missing_kwds={"color": "#f0f0f0", "edgecolor": EDGE_KW["edgecolor"]})

# 地图轴=网格外边界；无 margin；等比例
ax_map.set_aspect('equal', adjustable='box')
ax_map.set_xlim(xL, xR); ax_map.set_ylim(yB, yT)
ax_map.margins(0)
ax_map.set_axis_off()

# 10% 统一边距；标题居于上边距内
plt.subplots_adjust(left=MARGIN, right=1-MARGIN, top=1-MARGIN, bottom=MARGIN)
fig.suptitle("Integrated Grid Vulnerability of Zhengzhou", y=1-MARGIN/2)

# 固定布局，用于像素级贴合
fig.canvas.draw()

# ========= 树状图（彩色 + 重标 + 压入 + 轴限=0..N） =========
_ = _draw_dendro_rescaled(Z_row, ax_row, n_rows, orientation="left",
                          colorize=TREE_COLOR, color_mode=TREE_COLOR_MODE,
                          color_threshold=TREE_COLOR_THRESHOLD, cmap=TREE_CMAP,
                          extend_frac=LEAF_EXT_FRAC)
_ = _draw_dendro_rescaled(Z_col, ax_col, n_cols, orientation="top",
                          colorize=TREE_COLOR, color_mode=TREE_COLOR_MODE,
                          color_threshold=TREE_COLOR_THRESHOLD, cmap=TREE_CMAP,
                          extend_frac=LEAF_EXT_FRAC)

# ========= 严格“长度一致” + 像素级重叠（2~3 px） =========
def _one_pixel(fig):
    return (1.0/(fig.get_figwidth()*fig.dpi),
            1.0/(fig.get_figheight()*fig.dpi))

pos_row = ax_row.get_position()
pos_col = ax_col.get_position()
pos_map = ax_map.get_position()
PX, PY = _one_pixel(fig)
OVERLAP_X = 2.5*PX
OVERLAP_Y = 2.5*PY

# 左树：高度=地图高度；右侧与地图左边重叠
ax_row.set_position([pos_map.x0 - pos_row.width + OVERLAP_X,
                     pos_map.y0,
                     pos_row.width + OVERLAP_X,
                     pos_map.height])

# 顶树：宽度=地图宽度；底部与地图上边重叠
ax_col.set_position([pos_map.x0,
                     pos_map.y1 - OVERLAP_Y,
                     pos_map.width,
                     pos_col.height + OVERLAP_Y])

# 树轴浮到地图之上，且无白底
for ax in (ax_row, ax_col):
    ax.set_zorder(ax_map.get_zorder() + 10)
    ax.patch.set_visible(False)
    ax.set_facecolor("none")

# ========= 色带（更窄） =========
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
cb.set_label("Vulnerability (0–1)")

# ========= 导出（保留 10% 边距）=========
out_png = os.path.join(DATA_DIR, "grid_map_with_dendrograms.png")
plt.savefig(out_png, dpi=DPI_EXPORT)  # 不用 bbox_inches="tight"，以保留 10% 边距
plt.close(fig)
print("✅ 保存：", out_png)
