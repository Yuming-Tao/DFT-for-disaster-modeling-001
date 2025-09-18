# -*- coding: utf-8 -*-
"""
16:9 固定画幅；标题与色标轴距页面边缘均为页高的 10%；
底部水平色标与 3D 底面投影等长；正射投影（方案A）。
在同一张图上加入三处局部放大图（就近贴图）。
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio import windows
from rasterio.features import geometry_mask
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

# ===== 全局字体（Times New Roman, 10.5pt）=====
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10.5
mpl.rcParams['axes.titlesize'] = 10.5
mpl.rcParams['axes.labelsize'] = 10.5
mpl.rcParams['xtick.labelsize'] = 10.5
mpl.rcParams['ytick.labelsize'] = 10.5

# ========= 栅格三要素合成 =========
class FieldMerger:
    def __init__(self, terrain_precomputed_path, river_path, soil_path,
                 output_dir, alpha=1.5, h_crit=0.15, depth_coeff=0.1):
        self.terrain_path = terrain_precomputed_path
        self.river_path   = river_path
        self.soil_path    = soil_path
        self.output_dir   = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.ref_crs = None; self.ref_transform = None; self.ref_shape = None
        self.alpha = alpha; self.h_crit = h_crit; self.depth_coeff = depth_coeff

    def _ensure_ref_grid(self, src):
        if self.ref_crs is None:
            self.ref_crs = src.crs
            self.ref_transform = src.transform
            self.ref_shape = (src.height, src.width)

    def _load_field(self, path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            self._ensure_ref_grid(src)
            need_reproj = (src.crs != self.ref_crs or src.transform != self.ref_transform
                           or src.height != self.ref_shape[0] or src.width != self.ref_shape[1])
            if need_reproj:
                dst = np.empty(self.ref_shape, dtype=np.float32)
                reproject(arr, dst, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=self.ref_transform, dst_crs=self.ref_crs,
                          resampling=Resampling.bilinear)
            else:
                dst = arr
            return np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0)

    def merge_fields(self):
        V_topo = self._load_field(self.terrain_path)
        phi_R  = self._load_field(self.river_path)
        phi_K  = self._load_field(self.soil_path)
        V_flood = V_topo * (1.0 + self.alpha * phi_R)
        V_final = V_flood * phi_K
        water_depth = V_final * self.depth_coeff
        V_critical  = np.where(water_depth > self.h_crit, V_final * 2.0, V_final)
        return V_critical.astype(np.float32)

    def save_tif(self, data, filename="final_risk_field.tif"):
        path = os.path.join(self.output_dir, filename)
        with rasterio.open(path, "w",
            driver="GTiff", height=self.ref_shape[0], width=self.ref_shape[1],
            count=1, dtype="float32", crs=self.ref_crs, transform=self.ref_transform,
            nodata=-9999.0, compress="LZW") as dst:
            dst.write(data, 1)
        return path

# ========= 栅格 -> 网格均值 =========
def raster_mean_to_grid(raster_path, grid_shp_path, out_dir,
                        field_name='geo_mean', all_touched=True):
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(raster_path) as src_md:
        r_crs = src_md.crs
        r_transform = src_md.transform
        r_bounds = src_md.bounds
        width, height = src_md.width, src_md.height

    gdf = gpd.read_file(grid_shp_path)
    if gdf.crs != r_crs:
        gdf = gdf.to_crs(r_crs)

    means = []
    with rasterio.open(raster_path) as src:
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                means.append(np.nan); continue
            x0, y0, x1, y1 = geom.bounds
            ib = (max(x0, r_bounds.left), max(y0, r_bounds.bottom),
                  min(x1, r_bounds.right), min(y1, r_bounds.top))
            if ib[0] >= ib[2] or ib[1] >= ib[3]:
                means.append(np.nan); continue

            win_f = windows.from_bounds(*ib, transform=r_transform)
            win = windows.Window(
                int(np.floor(win_f.col_off)), int(np.floor(win_f.row_off)),
                int(np.ceil(win_f.width)),    int(np.ceil(win_f.height))
            ).intersection(windows.Window(0, 0, width, height))

            data = src.read(1, window=win, masked=True)
            win_transform = windows.transform(win, r_transform)

            poly_mask = geometry_mask([geom], out_shape=data.shape,
                                      transform=win_transform,
                                      invert=True, all_touched=all_touched)
            total_mask = (~poly_mask) | np.ma.getmaskarray(data)
            vals = np.ma.array(data, mask=total_mask).compressed()
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if vals.size > 0 else np.nan)

    gdf[field_name] = means
    out_shp = os.path.join(out_dir, "Zhengzhou_Grid_GeoVuln_Mean.shp")
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")
    return gdf, out_shp

# ========= 稳健 0–1 =========
def robust_minmax_normalize(series, q_lo=0.02, q_hi=0.98):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = s.quantile(q_lo), s.quantile(q_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index), float(lo), float(hi)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0), float(lo), float(hi)

# ========= 绘制：正射投影 + 10% 页边距 + 三处局部放大 =========
def _shade_color(base_rgba, normal_xy, light_xy=(-0.6, -0.8), min_bright=0.35, max_bright=1.0):
    bx, by = normal_xy; lx, ly = light_xy
    n = np.hypot(bx, by); l = np.hypot(lx, ly)
    if n == 0 or l == 0: w = 1.0
    else:
        w = (bx*lx + by*ly) / (n*l); w = (w + 1) / 2
    w = min_bright + (max_bright - min_bright) * w
    r,g,b,a = mpl.colors.to_rgba(base_rgba)
    return (r*w, g*w, b*w, a)

def _iter_polys(geom):
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms: yield p

def _with_alpha(color, alpha):
    r,g,b,_ = mpl.colors.to_rgba(color)
    return (r,g,b,float(np.clip(alpha,0,1)))

def _build_faces(gdf, v01, h_scale=1.0, cmap="Blues",
                 light_xy=(-0.6,-0.8), edge_decimals=6, equal_height_tol=1e-9):
    cm = mpl.cm.get_cmap(cmap)
    cores, heights = [], []
    for geom, v in zip(gdf.geometry, v01):
        if geom is None or geom.is_empty or not np.isfinite(v):
            cores.append(None); heights.append(np.nan); continue
        poly = max(_iter_polys(geom), key=lambda p: p.area)
        cores.append(list(poly.exterior.coords))
        heights.append(float(np.clip(v*h_scale, 0, 1)))
    def _canon(pt): return (round(pt[0], edge_decimals), round(pt[1], edge_decimals))
    def _edge_key(a,b):
        a=_canon(a); b=_canon(b); return (a,b) if a<b else (b,a)
    edge2={}
    for idx,coords in enumerate(cores):
        if coords is None: continue
        for i in range(len(coords)-1):
            edge2.setdefault(_edge_key(coords[i],coords[i+1]),[]).append(idx)
    cent = gdf.geometry.representative_point()
    order = np.lexsort((cent.x.values, -cent.y.values))
    faces, cols = [], []
    for idx in order:
        coords=cores[idx]
        if coords is None: continue
        h=heights[idx]; v=v01[idx]
        faces.append([(x,y,h) for (x,y) in coords]); cols.append(_with_alpha(cm(v), 0.9))
        for i in range(len(coords)-1):
            a,b=coords[i],coords[i+1]
            owners=edge2.get(_edge_key(a,b),[idx])
            nx,ny=(b[1]-a[1]),-(b[0]-a[0])
            c_side=_with_alpha(_shade_color(cm(v),(nx,ny),light_xy=light_xy),0.75)
            if len(owners)==1:
                faces.append([(a[0],a[1],0.0),(b[0],b[1],0.0),(b[0],b[1],h),(a[0],a[1],h)])
                cols.append(c_side)
            else:
                jdx=owners[0] if owners[1]==idx else owners[1]
                hj=heights[jdx]
                if (not np.isfinite(hj)) or (h>hj and abs(h-hj)>equal_height_tol):
                    z0=hj if np.isfinite(hj) else 0.0
                    faces.append([(a[0],a[1],z0),(b[0],b[1],z0),(b[0],b[1],h),(a[0],a[1],h)])
                    cols.append(c_side)
    return faces, cols

def _draw_roi_rect_on_main(ax, rect_xy, z=0, color='crimson', lw=1.2):
    rx0, rx1, ry0, ry1 = rect_xy
    xs=[rx0,rx1,rx1,rx0,rx0]; ys=[ry0,ry0,ry1,ry1,ry0]; zs=[z]*5
    ax.plot(xs, ys, zs, color=color, lw=lw, alpha=0.9)

def plot_grid_3d_prisms_z01(
    gdf, value_field, out_png,
    title="Geographic Vulnerability of Zhengzhou",
    cmap="Blues",
    elev=22, azim=-55,
    light_xy=(-0.6, -0.8),
    z_aspect=0.24, h_scale=1.00,
    proj='ortho',
    # ----- 版式 -----
    figsize=(16, 9),
    top_margin_frac=0.10,
    bottom_margin_frac=0.10,
    left_margin=0.05, right_margin=0.05,  # ← 主图更大
    cbar_height=0.022, cbar_pad=0.008,
    # ----- 三处局部放大（位置与范围）-----
    INSET_RECTS=((0.76,0.58,0.20,0.19),   # 右上
                 (0.07,0.14,0.22,0.18),   # 左下
                 (0.82,0.10,0.16,0.18)),  # 右下
    ROI_FRACS=((0.08,0.24,0.18,0.36),     # 左前
               (0.42,0.58,0.62,0.80),     # 中上
               (0.70,0.88,0.40,0.56))     # 右中
):
    cm = mpl.cm.get_cmap(cmap)

    # ---------- 小工具：3D 数据点 -> Figure 坐标 ----------
    def data_to_fig_xy(ax, x, y, z=0.0):
        Xp, Yp, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        xd, yd = ax.transData.transform((Xp, Yp))
        return ax.figure.transFigure.inverted().transform((xd, yd))

    # ---------- 归一化 ----------
    vals = pd.to_numeric(gdf[value_field], errors="coerce").astype(float).values
    vals = np.where(np.isfinite(vals), vals, np.nan)
    if np.nanmin(vals) < -1e-9 or np.nanmax(vals) > 1 + 1e-9:
        lo = np.nanpercentile(vals, 2); hi = np.nanpercentile(vals, 98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(vals), np.nanmax(vals)
        v01 = np.clip((vals - lo) / (hi - lo), 0, 1)
    else:
        v01 = np.clip(vals, 0, 1)

    # ---------- 版式 ----------
    ax_bottom = bottom_margin_frac + cbar_height + cbar_pad
    ax_top    = 1.0 - top_margin_frac
    ax_height = max(0.05, ax_top - ax_bottom)
    ax_rect = (left_margin, ax_bottom, 1.0 - left_margin - right_margin, ax_height)

    fig = plt.figure(figsize=figsize)
    fig.text(0.5, 1.0 - top_margin_frac, title, ha='center', va='top', fontsize=21)

    # ---------- 主图 ----------
    ax = fig.add_axes(ax_rect, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    try: ax.set_proj_type('ortho')
    except Exception: pass

    tb = gdf.total_bounds; x0,y0,x1,y1 = tb
    xran, yran = x1-x0, y1-y0

    # —— 组装主图棱柱面
    faces, facecols = _build_faces(gdf, v01, h_scale=h_scale, cmap=cmap, light_xy=light_xy)
    coll = Poly3DCollection(faces, facecolors=facecols, edgecolors=(0,0,0,0.35), linewidths=0.2)
    coll.set_zsort('average'); ax.add_collection3d(coll)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(0, 1)
    try: ax.set_box_aspect((xran, yran, z_aspect * max(xran, yran)))
    except Exception: pass
    ax.set_xlabel("X", labelpad=2); ax.set_ylabel("Y", labelpad=2); ax.set_zlabel("Value (0–1)", labelpad=2)
    ax.set_xticks(np.linspace(x0, x1, 5)); ax.set_yticks(np.linspace(y0, y1, 5)); ax.set_zticks(np.linspace(0, 1, 6))
    ax.tick_params(axis='both', which='major', pad=1.0); ax.zaxis.set_tick_params(labelsize=10.5, pad=1.0)

    # —— 主图标出 ROI，并记录中心点（用于连线）
    roi_rects, roi_centers = [], []
    for (fx0,fx1,fy0,fy1) in ROI_FRACS:
        rx0 = x0 + fx0*xran; rx1 = x0 + fx1*xran
        ry0 = y0 + fy0*yran; ry1 = y0 + fy1*yran
        roi_rects.append((rx0,rx1,ry0,ry1))
        _draw_roi_rect_on_main(ax, (rx0,rx1,ry0,ry1), z=0, color='crimson', lw=1.2)
        roi_centers.append(((rx0+rx1)/2.0, (ry0+ry1)/2.0))

    # ---------- 色标（等长且置顶刻度） ----------
    fig.canvas.draw()
    base_corners = np.array([[x0,y0,0.0],[x1,y0,0.0],[x1,y1,0.0],[x0,y1,0.0]])
    Xp, Yp, _ = proj3d.proj_transform(base_corners[:,0], base_corners[:,1], base_corners[:,2], ax.get_proj())
    pts_disp = ax.transData.transform(np.c_[Xp, Yp])
    pts_fig  = fig.transFigure.inverted().transform(pts_disp)
    foot_left, foot_right = float(pts_fig[:,0].min()), float(pts_fig[:,0].max())
    foot_width = foot_right - foot_left
    pad_lr = 0.01 * foot_width
    cbar_left   = max(left_margin, foot_left + pad_lr)
    cbar_width  = max(0.05, foot_width - 2*pad_lr)
    cbar_bottom = bottom_margin_frac
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks(np.linspace(0, 1, 6)); cb.ax.tick_params(length=2, pad=1)
    cb.set_label("Value (0–1)")
    cb.ax.xaxis.set_ticks_position('top'); cb.ax.xaxis.set_label_position('top')

    # ---------- 三个 inset ----------
    from matplotlib.patches import FancyBboxPatch
    connectors = []  # (x0,y0,x1,y1) in figure coords

    for rect, (rx0,rx1,ry0,ry1), (cx,cy) in zip(INSET_RECTS, roi_rects, roi_centers):
        # inset 边框（先画白底圆角框，再放 3D 轴）
        pad = 0.006
        bx, by, bw, bh = rect
        box = FancyBboxPatch((bx-pad, by-pad), bw+2*pad, bh+2*pad,
                             boxstyle="round,pad=0.004,rounding_size=0.004",
                             lw=0.8, edgecolor='0.3', facecolor='white',
                             transform=fig.transFigure, zorder=2, alpha=0.95)
        fig.add_artist(box)

        ax_i = fig.add_axes(rect, projection='3d', zorder=3)
        ax_i.view_init(elev=elev, azim=azim)
        try: ax_i.set_proj_type('ortho')
        except Exception: pass

        # ROI 子集
        reps = gdf.geometry.representative_point()
        m = (reps.x>=rx0)&(reps.x<=rx1)&(reps.y>=ry0)&(reps.y<=ry1)
        gsub = gdf.loc[m].copy()
        vsub = np.clip(pd.to_numeric(gsub[value_field], errors='coerce').astype(float).values, 0, 1)

        faces_i, cols_i = _build_faces(gsub, vsub, h_scale=h_scale, cmap=cmap, light_xy=light_xy)
        coll_i = Poly3DCollection(faces_i, facecolors=cols_i, edgecolors=(0,0,0,0.35), linewidths=0.18)
        coll_i.set_zsort('average'); ax_i.add_collection3d(coll_i)

        ax_i.set_xlim(rx0,rx1); ax_i.set_ylim(ry0,ry1); ax_i.set_zlim(0,1)
        try: ax_i.set_box_aspect(((rx1-rx0),(ry1-ry0), z_aspect*max(rx1-rx0, ry1-ry0)))
        except Exception: pass
        # 去除轴元素
        for spine in ("xaxis","yaxis","zaxis"):
            try: getattr(ax_i, spine).pane.set_visible(False)
            except Exception: pass
        ax_i.set_xticks([]); ax_i.set_yticks([]); ax_i.set_zticks([])
        ax_i.set_xlabel(""); ax_i.set_ylabel(""); ax_i.set_zlabel(""); ax_i.grid(False)

        # 记录连接线：主图 ROI 中心 -> inset 中心（figure 坐标）
        x0f, y0f = data_to_fig_xy(ax, cx, cy, z=0.0)
        bbox = ax_i.get_position()
        x1f, y1f = bbox.x0 + bbox.width*0.5, bbox.y0 + bbox.height*0.5
        connectors.append((x0f, y0f, x1f, y1f))

    # 画连接线（figure 坐标）
    for (x0f,y0f,x1f,y1f) in connectors:
        line = mpl.lines.Line2D([x0f,x1f], [y0f,y1f], transform=fig.transFigure,
                                color='0.3', lw=1.0, ls='--', alpha=0.8, zorder=1)
        fig.add_artist(line)

    # 淡化主图网格
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try: axis._axinfo["grid"]["color"] = (0, 0, 0, 0.12)
        except Exception: pass

    fig.savefig(out_png, dpi=360, bbox_inches=None)
    plt.close(fig)

# ========= 主程序 =========
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR    = os.path.join(SCRIPT_DIR, "综合风险评估")
    os.makedirs(OUT_DIR, exist_ok=True)

    TERRAIN_PRECOMPUTED_PATH = r"C:\Users\Administrator\Desktop\输出结果\地形要素势场\comprehensive_potential.tif"
    RIVER_FIELD_PATH         = r"C:\Users\Administrator\Desktop\输出结果\河道要素势场\river_potential_field.tif"
    SOIL_FIELD_PATH          = r"C:\Users\Administrator\Desktop\输出结果\soil_sigmoid_potential\soil_sigmoid_potential.tif"
    GRID_SHP_PATH            = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\郑州网格shp\郑州分析网格.shp"

    RASTER_PATH = os.path.join(OUT_DIR, "final_risk_field.tif")
    if not os.path.exists(RASTER_PATH):
        print("ℹ️ 未发现风险栅格，正在合成……")
        merger = FieldMerger(TERRAIN_PRECOMPUTED_PATH, RIVER_FIELD_PATH, SOIL_FIELD_PATH, OUT_DIR)
        raw_field = merger.merge_fields()
        RASTER_PATH = merger.save_tif(raw_field, filename="final_risk_field.tif")
        print("✅ 合成完成：", RASTER_PATH)
    else:
        print("✅ 使用现有风险栅格：", RASTER_PATH)

    gdf_mean, out_shp = raster_mean_to_grid(
        raster_path=RASTER_PATH,
        grid_shp_path=GRID_SHP_PATH,
        out_dir=OUT_DIR,
        field_name="geo_mean",
        all_touched=True
    )
    print("✅ 已输出网格（含 geo_mean）：", out_shp)

    gdf_mean = gdf_mean.copy()
    gdf_mean["geo_mean_01"], lo, hi = robust_minmax_normalize(gdf_mean["geo_mean"], 0.02, 0.98)
    print(f"ℹ️ 归一化分位区间：2%={lo:.6f}, 98%={hi:.6f}")

    out_png = os.path.join(OUT_DIR, "grid_true3D_16x9_with_insets.png")
    plot_grid_3d_prisms_z01(
        gdf=gdf_mean,
        value_field="geo_mean_01",
        out_png=out_png,
        title="Geographic Vulnerability of Zhengzhou",
        cmap="Blues",
        elev=22, azim=-55,
        proj='ortho',
        z_aspect=0.24, h_scale=1.00,
        figsize=(16, 9),
        top_margin_frac=0.10,
        bottom_margin_frac=0.10,
        left_margin=0.06, right_margin=0.06,
        cbar_height=0.022, cbar_pad=0.008,
        # —— 可按需微调的三个 ROI 与小图位置 ——
        INSET_RECTS=((0.70,0.58,0.26,0.30), (0.06,0.18,0.26,0.26), (0.82,0.10,0.16,0.24)),
        ROI_FRACS=((0.08,0.24,0.18,0.36), (0.42,0.58,0.62,0.80), (0.70,0.88,0.40,0.56))
    )
    print("✅ 已导出成品（含三处局部放大）：", out_png)
