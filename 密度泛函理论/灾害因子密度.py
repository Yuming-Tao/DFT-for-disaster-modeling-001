# -*- coding: utf-8 -*-
"""
NHPP + Monte Carlo（仅网格着色；线性色标轴）
- 支持两种“增量”定义（绘制面板用）：
  1) arrival:         每小时到达量 inc(h) ~ Poisson(mu_h * p)
  2) inventory_diff:  邻近两时刻库存差 ΔN(h) = N(h) - N(h-1)
- 输出：两张面板（增量 / 衰减库存），各自共享线性色标
- 新增：只对【库存面板】按“每一帧”生成统计表（跨所有网格求：均值、标准差、极差、分位数），共 12 行
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import ticker

# =================== 路径与主控参数 ===================
GRID_PATH   = r"D:\Cascading effect predicate\综合脆弱性\融合结果\vulnerability_integrated.geojson"
LAMBDA_CSV  = r"D:\Cascading effect predicate\密度泛函理论\hourly_mu_calibrated.csv"
HAZARD_CSV  = r"D:\Cascading effect predicate\密度泛函理论\hazard_outputs\hazard_density_rho.csv"  # 可选：含 'grid_id','p'
OUT_DIR     = r"D:\Cascading effect predicate\密度泛函理论\nhpp_grid_panels_linear"
os.makedirs(OUT_DIR, exist_ok=True)

START_FOR_FALLBACK = "2021-07-17 08:00"

# —— 关键开关：增量模式（仅影响增量面板的绘制） —— #
# "arrival" 或 "inventory_diff"
INCREMENT_MODE = "inventory_diff"

# 空间偏好
USE_P_FROM_HAZARD_CSV = True
PREF_COLUMN = "v_total"
USE_AREA_IN_P = False

# 衰减（小时半衰期）
HALF_LIFE_H = 12.0

# 选帧（12 帧：3×4）
FRAME_STRIDE_H   = 3
PEAK_PERCENTILE  = 85
MAX_FRAMES       = 12
ALWAYS_INCLUDE_PEAK = True

PANEL_ROWS = 3
PANEL_COLS = 4

# 单位与外观
PER_KM2 = False
ZERO_FILL_COLOR = "#f4f4f4"   # 仅对到达量模式的 <=0 使用；差分模式不用
BOUND_COLOR = "#444444"
BOUND_LW    = 0.6
BOUND_ALPHA = 0.6

# 线性色标（增量与库存可用不同策略）
# 到达量（非负，顺序色带）
Q_LO_ARR, Q_HI_ARR = 1.0, 95.0
TOPPAD_ARR = 1.00
VMIN_ARR, VMAX_ARR = None, None

# 差分（可正可负，零居中对称）
Q_ABS_HI_DIFF = 99.0   # |ΔN| 的上分位
TOPPAD_DIFF   = 1.00
VABS_DIFF     = None   # 若指定则强制对称范围 [-VABS, +VABS]

# 库存（非负，顺序色带）
Q_LO_DEC, Q_HI_DEC = 1.0, 99.5
TOPPAD_DEC = 1.10
VMIN_DEC, VMAX_DEC = None, None

# 画布
FIG_W, FIG_H = 18, 10.5
DPI = 240
FONT = "Times New Roman"
TITLE_FONTSIZE = 21
MARGIN = 0.08
WSPACE = 0.05
HSPACE = 0.08

TARGET_PROJ_CRS = "EPSG:3857"
BASE_SEED = 1234

# ===== “每帧统计表”分位数参数 =====
QUANTILES = [5, 25, 50, 75, 95]  # 百分位

matplotlib.rcParams["font.family"] = FONT
matplotlib.rcParams["axes.unicode_minus"] = False

GRID_ID_CANDS = ["grid_id", "Grid_ID", "gridId", "id", "ID", "gid"]
VAL_COLS      = ["mu_hour", "lambda", "arrival", "arrivals", "count", "value", "rate"]

# =================== 辅助函数 ===================
def find_grid_id_column(df):
    for c in GRID_ID_CANDS:
        if c in df.columns:
            return c
    for c in df.columns:
        if str(c).lower().endswith("id") or str(c).lower()=="gid":
            return c
    df["grid_id"] = np.arange(1, len(df)+1).astype(str)
    return "grid_id"

def _guess_time_column(df: pd.DataFrame):
    best, hits = None, 0
    for c in df.columns:
        ser = pd.to_datetime(df[c], errors="coerce", utc=False)
        h = ser.notna().sum()
        if h > hits:
            best, hits = c, h
    return best if best is not None and hits >= max(3, int(0.5*len(df))) else None

def _pick_value_column(df: pd.DataFrame):
    for c in VAL_COLS:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            return c
    raise RuntimeError("hourly CSV 未找到数值列")

def read_hourly_lambda(csv_path, start_for_fallback: str):
    df = pd.read_csv(csv_path)
    vcol = _pick_value_column(df)
    vals = pd.to_numeric(df[vcol], errors="coerce").fillna(0.0)
    tcol = _guess_time_column(df)
    if tcol is not None:
        t = pd.to_datetime(df[tcol], errors="coerce", utc=False)
        ok = t.notna() & vals.notna()
        df2 = pd.DataFrame({"time": t[ok].dt.floor("H"), "val": vals[ok]})
        hourly = df2.groupby("time", as_index=False)["val"].sum().sort_values("time").reset_index(drop=True)
        return hourly["time"].to_numpy(), hourly["val"].to_numpy(float)
    else:
        print("⚠️ hourly CSV 未识别到时间列，将按 START_FOR_FALLBACK 生成逐小时序列。")
        H = len(vals)
        t0 = pd.to_datetime(start_for_fallback)
        times = pd.date_range(start=t0, periods=H, freq="H")
        return times.to_numpy(), vals.to_numpy(float)

def ensure_projected(gdf):
    if gdf.crs is None:
        print("⚠️ GRID 没有 CRS；假定 EPSG:4326 再投影到米制。")
        return gdf.set_crs("EPSG:4326", allow_override=True).to_crs(TARGET_PROJ_CRS)
    try:
        if gdf.crs.is_geographic:
            return gdf.to_crs(TARGET_PROJ_CRS)
        return gdf
    except Exception:
        return gdf.to_crs(TARGET_PROJ_CRS)

def make_p(gdf, hazard_csv_path, prefer_hazard_csv=True, pref_col="v_total", use_area=False):
    if prefer_hazard_csv and hazard_csv_path and os.path.exists(hazard_csv_path):
        h = pd.read_csv(hazard_csv_path)
        if "grid_id" in h.columns and "p" in h.columns:
            gid_col = find_grid_id_column(gdf)
            g = gdf.merge(h[["grid_id","p"]].astype({"grid_id":str}),
                          left_on=gid_col, right_on="grid_id", how="left")
            p = pd.to_numeric(g["p"], errors="coerce").fillna(0.0).to_numpy(float)
            if p.sum() > 0:
                return p / p.sum()
    if pref_col not in gdf.columns:
        print("⚠️ 缺少偏好列 v_total，改用均匀分布 p。")
        return np.full(len(gdf), 1.0/len(gdf))
    w = pd.to_numeric(gdf[pref_col], errors="coerce").fillna(0.0).to_numpy(float)
    if use_area:
        if "area_m2" not in gdf.columns and "geometry" in gdf.columns:
            gdf["area_m2"] = gdf.geometry.area
        a = gdf["area_m2"].to_numpy(float) if "area_m2" in gdf.columns else np.ones(len(gdf))
        w = w * np.clip(a, 1e-12, None)
    s = w.sum()
    return (w/s) if s>0 else np.full(len(w), 1.0/len(w))

def select_frames(times_h, mu_hour, stride_h, peak_pct, max_frames, include_peak=True):
    H = len(times_h)
    idx_all = np.arange(H)
    stride_idx = idx_all[::max(1, int(stride_h))]
    thr = np.percentile(mu_hour, peak_pct)
    peak_idx = np.where(mu_hour >= thr)[0]
    sel = np.unique(np.concatenate([stride_idx, peak_idx]))
    if include_peak:
        k_peak = int(np.argmax(mu_hour))
        sel = np.unique(np.concatenate([sel, [k_peak]]))
    if len(sel) > max_frames:
        pick = np.linspace(0, len(sel)-1, max_frames)
        sel = sel[np.round(pick).astype(int)]
    return sel.astype(int)

def rng_for_hour(base_seed, h):
    return np.random.default_rng(base_seed + int(h))

def draw_increment_counts(mu_h, p, h):
    rng = rng_for_hour(BASE_SEED, h)
    lam = np.maximum(mu_h * p, 0.0)
    return rng.poisson(lam)

# =================== 色图 ===================
def make_seq_cmap_lb_yrp():
    stops = [
        (0.00, '#e6f4ff'),
        (0.15, '#cfe9ff'),
        (0.35, '#89c2ff'),
        (0.50, '#fde047'),
        (0.75, '#ef4444'),
        (1.00, '#7c3aed')
    ]
    cmap = LinearSegmentedColormap.from_list('lb_yrp', stops, N=256)
    cmap.set_bad('#f2f2f2')
    return cmap

def make_div_cmap_blue_white_pos_yrp():
    stops = [
        (0.00, '#1e40af'),
        (0.25, '#93c5fd'),
        (0.50, '#ffffff'),
        (0.62, '#fde047'),
        (0.80, '#ef4444'),
        (1.00, '#7c3aed')
    ]
    cmap = LinearSegmentedColormap.from_list('div_bw_yrp', stops, N=256)
    cmap.set_bad('#f2f2f2')
    return cmap

def compute_norm_seq(values_list, q_lo, q_hi, vmin_override, vmax_override, toppad):
    allv = np.concatenate([v[np.isfinite(v) & (v >= 0)] for v in values_list]) if values_list else np.array([0.0, 1.0])
    if allv.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.percentile(allv, q_lo)
        vmax = np.percentile(allv, q_hi)
        vmin = max(0.0, float(vmin))
        if vmax <= vmin:
            vmax = vmin + 1.0
        vmax *= float(toppad)
    if vmin_override is not None: vmin = float(vmin_override)
    if vmax_override is not None: vmax = float(vmax_override)
    return Normalize(vmin=vmin, vmax=vmax), vmin, vmax

def compute_norm_diff(values_list, q_abs_hi, toppad, vabs_override):
    allv = np.concatenate([v[np.isfinite(v)] for v in values_list]) if values_list else np.array([0.0])
    if allv.size == 0:
        vabs = 1.0
    else:
        vabs = np.percentile(np.abs(allv), q_abs_hi)
        vabs = max(1e-12, float(vabs)) * float(toppad)
    if vabs_override is not None:
        vabs = float(vabs_override)
    return Normalize(vmin=-vabs, vmax=+vabs), -vabs, +vabs

def add_shared_colorbar(fig, axes, cmap, norm, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.026, pad=0.016)
    cb.set_label(label, fontsize=12)
    cb.locator = ticker.MaxNLocator(nbins=6)
    cb.formatter = ticker.ScalarFormatter(useMathText=True)
    cb.update_ticks()
    return cb

def outer_boundary(gdf_proj: gpd.GeoDataFrame) -> gpd.GeoSeries:
    dissolved = gdf_proj.assign(_one=1).dissolve(by="_one")
    return dissolved.boundary

# ===== 新增：按“帧”统计（跨所有网格） =====
def frame_stats_from_values_dict(values_dict, sel_idx, times, quantiles):
    """
    逐帧生成统计（对每一帧，跨所有网格做聚合）：
    返回 DataFrame，每行一帧，列含：time_str, frame_idx, n_grids, mean, std, min, max, range, qXX...
    """
    rows = []
    for h in sel_idx:
        arr = values_dict.get(h, None)
        if arr is None:
            continue
        v = np.asarray(arr, dtype=float)
        m = np.isfinite(v)
        vv = v[m]
        if vv.size == 0:
            continue
        row = {
            "frame_idx": int(h),
            "time_str": pd.to_datetime(times[h]).strftime("%Y-%m-%d %H:%M"),
            "n_grids": int(vv.size),
            "mean": float(np.nanmean(vv)),
            "std":  float(np.nanstd(vv, ddof=0)),
            "min":  float(np.nanmin(vv)),
            "max":  float(np.nanmax(vv)),
        }
        row["range"] = row["max"] - row["min"]
        qvals = np.nanpercentile(vv, np.array(quantiles, dtype=float), method="linear")
        for i, qv in enumerate(quantiles):
            row[f"q{int(qv):02d}"] = float(qvals[i])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)

# =================== 主流程 ===================
def main():
    # 1) 读网格并投影
    gdf = gpd.read_file(GRID_PATH)
    gid_col = find_grid_id_column(gdf)
    gdf[gid_col] = gdf[gid_col].astype(str)
    gdf_proj = ensure_projected(gdf)
    bbox = gdf_proj.total_bounds

    # 面积
    gdf_proj["area_m2"] = gdf_proj.geometry.area
    a_km2 = gdf_proj["area_m2"].to_numpy(float) / 1e6

    # 外边界
    region_boundary = outer_boundary(gdf_proj)

    # 2) 空间偏好 p
    p = make_p(gdf_proj, HAZARD_CSV, prefer_hazard_csv=USE_P_FROM_HAZARD_CSV,
               pref_col=PREF_COLUMN, use_area=USE_AREA_IN_P)
    p = np.clip(p, 0.0, None)
    p = p / p.sum() if p.sum()>0 else np.full_like(p, 1.0/len(p))
    print(f"sum p = {p.sum():.6f} (should be 1.0)")

    # 3) 每小时到达期望
    times_h, mu_hour = read_hourly_lambda(LAMBDA_CSV, start_for_fallback=START_FOR_FALLBACK)
    H = len(times_h)
    print(f"hours loaded = {H}, total expected arrivals = {mu_hour.sum():.0f}")

    # 4) 选帧
    sel_idx = select_frames(times_h, mu_hour, FRAME_STRIDE_H, PEAK_PERCENTILE,
                            MAX_FRAMES, include_peak=ALWAYS_INCLUDE_PEAK)
    print(f"selected {len(sel_idx)} frames (indices): {sel_idx}")

    # 5) 扫过小时，逐步计算 N、ΔN
    kappa = np.log(2.0) / max(HALF_LIFE_H, 1e-6)
    N = np.zeros(len(p), dtype=float)         # 库存
    prev_dec_v = np.zeros(len(p), dtype=float)  # N(h-1) 的“每单位”版
    vals_inc = {}     # "inc"：arrival 或 ΔN（用于绘图）
    vals_deci = {}    # 库存（用于绘图与帧统计）

    for h in range(H):
        # 到达（整数）
        inc = draw_increment_counts(mu_hour[h], p, h)

        # 更新库存
        N = np.exp(-kappa) * N + inc

        # 输出尺度
        if PER_KM2:
            inc_v = inc / (a_km2 + 1e-12)       # 到达量（非负）
            dec_v = N   / (a_km2 + 1e-12)       # 库存（非负）
        else:
            inc_v = inc.astype(float)
            dec_v = N.astype(float)

        # 差分：ΔN = dec_v - prev_dec_v  （可能为负）
        diff_v = dec_v - prev_dec_v

        # 视模式存入“增量”；库存恒存
        if h in sel_idx:
            if INCREMENT_MODE == "inventory_diff":
                vals_inc[h] = diff_v.copy()
            else:
                vals_inc[h] = inc_v.copy()
            vals_deci[h] = dec_v.copy()

        prev_dec_v = dec_v

    # ===== 新增：生成“每帧（12 帧）统计表”，针对库存面板（跨网格求统计） =====
    df_frame_stats = frame_stats_from_values_dict(
        values_dict=vals_deci,
        sel_idx=sel_idx,
        times=times_h,
        quantiles=QUANTILES
    )
    out_frame_csv = Path(OUT_DIR) / "stats_inventory_frames.csv"
    df_frame_stats.to_csv(out_frame_csv, index=False, encoding="utf-8-sig")
    print("✅ saved frame-wise inventory stats:", out_frame_csv)

    # 6) 共享色标
    if INCREMENT_MODE == "inventory_diff":
        cmap_inc = make_div_cmap_blue_white_pos_yrp()
        inc_list = [vals_inc[h] for h in sel_idx if h in vals_inc]
        norm_inc, vmin_inc, vmax_inc = compute_norm_diff(
            inc_list, Q_ABS_HI_DIFF, TOPPAD_DIFF, VABS_DIFF
        )
        inc_label = "Inventory change per period" + (" (per km²)" if PER_KM2 else " (per grid)")
    else:
        cmap_inc = make_seq_cmap_lb_yrp()
        inc_list = [vals_inc[h] for h in sel_idx if h in vals_inc]
        norm_inc, vmin_inc, vmax_inc = compute_norm_seq(
            inc_list, Q_LO_ARR, Q_HI_ARR, VMIN_ARR, VMAX_ARR, TOPPAD_ARR
        )
        inc_label = "Hourly arrivals" + (" per km²" if PER_KM2 else " per grid")

    cmap_dec = make_seq_cmap_lb_yrp()
    dec_list = [vals_deci[h] for h in sel_idx if h in vals_deci]
    norm_dec, vmin_dec, vmax_dec = compute_norm_seq(
        dec_list, Q_LO_DEC, Q_HI_DEC, VMIN_DEC, VMAX_DEC, TOPPAD_DEC
    )

    unit_label = "per km²" if PER_KM2 else "per grid"
    print(f"[incremental/{INCREMENT_MODE}] shared range: {norm_inc.vmin:.4g} ~ {norm_inc.vmax:.4g} (linear)")
    print(f"[decayed]                  shared range: {vmin_dec:.4g} ~ {vmax_dec:.4g} (linear, {unit_label})")

    # 7) 画面板（增量）
    fig_inc, axes_inc = plt.subplots(PANEL_ROWS, PANEL_COLS, figsize=(FIG_W, FIG_H), dpi=DPI)
    plt.subplots_adjust(left=MARGIN, right=1-MARGIN, top=1-MARGIN, bottom=MARGIN, wspace=WSPACE, hspace=HSPACE)
    axes_inc = np.atleast_2d(axes_inc)

    for k, h in enumerate(sel_idx):
        r, c = divmod(k, PANEL_COLS); ax = axes_inc[r, c]
        if h not in vals_inc:
            ax.axis("off"); continue
        val = vals_inc[h]

        if INCREMENT_MODE == "arrival":
            zeros = ~np.isfinite(val) | (val <= 0)
            if zeros.any():
                gdf_proj.loc[zeros].plot(ax=ax, color=ZERO_FILL_COLOR, edgecolor='none', linewidth=0)
            pos = ~zeros
            if pos.any():
                gdf_proj.loc[pos, "___val___"] = val[pos]
                gdf_proj.loc[pos].plot(ax=ax, column="___val___", cmap=cmap_inc, norm=norm_inc, edgecolor='none')
        else:
            m = np.isfinite(val)
            gdf_proj.loc[m, "___val___"] = val[m]
            gdf_proj.loc[m].plot(ax=ax, column="___val___", cmap=cmap_inc, norm=norm_inc, edgecolor='none')
            if (~m).any():
                gdf_proj.loc[~m].plot(ax=ax, color="#f2f2f2", edgecolor='none', linewidth=0)

        try:
            region_boundary.plot(ax=ax, color=BOUND_COLOR, linewidth=BOUND_LW, alpha=BOUND_ALPHA)
        except Exception:
            pass

        ax.set_xticks([]); ax.set_yticks([])
        if bbox is not None:
            ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect("equal", adjustable="box")
        tlabel = pd.to_datetime(times_h[h]).strftime("%Y-%m-%d %H:%M")
        title = f"ΔN {tlabel}" if INCREMENT_MODE=="inventory_diff" else f"Hour {tlabel}"
        ax.set_title(title, fontsize=12)

    for k in range(len(sel_idx), PANEL_ROWS*PANEL_COLS):
        r, c = divmod(k, PANEL_COLS); axes_inc[r, c].axis("off")

    fig_inc.suptitle(
        ("Inventory change by period" if INCREMENT_MODE=="inventory_diff" else f"Hazard factor hourly arrivals {unit_label}")
        + " selected hours",
        fontsize=TITLE_FONTSIZE
    )
    add_shared_colorbar(fig_inc, axes_inc, cmap_inc, norm_inc, label=inc_label)
    out_inc = Path(OUT_DIR) / ("panel_grid_incremental_diff_linear.png" if INCREMENT_MODE=="inventory_diff"
                               else "panel_grid_incremental_linear.png")
    fig_inc.savefig(out_inc, dpi=DPI)
    plt.close(fig_inc)

    # 8) 画面板（库存）
    fig_dec, axes_dec = plt.subplots(PANEL_ROWS, PANEL_COLS, figsize=(FIG_W, FIG_H), dpi=DPI)
    plt.subplots_adjust(left=MARGIN, right=1-MARGIN, top=1-MARGIN, bottom=MARGIN, wspace=WSPACE, hspace=HSPACE)
    axes_dec = np.atleast_2d(axes_dec)

    for k, h in enumerate(sel_idx):
        r, c = divmod(k, PANEL_COLS); ax = axes_dec[r, c]
        if h not in vals_deci:
            ax.axis("off"); continue
        val = vals_deci[h]
        m = np.isfinite(val)
        gdf_proj.loc[m, "___val___"] = val[m]
        gdf_proj.loc[m].plot(ax=ax, column="___val___", cmap=make_seq_cmap_lb_yrp(), norm=norm_dec, edgecolor='none')
        if (~m).any():
            gdf_proj.loc[~m].plot(ax=ax, color="#f2f2f2", edgecolor='none', linewidth=0)

        try:
            region_boundary.plot(ax=ax, color=BOUND_COLOR, linewidth=BOUND_LW, alpha=BOUND_ALPHA)
        except Exception:
            pass

        ax.set_xticks([]); ax.set_yticks([])
        if bbox is not None:
            ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect("equal", adjustable="box")
        tlabel = pd.to_datetime(times_h[h]).strftime("%Y-%m-%d %H:%M")
        ax.set_title(f"Decayed to {tlabel}", fontsize=12)

    for k in range(len(sel_idx), PANEL_ROWS*PANEL_COLS):
        r, c = divmod(k, PANEL_COLS); axes_dec[r, c].axis("off")

    fig_dec.suptitle(f"Hazard factor decayed inventory {('per km²' if PER_KM2 else 'per grid')} selected hours",
                     fontsize=TITLE_FONTSIZE)
    add_shared_colorbar(fig_dec, axes_dec, make_seq_cmap_lb_yrp(), norm_dec,
                        label=f"Inventory {('per km²' if PER_KM2 else 'per grid')}")
    out_dec = Path(OUT_DIR) / "panel_grid_decayed_linear.png"
    fig_dec.savefig(out_dec, dpi=DPI)
    plt.close(fig_dec)

    print("saved figures:", out_inc, "and", out_dec)

if __name__ == "__main__":
    main()
