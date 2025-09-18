# -*- coding: utf-8 -*-
"""
综合脆弱性势场（融合四个网格：供应链 / 人口分布 / 流动 / 地理）
- 自动识别网格ID与脆弱性字段；非 0–1 自动归一化
- 线性权重融合 -> v_total_raw，再对融合结果进行二次归一化 -> v_total（0–1）
- 输出：CSV / Shapefile / GeoJSON / PNG / GPKG
- 输出目录：D:\Cascading effect predicate\综合脆弱性\融合结果
- 绘图部分：仅图内文字（标题、图例标签）使用英文；文件名与日志中文/英文按需
"""

import os
import warnings
from typing import Union, Optional, Dict

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============ Matplotlib 字体设置（仅图内英文） ============
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 图内渲染使用英文字体，避免中文缺字
mpl.rcParams['axes.unicode_minus'] = False

# ============ 【用户需要修改的路径】 ============
SUPPLY_PATH   = r"D:\Cascading effect predicate\供应链脆弱性\vulnerability_vector.shp"   # 供应链网格
POP_PATH      = r"D:\Cascading effect predicate\手机信令数据\vulnerability_vector_output\vulnerability_vector.shp"  # 人口分布网格
MOBILITY_PATH = r"D:\Cascading effect predicate\手机信令数据\vulnerability_results\vulnerability_spatial.shp"      # 流动网格
GEO_PATH      = "D:\Cascading effect predicate\地理脆弱性\综合风险评估\Zhengzhou_Grid_GeoVuln_Mean.shp"  # 地理脆弱性网格（含 geo_mean）

# 输出目录（绝对路径）
OUT_DIR = r"D:\Cascading effect predicate\综合脆弱性\融合结果"

# 融合权重（四要素；和将自动归一化为1）
WEIGHTS = {"supply": 0.30, "population": 0.25, "mobility": 0.20, "geo": 0.25}

# 融合后结果的二次归一化策略
# 可选："minmax"（默认）、"quantile"、"none"
TOTAL_NORMALIZE = "minmax"
# 若用分位归一化（抗极端值），可调这里（例如 0.02～0.98）
TOTAL_Q_LOW  = 0.02
TOTAL_Q_HIGH = 0.98

# 输出文件基础名
OUTPUT_BASENAME = "vulnerability_integrated"

# 可选：目标坐标系（如不为空，则强制统一到该CRS）
TARGET_CRS = ""

# ============ 精简警告 ============
warnings.filterwarnings("ignore", category=UserWarning, module="pyogrio.raw")
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")

# ============ 工具函数 ============
def read_any(path: str) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".shp", ".geojson", ".json", ".gpkg"):
        return gpd.read_file(path)
    elif ext == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    raise ValueError("不支持的文件类型: {}".format(ext))

def ensure_geodf(df: Union[gpd.GeoDataFrame, pd.DataFrame],
                 reference_crs: Optional[Union[str, dict]] = None) -> gpd.GeoDataFrame:
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df
        if reference_crs and gdf.crs and str(gdf.crs) != str(reference_crs):
            try:
                gdf = gdf.to_crs(reference_crs)
            except Exception:
                pass
        return gdf
    return gpd.GeoDataFrame(df.copy(), geometry=None, crs=reference_crs)

def detect_id_column(df: Union[gpd.GeoDataFrame, pd.DataFrame]) -> str:
    for c in ["grid_id", "tid", "TID", "id", "ID", "code", "gridcode"]:
        if c in df.columns:
            return c
    raise KeyError("未找到网格ID字段（候选: grid_id/tid/TID/id/ID/code/gridcode）")

def detect_vuln_column(df: Union[gpd.GeoDataFrame, pd.DataFrame]) -> str:
    # 通用名
    for c in ["vulnerability", "vulnerabil", "vulnerability_index"]:
        if c in df.columns:
            return c
    # 简写/自定义
    for c in ["vul", "vuln", "v_index", "vscore", "score"]:
        if c in df.columns:
            return c
    # 地理脆弱性常见命名（如 geo_mean）
    for c in ["geo_mean", "geo_vuln", "geovuln", "geo", "final_risk", "risk", "g_mean"]:
        if c in df.columns:
            return c
    raise KeyError("未找到脆弱性字段（候选: vulnerability/geo_mean/v_index 等）")

def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    v = arr.astype(float)
    mask = np.isfinite(v)
    if mask.sum() == 0:
        return np.zeros_like(v)
    lo, hi = np.nanmin(v[mask]), np.nanmax(v[mask])
    if hi <= lo:
        return np.zeros_like(v)
    out = (v - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

def normalize_quantile(arr: np.ndarray, q_low=0.02, q_high=0.98) -> np.ndarray:
    v = arr.astype(float)
    mask = np.isfinite(v)
    if mask.sum() == 0:
        return np.zeros_like(v)
    lo, hi = np.nanquantile(v[mask], [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return normalize_minmax(v)
    out = (v - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

def normalize_0_1(series: pd.Series) -> pd.Series:
    """单列 min-max 归一化（用于各子层进入融合前）"""
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    if v.size == 0 or np.all(np.isnan(v)):
        return pd.Series(np.zeros_like(v), index=series.index)
    valid = v[~np.isnan(v)]
    vmin, vmax = np.min(valid), np.max(valid)
    n = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)
    n = np.clip(np.nan_to_num(n, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return pd.Series(n, index=series.index)

def load_single_layer(path: str,
                      reference_crs: Optional[Union[str, dict]] = None,
                      tag: str = "") -> gpd.GeoDataFrame:
    df = read_any(path)
    gdf = ensure_geodf(df, reference_crs=reference_crs)

    id_col = detect_id_column(gdf)
    if id_col != "grid_id":
        gdf = gdf.rename(columns={id_col: "grid_id"})
    gdf["grid_id"] = gdf["grid_id"].astype(str).str.strip()

    vcol = detect_vuln_column(gdf)
    vvals = pd.to_numeric(gdf[vcol], errors="coerce").fillna(0.0)
    if (vvals.min() >= -1e-9 and vvals.max() <= 1+1e-9):
        v_norm = vvals.clip(0.0, 1.0)
    else:
        v_norm = normalize_0_1(vvals)

    gdf[f"v_{tag}"] = v_norm.values

    keep = ["grid_id", f"v_{tag}"]
    if isinstance(gdf, gpd.GeoDataFrame) and ("geometry" in gdf.columns):
        keep.append("geometry")
        out = gdf[keep].copy()
    else:
        out = gdf[keep].copy()

    # grid_id 唯一性检查（若重复则保留首次）
    dups = out["grid_id"].astype(str).duplicated(keep=False)
    if dups.any():
        cnt = int(dups.sum())
        ids = out.loc[dups, "grid_id"].astype(str).value_counts().head(5).index.tolist()
        print(f"⚠️ {tag} 存在重复 grid_id：{cnt} 条，例如 {ids} ... 将保留首次出现")
        out.drop_duplicates("grid_id", keep="first", inplace=True)

    return out

def resolve_out_dir(out_dir: str) -> str:
    """
    输出目录解析：
    - 绝对路径（带盘符或 UNC）→ 原样使用
    - 相对路径 → 相对当前工作目录解析
    """
    p = out_dir.replace("/", os.sep).replace("\\", os.sep).strip()
    is_abs = (len(p) > 1 and p[1] == ':') or p.startswith('\\\\')
    if is_abs:
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(os.getcwd(), p))

def normalize_weights(w: Dict[str, float],
                      required=("supply", "population", "mobility", "geo")) -> Dict[str, float]:
    for k in required:
        if k not in w:
            raise KeyError(f"WEIGHTS 缺少键：{k}")
    s = sum(float(w[k]) for k in required)
    if s <= 0:
        raise ValueError("WEIGHTS 权重和必须为正数")
    if abs(s - 1.0) > 1e-6:
        print(f"ℹ️ 权重和为 {s:.6f}，已自动归一化为 1")
    return {k: float(w[k]) / s for k in required}

def finalize_and_save(gdf: gpd.GeoDataFrame, out_dir: str, weights: dict) -> None:
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 1) 融合（加权和）→ v_total_raw（未归一化）
    v_total_raw = (
        gdf.get("v_supply", 0.0)     * weights.get("supply", 0.0) +
        gdf.get("v_population", 0.0) * weights.get("population", 0.0) +
        gdf.get("v_mobility", 0.0)   * weights.get("mobility", 0.0) +
        gdf.get("v_geo", 0.0)        * weights.get("geo", 0.0)
    ).astype(float)

    gdf["v_total_raw"] = v_total_raw.values

    # 2) 对融合结果做二次归一化 → v_total（0–1）
    raw_arr = v_total_raw.to_numpy()
    if TOTAL_NORMALIZE.lower() == "quantile":
        v_total = normalize_quantile(raw_arr, TOTAL_Q_LOW, TOTAL_Q_HIGH)
    elif TOTAL_NORMALIZE.lower() == "minmax":
        v_total = normalize_minmax(raw_arr)
    else:
        # 不做二次归一化时，为安全仍 clip 到 [0,1]
        v_total = np.clip(np.nan_to_num(raw_arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    gdf["v_total"] = v_total

    base = OUTPUT_BASENAME
    title_en = "Integrated Vulnerability Field"
    cbar_label_en = "Integrated Vulnerability (0–1)"

    # CSV（包含 v_total_raw 与 v_total）
    csv_path = os.path.join(out_dir, f"{base}.csv")
    gdf.drop(columns="geometry", errors="ignore").to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("✅ 已保存 CSV:", csv_path)

    # Shapefile（显式驱动、编码；几何修复；侧车文件自检；若缺 .cpg 则补写）
    shp_path = os.path.join(out_dir, f"{base}.shp")
    gdf_shp = gdf.rename(columns={
        "v_population": "v_pop",
        "v_mobility":   "v_move",
        "v_supply":     "v_sup",
        "v_geo":        "v_geo",
        "v_total_raw":  "v_t_raw",   # 兼容 10 字符限制
        "v_total":      "v_total"
    }).copy()

    # 几何修复（防止无效几何导致写出失败）
    if "geometry" in gdf_shp.columns and gdf_shp.geometry.notna().any():
        try:
            gdf_shp["geometry"] = gdf_shp.geometry.buffer(0)
        except Exception:
            pass

    # CRS 提示
    if getattr(gdf_shp, "crs", None) is None:
        print("⚠️ 当前数据无 CRS，写出将缺少 .prj；如需 .prj，请设置 TARGET_CRS 或源层 CRS。")

    # 显式指定驱动与编码
    gdf_shp.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    # 若缺 .cpg，则补写 UTF-8 声明
    base_noext = os.path.splitext(shp_path)[0]
    cpg_path = base_noext + ".cpg"
    if not os.path.exists(cpg_path):
        try:
            with open(cpg_path, "w", encoding="ascii") as f:
                f.write("UTF-8")
        except Exception:
            pass
    # 侧车文件自检
    sidecars = [".shp", ".dbf", ".shx", ".prj", ".cpg"]
    missing = [ext for ext in sidecars if not os.path.exists(base_noext + ext)]
    print("✅ 已保存 Shapefile:", shp_path)
    if missing:
        print("⚠️ Shapefile 缺少以下配套文件：", ", ".join(missing),
              "（GeoJSON/GPKG 正常可用；必要时设置 CRS 后再导出 SHP）")

    # GeoJSON（UTF-8，单文件稳定）
    geojson_path = os.path.join(out_dir, f"{base}.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON", encoding="utf-8")
    print("✅ 已保存 GeoJSON:", geojson_path)

    # GPKG（现代格式，字段限制少，跨软件更稳）
    gpkg_path = os.path.join(out_dir, f"{base}.gpkg")
    gdf.to_file(gpkg_path, layer=base, driver="GPKG")
    print("✅ 已保存 GPKG:", gpkg_path)

    # PNG（仅图内文字用英文；显示 v_total 0–1）
    if "geometry" in gdf.columns and gdf.geometry.notna().any():
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(
            column="v_total", cmap="OrRd", legend=True,
            legend_kwds={"label": cbar_label_en, "orientation": "horizontal"},
            ax=ax, edgecolor="0.8", linewidth=0.2, vmin=0, vmax=1, rasterized=True
        )
        ax.set_title(title_en, fontsize=16)
        ax.set_axis_off()
        fig.tight_layout()

        png_path = os.path.join(out_dir, f"{base}.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("✅ 已保存 PNG:", png_path)
    else:
        print("ℹ️ 未发现几何信息，已跳过地图绘制。")

# ============ 主流程 ============
def main():
    # 参考 CRS
    supply_raw = read_any(SUPPLY_PATH)
    if isinstance(supply_raw, gpd.GeoDataFrame) and supply_raw.crs:
        ref_crs = TARGET_CRS if TARGET_CRS else supply_raw.crs
    else:
        ref_crs = TARGET_CRS if TARGET_CRS else None
        if ref_crs is None:
            for p in [POP_PATH, MOBILITY_PATH, GEO_PATH]:
                tmp = read_any(p)
                if isinstance(tmp, gpd.GeoDataFrame) and tmp.crs:
                    ref_crs = tmp.crs
                    break

    # 逐层读取
    gdf_supply   = load_single_layer(SUPPLY_PATH,   reference_crs=ref_crs, tag="supply")
    gdf_pop      = load_single_layer(POP_PATH,      reference_crs=ref_crs, tag="population")
    gdf_mobility = load_single_layer(MOBILITY_PATH, reference_crs=ref_crs, tag="mobility")
    gdf_geo      = load_single_layer(GEO_PATH,      reference_crs=ref_crs, tag="geo")

    # 选择一个含几何的源作为几何参考
    geometry_source = None
    for g in (gdf_supply, gdf_pop, gdf_mobility, gdf_geo):
        if isinstance(g, gpd.GeoDataFrame) and ("geometry" in g.columns):
            geometry_source = g[["grid_id", "geometry"]].drop_duplicates("grid_id")
            break

    # 仅属性表
    s_attr = gdf_supply[["grid_id", "v_supply"]].copy()
    p_attr = gdf_pop[["grid_id", "v_population"]].copy()
    m_attr = gdf_mobility[["grid_id", "v_mobility"]].copy()
    g_attr = gdf_geo[["grid_id", "v_geo"]].copy()

    # 外连接汇总
    df_merge = (
        s_attr.merge(p_attr, on="grid_id", how="outer")
              .merge(m_attr, on="grid_id", how="outer")
              .merge(g_attr, on="grid_id", how="outer")
    )
    for c in ["v_supply", "v_population", "v_mobility", "v_geo"]:
        df_merge[c] = pd.to_numeric(df_merge[c], errors="coerce").fillna(0.0)

    # 组装 GeoDataFrame
    if geometry_source is not None:
        gdf_merge = geometry_source.merge(df_merge, on="grid_id", how="right")
        gdf_merge = gpd.GeoDataFrame(gdf_merge, geometry="geometry", crs=ref_crs)
    else:
        gdf_merge = gpd.GeoDataFrame(df_merge, geometry=None, crs=ref_crs)

    # 输出目录
    out_dir_final = resolve_out_dir(OUT_DIR)

    # 权重归一化
    weights_final = normalize_weights(WEIGHTS)

    # 保存（含二次归一化 v_total）
    finalize_and_save(gdf_merge, out_dir_final, weights_final)

    # 日志
    print("\n=== 融合完成（综合脆弱性势场） ===")
    print("- 记录数：{}".format(len(gdf_merge)))
    if "v_total" in gdf_merge.columns:
        print("- 融合值（raw）范围：{:.4f} - {:.4f}".format(gdf_merge["v_total_raw"].min(), gdf_merge["v_total_raw"].max()))
        print("- 归一化后范围：{:.4f} - {:.4f}".format(gdf_merge["v_total"].min(), gdf_merge["v_total"].max()))
    print(f"- 融合后归一化方式：{TOTAL_NORMALIZE}" + (f"（q_low={TOTAL_Q_LOW}, q_high={TOTAL_Q_HIGH}）" if TOTAL_NORMALIZE=='quantile' else ""))
    print("- 权重（归一化后）：", weights_final)
    print("- 输出目录：{}".format(out_dir_final))

if __name__ == "__main__":
    main()
