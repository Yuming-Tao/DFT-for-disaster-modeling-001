# integrator.py —— 供应链（企业→网格）整合：仅采用“风险扩散”策略
# 1) 企业→网格聚合：0.6*均值(>0)+0.4*最大值
# 2) 非空网格归一化到 [eps,1]（默认 eps=0.02）
# 3) 合并到全量网格后，用 Queen 邻接的“邻居均值×alpha”对空网格做风险扩散（可多步）
# 4) 导出 CSV / Shapefile / GeoJSON（Shapefile 字段名≤10）

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

from geo_processor import (
    load_fixed_grid,       # 识别网格ID并读入固定网格
    assign_to_fixed_grid,  # 点匹配网格
    get_grid_centers       # 网格中心点(lng, lat)
)

logger = logging.getLogger(__name__)


# ------------------------ 工具函数 ------------------------
def _normalize_floor(series: pd.Series, eps: float = 0.02) -> pd.Series:
    """
    将序列归一化到 [eps, 1]；若无差异则置为 eps。
    仅对“有企业的网格”的聚合值使用；空网格不在此序列中处理。
    """
    v = series.to_numpy(dtype=float)
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if vmax > vmin:
        r = (v - vmin) / (vmax - vmin)
        return pd.Series(eps + (1.0 - eps) * r, index=series.index)
    return pd.Series(np.full_like(v, eps), index=series.index)


def _neighbor_diffuse(gdf: gpd.GeoDataFrame,
                      value_col: str,
                      alpha: float = 0.6,
                      steps: int = 1) -> pd.Series:
    """
    风险扩散：对 value_col 中的缺失/为0的空网格，用 Queen 邻接的“邻居均值×alpha”填充。
    可进行多步扩散（每一步都基于上一步的结果）。
    说明：
      - 仅对“原始为空/NaN”的网格进行填充；已有值的网格保持不变。
      - alpha ∈ (0,1]，建议 0.5~0.7。
    """
    try:
        from libpysal.weights import Queen
        from scipy import sparse  # noqa: F401 仅为确保依赖存在
    except Exception as e:
        raise ImportError("风险扩散需要依赖 libpysal 与 scipy，请先安装：pip install libpysal scipy") from e

    vals = gdf[value_col].to_numpy(dtype=float)
    mask_nan = ~np.isfinite(vals)  # NaN 的位置（空网格）
    # 如果你希望“值为0的空网格也扩散”，打开下一行。当前默认仅填 NaN：
    # mask_nan = (~np.isfinite(vals)) | (vals == 0.0)

    # 权重矩阵（按 gdf 行顺序）
    w = Queen.from_dataframe(gdf, use_index=True)
    W = w.sparse

    filled = vals.copy()
    for _ in range(max(1, int(steps))):
        # 仅对需要填充的位置计算邻居均值
        ind = np.isfinite(filled).astype(float)
        cnt = W.dot(ind)                       # 有效邻居个数
        s = W.dot(np.nan_to_num(filled, nan=0.0))  # 邻居值求和（NaN当0）
        mean_nb = np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0)

        # 只更新原本是 NaN 的网格
        new_vals = filled.copy()
        new_vals[mask_nan] = alpha * mean_nb[mask_nan]

        # 对于仍无邻居/均值为0的，保持为0（或继续下一步迭代）
        filled = np.where(np.isfinite(new_vals), new_vals, 0.0)

    return pd.Series(np.clip(filled, 0.0, 1.0), index=gdf.index)


# ------------------------ 主整合函数 ------------------------
def integrate_results(
    enterprises: pd.DataFrame,
    vulnerability_result: pd.DataFrame,
    csv_path: str = "vulnerability_vector.csv",
    shp_path: str = "vulnerability_vector.shp",
    geojson_path: str = "vulnerability_vector.geojson",
    eps_floor: float = 0.02,     # 非空网格归一化下限（0~1）
    alpha: float = 0.6,          # 风险扩散比例（邻居均值×alpha）
    diffusion_steps: int = 1     # 扩散步数（>=1）
) -> pd.DataFrame:
    """
    企业层脆弱性 → 网格层聚合 → 归一化([eps,1]) → 合并到全量网格 → 风险扩散填充空网格。
    返回：仅“非空网格”的检查表（grid_id, vulnerability, lng, lat）。
    全量网格结果写入 CSV / SHP / GeoJSON。
    """
    try:
        # 1) 载入固定网格并统一ID
        grid_gdf, grid_id_col = load_fixed_grid()
        if grid_id_col != "grid_id":
            if grid_id_col in grid_gdf.columns:
                grid_gdf = grid_gdf.rename(columns={grid_id_col: "grid_id"})
                logger.info("网格ID列由 %s 重命名为 'grid_id'", grid_id_col)
            else:
                raise KeyError("网格中未找到识别到的ID列：{}".format(grid_id_col))

        # 2) 企业坐标 → 网格ID
        valid = vulnerability_result.dropna(subset=["lng", "lat"]).copy()
        coords = list(zip(valid["lng"], valid["lat"]))
        valid["grid_id"] = assign_to_fixed_grid(coords, grid_gdf, grid_id_col="grid_id")
        valid = valid.dropna(subset=["grid_id"])
        logger.info("参与网格分配的有效企业数：%d；匹配成功：%d", len(vulnerability_result), len(valid))

        if valid.empty:
            logger.warning("没有企业匹配到网格，返回空结果")
            return pd.DataFrame(columns=["grid_id", "vulnerability", "lng", "lat"])

        # 3) 网格聚合：均值(>0)与最大值的加权
        group = valid.groupby("grid_id")["vulnerability"]
        stats = group.agg(["mean", "max", "count"]).reset_index()
        mean_nz = group.apply(lambda s: s[s > 0].mean() if (s > 0).any() else np.nan)
        stats["mean_nz"] = mean_nz.values
        stats["mean_nz"] = stats["mean_nz"].fillna(0.0)

        agg_value = 0.6 * stats["mean_nz"] + 0.4 * stats["max"]
        grid_vuln = pd.DataFrame({"grid_id": stats["grid_id"], "vulnerability": agg_value})

        logger.info(
            "网格聚合完成：%d 个网格；聚合范围 [%.6f, %.6f]",
            len(grid_vuln),
            float(grid_vuln["vulnerability"].min()),
            float(grid_vuln["vulnerability"].max())
        )

        # 4) 对“非空网格”归一化到 [eps,1]
        before_min = float(grid_vuln["vulnerability"].min())
        before_max = float(grid_vuln["vulnerability"].max())
        grid_vuln = grid_vuln.assign(
            vulnerability=_normalize_floor(grid_vuln["vulnerability"], eps=eps_floor).values
        )
        logger.info("非空网格归一化：原范围 [%.6f, %.6f] → 新范围 [%.2f, 1.00]",
                    before_min, before_max, eps_floor)

        # 5) 合并到全量网格（空网格先置 NaN，随后用扩散填充）
        full = grid_gdf.merge(grid_vuln, on="grid_id", how="left")
        # 记录哪些是“本来就为空”的网格（NaN）
        is_empty = full["vulnerability"].isna()

        # 6) 风险扩散：邻居均值 × alpha（可多步）
        full["vulnerability"] = _neighbor_diffuse(
            full, "vulnerability", alpha=float(alpha), steps=int(diffusion_steps)
        )

        # 7) 生成非空网格检查表（带中心点）
        centers = get_grid_centers(grid_gdf, grid_id_col="grid_id")
        nonempty_table = grid_vuln.merge(centers, on="grid_id", how="left").fillna({"lng": 0.0, "lat": 0.0})

        # 8) 导出：CSV / SHP / GeoJSON
        centers_all = centers  # 全量中心点
        full_csv = full.merge(centers_all, on="grid_id", how="left")[["grid_id", "vulnerability", "lng", "lat"]]

        # 8.1 CSV
        try:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            full_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info("已保存CSV：%s（风险扩散：alpha=%.2f，steps=%d；原空网格数=%d）",
                        csv_path, alpha, diffusion_steps, int(is_empty.sum()))
        except Exception as e:
            logger.warning("保存CSV失败：%s", e)

        # 8.2 Shapefile（字段名≤10）
        try:
            Path(shp_path).parent.mkdir(parents=True, exist_ok=True)
            shp_gdf = full.copy().rename(columns={"vulnerability": "vulnerabil"})
            shp_gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
            logger.info("已保存Shapefile：%s", shp_path)
        except Exception as e:
            logger.warning("保存Shapefile失败：%s", e)

        # 8.3 GeoJSON（保留完整字段名）
        try:
            Path(geojson_path).parent.mkdir(parents=True, exist_ok=True)
            full.to_file(geojson_path, driver="GeoJSON", encoding="utf-8")
            logger.info("已保存GeoJSON：%s", geojson_path)
        except Exception as e:
            logger.warning("保存GeoJSON失败：%s", e)

        return nonempty_table

    except Exception as e:
        logger.error("整合失败：%s", e, exc_info=True)
        return pd.DataFrame()
