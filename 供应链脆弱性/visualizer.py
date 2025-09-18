# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import contextily as ctx
import geopandas as gpd
import numpy as np
import logging
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === 修改为你的郑州网格路径 ===
ZHENGZHOU_GRID_PATH = r"C:\Users\Administrator\Desktop\郑州数据交付\郑州数据交付\郑州网格shp\郑州分析网格.shp"

# ---------- 全局版式：16:9、10%页边距、Times New Roman ----------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10.5,            # 其余文本默认 10.5pt
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 21,
    "axes.unicode_minus": False,
})

# ---------- 工具：网格ID识别 ----------
def _get_grid_id_column(grid_gdf: gpd.GeoDataFrame) -> str:
    candidates = ['ID', 'id', 'grid_id', 'TID', 'CODE']
    for col in candidates:
        if col in grid_gdf.columns and grid_gdf[col].nunique() == len(grid_gdf):
            return col
    for col in grid_gdf.columns:
        if col != 'geometry' and grid_gdf[col].nunique() == len(grid_gdf):
            logger.warning(f"Using non-standard grid ID: {col}")
            return col
    raise ValueError("No valid grid ID column found!")

# ---------- 显色增强：分位裁剪 + Gamma + 最小着色 ----------
def _smart_red_stretch(s: pd.Series, q_low=0.02, q_high=0.98, gamma=0.65, min_tint=0.06) -> pd.Series:
    """
    返回 0~1 的“可视化强度”，用于着色：
    1) 分位裁剪（去极值） 2) 归一化 3) Gamma 增益（γ<1 提升低端层次）
    4) 最小着色，避免近白
    """
    x = pd.to_numeric(s, errors="coerce").astype(float)
    lo, hi = x.quantile(q_low), x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(hi - lo) or hi <= lo:
            return pd.Series(np.zeros(len(x)), index=x.index)
    y = (x.clip(lo, hi) - lo) / (hi - lo)      # 0..1
    y = np.power(y, gamma)                     # gamma 增益
    y = min_tint + (1.0 - min_tint) * y        # 保底着色
    return pd.Series(y).fillna(min_tint).clip(0, 1)

# 备用：原量纲 0~1 的“分位拉伸”（不做 gamma）
def _quantile_stretch(s: pd.Series, q_low=0.02, q_high=0.98) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce').astype(float)
    lo, hi = x.quantile(q_low), x.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(hi - lo) or hi <= lo:
            return pd.Series(np.zeros(len(x)), index=x.index)
    y = (x.clip(lo, hi) - lo) / (hi - lo)
    return pd.Series(y).fillna(0.0).clip(0, 1)

# ---------- 自定义“增强红系”色带（仍为红色系） ----------
def make_boosted_reds():
    """
    低端加密、避免过白；高端更深。
    依然是红色系：浅粉 -> 珊瑚红 -> 深红
    """
    stops = [
        (0.00, "#fff1f1"),
        (0.08, "#fee2e2"),
        (0.18, "#fdc9c9"),
        (0.35, "#fca5a5"),
        (0.55, "#f87171"),
        (0.75, "#ef4444"),
        (0.90, "#dc2626"),
        (1.00, "#991b1b"),
    ]
    cmap = LinearSegmentedColormap.from_list("boostedReds", stops, N=256)
    cmap.set_bad("#f5f5f5")
    return cmap

# ---------- PDF/统计：scipy 优先 + 回退 ----------
def _safe_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    dens = None
    try:
        from scipy.stats import gaussian_kde
        if np.unique(values).size >= 3:
            kde = gaussian_kde(values)
            d = kde(grid)
            if np.all(np.isfinite(d)) and d.max() > 0:
                dens = d
    except Exception as e:
        logger.info(f"KDE unavailable -> histogram fallback. Reason: {e}")
    if dens is None:
        hist, bins = np.histogram(values, bins=60, range=(0, 1), density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        dens = np.interp(grid, centers, hist, left=0, right=0)
    return dens

def _fit_normal(v: np.ndarray):
    mu = float(np.mean(v))
    sigma = float(np.std(v, ddof=1))
    return mu, sigma

def _normal_pdf(grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0 or not np.isfinite(mu) or not np.isfinite(sigma):
        return np.zeros_like(grid)
    try:
        from scipy.stats import norm
        return norm.pdf(grid, loc=mu, scale=sigma)
    except Exception:
        inv = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        z = (grid - mu) / sigma
        return inv * np.exp(-0.5 * z * z)

def _fit_lognorm(v: np.ndarray):
    v_pos = v[v > 0]
    if v_pos.size < 3:
        return None
    try:
        from scipy.stats import lognorm
        shape, loc, scale = lognorm.fit(v_pos, floc=0)
        return ("scipy", shape, loc, scale)
    except Exception:
        logv = np.log(v_pos)
        m = float(np.mean(logv))
        s = float(np.std(logv, ddof=1))
        return ("fallback", s, 0.0, np.exp(m))

def _lognorm_pdf(grid: np.ndarray, fit) -> np.ndarray:
    if fit is None:
        return np.zeros_like(grid)
    tag, p1, p2, p3 = fit
    try:
        if tag == "scipy":
            from scipy.stats import lognorm
            return lognorm.pdf(grid, p1, loc=p2, scale=p3)
        else:
            s, _, scale = p1, p2, p3
            x = np.clip(grid, 1e-12, None)
            inv = 1.0 / (x * s * np.sqrt(2*np.pi))
            return inv * np.exp(- (np.log(x) - np.log(scale))**2 / (2*s*s))
    except Exception:
        return np.zeros_like(grid)

def _fit_expon(v: np.ndarray):
    try:
        from scipy.stats import expon
        loc, scale = expon.fit(v, floc=0)
        return ("scipy", loc, scale)
    except Exception:
        mean = float(np.mean(v))
        scale = max(mean, 1e-12)
        return ("fallback", 0.0, scale)

def _expon_pdf(grid: np.ndarray, fit) -> np.ndarray:
    tag, loc, scale = fit
    try:
        if tag == "scipy":
            from scipy.stats import expon
            return expon.pdf(grid, loc=loc, scale=scale)
        else:
            lam = 1.0 / max(scale, 1e-12)
            x = np.clip(grid, 0.0, None)
            return lam * np.exp(-lam * x)
    except Exception:
        return np.zeros_like(grid)

def _stats_pack(v: np.ndarray, y_grid: np.ndarray, mu: float, sigma: float):
    try:
        from scipy.stats import skew, kurtosis, kstest
        skew_v  = float(skew(v, bias=False))
        kurt_v  = float(kurtosis(v, fisher=True, bias=False))
        ks_stat = float(kstest(v, 'norm', args=(mu, sigma)).statistic)
    except Exception:
        s = sigma if sigma > 0 else 1.0
        skew_v = float(np.mean(((v - mu) / s) ** 3))
        kurt_v = float(np.mean(((v - mu) / s) ** 4) - 3.0)
        x_sorted = np.sort(v)
        ecdf = np.searchsorted(x_sorted, y_grid, side='right') / x_sorted.size
        z = (y_grid - mu) / s
        t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
        b = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
        poly = (((((b[4]*t + b[3])*t) + b[2])*t + b[1])*t + b[0]) * t
        pdf0 = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*z*z)
        ncdf = 1.0 - pdf0 * poly
        ncdf = np.where(z >= 0, ncdf, 1.0 - ncdf)
        ks_stat = float(np.max(np.abs(ecdf - ncdf)))
    return skew_v, kurt_v, ks_stat

# 小工具：把 "2.8%" -> 0.028
def _pct_to_frac(x):
    if isinstance(x, str) and x.endswith("%"):
        return float(x[:-1]) / 100.0
    return float(x)

# ---------- 主函数 ----------
def visualize_results(grid_data: pd.DataFrame,
                      show_basemap: bool = False,
                      # 显色增强参数（可按需调节）
                      q_low: float = 0.02, q_high: float = 0.98,
                      gamma: float = 0.65,
                      min_tint: float = 0.06,
                      # 布局：主图等高色标 + 紧凑 PDF
                      colorbar_size: str = "2.6%",
                      colorbar_pad: float = 0.06,
                      pdf_width: str   = "18%",
                      gap_between: float = 0.012
                      ) -> str:
    """
    布局：主图 ax（四边 10% 页边距） -> 右侧等高色标 cax -> 更贴近的 PDF 面板 pdf_ax
    显色：分位裁剪 + Gamma + 最小着色（红色系）
    """
    try:
        if not Path(ZHENGZHOU_GRID_PATH).exists():
            raise FileNotFoundError(f"Grid file not found: {ZHENGZHOU_GRID_PATH}")

        grid_gdf = gpd.read_file(ZHENGZHOU_GRID_PATH)
        if grid_gdf.crs != "EPSG:3857":
            grid_gdf = grid_gdf.to_crs("EPSG:3857")

        grid_id_col = _get_grid_id_column(grid_gdf)
        if grid_id_col != 'grid_id':
            grid_gdf = grid_gdf.rename(columns={grid_id_col: 'grid_id'})

        if 'grid_id' not in grid_data.columns or 'vulnerability' not in grid_data.columns:
            raise KeyError("grid_data 必须包含 'grid_id' 与 'vulnerability' 两列")

        # === 计算“可视化强度” ===
        vals = pd.to_numeric(grid_data['vulnerability'], errors='coerce').astype(float)
        grid_df = grid_data.copy()
        grid_df['vuln_vis'] = _smart_red_stretch(vals, q_low=q_low, q_high=q_high, gamma=gamma, min_tint=min_tint)

        # 合并到网格
        merged = grid_gdf.merge(grid_df[['grid_id', 'vuln_vis']], on='grid_id', how='left')
        merged['vuln_vis'] = merged['vuln_vis'].fillna(min_tint)

        # ===== 画布：16:9 + 四边 10% 页边距 =====
        fig = plt.figure(figsize=(16, 9), dpi=300)
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.90)
        ax = fig.add_subplot(1, 1, 1)

        # 等高色标 + 紧凑 PDF
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
        pdf_pad = colorbar_pad + _pct_to_frac(colorbar_size) + gap_between
        pdf_ax = divider.append_axes("right", size=pdf_width, pad=pdf_pad)

        # 自定义增强红系色带
        cmap = make_boosted_reds()

        # 主图
        merged.plot(
            column='vuln_vis',
            ax=ax,
            cmap=cmap,
            alpha=0.95,
            edgecolor='none',     # 去网格描边，增强色块辨识
            linewidth=0.0,
            legend=True,
            legend_kwds={'label': '', 'cax': cax},
            vmin=0, vmax=1
        )

        # 色标（10.5pt）
        ticks = [0.00, 0.25, 0.50, 0.75, 1.00]
        cax.set_ylim(0, 1)
        cax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        cax.yaxis.set_major_formatter(mticker.FixedFormatter([f"{t:.2f}" for t in ticks]))
        cax.set_ylabel("Vulnerability (stretched 0–1)", fontsize=10.5, labelpad=8)
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')
        cax.tick_params(axis='y', labelsize=10.5, left=True, right=False)
        for spine in cax.spines.values():
            spine.set_linewidth(0.9)

        # 可选底图
        if show_basemap:
            try:
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11, alpha=0.5)
            except Exception as e:
                logger.warning(f"Add basemap failed: {e}")

        # ===== PDF（展示可视化强度的分布） =====
        v = merged['vuln_vis'].to_numpy(float)
        v = np.clip(v[np.isfinite(v)], 0.0, 1.0)
        y_grid = np.linspace(0.0, 1.0, 400)
        from numpy import zeros_like

        def _scale(x):
            m = np.nanmax(x)
            return (x / m) if (np.isfinite(m) and m > 0) else np.zeros_like(x)

        pdf_emp = _safe_kde(v, y_grid)
        mu, sigma = _fit_normal(v)
        pdf_norm = _normal_pdf(y_grid, mu, sigma)
        fit_ln   = _fit_lognorm(v)
        pdf_ln   = _lognorm_pdf(y_grid, fit_ln)
        fit_ex   = _fit_expon(v)
        pdf_ex   = _expon_pdf(y_grid, fit_ex)

        x_emp  = _scale(pdf_emp)
        x_norm = _scale(pdf_norm)
        x_ln   = _scale(pdf_ln)
        x_ex   = _scale(pdf_ex)

        pdf_ax.get_shared_y_axes().join(pdf_ax, cax)
        pdf_ax.set_ylim(cax.get_ylim())
        pdf_ax.yaxis.set_visible(False)
        for k in ['left', 'right']:
            pdf_ax.spines[k].set_visible(False)

        handles = []
        pdf_ax.fill_betweenx(y_grid, 0, x_emp, alpha=0.16, color='C0')
        h_emp,  = pdf_ax.plot(x_emp,  y_grid, lw=2.0, color='C0', label="Empirical KDE"); handles.append(h_emp)
        h_n,    = pdf_ax.plot(x_norm, y_grid, lw=1.8, color='C1', ls='--', label="Normal fit"); handles.append(h_n)
        h_ln,   = pdf_ax.plot(x_ln,   y_grid, lw=1.8, color='C2', ls='-.', label="Lognormal fit"); handles.append(h_ln)
        h_ex,   = pdf_ax.plot(x_ex,   y_grid, lw=1.8, color='C3', ls=':',  label="Exponential fit"); handles.append(h_ex)

        pdf_ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.55, 0.60),
                      frameon=False, fontsize=10.5)

        skew_v, kurt_v, ks_stat = _stats_pack(v, y_grid, mu, sigma)
        stats_txt = (f"μ = {mu:.3f}\nσ = {sigma:.3f}\n"
                     f"skew = {skew_v:.2f}\nkurt = {kurt_v:.2f}\nKS = {ks_stat:.3f}")
        pdf_ax.text(0.95, 0.95, stats_txt, ha='right', va='top',
                    transform=pdf_ax.transAxes, fontsize=10.5)

        pdf_ax.set_xlabel("Density (scaled)", fontsize=10.5)
        pdf_ax.set_title("PDF vs Fitted Distributions", pad=4, fontsize=10.5)
        pdf_ax.set_xlim(-0.05, 1.0)
        pdf_ax.grid(False)

        # 主图去轴；标题 21pt
        ax.set_axis_off()
        fig.suptitle('Supply Chain Vulnerability Index', y=0.965, fontsize=21)

        out_path = "supply_chain_vulnerability_with_multi_pdf.png"
        fig.savefig(out_path, dpi=300)
        logger.info(f"Saved to: {out_path}")
        return out_path

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return None

# ---------- 示例 ----------
def test_visualization():
    grid_data = pd.DataFrame({
        'grid_id': np.arange(1, 2001),
        'vulnerability': np.clip(np.random.beta(1.5, 6.0, 2000), 0, 1)
    })
    visualize_results(
        grid_data,
        show_basemap=False,
        q_low=0.02, q_high=0.98,
        gamma=0.65,       # ↓ 0.55~0.75 都不错；越小低值越显色
        min_tint=0.06,    # 最浅色的“着色下限”
        colorbar_size="2.6%", colorbar_pad=0.06,
        pdf_width="18%",  gap_between=0.012
    )

if __name__ == "__main__":
    try:
        if hasattr(ctx.tile._fetch_tile, 'cache_clear'):
            ctx.tile._fetch_tile.cache_clear()
    except Exception:
        pass
    test_visualization()
