# -*- coding: utf-8 -*-
"""
双窗口交互版（纵向控件）
- 色标轴保留刻度/标题
- 右侧提琴图：删除 y 轴数字标识，但保留刻度线
- 金色系配色
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 如遇 Tk 关闭报错，可改用 Qt 后端（按需开启）：
# mpl.use("Qt5Agg")

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
# ======= 路径设置 =======
CSV_PATH = r"/地理脆弱性\综合风险评估\Zhengzhou_Grid_GeoVuln_Mean.csv"
OUT_PNG  = r"D:\Cascading effect predicate\地理脆弱性势场\综合风险评估\risk_layout_defaults_gold_ticked.png"

# ======= 配色 =======
CMAP_ROSE = "YlOrBr"   # 金色系
CMAP_CBAR = "YlOrBr"

# ---------- 工具 ----------
def sector_edges_labels(n_dir: int):
    edges  = np.linspace(0, 2*np.pi, n_dir+1)
    labels = [f"{int(np.degrees(edges[i]))}–{int(np.degrees(edges[i+1]))}°" for i in range(n_dir)]
    mids   = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    return edges, labels, mids, widths

def quantile_rings(rad, n_rad):
    q = np.linspace(0, 1, n_rad + 1)
    edges = np.unique(np.quantile(rad, q))
    if len(edges) < n_rad + 1:
        edges = np.linspace(np.nanmin(rad), np.nanmax(rad), n_rad + 1)
    return edges

def pastel(n, s=0.55, v=0.88, alpha=0.28):
    cols = []
    for i in range(n):
        h = (i / n) % 1.0
        rgb = mpl.colors.hsv_to_rgb((h, s, v))
        cols.append((*rgb, alpha))
    return cols

# ---------- 绘图块 ----------
def draw_risk_rose(ax_polar, x, y, val, n_dir=8, n_rad=6, cmap=CMAP_ROSE):
    cx, cy = float(np.nanmean(x)), float(np.nanmean(y))
    dx, dy = x - cx, y - cy
    theta  = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
    rad    = np.hypot(dx, dy)

    t_edges, _, t_mid, t_w = sector_edges_labels(n_dir)
    r_edges = quantile_rings(rad, n_rad)

    H = np.full((n_dir, n_rad), np.nan, float)
    for i in range(n_dir):
        t0, t1 = t_edges[i], t_edges[i+1]
        mdir = (theta >= t0) & (theta < t1) if i < n_dir-1 else (theta >= t0) & (theta <= t1)
        if not np.any(mdir):
            continue
        tr = rad[mdir]; tv = val[mdir]
        idx = np.digitize(tr, r_edges, right=False) - 1
        idx = np.clip(idx, 0, n_rad-1)
        for j in range(n_rad):
            sel = idx == j
            if np.any(sel):
                H[i, j] = np.nanmean(tv[sel])

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cm   = mpl.cm.get_cmap(cmap)
    for i in range(n_dir):
        for j in range(n_rad):
            c = cm(norm(H[i, j])) if np.isfinite(H[i, j]) else (1,1,1,0)
            ax_polar.bar(x=t_mid[i],
                         height=r_edges[j+1]-r_edges[j],
                         width=t_w[i],
                         bottom=r_edges[j],
                         color=c, edgecolor='white', linewidth=0.8, align='center')

    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(-1)
    ax_polar.grid(alpha=0.25)
    ax_polar.set_yticklabels([])
    ax_polar.set_title("Zhengzhou — Directional Vulnerability Rose", pad=6)

def draw_xy(ax, x, y, title="Study Area (Zhengzhou) & Analysis Center"):
    cx, cy = float(np.nanmean(x)), float(np.nanmean(y))
    ax.scatter(x, y, s=2, alpha=0.10, color="#1f77b4")
    ax.scatter([cx], [cy], s=40, marker="*", color="#d62728", zorder=3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(title, pad=6)
    ax.grid(alpha=0.12, linewidth=0.6)

def draw_violin(ax, theta, val, t_edges, labels=None,
                jitter_width=0.72, sample_frac=0.35, cat_gap=1.25):
    """
    提琴 + 盒须 + 抖动
    ——需求：删除 y 轴“数字标识”，但保留刻度线（ticks）
    """
    n_dir = len(t_edges) - 1
    bins = np.digitize(theta, t_edges, right=False) - 1
    bins = np.clip(bins, 0, n_dir-1)
    groups = [val[bins == i] for i in range(n_dir)]
    if labels is None:
        labels = [f"{int(np.degrees(t_edges[i]))}–{int(np.degrees(t_edges[i+1]))}°" for i in range(n_dir)]
    cols = pastel(n_dir)

    pos = 1 + np.arange(n_dir) * float(cat_gap)
    width_v = min(0.9, 0.8 * cat_gap)
    parts = ax.violinplot(groups, positions=pos, widths=width_v,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, b in enumerate(parts['bodies']):
        b.set_facecolor(cols[i]); b.set_edgecolor('none'); b.set_alpha(cols[i][3])

    for i, g in enumerate(groups):
        if g.size == 0: continue
        q1, med, q3 = np.percentile(g, [25, 50, 75])
        iqr = q3 - q1
        whisk_low  = np.min(g[g >= q1 - 1.5*iqr]) if np.any(g >= q1 - 1.5*iqr) else np.min(g)
        whisk_high = np.max(g[g <= q3 + 1.5*iqr]) if np.any(g <= q3 + 1.5*iqr) else np.max(g)
        x0 = pos[i]
        box_w = min(0.38, 0.32 * cat_gap)
        edge  = mpl.colors.to_hex(cols[i][:3])
        ax.add_patch(plt.Rectangle((x0 - box_w/2, q1), box_w, q3 - q1,
                                   facecolor='white', edgecolor=edge, lw=2, alpha=0.98, zorder=3))
        ax.plot([x0 - box_w/2, x0 + box_w/2], [med, med], color=edge, lw=2, zorder=4)
        ax.plot([x0, x0], [whisk_low, q1], color=edge, lw=1.4, zorder=3)
        ax.plot([x0, x0], [q3, whisk_high], color=edge, lw=1.4, zorder=3)
        ax.plot([x0 - box_w/4, x0 + box_w/4], [whisk_low, whisk_low], color=edge, lw=1.4, zorder=3)
        ax.plot([x0 - box_w/4, x0 + box_w/4], [whisk_high, whisk_high], color=edge, lw=1.4, zorder=3)
        ax.text(x0, 0.02, f"n={len(g)}", ha='center', va='bottom', fontsize=9, color='#555')

    rng = np.random.default_rng(2025)
    for i, g in enumerate(groups):
        if g.size == 0: continue
        k = max(1, int(len(g) * sample_frac))
        idx = rng.choice(len(g), size=k, replace=False)
        gj  = g[idx]
        jitter_w = min(jitter_width, 0.45 * cat_gap)
        jitter = (rng.random(k) - 0.5) * jitter_w
        ax.scatter(pos[i] + jitter, gj, s=10, alpha=0.35, color='k', zorder=2, linewidths=0)

    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=25, ha='right')

    # ——关键改动：保留刻度线，但隐藏数字标识（标签）——
    ax.set_yticks(np.linspace(0, 1, 6))              # 保留 tick 位置
    ax.tick_params(axis='y', which='both',
                   labelleft=False, length=4, width=0.8)  # 隐藏标签，仅显示刻度线
    ax.set_ylabel("")                                 # 不显示 y 轴标题
    ax.grid(False)                                     # 不画网格线

    ax.set_title("Vulnerability by Direction (Violin • Box • Jitter)", pad=6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ---------- 版式/缩放 ----------
def _centered_scaled_rect(box, scale):
    x, y, w, h = box
    s = float(scale)
    nw = min(w*s, w); nh = min(h*s, h)
    cx, cy = x + w/2, y + h/2
    nx, ny = cx - nw/2, cy - nh/2
    nx = max(x, nx); ny = max(y, ny)
    if nx + nw > x + w: nx = x + w - nw
    if ny + nh > y + h: ny = y + h - nh
    return [nx, ny, nw, nh]

def compute_rects(p):
    L = p['left']; R = 1 - p['right']; T = 1 - p['top']; B = p['bottom']
    W = R - L;     H = T - B

    leftW  = W * p['left_col_w']
    gapW   = W * p['wspace']
    rightW = W - leftW - gapW
    xL = L
    xGapL = xL + leftW
    xR = xGapL + gapW

    roseH = H * p['top_frac']
    xyH   = H - roseH - p['vspace'] * H
    yRose = B + (H - roseH)
    yXY   = B

    rose_box = [xL, yRose, leftW, roseH]
    xy_box   = [xL, yXY,   leftW, xyH]
    vio_box  = [xR, B,     rightW, H]

    rect_rose = _centered_scaled_rect(rose_box, p.get('rose_scale', 1.0))
    rect_xy   = _centered_scaled_rect(xy_box,   p.get('xy_scale',   1.0))
    rect_vio  = _centered_scaled_rect(vio_box,  p.get('vio_scale',  1.0))

    # 色标：宽度限制在中缝，允许水平偏移
    cbar_w_abs = min(p['cbar_w'] * W, gapW * 0.9)
    base_x = xGapL + (gapW - cbar_w_abs) / 2.0
    offset = float(p.get('cbar_offset', 0.0)) * gapW  # 相对中缝宽度
    cbar_x = base_x + offset
    cbar_x = max(xGapL, min(cbar_x, xGapL + gapW - cbar_w_abs))
    rect_cbar = [cbar_x, B, cbar_w_abs, H]

    return rect_rose, rect_xy, rect_vio, rect_cbar

# ---------- 主流程（双窗口） ----------
def main():
    df = pd.read_csv(CSV_PATH)[["geo_mean_01","centroid_x","centroid_y"]].dropna()
    v = df["geo_mean_01"].to_numpy()
    x = df["centroid_x"].to_numpy()
    y = df["centroid_y"].to_numpy()

    cx, cy = float(np.mean(x)), float(np.mean(y))
    theta = (np.arctan2(y - cy, x - cx) + 2*np.pi) % (2*np.pi)
    edges, labels, _, _ = sector_edges_labels(8)

    # ——默认值：记录自你上次面板参数——
    P = dict(
        left=0.0604, right=0.0540, top=0.1424, bottom=0.1024,
        left_col_w=0.3000, wspace=0.0420,
        top_frac=0.7063, vspace=0.0981,
        cbar_w=0.0196, cbar_offset=-0.0042,
        rose_scale=0.7755, xy_scale=1.1294, vio_scale=1.0014,
        cat_gap=2.0000
    )

    # ——主窗口——
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Descriptive Statistics of Geographic Vulnerability in Zhengzhou",
                 y=0.98, fontsize=21)

    r_rose, r_xy, r_vio, r_cbar = compute_rects(P)
    ax_rose = fig.add_axes(r_rose, projection='polar')
    ax_xy   = fig.add_axes(r_xy)
    ax_vio  = fig.add_axes(r_vio)
    ax_cbar = fig.add_axes(r_cbar, sharey=ax_vio)

    draw_risk_rose(ax_rose, x, y, v, 8, 6, cmap=CMAP_ROSE)
    draw_xy(ax_xy, x, y)
    draw_violin(ax_vio, theta, v, edges, labels, cat_gap=P['cat_gap'])

    # 中间色标（与右图共享 y 轴 0–1）——保留刻度与标题
    grad = np.linspace(0, 1, 256)[:, None]
    ax_cbar.imshow(grad, aspect='auto', cmap=CMAP_CBAR, origin='lower', extent=[0, 1, 0, 1])
    ax_cbar.set_xticks([])
    ax_cbar.set_ylim(ax_vio.get_ylim())
    ax_cbar.yaxis.set_ticks_position('left')
    ax_cbar.yaxis.set_label_position('left')
    ax_cbar.set_ylabel("Vulnerability (0–1)")      # 标题保留
    ax_cbar.set_yticks(np.linspace(0, 1, 6))       # 刻度保留（0,0.2,...,1）

    # ——控制窗口（纵向单列）——
    figc = plt.figure(figsize=(5.4, 11))
    figc.suptitle("Layout Controls (vertical)", y=0.985, fontsize=12)

    sliders = {}
    bax_save  = figc.add_axes([0.08, 0.94, 0.38, 0.04])
    bax_reset = figc.add_axes([0.54, 0.94, 0.38, 0.04])
    btn_save  = Button(bax_save, "Save PNG", color="#3A7", hovercolor="#49A")
    btn_reset = Button(bax_reset, "Reset",    color="#777", hovercolor="#999")

    left = 0.10; width = 0.80; h = 0.033; gap = 0.012
    y0 = 0.92 - (h + gap)
    items = [
        ('left','Left',P['left'],0.00,0.15),
        ('right','Right',P['right'],0.00,0.15),
        ('top','Top',P['top'],0.00,0.20),
        ('bottom','Bottom',P['bottom'],0.00,0.20),
        ('left_col_w','Left Col W',P['left_col_w'],0.30,0.70),
        ('wspace','Wspace',P['wspace'],0.00,0.10),
        ('top_frac','TopFrac',P['top_frac'],0.40,0.90),
        ('vspace','Vspace',P['vspace'],0.00,0.20),
        ('cbar_w','Cbar W',P['cbar_w'],0.010,0.050),
        ('cbar_offset','Cbar Offset',P['cbar_offset'],-0.45,0.45),
        ('rose_scale','Rose Scale',P['rose_scale'],0.70,1.20),
        ('xy_scale','XY Scale',P['xy_scale'],0.70,1.30),
        ('vio_scale','Violin Scale',P['vio_scale'],0.70,1.20),
        ('cat_gap','Violin Spacing',P['cat_gap'],1.00,2.00),
    ]
    y = y0
    for name, label, val, vmin, vmax in items:
        ax_s = figc.add_axes([left, y, width, h])
        sliders[name] = Slider(ax_s, label, vmin, vmax, valinit=val)
        y -= (h + gap)

    def redraw_violin():
        ax_vio.clear()
        draw_violin(ax_vio, theta, v, edges, labels, cat_gap=sliders['cat_gap'].val)
        ax_cbar.set_ylim(ax_vio.get_ylim())
        ax_cbar.set_yticks(np.linspace(0, 1, 6))

    def update(_=None):
        for k in P.keys():
            if k in sliders:
                P[k] = sliders[k].val
        r1, r2, r3, r4 = compute_rects(P)
        ax_rose.set_position(r1); ax_xy.set_position(r2)
        ax_vio.set_position(r3);  ax_cbar.set_position(r4)
        redraw_violin()
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    def do_save(event):
        fig.savefig(OUT_PNG, dpi=360, bbox_inches=None)
        print("✅ Saved:", OUT_PNG)

    def do_reset(event):
        defaults = dict(
            left=0.0604, right=0.0540, top=0.1424, bottom=0.1024,
            left_col_w=0.3000, wspace=0.0420,
            top_frac=0.7063, vspace=0.0981,
            cbar_w=0.0196, cbar_offset=-0.0042,
            rose_scale=0.7755, xy_scale=1.1294, vio_scale=1.0014,
            cat_gap=2.0000
        )
        for k,v0 in defaults.items():
            if k in sliders:
                sliders[k].set_val(v0)
        update()

    btn_save.on_clicked(do_save)
    btn_reset.on_clicked(do_reset)

    plt.show()

if __name__ == "__main__":
    main()
