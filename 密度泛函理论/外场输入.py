# -*- coding: utf-8 -*-
"""
Direct mapping with calibrated simulated rainfall, plain-text labels,
16:9 figure, Times New Roman, 10% margins, numeric day ticks on top subplot.
Exports:
  - lambda_5min_timeseries.csv  (time, I_theta_mmph, lambda_per_hr, expected_5min)
  - hourly_mu_calibrated.csv    (time, mu_hour)  ← 给 NHPP/密度脚本使用
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------- Font ----------
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif", "Times"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ====================== USER SETTINGS ======================
START  = "2021-07-17 08:00"
END    = "2021-07-23 08:00"
DT_MIN = 5

PEAK_TIME     = "2021-07-20 16:30"
PEAK_MM_PER_H = 201.9
S_WIN_MM      = 534.0

CALIBRATE_I_TOTAL = True

PULSES_BASE = [
    ("2021-07-19 14:30", 0.22, 1.0, 2.0),
    ("2021-07-20 16:30", 1.00, 0.8, 2.0),  # main peak
    ("2021-07-20 22:30", 0.30, 0.8, 2.2),
    ("2021-07-21 06:00", 0.20, 1.0, 2.8),
]

# Poisson 生成许多小脉冲
EXTRA_MODE = "poisson"
RATE_PER_HR_BASE = 2.0
RATE_PEAK_BOOST  = 3.0
EXCLUDE_AROUND_PEAK_HR = 1.5
MIN_GAP_HR       = 0.20
EXTRA_AMP_REL_MIN, EXTRA_AMP_REL_MAX = 0.03, 0.10
EXTRA_SIG_MIN_HR,  EXTRA_SIG_MAX_HR  = 0.35, 1.0
RANDOM_SEED = 123

# 低幅背景雨（可变）
FLOOR_BASE_MM_PER_H  = 0.45
FLOOR_PEAK_BOOST     = 4.0
FLOOR_NOISE_AMP      = 1.2
FLOOR_NOISE_SIG_HR   = 0.6
FLOOR_SHAPE_P        = 1.3

# 脉冲宽度缩放
FIX_WIDTH_SCALE = 0.300

# 直接映射 g(I)；alpha 用 ∑λΔt = TARGET_PARTICLES 定标
MAP_MODE   = 'hill'   # 'linear' | 'power' | 'hill'
GAMMA_EXP  = 1.15
ES_MM_PER_H = 35.0

TARGET_PARTICLES = 6e5

# 图形
PLOT_MODE = 'symlog'  # 'symlog' | 'broken' | 'cap'
TITLE_I = "Simulated rainfall intensity time series Zhengzhou 17 to 23 July 2021"
TITLE_L = "Arrival rate time series derived from simulated rainfall intensity Zhengzhou 17 to 23 July 2021"
SAVE_PNG = True
PNG_PATH = f"Itheta_lambda_{PLOT_MODE}_16x9_numeric_ticks_TOP.png"
DPI = 300
FIGSIZE = (16, 9)  # width, height → 16:9
MARGIN  = 0.10     # 10% four-side margins
HSPACE  = 0.22     # spacing between subplots
# 导出 CSV 路径（按你的盘符）
CSV_5MIN  = r"D:\Cascading effect predicate\密度泛函理论\lambda_5min_timeseries.csv"
CSV_HOURLY = r"D:\Cascading effect predicate\密度泛函理论\hourly_mu_calibrated.csv"
# ===========================================================

# ---------------------- helpers ----------------------
def parse_time(s): return datetime.strptime(s, "%Y-%m-%d %H:%M")

def build_time_axis(start_str, end_str, dt_min=5):
    t0 = parse_time(start_str); t1 = parse_time(end_str)
    times = []
    t = t0; delta = timedelta(minutes=dt_min)
    while t <= t1 + timedelta(seconds=1):
        times.append(t); t += delta
    times = np.array(times)
    th = np.array([(tt - t0).total_seconds()/3600.0 for tt in times])
    T_hr = (t1 - t0).total_seconds()/3600.0
    dt_hr = dt_min / 60.0
    return t0, times, th, T_hr, dt_hr

def to_rel_hour(dt, t0): return (dt - t0).total_seconds()/3600.0

def skew_gauss_pulse(th, tp, amp, sigL, sigR):
    sig = np.where(th <= tp, sigL, sigR)
    return amp * np.exp(-(th - tp)**2 / (2.0 * sig**2))

def gaussian_kernel(n_sigma=3, sigma_steps=5):
    half = int(n_sigma * sigma_steps)
    x = np.arange(-half, half+1)
    ker = np.exp(-0.5*(x/sigma_steps)**2); ker /= ker.sum()
    return ker

def smooth_noise(n, sigma_steps=5, seed=None):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    ker = gaussian_kernel(sigma_steps=sigma_steps)
    return np.convolve(w, ker, mode="same")

def raised_cosine_env(th, T_hr, p=1.3):
    x = np.clip(th / T_hr, 0.0, 1.0)
    return (0.5*(1 - np.cos(2*np.pi*x)))**p

def make_variable_floor(th, T_hr, base, peak_boost, noise_amp, noise_sig_hr, dt_hr, p, seed):
    env = raised_cosine_env(th, T_hr, p=p)
    n = len(th)
    sigma_steps = max(1, int(round(noise_sig_hr / dt_hr)))
    z = smooth_noise(n, sigma_steps=sigma_steps, seed=seed)
    z = (z - z.mean()) / (z.std() + 1e-9)
    floor = base + peak_boost*env + noise_amp*z
    return np.clip(floor, 0.0, None)

def time_envelope(th, t0):
    c1 = to_rel_hour(parse_time("2021-07-19 12:00"), t0)
    c2 = to_rel_hour(parse_time("2021-07-20 18:00"), t0)
    c3 = to_rel_hour(parse_time("2021-07-21 12:00"), t0)
    s1, s2, s3 = 14.0, 8.0, 12.0
    env = (np.exp(-0.5*((th-c1)/s1)**2) +
           np.exp(-0.5*((th-c2)/s2)**2) +
           np.exp(-0.5*((th-c3)/s3)**2))
    return env / env.max()

def generate_extra_pulses_poisson(th, t0, dt_hr,
                                  rate_base_per_hr, peak_boost,
                                  min_gap_hr, exclude_around_peak_hr,
                                  amp_rel_rng, sig_rng_hr, seed):
    rng = np.random.default_rng(seed)
    tp_main = to_rel_hour(parse_time(PEAK_TIME), t0)
    env = time_envelope(th, t0)
    rate = rate_base_per_hr * (1.0 + peak_boost*env)
    p_per_step = np.maximum(rate * dt_hr, 0.0)

    times_hr = []
    for i, t_hr in enumerate(th):
        if abs(t_hr - tp_main) < exclude_around_peak_hr:
            continue
        k = rng.poisson(p_per_step[i])
        for _ in range(k):
            cand = t_hr + rng.uniform(0, dt_hr)
            if any(abs(cand - h) < min_gap_hr for h in times_hr):
                continue
            times_hr.append(cand)

    pulses = []
    for cand in times_hr:
        amp_rel = rng.uniform(*amp_rel_rng)
        sigL = rng.uniform(*sig_rng_hr)
        sigR = rng.uniform(*sig_rng_hr)
        ts = (t0 + timedelta(hours=float(cand))).strftime("%Y-%m-%d %H:%M")
        pulses.append((ts, amp_rel, sigL, sigR))
    return pulses

# ---------------------- I_theta construction ----------------------
def build_Itheta(th, t0, peak_mmph, pulses_all, width_scale, floor):
    tp_main = to_rel_hour(parse_time(PEAK_TIME), t0)
    idx_peak = np.argmin(np.abs(th - tp_main))
    F = np.clip(floor, 0.0, None)
    main_base0 = max(peak_mmph - F[idx_peak], 0.0)

    I = F.copy()
    for ts, amp_rel, sigL, sigR in pulses_all:
        tp = to_rel_hour(parse_time(ts), t0)
        A = main_base0 * (1.0 if ts == PEAK_TIME else amp_rel)
        I += skew_gauss_pulse(th, tp, A, width_scale*sigL, width_scale*sigR)
    return np.clip(I, 0.0, None)

# ---------------------- analytic calibration ----------------------
def recalibrate_Itheta_preserve_peak(th, t0, floor, pulses_all, width_scale,
                                     peak_mmph, S_target, dt_hr, peak_time_str):
    # 找到主峰的形状
    main_sigL = main_sigR = None
    for ts, amp_rel, sigL, sigR in PULSES_BASE:
        if ts == peak_time_str:
            main_sigL, main_sigR = sigL, sigR
            break
    if main_sigL is None:
        raise ValueError("Main pulse not found in PULSES_BASE.")

    tp_main = (parse_time(peak_time_str) - t0).total_seconds()/3600.0
    idx_peak = int(np.argmin(np.abs(th - tp_main)))

    F = np.clip(floor, 0.0, None)
    main_base0 = max(peak_mmph - F[idx_peak], 0.0)

    # 非主峰叠加（相对主峰幅度）
    R = np.zeros_like(th)
    for ts, amp_rel, sigL, sigR in pulses_all:
        if ts == peak_time_str:
            continue
        tp = (parse_time(ts) - t0).total_seconds()/3600.0
        R += skew_gauss_pulse(th, tp, amp_rel*main_base0,
                              width_scale*sigL, width_scale*sigR)

    # 主峰单位形状
    M1 = skew_gauss_pulse(th, tp_main, 1.0,
                          width_scale*main_sigL, width_scale*main_sigR)

    B_peak = F[idx_peak] + R[idx_peak]
    S_B   = (F + R).sum() * dt_hr
    S_M   = M1.sum() * dt_hr

    denom = (S_B - B_peak * S_M)
    if denom <= 1e-12:
        k = 1.0
        A = max(PEAK_MM_PER_H - B_peak, 0.0)
    else:
        k = (S_target - PEAK_MM_PER_H * S_M) / denom
        A = max(PEAK_MM_PER_H - k * B_peak, 0.0)

    I_new = np.clip(k*(F + R) + A*M1, 0.0, None)
    return I_new, k, A

# ---------------------- direct mapping ----------------------
def map_I_to_lambda_direct(I, dt_hr, map_mode='hill', gamma=1.15, Es=35.0, target_particles=6e5):
    if map_mode == 'linear':
        E = I.copy()
    elif map_mode == 'power':
        E = np.power(np.clip(I, 0, None), gamma)
    else:  # 'hill'
        num = np.power(np.clip(I, 0, None), gamma)
        den = num + np.power(Es, gamma)
        E = np.where(den > 0, num/den, 0.0)
    alpha = 0.0 if E.sum() == 0 else (target_particles / (E.sum() * dt_hr))
    return alpha * E, alpha

def _auto_linthresh(y, lo=5.0, hi=50.0, pct=95, scale=0.25):
    y = np.asarray(y)
    ypos = y[y > 0]
    if ypos.size == 0: return lo
    t = np.percentile(ypos, pct) * scale
    return float(np.clip(t, lo, hi))

def set_numeric_day_ticks(ax):
    """Show day numbers 17..23 on a datetime axis."""
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.tick_params(axis='x', labelsize=12)

# =========================== MAIN ===========================
t0, times, th, T_hr, dt_hr = build_time_axis(START, END, DT_MIN)

# 背景
floor = make_variable_floor(
    th, T_hr,
    base=FLOOR_BASE_MM_PER_H,
    peak_boost=FLOOR_PEAK_BOOST,
    noise_amp=FLOOR_NOISE_AMP,
    noise_sig_hr=FLOOR_NOISE_SIG_HR,
    dt_hr=dt_hr,
    p=FLOOR_SHAPE_P,
    seed=RANDOM_SEED
)

# 脉冲
pulses_all = list(PULSES_BASE)
if EXTRA_MODE == "poisson":
    extra = generate_extra_pulses_poisson(
        th, t0, dt_hr,
        rate_base_per_hr=RATE_PER_HR_BASE,
        peak_boost=RATE_PEAK_BOOST,
        min_gap_hr=MIN_GAP_HR,
        exclude_around_peak_hr=EXCLUDE_AROUND_PEAK_HR,
        amp_rel_rng=(EXTRA_AMP_REL_MIN, EXTRA_AMP_REL_MAX),
        sig_rng_hr=(EXTRA_SIG_MIN_HR, EXTRA_SIG_MAX_HR),
        seed=RANDOM_SEED
    )
    pulses_all.extend(extra)

# 构造 & 校准 I_theta
w_scale = float(FIX_WIDTH_SCALE)
I_theta = build_Itheta(th, t0, PEAK_MM_PER_H, pulses_all, w_scale, floor)
if CALIBRATE_I_TOTAL:
    I_theta, k_nonmain, A_main = recalibrate_Itheta_preserve_peak(
        th, t0, floor, pulses_all, w_scale,
        PEAK_MM_PER_H, S_WIN_MM, dt_hr, PEAK_TIME
    )

# 直接映射→lambda
lam, alpha = map_I_to_lambda_direct(
    I_theta, dt_hr,
    map_mode=MAP_MODE, gamma=GAMMA_EXP, Es=ES_MM_PER_H,
    target_particles=TARGET_PARTICLES
)

# ---- 诊断 ----
i_peak = np.argmax(I_theta)
print("—— Stats ——")
print(f"Window: {START} ~ {END} (dt={DT_MIN} min)")
print(f"I_theta peak ≈ {I_theta[i_peak]:.1f} mm/h @ {times[i_peak]}")
print(f"I_theta total ≈ {I_theta.sum()*dt_hr:.1f} mm  (target {S_WIN_MM})")
print(f"sum(lambda*dt) ≈ {(lam.sum()*dt_hr):.0f}  (target {TARGET_PARTICLES:.0f})")
print(f"alpha = {alpha:.3g}")

# ===== 导出（5min 与 hourly） =====
df5 = pd.DataFrame({
    "time": pd.to_datetime(times),                 # 5-min 时间戳
    "I_theta_mmph": I_theta.astype(float),
    "lambda_per_hr": lam.astype(float)
})
df5["expected_5min"] = df5["lambda_per_hr"] * dt_hr   # 每 5 分钟的期望到达

df5.to_csv(CSV_5MIN, index=False, encoding="utf-8-sig")
hourly = (
    df5.assign(hour=df5["time"].dt.floor("H"))
       .groupby("hour", as_index=False)["expected_5min"].sum()
       .rename(columns={"hour": "time", "expected_5min": "mu_hour"})
)
print(f"hourly sum mu_hour ≈ {hourly['mu_hour'].sum():.0f}  (target {TARGET_PARTICLES:.0f})")
hourly.to_csv(CSV_HOURLY, index=False, encoding="utf-8-sig")
print(f"saved 5-min CSV -> {CSV_5MIN}")
print(f"saved hourly CSV -> {CSV_HOURLY}")

# --------------------------- plotting ---------------------------
if PLOT_MODE == 'symlog':
    ith_linthresh = _auto_linthresh(I_theta, lo=5, hi=30, pct=95, scale=0.30)
    lam_linthresh = _auto_linthresh(lam,      lo=500, hi=5000, pct=95, scale=0.25)

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=MARGIN, right=1-MARGIN, bottom=MARGIN, top=1-MARGIN, hspace=HSPACE)

    ax0 = axes[0]
    ax0.plot(times, I_theta, lw=1.6)
    ax0.set_yscale('symlog', linthresh=ith_linthresh, linscale=1.0)
    ax0.set_ylabel("Rainfall intensity over time", fontsize=14)
    ax0.set_title(TITLE_I, fontsize=21, pad=16)
    ax0.axvline(parse_time(PEAK_TIME), ls=":", lw=1.0)
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    set_numeric_day_ticks(ax0)
    ax0.tick_params(axis='x', labelbottom=True)
    for lbl in ax0.get_xticklabels():
        lbl.set_visible(True)

    ax1 = axes[1]
    ax1.plot(times, lam, lw=1.8)
    ax1.set_yscale('symlog', linthresh=lam_linthresh, linscale=1.0)
    ax1.set_ylabel("Arrival rate over time", fontsize=14)
    ax1.set_title(TITLE_L, fontsize=21, pad=12)
    ax1.grid(True, ls=":", lw=0.6, alpha=0.6)
    set_numeric_day_ticks(ax1)
    ax1.set_xlabel("Time BJT")

else:
    raise ValueError("Only 'symlog' mode implemented here for brevity.")

if SAVE_PNG:
    plt.savefig(PNG_PATH, dpi=DPI)
    print(f"\nSaved figure: {PNG_PATH} (dpi={DPI})")

plt.show()
