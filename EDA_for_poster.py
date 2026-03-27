import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# ── Paste your file paths here ─────────────────────────────────────────────
DF1_PATH = "cleaned_air_quality_merged.csv"   # Air quality time-series
DF2_PATH = "city_pollutant_health_merged_v2.csv"   # Multi-city health dataset
# ───────────────────────────────────────────────────────────────────────────
TOL_BRIGHT = [
    "#4477AA",  # blue
    "#EE6677",  # rose  (replaces red — safe with blue)
    "#228833",  # green (safe when not paired with red)
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
]
 
# Extended to 10 by adding Tol 'muted' extras
TOL_10 = TOL_BRIGHT + [
    "#BBBBBB",  # light gray
    "#000000",  # black
    "#994F00",  # brown
    "#006CD1",  # sky blue
]
 
# ── Tol 'sunset' diverging colormap (blue → white → red) ───────────────────
# Perceptually uniform, safe for all CVD types. Replaces blue/red diverging.
_SUNSET_COLORS = [
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1",
    "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366",
    "#F67E4B", "#DD3D2D", "#A50026",
]
CMAP_DIVERGING = LinearSegmentedColormap.from_list("tol_sunset", _SUNSET_COLORS)
 
# ── Tol sequential (white → orange → brown) for missing-data heatmap ───────
_SEQ_COLORS = ["#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C",
               "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"]
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list("tol_seq", _SEQ_COLORS)
 
THEME = {
    "bg":      "#FAFAF8",
    "panel":   "#FFFFFF",
    "border":  "#D3D1C7",
    "text":    "#2C2C2A",
    "muted":   "#73726C",
    # Semantic aliases — always from TOL_BRIGHT so they stay CB-safe
    "blue":    TOL_BRIGHT[0],   # #4477AA
    "rose":    TOL_BRIGHT[1],   # #EE6677  (replaces red)
    "green":   TOL_BRIGHT[2],   # #228833
    "yellow":  TOL_BRIGHT[3],   # #CCBB44
    "cyan":    TOL_BRIGHT[4],   # #66CCEE
    "purple":  TOL_BRIGHT[5],   # #AA3377
    "gray":    TOL_10[6],       # #BBBBBB
    "brown":   TOL_10[8],       # #994F00
    "sky":     TOL_10[9],       # #006CD1
}
 
# Pollutant → color mapping using only CB-safe colors
POLLUTANT_COLORS = {
    "NO2":        THEME["blue"],
    "NO":         THEME["cyan"],
    "SO2":        THEME["yellow"],
    "CO":         THEME["brown"],
    "PM25":       THEME["rose"],
    "Ox":         THEME["purple"],
    "H2S":        THEME["gray"],
    "PM10":       THEME["green"],
    "NOx":        THEME["sky"],
    "UFP":        TOL_10[7],    # black
    "O3":         THEME["green"],
    "BC":         THEME["gray"],
    "NH3":        THEME["yellow"],
    "NO2_palmes": THEME["brown"],
}
 
DISEASE_COLS = {
    "TotaalZiektenVanHartEnVaatstelsel_43":    "Cardiovascular",
    "TotaalZiektenVanDeAdemhalingsorganen_50": "Respiratory",
    "TotaalNieuwvormingen_8":                  "Cancer",
    "TotaalPsychischeStoornissen_35":          "Psychiatric",
    "TotaalZiektenSpierenBeendBindwfsl_64":    "Musculoskeletal",
    "TotaalEndocrieneVoedingsStofwZ_32":       "Endocrine",
    "k_81Griep_51":                            "Influenza",
    "TotaalChronischeAandOndersteLucht_53":    "Chronic lower resp.",
    "k_82Longontsteking_52":                   "Pneumonia",
    "k_711AcuutHartinfarct_45":                "Acute MI",
}
 
DF1_POLLUTANTS = ["NO2", "SO2", "CO", "PM25", "Ox", "NO", "H2S"]
 
 
def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(THEME["panel"])
    ax.tick_params(colors=THEME["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["border"])
        spine.set_linewidth(0.5)
    ax.grid(True, color=THEME["border"], linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", color=THEME["text"], pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=THEME["muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=THEME["muted"])
 
 
# ════════════════════════════════════════════════════════════════════════════
# DATASET 1 — AIR QUALITY TIME-SERIES
# ════════════════════════════════════════════════════════════════════════════
 
def load_df1(path):
    # Auto-detect separator (tab or comma)
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(path, sep=sep, engine="python")
 
    # Normalise column names: strip whitespace and BOM characters
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
 
    # Find the datetime column regardless of exact capitalisation
    dt_col = next((c for c in df.columns if c.lower() == "datetime"), None)
    if dt_col is None:
        raise ValueError(
            f"No 'datetime' column found in DF1. Columns present: {list(df.columns)}"
        )
 
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.rename(columns={dt_col: "datetime"})
    df = df.set_index("datetime").sort_index()
 
    for col in DF1_POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
 
 
def plot_df1(df):
    fig = plt.figure(figsize=(16, 12), facecolor=THEME["bg"])
    fig.suptitle("Dataset 1 — Air Quality Time-Series (Eindhoven)",
                 fontsize=13, fontweight="bold", color=THEME["text"], y=0.98)
 
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)
 
    cols = [c for c in DF1_POLLUTANTS if c in df.columns]
 
    # ── 1. Full time-series (top row, full width) ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    linestyles = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]
    for i, col in enumerate([c for c in cols if c != "CO"]):
        ax1.plot(df.index, df[col], linewidth=0.9, alpha=0.85,
                 label=col, color=POLLUTANT_COLORS.get(col, THEME["gray"]),
                 linestyle=linestyles[i % len(linestyles)])
    style_ax(ax1, title="Pollutant concentrations over time (excl. CO)", ylabel="µg/m³")
    ax1.legend(fontsize=7, framealpha=0.4, ncol=len(cols),
               loc="upper right", labelcolor=THEME["text"])
    ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha="right")
 
    # ── 2. CO separately (different magnitude) ───────────────────────────
    ax_co = ax1.twinx()
    if "CO" in df.columns:
        ax_co.plot(df.index, df["CO"], linewidth=0.7, alpha=0.5,
                   color=POLLUTANT_COLORS["CO"], linestyle="--", label="CO")
        ax_co.set_ylabel("CO (µg/m³)", fontsize=8, color=POLLUTANT_COLORS["CO"])
        ax_co.tick_params(axis="y", colors=POLLUTANT_COLORS["CO"], labelsize=7)
        ax_co.legend(fontsize=7, loc="upper left", framealpha=0.4)
        for spine in ax_co.spines.values():
            spine.set_visible(False)
 
    # ── 3. Diurnal pattern (mean by hour) ────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    hourly = df.groupby(df.index.hour)[cols].mean()
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles2 = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]
    for i, col in enumerate([c for c in cols if c != "CO"]):
        ax2.plot(hourly.index, hourly[col], marker=markers[i % len(markers)],
                 markersize=3, linewidth=1.5, label=col,
                 linestyle=linestyles2[i % len(linestyles2)],
                 color=POLLUTANT_COLORS.get(col, THEME["gray"]))
    style_ax(ax2, title="Mean diurnal pattern by hour", xlabel="Hour of day", ylabel="µg/m³")
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend(fontsize=7, framealpha=0.4, ncol=3, labelcolor=THEME["text"])
 
    # ── 4. Distribution boxplots ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    plot_cols = [c for c in cols if c not in ("CO", "PM25")]
    data_box = [df[c].dropna().values for c in plot_cols]
    bp = ax3.boxplot(data_box, patch_artist=True, widths=0.5,
                     medianprops=dict(color=THEME["text"], linewidth=1.5),
                     whiskerprops=dict(color=THEME["muted"]),
                     capprops=dict(color=THEME["muted"]),
                     flierprops=dict(marker=".", markersize=2,
                                     color=THEME["muted"], alpha=0.4))
    for patch, col in zip(bp["boxes"], plot_cols):
        patch.set_facecolor(POLLUTANT_COLORS.get(col, THEME["gray"]))
        patch.set_alpha(0.7)
    ax3.set_xticklabels(plot_cols, fontsize=8, rotation=30, ha="right",
                        color=THEME["muted"])
    style_ax(ax3, title="Distribution (excl. CO, PM₂.₅)", ylabel="µg/m³")
 
    # ── 5. Correlation heatmap ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    corr_df = df[cols].dropna().corr()
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax4.imshow(corr_df.values, cmap=CMAP_DIVERGING, norm=norm, aspect="auto")
    ax4.set_xticks(range(len(cols)))
    ax4.set_yticks(range(len(cols)))
    ax4.set_xticklabels(corr_df.columns, fontsize=8, rotation=40, ha="right",
                        color=THEME["muted"])
    ax4.set_yticklabels(corr_df.index, fontsize=8, color=THEME["muted"])
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr_df.values[i, j]
            ax4.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=7, color="white" if abs(v) > 0.5 else THEME["text"])
    plt.colorbar(im, ax=ax4, shrink=0.8, pad=0.02)
    style_ax(ax4, title="Pollutant pairwise correlation (Pearson r)")
    ax4.grid(False)
 
    # ── 6. Missing values ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    missing_pct = df[cols].isna().mean() * 100
    bars = ax5.barh(missing_pct.index, missing_pct.values,
                    color=[POLLUTANT_COLORS.get(c, THEME["gray"]) for c in missing_pct.index],
                    height=0.6, alpha=0.8)
    for bar, val in zip(bars, missing_pct.values):
        ax5.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=7, color=THEME["muted"])
    style_ax(ax5, title="Missing values per pollutant", xlabel="% missing")
    ax5.set_xlim(0, max(missing_pct.values) * 1.25 + 2)
    ax5.tick_params(axis="y", labelsize=8, colors=THEME["muted"])
 
    plt.savefig("eda_df1.png", dpi=180, bbox_inches="tight",
                facecolor=THEME["bg"])
    print("✓ Saved eda_df1.png")
    plt.show()
 
 
# ════════════════════════════════════════════════════════════════════════════
# DATASET 2 — MULTI-CITY HEALTH DATASET
# ════════════════════════════════════════════════════════════════════════════
 
def load_df2(path):
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
 
    dt_col = next((c for c in df.columns if c.lower() == "datetime"), None)
    if dt_col is None:
        raise ValueError(
            f"No 'datetime' column found in DF2. Columns present: {list(df.columns)}"
        )
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.rename(columns={dt_col: "datetime"})
 
    for col in df.columns:
        if col not in ("datetime", "City"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
 
 
def plot_df2(df):
    fig = plt.figure(figsize=(18, 14), facecolor=THEME["bg"])
    fig.suptitle("Dataset 2 — Multi-City Air Quality & Health (Netherlands 2023)",
                 fontsize=13, fontweight="bold", color=THEME["text"], y=0.98)
 
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.40,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)
 
    poll_cols = [c for c in ["NO2", "NO", "SO2", "PM25", "Ox", "NOx", "O3", "PM10"]
                 if c in df.columns]
    dis_cols  = {k: v for k, v in DISEASE_COLS.items() if k in df.columns}
 
    # ── 1. Mean NO₂ by city ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    city_no2 = (df.groupby("City")["NO2"].mean()
                  .dropna().sort_values(ascending=True))
    colors_city = [THEME["blue"] if v >= city_no2.median() else THEME["cyan"]
                   for v in city_no2.values]
    bars = ax1.barh(city_no2.index, city_no2.values,
                    color=colors_city, height=0.6, alpha=0.85)
    ax1.axvline(city_no2.median(), color=THEME["rose"], linewidth=1,
                linestyle="--", alpha=0.7, label=f"Median {city_no2.median():.1f}")
    ax1.legend(fontsize=7, framealpha=0.4)
    style_ax(ax1, title="Mean NO₂ by city", xlabel="µg/m³")
    ax1.tick_params(axis="y", labelsize=8, colors=THEME["muted"])
 
    # ── 2. Disease burden bar chart ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if dis_cols:
        dis_means = {dis_cols[k]: df[k].dropna().mean() / 1000
                     for k in dis_cols if k in df.columns}
        dis_sorted = dict(sorted(dis_means.items(), key=lambda x: x[1], reverse=True))
        d_colors = TOL_10[:len(dis_sorted)]
        ax2.bar(range(len(dis_sorted)), list(dis_sorted.values()),
                color=d_colors[:len(dis_sorted)], alpha=0.85, width=0.6)
        ax2.set_xticks(range(len(dis_sorted)))
        ax2.set_xticklabels(list(dis_sorted.keys()), rotation=35, ha="right",
                            fontsize=7, color=THEME["muted"])
    style_ax(ax2, title="Mean disease cases by category", ylabel="Cases (×1000)")
 
    # ── 3. Missing data per pollutant across cities ───────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    miss = df.groupby("City")[poll_cols].apply(lambda g: g.isna().mean() * 100)
    im3 = ax3.imshow(miss.T.values, cmap=CMAP_SEQUENTIAL, aspect="auto", vmin=0, vmax=100)
    ax3.set_xticks(range(len(miss.index)))
    ax3.set_xticklabels(miss.index, rotation=45, ha="right", fontsize=7,
                        color=THEME["muted"])
    ax3.set_yticks(range(len(poll_cols)))
    ax3.set_yticklabels(poll_cols, fontsize=8, color=THEME["muted"])
    plt.colorbar(im3, ax=ax3, shrink=0.8, label="% missing")
    style_ax(ax3, title="Missing data: pollutant × city (%)")
    ax3.grid(False)
 
    # ── 4. Pollutant × disease correlation heatmap ───────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    if dis_cols:
        city_agg = df.groupby("City")[poll_cols + list(dis_cols.keys())].mean()
        corr_rows, pval_rows = {}, {}
        for p in poll_cols:
            row_r, row_p = {}, {}
            for d_col, d_name in dis_cols.items():
                sub = city_agg[[p, d_col]].dropna()
                if len(sub) >= 3:
                    r, pv = pearsonr(sub[p], sub[d_col])
                    row_r[d_name] = round(r, 2)
                    row_p[d_name] = pv
                else:
                    row_r[d_name] = np.nan
                    row_p[d_name] = np.nan
            corr_rows[p] = row_r
            pval_rows[p] = row_p
 
        corr_mat = pd.DataFrame(corr_rows).T  # shape: pollutants × diseases
        pval_mat = pd.DataFrame(pval_rows).T
 
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im4 = ax4.imshow(corr_mat.values, cmap=CMAP_DIVERGING, norm=norm, aspect="auto")
        ax4.set_xticks(range(corr_mat.shape[1]))
        ax4.set_yticks(range(corr_mat.shape[0]))
        ax4.set_xticklabels(corr_mat.columns, rotation=35, ha="right",
                            fontsize=8, color=THEME["muted"])
        ax4.set_yticklabels(corr_mat.index, fontsize=8, color=THEME["muted"])
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                v = corr_mat.values[i, j]
                if np.isnan(v):
                    continue
                sig = pval_mat.values[i, j] < 0.05
                txt = f"{v:.2f}{'*' if sig else ''}"
                ax4.text(j, i, txt, ha="center", va="center",
                         fontsize=7,
                         color="white" if abs(v) > 0.5 else THEME["text"],
                         fontweight="bold" if sig else "normal")
        plt.colorbar(im4, ax=ax4, shrink=0.6, pad=0.02, label="Pearson r")
        ax4.grid(False)
    style_ax(ax4, title="Pollutant × disease correlation heatmap (* p < 0.05)")
 
    # ── 5. City pollutant profile (radar-like bar) ────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    city_means = df.groupby("City")[poll_cols].mean()
    city_norm = city_means.div(city_means.max())
    for i, city in enumerate(city_norm.index):
        ax5.plot(range(len(poll_cols)), city_norm.loc[city].values,
                 alpha=0.5, linewidth=0.9, marker="o", markersize=2)
    ax5.set_xticks(range(len(poll_cols)))
    ax5.set_xticklabels(poll_cols, rotation=35, ha="right", fontsize=7,
                        color=THEME["muted"])
    style_ax(ax5, title="Normalised pollutant profiles by city", ylabel="Normalised value")
 
    # ── 6. Top disease associations scatter ──────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    if "NO2" in df.columns and dis_cols:
        city_agg2 = df.groupby("City")[["NO2"] + list(dis_cols.keys())].mean()
        target_col = list(dis_cols.keys())[0]
        target_name = dis_cols[target_col]
        sub = city_agg2[["NO2", target_col]].dropna()
        ax6.scatter(sub["NO2"], sub[target_col] / 1000, s=60, alpha=0.8,
                    color=THEME["blue"], zorder=3)
        for city, row in sub.iterrows():
            ax6.annotate(city, (row["NO2"], row[target_col] / 1000),
                         fontsize=6, color=THEME["muted"],
                         xytext=(3, 3), textcoords="offset points")
        if len(sub) >= 3:
            m, b = np.polyfit(sub["NO2"], sub[target_col] / 1000, 1)
            xs = np.linspace(sub["NO2"].min(), sub["NO2"].max(), 100)
            ax6.plot(xs, m * xs + b, color=THEME["rose"], linewidth=1,
                     linestyle="--", alpha=0.7)
    style_ax(ax6, title=f"NO₂ vs {target_name} by city",
             xlabel="Mean NO₂ (µg/m³)", ylabel="Cases (×1000)")
 
    # ── 7. Disease correlation with multiple pollutants ───────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    if dis_cols:
        city_agg3 = df.groupby("City")[poll_cols + list(dis_cols.keys())].mean()
        target_col = list(dis_cols.keys())[0]
        rs = []
        for p in poll_cols:
            sub = city_agg3[[p, target_col]].dropna()
            if len(sub) >= 3:
                r, _ = pearsonr(sub[p], sub[target_col])
                rs.append((p, r))
        rs.sort(key=lambda x: x[1])
        pnames, rvals = zip(*rs) if rs else ([], [])
        bar_cols = [THEME["blue"] if v >= 0 else THEME["rose"] for v in rvals]
        ax7.barh(pnames, rvals, color=bar_cols, height=0.55, alpha=0.85)
        ax7.axvline(0, color=THEME["border"], linewidth=0.8)
        style_ax(ax7, title=f"Pollutant correlation with {list(dis_cols.values())[0]}",
                 xlabel="Pearson r")
        ax7.tick_params(axis="y", labelsize=8, colors=THEME["muted"])
 
    # ── 8. PM25 distribution across cities ───────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    if "PM25" in df.columns:
        city_pm = {city: grp["PM25"].dropna().values
                   for city, grp in df.groupby("City")
                   if grp["PM25"].dropna().shape[0] > 0}
        if city_pm:
            sorted_cities = sorted(city_pm.items(),
                                   key=lambda x: np.median(x[1]))
            positions = range(len(sorted_cities))
            bp = ax8.boxplot([v for _, v in sorted_cities],
                             positions=list(positions),
                             vert=False, patch_artist=True, widths=0.5,
                             medianprops=dict(color=THEME["text"], linewidth=1.5),
                             whiskerprops=dict(color=THEME["muted"]),
                             capprops=dict(color=THEME["muted"]),
                             flierprops=dict(marker=".", markersize=2,
                                             alpha=0.4, color=THEME["muted"]))
            for patch in bp["boxes"]:
                patch.set_facecolor(THEME["cyan"])
                patch.set_alpha(0.8)
            ax8.set_yticks(list(positions))
            ax8.set_yticklabels([c for c, _ in sorted_cities],
                                fontsize=7, color=THEME["muted"])
    style_ax(ax8, title="PM₂.₅ distribution by city", xlabel="µg/m³")
 
    plt.savefig("eda_df2.png", dpi=180, bbox_inches="tight",
                facecolor=THEME["bg"])
    print("✓ Saved eda_df2.png")
    plt.show()
 
 
# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    print("Loading DF1...")
    df1 = load_df1(DF1_PATH)
    print(f"  Shape: {df1.shape}, range: {df1.index.min()} → {df1.index.max()}")
    print(f"  Columns: {list(df1.columns)}")
    plot_df1(df1)
 
    print("\nLoading DF2...")
    df2 = load_df2(DF2_PATH)
    print(f"  Shape: {df2.shape}")
    print(f"  Cities: {sorted(df2['City'].unique())}")
    plot_df2(df2)
 
    print("\nDone. Output files: eda_df1.png, eda_df2.png")
 