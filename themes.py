# analysis_emotions_rq2_simple.py
# Direct analysis for RQ2 (no clustering).
#
# Provides:
#  1. Yearly trends (line charts + heatmap)
#  2. Dominant emotions per year
#  3. Emotion co-occurrence (correlation heatmap only)
#  4. Period comparisons (effect sizes)
#  5. NEW: Top-10 most prevalent emotions:
#       - box + whisker by year (with legend)
#       - histogram (overall)

import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_per_posts(pattern="analysis/per_post_*.csv", year_min=2018, year_max=2025):
    frames = []
    for p in glob.glob(pattern):
        try:
            y = int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        except Exception:
            continue
        if not (year_min <= y <= year_max):
            continue
        df = pd.read_csv(p)
        df["year"] = y
        for e in EMOTIONS:
            if e in df.columns:
                df[e] = pd.to_numeric(df[e], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _write_emotion_means_csv(yearly: pd.DataFrame, outdir: str):
    """
    Writes a CSV with mean score per emotion by year.
    Includes an extra 'ALL' row that is the mean across all years.
    """
    out_path = os.path.join(outdir, "emotion_yearly_means.csv")
    table = yearly.copy()
    overall = table.mean(axis=0).to_frame().T
    overall.index = ["ALL"]
    table = pd.concat([table, overall], axis=0)
    table.index.name = "year"
    table.to_csv(out_path)
    print("[WROTE]", out_path)

def plot_trends(per_posts, outdir):
    yearly = per_posts.groupby("year")[EMOTIONS].mean()
    if yearly.empty:
        return

    # Ensure chronological order
    yearly = yearly.sort_index()

    # -------- Heatmap (kept) --------
    plt.figure(figsize=(12, 6))
    plt.imshow(yearly.T, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Mean score")
    plt.title("Year × Emotion Heatmap (means)")
    plt.xlabel("Year")
    plt.ylabel("Emotion")
    plt.xticks(range(len(yearly.index)), yearly.index, rotation=45)
    plt.yticks(range(len(yearly.columns)), yearly.columns)
    plt.tight_layout()
    path = os.path.join(outdir, "year_emotion_heatmap.png")
    plt.savefig(path, dpi=200); plt.close(); print("[WROTE]", path)

    # -------- All emotions in one line chart --------
    plt.figure(figsize=(14, 8))
    for e in EMOTIONS:
        if e in yearly.columns:
            plt.plot(yearly.index, yearly[e], marker=".", linewidth=1.0, alpha=0.9, label=e)
    plt.title("Emotion trends (all 28 emotions)")
    plt.xlabel("Year")
    plt.ylabel("Mean score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Emotions", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    path = os.path.join(outdir, "trends_all.png")
    plt.savefig(path, dpi=200); plt.close(); print("[WROTE]", path)

    # -------- Top-10 by variance (over time) --------
    var = yearly.var().sort_values(ascending=False)
    top10 = var.head(10).index.tolist()

    plt.figure(figsize=(14, 8))
    for e in top10:
        plt.plot(yearly.index, yearly[e], marker="o", linewidth=1.5, alpha=0.95, label=e)
    plt.title("Emotion trends (top-10 by variance)")
    plt.xlabel("Year")
    plt.ylabel("Mean score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Emotions", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    path = os.path.join(outdir, "trends_top10.png")
    plt.savefig(path, dpi=200); plt.close(); print("[WROTE]", path)

    # -------- CSV with yearly means (+ ALL) --------
    _write_emotion_means_csv(yearly, outdir)

def dominant_emotions_per_year(per_posts, outdir):
    """
    For each year:
      - compute per-emotion mean and standard deviation across posts
      - select the top 3 emotions by mean
      - plot bars with STDEV error bars
    Also writes a CSV with the top-3 (mean & stdev) per year.
    """
    if per_posts.empty:
        return

    # Ensure we have numeric columns
    for e in EMOTIONS:
        if e in per_posts.columns:
            per_posts[e] = pd.to_numeric(per_posts[e], errors="coerce")

    # Aggregate mean and std per year
    yearly_mean = per_posts.groupby("year")[EMOTIONS].mean().sort_index()
    yearly_std  = per_posts.groupby("year")[EMOTIONS].std(ddof=1).sort_index()

    if yearly_mean.empty:
        return

    # CSV of top-3 per year (with mean & stdev)
    rows = []
    for year in yearly_mean.index:
        means = yearly_mean.loc[year]
        stdev = yearly_std.loc[year]

        # Pick top 3 by mean
        top3_idx = means.sort_values(ascending=False).head(3).index
        for emo in top3_idx:
            rows.append({
                "year": year,
                "emotion": emo,
                "mean": float(means[emo]),
                "stdev": float(0.0 if pd.isna(stdev[emo]) else stdev[emo]),
            })

        # ---- Plot: bars with STDEV error bars ----
        top_means = means[top3_idx]
        top_stdev = stdev[top3_idx].fillna(0.0)

        plt.figure(figsize=(6, 4))
        plt.bar(top_means.index, top_means.values, yerr=top_stdev.values,
                capsize=5, alpha=0.9)
        plt.title(f"Top 3 emotions in {year} (mean ± stdev)")
        plt.ylabel("Mean score")
        plt.ylim(0, 1)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        path = os.path.join(outdir, f"top3_{year}_stdev.png")
        plt.tight_layout()
        plt.savefig(path, dpi=200); plt.close(); print("[WROTE]", path)

    # Write the CSV
    df_top3 = pd.DataFrame(rows, columns=["year", "emotion", "mean", "stdev"])
    csv_path = os.path.join(outdir, "dominant_emotions_per_year_top3_mean_stdev.csv")
    df_top3.to_csv(csv_path, index=False)
    print("[WROTE]", csv_path)

def cooccurrence_analysis(per_posts, outdir):
    corr = per_posts[EMOTIONS].corr()
    plt.figure(figsize=(10,8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Emotion co-occurrence (correlation heatmap)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.tight_layout()
    path = os.path.join(outdir, "emotion_cooccurrence_heatmap.png")
    plt.savefig(path, dpi=200); plt.close(); print("[WROTE]", path)

def period_contrasts(per_posts, outdir):
    def period(y):
        if y <= 2019: return "pre"
        if 2020 <= y <= 2021: return "covid"
        return "post"
    per_posts["period"] = per_posts["year"].apply(period)

    results = []
    for e in EMOTIONS:
        if e not in per_posts.columns:
            continue
        for a,b in [("covid","pre"), ("post","pre"), ("post","covid")]:
            A = per_posts.loc[per_posts["period"]==a, e].dropna()
            B = per_posts.loc[per_posts["period"]==b, e].dropna()
            if len(A)<2 or len(B)<2: continue
            diff = A.mean()-B.mean()
            sp = np.sqrt(((len(A)-1)*A.std()**2 + (len(B)-1)*B.std()**2) / max(len(A)+len(B)-2,1))
            d = diff/(sp+1e-9)
            results.append({"emotion":e,"contrast":f"{a}-{b}","mean_diff":diff,"cohen_d":d})
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(outdir, "period_effect_sizes.csv"), index=False)
    print("[WROTE] period_effect_sizes.csv")

    sub = out[out["contrast"]=="covid-pre"].copy()
    if not sub.empty:
        sub = sub.reindex(sub["cohen_d"].abs().sort_values(ascending=False).index)[:8]
        plt.figure()
        plt.bar(sub["emotion"], sub["cohen_d"], color="lightcoral")
        plt.title("COVID vs Pre period: effect sizes (Cohen's d)")
        plt.xticks(rotation=45, ha="right")
        path = os.path.join(outdir, "effect_sizes_covid_vs_pre.png")
        plt.tight_layout()
        plt.savefig(path, dpi=200); plt.close(); print("[WROTE]", path)

# -----------------------
# NEW: Boxplot legend + per-emotion plotting helpers
# -----------------------

def _boxplot_legend():
    """Build a clean legend explaining boxplot elements."""
    handles = [
        Patch(alpha=0.35, label="IQR box (Q1–Q3)"),
        Line2D([0], [0], color="C7", lw=2, label="Median"),
        Line2D([0], [0], color="C3", lw=2, label="Mean (line)"),
        Line2D([0], [0], color="0.3", lw=1.5, label="Whiskers (to 1.5×IQR)"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=6,
               markerfacecolor='none', markeredgecolor='0.2', label="Outliers"),
    ]
    return handles

def _boxplot_emotion_by_year(per_posts: pd.DataFrame, emotion: str, outdir: str):
    """Box + whiskers for a single emotion split by year, with legend."""
    if per_posts.empty or "year" not in per_posts.columns or emotion not in per_posts.columns:
        return
    years = sorted(pd.to_numeric(per_posts["year"], errors="coerce").dropna().unique().tolist())
    data = [per_posts.loc[per_posts["year"] == y, emotion].dropna().values for y in years]
    if all(len(d) == 0 for d in data):
        return

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(
        data,
        labels=years,
        showmeans=True,
        meanline=True,      # plots the mean as a line
        whis=1.5,
        patch_artist=True,  # so we can color boxes
        flierprops=dict(marker='o', markersize=4, markerfacecolor='none', markeredgecolor='0.2', alpha=0.9),
        boxprops=dict(facecolor="C0", alpha=0.35, edgecolor="C0"),
        medianprops=dict(color="C7", linewidth=2),
        meanprops=dict(color="C3", linewidth=2),
        whiskerprops=dict(color="0.3", linewidth=1.5),
        capprops=dict(color="0.3", linewidth=1.5),
    )

    plt.title(f"Distribution of '{emotion}' by year (box + whiskers)")
    plt.xlabel("Year")
    plt.ylabel(f"'{emotion}' score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(
    handles=_boxplot_legend(),
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0.,
    frameon=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    path = os.path.join(outdir, f"boxplot_{emotion}_by_year.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close(); print("[WROTE]", path)

def _histogram_emotion(per_posts: pd.DataFrame, emotion: str, outdir: str, bins: int = 30):
    """Simple overall histogram for a single emotion."""
    if emotion not in per_posts.columns:
        return
    s = pd.to_numeric(per_posts[emotion], errors="coerce").dropna()
    if s.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=bins, alpha=0.85, edgecolor="black")
    plt.title(f"Histogram of '{emotion}' scores (all years)")
    plt.xlabel(f"'{emotion}' score")
    plt.ylabel("Count")
    plt.xlim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    path = os.path.join(outdir, f"hist_{emotion}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close(); print("[WROTE]", path)

def generate_top_emotion_distributions(per_posts: pd.DataFrame, outdir: str, top_n: int = 10, bins: int = 30):
    """
    Finds the top-N most prevalent emotions (highest overall mean),
    and for each:
      - saves a box+whisker plot by year (with legend)
      - saves an overall histogram
    Also writes a small CSV listing the chosen top-N by mean.
    """
    if per_posts.empty:
        return

    # Ensure numeric
    for e in EMOTIONS:
        if e in per_posts.columns:
            per_posts[e] = pd.to_numeric(per_posts[e], errors="coerce")

    # Rank by overall mean (prevalence)
    overall_means = per_posts[EMOTIONS].mean(numeric_only=True).dropna().sort_values(ascending=False)
    top_emotions = overall_means.head(top_n).index.tolist()
    if not top_emotions:
        return

    # Write a CSV listing the selected top emotions
    top_df = overall_means.head(top_n).to_frame(name="overall_mean").reset_index(names=["emotion"])
    csv_path = os.path.join(outdir, f"top_{top_n}_emotions_by_mean.csv")
    top_df.to_csv(csv_path, index=False)
    print("[WROTE]", csv_path)

    # Generate plots
    for emo in top_emotions:
        _boxplot_emotion_by_year(per_posts, emo, outdir)
        _histogram_emotion(per_posts, emo, outdir, bins=bins)

# -----------------------
# Driver
# -----------------------

def main():
    outdir = "analysis/rq2_simple"
    ensure_dir(outdir)
    per_posts = load_per_posts()
    if per_posts.empty:
        print("[WARN] No data found."); return

    plot_trends(per_posts, outdir)
    dominant_emotions_per_year(per_posts, outdir)
    cooccurrence_analysis(per_posts, outdir)
    period_contrasts(per_posts, outdir)

    # NEW: top-10 prevalence distributions (boxplots + histograms)
    generate_top_emotion_distributions(per_posts, outdir, top_n=10, bins=30)

if __name__ == "__main__":
    main()
