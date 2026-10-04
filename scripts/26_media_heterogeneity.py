# -*- coding: utf-8 -*-
"""
[26] 매체별 이질성 분석

news_tagged.csv 의 newspaper 컬럼으로 매체별 감성/프레임 분포 차이를 비교한다.
기사 약 3만 건에서는 카이제곱 p값이 작은 차이에도 0에 가까워지므로 Cramér's V 를
함께 보고하고, 월별 매체 구성 변화가 전국 감성 지표를 움직이는 정도를 점검한다.

출력:
  - outputs/kg/tables/tbl_media_sentiment.csv
  - outputs/kg/tables/tbl_media_frames.csv
  - outputs/kg/tables/tbl_media_frame_effect.csv
  - outputs/kg/tables/tbl_media_composition.csv
  - outputs/kg/tables/tbl_media_composition_sensitivity.csv
  - outputs/kg/figures/media_sentiment.png
  - outputs/kg/figures/media_frames.png
  - outputs/kg/tables/media_heterogeneity_summary.md
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

warnings.filterwarnings("ignore")

SENT_INDICATORS = {
    "neg_share": lambda s: (s == "부정").astype(float),
    "pos_share": lambda s: (s == "긍정").astype(float),
    "net_sentiment": lambda s: (s == "긍정").astype(float) - (s == "부정").astype(float),
}


def cramers_v(table: np.ndarray) -> tuple[float, float, int, float]:
    chi2, p, dof, _ = chi2_contingency(table)
    k = min(table.shape) - 1
    return chi2, p, dof, float(np.sqrt(chi2 / (table.sum() * k)))


def frame_effects(nt: pd.DataFrame) -> pd.DataFrame:
    # 한 기사가 여러 프레임을 가지므로 프레임마다 기사 단위 등장 여부(2 x 매체)로 따로 검정한다.
    frames = nt["frames"].fillna("").str.split(";")
    rows = []
    for fr in kc.FRAME_NAMES:
        has = frames.apply(lambda xs: fr in xs)
        ct = pd.crosstab(has, nt["newspaper"]).values
        chi2, p, dof, v = cramers_v(ct)
        rows.append({"frame": fr, "chi2": round(chi2, 1), "dof": dof,
                     "p": p, "cramers_v": round(v, 4)})
    return pd.DataFrame(rows)


def composition_sensitivity(nt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = nt[(nt["ym"] >= kc.ANALYSIS_START) & (nt["ym"] <= kc.ANALYSIS_END)].copy()
    comp = pd.crosstab(df["ym"], df["newspaper"], normalize="index")
    comp_summary = comp.describe().loc[["mean", "std", "min", "max"]].round(4)

    rows = []
    for name, fn in SENT_INDICATORS.items():
        df["_v"] = fn(df["sentiment"])
        pooled = df.groupby("ym")["_v"].mean()
        equal = df.groupby(["ym", "newspaper"])["_v"].mean().unstack().mean(axis=1)
        diff = (pooled - equal).abs()
        rows.append({
            "indicator": name,
            "months": len(pooled),
            "corr_level": round(pooled.corr(equal), 4),
            "corr_diff": round(pooled.diff().corr(equal.diff()), 4),
            "mean_abs_gap_pp": round(diff.mean() * 100, 3),
            "max_abs_gap_pp": round(diff.max() * 100, 3),
            "pooled_sd_pp": round(pooled.std() * 100, 3),
        })
    return comp_summary, pd.DataFrame(rows)


def main():
    kc.ensure_dirs()
    kc.set_korean_font()
    nt = pd.read_csv(kc.NEWS_TAGGED_CSV)

    # --- 감성 분포 (매체 x 감성) ---
    sent_ct = pd.crosstab(nt["newspaper"], nt["sentiment"])
    sent_share = sent_ct.div(sent_ct.sum(axis=1), axis=0).round(4)
    sent_share["n"] = sent_ct.sum(axis=1)
    sent_share.to_csv(os.path.join(kc.TABLE_DIR, "tbl_media_sentiment.csv"),
                      encoding="utf-8-sig")
    chi2_s, p_s, dof_s, v_s = cramers_v(sent_ct.values)

    # --- 프레임 분포 (매체별 프레임 등장 비율) ---
    frame_rows = []
    for paper, g in nt.groupby("newspaper"):
        n = len(g)
        row = {"newspaper": paper, "n": n}
        exploded = g["frames"].fillna("").str.split(";")
        for fr in kc.FRAME_NAMES:
            cnt = exploded.apply(lambda xs: fr in xs).sum()
            row[fr] = round(cnt / n, 4) if n else 0.0
        frame_rows.append(row)
    frame_df = pd.DataFrame(frame_rows).set_index("newspaper")
    frame_df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_media_frames.csv"),
                    encoding="utf-8-sig")

    frame_eff = frame_effects(nt)
    frame_eff.to_csv(os.path.join(kc.TABLE_DIR, "tbl_media_frame_effect.csv"),
                     index=False, encoding="utf-8-sig")

    comp_summary, comp_sens = composition_sensitivity(nt)
    comp_summary.to_csv(os.path.join(kc.TABLE_DIR, "tbl_media_composition.csv"),
                        encoding="utf-8-sig")
    comp_sens.to_csv(os.path.join(kc.TABLE_DIR, "tbl_media_composition_sensitivity.csv"),
                     index=False, encoding="utf-8-sig")

    # --- 그림: 감성 스택바 ---
    fig, ax = plt.subplots(figsize=(9, 5))
    order = [c for c in ["긍정", "중립", "부정"] if c in sent_share.columns]
    colors = {"긍정": "#4c9f70", "중립": "#b0b0b0", "부정": "#d1495b"}
    sent_share[order].plot(kind="bar", stacked=True, ax=ax,
                           color=[colors[c] for c in order])
    ax.set_ylabel("감성 비중")
    ax.set_title(f"매체별 감성 분포 (Cramér's V={v_s:.3f})")
    ax.legend(title="감성", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(kc.FIG_DIR, "media_sentiment.png"), dpi=150)
    plt.close(fig)

    # --- 그림: 프레임 히트맵 스타일(막대) ---
    fig, ax = plt.subplots(figsize=(11, 5))
    frame_df[kc.FRAME_NAMES].T.plot(kind="bar", ax=ax)
    ax.set_ylabel("프레임 등장 비율")
    ax.set_title("매체별 프레임 분포")
    ax.legend(title="매체", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(kc.FIG_DIR, "media_frames.png"), dpi=150)
    plt.close(fig)

    lines = ["# 매체별 이질성 분석 요약\n"]
    lines.append(f"- 매체 수: {sent_ct.shape[0]}, 총 기사: {int(sent_ct.values.sum())}\n")
    lines.append("## 매체별 감성 분포 (비중)\n")
    lines.append(sent_share.to_markdown())
    lines.append("")
    lines.append(f"- 감성×매체: chi2={chi2_s:.1f}, dof={dof_s}, p={p_s:.3g}, "
                 f"**Cramér's V={v_s:.4f}**")
    lines.append("- 기사 수가 많아 p값은 작은 차이에도 0에 가까워진다. 차이의 크기는 V로 판단한다"
                 "(0.1 미만은 작은 효과).")
    lines.append("")
    lines.append("## 매체별 프레임 등장 비율\n")
    lines.append(frame_df.to_markdown())
    lines.append("")
    lines.append("한 기사가 여러 프레임을 가지므로 프레임을 한 표에 합쳐 검정하지 않고, "
                 "프레임마다 기사 단위 등장 여부로 따로 검정했다.\n")
    lines.append(frame_eff.to_markdown(index=False))
    lines.append("")
    lines.append(f"## 월별 매체 구성과 전국 감성 지표 ({kc.ANALYSIS_START}~{kc.ANALYSIS_END})\n")
    lines.append("월별 매체 비중:\n")
    lines.append(comp_summary.to_markdown())
    lines.append("")
    lines.append("기사를 그대로 합친 지표(pooled)와 세 매체에 같은 가중치를 준 지표(equal)의 비교:\n")
    lines.append(comp_sens.to_markdown(index=False))
    lines.append("")
    lines.append("- 두 지표의 차이가 pooled 지표의 월별 표준편차보다 훨씬 작으면, 월별 매체 구성 "
                 "변화가 전국 지표를 움직이는 정도는 작다.")

    path = os.path.join(kc.TABLE_DIR, "media_heterogeneity_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[26] 저장: {path}")


if __name__ == "__main__":
    main()
