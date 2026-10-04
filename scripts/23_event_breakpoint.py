# -*- coding: utf-8 -*-
"""
[23] 구조변화(breakpoint) 탐지 — 형식적 사건연구

kg_monthly_features.csv 의 담론지표(safety_neg_intensity, cent_safety 등)에 대해
(1) OLS-CUSUM 구조안정성 검정, (2) supF 단일 평균전환 탐지를 적용해
2024-08(전기차 화재 급증) 전환점을 통계적으로 확인한다.

출력 (outputs/kg/):
  - tables/tbl_breakpoints.csv
  - figures/breakpoint_<series>.png
  - tables/breakpoint_summary.md
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
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

warnings.filterwarnings("ignore")

SERIES = ["safety_neg_intensity", "cent_safety", "neg_event_propagation", "salience_newmodel"]
TRIM = 0.15


def supf_meanshift(y):
    """단일 평균전환 supF 검정. (best_idx, supF) 반환."""
    n = len(y)
    lo = int(np.floor(n * TRIM))
    hi = int(np.ceil(n * (1 - TRIM)))
    y = np.asarray(y, dtype=float)
    x_const = np.ones((n, 1))
    rss_r = np.sum((y - y.mean()) ** 2)
    best_f, best_i = -np.inf, None
    for i in range(lo, hi):
        d = np.zeros(n)
        d[i:] = 1.0
        X = np.column_stack([x_const[:, 0], d])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rss_u = np.sum(resid ** 2)
        # 단일 제약(전환더미 1개)에 대한 F
        f = (rss_r - rss_u) / (rss_u / (n - 2)) if rss_u > 0 else 0.0
        if f > best_f:
            best_f, best_i = f, i
    return best_i, float(best_f)


def main():
    kc.ensure_dirs()
    kc.set_korean_font()
    df = pd.read_csv(kc.KG_MONTHLY_FEATURES_CSV).sort_values("ym").reset_index(drop=True)
    ym = df["ym"].tolist()

    rows = []
    for s in SERIES:
        if s not in df.columns:
            continue
        y = df[s].astype(float).values

        # (1) OLS-CUSUM 검정 (상수항 회귀 잔차 기반)
        X = sm.add_constant(np.arange(len(y), dtype=float))
        resid = sm.OLS(y, X).fit().resid
        try:
            cusum_stat, cusum_p, _ = breaks_cusumolsresid(resid.reshape(-1, 1))
        except Exception:
            cusum_stat, cusum_p = np.nan, np.nan

        # (2) supF 단일 전환점
        bi, supf = supf_meanshift(y)
        bym = ym[bi] if bi is not None else None
        pre_mean = float(np.mean(y[:bi])) if bi else np.nan
        post_mean = float(np.mean(y[bi:])) if bi else np.nan

        rows.append({
            "series": s, "cusum_stat": round(float(cusum_stat), 4),
            "cusum_p": round(float(cusum_p), 4),
            "break_ym": bym, "supF": round(supf, 3),
            "pre_mean": round(pre_mean, 4), "post_mean": round(post_mean, 4),
            "shift_pct": round((post_mean / pre_mean - 1) * 100, 1) if pre_mean else np.nan,
        })

        # 그림
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(range(len(y)), y, color="#333", lw=1.2, label=s)
        if bi is not None:
            ax.axvline(bi, color="crimson", ls="--", lw=1.5, label=f"전환점 {bym}")
            ax.hlines(pre_mean, 0, bi, color="steelblue", lw=2, alpha=0.7)
            ax.hlines(post_mean, bi, len(y) - 1, color="darkorange", lw=2, alpha=0.7)
        step = max(1, len(ym) // 12)
        ax.set_xticks(range(0, len(ym), step))
        ax.set_xticklabels([ym[i] for i in range(0, len(ym), step)], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{s} 구조변화 (supF={supf:.1f}, CUSUM p={cusum_p:.3f})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(kc.FIG_DIR, f"breakpoint_{s}.png"), dpi=150)
        plt.close(fig)

    bp = pd.DataFrame(rows)
    bp.to_csv(os.path.join(kc.TABLE_DIR, "tbl_breakpoints.csv"),
              index=False, encoding="utf-8-sig")

    lines = ["# 구조변화(breakpoint) 탐지 요약\n"]
    lines.append("## CUSUM 안정성 검정 + supF 단일 전환점\n")
    lines.append(bp.to_markdown(index=False))
    lines.append("")
    lines.append("- CUSUM p<0.05 이면 구조안정성 기각(구조변화 존재).")
    lines.append("- break_ym이 2024-08 부근이면 화재 사건을 통계적으로 포착한 것.")
    path = os.path.join(kc.TABLE_DIR, "breakpoint_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[23] 저장: {path}")


if __name__ == "__main__":
    main()
