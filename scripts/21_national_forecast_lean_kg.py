# -*- coding: utf-8 -*-
"""
[21] 이론기반 선별형 KG 예측 (4단 비교) + 블록 부트스트랩

3단(기준/단순뉴스/KG전체)에 "이론기반 소수 KG" 단계를 추가해 4단으로 확장한다.
과적합(49개 특성) 대신 Granger에서 상대적으로 유의했던 소수 KG 지표만 사용한다.

입력: data/processed/national_ev_news_kg.csv
출력 (outputs/kg/tables/):
  - tbl_national_forecast_4tier.csv
  - tbl_bootstrap_rmse.csv
  - forecast_lean_summary.md
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc
from importlib import import_module

warnings.filterwarnings("ignore")

# 18번 모듈 재사용 (파일명이 숫자로 시작해 importlib 사용)
m18 = import_module("18_models_with_kg")
LAGS = m18.LAGS
SIMPLE_BASE = m18.SIMPLE_BASE
KG_BASE = m18.KG_BASE
lag_names = m18.lag_names
add_target_lags = m18.add_target_lags
expanding_forecast = m18.expanding_forecast
diebold_mariano = m18.diebold_mariano

# Granger(20)에서 순방향 상대적으로 유의했던 이론기반 소수 KG 지표
LEAN_KG = ["neg_event_propagation", "salience_newmodel", "frame_entropy"]


def block_bootstrap_ci(err_a, err_b, n_boot=2000, block=6, seed=42):
    """페어드 이동블록 부트스트랩. RMSE 차(A-B) 분포의 95% CI 반환.
    양수면 A(단순)의 오차가 커서 B(선별KG)가 우수."""
    rng = np.random.default_rng(seed)
    n = len(err_a)
    n_blocks = int(np.ceil(n / block))
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        a = err_a[idx]
        b = err_b[idx]
        diffs.append(np.sqrt(np.mean(a ** 2)) - np.sqrt(np.mean(b ** 2)))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(diffs.mean())


def main():
    kc.ensure_dirs()
    df = pd.read_csv(kc.NATIONAL_CSV).sort_values("ym").reset_index(drop=True)
    df = add_target_lags(df, "ev_net_add", lags=(1, 2, 3, 12))

    base_feats = ["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3", "ev_net_add_lag12"]
    simple_feats = base_feats + lag_names(SIMPLE_BASE)
    lean_feats = simple_feats + lag_names(LEAN_KG)
    kg_feats = simple_feats + lag_names(KG_BASE)

    # 4단 모두 동일 평가구간(최대 특성 기준 dropna)
    common = df.dropna(subset=["ev_net_add"] + kg_feats).reset_index(drop=True)

    tiers = [
        ("기준(AR)", base_feats),
        ("단순뉴스", simple_feats),
        ("KG선별(이론3)", lean_feats),
        ("KG전체", kg_feats),
    ]
    results, err_store = [], {}
    for name, feats in tiers:
        rmse, mae, r2, da, preds, actuals, _ = expanding_forecast(common, "ev_net_add", feats)
        results.append({"model": name, "n_features": len(feats),
                        "RMSE": round(rmse, 1), "MAE": round(mae, 1),
                        "R2": round(r2, 4), "direction_acc": round(da, 3)})
        err_store[name] = actuals - preds

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_national_forecast_4tier.csv"),
                  index=False, encoding="utf-8-sig")

    # DM + 블록 부트스트랩: 단순뉴스 vs KG선별
    dm_lean = diebold_mariano(err_store["단순뉴스"], err_store["KG선별(이론3)"], h=1)
    lo, hi, mean_d = block_bootstrap_ci(err_store["단순뉴스"], err_store["KG선별(이론3)"])
    dm_full = diebold_mariano(err_store["단순뉴스"], err_store["KG전체"], h=1)
    lo2, hi2, mean_d2 = block_bootstrap_ci(err_store["단순뉴스"], err_store["KG전체"])

    boot = pd.DataFrame([
        {"비교": "단순뉴스 - KG선별", "RMSE차_평균": round(mean_d, 1),
         "CI2.5": round(lo, 1), "CI97.5": round(hi, 1), "DM": round(dm_lean, 3),
         "선별KG우수": lo > 0},
        {"비교": "단순뉴스 - KG전체", "RMSE차_평균": round(mean_d2, 1),
         "CI2.5": round(lo2, 1), "CI97.5": round(hi2, 1), "DM": round(dm_full, 3),
         "선별KG우수": lo2 > 0},
    ])
    boot.to_csv(os.path.join(kc.TABLE_DIR, "tbl_bootstrap_rmse.csv"),
                index=False, encoding="utf-8-sig")

    lines = ["# 선별형 KG 예측 (4단) + 블록 부트스트랩\n"]
    lines.append("## 1. 4단 예측 성능 (확장창 1기 예측, 동일 평가구간)\n")
    lines.append(res_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 2. 블록 부트스트랩 RMSE 차 (95% CI, block=6, n=2000)\n")
    lines.append(boot.to_markdown(index=False))
    lines.append("")
    best = res_df.loc[res_df["RMSE"].idxmin(), "model"]
    lines.append(f"- 최저 RMSE 모형: **{best}**")
    lean_r = res_df.loc[res_df["model"] == "KG선별(이론3)", "RMSE"].values[0]
    simp_r = res_df.loc[res_df["model"] == "단순뉴스", "RMSE"].values[0]
    lines.append(f"- KG선별 RMSE={lean_r} vs 단순뉴스 RMSE={simp_r} → "
                 f"{'개선' if lean_r < simp_r else '미개선'}")
    lines.append(f"- 부트스트랩 CI가 0을 포함하면 개선은 통계적으로 불확실 "
                 f"(단순-KG선별 CI=[{round(lo,1)}, {round(hi,1)}]).")

    path = os.path.join(kc.TABLE_DIR, "forecast_lean_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[21] 저장: {path}")


if __name__ == "__main__":
    main()
