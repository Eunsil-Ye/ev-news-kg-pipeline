# -*- coding: utf-8 -*-
"""
[22] 패널 강건성: 계절성 통제 / 대안 종속변수 / 사건월 제외

기준 패널모형(18번 run_panel: 지역FE + AR + 뉴스/KG 주효과 + 상호작용)에 대해
세 가지 강건성 점검을 수행하고 상호작용 계수의 방향·유의성 안정성을 비교한다.

입력: data/processed/panel_ev_news_kg.csv
출력 (outputs/kg/tables/):
  - tbl_panel_robustness.csv    (스펙별 R2/상호작용 계수/유의성)
  - panel_robustness_summary.md
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc
from importlib import import_module

warnings.filterwarnings("ignore")

m18 = import_module("18_models_with_kg")
lag_names = m18.lag_names
add_target_lags = m18.add_target_lags

INTER_MAIN = ["safety_neg_intensity_lag1", "salience_newmodel_lag1", "cent_safety_lag1"]
MODERATORS = ["charger_per_ev", "ev_share"]
MAIN_LAGS = lag_names(["neg_share", "frshare_newmodel", "frshare_safety"])
INTER_NAMES = ["safetyNeg_x_chgPerEv", "newmodel_x_evshare", "centSafety_x_chgPerEv"]


def build_alt_dvs(df):
    """대안 종속변수 생성 (지역별)."""
    g = df.groupby("region", group_keys=False)
    df["ev_net_add_ma3"] = g["ev_net_add"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    df["ev_net_add_yoy"] = g["ev_net_add"].transform(lambda s: s.pct_change(12))
    df["ev_net_add_yoy"] = df["ev_net_add_yoy"].replace([np.inf, -np.inf], np.nan)
    return df


def fit_spec(df, dv, add_month_fe=False, exclude=None, label=""):
    d = df.copy()
    if exclude is not None:
        d = d[~d["ym"].isin(exclude)]

    need = ([dv, "ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3", "region"]
            + INTER_MAIN + MODERATORS + MAIN_LAGS)
    d = d.dropna(subset=[c for c in need if c in d.columns]).reset_index(drop=True)
    if len(d) < 60:
        return None

    cont = (["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3"]
            + INTER_MAIN + MODERATORS + MAIN_LAGS)
    cont = [c for c in cont if c in d.columns]
    dz = d.copy()
    for c in cont:
        s = dz[c].std()
        dz[c] = (dz[c] - dz[c].mean()) / s if s > 0 else 0.0

    region_dum = pd.get_dummies(dz["region"], prefix="reg", drop_first=True).astype(float)
    ar = dz[["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3"]].reset_index(drop=True)
    main = dz[MAIN_LAGS + INTER_MAIN].reset_index(drop=True)

    inter = pd.DataFrame(index=dz.index)
    inter["safetyNeg_x_chgPerEv"] = dz["safety_neg_intensity_lag1"] * dz["charger_per_ev"]
    inter["newmodel_x_evshare"] = dz["salience_newmodel_lag1"] * dz["ev_share"]
    inter["centSafety_x_chgPerEv"] = dz["cent_safety_lag1"] * dz["charger_per_ev"]
    inter = inter.reset_index(drop=True)

    parts = [ar, main, inter, region_dum.reset_index(drop=True)]
    if add_month_fe:
        month = pd.to_datetime(d["ym"] + "-01").dt.month
        month_dum = pd.get_dummies(month, prefix="m", drop_first=True).astype(float).reset_index(drop=True)
        parts.append(month_dum)

    X = pd.concat(parts, axis=1)
    X = sm.add_constant(X)
    y = d[dv].values
    m = sm.OLS(y, X.values).fit(cov_type="HC1")
    # 뉴스/KG 지표가 월 내 17개 지역에 공통이므로 18번과 같이 월 클러스터를 주 스펙으로 둔다.
    ym_groups = pd.factorize(d["ym"])[0]
    mc = sm.OLS(y, X.values).fit(cov_type="cluster", cov_kwds={"groups": ym_groups})
    cols = list(X.columns)

    row = {"spec": label, "dv": dv, "n": int(m.nobs), "n_months": int(ym_groups.max() + 1),
           "r2": round(m.rsquared, 4), "adj_r2": round(m.rsquared_adj, 4)}
    for nm in INTER_NAMES:
        if nm in cols:
            idx = cols.index(nm)
            row[f"{nm}_coef"] = round(m.params[idx], 4)
            row[f"{nm}_p"] = round(m.pvalues[idx], 4)
            row[f"{nm}_p_cluster"] = round(mc.pvalues[idx], 4)
    return row


def main():
    kc.ensure_dirs()
    df = pd.read_csv(kc.PANEL_CSV).sort_values(["region", "ym"]).reset_index(drop=True)
    df = add_target_lags(df, "ev_net_add", lags=(1, 2, 3), group="region")
    df = build_alt_dvs(df)

    covid = [f"2020-{mm:02d}" for mm in range(1, 13)]
    specs = []
    specs.append(fit_spec(df, "ev_net_add", label="기준(재현)"))
    specs.append(fit_spec(df, "ev_net_add", add_month_fe=True, label="+월고정효과"))
    specs.append(fit_spec(df, "ev_net_add", exclude=["2024-08"], label="사건월제외(2024-08)"))
    specs.append(fit_spec(df, "ev_net_add", exclude=covid, label="코로나년제외(2020)"))
    specs.append(fit_spec(df, "ev_net_add", exclude=["2024-08"] + covid, label="사건월+코로나제외"))
    specs.append(fit_spec(df, "ev_share", label="대안DV: ev_share"))
    specs.append(fit_spec(df, "ev_net_add_ma3", label="대안DV: 3개월MA"))
    specs.append(fit_spec(df, "ev_net_add_yoy", label="대안DV: YoY증가율"))

    rows = [s for s in specs if s is not None]
    rob = pd.DataFrame(rows)
    rob.to_csv(os.path.join(kc.TABLE_DIR, "tbl_panel_robustness.csv"),
               index=False, encoding="utf-8-sig")

    # 상호작용 계수 방향 안정성 요약
    lines = ["# 패널 강건성 점검 요약\n"]
    lines.append("## 스펙별 R2 및 상호작용 계수/유의성\n")
    lines.append("`_p_cluster` = 월 클러스터(주 스펙), `_p` = HC1(참고).\n")
    show_cols = ["spec", "dv", "n", "n_months", "r2", "adj_r2"]
    for nm in INTER_NAMES:
        show_cols += [f"{nm}_coef", f"{nm}_p_cluster", f"{nm}_p"]
    show_cols = [c for c in show_cols if c in rob.columns]
    lines.append(rob[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 상호작용 계수 부호 안정성\n")
    for nm in INTER_NAMES:
        cc = f"{nm}_coef"
        if cc not in rob.columns:
            continue
        signs = np.sign(rob[cc].dropna())
        stable = "일관" if signs.nunique() == 1 else "혼재"
        n_all = int(rob[cc].notna().sum())
        parts = []
        for tag, pc in (("월 클러스터", f"{nm}_p_cluster"), ("HC1", f"{nm}_p")):
            nsig = int((rob[pc] < kc.ALPHA).sum())
            nsig10 = int((rob[pc] < 0.1).sum())
            parts.append(f"{tag} p<{kc.ALPHA} {nsig}/{n_all} (참고: p<0.1 {nsig10}개)")
        lines.append(f"- {nm}: 부호 {stable}, " + "; ".join(parts))

    path = os.path.join(kc.TABLE_DIR, "panel_robustness_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[22] 저장: {path}")


if __name__ == "__main__":
    main()
