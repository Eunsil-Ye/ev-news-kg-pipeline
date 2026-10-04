# -*- coding: utf-8 -*-
"""
[20] 다중비교 보정 + 양방향 Granger + VAR/IRF/FEVD

입력: data/processed/national_ev_news_kg.csv
출력 (outputs/kg/tables/, outputs/kg/figures/):
  - tbl_granger_corrected.csv   양방향 Granger + BH(FDR)/Bonferroni 보정
  - tbl_var_fevd.csv            VAR 분산분해
  - var_irf.png                 충격반응함수
  - granger_var_summary.md

다중비교 보정(BH, Bonferroni)은 정방향 검정에만 적용한다. 역방향 검정은 역인과를
진단하는 보조 검정이므로 보정하지 않는다.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SIMPLE_BASE = ["news_count", "neg_share", "net_sentiment", "pos_share",
               "frshare_safety", "frshare_newmodel", "frshare_subsidy", "frshare_charging"]
KG_BASE = ["cent_safety", "salience_newmodel", "safety_neg_intensity",
           "newmodel_pos_intensity", "frame_entropy", "neg_event_propagation", "company_hhi"]
MAXLAG = 3


def best_granger(df2, cause, effect):
    """cause -> effect 최소 p-value와 시차. grangercausalitytests 입력은 [effect, cause] 순."""
    sub = df2[[effect, cause]].diff().dropna()
    if len(sub) < 20 or sub[cause].std() == 0 or sub[effect].std() == 0:
        return np.nan, np.nan
    try:
        gc = grangercausalitytests(sub[[effect, cause]], maxlag=MAXLAG, verbose=False)
        pvals = {k: gc[k][0]["ssr_ftest"][1] for k in gc}
        best = min(pvals, key=pvals.get)
        return best, pvals[best]
    except Exception:
        return np.nan, np.nan


def run_bidirectional(df):
    rows = []
    for f in SIMPLE_BASE + KG_BASE:
        if f not in df.columns:
            continue
        fwd_lag, fwd_p = best_granger(df, cause=f, effect="ev_net_add")
        rev_lag, rev_p = best_granger(df, cause="ev_net_add", effect=f)
        rows.append({
            "feature": f,
            "type": "KG" if f in KG_BASE else "simple",
            "fwd_best_lag": fwd_lag, "fwd_p": fwd_p,
            "rev_best_lag": rev_lag, "rev_p": rev_p,
        })
    res = pd.DataFrame(rows)
    # 순방향 p에 대해 다중비교 보정
    valid = res["fwd_p"].notna()
    res["fwd_p_bonf"] = np.nan
    res["fwd_p_bh"] = np.nan
    if valid.sum() > 0:
        pv = res.loc[valid, "fwd_p"].values
        res.loc[valid, "fwd_p_bonf"] = multipletests(pv, method="bonferroni")[1]
        res.loc[valid, "fwd_p_bh"] = multipletests(pv, method="fdr_bh")[1]
    a = kc.ALPHA
    res["fwd_sig_raw"] = res["fwd_p"] < a
    res["fwd_sig_bh"] = res["fwd_p_bh"] < a
    res["direction"] = np.where(
        (res["fwd_p"] < a) & (res["rev_p"] >= a), "뉴스선행",
        np.where((res["fwd_p"] >= a) & (res["rev_p"] < a), "등록선행",
                 np.where((res["fwd_p"] < a) & (res["rev_p"] < a), "양방향", "무관")))
    res = res.sort_values("fwd_p")
    for c in ["fwd_p", "rev_p", "fwd_p_bonf", "fwd_p_bh"]:
        res[c] = res[c].round(4)
    res.to_csv(os.path.join(kc.TABLE_DIR, "tbl_granger_corrected.csv"),
               index=False, encoding="utf-8-sig")
    return res


def run_var(df):
    vars_ = ["ev_net_add", "news_count", "neg_event_propagation", "salience_newmodel"]
    d = df[["ym"] + vars_].dropna().copy()
    data = d[vars_].astype(float)
    # 1차 차분으로 정상성 확보
    dd = data.diff().dropna()
    adf = {c: round(adfuller(dd[c])[1], 4) for c in vars_}

    model = VAR(dd)
    sel = model.select_order(maxlags=4)
    p = sel.aic if isinstance(sel.aic, int) and sel.aic > 0 else 2
    p = max(1, min(p, 3))
    res = model.fit(p)

    # IRF
    irf = res.irf(12)
    fig = irf.plot(orth=True)
    fig.set_size_inches(13, 11)
    kc.set_korean_font()
    fig.suptitle(f"직교 충격반응함수 (VAR p={p})", fontsize=14)
    irf_path = os.path.join(kc.FIG_DIR, "var_irf.png")
    fig.savefig(irf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # FEVD (12기)
    fevd = res.fevd(12)
    fevd_rows = []
    for i, target in enumerate(vars_):
        dec = fevd.decomp[i][-1]  # 12기 시점 분해
        for j, src in enumerate(vars_):
            fevd_rows.append({"target": target, "shock_from": src,
                              "contrib_h12": round(float(dec[j]), 4)})
    fevd_df = pd.DataFrame(fevd_rows)
    fevd_df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_var_fevd.csv"),
                   index=False, encoding="utf-8-sig")
    return p, adf, fevd_df, irf_path


def main():
    kc.ensure_dirs()
    df = pd.read_csv(kc.NATIONAL_CSV).sort_values("ym").reset_index(drop=True)

    print("[20] 양방향 Granger + 다중비교 보정...")
    g = run_bidirectional(df)

    print("[20] VAR/IRF/FEVD...")
    p, adf, fevd_df, irf_path = run_var(df)

    lines = ["# 양방향 Granger / 다중비교 보정 / VAR 요약\n"]
    lines.append(f"## 1. 양방향 Granger (차분, maxlag=3, 판정 α={kc.ALPHA})\n")
    lines.append(g.to_markdown(index=False))
    sig_bh = g[g["fwd_sig_bh"]]
    lines.append("")
    lines.append(f"- 순방향 raw p<{kc.ALPHA}: {int(g['fwd_sig_raw'].sum())}개, "
                 f"BH(FDR) 보정 후 p<{kc.ALPHA}: {int(g['fwd_sig_bh'].sum())}개")
    lines.append(f"- `direction` 열은 정방향·역방향에 동일한 α={kc.ALPHA} 를 적용한다. "
                 "정방향에 느슨한 기준을 쓰면 선행성 주장이 관대해지고, 역방향에 느슨한 "
                 "기준을 쓰면 역인과 판정이 관대해지므로 양쪽을 대칭으로 맞춘다.")
    if len(sig_bh):
        lines.append(f"- BH 보정 후에도 유의: {', '.join(sig_bh['feature'])}")
    else:
        lines.append("- BH 보정 후 유의 지표 없음 → 소표본에서 선행성 증거는 약함(정직하게 명시).")
    lines.append("")
    lines.append(f"## 2. VAR (p={p}), 차분 후 ADF p-value: {adf}\n")
    lines.append("### 분산분해(FEVD, h=12)\n")
    lines.append(fevd_df.to_markdown(index=False))
    lines.append("")
    lines.append(f"- IRF 그림: {os.path.relpath(irf_path, kc.BASE_DIR)}")

    path = os.path.join(kc.TABLE_DIR, "granger_var_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[20] 저장: {path}")


if __name__ == "__main__":
    main()
