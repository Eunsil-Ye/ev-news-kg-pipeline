# -*- coding: utf-8 -*-
"""
[37] LLM 감성 라벨로 H1 재검정

36번의 gpt-6-luna 라벨(data/processed/news_llm_labels.csv)로 월별 감성 지표를 다시 만들고,
규칙 기반 감성으로 수행한 H1 검정 두 가지를 반복한다. 기존 결과 파일은 덮어쓰지 않는다.
  1) 전국 시계열 양방향 Granger (20번과 같은 함수: 차분, maxlag=3, α=kc.ALPHA)
  2) 지역 패널 P1 (18번과 같은 모형·월 클러스터 SE), neg_share 시차 3개만 LLM 지표로 교체

주 검정(사전 고정, BH 보정 단위): llm_neg_share, llm_net_sentiment,
                                 llmrel_neg_share, llmrel_net_sentiment
  - llm_*    : 키워드 코퍼스 전체 기사 기준
  - llmrel_* : LLM 이 '관련'으로 판정한 기사만 기준
보조: llm_pos_share, llmrel_pos_share, llmrel_news_count, rel_share (보정 없이 참고)

출력:
  - data/processed/news_monthly_llm.csv
  - outputs/kg/tables/tbl_h1_llm_granger.csv, tbl_h1_llm_panel.csv, tbl_h1_llm_panel_yearfe.csv
  - outputs/kg/tables/h1_llm_retest_summary.md
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

warnings.filterwarnings("ignore")

LLM_LABELS_CSV = os.path.join(kc.DATA_PROC, "news_llm_labels.csv")
MONTHLY_CSV = os.path.join(kc.DATA_PROC, "news_monthly_llm.csv")
MAIN = ["llm_neg_share", "llm_net_sentiment", "llmrel_neg_share", "llmrel_net_sentiment"]
AUX = ["llm_pos_share", "llmrel_pos_share", "llmrel_news_count", "rel_share"]
BASE = ["neg_share", "net_sentiment", "pos_share", "news_count"]
PANEL_SUBS = ["neg_share", "llm_neg_share", "llmrel_neg_share"]
MAX_MISSING = 0.01


def load_script(fname: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    spec = importlib.util.spec_from_file_location(fname[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_monthly() -> pd.DataFrame:
    d = pd.read_csv(LLM_LABELS_CSV, encoding="utf-8-sig")
    d = d[(d["ym"] >= kc.ANALYSIS_START) & (d["ym"] <= kc.ANALYSIS_END)]
    miss = d["llm_label"].isna().mean()
    if miss > MAX_MISSING:
        raise SystemExit(f"분석 기간 LLM 라벨 결측 {miss:.1%} (허용 {MAX_MISSING:.0%}). 36번 실행을 마친다.")
    d = d[d["llm_label"].notna()]
    d["pos"] = (d["llm_label"] == "긍정").astype(int)
    d["neg"] = (d["llm_label"] == "부정").astype(int)
    d["rel"] = (d["llm_relevance"] == "관련").astype(int)
    rows = []
    for ym, g in d.groupby("ym"):
        r = g[g["rel"] == 1]
        rows.append({
            "ym": ym,
            "llm_n": len(g),
            "llm_pos_share": g["pos"].mean(),
            "llm_neg_share": g["neg"].mean(),
            "llm_net_sentiment": g["pos"].mean() - g["neg"].mean(),
            "rel_share": g["rel"].mean(),
            "llmrel_news_count": len(r),
            "llmrel_pos_share": r["pos"].mean() if len(r) else np.nan,
            "llmrel_neg_share": r["neg"].mean() if len(r) else np.nan,
            "llmrel_net_sentiment": (r["pos"].mean() - r["neg"].mean()) if len(r) else np.nan,
        })
    m = pd.DataFrame(rows).sort_values("ym").reset_index(drop=True)
    m.to_csv(MONTHLY_CSV, index=False, encoding="utf-8-sig")
    return m, miss


def granger(nat: pd.DataFrame) -> pd.DataFrame:
    g20 = load_script("20_granger_multi_correction.py")
    rows = []
    for f in BASE + MAIN + AUX:
        fl, fp = g20.best_granger(nat, cause=f, effect="ev_net_add")
        rl, rp = g20.best_granger(nat, cause="ev_net_add", effect=f)
        rows.append({"feature": f,
                     "family": "기존(규칙)" if f in BASE else ("주 검정" if f in MAIN else "보조"),
                     "fwd_best_lag": fl, "fwd_p": fp, "rev_best_lag": rl, "rev_p": rp})
    res = pd.DataFrame(rows)
    main = res["family"] == "주 검정"
    res["fwd_p_bh"] = np.nan
    res.loc[main, "fwd_p_bh"] = multipletests(res.loc[main, "fwd_p"], method="fdr_bh")[1]
    a = kc.ALPHA
    res["direction"] = np.where(
        (res.fwd_p < a) & (res.rev_p >= a), "뉴스선행",
        np.where((res.fwd_p >= a) & (res.rev_p < a), "등록선행",
                 np.where((res.fwd_p < a) & (res.rev_p < a), "양방향", "무관")))
    for c in ["fwd_p", "rev_p", "fwd_p_bh"]:
        res[c] = res[c].round(4)
    return res


def panel(monthly: pd.DataFrame) -> pd.DataFrame:
    """18번 run_panel 을 그대로 호출하되 neg_share 시차만 교체. 결과 파일은 임시 폴더로 보낸다."""
    m18 = load_script("18_models_with_kg.py")
    base_panel = pd.read_csv(kc.PANEL_CSV)
    orig_panel_csv, orig_table_dir = kc.PANEL_CSV, kc.TABLE_DIR
    tmp = tempfile.mkdtemp(prefix="h1_llm_")
    rows = []
    try:
        kc.TABLE_DIR = tmp
        for src in PANEL_SUBS:
            p = base_panel.copy()
            if src != "neg_share":
                s = monthly.set_index("ym")[src].sort_index()
                for k in (1, 2, 3):
                    p[f"neg_share_lag{k}"] = p["ym"].map(s.shift(k))
            path = os.path.join(tmp, f"panel_{src}.csv")
            p.to_csv(path, index=False, encoding="utf-8-sig")
            kc.PANEL_CSV = path
            _, _, main_df, joint_df = m18.run_panel()
            j = joint_df[joint_df.var_group == "neg_share"].iloc[0]
            coefs = main_df.set_index("term")
            rows.append({
                "neg_share_source": src,
                **{f"lag{k}_coef": coefs.loc[f"neg_share_lag{k}", "coef"] for k in (1, 2, 3)},
                **{f"lag{k}_p_cluster": coefs.loc[f"neg_share_lag{k}", "p_cluster_ym"]
                   for k in (1, 2, 3)},
                "joint_F_cluster": j.joint_F_cluster, "joint_p_cluster": j.joint_p_cluster,
                "cum_coef": j.cum_coef, "cum_p_cluster": j.cum_p_cluster,
                "cum_ci_low_cluster": j.cum_ci_low_cluster,
                "cum_ci_high_cluster": j.cum_ci_high_cluster,
            })
    finally:
        kc.PANEL_CSV, kc.TABLE_DIR = orig_panel_csv, orig_table_dir
    return pd.DataFrame(rows)


def panel_year_fe(monthly: pd.DataFrame) -> pd.DataFrame:
    """P1 과 같은 설계에 연도 더미를 추가. LLM 부정 비중은 시간 추세가 강해(r≈0.7) 수준 회귀가
    공통 추세를 잡을 수 있으므로, 연도 내 월 변동만으로도 결과가 유지되는지 본다."""
    import statsmodels.api as sm
    m18 = load_script("18_models_with_kg.py")
    base = pd.read_csv(kc.PANEL_CSV).sort_values(["region", "ym"]).reset_index(drop=True)
    base = m18.add_target_lags(base, "ev_net_add", lags=(1, 2, 3), group="region")
    ar = [f"ev_net_add_lag{k}" for k in (1, 2, 3)]
    negs = [f"neg_share_lag{k}" for k in (1, 2, 3)]
    frs = [f"{b}_lag{k}" for b in ["frshare_newmodel", "frshare_safety"] for k in (1, 2, 3)]
    inter = ["safety_neg_intensity_lag1", "salience_newmodel_lag1", "cent_safety_lag1"]
    rows = []
    for src in PANEL_SUBS:
        for year_fe in (False, True):
            d = base.copy()
            if src != "neg_share":
                s = monthly.set_index("ym")[src].sort_index()
                for k in (1, 2, 3):
                    d[f"neg_share_lag{k}"] = d["ym"].map(s.shift(k))
            d = d.dropna(subset=["ev_net_add", *ar, *negs, *frs, *inter,
                                 "charger_per_ev", "ev_share"]).reset_index(drop=True)
            X = d[ar + negs + frs + inter]
            X = (X - X.mean()) / X.std()
            parts = [X, pd.get_dummies(d.region, prefix="r", drop_first=True).astype(float)]
            if year_fe:
                parts.append(pd.get_dummies(d.ym.str[:4], prefix="y", drop_first=True).astype(float))
            X = sm.add_constant(pd.concat(parts, axis=1))
            m = sm.OLS(d.ev_net_add.values, X).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ym)[0]})
            R = np.zeros((3, X.shape[1]))
            for i, n in enumerate(negs):
                R[i, list(X.columns).index(n)] = 1
            f, t = m.f_test(R), m.t_test(R.sum(0, keepdims=True))
            rows.append({"neg_share_source": src, "year_fe": year_fe, "n": len(d),
                         "joint_p_cluster": round(float(np.squeeze(f.pvalue)), 4),
                         "cum_coef": round(float(np.squeeze(t.effect)), 2),
                         "cum_p_cluster": round(float(np.squeeze(t.pvalue)), 4)})
    return pd.DataFrame(rows)


def main():
    kc.ensure_dirs()
    monthly, miss = build_monthly()
    nat = pd.read_csv(kc.NATIONAL_CSV).sort_values("ym").reset_index(drop=True)
    nat = nat.merge(monthly, on="ym", how="left")

    corr = {}
    for a, b in [("neg_share", "llm_neg_share"), ("neg_share", "llmrel_neg_share"),
                 ("net_sentiment", "llm_net_sentiment"),
                 ("net_sentiment", "llmrel_net_sentiment"),
                 ("news_count", "llmrel_news_count")]:
        corr[f"{a} ~ {b}"] = (round(nat[a].corr(nat[b]), 3),
                              round(nat[a].diff().corr(nat[b].diff()), 3))

    g = granger(nat)
    g.to_csv(os.path.join(kc.TABLE_DIR, "tbl_h1_llm_granger.csv"), index=False,
             encoding="utf-8-sig")
    p = panel(monthly)
    p.to_csv(os.path.join(kc.TABLE_DIR, "tbl_h1_llm_panel.csv"), index=False,
             encoding="utf-8-sig")
    pfe = panel_year_fe(monthly)
    pfe.to_csv(os.path.join(kc.TABLE_DIR, "tbl_h1_llm_panel_yearfe.csv"), index=False,
               encoding="utf-8-sig")
    t = np.arange(len(monthly))
    trend = {c: round(np.corrcoef(t, monthly[c])[0, 1], 3)
             for c in ["llm_neg_share", "llmrel_neg_share"]}

    rel = monthly.set_index("ym")["rel_share"]
    yearly_rel = monthly.assign(year=monthly.ym.str[:4]).groupby("year").apply(
        lambda x: x.llmrel_news_count.sum() / x.llm_n.sum()).round(3)
    lines = ["# H1 재검정: LLM 감성 라벨 (gpt-6-luna; 감성 v1, 관련성 v3 프롬프트)\n",
             f"- 분석 기간 {kc.ANALYSIS_START}~{kc.ANALYSIS_END}, 라벨 결측 {miss:.2%}",
             f"- LLM 관련 판정 비율: 월평균 {rel.mean():.3f} (최소 {rel.min():.3f}, 최대 {rel.max():.3f})",
             f"- 연도별 관련 비율: {yearly_rel.to_dict()}", "",
             "## 월별 지표 상관 (수준 / 1차 차분)\n",
             *[f"- {k}: {v[0]} / {v[1]}" for k, v in corr.items()], "",
             f"## 1. 전국 양방향 Granger (차분, maxlag=3, α={kc.ALPHA})\n",
             "BH 보정은 '주 검정' 4개 지표에만 적용한다.\n",
             g.to_markdown(index=False), "",
             "## 2. 지역 패널 P1 (18번 모형, neg_share 시차만 교체, 월 클러스터 SE)\n",
             p.to_markdown(index=False), "",
             "- 첫 행(neg_share)은 기존 규칙 기반 결과의 재현이다(18번 tbl_panel_models.csv 의 P1 과 같아야 한다).", "",
             "## 3. 추세 점검: 연도 더미 추가 (P1 과 같은 설계, 월 클러스터 SE)\n",
             f"- LLM 부정 비중과 시간의 상관: {trend}\n",
             pfe.to_markdown(index=False)]
    path = os.path.join(kc.TABLE_DIR, "h1_llm_retest_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[37] 저장: {path}")


if __name__ == "__main__":
    main()
