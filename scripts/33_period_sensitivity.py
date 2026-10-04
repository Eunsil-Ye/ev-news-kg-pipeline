# -*- coding: utf-8 -*-
"""
[33] 분석기간 민감도: 주 스펙(2018-01~, 94개월) vs 제한(2019-01~, 82개월)

주 스펙은 `kg_common.ANALYSIS_START = "2018-01"` 이다. 초기 확산기(2018년)를 포함한 것이
결과를 좌우하는지 확인하기 위해, 2019-01 로 제한한 변이를 만들어 나란히 비교한다.

14번(태깅)은 기간으로 필터하지 않으므로 `news_tagged.csv` 는 재생성하지 않는다.
따라서 사람 라벨에 기반한 지표(개체/프레임 정밀도, 감성 kappa, BERT, 관련성 분류기)는
기간과 무관하며 이 비교의 대상이 아니다.

원본 산출물은 읽기만 하고 수정하지 않는다. 변이 산출물은 모두 `_p2019` 접미사로 분리한다.

출력:
  data/processed/{kg_monthly_features,news_monthly_simple,national_ev_news_kg,
                  panel_ev_news_kg}_p2019.csv
  outputs/kg/tables/tbl_period_granger.csv
  outputs/kg/tables/tbl_period_fevd.csv
  outputs/kg/tables/tbl_period_panel.csv
  outputs/kg/tables/tbl_period_breakpoint.csv
  outputs/kg/tables/period_sensitivity.md
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

m15 = import_module("15_build_temporal_kg")
m16 = import_module("16_kg_monthly_indicators")
m17 = import_module("17_merge_kg_into_panel")
m18 = import_module("18_models_with_kg")
m20 = import_module("20_granger_multi_correction")
m23 = import_module("23_event_breakpoint")

from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests

# ------------------------------------------------------------
# 변이 정의
# ------------------------------------------------------------
VARIANT_START = "2019-01"
SUFFIX = "_p2019"

V_KG_DIR = os.path.join(kc.KG_DIR, f"variant{SUFFIX}")
V_MONTHLY_DIR = os.path.join(V_KG_DIR, "monthly")
V_FEATURES = os.path.join(kc.DATA_PROC, f"kg_monthly_features{SUFFIX}.csv")
V_NEWS_SIMPLE = os.path.join(kc.DATA_PROC, f"news_monthly_simple{SUFFIX}.csv")
V_NATIONAL = os.path.join(kc.DATA_PROC, f"national_ev_news_kg{SUFFIX}.csv")
V_PANEL = os.path.join(kc.DATA_PROC, f"panel_ev_news_kg{SUFFIX}.csv")

GRANGER_FEATS = m20.SIMPLE_BASE + m20.KG_BASE
VAR_VARS = ["ev_net_add", "news_count", "neg_event_propagation", "salience_newmodel"]
BREAK_SERIES = ["safety_neg_intensity", "cent_safety", "neg_event_propagation",
                "salience_newmodel"]
INTER_MAIN = ["safety_neg_intensity_lag1", "salience_newmodel_lag1", "cent_safety_lag1"]
INTER_NAMES = ["safetyNeg_x_chgPerEv", "newmodel_x_evshare", "centSafety_x_chgPerEv"]
MAIN_LAGS = m18.lag_names(["neg_share", "frshare_newmodel", "frshare_safety"])
JOINT_GROUPS = ["neg_share", "frshare_safety", "frshare_newmodel"]


# ============================================================
# 1. 제한 기간 변이 생성 (15 -> 16 -> 17 재실행)
# ============================================================
def build_variant():
    """kc 모듈 상수를 일시적으로 제한 기간/변이 경로로 바꿔 15~17번을 재실행한다.

    29번이 정제판을 만들 때 쓰는 방식과 동일하다. finally 로 반드시 되돌려
    이후 비교 단계가 원본 경로를 읽도록 보장한다.
    """
    saved = (kc.ANALYSIS_START, kc.KG_DIR, kc.KG_MONTHLY_DIR,
             kc.KG_MONTHLY_FEATURES_CSV, kc.NEWS_MONTHLY_SIMPLE_CSV,
             kc.NATIONAL_CSV, kc.PANEL_CSV)
    try:
        kc.ANALYSIS_START = VARIANT_START
        kc.KG_DIR = V_KG_DIR
        kc.KG_MONTHLY_DIR = V_MONTHLY_DIR
        kc.KG_MONTHLY_FEATURES_CSV = V_FEATURES
        kc.NEWS_MONTHLY_SIMPLE_CSV = V_NEWS_SIMPLE
        kc.NATIONAL_CSV = V_NATIONAL
        kc.PANEL_CSV = V_PANEL
        os.makedirs(V_MONTHLY_DIR, exist_ok=True)

        print(f"[33] 변이 생성 시작 (ANALYSIS_START={VARIANT_START})")
        m15.main()
        m16.main()
        m17.main()
    finally:
        (kc.ANALYSIS_START, kc.KG_DIR, kc.KG_MONTHLY_DIR,
         kc.KG_MONTHLY_FEATURES_CSV, kc.NEWS_MONTHLY_SIMPLE_CSV,
         kc.NATIONAL_CSV, kc.PANEL_CSV) = saved
    print(f"[33] 변이 생성 완료 -> {V_NATIONAL}")


# ============================================================
# 2. R1 양방향 Granger + BH 비교
# ============================================================
def granger_block(df, tag):
    rows = []
    for f in GRANGER_FEATS:
        if f not in df.columns:
            continue
        fwd_lag, fwd_p = m20.best_granger(df, cause=f, effect="ev_net_add")
        rev_lag, rev_p = m20.best_granger(df, cause="ev_net_add", effect=f)
        rows.append({"feature": f,
                     "type": "KG" if f in m20.KG_BASE else "simple",
                     f"{tag}_fwd_lag": fwd_lag, f"{tag}_fwd_p": fwd_p,
                     f"{tag}_rev_p": rev_p})
    out = pd.DataFrame(rows)
    ok = out[f"{tag}_fwd_p"].notna()
    out[f"{tag}_fwd_bh"] = np.nan
    if ok.sum():
        out.loc[ok, f"{tag}_fwd_bh"] = multipletests(
            out.loc[ok, f"{tag}_fwd_p"].values, method="fdr_bh")[1]
    a = kc.ALPHA
    # 선행성 판정: 정방향 유의 + 역방향 비유의.
    out[f"{tag}_pass"] = (out[f"{tag}_fwd_p"] < a) & (out[f"{tag}_rev_p"] >= a)
    return out


def granger_compare(base, restr):
    b = granger_block(base, "base")
    e = granger_block(restr, "restr")
    df = b.merge(e.drop(columns=["type"]), on="feature", how="outer")
    df["conclusion"] = np.where(
        df["base_pass"] & df["restr_pass"], "양쪽 충족(유지)",
        np.where(df["base_pass"] & ~df["restr_pass"], "제한 후 소멸",
                 np.where(~df["base_pass"] & df["restr_pass"], "제한 후 등장", "양쪽 미충족")))
    df["p_delta"] = (df["restr_fwd_p"] - df["base_fwd_p"]).round(4)
    for c in df.columns:
        if c.endswith(("_fwd_p", "_rev_p", "_fwd_bh")):
            df[c] = df[c].astype(float).round(4)
    return df.sort_values("base_fwd_p").reset_index(drop=True)


# ============================================================
# 3. R2 VAR / FEVD 비교
# ============================================================
def fevd_block(df, tag):
    d = df[["ym"] + VAR_VARS].dropna().copy()
    dd = d[VAR_VARS].astype(float).diff().dropna()
    model = VAR(dd)
    sel = model.select_order(maxlags=4)
    p = sel.aic if isinstance(sel.aic, int) and sel.aic > 0 else 2
    p = max(1, min(p, 3))
    res = model.fit(p)
    dec = res.fevd(12).decomp
    rows = []
    for i, target in enumerate(VAR_VARS):
        for j, src in enumerate(VAR_VARS):
            rows.append({"target": target, "shock_from": src,
                         f"{tag}_contrib_h12": round(float(dec[i][-1][j]), 4)})
    return pd.DataFrame(rows), p, int(len(dd))


def fevd_compare(base, restr):
    fb, pb, nb = fevd_block(base, "base")
    fe, pe, ne = fevd_block(restr, "restr")
    df = fb.merge(fe, on=["target", "shock_from"], how="outer")
    df["delta"] = (df["restr_contrib_h12"] - df["base_contrib_h12"]).round(4)
    return df, {"base_lag": pb, "base_n": nb, "restr_lag": pe, "restr_n": ne}


# ============================================================
# 4. R3 패널 비교 (P1 주효과 월클러스터 + P2 상호작용)
# ============================================================
def panel_fit(panel_path, label):
    df = pd.read_csv(panel_path).sort_values(["region", "ym"]).reset_index(drop=True)
    df = m18.add_target_lags(df, "ev_net_add", lags=(1, 2, 3), group="region")

    moderators = ["charger_per_ev", "ev_share"]
    ar_cols = ["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3"]
    need = ["ev_net_add", "region"] + ar_cols + INTER_MAIN + moderators + MAIN_LAGS
    d = df.dropna(subset=[c for c in need if c in df.columns]).reset_index(drop=True)

    cont = ar_cols + INTER_MAIN + moderators + MAIN_LAGS
    cont = [c for c in cont if c in d.columns]
    dz = d.copy()
    for c in cont:
        s = dz[c].std()
        dz[c] = (dz[c] - dz[c].mean()) / s if s > 0 else 0.0

    region_dum = pd.get_dummies(dz["region"], prefix="reg", drop_first=True).astype(float)
    ar = dz[ar_cols]
    main = dz[MAIN_LAGS + INTER_MAIN]
    y = d["ev_net_add"].values
    # 뉴스/KG 지표는 전국 공통값이므로 월 클러스터가 주 스펙이다(Moulton 문제).
    ym_groups = pd.factorize(d["ym"])[0]

    inter = pd.DataFrame(index=dz.index)
    inter["safetyNeg_x_chgPerEv"] = dz["safety_neg_intensity_lag1"] * dz["charger_per_ev"]
    inter["newmodel_x_evshare"] = dz["salience_newmodel_lag1"] * dz["ev_share"]
    inter["centSafety_x_chgPerEv"] = dz["cent_safety_lag1"] * dz["charger_per_ev"]

    X1 = sm.add_constant(pd.concat([ar, main, region_dum], axis=1))
    m1c = sm.OLS(y, X1.values).fit(cov_type="cluster", cov_kwds={"groups": ym_groups})
    cols1 = list(X1.columns)

    X2 = sm.add_constant(pd.concat([ar, main, inter, region_dum], axis=1))
    m2 = sm.OLS(y, X2.values).fit(cov_type="HC1")
    cols2 = list(X2.columns)

    info = {"spec": label, "n": int(m1c.nobs), "months": int(d["ym"].nunique()),
            "ym_min": d["ym"].min(), "ym_max": d["ym"].max(),
            "r2_P1": round(m1c.rsquared, 4), "r2_P2": round(m2.rsquared, 4)}

    main_rows = []
    for nm in MAIN_LAGS + INTER_MAIN:
        if nm in cols1:
            i = cols1.index(nm)
            main_rows.append({"spec": label, "term": nm,
                              "coef": round(m1c.params[i], 2),
                              "p_cluster": round(m1c.pvalues[i], 4)})
    for nm in INTER_NAMES:
        if nm in cols2:
            i = cols2.index(nm)
            main_rows.append({"spec": label, "term": nm,
                              "coef": round(m2.params[i], 2),
                              "p_hc1": round(m2.pvalues[i], 4)})

    # 시차군 결합검정(월 클러스터) — H1 감성 판정의 근거
    joint_rows = []
    for base in JOINT_GROUPS:
        names = [f"{base}_lag{k}" for k in (1, 2, 3)]
        idx = [cols1.index(n) for n in names if n in cols1]
        if len(idx) != 3:
            continue
        R = np.zeros((3, len(cols1)))
        for r, i in enumerate(idx):
            R[r, i] = 1.0
        ft = m1c.f_test(R)
        Rc = np.zeros((1, len(cols1)))
        for i in idx:
            Rc[0, i] = 1.0
        tt = m1c.t_test(Rc)
        joint_rows.append({"spec": label, "var_group": base,
                           "joint_F": round(float(np.squeeze(ft.fvalue)), 3),
                           "joint_p": round(float(np.squeeze(ft.pvalue)), 4),
                           "cum_coef": round(float(np.squeeze(tt.effect)), 2),
                           "cum_p": round(float(np.squeeze(tt.pvalue)), 4)})
    return info, main_rows, joint_rows


# ============================================================
# 5. R4 구조 전환점 비교
# ============================================================
def breakpoint_compare():
    base = pd.read_csv(kc.KG_MONTHLY_FEATURES_CSV).sort_values("ym").reset_index(drop=True)
    restr = pd.read_csv(V_FEATURES).sort_values("ym").reset_index(drop=True)
    rows = []
    for s in BREAK_SERIES:
        rec = {"series": s}
        for tag, df in [("base", base), ("restr", restr)]:
            if s not in df.columns:
                rec[f"{tag}_break_ym"] = None
                rec[f"{tag}_supF"] = np.nan
                continue
            y = df[s].astype(float).values
            bi, supf = m23.supf_meanshift(y)
            rec[f"{tag}_break_ym"] = df["ym"].tolist()[bi] if bi is not None else None
            rec[f"{tag}_supF"] = round(supf, 2)
        rec["same"] = rec.get("base_break_ym") == rec.get("restr_break_ym")
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================
# 6. 리포트
# ============================================================
def main():
    kc.ensure_dirs()
    if not os.path.exists(kc.NEWS_TAGGED_CSV):
        print("[33] news_tagged.csv 가 없다. 먼저 14번을 실행하라.")
        return

    build_variant()

    base = pd.read_csv(kc.NATIONAL_CSV).sort_values("ym").reset_index(drop=True)
    restr = pd.read_csv(V_NATIONAL).sort_values("ym").reset_index(drop=True)

    print("[33] R1 양방향 Granger 비교...")
    g = granger_compare(base, restr)
    print("[33] R2 VAR/FEVD 비교...")
    fv, vinfo = fevd_compare(base, restr)
    print("[33] R3 패널 비교...")
    i_b, mr_b, jr_b = panel_fit(kc.PANEL_CSV, f"주스펙({kc.ANALYSIS_START}~)")
    i_e, mr_e, jr_e = panel_fit(V_PANEL, f"제한({VARIANT_START}~)")
    pinfo = pd.DataFrame([i_b, i_e])
    pmain = pd.DataFrame(mr_b + mr_e)
    pjoint = pd.DataFrame(jr_b + jr_e)
    print("[33] R4 구조 전환점 비교...")
    b = breakpoint_compare()

    g.to_csv(os.path.join(kc.TABLE_DIR, "tbl_period_granger.csv"),
             index=False, encoding="utf-8-sig")
    fv.to_csv(os.path.join(kc.TABLE_DIR, "tbl_period_fevd.csv"),
              index=False, encoding="utf-8-sig")
    pmain.to_csv(os.path.join(kc.TABLE_DIR, "tbl_period_panel.csv"),
                 index=False, encoding="utf-8-sig")
    b.to_csv(os.path.join(kc.TABLE_DIR, "tbl_period_breakpoint.csv"),
             index=False, encoding="utf-8-sig")

    L = [f"# 분석기간 민감도: 주 스펙({kc.ANALYSIS_START}~) vs 제한({VARIANT_START}~)\n"]
    L.append(f"판정 기준은 α={kc.ALPHA}(정방향 유의 + 역방향 비유의)다.\n")
    L.append("주 스펙은 사전에 정한 2018-01 시작을 따른다. 여기서는 "
             "초기 확산기(2018년)를 포함한 것이 결론을 좌우하는지 확인하기 위해 "
             "2019-01 로 제한한 변이와 나란히 비교한다.\n")
    L.append(f"- 주 스펙 전국 시계열: {len(base)}개월 ({base.ym.min()}~{base.ym.max()})")
    L.append(f"- 제한 전국 시계열: {len(restr)}개월 ({restr.ym.min()}~{restr.ym.max()})")
    L.append("- 태깅 자체(14번)는 기간 필터를 쓰지 않으므로 사람 라벨 기반 지표"
             "(개체/프레임 정밀도, 감성 kappa, BERT, 관련성 분류기)는 두 스펙에서 동일하다.\n")

    L.append("## R1. 양방향 Granger 선행성\n")
    L.append(g.to_markdown(index=False))
    L.append("")
    for key in ["양쪽 충족(유지)", "제한 후 소멸", "제한 후 등장"]:
        items = g[g["conclusion"] == key]["feature"].tolist()
        L.append(f"- {key}: {', '.join(items) if items else '없음'}")
    n_bh_b = int((g["base_fwd_bh"] < kc.ALPHA).sum())
    n_bh_e = int((g["restr_fwd_bh"] < kc.ALPHA).sum())
    L.append(f"- BH(FDR) 보정 후 유의 지표 수: 주 스펙 {n_bh_b}개 / 제한 {n_bh_e}개")
    L.append("")

    L.append(f"## R2. VAR 분산분해 (h=12, 차분)\n")
    L.append(f"- 선택 시차: 주 스펙 p={vinfo['base_lag']}(n={vinfo['base_n']}), "
             f"제한 p={vinfo['restr_lag']}(n={vinfo['restr_n']})\n")
    ev = fv[fv["target"] == "ev_net_add"]
    L.append(ev.to_markdown(index=False))
    L.append("")

    L.append("## R3. 지역 패널\n")
    L.append(pinfo.to_markdown(index=False))
    L.append("")
    L.append("### P1 주효과(월 클러스터 SE) / P2 상호작용(HC1)\n")
    L.append(pmain.to_markdown(index=False))
    L.append("")
    L.append("### 시차군 결합검정 + 3개월 누적효과 (월 클러스터)\n")
    L.append(pjoint.to_markdown(index=False))
    L.append("")
    L.append("- 충전기 자료가 2020-12 부터 시작하므로 `charger_per_ev` 결측으로 인해 "
             "패널 유효표본은 두 스펙에서 사실상 동일하다. 즉 기간 선택의 영향은 "
             "전국 시계열(R1·R2)에 한정되고, 지역 패널 결론은 기간과 무관하다.")
    L.append("")

    L.append("## R4. 구조변화 전환점\n")
    L.append(b.to_markdown(index=False))
    L.append("")
    same = b[b["same"]]["series"].tolist()
    diff = b[~b["same"]]["series"].tolist()
    L.append(f"- 전환점 동일: {', '.join(same) if same else '없음'}")
    L.append(f"- 전환점 변동: {', '.join(diff) if diff else '없음'}")

    path = os.path.join(kc.TABLE_DIR, "period_sensitivity.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # 콘솔에는 핵심만 (cp949 콘솔에서 깨지지 않도록 표는 파일로만 저장)
    print()
    print(f"[33] 주 스펙 {len(base)}개월 vs 제한 {len(restr)}개월")
    print("[33] --- Granger 핵심 3종 ---")
    for f_ in ["news_count", "neg_event_propagation", "salience_newmodel"]:
        r = g[g["feature"] == f_]
        if len(r):
            r = r.iloc[0]
            print(f"  {f_:24s} base fwd={r['base_fwd_p']} rev={r['base_rev_p']} "
                  f"| restr fwd={r['restr_fwd_p']} rev={r['restr_rev_p']} -> {r['conclusion']}")
    print(f"[33] BH 보정 후 유의: base {n_bh_b} / restr {n_bh_e}")
    print("[33] --- FEVD(ev_net_add) ---")
    for _, r in ev.iterrows():
        print(f"  from {r['shock_from']:24s} base={r['base_contrib_h12']} "
              f"restr={r['restr_contrib_h12']} (delta {r['delta']})")
    print("[33] --- 패널 유효표본 ---")
    for _, r in pinfo.iterrows():
        print(f"  {r['spec']}: n={r['n']} months={r['months']} "
              f"({r['ym_min']}~{r['ym_max']})")
    print(f"[33] 저장: {path}")


if __name__ == "__main__":
    main()
