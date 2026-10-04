# -*- coding: utf-8 -*-
"""
[30] 원본 KG vs 정제판 KG — 핵심 결과 강건성 재검증

`29_kg_clean_variant.py` 가 만든 정제판 지표로 핵심 결과 세 가지를 재실행해
결론이 유지되는지 확인한다.

  R1 Granger 선행성      news_count / neg_event_propagation / salience_newmodel /
                         frame_entropy / cent_safety / safety_neg_intensity
                         (원본 p vs 정제판 p, BH 보정 포함)
  R2 구조변화 전환점      safety_neg_intensity, cent_safety, neg_event_propagation,
                         salience_newmodel 의 supF 전환 시점 유지 여부
  R3 패널 상호작용        상호작용 3항의 계수 부호와 월 클러스터 p

원본 산출물은 읽기만 하고 수정하지 않는다.

출력:
  outputs/kg/tables/tbl_kg_clean_comparison.csv
  outputs/kg/tables/kg_clean_robustness.md
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
m20 = import_module("20_granger_multi_correction")
m23 = import_module("23_event_breakpoint")

from statsmodels.stats.multitest import multipletests

CLEAN_FEATURES = os.path.join(kc.DATA_PROC, "kg_monthly_features_clean.csv")

GRANGER_FEATS = ["news_count", "neg_event_propagation", "salience_newmodel",
                 "frame_entropy", "cent_safety", "safety_neg_intensity"]
BREAK_SERIES = ["safety_neg_intensity", "cent_safety", "neg_event_propagation",
                "salience_newmodel"]
KG_COLS = ["safety_neg_intensity", "salience_newmodel", "cent_safety"]
INTER_NAMES = ["safetyNeg_x_chgPerEv", "newmodel_x_evshare", "centSafety_x_chgPerEv"]
MAIN_LAGS = m18.lag_names(["neg_share", "frshare_newmodel", "frshare_safety"])


def shift_ym(ym: str, k: int) -> str:
    """'2024-08' 을 k개월 뒤로 이동한 ym 문자열."""
    y, m = int(ym[:4]), int(ym[5:7])
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


# ============================================================
# R1. Granger 비교
# ============================================================
def build_national_clean():
    """전국 시계열의 KG 지표를 정제판으로 교체."""
    nat = pd.read_csv(kc.NATIONAL_CSV).sort_values("ym").reset_index(drop=True)
    clean = pd.read_csv(CLEAN_FEATURES)
    kg_cols = [c for c in clean.columns if c != "ym"]
    base = nat.drop(columns=[c for c in kg_cols if c in nat.columns])
    return base.merge(clean, on="ym", how="left").sort_values("ym").reset_index(drop=True)


def granger_compare(nat_orig, nat_clean):
    rows = []
    for f in GRANGER_FEATS:
        o_lag, o_p = m20.best_granger(nat_orig, cause=f, effect="ev_net_add") \
            if f in nat_orig.columns else (np.nan, np.nan)
        c_lag, c_p = m20.best_granger(nat_clean, cause=f, effect="ev_net_add") \
            if f in nat_clean.columns else (np.nan, np.nan)
        rows.append({"feature": f, "orig_best_lag": o_lag, "orig_p": o_p,
                     "clean_best_lag": c_lag, "clean_p": c_p})
    df = pd.DataFrame(rows)
    for side in ["orig", "clean"]:
        col = f"{side}_p"
        ok = df[col].notna()
        df[f"{side}_p_bh"] = np.nan
        if ok.sum() > 0:
            df.loc[ok, f"{side}_p_bh"] = multipletests(df.loc[ok, col].values,
                                                       method="fdr_bh")[1]
    df["sig_orig"] = df["orig_p"] < kc.ALPHA
    df["sig_clean"] = df["clean_p"] < kc.ALPHA
    df["conclusion"] = np.where(df["sig_orig"] & df["sig_clean"], "양쪽 유의(유지)",
                        np.where(df["sig_orig"] & ~df["sig_clean"], "정제 후 소멸",
                        np.where(~df["sig_orig"] & df["sig_clean"], "정제 후 등장", "양쪽 비유의")))
    # 유의성 판정과 별개로 p값이 규칙 변경에 얼마나 흔들리는지 기록한다.
    # 두 버전 모두 비유의여도 p값이 거의 같으면 '규칙에 견고하나 유의하지 않은' 지표이며,
    # 이 둘을 구분하지 않으면 강건성과 유의성이 뒤섞인 서술이 나온다.
    df["p_delta"] = (df["clean_p"] - df["orig_p"]).astype(float).round(4)
    for c in ["orig_p", "clean_p", "orig_p_bh", "clean_p_bh"]:
        df[c] = df[c].astype(float).round(4)
    return df


# ============================================================
# R2. 구조변화 전환점 비교
# ============================================================
def breakpoint_compare():
    orig = pd.read_csv(kc.KG_MONTHLY_FEATURES_CSV).sort_values("ym").reset_index(drop=True)
    clean = pd.read_csv(CLEAN_FEATURES).sort_values("ym").reset_index(drop=True)
    rows = []
    for s in BREAK_SERIES:
        rec = {"series": s}
        for name, df in [("orig", orig), ("clean", clean)]:
            if s not in df.columns:
                rec[f"{name}_break_ym"] = None
                rec[f"{name}_supF"] = np.nan
                rec[f"{name}_shift_pct"] = np.nan
                continue
            y = df[s].astype(float).values
            bi, supf = m23.supf_meanshift(y)
            ym_list = df["ym"].tolist()
            pre = float(np.mean(y[:bi])) if bi else np.nan
            post = float(np.mean(y[bi:])) if bi else np.nan
            rec[f"{name}_break_ym"] = ym_list[bi] if bi is not None else None
            rec[f"{name}_supF"] = round(supf, 2)
            rec[f"{name}_shift_pct"] = round((post / pre - 1) * 100, 1) if pre else np.nan
        rec["break_ym_same"] = rec.get("orig_break_ym") == rec.get("clean_break_ym")
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================
# R3. 패널 상호작용 비교
# ============================================================
def panel_fit(kg_source_df, label):
    """kg_source_df: ym + KG_COLS 를 담은 전국 시계열(원본 또는 정제판)."""
    panel = pd.read_csv(kc.PANEL_CSV).sort_values(["region", "ym"]).reset_index(drop=True)
    panel = m18.add_target_lags(panel, "ev_net_add", lags=(1, 2, 3), group="region")

    # 전국 KG 지표의 1개월 시차를 ym 기준으로 결합
    lagged = kg_source_df[["ym"] + KG_COLS].copy()
    lagged["ym"] = lagged["ym"].apply(lambda s: shift_ym(s, 1))
    lagged = lagged.rename(columns={c: f"{c}_lag1_new" for c in KG_COLS})

    d = panel.drop(columns=[f"{c}_lag1" for c in KG_COLS if f"{c}_lag1" in panel.columns])
    d = d.merge(lagged, on="ym", how="left")
    for c in KG_COLS:
        d[f"{c}_lag1"] = d[f"{c}_lag1_new"]

    inter_main = [f"{c}_lag1" for c in KG_COLS]
    moderators = ["charger_per_ev", "ev_share"]
    need = (["ev_net_add", "ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3", "region"]
            + inter_main + moderators + MAIN_LAGS)
    d = d.dropna(subset=[c for c in need if c in d.columns]).reset_index(drop=True)

    cont = (["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3"]
            + inter_main + moderators + MAIN_LAGS)
    cont = [c for c in cont if c in d.columns]
    dz = d.copy()
    for c in cont:
        s = dz[c].std()
        dz[c] = (dz[c] - dz[c].mean()) / s if s > 0 else 0.0

    region_dum = pd.get_dummies(dz["region"], prefix="reg", drop_first=True).astype(float)
    ar = dz[["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3"]]
    main = dz[MAIN_LAGS + inter_main]
    inter = pd.DataFrame(index=dz.index)
    inter["safetyNeg_x_chgPerEv"] = dz["safety_neg_intensity_lag1"] * dz["charger_per_ev"]
    inter["newmodel_x_evshare"] = dz["salience_newmodel_lag1"] * dz["ev_share"]
    inter["centSafety_x_chgPerEv"] = dz["cent_safety_lag1"] * dz["charger_per_ev"]

    X = sm.add_constant(pd.concat([ar, main, inter, region_dum], axis=1))
    m = sm.OLS(d["ev_net_add"].values, X.values).fit(cov_type="HC1")
    # 뉴스/KG 지표가 월 내 17개 지역에 공통이므로 18번과 같이 월 클러스터를 주 스펙으로 둔다.
    ym_groups = pd.factorize(d["ym"])[0]
    mc = sm.OLS(d["ev_net_add"].values, X.values).fit(
        cov_type="cluster", cov_kwds={"groups": ym_groups})
    cols = list(X.columns)

    row = {"spec": label, "n": int(m.nobs), "n_months": int(ym_groups.max() + 1),
           "r2": round(m.rsquared, 4), "adj_r2": round(m.rsquared_adj, 4)}
    for nm in INTER_NAMES:
        if nm in cols:
            i = cols.index(nm)
            row[f"{nm}_coef"] = round(m.params[i], 2)
            row[f"{nm}_p"] = round(m.pvalues[i], 4)
            row[f"{nm}_p_cluster"] = round(mc.pvalues[i], 4)
    return row


def panel_compare(nat_orig, nat_clean):
    rows = [panel_fit(nat_orig, "원본KG"), panel_fit(nat_clean, "정제판KG")]
    return pd.DataFrame(rows)


def main():
    kc.ensure_dirs()
    if not os.path.exists(CLEAN_FEATURES):
        print("[30] 정제판 지표가 없습니다. 먼저 29번을 실행하세요.")
        return

    print("[30] 전국 시계열 구성...")
    nat_orig = pd.read_csv(kc.NATIONAL_CSV).sort_values("ym").reset_index(drop=True)
    nat_clean = build_national_clean()

    print("[30] R1 Granger 비교...")
    g = granger_compare(nat_orig, nat_clean)

    print("[30] R2 구조변화 비교...")
    b = breakpoint_compare()

    print("[30] R3 패널 상호작용 비교...")
    p = panel_compare(pd.read_csv(kc.KG_MONTHLY_FEATURES_CSV),
                      pd.read_csv(CLEAN_FEATURES))

    # 통합 비교표 저장
    g.assign(block="R1_granger").to_csv(
        os.path.join(kc.TABLE_DIR, "tbl_kg_clean_comparison.csv"),
        index=False, encoding="utf-8-sig")
    b.to_csv(os.path.join(kc.TABLE_DIR, "tbl_kg_clean_breakpoint.csv"),
             index=False, encoding="utf-8-sig")
    p.to_csv(os.path.join(kc.TABLE_DIR, "tbl_kg_clean_panel.csv"),
             index=False, encoding="utf-8-sig")

    L = ["# 원본 KG vs 정제판 KG 강건성 재검증\n"]
    L.append("정제판은 동어반복 엣지 제외, 최장일치(이중계상 제거), 개체 단어경계 오탐 차단, "
             "충전난 별칭 확장, 산업 프레임 정밀화, 프레임 키워드 오탐 차단"
             "(가격의 '리스', 배터리의 '셀'을 정문맥 정규식으로 판정)을 적용한 변이다. "
             "감성 분류는 원본과 동일하다.\n")
    L.append("주의: `neg_event_propagation` 은 부정사건 개체가 닿은 distinct 프레임 수로 "
             "정의되므로, 프레임 태깅을 정밀화하면 기계적으로 감소한다. 이 지표의 변동은 "
             "정제 작업의 부작용이 아니라 지표 정의상 예상되는 민감성이다.\n")

    L.append(f"## R1. Granger 선행성 (차분, maxlag=3, 판정 α={kc.ALPHA})\n")
    L.append(g.to_markdown(index=False))
    L.append("")
    kept = g[g["conclusion"] == "양쪽 유의(유지)"]["feature"].tolist()
    lost = g[g["conclusion"] == "정제 후 소멸"]["feature"].tolist()
    gained = g[g["conclusion"] == "정제 후 등장"]["feature"].tolist()
    n_orig_sig = int(g["sig_orig"].sum())
    L.append(f"- 유지: {', '.join(kept) if kept else '없음'} "
             f"(원본에서 유의한 {n_orig_sig}개 중 {len(kept)}개)")
    L.append(f"- 정제 후 소멸: {', '.join(lost) if lost else '없음'}")
    L.append(f"- 정제 후 등장: {', '.join(gained) if gained else '없음'}")
    L.append("- 주의: 두 스펙 모두 BH(FDR) 보정 후에는 유의성이 유지되지 않을 수 있다"
             "(원본 분석과 동일한 한계).")
    L.append("")
    stable = g[(~g["sig_orig"]) & (~g["sig_clean"]) & (g["p_delta"].abs() < 0.02)]
    if len(stable):
        desc = ", ".join(f"{r['feature']}({r['orig_p']}->{r['clean_p']})"
                         for _, r in stable.iterrows())
        L.append(f"- 유의성과 별개인 p값 안정성: {desc} 는 규칙을 바꿔도 p값이 거의 "
                 f"변하지 않지만 두 버전 모두 α={kc.ALPHA} 에 미달한다. "
                 "따라서 **'정제판에서도 유지'라는 표현을 쓰지 말 것**. "
                 "규칙 강건성은 유의성 근거가 되지 않는다.")
        L.append("")

    L.append("## R2. 구조변화 전환점\n")
    L.append(b.to_markdown(index=False))
    L.append("")
    same = b[b["break_ym_same"]]["series"].tolist()
    diff = b[~b["break_ym_same"]]["series"].tolist()
    L.append(f"- 전환점 동일: {', '.join(same) if same else '없음'}")
    L.append(f"- 전환점 변동: {', '.join(diff) if diff else '없음'}")
    sni = b[b["series"] == "safety_neg_intensity"]
    if len(sni):
        r = sni.iloc[0]
        L.append(f"- 핵심(safety_neg_intensity): 원본 {r['orig_break_ym']} vs "
                 f"정제판 {r['clean_break_ym']} -> "
                 f"{'2024-08 화재 사건 포착 유지' if r['clean_break_ym'] == '2024-08' else '재확인 필요'}")
    L.append("")

    L.append("## R3. 패널 상호작용 (지역FE + AR + 주효과 + 상호작용)\n")
    L.append("`_p_cluster` = 월 클러스터(주 스펙), `_p` = HC1(참고).\n")
    L.append(p.to_markdown(index=False))
    L.append("")
    if "centSafety_x_chgPerEv_p_cluster" in p.columns:
        o = p[p["spec"] == "원본KG"].iloc[0]
        c = p[p["spec"] == "정제판KG"].iloc[0]
        same_sign = np.sign(o["centSafety_x_chgPerEv_coef"]) == np.sign(c["centSafety_x_chgPerEv_coef"])
        L.append(f"- centSafety x charger_per_ev: 원본 계수 {o['centSafety_x_chgPerEv_coef']} "
                 f"(월 클러스터 p={o['centSafety_x_chgPerEv_p_cluster']}) vs 정제판 "
                 f"{c['centSafety_x_chgPerEv_coef']} "
                 f"(월 클러스터 p={c['centSafety_x_chgPerEv_p_cluster']})")
        L.append(f"- 부호 {'일관' if same_sign else '반전'}, "
                 f"정제판 유의성(월 클러스터) "
                 f"{f'유의(p<{kc.ALPHA})' if c['centSafety_x_chgPerEv_p_cluster'] < kc.ALPHA else '비유의'}")
    L.append("")

    path = os.path.join(kc.TABLE_DIR, "kg_clean_robustness.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[30] 저장: {path}")


if __name__ == "__main__":
    main()
