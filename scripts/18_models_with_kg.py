# -*- coding: utf-8 -*-
"""
[18] 전국 예측 3단 비교 + Granger + Diebold-Mariano + 지역 패널 고정효과

입력:
  - data/processed/national_ev_news_kg.csv
  - data/processed/panel_ev_news_kg.csv
출력 (outputs/kg/tables/):
  - tbl_national_forecast.csv   (기준 vs 단순뉴스 vs +KG 확장창 1기 예측 성능)
  - tbl_granger.csv             (뉴스/KG 지표 -> 등록 순증가 선행성)
  - tbl_panel_models.csv        (패널 P0/P1/P2 설명력)
  - tbl_panel_interactions.csv  (P2 상호작용 계수, HC1 및 월 클러스터 p)
  - tbl_panel_main_effects.csv  (P1 주효과 계수와 95% CI)
  - tbl_panel_lag_joint.csv     (시차군 결합검정과 3개월 누적효과)
  - model_summary.md

21·22·30·33·37번이 이 모듈의 함수와 변수 목록을 재사용한다.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

LAGS = [1, 2, 3]

SIMPLE_BASE = ["news_count", "neg_share", "net_sentiment", "pos_share",
               "frshare_safety", "frshare_newmodel", "frshare_subsidy", "frshare_charging"]
KG_BASE = ["cent_safety", "salience_newmodel", "safety_neg_intensity",
           "newmodel_pos_intensity", "frame_entropy", "neg_event_propagation", "company_hhi"]


def lag_names(bases):
    return [f"{b}_lag{k}" for b in bases for k in LAGS]


def add_target_lags(df, target, lags=(1, 2, 3, 12), group=None):
    for k in lags:
        col = f"{target}_lag{k}"
        df[col] = df.groupby(group)[target].shift(k) if group else df[target].shift(k)
    return df


# ============================================================
# 전국 시계열: 확장창 1기 예측 성능
# ============================================================
def expanding_forecast(df, target, features, min_train=42):
    d = df.dropna(subset=[target] + features).reset_index(drop=True)
    n = len(d)
    if n <= min_train + 5:
        min_train = max(int(n * 0.6), 10)
    preds, actuals = [], []
    for t in range(min_train, n):
        train = d.iloc[:t]
        test = d.iloc[t:t + 1]
        Xtr = train[features].values
        ytr = train[target].values
        scaler = StandardScaler().fit(Xtr)
        model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=20000)
        model.fit(scaler.transform(Xtr), ytr)
        pred = model.predict(scaler.transform(test[features].values))[0]
        preds.append(pred)
        actuals.append(test[target].values[0])
    actuals = np.array(actuals)
    preds = np.array(preds)
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    # 방향성 정확도(전월 대비 상승/하락)
    if len(actuals) > 1:
        da = np.mean(np.sign(np.diff(actuals)) == np.sign(np.diff(preds)))
    else:
        da = np.nan
    return rmse, mae, r2, float(da), preds, actuals, d


def diebold_mariano(e1, e2, h=1):
    """간이 DM 검정 (제곱오차 손실). 양수 stat = model1 오차가 더 큼."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    # Newey-West 유사 분산 (lag h-1)
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * (1 - k / h) * gk
    dm = dbar / np.sqrt(var / n) if var > 0 else np.nan
    return float(dm)


def run_national():
    df = pd.read_csv(kc.NATIONAL_CSV)
    df = df.sort_values("ym").reset_index(drop=True)
    df = add_target_lags(df, "ev_net_add", lags=(1, 2, 3, 12))

    base_feats = ["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3", "ev_net_add_lag12"]
    simple_feats = base_feats + lag_names(SIMPLE_BASE)
    kg_feats = simple_feats + lag_names(KG_BASE)

    # 공통 평가 구간 정렬을 위해 kg_feats 기준으로 dropna
    common = df.dropna(subset=["ev_net_add"] + kg_feats).reset_index(drop=True)

    results = []
    err_store = {}
    for name, feats in [("기준(AR)", base_feats), ("단순뉴스", simple_feats), ("KG포함", kg_feats)]:
        rmse, mae, r2, da, preds, actuals, _ = expanding_forecast(common, "ev_net_add", feats)
        results.append({"model": name, "n_features": len(feats),
                        "RMSE": round(rmse, 1), "MAE": round(mae, 1),
                        "R2": round(r2, 4), "direction_acc": round(da, 3)})
        err_store[name] = actuals - preds

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_national_forecast.csv"),
                  index=False, encoding="utf-8-sig")

    # DM 검정: 단순뉴스 vs KG포함
    dm = diebold_mariano(err_store["단순뉴스"], err_store["KG포함"], h=1)

    # Granger: 후보 지표 -> ev_net_add (차분 후)
    granger_rows = []
    cand = SIMPLE_BASE + KG_BASE
    for f in cand:
        if f not in df.columns:
            continue
        sub = df[["ev_net_add", f]].dropna()
        # 차분으로 정상성 확보
        sub = sub.diff().dropna()
        if len(sub) < 20 or sub[f].std() == 0:
            continue
        try:
            gc = grangercausalitytests(sub[["ev_net_add", f]], maxlag=3, verbose=False)
            pvals = {k: gc[k][0]["ssr_ftest"][1] for k in gc}
            best_lag = min(pvals, key=pvals.get)
            granger_rows.append({"feature": f, "best_lag": best_lag,
                                 "p_value": round(pvals[best_lag], 4),
                                 "type": "KG" if f in KG_BASE else "simple"})
        except Exception:
            continue
    g_df = pd.DataFrame(granger_rows).sort_values("p_value")
    g_df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_granger.csv"),
                index=False, encoding="utf-8-sig")
    return res_df, dm, g_df


# ============================================================
# 지역 패널: 고정효과 + 상호작용
# ============================================================
def run_panel():
    df = pd.read_csv(kc.PANEL_CSV)
    df = df.sort_values(["region", "ym"]).reset_index(drop=True)
    df = add_target_lags(df, "ev_net_add", lags=(1, 2, 3), group="region")

    # 표준화 대상
    inter_main = ["safety_neg_intensity_lag1", "salience_newmodel_lag1", "cent_safety_lag1"]
    moderators = ["charger_per_ev", "ev_share"]
    need = (["ev_net_add", "ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3",
             "region"] + inter_main + moderators +
            lag_names(["neg_share", "frshare_newmodel", "frshare_safety"]))
    d = df.dropna(subset=[c for c in need if c in df.columns]).reset_index(drop=True)

    # 연속변수 표준화
    cont = (["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3"] +
            inter_main + moderators +
            lag_names(["neg_share", "frshare_newmodel", "frshare_safety"]))
    cont = [c for c in cont if c in d.columns]
    dz = d.copy()
    for c in cont:
        s = dz[c].std()
        dz[c] = (dz[c] - dz[c].mean()) / s if s > 0 else 0.0

    region_dum = pd.get_dummies(dz["region"], prefix="reg", drop_first=True).astype(float)
    ar = dz[["ev_net_add_lag1", "ev_net_add_lag2", "ev_net_add_lag3"]]
    y = d["ev_net_add"].values

    # 뉴스/KG 지표는 전국 공통값이라 같은 달의 17개 지역이 동일한 값을 갖는다.
    # 관측치가 월 단위로 독립이 아니어서 HC1 은 표준오차를 과소추정하므로(Moulton 문제)
    # 월 클러스터 표준오차를 주 결과로 함께 산출한다.
    ym_groups = pd.factorize(d["ym"])[0]

    def fit(Xparts, label, cov="HC1"):
        X = pd.concat([ar] + Xparts + [region_dum], axis=1)
        X = sm.add_constant(X)
        if cov == "cluster":
            m = sm.OLS(y, X.values).fit(cov_type="cluster",
                                        cov_kwds={"groups": ym_groups})
        else:
            m = sm.OLS(y, X.values).fit(cov_type="HC1")
        return {"model": label, "r2": round(m.rsquared, 4),
                "adj_r2": round(m.rsquared_adj, 4), "n": int(m.nobs),
                "k": X.shape[1]}, m, list(X.columns)

    # P0: FE + AR
    r0, m0, _ = fit([], "P0_FE_AR")
    # P1: + 뉴스/KG 주효과
    main_names = lag_names(["neg_share", "frshare_newmodel", "frshare_safety"]) + inter_main
    main = dz[main_names]
    r1, m1, cols1 = fit([main], "P1_main")
    # P2: + 상호작용
    inter = pd.DataFrame(index=dz.index)
    inter["safetyNeg_x_chgPerEv"] = dz["safety_neg_intensity_lag1"] * dz["charger_per_ev"]
    inter["newmodel_x_evshare"] = dz["salience_newmodel_lag1"] * dz["ev_share"]
    inter["centSafety_x_chgPerEv"] = dz["cent_safety_lag1"] * dz["charger_per_ev"]
    r2, m2, cols2 = fit([main, inter], "P2_interaction")

    rows = [r0, r1, r2]
    pdf = pd.DataFrame(rows)
    pdf.to_csv(os.path.join(kc.TABLE_DIR, "tbl_panel_models.csv"),
               index=False, encoding="utf-8-sig")

    # 상호작용 계수 추출
    # 상호작용항도 월 공통 뉴스/KG 지표를 포함하므로 P1 과 같은 이유로 월 클러스터를 병행한다.
    _, m2c, _ = fit([main, inter], "P2_interaction_clustered", cov="cluster")
    inter_names = ["safetyNeg_x_chgPerEv", "newmodel_x_evshare", "centSafety_x_chgPerEv"]
    coef_rows = []
    for nm in inter_names:
        if nm in cols2:
            # cols2 는 add_constant 적용 후의 컬럼명이라 'const' 가 이미 포함되어 있다.
            # 여기에 +1 을 더하면 한 칸 뒤(지역 더미) 계수를 읽게 되므로 그대로 사용한다.
            idx = cols2.index(nm)
            coef_rows.append({"term": nm, "coef": round(m2.params[idx], 2),
                              "p_value": round(m2.pvalues[idx], 4),
                              "p_cluster_ym": round(m2c.pvalues[idx], 4)})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_panel_interactions.csv"),
                   index=False, encoding="utf-8-sig")

    # P1 주효과 계수 + 95% CI
    # 감성의 시차 선행성(H1)에 대한 패널 근거다. 귀무를 서술할 때는 p값만으로
    # 부족하므로 신뢰구간을 함께 남긴다.
    _, m1c, _ = fit([main], "P1_main_clustered", cov="cluster")
    ci1 = m1.conf_int()
    ci1c = m1c.conf_int()
    main_rows = []
    for nm in main_names:
        if nm not in cols1:
            continue
        idx = cols1.index(nm)
        main_rows.append({
            "term": nm,
            "coef": round(m1.params[idx], 2),
            "se_hc1": round(m1.bse[idx], 2),
            "p_hc1": round(m1.pvalues[idx], 4),
            "ci_low_hc1": round(ci1[idx][0], 2),
            "ci_high_hc1": round(ci1[idx][1], 2),
            "se_cluster_ym": round(m1c.bse[idx], 2),
            "p_cluster_ym": round(m1c.pvalues[idx], 4),
            "ci_low_cluster": round(ci1c[idx][0], 2),
            "ci_high_cluster": round(ci1c[idx][1], 2),
            # sig05 가 판정 기준(kc.ALPHA)이고 sig10 은 참고용이다.
            "sig05_hc1": bool(m1.pvalues[idx] < kc.ALPHA),
            "sig05_cluster": bool(m1c.pvalues[idx] < kc.ALPHA),
            "sig10_hc1": bool(m1.pvalues[idx] < 0.1),
            "sig10_cluster": bool(m1c.pvalues[idx] < 0.1),
        })
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_panel_main_effects.csv"),
                   index=False, encoding="utf-8-sig")

    # 시차군 결합검정 + 누적효과
    # 개별 시차 계수는 인접 시차 간 다중공선성으로 부호가 뒤집힐 수 있으므로,
    # 시차 전체의 결합 F검정과 계수 합(3개월 누적 순효과)을 함께 산출한다.
    joint_rows = []
    for label, base in [("neg_share", "neg_share"),
                        ("frshare_safety", "frshare_safety"),
                        ("frshare_newmodel", "frshare_newmodel")]:
        idxs = [cols1.index(f"{base}_lag{k}") for k in LAGS
                if f"{base}_lag{k}" in cols1]
        if len(idxs) != len(LAGS):
            continue
        k_tot = len(cols1)
        R_joint = np.zeros((len(idxs), k_tot))
        for r, i in enumerate(idxs):
            R_joint[r, i] = 1.0
        R_sum = np.zeros((1, k_tot))
        for i in idxs:
            R_sum[0, i] = 1.0
        ft, ftc = m1.f_test(R_joint), m1c.f_test(R_joint)
        tt, ttc = m1.t_test(R_sum), m1c.t_test(R_sum)
        joint_rows.append({
            "var_group": label,
            "joint_F_hc1": round(float(np.squeeze(ft.fvalue)), 3),
            "joint_p_hc1": round(float(np.squeeze(ft.pvalue)), 4),
            "joint_F_cluster": round(float(np.squeeze(ftc.fvalue)), 3),
            "joint_p_cluster": round(float(np.squeeze(ftc.pvalue)), 4),
            "cum_coef": round(float(np.squeeze(tt.effect)), 2),
            "cum_p_hc1": round(float(np.squeeze(tt.pvalue)), 4),
            "cum_se_cluster": round(float(np.squeeze(ttc.sd)), 2),
            "cum_ci_low_cluster": round(float(np.squeeze(ttc.conf_int()[0][0])), 2),
            "cum_ci_high_cluster": round(float(np.squeeze(ttc.conf_int()[0][1])), 2),
            "cum_p_cluster": round(float(np.squeeze(ttc.pvalue)), 4),
        })
    joint_df = pd.DataFrame(joint_rows)
    joint_df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_panel_lag_joint.csv"),
                    index=False, encoding="utf-8-sig")
    return pdf, coef_df, main_df, joint_df


def write_summary(res_df, dm, g_df, pdf, coef_df, main_df, joint_df):
    lines = ["# KG 결합 모형 결과 요약\n"]
    lines.append("## 1. 전국 3단 예측 성능 (확장창 1기 예측)\n")
    lines.append(res_df.to_markdown(index=False))
    lines.append("")
    lines.append(f"- Diebold-Mariano (단순뉴스 vs KG포함): DM={dm:.3f} "
                 f"(양수면 KG포함 모형의 오차가 더 작음)\n")
    lines.append(f"## 2. Granger 선행성 (차분 후, maxlag=3, p<{kc.ALPHA} 유의)\n")
    sig = g_df[g_df["p_value"] < kc.ALPHA]
    lines.append((sig if len(sig) else g_df.head(10)).to_markdown(index=False))
    lines.append("")
    lines.append("## 3. 지역 패널 고정효과 모형\n")
    lines.append(pdf.to_markdown(index=False))
    lines.append("")
    lines.append("### 상호작용 계수 (P2, 표준화; p_value=HC1, p_cluster_ym=월 클러스터)\n")
    lines.append(coef_df.to_markdown(index=False) if len(coef_df) else "(없음)")
    lines.append("")
    lines.append("### P1 주효과 계수 + 95% CI (표준화, HC1)\n")
    lines.append(main_df.to_markdown(index=False) if len(main_df) else "(없음)")
    lines.append("")
    lines.append("- `neg_share_lag1/2/3` 은 감성의 시차 선행성(H1) 검정의 패널 근거다.\n")
    lines.append("- **HC1 과 월 클러스터를 함께 볼 것.** 뉴스/KG 지표는 전국 공통값이라 "
                 "같은 달의 17개 지역이 동일한 값을 가지므로 관측치가 월 단위로 독립이 아니고, "
                 "HC1 은 표준오차를 과소추정한다. **월 클러스터 결과가 주 결과**이고 "
                 "HC1 은 참고값이다.\n")
    lines.append("- 귀무를 서술할 때는 p값이 아니라 **신뢰구간**을 인용한다. CI 폭이 넓으면 "
                 "'효과 없음'이 아니라 '탐지할 만한 검정력이 없음'이 정확한 진술이다.\n")
    lines.append("### 시차군 결합검정 + 누적효과 (P1 기준)\n")
    lines.append(joint_df.to_markdown(index=False) if len(joint_df) else "(없음)")
    lines.append("")
    lines.append("- `joint_p` 는 '해당 변수의 1~3개월 시차 계수가 모두 0'이라는 귀무가설의 "
                 "F검정이다. 개별 시차 계수는 인접 시차 간 다중공선성으로 부호가 뒤집힐 수 "
                 "있어 단독 해석이 위험하므로, 시차군 전체의 유의성은 이 값으로 판단한다.\n")
    lines.append("- `cum_coef` 는 세 시차 계수의 합(3개월 누적 순효과)이다. 개별 계수가 "
                 "유의해도 누적이 0 근처면 '즉각 반응 후 반등'으로 해석해야 하며, "
                 "'부정 보도가 등록을 지속적으로 위축시킨다'는 주장은 성립하지 않는다.\n")
    path = os.path.join(kc.TABLE_DIR, "model_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return path


def main():
    kc.ensure_dirs()
    print("[18] 전국 모형 실행...")
    res_df, dm, g_df = run_national()
    print("[18] 패널 모형 실행...")
    pdf, coef_df, main_df, joint_df = run_panel()
    path = write_summary(res_df, dm, g_df, pdf, coef_df, main_df, joint_df)
    print(f"\n[18] 요약 저장: {path}")


if __name__ == "__main__":
    main()
