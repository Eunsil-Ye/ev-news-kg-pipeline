# -*- coding: utf-8 -*-
"""
[17] 등록/뉴스/KG 결합 데이터셋 구축

입력:
  - 201801_202605_전기차등록현황.xls        (전기차 누적 등록, 월 x 17시도)
  - data/processed/total_vehicle_registration_month_region_201801_202510.csv
  - data/processed/ev_charger_region_speed_snapshots.csv
  - data/processed/news_tagged.csv
  - data/processed/kg_monthly_features.csv
출력:
  - data/processed/news_monthly_simple.csv   (단순 뉴스 월별 지표)
  - data/processed/national_ev_news_kg.csv   (전국 월별: 종속 + 뉴스 + KG + 시차)
  - data/processed/panel_ev_news_kg.csv      (17시도 x 월 패널)

종속변수: 누적 등록의 전월 차분 = 전기차 등록 순증가(ev_net_add).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

REGIONS17 = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
             "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]

FRAME_ROMAN = {
    "정책": "policy", "보조금": "subsidy", "충전": "charging", "안전": "safety",
    "가격": "price", "배터리": "battery", "신차": "newmodel",
    "환경": "environment", "산업": "industry",
}

# 충전기 8권역 -> 17시도 매핑
REGION_TO_GROUP = {
    "서울": "서울", "경기": "경기", "인천": "인천", "강원": "강원",
    "대전": "충청", "세종": "충청", "충북": "충청", "충남": "충청",
    "광주": "전라", "전북": "전라", "전남": "전라",
    "부산": "경상", "대구": "경상", "울산": "경상", "경북": "경상", "경남": "경상",
    "제주": "제주",
}


def parse_ev_registration():
    raw = pd.read_excel(kc.EV_REGISTRATION_XLS, header=2)
    # 원본 xls 하단에는 연도별 합계를 담은 별도의 footer 표가 있고, 그 표를 편집한
    # 흔적으로 본 표 우측에 빈 열이 더 붙는 경우가 있다(19열 -> 22열). 열 개수를
    # 가정하지 않고 라벨 행에서 '합계' 위치를 찾아 본 표 범위만 잘라 쓴다.
    labels = raw.iloc[0].astype(str).str.strip().tolist()
    ncol = labels.index("합계") + 1 if "합계" in labels else len(REGIONS17) + 2
    if labels[1:ncol - 1] != REGIONS17:
        raise ValueError(
            f"등록자료 지역 열 구성이 예상과 다르다: {labels[1:ncol - 1]}")
    raw = raw.iloc[1:, :ncol].copy()  # 첫 데이터행(지역명 라벨) 제거 + 본 표만
    raw.columns = ["ym"] + REGIONS17 + ["합계"]
    raw["ym"] = raw["ym"].astype(str).str.strip()
    raw = raw[raw["ym"].str.match(r"^\d{4}-\d{2}$")].copy()
    for col in REGIONS17 + ["합계"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.sort_values("ym").reset_index(drop=True)

    # long 형태 (지역)
    long = raw.melt(id_vars="ym", value_vars=REGIONS17,
                    var_name="region", value_name="ev_stock")
    long = long.sort_values(["region", "ym"]).reset_index(drop=True)
    long["ev_net_add"] = long.groupby("region")["ev_stock"].diff()
    long["ev_growth_mom"] = long.groupby("region")["ev_stock"].pct_change()

    # national
    nat = raw[["ym", "합계"]].rename(columns={"합계": "ev_stock"}).copy()
    nat = nat.sort_values("ym").reset_index(drop=True)
    nat["ev_net_add"] = nat["ev_stock"].diff()
    nat["ev_growth_mom"] = nat["ev_stock"].pct_change()
    return nat, long


def load_total_vehicle():
    t = pd.read_csv(kc.TOTAL_VEH_REGION_CSV)
    t = t[["ym", "region", "total_car_stock"]].copy()
    nat = t.groupby("ym", as_index=False)["total_car_stock"].sum()
    return nat, t


def load_charger_monthly():
    """충전기 스냅샷을 월별로 forward-fill. 권역별 + 전국.

    원본 스냅샷에는 8개 권역 행과 함께 전국 '합계' 행이 같이 들어 있다.
    구분 없이 더하면 전국 값이 정확히 2배가 되므로 먼저 합계 행을 떨어낸다.
    권역별 집계(grp)는 '합계'가 REGION_TO_GROUP 에 없어 병합에서 자연히 빠지지만,
    같은 실수를 되풀이하지 않도록 여기서 한 번에 제거한다.
    """
    c = pd.read_csv(kc.CHARGER_CSV)
    c = c[c["region_group"] != "합계"].copy()
    grp = c.groupby(["ym", "region_group"], as_index=False)["charger_count"].sum()
    nat = c.groupby("ym", as_index=False)["charger_count"].sum().rename(
        columns={"charger_count": "charger_total"})
    return nat, grp


def build_news_simple():
    df = pd.read_csv(kc.NEWS_TAGGED_CSV, dtype=str, keep_default_na=False)
    df = df[(df["ym"] >= kc.ANALYSIS_START) & (df["ym"] <= kc.ANALYSIS_END)].copy()
    df["score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    rows = []
    for ym, g in df.groupby("ym"):
        n = len(g)
        pos = (g["sentiment"] == "긍정").sum()
        neg = (g["sentiment"] == "부정").sum()
        neu = (g["sentiment"] == "중립").sum()
        rec = {
            "ym": ym,
            "news_count": n,
            "pos_share": pos / n,
            "neg_share": neg / n,
            "neu_share": neu / n,
            "net_sentiment": (pos - neg) / n,
            "avg_sentiment_score": g["score"].mean(),
        }
        for fr, roman in FRAME_ROMAN.items():
            cnt = g["frames"].str.split(";").apply(lambda xs: fr in xs).sum()
            rec[f"frshare_{roman}"] = cnt / n
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("ym").reset_index(drop=True)
    out.to_csv(kc.NEWS_MONTHLY_SIMPLE_CSV, index=False, encoding="utf-8-sig")
    return out


def add_lags(df, cols, lags, group=None):
    for c in cols:
        for k in lags:
            newc = f"{c}_lag{k}"
            if group:
                df[newc] = df.groupby(group)[c].shift(k)
            else:
                df[newc] = df[c].shift(k)
    return df


def main():
    kc.ensure_dirs()
    lags = [1, 2, 3, 6]

    ev_nat, ev_reg = parse_ev_registration()
    tot_nat, tot_reg = load_total_vehicle()
    chg_nat, chg_grp = load_charger_monthly()
    news = build_news_simple()
    kgf = pd.read_csv(kc.KG_MONTHLY_FEATURES_CSV)

    news_cols = [c for c in news.columns if c != "ym"]
    kg_cols = [c for c in kgf.columns if c != "ym"]

    # ---------------- 전국 데이터셋 ----------------
    nat = ev_nat.merge(tot_nat, on="ym", how="left")
    nat["ev_share"] = nat["ev_stock"] / nat["total_car_stock"]
    # 충전기 월별 forward fill
    allmonths = pd.DataFrame({"ym": sorted(nat["ym"].unique())})
    chg_nat_full = allmonths.merge(chg_nat, on="ym", how="left").sort_values("ym")
    chg_nat_full["charger_total"] = chg_nat_full["charger_total"].ffill()
    nat = nat.merge(chg_nat_full, on="ym", how="left")
    nat = nat.merge(news, on="ym", how="left").merge(kgf, on="ym", how="left")
    nat["log_ev_net_add"] = np.log1p(nat["ev_net_add"].clip(lower=0))
    nat = nat[(nat["ym"] >= kc.ANALYSIS_START) & (nat["ym"] <= kc.ANALYSIS_END)].copy()
    nat = nat.sort_values("ym").reset_index(drop=True)
    nat = add_lags(nat, news_cols + kg_cols, lags)
    nat.to_csv(kc.NATIONAL_CSV, index=False, encoding="utf-8-sig")

    # ---------------- 지역 패널 ----------------
    panel = ev_reg.merge(tot_reg, on=["ym", "region"], how="left")
    panel["ev_share"] = panel["ev_stock"] / panel["total_car_stock"]
    panel["region_group"] = panel["region"].map(REGION_TO_GROUP)
    # 권역 충전기를 월별로 채운다. 원본 공표 주기가 2023-12 까지 반기이고 2025-05
    # 이후가 불규칙하므로, 선형보간이 아니라 직전 관측값 유지(ffill)로 메운다.
    # 충전기 자료가 있는 59개월 중 실관측은 24개월이고 35개월이 채운 값이다.
    grp_months = pd.MultiIndex.from_product(
        [sorted(panel["ym"].unique()), sorted(chg_grp["region_group"].unique())],
        names=["ym", "region_group"]).to_frame(index=False)
    chg_grp_full = grp_months.merge(chg_grp, on=["ym", "region_group"], how="left")
    chg_grp_full = chg_grp_full.sort_values(["region_group", "ym"])
    chg_grp_full["charger_count"] = chg_grp_full.groupby("region_group")["charger_count"].ffill()
    chg_grp_full = chg_grp_full.rename(columns={"charger_count": "charger_group"})
    panel = panel.merge(chg_grp_full, on=["ym", "region_group"], how="left")
    # 전국 뉴스 + KG (모든 지역 공통)
    panel = panel.merge(news, on="ym", how="left").merge(kgf, on="ym", how="left")
    # 분자는 권역 단위, 분모는 시도 단위라 단위가 일치하지 않는다. 충전기가 8권역으로만
    # 공표되기 때문이며, 1:1 대응인 5개 시도(서울·경기·인천·강원·제주)를 제외하면
    # "해당 시도의 충전기 밀도"로 읽을 수 없다. 권역 인프라 여건의 통제항으로만 쓴다.
    panel["charger_per_ev"] = panel["charger_group"] / panel["ev_stock"].replace(0, np.nan)
    panel["log_ev_net_add"] = np.log1p(panel["ev_net_add"].clip(lower=0))
    panel = panel[(panel["ym"] >= kc.ANALYSIS_START) & (panel["ym"] <= kc.ANALYSIS_END)].copy()
    panel = panel.sort_values(["region", "ym"]).reset_index(drop=True)
    panel = add_lags(panel, news_cols + kg_cols, lags, group="region")
    panel.to_csv(kc.PANEL_CSV, index=False, encoding="utf-8-sig")

    print(f"[17] 전국 데이터셋: {nat.shape} -> {kc.NATIONAL_CSV}")
    print(f"[17]   기간 {nat['ym'].min()}~{nat['ym'].max()}, 종속 ev_net_add 결측 {nat['ev_net_add'].isna().sum()}")
    print(f"[17] 지역 패널: {panel.shape} -> {kc.PANEL_CSV}")
    print(f"[17]   지역수 {panel['region'].nunique()}, 월수 {panel['ym'].nunique()}")
    print(f"[17] 단순 뉴스 지표: {news.shape} -> {kc.NEWS_MONTHLY_SIMPLE_CSV}")
    print(f"[17] 뉴스 컬럼 {len(news_cols)}, KG 컬럼 {len(kg_cols)}")


if __name__ == "__main__":
    main()
