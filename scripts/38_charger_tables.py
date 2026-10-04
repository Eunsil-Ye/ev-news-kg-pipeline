# -*- coding: utf-8 -*-
"""
[38] 충전기 설치 현황 표

ev_charger_region_speed_snapshots.csv(권역 × 충전속도 × 기준월 스냅샷)에서
전국 급속·완속 추이, 권역별 누적 대수, 스냅샷 기준월 목록을 만든다.
스냅샷에는 8개 권역 행과 전국 '합계' 행이 함께 있으므로 권역 행만 더하고,
그 값이 '합계' 행과 일치하는지 확인한다.

출력 (outputs/kg/tables/):
  - tbl_charger_national.csv
  - tbl_charger_region.csv
  - tbl_charger_calendar.csv
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

GROUPS = [
    ("서울", "서울"),
    ("경기", "경기"),
    ("인천", "인천"),
    ("강원", "강원"),
    ("충청", "대전·세종·충북·충남"),
    ("전라", "광주·전북·전남"),
    ("경상", "부산·대구·울산·경북·경남"),
    ("제주", "제주"),
]
REGION_MONTHS = ["2020-12", "2023-12", "2024-12", "2025-08", "2025-12"]


def month_gap(a: str, b: str) -> int:
    ya, ma = map(int, a.split("-"))
    yb, mb = map(int, b.split("-"))
    return (yb - ya) * 12 + (mb - ma)


def main():
    kc.ensure_dirs()
    raw = pd.read_csv(kc.CHARGER_CSV, encoding="utf-8-sig")
    reg = raw[raw["region_group"] != "합계"]
    tot = raw[raw["region_group"] == "합계"].groupby("ym")["charger_count"].sum()

    speed = reg.pivot_table(index="ym", columns="charger_speed",
                            values="charger_count", aggfunc="sum").astype(int)
    speed["합계"] = speed["급속"] + speed["완속"]
    gap = (speed["합계"] - tot.reindex(speed.index)).abs().max()
    print(f"[38] 권역 합 vs '합계' 행 최대 차이: {gap}")

    months = list(speed.index)
    in_period = [m for m in months if m <= kc.ANALYSIS_END]
    last_obs = in_period[-1]

    rows = []
    for m in in_period + ([months[-1]] if months[-1] != last_obs else []):
        if m > kc.ANALYSIS_END:
            note = f"원자료 최신. 종속변수가 {kc.ANALYSIS_END}까지라 분석 미사용"
        elif m == months[0]:
            note = "자료 시작"
        elif m.endswith("-12"):
            note = "연말"
        elif m == last_obs:
            fill_from = pd.Period(m, "M") + 1
            note = (f"분석기간({kc.ANALYSIS_END}) 내 최신 실관측. "
                    f"{fill_from}~{kc.ANALYSIS_END[-2:]}은 이 값을 유지")
        else:
            continue
        s = speed.loc[m]
        rows.append({"기준월": m, "급속": s["급속"], "완속": s["완속"], "합계": s["합계"],
                     "급속비중_pct": round(s["급속"] / s["합계"] * 100, 1), "비고": note})
    national = pd.DataFrame(rows)

    by_group = reg.groupby(["region_group", "ym"])["charger_count"].sum().unstack("ym")
    region = pd.DataFrame(GROUPS, columns=["권역", "시도 대응"])
    for m in REGION_MONTHS:
        region[m] = region["권역"].map(by_group[m]).astype(int)
    total = {"권역": "합계", "시도 대응": None, **{m: int(region[m].sum()) for m in REGION_MONTHS}}
    region = pd.concat([region, pd.DataFrame([total])], ignore_index=True)

    cal = []
    for i, m in enumerate(months):
        g = month_gap(months[i - 1], m) if i else 6
        if g == 1:
            cycle = "월별"
        elif g == 6:
            cycle = "반기"
            if i and speed.loc[m, "합계"] == speed.loc[months[i - 1], "합계"]:
                cycle = f"반기({months[i - 1]}과 동일값)"
        else:
            cycle = "불규칙"
        inc = "Y" if m <= kc.ANALYSIS_END else f"N ({kc.ANALYSIS_END} 이후)"
        cal.append({"기준월": m, "주기": cycle, "분석기간 포함": inc})
    calendar = pd.DataFrame(cal)

    for name, df in [("national", national), ("region", region), ("calendar", calendar)]:
        path = os.path.join(kc.TABLE_DIR, f"tbl_charger_{name}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[38] 저장 {path} ({len(df)}행)")
    print(f"[38] 분석기간 내 실관측 기준월 {len(in_period)}개")


if __name__ == "__main__":
    main()
