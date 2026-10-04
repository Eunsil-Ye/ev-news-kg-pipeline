# -*- coding: utf-8 -*-
"""
오탐 검토표본(`outputs/kg_falsepositive_review_sample.csv`)의 기계 판독 라벨.

판정 기준
    `match_context` 안에서 매칭된 문자열이 실제로 그 개체(canonical)를 지칭하는가.
    기사의 주제가 그 개체인지는 묻지 않는다. 예를 들어 IRA 정책 기사에 등장한
    "현대·기아차"의 '기아'는 주제가 정책이더라도 기아자동차를 지칭하므로 정탐이다.

계층 중복(아이오닉5 를 '아이오닉' 으로도 세는 문제)은 오탐이 아니라 별도 결함이며
`28_kg_quality_audit.py` A5 에서 따로 측정한다. 여기서는 정탐으로 본다.

아래 목록은 100건의 문맥을 하나씩 읽어 오탐으로 판정한 22건이다.
규칙으로 자동 생성하지 않고 (별칭, Url) 로 고정해 둔다. 정제판 정규식의 성능을
이 라벨로 평가하기 때문에, 라벨이 규칙 변경에 따라 함께 움직이면 순환논증이 된다.
"""

from __future__ import annotations

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP_REVIEW_PATH = os.path.join(BASE_DIR, "outputs", "kg_falsepositive_review_sample.csv")

# (별칭, Url): 오탐 사유
FALSE_POSITIVES = {
    # --- 리프: 닛산 Leaf 가 아닌 것 (16건) ---
    ("리프", "https://www.joongang.co.kr/article/25248747"): "정비용 리프트",
    ("리프", "https://www.khan.co.kr/article/202409241020001"): "드리프트",
    ("리프", "https://www.hankyung.com/article/2025082507481"): "폴리프로필렌",
    ("리프", "https://www.joongang.co.kr/article/24084355"): "차량공유 리프트(Lyft)",
    ("리프", "https://www.hankyung.com/article/2020022151301"): "페이스리프트",
    ("리프", "https://www.joongang.co.kr/article/25196459"): "드리프트",
    ("리프", "https://www.joongang.co.kr/article/25278112"): "데드리프트",
    ("리프", "https://www.khan.co.kr/article/202509072056015"): "아파트 브랜드 엘리프",
    ("리프", "https://www.khan.co.kr/article/202301101616001"): "폴리프로필렌",
    ("리프", "https://www.khan.co.kr/article/202108051035001"): "리프레시 코스",
    ("리프", "https://www.hankyung.com/article/2021092279851"): "차량공유 리프트(Lyft)",
    ("리프", "https://www.hankyung.com/article/2024012205831"): "드리프트",
    ("리프", "https://www.hankyung.com/article/2020062879521"): "페이스리프트",
    ("리프", "https://www.joongang.co.kr/article/24000602"): "페이스리프트",
    ("리프", "https://www.joongang.co.kr/article/25365668"): "중국 리프모터(Leapmotor)",
    ("리프", "https://www.joongang.co.kr/article/23584561"): "차량호출 리프트(Lyft)",
    # --- GM: 영문 약칭 내부 매칭 (5건) ---
    ("GM", "https://www.joongang.co.kr/article/25322132"): "HMGMA(현대차 미국공장)",
    ("GM", "https://www.khan.co.kr/article/202212090945001"): "E-GMP 플랫폼",
    ("GM", "https://www.joongang.co.kr/article/25318785"): "HMGMA(현대차 미국공장)",
    ("GM", "https://www.hankyung.com/article/2021082438581"): "GGM(광주글로벌모터스)",
    ("GM", "https://www.khan.co.kr/article/202406231618001"): "KGM(KG모빌리티)",
    # --- 기아: 굶주림 (1건) ---
    ("기아", "https://www.khan.co.kr/article/202407311609011"): "기아 퇴치(굶주림)",
}


def apply_labels(path: str = FP_REVIEW_PATH) -> pd.DataFrame:
    """검토표본에 agent_correct / agent_reason 열을 채워 저장한다.

    사람이 기입한 human_correct·review_memo 는 건드리지 않는다.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]

    keys = list(zip(df["alias"].astype(str), df["Url"].astype(str)))
    dup = [k for k in set(keys) if keys.count(k) > 1]
    if dup:
        raise ValueError(f"(alias, Url) 키가 중복된다: {dup}")

    unknown = set(FALSE_POSITIVES) - set(keys)
    if unknown:
        raise ValueError(f"표본에 없는 오탐 키가 지정됐다: {unknown}")

    df["agent_correct"] = [0 if k in FALSE_POSITIVES else 1 for k in keys]
    df["agent_reason"] = [FALSE_POSITIVES.get(k, "") for k in keys]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def agreement(df: pd.DataFrame) -> dict:
    """사람이 기입한 행에 한해 기계 판독과의 일치도를 계산한다."""
    sub = df[df["human_correct"].notna()]
    if sub.empty:
        return {"n": 0, "agree": 0, "rate": None}
    n_agree = int((sub["human_correct"].astype(int) == sub["agent_correct"]).sum())
    return {"n": len(sub), "agree": n_agree, "rate": round(n_agree / len(sub), 4)}


if __name__ == "__main__":
    d = apply_labels()
    print(f"라벨 기입 완료: {len(d)}건 (오탐 {int((d['agent_correct'] == 0).sum())}건)")
    print(f"사람 기입분 대조: {agreement(d)}")
    print()
    g = d.groupby("alias")["agent_correct"].agg(["sum", "count"])
    g["precision"] = (g["sum"] / g["count"]).round(4)
    print(g.rename(columns={"sum": "정탐", "count": "검토"}))
    print(f"\n전체 정밀도: {d['agent_correct'].mean():.4f}")
