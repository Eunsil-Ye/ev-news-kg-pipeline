# -*- coding: utf-8 -*-
"""
[31] 개체/프레임 검토표본에 판정 근거(문맥·매칭어) 열 추가

`25_entity_frame_review_sample.py` 가 만든 표본에는 제목만 있어서, 사람이 판정하려면
기사 원문을 일일이 열어야 한다. 태깅이 실제로 본 텍스트(제목 + 본문 앞 1000자)에서
`target` 을 유발한 문자열과 그 주변 문맥을 뽑아 붙인다.

추가 열
  matched_terms   target 을 유발한 별칭/키워드 (여러 개면 세미콜론 구분)
  match_context   첫 매칭 위치 주변 +-40자
  context_status  정상 / 매칭없음 / 원문없음

판정 기준(느슨한 기준)
  entity 행: matched_terms 의 별칭이 그 문맥에서 실제 그 기업·모델을 가리키면 1
  frame  행: matched_terms 의 키워드가 그 문맥에서 해당 프레임 의미로 쓰였으면 1
             (예: target=충전 인데 "휴대폰 충전" 이면 0, "급속충전기 설치" 면 1)

기존 열과 이미 기입된 값은 보존한다. 여러 번 실행해도 안전하다.
"""

from __future__ import annotations

import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

REVIEW_PATH = os.path.join(kc.OUT_DIR, "entity_frame_review_sample.csv")
CONTEXT_WINDOW = 40
CONTENT_LIMIT = 1000  # 14_extract_entities.py 와 동일해야 한다

COLUMN_ORDER = [
    "check_type", "target", "article_id", "ym", "newspaper", "Title",
    "tagged_frames", "tagged_entities",
    "matched_terms", "match_context",
    "human_correct", "review_memo", "context_status", "Url",
]


def load_article_text():
    """Url -> 태깅이 실제로 본 텍스트(제목 + 정규화 본문 앞 1000자).

    article_id 는 태깅 단계에서 새로 부여한 일련번호라 원문에 없다. Url 로 조인한다.
    """
    csv.field_size_limit(10 ** 7)
    by_url = {}
    with open(kc.ALL_NEWS, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = kc.norm_text(row.get("Url"))
            if not url or url in by_url:
                continue
            # 14번과 동일한 순서: 정규화 후 절단
            title = kc.norm_text(row.get("Title"))
            content = kc.norm_text(row.get("Content"))
            by_url[url] = f"{title} {content[:CONTENT_LIMIT]}"
    return by_url


def terms_for(check_type: str, target: str):
    """target 을 유발할 수 있는 별칭/키워드 목록."""
    if check_type == "frame":
        return list(kc.FRAME_KEYWORDS.get(target, []))
    # entity: target 은 "type:canonical" 형식
    return [alias for alias, (etype, canon) in kc.ENTITY_DICT.items()
            if f"{etype}:{canon}" == target and alias != "EV"]


def find_matches(text: str, terms):
    """(매칭된 용어 목록, 가장 앞선 매칭 위치) 반환. detect_* 와 동일한 규칙."""
    low = text.lower()
    hits, first = [], None
    for t in terms:
        pos = low.find(t.lower())
        if pos < 0:
            continue
        hits.append(t)
        if first is None or pos < first:
            first = pos
    return hits, first


def main():
    if not os.path.exists(REVIEW_PATH):
        print(f"[31] 검토표본이 없다: {REVIEW_PATH} -> 25번을 먼저 실행하라.")
        return
    if not os.path.exists(kc.ALL_NEWS):
        print(f"[31] 원문 코퍼스가 없다: {kc.ALL_NEWS}")
        return

    df = pd.read_csv(REVIEW_PATH, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    print(f"[31] 검토표본 {len(df)}건 로드")

    print("[31] 원문 코퍼스 인덱싱...")
    by_url = load_article_text()
    print(f"  Url 인덱스 {len(by_url)}건")

    terms_col, ctx_col, status_col = [], [], []
    for _, r in df.iterrows():
        url = kc.norm_text(r.get("Url"))
        text = by_url.get(url)
        if not text:
            terms_col.append("")
            ctx_col.append("")
            status_col.append("원문없음")
            continue

        hits, first = find_matches(text, terms_for(str(r["check_type"]), str(r["target"])))
        if not hits:
            terms_col.append("")
            ctx_col.append(text[:120])
            status_col.append("매칭없음")
            continue

        st = max(0, first - CONTEXT_WINDOW)
        en = min(len(text), first + CONTEXT_WINDOW * 2)
        terms_col.append(";".join(hits))
        ctx_col.append(text[st:en].replace("\n", " "))
        status_col.append("정상")

    df["matched_terms"] = terms_col
    df["match_context"] = ctx_col
    df["context_status"] = status_col

    for c in ("human_correct", "review_memo"):
        if c not in df.columns:
            df[c] = ""
    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]

    print()
    print(df["context_status"].value_counts().to_string())

    try:
        df.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
        print(f"\n[31] 저장: {REVIEW_PATH}")
    except PermissionError:
        # 엑셀 등에서 파일을 열어두면 잠긴다. 계산은 끝났으므로 안내만 하고 종료한다.
        print(f"\n[31] 저장 실패: 파일이 다른 프로그램(엑셀 등)에서 열려 있다.")
        print(f"     {REVIEW_PATH}")
        print("     해당 파일을 닫고 이 스크립트를 다시 실행하면 된다.")
        return

    bad = df[df["context_status"] != "정상"]
    if len(bad):
        print(f"\n[31] 재현 불가 {len(bad)}건 (검토 대상에서 제외하고 원인 확인 필요)")
        print(bad[["check_type", "target", "context_status", "Title"]]
              .head(20).to_string(index=False))
    else:
        print("\n[31] 300건 전부 태깅 근거를 재현했다 -> 태깅 로직과 조인 모두 정상")

    n_frame = int((df["check_type"] == "frame").sum())
    n_ent = int((df["check_type"] == "entity").sum())
    print(f"\n[31] 검토 대상: 개체 {n_ent}건 / 프레임 {n_frame}건")
    print("  matched_terms 와 match_context 를 보고 human_correct 에 1/0 을 기입한 뒤")
    print("  `python scripts/25_entity_frame_review_sample.py` 를 실행하면 정밀도가 산출된다.")


if __name__ == "__main__":
    main()
