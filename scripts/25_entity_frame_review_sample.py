# -*- coding: utf-8 -*-
"""
[25] 개체/프레임 사전추출 사람 검증 표본 생성 + (파일 채워지면) 정밀도/재현율 계산

사전기반 개체/프레임 태깅의 신뢰도를 사람 검토로 검증하기 위한 층화표본을 만든다.
- 프레임: 각 프레임이 태깅된 기사 중 층화추출(기본 프레임당 20건)
- 개체: 주요 canonical 개체가 태깅된 기사 중 층화추출(기본 개체당 10건)

검토자는 생성된 CSV 의 human_frame_correct / human_entity_correct 열에
1(맞음)/0(틀림) 을 기입한다. 파일이 채워진 뒤 다시 실행하면 정밀도를 계산한다.

출력:
  - outputs/entity_frame_review_sample.csv   (사람 입력 대기)
  - outputs/kg/tables/tbl_entity_frame_precision.csv (채워진 뒤)
"""

from __future__ import annotations

import os
import sys
import random

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

SEED = 20260720
PER_FRAME = 20
PER_ENTITY = 10
TOP_ENTITIES = 12
REVIEW_PATH = os.path.join(kc.OUT_DIR, "entity_frame_review_sample.csv")


def build_sample(nt):
    rng = random.Random(SEED)
    rows = []
    seen = set()

    # 프레임 층화표본
    for fr in kc.FRAME_NAMES:
        mask = nt["frames"].fillna("").str.split(";").apply(lambda xs: fr in xs)
        pool = nt[mask].index.tolist()
        rng.shuffle(pool)
        for idx in pool[:PER_FRAME]:
            rows.append(("frame", fr, idx))

    # 개체 층화표본 (빈도 상위 canonical)
    ent_counter = {}
    for ents in nt["entities"].fillna(""):
        for e in ents.split(";"):
            if e:
                ent_counter[e] = ent_counter.get(e, 0) + 1
    top = sorted(ent_counter, key=ent_counter.get, reverse=True)[:TOP_ENTITIES]
    for ent in top:
        mask = nt["entities"].fillna("").str.split(";").apply(lambda xs: ent in xs)
        pool = nt[mask].index.tolist()
        rng.shuffle(pool)
        for idx in pool[:PER_ENTITY]:
            rows.append(("entity", ent, idx))

    out = []
    for kind, target, idx in rows:
        key = (kind, target, idx)
        if key in seen:
            continue
        seen.add(key)
        r = nt.loc[idx]
        out.append({
            "check_type": kind,          # frame / entity
            "target": target,            # 검증 대상 프레임/개체명
            "article_id": r.get("article_id", idx),
            "ym": r.get("ym", ""),
            "newspaper": r.get("newspaper", ""),
            "Title": str(r.get("Title", ""))[:200],
            "tagged_frames": r.get("frames", ""),
            "tagged_entities": r.get("entities", ""),
            "human_correct": "",         # 1=대상이 실제로 기사에 존재, 0=오탐
            "review_memo": "",
            "Url": r.get("Url", ""),
        })
    df = pd.DataFrame(out)
    # 검토 편의를 위해 유형/대상별 정렬 + 섞기
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def score_if_filled():
    if not os.path.exists(REVIEW_PATH):
        return False
    df = pd.read_csv(REVIEW_PATH)
    filled = df[df["human_correct"].isin([0, 1, "0", "1"])].copy()
    if len(filled) == 0:
        print("  [25] 아직 human_correct 미기입 → 정밀도 계산 생략.")
        return False
    filled["human_correct"] = filled["human_correct"].astype(int)
    rows = []
    for kind in ["frame", "entity"]:
        sub = filled[filled["check_type"] == kind]
        if len(sub) == 0:
            continue
        rows.append({"check_type": kind, "n_reviewed": len(sub),
                     "precision": round(sub["human_correct"].mean(), 3)})
        # 대상별 정밀도
        for tgt, g in sub.groupby("target"):
            rows.append({"check_type": f"{kind}:{tgt}", "n_reviewed": len(g),
                         "precision": round(g["human_correct"].mean(), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(kc.TABLE_DIR, "tbl_entity_frame_precision.csv"),
               index=False, encoding="utf-8-sig")
    print(res.to_markdown(index=False))
    print("  [25] 정밀도 표 저장: tbl_entity_frame_precision.csv")
    return True


def main():
    kc.ensure_dirs()
    nt = pd.read_csv(kc.NEWS_TAGGED_CSV)

    if os.path.exists(REVIEW_PATH):
        # 이미 존재하면 채워졌는지 확인해 채점 시도
        if score_if_filled():
            return
        print(f"  [25] 기존 검토표본 유지: {REVIEW_PATH} (사람 입력 대기)")
        return

    df = build_sample(nt)
    df.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    print(f"[25] 검토표본 생성: {REVIEW_PATH}")
    print(f"  - 총 {len(df)}건 (프레임 {(df['check_type']=='frame').sum()} / "
          f"개체 {(df['check_type']=='entity').sum()})")
    print("  - human_correct 열에 1(정탐)/0(오탐) 기입 후 이 스크립트를 다시 실행하면 정밀도가 계산됩니다.")


if __name__ == "__main__":
    main()
