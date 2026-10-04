# -*- coding: utf-8 -*-
"""
[29] 정제판(clean) KG 변이 생성

`28_kg_quality_audit.py` 의 진단 결과를 교정한 KG 를 **별도 경로**에 생성한다.
원본 산출물(news_tagged.csv, corpus_*.csv, kg_monthly_*, kg_monthly_features.csv)은
일절 수정하지 않는다.

적용 교정 (scripts/kg_clean_rules.py)
  C1 최장일치 우선 개체 추출 (아이오닉5 -> 아이오닉 이중계상 제거)
  C2 단어경계/부정문맥 규칙 (GM, 리프, 니로, 기아 오탐 차단)
  C3 동어반복 ENTITY_FRAME 엣지 제외
  C4 충전난 별칭 확장
  C5 산업 프레임 키워드 정밀화
  C6 프레임 키워드 부분문자열 오탐 차단 (리스, 셀)

감성 분류는 원본과 동일하게 유지한다(KG 구축 효과만 분리하기 위함).

출력:
  data/processed/news_tagged_clean.csv
  outputs/kg/clean/monthly/kg_monthly_nodes.csv, kg_monthly_edges.csv
  outputs/kg/clean/corpus_nodes.csv, corpus_edges.csv
  data/processed/kg_monthly_features_clean.csv
  outputs/kg/tables/tbl_clean_vs_original_tagging.csv
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from importlib import import_module
from itertools import combinations

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc
import kg_clean_rules as kr

CLEAN_TAGGED = os.path.join(kc.DATA_PROC, "news_tagged_clean.csv")
CLEAN_KG_DIR = os.path.join(kc.KG_DIR, "clean")
CLEAN_MONTHLY_DIR = os.path.join(CLEAN_KG_DIR, "monthly")
CLEAN_FEATURES = os.path.join(kc.DATA_PROC, "kg_monthly_features_clean.csv")

SENTI_IDX = {"긍정": "n_pos", "중립": "n_neu", "부정": "n_neg"}


def ent_node(en: str) -> str:
    t, n = en.split(":", 1)
    return f"Entity_{t}:{n}"


# ============================================================
# STEP 1: 정제 규칙으로 재태깅
# ============================================================
def retag():
    csv.field_size_limit(10 ** 7)
    rows_out = []
    n_total = n_ev = 0
    frame_counter = Counter()
    entity_counter = Counter()

    with open(kc.ALL_NEWS, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            n_total += 1
            title = kc.norm_text(row.get("Title"))
            content = kc.norm_text(row.get("Content"))
            text = f"{title} {content}"
            if not kc.contains_any(text, kc.EV_KEYWORDS):
                continue
            n_ev += 1

            # 원본(14번)과 동일한 판정 텍스트 범위
            text_clf = f"{title} {content[:1000]}"
            # 감성은 원본과 동일 로직 유지
            label, score, n_pos, n_neg = kc.classify_sentiment(text_clf)
            frames = kr.detect_frames_clean(text_clf)
            entities = kr.detect_entities_clean(text_clf)

            for fr in frames:
                frame_counter[fr] += 1
            for etype, canon in entities:
                entity_counter[f"{etype}:{canon}"] += 1

            year = kc.norm_text(row.get("Year"))
            month = kc.norm_text(row.get("Month"))
            ym = f"{int(year):04d}-{int(month):02d}" if year and month else ""

            rows_out.append({
                "article_id": f"A{n_ev:06d}",
                "Date": kc.norm_text(row.get("Date")),
                "ym": ym,
                "Year": year,
                "Month": month,
                "newspaper": kc.norm_text(row.get("newspaper")),
                "Title": title,
                "sentiment": label,
                "sentiment_score": f"{score:.4f}",
                "n_pos": n_pos,
                "n_neg": n_neg,
                "frames": ";".join(frames),
                "entities": ";".join(f"{t}:{c}" for t, c in entities),
                "Url": kc.norm_text(row.get("Url")),
            })

    fieldnames = ["article_id", "Date", "ym", "Year", "Month", "newspaper", "Title",
                  "sentiment", "sentiment_score", "n_pos", "n_neg", "frames",
                  "entities", "Url"]
    with open(CLEAN_TAGGED, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"  [재태깅] 전체 {n_total} / 전기차 관련 {n_ev} -> {CLEAN_TAGGED}")
    return frame_counter, entity_counter, n_ev


# ============================================================
# STEP 2: 정제판 월별/통합 KG (동어반복 엣지 제외)
# ============================================================
def build_clean_kg():
    os.makedirs(CLEAN_MONTHLY_DIR, exist_ok=True)
    csv.field_size_limit(10 ** 7)

    node_stat = defaultdict(lambda: {"count": 0, "n_pos": 0, "n_neu": 0, "n_neg": 0})
    edge_stat = defaultdict(int)
    corpus_node = Counter()
    corpus_edge = Counter()
    n_used = 0
    n_taut_suppressed = 0

    with open(CLEAN_TAGGED, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ym = row["ym"]
            if not ym or not (kc.ANALYSIS_START <= ym <= kc.ANALYSIS_END):
                continue
            n_used += 1
            senti_key = SENTI_IDX.get(row["sentiment"], "n_neu")
            frames = [x for x in row["frames"].split(";") if x]
            ents = [x for x in row["entities"].split(";") if x]

            for fr in frames:
                node = f"Frame:{fr}"
                s = node_stat[(ym, "Frame", node)]
                s["count"] += 1
                s[senti_key] += 1
                corpus_node[("Frame", node)] += 1
            for en in ents:
                node = ent_node(en)
                s = node_stat[(ym, "Entity", node)]
                s["count"] += 1
                s[senti_key] += 1
                corpus_node[("Entity", node)] += 1

            for a, b in combinations(sorted(set(frames)), 2):
                edge_stat[(ym, "FRAME_COOC", f"Frame:{a}", f"Frame:{b}")] += 1
                corpus_edge[("FRAME_COOC", f"Frame:{a}", f"Frame:{b}")] += 1
            for a, b in combinations(sorted(set(ents)), 2):
                edge_stat[(ym, "ENTITY_COOC", ent_node(a), ent_node(b))] += 1
                corpus_edge[("ENTITY_COOC", ent_node(a), ent_node(b))] += 1
            for en in set(ents):
                canon = en.split(":", 1)[1]
                enode = ent_node(en)
                for fr in set(frames):
                    # C3: 사전 구성상 100% 보장되는 쌍은 정보량이 없으므로 제외
                    if kr.is_tautological(canon, fr):
                        n_taut_suppressed += 1
                        continue
                    edge_stat[(ym, "ENTITY_FRAME", enode, f"Frame:{fr}")] += 1
                    corpus_edge[("ENTITY_FRAME", enode, f"Frame:{fr}")] += 1

    npath = os.path.join(CLEAN_MONTHLY_DIR, "kg_monthly_nodes.csv")
    with open(npath, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ym", "node_type", "node", "count", "n_pos", "n_neu", "n_neg"])
        for (ym, ntype, node), s in sorted(node_stat.items()):
            w.writerow([ym, ntype, node, s["count"], s["n_pos"], s["n_neu"], s["n_neg"]])

    epath = os.path.join(CLEAN_MONTHLY_DIR, "kg_monthly_edges.csv")
    with open(epath, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ym", "edge_type", "source", "target", "weight"])
        for (ym, etype, s_, d_), wgt in sorted(edge_stat.items()):
            w.writerow([ym, etype, s_, d_, wgt])

    with open(os.path.join(CLEAN_KG_DIR, "corpus_nodes.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_type", "node", "count"])
        for (ntype, node), c in corpus_node.most_common():
            w.writerow([ntype, node, c])
    with open(os.path.join(CLEAN_KG_DIR, "corpus_edges.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_type", "source", "target", "weight"])
        for (etype, s_, d_), c in corpus_edge.most_common():
            w.writerow([etype, s_, d_, c])

    print(f"  [정제KG] 사용 기사 {n_used}, 동어반복 엣지 인스턴스 {n_taut_suppressed}건 제외")
    print(f"  [정제KG] 통합 노드 {len(corpus_node)}, 통합 엣지 {len(corpus_edge)}")
    return n_used, n_taut_suppressed, corpus_node


# ============================================================
# STEP 3: 정제판 월별 지표 (16번 로직 재사용, 경로만 치환)
# ============================================================
def build_clean_features():
    m16 = import_module("16_kg_monthly_indicators")
    saved = (kc.KG_MONTHLY_DIR, kc.KG_MONTHLY_FEATURES_CSV, kc.NEWS_TAGGED_CSV)
    try:
        kc.KG_MONTHLY_DIR = CLEAN_MONTHLY_DIR
        kc.KG_MONTHLY_FEATURES_CSV = CLEAN_FEATURES
        kc.NEWS_TAGGED_CSV = CLEAN_TAGGED
        m16.main()
    finally:
        kc.KG_MONTHLY_DIR, kc.KG_MONTHLY_FEATURES_CSV, kc.NEWS_TAGGED_CSV = saved


# ============================================================
# STEP 4: 원본 대비 태깅 변화 비교표
# ============================================================
def compare_tagging(clean_nodes, n_used):
    orig = {}
    with open(os.path.join(kc.KG_DIR, "corpus_nodes.csv"),
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            orig[(r["node_type"], r["node"])] = int(r["count"])

    rows = []
    for key in set(orig) | set(clean_nodes):
        o = orig.get(key, 0)
        c = clean_nodes.get(key, 0)
        if o == c:
            continue
        rows.append({"node_type": key[0], "node": key[1], "original": o, "clean": c,
                     "diff": c - o,
                     "pct_change": round((c - o) / o * 100, 1) if o else None})
    df = pd.DataFrame(rows).sort_values("diff")
    df.to_csv(os.path.join(kc.TABLE_DIR, "tbl_clean_vs_original_tagging.csv"),
              index=False, encoding="utf-8-sig")
    return df, orig


def main():
    kc.ensure_dirs()
    os.makedirs(CLEAN_KG_DIR, exist_ok=True)

    print("[29] STEP1 정제 규칙으로 재태깅...")
    frame_counter, entity_counter, n_ev = retag()

    print("[29] STEP2 정제판 KG 구축...")
    n_used, n_taut, clean_nodes = build_clean_kg()

    print("[29] STEP3 정제판 월별 지표 산출...")
    build_clean_features()

    print("[29] STEP4 원본 대비 비교...")
    diff_df, orig_nodes = compare_tagging(clean_nodes, n_used)

    # 요약
    L = ["# 정제판 KG 생성 요약\n"]
    L.append(f"- 동어반복 (canonical, frame) 쌍: {len(kr.TAUTOLOGICAL_PAIRS)}종, "
             f"엣지 인스턴스 {n_taut}건 제외")
    L.append(f"- 분석기간 사용 기사: {n_used}건 (원본과 동일 코퍼스)")
    L.append("")
    L.append("## 프레임 출현율 변화 (정제 후)\n")
    frows = []
    for fr in kr.FRAME_NAMES_CLEAN:
        c = clean_nodes.get(("Frame", f"Frame:{fr}"), 0)
        o = orig_nodes.get(("Frame", f"Frame:{fr}"), 0)
        frows.append({"frame": fr, "original": o, "clean": c,
                      "clean_share": round(c / n_used, 4) if n_used else 0.0})
    fdf = pd.DataFrame(frows).sort_values("clean", ascending=False)
    L.append(fdf.to_markdown(index=False))
    L.append("")
    L.append("## 개체 노드 변화 상위 (감소순)\n")
    if len(diff_df):
        show = diff_df[diff_df["node_type"] == "Entity"].head(12)
        L.append(show.to_markdown(index=False))
    L.append("")
    L.append(f"- 전체 변화 노드 {len(diff_df)}개 -> `tbl_clean_vs_original_tagging.csv`")
    L.append("- 핵심 결과 유지 여부는 `scripts/30_kg_clean_robustness.py` 에서 검증한다.")

    path = os.path.join(kc.TABLE_DIR, "kg_clean_build_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[29] 저장: {path}")


if __name__ == "__main__":
    main()
