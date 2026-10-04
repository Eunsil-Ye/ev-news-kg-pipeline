# -*- coding: utf-8 -*-
"""
[15] 월별 시간적 지식그래프(KG) 스냅샷 구축

입력: data/processed/news_tagged.csv
출력:
  - outputs/kg/monthly/kg_monthly_nodes.csv  (ym, node_type, node, count, n_pos, n_neu, n_neg)
  - outputs/kg/monthly/kg_monthly_edges.csv  (ym, edge_type, source, target, weight)
  - outputs/kg/corpus_nodes.csv / corpus_edges.csv  (전체 기간 통합 그래프, 시각화용)

각 월(ym)마다 다음을 집계한다.
  - Frame/Entity 노드의 등장 기사 수 + 감성 분해
  - Frame-Frame, Entity-Entity, Entity-Frame 공출현(같은 기사 동시 등장) 가중치
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

SENTI_IDX = {"긍정": "n_pos", "중립": "n_neu", "부정": "n_neg"}


def in_period(ym: str) -> bool:
    return bool(ym) and kc.ANALYSIS_START <= ym <= kc.ANALYSIS_END


def main():
    kc.ensure_dirs()
    csv.field_size_limit(10 ** 7)

    # 월별 노드 카운트: (ym, node_type, node) -> {count, n_pos, n_neu, n_neg}
    node_stat = defaultdict(lambda: {"count": 0, "n_pos": 0, "n_neu": 0, "n_neg": 0})
    # 월별 엣지 카운트: (ym, edge_type, src, dst) -> weight
    edge_stat = defaultdict(int)

    # 전체 기간 통합용
    corpus_node = defaultdict(lambda: {"count": 0})
    corpus_edge = defaultdict(int)

    n_used = 0
    with open(kc.NEWS_TAGGED_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ym = row["ym"]
            if not in_period(ym):
                continue
            n_used += 1
            senti = row["sentiment"]
            senti_key = SENTI_IDX.get(senti, "n_neu")

            frames = [x for x in row["frames"].split(";") if x]
            ents = [x for x in row["entities"].split(";") if x]  # "type:name"

            # --- 노드 통계 ---
            for fr in frames:
                node = f"Frame:{fr}"
                s = node_stat[(ym, "Frame", node)]
                s["count"] += 1
                s[senti_key] += 1
                corpus_node[("Frame", node)]["count"] += 1
            for en in ents:
                etype = en.split(":", 1)[0]
                node = f"Entity_{etype}:{en.split(':',1)[1]}"
                s = node_stat[(ym, "Entity", node)]
                s["count"] += 1
                s[senti_key] += 1
                corpus_node[("Entity", node)]["count"] += 1

            # --- 엣지 통계 (공출현) ---
            for a, b in combinations(sorted(set(frames)), 2):
                edge_stat[(ym, "FRAME_COOC", f"Frame:{a}", f"Frame:{b}")] += 1
                corpus_edge[("FRAME_COOC", f"Frame:{a}", f"Frame:{b}")] += 1
            for a, b in combinations(sorted(set(ents)), 2):
                s = f"Entity_{a.split(':',1)[0]}:{a.split(':',1)[1]}"
                d = f"Entity_{b.split(':',1)[0]}:{b.split(':',1)[1]}"
                edge_stat[(ym, "ENTITY_COOC", s, d)] += 1
                corpus_edge[("ENTITY_COOC", s, d)] += 1
            for en in set(ents):
                enode = f"Entity_{en.split(':',1)[0]}:{en.split(':',1)[1]}"
                for fr in set(frames):
                    edge_stat[(ym, "ENTITY_FRAME", enode, f"Frame:{fr}")] += 1
                    corpus_edge[("ENTITY_FRAME", enode, f"Frame:{fr}")] += 1

    # --- 월별 노드 저장 ---
    nodes_path = os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_nodes.csv")
    with open(nodes_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ym", "node_type", "node", "count", "n_pos", "n_neu", "n_neg"])
        for (ym, ntype, node), s in sorted(node_stat.items()):
            w.writerow([ym, ntype, node, s["count"], s["n_pos"], s["n_neu"], s["n_neg"]])

    # --- 월별 엣지 저장 ---
    edges_path = os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_edges.csv")
    with open(edges_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ym", "edge_type", "source", "target", "weight"])
        for (ym, etype, s_, d_), wgt in sorted(edge_stat.items()):
            w.writerow([ym, etype, s_, d_, wgt])

    # --- 통합 그래프 저장 ---
    cn_path = os.path.join(kc.KG_DIR, "corpus_nodes.csv")
    with open(cn_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_type", "node", "count"])
        for (ntype, node), s in sorted(corpus_node.items(), key=lambda x: -x[1]["count"]):
            w.writerow([ntype, node, s["count"]])
    ce_path = os.path.join(kc.KG_DIR, "corpus_edges.csv")
    with open(ce_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_type", "source", "target", "weight"])
        for (etype, s_, d_), wgt in sorted(corpus_edge.items(), key=lambda x: -x[1]):
            w.writerow([etype, s_, d_, wgt])

    months = sorted({k[0] for k in node_stat})
    print(f"[15] 사용 기사 {n_used}, 월 수 {len(months)} ({months[0]}~{months[-1]})")
    print(f"[15] 월별 노드행 {len(node_stat)}, 월별 엣지행 {len(edge_stat)}")
    print(f"[15] 통합 노드 {len(corpus_node)}, 통합 엣지 {len(corpus_edge)}")
    print(f"[15] 저장: {nodes_path}")
    print(f"[15] 저장: {edges_path}")


if __name__ == "__main__":
    main()
