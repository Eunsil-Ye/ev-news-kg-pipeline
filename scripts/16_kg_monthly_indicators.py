# -*- coding: utf-8 -*-
"""
[16] KG 파생 월별 담론 지표 산출

입력:
  - outputs/kg/monthly/kg_monthly_nodes.csv
  - outputs/kg/monthly/kg_monthly_edges.csv
  - data/processed/news_tagged.csv   (월별 기사 수)
출력:
  - data/processed/kg_monthly_features.csv   (ym x 담론 구조 지표)

지표(모두 월별):
  - frame_density        프레임 공출현 네트워크 밀도
  - frame_entropy        프레임 분포 엔트로피(담론 다양성)
  - frame_modularity     프레임 네트워크 모듈러리티(담론 분절도)
  - frame_n_communities  담론 군집 수
  - cent_safety/신차/보조금/충전  주요 프레임 고유벡터 중심성
  - cooc_safety_battery, cooc_price_subsidy, cooc_safety_charging  특정 공출현율
  - safety_neg_intensity 안전 프레임 부정 노출 강도
  - newmodel_pos_intensity 신차 프레임 긍정 노출 강도
  - salience_tesla, salience_hyundai, salience_newmodel  개체 현저성
  - company_hhi          기업 담론 집중도(허핀달)
  - neg_event_propagation 부정 사건 개체의 프레임 전파도
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc


def load_monthly():
    nodes = defaultdict(list)  # ym -> [(node_type, node, count, n_pos, n_neu, n_neg)]
    npath = os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_nodes.csv")
    with open(npath, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            nodes[r["ym"]].append((
                r["node_type"], r["node"], int(r["count"]),
                int(r["n_pos"]), int(r["n_neu"]), int(r["n_neg"]),
            ))
    edges = defaultdict(list)  # ym -> [(edge_type, src, dst, weight)]
    epath = os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_edges.csv")
    with open(epath, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            edges[r["ym"]].append((r["edge_type"], r["source"], r["target"], int(r["weight"])))
    return nodes, edges


def month_article_counts():
    counts = defaultdict(int)
    with open(kc.NEWS_TAGGED_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ym = r["ym"]
            if ym and kc.ANALYSIS_START <= ym <= kc.ANALYSIS_END:
                counts[ym] += 1
    return counts


def safe_eigenvector(G, node):
    if node not in G or G.number_of_edges() == 0:
        return 0.0
    try:
        cent = nx.eigenvector_centrality_numpy(G, weight="weight")
        return float(cent.get(node, 0.0))
    except Exception:
        try:
            cent = nx.degree_centrality(G)
            return float(cent.get(node, 0.0))
        except Exception:
            return 0.0


def shannon_entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p, 2)
    return ent


def main():
    kc.ensure_dirs()
    nodes, edges = load_monthly()
    art_counts = month_article_counts()
    months = sorted(nodes.keys())

    key_frames = ["안전", "신차", "보조금", "충전"]
    rows = []

    for ym in months:
        n_art = max(art_counts.get(ym, 0), 1)

        # 노드 집계
        frame_count = {}
        frame_pos = {}
        frame_neg = {}
        company_count = defaultdict(int)
        entity_count = defaultdict(int)  # canonical name -> count
        model_total = 0
        for ntype, node, cnt, npos, nneu, nneg in nodes[ym]:
            if ntype == "Frame":
                fr = node.split(":", 1)[1]
                frame_count[fr] = cnt
                frame_pos[fr] = npos
                frame_neg[fr] = nneg
            elif ntype == "Entity":
                # node = "Entity_company:현대차"
                sub = node.split(":", 1)[1]
                etype = node.split(":", 1)[0].replace("Entity_", "")
                entity_count[sub] += cnt
                if etype == "company":
                    company_count[sub] += cnt
                if etype == "model":
                    model_total += cnt

        # 프레임 네트워크 구성
        Gf = nx.Graph()
        for fr in frame_count:
            Gf.add_node(f"Frame:{fr}")
        for etype, s, d, w in edges[ym]:
            if etype == "FRAME_COOC":
                Gf.add_edge(s, d, weight=w)

        # 개체 네트워크 (동시등장)
        Ge = nx.Graph()
        for etype, s, d, w in edges[ym]:
            if etype == "ENTITY_COOC":
                Ge.add_edge(s, d, weight=w)

        # --- 지표 계산 ---
        frame_density = nx.density(Gf) if Gf.number_of_nodes() > 1 else 0.0
        frame_entropy = shannon_entropy(list(frame_count.values()))

        try:
            from networkx.algorithms.community import greedy_modularity_communities, modularity
            if Gf.number_of_edges() > 0:
                comms = list(greedy_modularity_communities(Gf, weight="weight"))
                frame_modularity = modularity(Gf, comms, weight="weight")
                frame_n_comm = len(comms)
            else:
                frame_modularity, frame_n_comm = 0.0, Gf.number_of_nodes()
        except Exception:
            frame_modularity, frame_n_comm = 0.0, 0

        cent = {}
        for fr in key_frames:
            cent[fr] = safe_eigenvector(Gf, f"Frame:{fr}")

        # 특정 공출현율 (같은 기사 동시 등장 수 / 월 기사수)
        cooc = {("안전", "배터리"): 0, ("가격", "보조금"): 0, ("안전", "충전"): 0}
        for etype, s, d, w in edges[ym]:
            if etype != "FRAME_COOC":
                continue
            a = s.split(":", 1)[1]
            b = d.split(":", 1)[1]
            for (x, y) in list(cooc.keys()):
                if {a, b} == {x, y}:
                    cooc[(x, y)] += w

        safety_neg_intensity = frame_neg.get("안전", 0) / n_art
        newmodel_pos_intensity = frame_pos.get("신차", 0) / n_art

        salience_tesla = entity_count.get("테슬라", 0) / n_art
        salience_hyundai = entity_count.get("현대차", 0) / n_art
        salience_newmodel = model_total / n_art

        # 기업 담론 집중도(HHI)
        comp_total = sum(company_count.values())
        if comp_total > 0:
            company_hhi = sum((c / comp_total) ** 2 for c in company_count.values())
        else:
            company_hhi = 0.0

        # 부정 사건 개체 프레임 전파도: 부정사건 개체가 닿은 distinct 프레임 수 평균
        ent_frame_map = defaultdict(set)
        for etype, s, d, w in edges[ym]:
            if etype == "ENTITY_FRAME":
                ename = s.split(":", 1)[1]
                ent_frame_map[ename].add(d)
        prop_vals = [len(ent_frame_map.get(e, set())) for e in kc.NEGATIVE_EVENT_ENTITIES]
        neg_event_propagation = (sum(prop_vals) / len(prop_vals)) if prop_vals else 0.0

        rows.append({
            "ym": ym,
            "frame_density": round(frame_density, 5),
            "frame_entropy": round(frame_entropy, 5),
            "frame_modularity": round(frame_modularity, 5),
            "frame_n_communities": frame_n_comm,
            "cent_safety": round(cent["안전"], 5),
            "cent_newmodel": round(cent["신차"], 5),
            "cent_subsidy": round(cent["보조금"], 5),
            "cent_charging": round(cent["충전"], 5),
            "cooc_safety_battery": round(cooc[("안전", "배터리")] / n_art, 5),
            "cooc_price_subsidy": round(cooc[("가격", "보조금")] / n_art, 5),
            "cooc_safety_charging": round(cooc[("안전", "충전")] / n_art, 5),
            "safety_neg_intensity": round(safety_neg_intensity, 5),
            "newmodel_pos_intensity": round(newmodel_pos_intensity, 5),
            "salience_tesla": round(salience_tesla, 5),
            "salience_hyundai": round(salience_hyundai, 5),
            "salience_newmodel": round(salience_newmodel, 5),
            "company_hhi": round(company_hhi, 5),
            "neg_event_propagation": round(neg_event_propagation, 5),
        })

    fieldnames = list(rows[0].keys())
    with open(kc.KG_MONTHLY_FEATURES_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[16] 월 수 {len(rows)}, 지표 수 {len(fieldnames) - 1}")
    print(f"[16] 지표: {[c for c in fieldnames if c != 'ym']}")
    print(f"[16] 저장: {kc.KG_MONTHLY_FEATURES_CSV}")


if __name__ == "__main__":
    main()
