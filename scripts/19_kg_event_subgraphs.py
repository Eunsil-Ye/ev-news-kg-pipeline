# -*- coding: utf-8 -*-
"""
[19] 담론 진화 시각화 + 사건 기반 서브그래프

입력:
  - data/processed/news_monthly_simple.csv
  - data/processed/kg_monthly_features.csv
  - outputs/kg/monthly/kg_monthly_edges.csv
  - data/processed/national_ev_news_kg.csv (등록 순증가 오버레이)
출력 (outputs/kg/figures/):
  - discourse_evolution.png       프레임 비중 누적면적(담론 진화)
  - kg_indicators_timeline.png    주요 KG 지표 + 등록 순증가 타임라인
  - event_subgraph_<ym>_<focus>.png  사건 월 개체-프레임 서브그래프
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

FRAME_ROMAN = {
    "정책": "policy", "보조금": "subsidy", "충전": "charging", "안전": "safety",
    "가격": "price", "배터리": "battery", "신차": "newmodel",
    "환경": "environment", "산업": "industry",
}


def load_monthly_edges():
    df = pd.read_csv(os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_edges.csv"))
    return df


def fig_discourse_evolution(news):
    kc.set_korean_font()
    cols = [f"frshare_{r}" for r in FRAME_ROMAN.values()]
    labels = list(FRAME_ROMAN.keys())
    x = pd.to_datetime(news["ym"] + "-01")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.stackplot(x, [news[c].values for c in cols], labels=labels, alpha=0.85)
    ax.set_title("전기차 담론 프레임 비중의 진화 (2019-2025)", fontsize=15, fontweight="bold")
    ax.set_ylabel("기사 중 프레임 등장 비중(중복 허용)")
    ax.legend(loc="upper left", ncol=5, fontsize=9)
    ax.margins(x=0)
    plt.tight_layout()
    p = os.path.join(kc.FIG_DIR, "discourse_evolution.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_indicators_timeline(kgf, nat):
    kc.set_korean_font()
    x = pd.to_datetime(kgf["ym"] + "-01")
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax1.plot(x, kgf["cent_safety"], color="#d62728", label="안전 프레임 중심성", linewidth=1.8)
    ax1.plot(x, kgf["salience_newmodel"], color="#1f77b4", label="신차 개체 현저성", linewidth=1.8)
    ax1.plot(x, kgf["safety_neg_intensity"], color="#ff7f0e",
             label="안전 부정 노출강도", linewidth=1.4, linestyle="--")
    ax1.set_ylabel("KG 담론 지표")
    ax1.set_xlabel("월")

    nat2 = nat[["ym", "ev_net_add"]].copy()
    nat2["dt"] = pd.to_datetime(nat2["ym"] + "-01")
    ax2 = ax1.twinx()
    ax2.plot(nat2["dt"], nat2["ev_net_add"], color="#2ca02c",
             label="전기차 등록 순증가(우축)", linewidth=1.2, alpha=0.5)
    ax2.set_ylabel("전기차 등록 순증가(대)")

    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=10)
    ax1.set_title("KG 담론 지표와 전기차 등록 순증가 타임라인", fontsize=15, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(kc.FIG_DIR, "kg_indicators_timeline.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def draw_event_subgraph(edges, ym, focus_nodes, title, fname, top_edges=40):
    """해당 월 edges에서 focus_nodes 이웃 서브그래프를 그린다."""
    kc.set_korean_font()
    sub = edges[edges["ym"] == ym]
    if sub.empty:
        return None

    G = nx.Graph()
    # focus 이웃 엣지 우선 수집
    focus_edges = sub[(sub["source"].isin(focus_nodes)) | (sub["target"].isin(focus_nodes))]
    focus_edges = focus_edges.sort_values("weight", ascending=False).head(top_edges)
    for _, r in focus_edges.iterrows():
        G.add_edge(r["source"], r["target"], weight=r["weight"], etype=r["edge_type"])
    # 이웃 노드 간 프레임 공출현도 일부 추가(맥락)
    nbrs = set(G.nodes())
    ctx = sub[(sub["edge_type"] == "FRAME_COOC") &
              (sub["source"].isin(nbrs)) & (sub["target"].isin(nbrs))]
    for _, r in ctx.iterrows():
        G.add_edge(r["source"], r["target"], weight=r["weight"], etype=r["edge_type"])
    if G.number_of_nodes() == 0:
        return None

    def ntype(n):
        if n.startswith("Frame:"):
            return "Frame"
        if n.startswith("Entity_company"):
            return "Company"
        if n.startswith("Entity_model"):
            return "Model"
        if n.startswith("Entity_event"):
            return "Event"
        if n.startswith("Entity_policy"):
            return "Policy"
        return "Other"

    colors = {"Frame": "#ffb703", "Company": "#8ecae6", "Model": "#90be6d",
              "Event": "#e63946", "Policy": "#b5838d", "Other": "#cccccc"}

    def label(n):
        return n.split(":", 1)[1] if ":" in n else n

    pos = nx.spring_layout(G, seed=42, k=0.9)
    fig, ax = plt.subplots(figsize=(14, 10))
    for t, col in colors.items():
        nodes = [n for n in G.nodes() if ntype(n) == t]
        if nodes:
            sizes = [900 + 120 * G.degree(n) for n in nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=col,
                                   node_size=sizes, edgecolors="#333", linewidths=0.8,
                                   ax=ax, label=t)
    ws = [G[u][v]["weight"] for u, v in G.edges()]
    mx = max(ws) if ws else 1
    nx.draw_networkx_edges(G, pos, width=[0.5 + 3 * (w / mx) for w in ws],
                           edge_color="#999", alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: label(n) for n in G.nodes()},
                            font_size=10, ax=ax)
    ax.legend(scatterpoints=1, fontsize=10, loc="upper right")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    p = os.path.join(kc.FIG_DIR, fname)
    plt.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    kc.ensure_dirs()
    news = pd.read_csv(kc.NEWS_MONTHLY_SIMPLE_CSV)
    kgf = pd.read_csv(kc.KG_MONTHLY_FEATURES_CSV)
    nat = pd.read_csv(kc.NATIONAL_CSV)
    edges = load_monthly_edges()

    outs = []
    outs.append(fig_discourse_evolution(news))
    outs.append(fig_indicators_timeline(kgf, nat))

    # 부정 사건 월: safety_neg_intensity 최대
    ym_safety = kgf.loc[kgf["safety_neg_intensity"].idxmax(), "ym"]
    p1 = draw_event_subgraph(
        edges, ym_safety,
        focus_nodes=["Entity_event:화재", "Entity_event:리콜", "Frame:안전"],
        title=f"안전 이슈 최고조 월({ym_safety}) 담론 서브그래프",
        fname=f"event_subgraph_{ym_safety}_safety.png")
    outs.append(p1)

    # 신차 현저성 최대 월
    ym_new = kgf.loc[kgf["salience_newmodel"].idxmax(), "ym"]
    p2 = draw_event_subgraph(
        edges, ym_new,
        focus_nodes=["Frame:신차", "Entity_company:현대차", "Entity_company:테슬라"],
        title=f"신차 현저성 최고 월({ym_new}) 담론 서브그래프",
        fname=f"event_subgraph_{ym_new}_newmodel.png")
    outs.append(p2)

    print(f"[19] safety 최고조 월={ym_safety}, 신차 최고 월={ym_new}")
    for o in outs:
        if o:
            print(f"[19] 저장: {os.path.relpath(o, kc.BASE_DIR)}")


if __name__ == "__main__":
    main()
