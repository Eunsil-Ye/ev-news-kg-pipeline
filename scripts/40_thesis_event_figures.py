# -*- coding: utf-8 -*-
"""
[40] 사건 탐지 그림 (2020-10 코나 화재·리콜, 2024-08 인천 화재)

입력:
  - data/processed/kg_monthly_features.csv
  - data/processed/news_tagged.csv          (월별 기사 수)
  - outputs/kg/monthly/kg_monthly_nodes.csv
  - outputs/kg/monthly/kg_monthly_edges.csv
  - outputs/kg/tables/tbl_breakpoints.csv
출력 (outputs/kg/figures/):
  - thesis_breakpoints_two_events.png   두 사건의 구조 전환점 (2패널)
  - thesis_event_subgraphs.png          두 사건 월의 부정 사건 하위그래프 (2패널)
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

EVENTS = [
    {"ym": "2024-08", "series": "safety_neg_intensity", "label": "안전 부정강도",
     "event": "인천 지하주차장 화재"},
    {"ym": "2020-10", "series": "neg_event_propagation", "label": "부정사건 전파",
     "event": "코나 일렉트릭 화재·리콜"},
]

NEG_EVENT_NODES = [f"Entity_event:{e}" for e in kc.NEGATIVE_EVENT_ENTITIES]
TOP_ENTITY_COOC = 10

TYPE_COLOR = {"Frame": "#ffb703", "Event": "#e63946", "company": "#8ecae6",
              "model": "#90be6d", "policy": "#b5838d", "event": "#f4a3a8"}
TYPE_NAME = {"Frame": "프레임", "Event": "부정 사건 개체", "company": "기업 개체",
             "model": "차종 개체", "policy": "정책 개체", "event": "기타 사건 개체"}


def month_article_counts():
    df = pd.read_csv(kc.NEWS_TAGGED_CSV, usecols=["ym"], dtype=str)
    return df["ym"].value_counts().to_dict()


def tautological_pairs():
    """별칭이 프레임 키워드와 겹치는 (개체 정규명, 프레임) 쌍."""
    aliases = {}
    for alias, (_, canon) in kc.ENTITY_DICT.items():
        aliases.setdefault(canon, []).append(alias)
    pairs = set()
    for canon, als in aliases.items():
        for frame, kws in kc.FRAME_KEYWORDS.items():
            if any(kw in a or a in kw for a in als for kw in kws):
                pairs.add((canon, frame))
    return pairs


def short(n):
    return n.split(":", 1)[1]


def node_kind(n):
    if n.startswith("Frame:"):
        return "Frame"
    if n in NEG_EVENT_NODES:
        return "Event"
    return n.split(":", 1)[0].replace("Entity_", "")


def fig_breakpoints(kgf, bp):
    kc.set_korean_font()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5))
    x = pd.to_datetime(kgf["ym"] + "-01")
    for ax, ev, tag in zip(axes, EVENTS, ["(a)", "(b)"]):
        s = ev["series"]
        row = bp.set_index("series").loc[s]
        bdt = pd.to_datetime(row["break_ym"] + "-01")
        y = kgf[s].astype(float)
        ax.plot(x, y, color="#333", linewidth=1.3)
        ax.hlines(row["pre_mean"], x.iloc[0], bdt, colors="#4c78a8", linewidth=2.2,
                  label=f"전환 이전 평균 {row['pre_mean']:.3f}")
        ax.hlines(row["post_mean"], bdt, x.iloc[-1], colors="#f58518", linewidth=2.2,
                  label=f"전환 이후 평균 {row['post_mean']:.3f}")
        ax.axvline(bdt, color="#d62728", linestyle="--", linewidth=1.4,
                   label=f"전환점 {row['break_ym']}")
        ax.set_title(f"{tag} {ev['label']} (supF={row['supF']:.1f}, 전후 변화 {row['shift_pct']:+.1f}%)",
                     fontsize=13, loc="left")
        ax.set_ylabel(ev["label"])
        ax.legend(loc="upper left", fontsize=9.5, frameon=True)
        ax.margins(x=0.01)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    p = os.path.join(kc.FIG_DIR, "thesis_breakpoints_two_events.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def build_event_graph(edges, nodes, ym):
    e = edges[edges["ym"] == ym]
    cnt = nodes[nodes["ym"] == ym].set_index("node")["count"].to_dict()
    focus = [n for n in NEG_EVENT_NODES if cnt.get(n, 0) > 0]
    inc = e[e["source"].isin(focus) | e["target"].isin(focus)]
    ef = inc[inc["edge_type"] == "ENTITY_FRAME"]
    ec = inc[inc["edge_type"] == "ENTITY_COOC"]
    ec_ev = ec[ec["source"].isin(focus) & ec["target"].isin(focus)]
    ec_other = ec.drop(ec_ev.index).sort_values("weight", ascending=False).head(TOP_ENTITY_COOC)
    G = nx.Graph()
    for f in kc.FRAME_KEYWORDS:
        G.add_node(f"Frame:{f}")
    for _, r in pd.concat([ef, ec_ev, ec_other]).iterrows():
        G.add_edge(r["source"], r["target"], weight=int(r["weight"]), etype=r["edge_type"])
    return G, cnt, focus


def layout(G, focus):
    pos = {}
    frames = list(kc.FRAME_KEYWORDS)
    for i, f in enumerate(frames):
        a = math.pi / 2 - 2 * math.pi * i / len(frames)
        pos[f"Frame:{f}"] = (1.0 * math.cos(a), 1.0 * math.sin(a))
    present = [n for n in focus if n in G]
    for i, n in enumerate(present):
        a = math.pi / 2 - 2 * math.pi * i / max(len(present), 1)
        r = 0.0 if len(present) == 1 else 0.45
        pos[n] = (r * math.cos(a), r * math.sin(a))
    others = sorted(n for n in G if n not in pos)
    for i, n in enumerate(others):
        a = math.pi / 2 + math.pi / max(len(others), 1) - 2 * math.pi * i / max(len(others), 1)
        pos[n] = (1.75 * math.cos(a), 1.75 * math.sin(a))
    return pos


def draw_panel(ax, G, cnt, focus, n_art, taut, title):
    pos = layout(G, focus)
    deg_frames = {n for u, v in G.edges() for n in (u, v) if n.startswith("Frame:")}
    for (u, v, d) in G.edges(data=True):
        share = d["weight"] / n_art
        width = 0.6 + 14 * share
        ent, fr = (u, v) if v.startswith("Frame:") else (v, u)
        is_taut = d["etype"] == "ENTITY_FRAME" and (short(ent), short(fr)) in taut
        color = "#c44e52" if d["etype"] == "ENTITY_FRAME" else "#7f7f7f"
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, ax=ax,
                               edge_color="#bbbbbb" if is_taut else color,
                               style="dashed" if is_taut else "solid",
                               alpha=0.55 if is_taut else 0.65)
    for n in G.nodes():
        kind = node_kind(n)
        if kind == "Frame":
            size = 1500
            alpha = 1.0 if n in deg_frames else 0.25
        else:
            size = min(500 + 6000 * cnt.get(n, 0) / n_art, 2600)
            alpha = 1.0
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_size=size, ax=ax, alpha=alpha,
                               node_color=TYPE_COLOR.get(kind, "#ccc"),
                               edgecolors="#333", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels={n: short(n) for n in G}, font_size=10.5, ax=ax,
                            font_family=plt.rcParams["font.family"])
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.15, 2.15)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_subgraphs(edges, nodes, kgf, art):
    kc.set_korean_font()
    taut = tautological_pairs()
    fig, axes = plt.subplots(1, 2, figsize=(18, 9.5))
    prop = kgf.set_index("ym")["neg_event_propagation"]
    for ax, ev, tag in zip(axes, EVENTS, ["(a)", "(b)"]):
        G, cnt, focus = build_event_graph(edges, nodes, ev["ym"])
        n_art = art.get(ev["ym"], 1)
        title = (f"{tag} {ev['ym']} {ev['event']}\n"
                 f"기사 {n_art:,}건, 부정사건 전파 {prop[ev['ym']]:.2f}")
        draw_panel(ax, G, cnt, focus, n_art, taut, title)

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=11,
                      markerfacecolor=TYPE_COLOR[k], markeredgecolor="#333", label=TYPE_NAME[k])
               for k in ["Frame", "Event", "company", "model"]]
    handles += [
        Line2D([0], [0], color="#c44e52", linewidth=3, label="개체–프레임 공출현"),
        Line2D([0], [0], color="#7f7f7f", linewidth=3, label="개체–개체 공출현"),
        Line2D([0], [0], color="#bbbbbb", linewidth=3, linestyle="--",
               label="사전 중첩 엣지(해석 제외)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(kc.FIG_DIR, "thesis_event_subgraphs.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    kc.ensure_dirs()
    kgf = pd.read_csv(kc.KG_MONTHLY_FEATURES_CSV).sort_values("ym").reset_index(drop=True)
    bp = pd.read_csv(os.path.join(kc.TABLE_DIR, "tbl_breakpoints.csv"))
    nodes = pd.read_csv(os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_nodes.csv"))
    edges = pd.read_csv(os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_edges.csv"))
    art = month_article_counts()
    for p in (fig_breakpoints(kgf, bp), fig_subgraphs(edges, nodes, kgf, art)):
        print(f"[40] 저장: {os.path.relpath(p, kc.BASE_DIR)}")


if __name__ == "__main__":
    main()
