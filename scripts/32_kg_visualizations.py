# -*- coding: utf-8 -*-
"""
[32] KG 도식 3종 생성

출력 (outputs/kg/figures/):
  - kg_corpus_overview.png    전체 기간 통합 KG
  - kg_schema.png             KG 스키마 + 처리 파이프라인
  - kg_clean_comparison.png   원본 vs 정제판 비교

입력:
  outputs/kg/corpus_nodes.csv, corpus_edges.csv
  outputs/kg/monthly/kg_monthly_nodes.csv, kg_monthly_edges.csv
  outputs/kg/clean/monthly/kg_monthly_nodes.csv, kg_monthly_edges.csv
  outputs/kg/tables/kg_audit_entity_frame_conditional.csv
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_clean_rules as kr
import kg_common as kc

CLEAN_MONTHLY_DIR = os.path.join(kc.KG_DIR, "clean", "monthly")
AUDIT_TAUT_CSV = os.path.join(kc.TABLE_DIR, "kg_audit_entity_frame_conditional.csv")

# 19번 스크립트와 동일한 팔레트를 써서 그림 간 색 의미를 일치시킨다.
NODE_COLORS = {
    "Frame": "#ffb703", "Company": "#8ecae6", "Model": "#90be6d",
    "Event": "#e63946", "Policy": "#b5838d",
}
NODE_LABELS = {
    "Frame": "Frame (담론 프레임)", "Company": "Entity/company (기업)",
    "Model": "Entity/model (차종)", "Event": "Entity/event (사건)",
    "Policy": "Entity/policy (정책)",
}
EDGE_COLORS = {
    "FRAME_COOC": "#e08a3c", "ENTITY_COOC": "#4d80b3", "ENTITY_FRAME": "#9aa3ad",
}
EDGE_LABELS = {
    "FRAME_COOC": "FRAME_COOC (프레임-프레임 공출현)",
    "ENTITY_COOC": "ENTITY_COOC (개체-개체 공출현)",
    "ENTITY_FRAME": "ENTITY_FRAME (개체-프레임 연결)",
}


def ntype(node: str) -> str:
    if node.startswith("Frame:"):
        return "Frame"
    if node.startswith("Entity_company"):
        return "Company"
    if node.startswith("Entity_model"):
        return "Model"
    if node.startswith("Entity_event"):
        return "Event"
    if node.startswith("Entity_policy"):
        return "Policy"
    return "Frame"


def short(node: str) -> str:
    return node.split(":", 1)[1] if ":" in node else node


def node_size(count: int, base: float = 260, mult: float = 13.0) -> float:
    """등장 기사 수를 면적에 반영하되 제곱근으로 눌러 편차를 완화.

    전체 기간 누적(수천~2만)과 월 단위(수십~수백)는 자릿수가 달라 배율을 분리한다.
    """
    return base + mult * math.sqrt(max(count, 1))


def draw_labels(G) -> dict:
    """짧은 이름이 유형 간에 겹치면(예: Frame:보조금 / Entity_policy:보조금) 유형을 병기."""
    dup = {}
    for n in G.nodes():
        dup.setdefault(short(n), []).append(n)
    out = {}
    for n in G.nodes():
        s = short(n)
        out[n] = f"{s}\n({ntype(n).lower()})" if len(dup[s]) > 1 else s
    return out


def draw_graph(ax, G, pos, counts, taut_pairs=None, font_size=10,
               base=260, mult=13.0):
    """공용 그래프 렌더러. taut_pairs 가 주어지면 동어반복 엣지를 빨간 점선으로 강조."""
    taut_pairs = taut_pairs or set()

    normal, taut = [], []
    for u, v, d in G.edges(data=True):
        if (u, v) in taut_pairs or (v, u) in taut_pairs:
            taut.append((u, v, d))
        else:
            normal.append((u, v, d))

    wmax = max([d["weight"] for _, _, d in G.edges(data=True)], default=1)
    for etype, color in EDGE_COLORS.items():
        sel = [(u, v) for u, v, d in normal if d["etype"] == etype]
        if not sel:
            continue
        widths = [0.6 + 3.4 * (G[u][v]["weight"] / wmax) for u, v in sel]
        nx.draw_networkx_edges(G, pos, edgelist=sel, width=widths,
                               edge_color=color, alpha=0.65, ax=ax)
    if taut:
        sel = [(u, v) for u, v, _ in taut]
        widths = [1.2 + 3.4 * (G[u][v]["weight"] / wmax) for u, v in sel]
        nx.draw_networkx_edges(G, pos, edgelist=sel, width=widths,
                               edge_color="#d90429", style="dashed", alpha=0.95, ax=ax)

    for t, color in NODE_COLORS.items():
        sel = [n for n in G.nodes() if ntype(n) == t]
        if not sel:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=sel, node_color=color,
            node_size=[node_size(counts.get(n, 1), base, mult) for n in sel],
            edgecolors="#333", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=draw_labels(G), font_size=font_size, ax=ax)
    ax.margins(0.10)
    ax.axis("off")


# ============================================================
# 도식 1. 전체 기간 통합 KG
# ============================================================
def corpus_scale():
    """그림 캡션에 넣을 규모 수치를 자료에서 직접 계산한다.

    분석기간 상수를 바꾸면 그림의 수치도 함께 바뀌도록 하드코딩하지 않는다.
    """
    raw = pd.read_csv(kc.ALL_NEWS, usecols=["Title"]) if os.path.exists(
        kc.ALL_NEWS) else None
    tagged = pd.read_csv(kc.NEWS_TAGGED_CSV, usecols=["ym"])
    ym = tagged["ym"].astype(str)
    in_p = ym.between(kc.ANALYSIS_START, kc.ANALYSIS_END)
    return {
        "n_raw": len(raw) if raw is not None else None,
        "n_ev": len(tagged),
        "n_period": int(in_p.sum()),
        "n_months": int(ym[in_p].nunique()),
    }


def fig_corpus_overview(top_entities: int = 15, per_node: int = 2, max_edges: int = 62):
    kc.set_korean_font()
    nodes = pd.read_csv(os.path.join(kc.KG_DIR, "corpus_nodes.csv"))
    edges = pd.read_csv(os.path.join(kc.KG_DIR, "corpus_edges.csv"))
    sc = corpus_scale()

    # 프레임 9종은 전부 남기고, 개체는 등장 상위 N개만 남긴다(hairball 방지).
    frames = nodes[nodes["node"].str.startswith("Frame:")]
    ents = nodes[nodes["node"].str.startswith("Entity_")].nlargest(top_entities, "count")
    keep = set(frames["node"]) | set(ents["node"])
    counts = dict(zip(nodes["node"], nodes["count"]))

    pool = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]

    # 전역 상위 엣지만 뽑으면 프레임끼리의 굵은 엣지가 자리를 다 차지해 개체 노드가
    # 통째로 사라진다. 노드마다 최강 엣지를 먼저 확보한 뒤 남은 자리를 상위 엣지로 채운다.
    idx = set()
    for n in keep:
        own = pool[(pool["source"] == n) | (pool["target"] == n)]
        idx.update(own.nlargest(per_node, "weight").index)
    remain = pool.drop(index=list(idx))
    if max_edges > len(idx):
        idx.update(remain.nlargest(max_edges - len(idx), "weight").index)
    e = pool.loc[sorted(idx)]

    G = nx.Graph()
    for _, r in e.iterrows():
        G.add_edge(r["source"], r["target"], weight=int(r["weight"]),
                   etype=r["edge_type"])

    pos = nx.spring_layout(G, seed=11, k=1.9, iterations=500, weight=None)

    fig, ax = plt.subplots(figsize=(15, 10.5))
    draw_graph(ax, G, pos, counts, font_size=11)

    ax.set_title(f"전기차 담론 지식그래프 전체 구조 "
                 f"({kc.ANALYSIS_START} ~ {kc.ANALYSIS_END}, 기사 {sc['n_period']:,}건)",
                 fontsize=16, fontweight="bold", pad=18)
    ax.text(0.5, -0.04,
            f"노드 크기 = 등장 기사 수 · 엣지 굵기 = 공출현 빈도 · 가독성을 위해 "
            f"프레임 9종 + 상위 개체 {top_entities}개(노드 {G.number_of_nodes()}개), "
            f"엣지 {G.number_of_edges()}개만 표시",
            transform=ax.transAxes, ha="center", fontsize=10, color="#555")

    leg1 = ax.legend(handles=[Line2D([0], [0], marker="o", color="none", label=NODE_LABELS[t],
                                     markerfacecolor=c, markeredgecolor="#333", markersize=13)
                              for t, c in NODE_COLORS.items()],
                     loc="upper left", fontsize=10, title="노드 유형", title_fontsize=10.5,
                     framealpha=0.9)
    ax.add_artist(leg1)
    # 우상단에 두면 '화재' 노드를 가리므로 노드 범례 바로 아래 왼쪽 가장자리에 붙인다.
    fig.canvas.draw()
    box = leg1.get_window_extent().transformed(ax.transAxes.inverted())
    ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=EDGE_LABELS[t])
                       for t, c in EDGE_COLORS.items()],
              loc="upper left", bbox_to_anchor=(box.x0, box.y0 - 0.01),
              borderaxespad=0, fontsize=10, title="엣지 유형", title_fontsize=10.5,
              framealpha=0.9)

    p = os.path.join(kc.FIG_DIR, "kg_corpus_overview.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[32] 통합 KG: 노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}")
    return p


# ============================================================
# 도식 2. KG 스키마 다이어그램
# ============================================================
def _box(ax, x, y, w, h, title, body, fc, fs_title=12.5, fs_body=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.10,rounding_size=0.18",
                                facecolor=fc, edgecolor="#333", linewidth=1.6, zorder=2))
    # 제목/본문을 상자 높이에 비례해 배치해야 큰 상자와 작은 상자 모두에서 여백이 맞는다.
    ty = y + h * (0.78 if body else 0.5)
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=fs_title, fontweight="bold", zorder=3)
    if body:
        ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center",
                fontsize=fs_body, zorder=3, linespacing=1.6, color="#222")


def _arrow(ax, xy_from, xy_to, style="-|>", color="#444", lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style, color=color,
                                 linewidth=lw, linestyle=ls, mutation_scale=16,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def fig_schema():
    kc.set_korean_font()
    sc = corpus_scale()
    fig, ax = plt.subplots(figsize=(15, 12))
    # 좌우 여백을 두어야 상자 바깥쪽 COOC 원호와 그 라벨이 잘리지 않는다.
    ax.set_xlim(-1.1, 11.1)
    # 아래쪽 여백은 병합·모형 단계 두 상자와 각주 두 줄을 담는다.
    ax.set_ylim(-2.05, 10)
    ax.axis("off")

    # --- 입력 ---
    _box(ax, 3.1, 9.00, 3.8, 0.85, "뉴스 기사 (Article)",
         f"전체 {sc['n_raw']:,}건 → 전기차 관련 {sc['n_ev']:,}건", "#e9ecef")
    _arrow(ax, (5.0, 9.00), (5.0, 8.55))

    # --- 처리 ---
    _box(ax, 2.0, 7.60, 6.0, 0.95, "규칙 기반 태깅 (kg_common.py)",
         "제목 + 본문 앞 1,000자 대상 · 감성 사전 / 프레임 키워드 / 개체 별칭 사전", "#dbe7f3")
    _arrow(ax, (3.8, 7.60), (2.4, 6.20), rad=0.12)
    _arrow(ax, (6.2, 7.60), (7.6, 6.20), rad=-0.12)

    # --- 노드 타입 ---
    _box(ax, 1.00, 4.20, 2.8, 2.0, "Frame 노드 (9종)",
         "정책 · 보조금 · 충전 · 안전\n가격 · 배터리 · 신차\n환경 · 산업", "#ffe2a8")
    _box(ax, 6.20, 4.20, 2.8, 2.0, "Entity 노드 (4범주)",
         "company (기업 20여)\nmodel (차종)\npolicy (정책)\nevent (사건)", "#cfe6f5")

    # --- 엣지 타입: 상자 사이(ENTITY_FRAME), 상자 바깥쪽(각 COOC) ---
    _arrow(ax, (6.20, 5.30), (3.80, 5.30), style="<|-|>",
           color=EDGE_COLORS["ENTITY_FRAME"], lw=2.4)
    ax.text(5.0, 5.52, "ENTITY_FRAME", ha="center", fontsize=11.5, fontweight="bold",
            color="#6b747d")
    ax.text(5.0, 4.98, "같은 기사에\n동시 등장", ha="center", va="top",
            fontsize=9.5, color="#666", linespacing=1.4)

    _arrow(ax, (1.00, 5.90), (1.00, 4.50), style="<|-|>",
           color=EDGE_COLORS["FRAME_COOC"], lw=2.4, rad=0.8)
    ax.text(0.22, 5.42, "FRAME_COOC", ha="right", va="center", fontsize=11.5,
            fontweight="bold", color=EDGE_COLORS["FRAME_COOC"])
    ax.text(0.22, 5.03, "프레임끼리 공출현", ha="right", va="center", fontsize=9,
            color="#777")

    _arrow(ax, (9.00, 5.90), (9.00, 4.50), style="<|-|>",
           color=EDGE_COLORS["ENTITY_COOC"], lw=2.4, rad=-0.8)
    ax.text(9.78, 5.42, "ENTITY_COOC", ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=EDGE_COLORS["ENTITY_COOC"])
    ax.text(9.78, 5.03, "개체끼리 공출현", ha="left", va="center", fontsize=9,
            color="#777")

    # --- 감성은 별도 노드가 아니라 노드 속성 ---
    _box(ax, 3.55, 2.75, 2.9, 1.10, "Sentiment (노드 속성)",
         "태깅 단계에서 기사별로 부여하여\n노드별 n_pos / n_neu / n_neg 로 누적", "#e8e0f0",
         fs_body=9)
    _arrow(ax, (3.55, 3.55), (2.40, 4.20), color="#8878a8", lw=1.4, ls="--", rad=0.12)
    _arrow(ax, (6.45, 3.55), (7.60, 4.20), color="#8878a8", lw=1.4, ls="--", rad=-0.12)

    # --- 산출 ---
    _arrow(ax, (4.30, 2.75), (2.60, 2.15), rad=0.12)
    ax.text(3.30, 2.60, "분석기간 필터 후\n월 단위 집계", ha="center", va="center",
            fontsize=9, color="#555", zorder=5, linespacing=1.4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    _box(ax, 0.50, 1.05, 4.1, 1.05, f"월별 그래프 스냅샷 {sc['n_months']}개",
         f"{kc.ANALYSIS_START} ~ {kc.ANALYSIS_END} · 기간 필터 후 {sc['n_period']:,}건\n"
         f"월 단위로 노드·엣지 재집계",
         "#d8f0dd", fs_body=9.5)
    _arrow(ax, (4.60, 1.58), (5.40, 1.58))
    _box(ax, 5.40, 1.05, 4.1, 1.05, "담론 특성 18종",
         "밀도 · 엔트로피 · 모듈러리티 · 중심성\n공출현률 · 현저성 · 전파도", "#d8f0dd",
         fs_body=9.5)

    # --- 정형 데이터 병합과 모형 단계 ---
    _arrow(ax, (7.45, 1.05), (7.45, 0.25))
    _box(ax, 5.40, -0.80, 4.1, 1.05, "[17] 정형 데이터 병합",
         f"EV 등록 순증가 · 충전기 · 총차량 등록\n"
         f"전국 {sc['n_months']}개월 / 지역패널 {sc['n_months']}개월 × 17개 시도",
         "#f2e6d0", fs_body=9)
    _arrow(ax, (5.40, -0.275), (4.60, -0.275))
    _box(ax, 0.50, -0.80, 4.1, 1.05, "[18~27] 통계 모형 · 검증",
         "Granger / VAR·FEVD · 패널 고정효과 · ElasticNetCV\n다중비교 보정 · BERT 벤치마크 · 정제판 재현",
         "#f2e6d0", fs_body=9)

    ax.text(5.0, -1.30,
            "엣지 가중치 = 두 노드가 같은 기사에 함께 등장한 횟수 (월 단위 누적)",
            ha="center", fontsize=10.5, color="#444")
    ax.text(5.0, -1.68,
            f"정제판에서는 사전 중첩으로 생긴 동어반복 ENTITY_FRAME 엣지 {len(kr.TAUTOLOGICAL_PAIRS)}개를 제외한다",
            ha="center", fontsize=9.5, color="#a03030")

    ax.set_title("지식그래프 스키마 및 처리 파이프라인",
                 fontsize=16, fontweight="bold", pad=16)
    p = os.path.join(kc.FIG_DIR, "kg_schema.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[32] 스키마 다이어그램 생성")
    return p


# ============================================================
# 도식 3. 원본 vs 정제판 비교
# ============================================================
def load_taut_pairs():
    df = pd.read_csv(AUDIT_TAUT_CSV)
    flag = df["tautological"].astype(str).str.lower() == "true"
    return set(zip(df.loc[flag, "entity"], df.loc[flag, "frame"]))


def month_graph(edges: pd.DataFrame, keep: set, show: set | None = None) -> nx.Graph:
    G = nx.Graph()
    for n in keep:
        G.add_node(n)
    sub = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]
    for _, r in sub.iterrows():
        pair = (r["source"], r["target"])
        if show is not None and pair not in show and pair[::-1] not in show:
            continue
        G.add_edge(r["source"], r["target"], weight=int(r["weight"]),
                   etype=r["edge_type"])
    return G


def fig_clean_comparison(ym: str = "2024-08", top_n: int = 9, n_taut_show: int = 5,
                         n_edges: int = 28):
    kc.set_korean_font()
    oe = pd.read_csv(os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_edges.csv"))
    ce = pd.read_csv(os.path.join(CLEAN_MONTHLY_DIR, "kg_monthly_edges.csv"))
    on = pd.read_csv(os.path.join(kc.KG_MONTHLY_DIR, "kg_monthly_nodes.csv"))
    cn = pd.read_csv(os.path.join(CLEAN_MONTHLY_DIR, "kg_monthly_nodes.csv"))
    oe, ce = oe[oe["ym"] == ym], ce[ce["ym"] == ym]
    on, cn = on[on["ym"] == ym], cn[cn["ym"] == ym]

    taut_pairs = load_taut_pairs()

    # 비교 가능성을 위해 '원본 기준 상위 노드'로 양쪽 노드 집합을 고정한다.
    keep = set(on.nlargest(top_n, "count")["node"])
    # 상위 노드만 뽑으면 동어반복 엣지가 거의 안 잡혀 이 그림의 목적이 사라진다.
    # 해당 월에 실제로 존재하는 동어반복 엣지 상위 몇 개의 양끝 노드를 명시적으로 추가한다.
    taut_mask = [(s, t) in taut_pairs for s, t in zip(oe["source"], oe["target"])]
    taut_month = oe[taut_mask].nlargest(n_taut_show, "weight")
    keep |= set(taut_month["source"]) | set(taut_month["target"])

    # 이 월의 엣지를 전부 그리면 700개가 넘어 판독이 불가능하다. 상위 엣지 + 동어반복
    # 엣지만 표시 대상으로 고정하고, 두 패널에 같은 표시 집합을 적용한다.
    pool = oe[oe["source"].isin(keep) & oe["target"].isin(keep)]
    show = set(map(tuple, pool.nlargest(n_edges, "weight")[["source", "target"]].values))
    show |= set(zip(taut_month["source"], taut_month["target"]))
    # 고립 노드가 생기면 spring layout 이 성분을 멀리 떼어놓아 본 군집이 뭉개진다.
    # 노드마다 최강 엣지를 하나씩 확보해 그래프를 연결 상태로 유지한다.
    for n in keep:
        own = pool[(pool["source"] == n) | (pool["target"] == n)]
        own = own[[(s, t) not in taut_pairs
                   for s, t in zip(own["source"], own["target"])]]
        if len(own):
            r = own.loc[own["weight"].idxmax()]
            show.add((r["source"], r["target"]))

    o_counts = dict(zip(on["node"], on["count"]))
    c_counts = dict(zip(cn["node"], cn["count"]))

    Go = month_graph(oe, keep, show)
    Gc = month_graph(ce, keep, show)

    # 동일 배치(layout)를 써야 두 패널의 차이가 눈에 들어온다.
    pos = nx.spring_layout(Go, seed=5, k=2.8, iterations=600, weight=None)

    def n_taut(G):
        return sum(1 for u, v in G.edges()
                   if (u, v) in taut_pairs or (v, u) in taut_pairs)

    # 표시 집합과 별개로, 해당 월 '전체' 규모도 각주에 병기한다.
    full_all_o, full_all_c = len(oe), len(ce)

    fig, axes = plt.subplots(1, 2, figsize=(19, 10.5))
    draw_graph(axes[0], Go, pos, o_counts, taut_pairs=taut_pairs, font_size=10.5,
               base=430, mult=50)
    axes[0].set_title(f"(a) 원본 KG — {ym}\n"
                      f"표시 엣지 {Go.number_of_edges()}개 중 "
                      f"동어반복 {n_taut(Go)}개 (빨간 점선)",
                      fontsize=13.5, fontweight="bold")

    draw_graph(axes[1], Gc, pos, c_counts, taut_pairs=taut_pairs, font_size=10.5,
               base=430, mult=50)
    axes[1].set_title(f"(b) 정제판 KG — {ym}\n"
                      f"표시 엣지 {Gc.number_of_edges()}개 · "
                      f"동어반복 {n_taut(Gc)}개",
                      fontsize=13.5, fontweight="bold")

    # 개체 계상 변화(C1 최장일치 · C2 단어경계)를 수치로 같이 보여준다.
    changed = []
    for n in sorted(keep):
        a, b = o_counts.get(n, 0), c_counts.get(n, 0)
        if a and b != a:
            changed.append(f"{short(n)} {a}→{b}({(b - a) / a * 100:+.0f}%)")
    note = " · ".join(changed[:6]) if changed else "이 월에는 개체 계상 변화 없음"

    fig.suptitle("KG 품질 검진: 원본 vs 정제판 (동일 노드 집합·동일 배치)",
                 fontsize=16.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.062,
             f"가독성을 위해 상위 엣지 {n_edges}개와 동어반복 엣지만 표시 "
             f"(이 월 전체 엣지는 {full_all_o:,}개 → 정제판 {full_all_c:,}개). "
             f"(b)에서 고립돼 보이는 개체는 표시 대상 엣지가 동어반복뿐이었을 뿐, "
             f"전체 그래프에서는 다른 엣지를 유지한다",
             ha="center", fontsize=10, color="#666")
    fig.text(0.5, 0.035,
             "정제 규칙  C1 최장일치 우선(아이오닉5를 아이오닉으로 이중계상 금지) · "
             "C2 단어경계(GM·리프·기아 오탐 차단) · C3 동어반복 엣지 제외",
             ha="center", fontsize=10.5, color="#444")
    fig.text(0.5, 0.010, f"개체 계상 변화: {note}",
             ha="center", fontsize=10, color="#a03030")

    handles = [Line2D([0], [0], marker="o", color="none", label=NODE_LABELS[t],
                      markerfacecolor=c, markeredgecolor="#333", markersize=11)
               for t, c in NODE_COLORS.items()]
    handles += [Line2D([0], [0], color=c, lw=3, label=EDGE_LABELS[t])
                for t, c in EDGE_COLORS.items()]
    handles.append(Line2D([0], [0], color="#d90429", lw=3, linestyle="--",
                          label="동어반복 엣지 P(frame|entity)=1.0 → 정제판에서 제외"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.082))

    plt.tight_layout(rect=(0, 0.175, 1, 0.955))
    p = os.path.join(kc.FIG_DIR, "kg_clean_comparison.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[32] 정제판 비교({ym}): 표시 엣지 {Go.number_of_edges()} -> "
          f"{Gc.number_of_edges()}, 동어반복 {n_taut(Go)} -> {n_taut(Gc)}, "
          f"월 전체 엣지 {full_all_o} -> {full_all_c}")
    return p


def main():
    kc.ensure_dirs()
    outs = [fig_corpus_overview(), fig_schema(), fig_clean_comparison()]
    for o in outs:
        print(f"[32] 저장: {os.path.relpath(o, kc.BASE_DIR)}")


if __name__ == "__main__":
    main()
