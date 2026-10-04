# -*- coding: utf-8 -*-
"""
감성분석 신뢰도 + 오분류 패턴 Knowledge Graph 생성

입력:
  - outputs/sentiment_review_sample_300.csv  (300건 표본, 200건 수작업 검토)
  - outputs/sentiment_mismatch_analysis.csv  (자동 vs 사람 라벨 불일치, analyze_sentiment_mismatches.py 산출)

출력 (모두 outputs/kg/):
  - nodes.csv            : 노드 목록
  - edges.csv            : 엣지 목록
  - kg_error_pattern.png : 정적 그림 (HitWord -> IssueType -> SuggestedRule)
  - kg_interactive.html  : pyvis 인터랙티브 전체 그래프
  - kg_build_summary.md  : 간단 통계 요약

KG 범위는 검증 표본(200~300건)에 한정된다. nodes.csv 는 28번 검진(B5)의 입력이다.
"""

import os
import re
from collections import Counter, defaultdict

import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
KG_DIR = os.path.join(OUT_DIR, "kg")

REVIEW_CSV = os.path.join(OUT_DIR, "sentiment_review_sample_300.csv")
MISMATCH_CSV = os.path.join(OUT_DIR, "sentiment_mismatch_analysis.csv")

os.makedirs(KG_DIR, exist_ok=True)


# ============================================================
# 한글 폰트 설정 (matplotlib)
# ============================================================
def set_korean_font():
    """윈도우 환경의 한글 폰트를 우선순위대로 시도해 설정한다."""
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim", "MS Gothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen:
        rcParams["font.family"] = chosen
        rcParams["font.sans-serif"] = [chosen] + list(rcParams.get("font.sans-serif", []))
        print(f"  [폰트] '{chosen}' 적용")
    else:
        print("  [폰트] 한글 폰트를 찾지 못해 기본 폰트를 사용합니다.")
    rcParams["axes.unicode_minus"] = False


# ============================================================
# 유틸: 콤마로 구분된 필드 파싱
# ============================================================
def split_multi(value):
    """'투자, 수요, 확대' 같은 콤마 구분 문자열을 정규화된 토큰 리스트로 변환."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []
    tokens = re.split(r"[,\n]", text)
    result = []
    for tok in tokens:
        t = tok.strip()
        if t:
            result.append(t)
    return result


def clean_scalar(value):
    """스칼라 값을 문자열로 정규화. 결측/빈값은 빈 문자열."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


# ============================================================
# STEP 1: 데이터 로드 및 결합
# ============================================================
def load_data():
    review = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
    mismatch = pd.read_csv(MISMATCH_CSV, dtype=str, keep_default_na=False)

    # 오분류 파일에서 sample_id -> (issue_type, suggested_rule) 매핑
    mismatch_info = {}
    for _, row in mismatch.iterrows():
        sid = clean_scalar(row.get("sample_id"))
        if sid:
            mismatch_info[sid] = {
                "issue_type": clean_scalar(row.get("issue_type")),
                "suggested_rule": clean_scalar(row.get("suggested_rule")),
            }
    return review, mismatch, mismatch_info


def derive_match_status(pred, human, relevance):
    """검토 결과로 기사별 match_status 파생."""
    if relevance == "무관":
        return "무관"
    if human == "":
        # 관련성은 입력됐지만 감정 미입력 → 미검토로 취급
        return "미검토"
    if pred == human:
        return "일치"
    return "오분류"


# ============================================================
# STEP 2: 그래프 구성
# ============================================================
def node_id(node_type, name):
    return f"{node_type}::{name}"


def build_graph(review, mismatch_info):
    G = nx.DiGraph()

    def add_node(ntype, name, **attrs):
        nid = node_id(ntype, name)
        if G.has_node(nid):
            # count 증가 (등장 빈도)
            G.nodes[nid]["count"] = G.nodes[nid].get("count", 1) + 1
            for k, v in attrs.items():
                if k not in G.nodes[nid] and v not in (None, ""):
                    G.nodes[nid][k] = v
        else:
            G.add_node(nid, node_type=ntype, label=name, count=1, **attrs)
        return nid

    def add_edge(src, dst, relation, **attrs):
        if G.has_edge(src, dst) and G[src][dst].get("relation") == relation:
            G[src][dst]["weight"] = G[src][dst].get("weight", 1) + 1
        else:
            G.add_edge(src, dst, relation=relation, weight=attrs.pop("weight", 1), **attrs)

    # HitWord -> IssueType 공출현 카운트 (집계 엣지용)
    hit_issue_counter = Counter()          # (hitword, polarity, issue_type) -> count
    issue_rule_pairs = set()

    for _, row in review.iterrows():
        sid = clean_scalar(row.get("sample_id"))
        if not sid:
            continue

        pred = clean_scalar(row.get("predicted_sentiment"))
        human = clean_scalar(row.get("human_sentiment"))
        relevance = clean_scalar(row.get("human_relevance"))
        match_status = derive_match_status(pred, human, relevance)

        info = mismatch_info.get(sid, {})
        issue_type = info.get("issue_type", "")
        suggested_rule = info.get("suggested_rule", "")

        # --- Article 노드 ---
        art = add_node(
            "Article", sid,
            title=clean_scalar(row.get("Title")),
            date=clean_scalar(row.get("Date")),
            year=clean_scalar(row.get("Year")),
            month=clean_scalar(row.get("Month")),
            url=clean_scalar(row.get("Url")),
            sentiment_score=clean_scalar(row.get("sentiment_score")),
            predicted_sentiment=pred,
            human_sentiment=human,
            human_relevance=relevance,
            match_status=match_status,
        )

        # --- Newspaper ---
        paper = clean_scalar(row.get("newspaper"))
        if paper:
            pid = add_node("Newspaper", paper)
            add_edge(art, pid, "PUBLISHED_IN")

        # --- Predicted / Human Sentiment ---
        if pred:
            sid_pred = add_node("Sentiment", pred)
            add_edge(art, sid_pred, "PREDICTED_AS")
        if human and relevance == "관련":
            sid_human = add_node("Sentiment", human)
            add_edge(art, sid_human, "HUMAN_LABELED_AS")

        # --- Topic ---
        topic_map = {
            "topic_policy": "정책",
            "topic_charging": "충전",
            "topic_safety": "안전",
            "topic_price": "가격",
            "topic_battery": "배터리",
            "topic_new_model": "신차",
        }
        for col, tname in topic_map.items():
            if clean_scalar(row.get(col)) == "1":
                tid = add_node("Topic", tname)
                add_edge(art, tid, "HAS_TOPIC")

        # --- EV Keyword ---
        for kw in split_multi(row.get("matched_ev_keywords")):
            kid = add_node("EVKeyword", kw)
            add_edge(art, kid, "MENTIONS")

        # --- Hit Words (positive / negative) ---
        pos_hits = split_multi(row.get("positive_hits"))
        neg_hits = split_multi(row.get("negative_hits"))
        for w in pos_hits:
            wid = add_node("HitWord", w, polarity="positive")
            add_edge(art, wid, "HAS_HIT", polarity="positive")
        for w in neg_hits:
            wid = add_node("HitWord", w, polarity="negative")
            add_edge(art, wid, "HAS_HIT", polarity="negative")

        # --- IssueType / SuggestedRule (오분류 기사만) ---
        if match_status == "오분류" and issue_type:
            iid = add_node("IssueType", issue_type)
            add_edge(art, iid, "HAS_ISSUE")

            if suggested_rule:
                rid = add_node("SuggestedRule", suggested_rule)
                add_edge(iid, rid, "SUGGESTS")
                issue_rule_pairs.add((issue_type, suggested_rule))

            # 오분류 기사에서 등장한 hit 단어 → 오류유형 공출현 집계
            for w in pos_hits:
                hit_issue_counter[(w, "positive", issue_type)] += 1
            for w in neg_hits:
                hit_issue_counter[(w, "negative", issue_type)] += 1

    # --- 집계 엣지: HitWord -CONTRIBUTES_TO-> IssueType ---
    for (w, polarity, issue_type), cnt in hit_issue_counter.items():
        wid = node_id("HitWord", w)
        iid = node_id("IssueType", issue_type)
        if G.has_node(wid) and G.has_node(iid):
            G.add_edge(wid, iid, relation="CONTRIBUTES_TO", weight=cnt, polarity=polarity)

    return G, hit_issue_counter


# ============================================================
# STEP 3: nodes.csv / edges.csv 저장
# ============================================================
def export_csv(G):
    node_rows = []
    for nid, data in G.nodes(data=True):
        row = {"node_id": nid}
        row.update(data)
        node_rows.append(row)
    nodes_df = pd.DataFrame(node_rows)
    # 컬럼 순서 정리
    front = ["node_id", "node_type", "label", "count"]
    cols = front + [c for c in nodes_df.columns if c not in front]
    nodes_df = nodes_df.reindex(columns=cols)
    nodes_path = os.path.join(KG_DIR, "nodes.csv")
    nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")

    edge_rows = []
    for src, dst, data in G.edges(data=True):
        row = {"source": src, "target": dst}
        row.update(data)
        edge_rows.append(row)
    edges_df = pd.DataFrame(edge_rows)
    front_e = ["source", "target", "relation", "weight"]
    cols_e = front_e + [c for c in edges_df.columns if c not in front_e]
    edges_df = edges_df.reindex(columns=cols_e)
    edges_path = os.path.join(KG_DIR, "edges.csv")
    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")

    return nodes_df, edges_df


# ============================================================
# STEP 4: 정적 그림 (HitWord -> IssueType -> SuggestedRule)
# ============================================================
def draw_error_pattern(G, hit_issue_counter, top_n_per_issue=6):
    set_korean_font()

    # 오류유형별 상위 hit 단어 선정
    issue_to_words = defaultdict(list)  # issue_type -> [(word, polarity, cnt)]
    for (w, polarity, issue_type), cnt in hit_issue_counter.items():
        issue_to_words[issue_type].append((w, polarity, cnt))

    SG = nx.DiGraph()
    for issue_type, words in issue_to_words.items():
        SG.add_node(node_id("IssueType", issue_type), node_type="IssueType", label=issue_type)
        top_words = sorted(words, key=lambda x: x[2], reverse=True)[:top_n_per_issue]
        for w, polarity, cnt in top_words:
            wid = node_id("HitWord", w)
            SG.add_node(wid, node_type="HitWord", label=w, polarity=polarity)
            SG.add_edge(wid, node_id("IssueType", issue_type), weight=cnt)

    # IssueType -> SuggestedRule
    for src, dst, data in G.edges(data=True):
        if data.get("relation") == "SUGGESTS":
            if SG.has_node(src):
                SG.add_node(dst, node_type="SuggestedRule",
                            label=G.nodes[dst].get("label", ""))
                SG.add_edge(src, dst, weight=data.get("weight", 1))

    if SG.number_of_nodes() == 0:
        print("[경고] 정적 그림용 서브그래프가 비어 있습니다.")
        return None

    # 레이어 배치: HitWord(왼쪽) -> IssueType(중앙) -> SuggestedRule(오른쪽)
    layers = {"HitWord": 0, "IssueType": 1, "SuggestedRule": 2}
    pos = {}
    col_nodes = defaultdict(list)
    for n, d in SG.nodes(data=True):
        col_nodes[d["node_type"]].append(n)
    for ntype, x in layers.items():
        nodes = sorted(col_nodes.get(ntype, []))
        k = len(nodes)
        for i, n in enumerate(nodes):
            y = 1.0 - (i / max(k - 1, 1))
            pos[n] = (x * 3.0, y * max(k, 1))

    color_map = {
        "HitWord": "#8ecae6",
        "IssueType": "#ffb703",
        "SuggestedRule": "#90be6d",
    }

    fig, ax = plt.subplots(figsize=(18, 12))

    for ntype, color in color_map.items():
        nodes = col_nodes.get(ntype, [])
        sizes = [1600 if ntype != "HitWord" else 900 for _ in nodes]
        nx.draw_networkx_nodes(SG, pos, nodelist=nodes, node_color=color,
                               node_size=sizes, ax=ax, edgecolors="#333333",
                               linewidths=0.8, alpha=0.95)

    # 엣지 (가중치 비례 굵기)
    weights = [SG[u][v].get("weight", 1) for u, v in SG.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.6 + 3.0 * (w / max_w) for w in weights]
    nx.draw_networkx_edges(SG, pos, ax=ax, width=edge_widths,
                           edge_color="#999999", arrows=True,
                           arrowsize=12, alpha=0.6,
                           connectionstyle="arc3,rad=0.05")

    labels = {n: d["label"] for n, d in SG.nodes(data=True)}
    nx.draw_networkx_labels(SG, pos, labels=labels, font_size=9, ax=ax)

    # 컬럼 헤더
    ax.text(0, max(len(col_nodes.get("HitWord", [])), 1) + 0.5,
            "감성 hit 단어", fontsize=14, fontweight="bold", ha="center")
    ax.text(3.0, max(len(col_nodes.get("IssueType", [])), 1) + 0.5,
            "오류 유형", fontsize=14, fontweight="bold", ha="center")
    ax.text(6.0, max(len(col_nodes.get("SuggestedRule", [])), 1) + 0.5,
            "보정 규칙 후보", fontsize=14, fontweight="bold", ha="center")

    ax.set_title("감정분석 오분류 패턴: 감성 hit 단어 → 오류 유형 → 보정 규칙",
                 fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    out_path = os.path.join(KG_DIR, "kg_error_pattern.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# STEP 5: 인터랙티브 HTML (pyvis)
# ============================================================
def draw_interactive(G):
    try:
        from pyvis.network import Network
    except ImportError:
        print("[경고] pyvis 미설치 → 인터랙티브 HTML 생략. 'pip install pyvis' 후 재실행.")
        return None

    color_map = {
        "Article": "#adb5bd",
        "Newspaper": "#f4a261",
        "Sentiment": "#e76f51",
        "Topic": "#2a9d8f",
        "EVKeyword": "#264653",
        "HitWord": "#8ecae6",
        "IssueType": "#ffb703",
        "SuggestedRule": "#90be6d",
    }

    net = Network(height="800px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    for nid, d in G.nodes(data=True):
        ntype = d.get("node_type", "")
        label = d.get("label", nid)
        count = d.get("count", 1)
        size = 10 + min(count, 30)
        title_bits = [f"type: {ntype}", f"label: {label}", f"count: {count}"]
        if ntype == "Article":
            title_bits += [
                f"predicted: {d.get('predicted_sentiment','')}",
                f"human: {d.get('human_sentiment','')}",
                f"match: {d.get('match_status','')}",
            ]
        net.add_node(nid, label=label, color=color_map.get(ntype, "#cccccc"),
                     size=size, title="\n".join(title_bits), group=ntype)

    for src, dst, d in G.edges(data=True):
        net.add_edge(src, dst, title=d.get("relation", ""),
                     value=d.get("weight", 1))

    net.set_options("""
    var options = {
      "interaction": {"hover": true, "tooltipDelay": 100},
      "physics": {"stabilization": {"iterations": 200}}
    }
    """)

    out_path = os.path.join(KG_DIR, "kg_interactive.html")
    net.write_html(out_path, notebook=False, open_browser=False)
    return out_path


# ============================================================
# STEP 6: 요약 통계
# ============================================================
def write_summary(G, review, mismatch, paths):
    type_counter = Counter(d["node_type"] for _, d in G.nodes(data=True))
    rel_counter = Counter(d["relation"] for _, _, d in G.edges(data=True))
    match_counter = Counter(
        d.get("match_status") for _, d in G.nodes(data=True)
        if d.get("node_type") == "Article"
    )

    lines = []
    lines.append("# 감정 신뢰도 Knowledge Graph 생성 요약\n")
    lines.append("## 입력 파일\n")
    lines.append(f"- 검토 표본: {len(review)} 행")
    lines.append(f"- 오분류 분석: {len(mismatch)} 행\n")

    lines.append("## 노드 수 (타입별)\n")
    lines.append("| 노드 타입 | 개수 |")
    lines.append("| --- | ---: |")
    for t, c in type_counter.most_common():
        lines.append(f"| {t} | {c} |")
    lines.append(f"| **합계** | **{G.number_of_nodes()}** |\n")

    lines.append("## 엣지 수 (관계별)\n")
    lines.append("| 관계 | 개수 |")
    lines.append("| --- | ---: |")
    for r, c in rel_counter.most_common():
        lines.append(f"| {r} | {c} |")
    lines.append(f"| **합계** | **{G.number_of_edges()}** |\n")

    lines.append("## 기사 match_status 분포\n")
    lines.append("| 상태 | 건수 |")
    lines.append("| --- | ---: |")
    for s, c in match_counter.most_common():
        lines.append(f"| {s} | {c} |")
    lines.append("")

    lines.append("## 생성 파일\n")
    for label, p in paths.items():
        if p:
            lines.append(f"- {label}: `{os.path.relpath(p, BASE_DIR)}`")
    lines.append("")

    summary_text = "\n".join(lines)
    out_path = os.path.join(KG_DIR, "kg_build_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # 콘솔 출력
    print(summary_text)
    return out_path


# ============================================================
# main
# ============================================================
def main():
    print("[1/6] 데이터 로드 중...")
    review, mismatch, mismatch_info = load_data()

    print("[2/6] 그래프 구성 중...")
    G, hit_issue_counter = build_graph(review, mismatch_info)
    print(f"  노드 {G.number_of_nodes()}개 / 엣지 {G.number_of_edges()}개")

    print("[3/6] nodes.csv / edges.csv 저장 중...")
    export_csv(G)

    print("[4/6] 정적 그림 생성 중...")
    png_path = draw_error_pattern(G, hit_issue_counter)

    print("[5/6] 인터랙티브 HTML 생성 중...")
    html_path = draw_interactive(G)

    print("[6/6] 요약 작성 중...")
    paths = {
        "nodes.csv": os.path.join(KG_DIR, "nodes.csv"),
        "edges.csv": os.path.join(KG_DIR, "edges.csv"),
        "정적 그림": png_path,
        "인터랙티브 HTML": html_path,
    }
    write_summary(G, review, mismatch, paths)

    print("\n완료. 산출물 경로:", KG_DIR)


if __name__ == "__main__":
    main()
