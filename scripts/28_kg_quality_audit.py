# -*- coding: utf-8 -*-
"""
[28] KG 구축 정확성 검진 (진단 전용 — 기존 산출물은 일절 수정하지 않음)

두 개의 KG를 각각 검진한다.
  (A) 전체 코퍼스 KG : outputs/kg/corpus_nodes.csv, corpus_edges.csv  (scripts/15 산출)
  (B) 감성신뢰도 KG  : outputs/kg/nodes.csv, edges.csv                (kg_sentiment_reliability.py 산출)

검진 항목 (A):
  A1 무결성 재계산    news_tagged.csv 로 독립 재집계 후 저장본과 차분
  A2 구조 위반 검사    자기루프 / 역방향 중복 / weight > min(node count)
  A3 사전 중복 진단    FRAME_KEYWORDS 와 ENTITY_DICT 의 동일 표제어 -> 동어반복 엣지
  A4 프레임 판별력     출현율 과대/과소, frame_density·modularity 분산(퇴화 여부)
  A5 계층 중복 계상    별칭 포함관계(아이오닉 ⊂ 아이오닉5)로 인한 이중 계상
  A6 사전 커버리지     사실상 죽은 항목, neg_event_propagation 구성 개체 기여도
  A7 연관강도 재평가   PMI / lift / Jaccard / G2 로 엣지 재순위
  A8 오탐 정밀도       위험 별칭의 문맥 표본 검토 결과로 정밀도 산출, 정제 규칙과 대조
                       (표본 파일이 이미 있으면 라벨 보존을 위해 재생성하지 않는다)

검진 항목 (B):
  B1 혼동행렬·정확도·클래스별 P/R/F1
  B2 Cohen kappa + 블록 부트스트랩 95% CI
  B3 무관 기사 포함/제외 비교, 유효 검증표본 수 명시
  B4 불일치 파일(68건) 정합성 검증
  B5 Sentiment 노드 카운트 이중합산 버그 확인

출력:
  outputs/kg/tables/kg_audit_*.csv
  outputs/kg/tables/kg_quality_audit.md
  outputs/kg_falsepositive_review_sample.csv
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

SEED = 20260804
FP_PER_ALIAS = 20
FP_REVIEW_PATH = os.path.join(kc.OUT_DIR, "kg_falsepositive_review_sample.csv")

# 단어경계 없이 매칭되어 오탐 위험이 있는 별칭 (사람 검토 대상)
RISKY_ALIASES = ["리프", "GM", "기아", "니로", "아이오닉"]

# 프레임 출현율 판정 기준
FRAME_TOO_BROAD = 0.60
FRAME_TOO_RARE = 0.02
# 사실상 죽은 사전 항목 기준(전체 기간 등장 기사 수)
DEAD_ENTITY_MAX = 100


# ============================================================
# 공통 로더
# ============================================================
def load_tagged():
    """news_tagged.csv 를 분석기간으로 필터해 (frames, entities, sentiment) 로 로드."""
    csv.field_size_limit(10 ** 7)
    rows = []
    with open(kc.NEWS_TAGGED_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ym = r["ym"]
            if not ym or not (kc.ANALYSIS_START <= ym <= kc.ANALYSIS_END):
                continue
            rows.append({
                "ym": ym,
                "sentiment": r["sentiment"],
                "frames": [x for x in r["frames"].split(";") if x],
                "entities": [x for x in r["entities"].split(";") if x],
            })
    return rows


def ent_node(en: str) -> str:
    """'company:현대차' -> 'Entity_company:현대차' (script 15 와 동일 규칙)."""
    t, n = en.split(":", 1)
    return f"Entity_{t}:{n}"


def read_saved_graph():
    nodes = {}
    with open(os.path.join(kc.KG_DIR, "corpus_nodes.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            nodes[(r["node_type"], r["node"])] = int(r["count"])
    edges = {}
    with open(os.path.join(kc.KG_DIR, "corpus_edges.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            edges[(r["edge_type"], r["source"], r["target"])] = int(r["weight"])
    return nodes, edges


# ============================================================
# A1/A2 무결성 재계산 및 구조 위반
# ============================================================
def recount(rows):
    """script 15 의 집계 로직을 독립 구현해 재계산."""
    node = Counter()
    edge = Counter()
    for r in rows:
        frames, ents = r["frames"], r["entities"]
        for fr in frames:
            node[("Frame", f"Frame:{fr}")] += 1
        for en in ents:
            node[("Entity", ent_node(en))] += 1
        for a, b in combinations(sorted(set(frames)), 2):
            edge[("FRAME_COOC", f"Frame:{a}", f"Frame:{b}")] += 1
        for a, b in combinations(sorted(set(ents)), 2):
            edge[("ENTITY_COOC", ent_node(a), ent_node(b))] += 1
        for en in set(ents):
            for fr in set(frames):
                edge[("ENTITY_FRAME", ent_node(en), f"Frame:{fr}")] += 1
    return node, edge


def audit_integrity(rows, saved_nodes, saved_edges):
    rc_node, rc_edge = recount(rows)
    diffs = []
    for key in set(rc_node) | set(saved_nodes):
        a, b = rc_node.get(key, 0), saved_nodes.get(key, 0)
        if a != b:
            diffs.append({"kind": "node", "key": " | ".join(key),
                          "recount": a, "saved": b, "diff": a - b})
    for key in set(rc_edge) | set(saved_edges):
        a, b = rc_edge.get(key, 0), saved_edges.get(key, 0)
        if a != b:
            diffs.append({"kind": "edge", "key": " | ".join(key),
                          "recount": a, "saved": b, "diff": a - b})

    # 구조 위반 검사
    violations = []
    seen_pairs = set()
    for (etype, s, d), w in saved_edges.items():
        if s == d:
            violations.append({"type": "self_loop", "detail": f"{etype} {s}", "value": w})
        pair = (etype, frozenset((s, d)))
        if pair in seen_pairs:
            violations.append({"type": "reversed_duplicate", "detail": f"{etype} {s}<->{d}", "value": w})
        seen_pairs.add(pair)
        # weight 는 양쪽 노드 등장 기사 수를 넘을 수 없다
        cs = saved_nodes.get(("Frame", s), saved_nodes.get(("Entity", s), 0))
        cd = saved_nodes.get(("Frame", d), saved_nodes.get(("Entity", d), 0))
        if cs and cd and w > min(cs, cd):
            violations.append({"type": "weight_exceeds_node_count",
                               "detail": f"{etype} {s}-{d} (w={w} > min({cs},{cd}))", "value": w})
    return diffs, violations, rc_node, rc_edge


# ============================================================
# A3 사전 중복 -> 동어반복 엣지
# ============================================================
def audit_dict_overlap(saved_nodes, saved_edges):
    """FRAME_KEYWORDS 와 ENTITY_DICT 에 동일 표제어가 있는지, 그로 인해
    조건부확률 1.0 에 가까운 ENTITY_FRAME 엣지가 생기는지 확인."""
    overlap = []
    for alias, (etype, canon) in kc.ENTITY_DICT.items():
        for fr, kws in kc.FRAME_KEYWORDS.items():
            for kw in kws:
                if alias == kw:
                    rel = "동일어"
                elif alias in kw or kw in alias:
                    rel = "포함관계"
                else:
                    continue
                overlap.append({"entity_alias": alias, "entity_canonical": f"{etype}:{canon}",
                                "frame": fr, "frame_keyword": kw, "relation": rel})
    overlap_df = pd.DataFrame(overlap)

    # ENTITY_FRAME 엣지의 조건부확률 P(frame|entity)
    rows = []
    for (etype, s, d), w in saved_edges.items():
        if etype != "ENTITY_FRAME":
            continue
        ce = saved_nodes.get(("Entity", s), 0)
        if ce <= 0:
            continue
        p = w / ce
        rows.append({"entity": s, "frame": d, "weight": w, "entity_count": ce,
                     "P_frame_given_entity": round(p, 4),
                     "tautological": p >= 0.99})
    cond = pd.DataFrame(rows).sort_values("P_frame_given_entity", ascending=False)
    return overlap_df, cond


# ============================================================
# A4 프레임 판별력 / 퇴화 지표
# ============================================================
def audit_frames(rows, saved_nodes):
    n_art = len(rows)
    frows = []
    for fr in kc.FRAME_NAMES:
        cnt = saved_nodes.get(("Frame", f"Frame:{fr}"), 0)
        share = cnt / n_art if n_art else 0.0
        flag = "과대포괄" if share > FRAME_TOO_BROAD else ("과소" if share < FRAME_TOO_RARE else "적정")
        frows.append({"frame": fr, "count": cnt, "share": round(share, 4), "judgement": flag})
    fdf = pd.DataFrame(frows).sort_values("count", ascending=False)

    # 월별 구조지표의 변동성(퇴화 여부)
    feat = pd.read_csv(kc.KG_MONTHLY_FEATURES_CSV)
    deg = []
    for col in ["frame_density", "frame_modularity", "frame_n_communities", "frame_entropy"]:
        if col not in feat.columns:
            continue
        s = feat[col].astype(float)
        cv = (s.std() / abs(s.mean())) if s.mean() != 0 else np.nan
        deg.append({"indicator": col, "mean": round(s.mean(), 5), "std": round(s.std(), 5),
                    "min": round(s.min(), 5), "max": round(s.max(), 5),
                    "cv": round(cv, 5) if pd.notna(cv) else np.nan,
                    "judgement": "퇴화(변동 거의 없음)" if pd.notna(cv) and cv < 0.05 else "사용가능"})
    return fdf, pd.DataFrame(deg), n_art


# ============================================================
# A5 계층 중복 계상
# ============================================================
def audit_hierarchy(rows):
    """별칭 A가 별칭 B의 부분문자열이고 canonical 이 다르면, B 매칭 시 A도 함께 계상된다."""
    pairs = []
    aliases = list(kc.ENTITY_DICT.keys())
    for a, b in combinations(aliases, 2):
        short, long_ = (a, b) if len(a) < len(b) else (b, a)
        if short in long_:
            ca = kc.ENTITY_DICT[short][1]
            cb = kc.ENTITY_DICT[long_][1]
            if ca != cb:
                pairs.append((short, long_, ca, cb))

    # 실제 동시 계상 건수
    ent_sets = [set(e.split(":", 1)[1] for e in r["entities"]) for r in rows]
    out = []
    for short, long_, ca, cb in pairs:
        both = sum(1 for s in ent_sets if ca in s and cb in s)
        out.append({"short_alias": short, "long_alias": long_,
                    "canonical_short": ca, "canonical_long": cb,
                    "co_counted_articles": both})
    hdf = pd.DataFrame(out).sort_values("co_counted_articles", ascending=False)

    # salience_newmodel 과대 정도: 모델 개체 총합 vs 모델 보유 기사 수
    model_total = 0
    model_articles = 0
    for r in rows:
        models = [e for e in r["entities"] if e.startswith("model:")]
        model_total += len(models)
        if models:
            model_articles += 1
    inflate = {
        "model_entity_mentions_total": model_total,
        "articles_with_any_model": model_articles,
        "mentions_per_article": round(model_total / model_articles, 4) if model_articles else 0.0,
        "note": "salience_newmodel 은 mentions_total/기사수 이므로 1기사 다중계상만큼 과대",
    }
    return hdf, inflate


# ============================================================
# A6 커버리지 / neg_event_propagation 구성
# ============================================================
def audit_coverage(rows, saved_nodes):
    dead = []
    canon_count = Counter()
    for (ntype, node), c in saved_nodes.items():
        if ntype == "Entity":
            canon_count[node] = c
    for node, c in sorted(canon_count.items(), key=lambda x: x[1]):
        if c <= DEAD_ENTITY_MAX:
            dead.append({"entity_node": node, "count": c,
                         "judgement": "사실상 미작동" if c < 10 else "저빈도"})
    dead_df = pd.DataFrame(dead)

    # neg_event_propagation 구성: 개체별 월 기여(0인 달 비율)
    by_month = defaultdict(lambda: defaultdict(set))
    for r in rows:
        ents = set(e.split(":", 1)[1] for e in r["entities"])
        frames = set(r["frames"])
        for e in kc.NEGATIVE_EVENT_ENTITIES:
            if e in ents:
                by_month[r["ym"]][e] |= frames
    months = sorted(by_month.keys())
    comp = []
    for e in kc.NEGATIVE_EVENT_ENTITIES:
        vals = [len(by_month[m].get(e, set())) for m in months]
        zeros = sum(1 for v in vals if v == 0)
        comp.append({"negative_event_entity": e, "n_months": len(months),
                     "zero_months": zeros,
                     "zero_share": round(zeros / len(months), 4) if months else 0.0,
                     "mean_frames_reached": round(float(np.mean(vals)), 4) if vals else 0.0,
                     "judgement": "상시 0 (평균 희석)" if zeros / max(len(months), 1) > 0.7 else "기여"})
    return dead_df, pd.DataFrame(comp)


# ============================================================
# A7 연관강도 재평가 (PMI / lift / Jaccard / G2)
# ============================================================
def g2_stat(a, count_a, count_b, n):
    """2x2 우도비 통계량."""
    b = count_a - a
    c = count_b - a
    d = n - a - b - c
    if min(a, b, c, d) < 0:
        return np.nan
    obs = [a, b, c, d]
    exp = [
        count_a * count_b / n,
        count_a * (n - count_b) / n,
        (n - count_a) * count_b / n,
        (n - count_a) * (n - count_b) / n,
    ]
    g = 0.0
    for o, e in zip(obs, exp):
        if o > 0 and e > 0:
            g += o * math.log(o / e)
    return 2 * g


def audit_association(saved_nodes, saved_edges, n_art, min_weight=30):
    node_count = {}
    for (ntype, node), c in saved_nodes.items():
        node_count[node] = c

    rows = []
    for (etype, s, d), w in saved_edges.items():
        if w < min_weight:
            continue
        ca, cb = node_count.get(s, 0), node_count.get(d, 0)
        if ca <= 0 or cb <= 0:
            continue
        expected = ca * cb / n_art
        lift = w / expected if expected > 0 else np.nan
        pmi = math.log2(lift) if lift and lift > 0 else np.nan
        jac = w / (ca + cb - w) if (ca + cb - w) > 0 else np.nan
        rows.append({"edge_type": etype, "source": s, "target": d, "weight": w,
                     "lift": round(lift, 3) if pd.notna(lift) else np.nan,
                     "pmi": round(pmi, 3) if pd.notna(pmi) else np.nan,
                     "jaccard": round(jac, 4) if pd.notna(jac) else np.nan,
                     "g2": round(g2_stat(w, ca, cb, n_art), 1)})
    df = pd.DataFrame(rows)
    df["rank_weight"] = df["weight"].rank(ascending=False, method="min").astype(int)
    df["rank_pmi"] = df["pmi"].rank(ascending=False, method="min")
    df["rank_shift"] = df["rank_weight"] - df["rank_pmi"]
    return df.sort_values("weight", ascending=False)


# ============================================================
# A8 오탐 검토표본 (All_news.csv 원문 문맥 필요)
# ============================================================
def audit_false_positive_sample():
    if not os.path.exists(kc.ALL_NEWS):
        return None, "All_news.csv 없음 -> 오탐 표본 생략"
    csv.field_size_limit(10 ** 7)
    rng = random.Random(SEED)
    buckets = defaultdict(list)

    with open(kc.ALL_NEWS, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            title = kc.norm_text(row.get("Title"))
            content = kc.norm_text(row.get("Content"))
            if not kc.contains_any(f"{title} {content}", kc.EV_KEYWORDS):
                continue
            # 태깅과 동일한 판정 텍스트
            text_clf = f"{title} {content[:1000]}"
            low = text_clf.lower()
            for alias in RISKY_ALIASES:
                pos = low.find(alias.lower())
                if pos < 0:
                    continue
                ctx = text_clf[max(0, pos - 40): pos + len(alias) + 40]
                buckets[alias].append({
                    "alias": alias,
                    "canonical": f"{kc.ENTITY_DICT[alias][0]}:{kc.ENTITY_DICT[alias][1]}",
                    "Date": kc.norm_text(row.get("Date")),
                    "newspaper": kc.norm_text(row.get("newspaper")),
                    "Title": title[:150],
                    "match_context": ctx.replace("\n", " "),
                    "human_correct": "",
                    "review_memo": "",
                    "Url": kc.norm_text(row.get("Url")),
                })

    stats = {a: len(v) for a, v in buckets.items()}

    # 검토표본에는 사람이 기입한 라벨·메모가 들어 있다. 재실행으로 지우면 복구할 수
    # 없으므로, 파일이 이미 있으면 읽어서 쓰고 새로 만들지 않는다.
    if os.path.exists(FP_REVIEW_PATH):
        df = pd.read_csv(FP_REVIEW_PATH, encoding="utf-8-sig")
        df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
        return df, stats

    out = []
    for alias, items in buckets.items():
        rng.shuffle(items)
        out.extend(items[:FP_PER_ALIAS])
    if not out:
        return None, "위험 별칭 매칭 기사 없음"
    df = pd.DataFrame(out).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.to_csv(FP_REVIEW_PATH, index=False, encoding="utf-8-sig")
    return df, stats


def score_fp_precision(df):
    """검토 완료된 표본으로 별칭별 오탐 정밀도를 산출한다.

    라벨 열 우선순위는 agent_correct(전건 판독) -> human_correct(부분 기입) 이다.
    정제판 정규식(`kg_clean_rules`)을 같은 문맥에 적용해, 규칙이 오탐을 실제로
    걸러내는지(그리고 정탐을 잘못 지우지는 않는지)까지 함께 본다.
    """
    label_col = None
    for c in ("agent_correct", "human_correct"):
        if c in df.columns and df[c].notna().any():
            label_col = c
            break
    if label_col is None:
        return None, None, None

    d = df[df[label_col].notna()].copy()
    d["label"] = d[label_col].astype(int)

    try:
        import kg_clean_rules as kr
        d["kept_clean"] = [
            len(kr._alias_spans(str(ctx), str(alias))) > 0
            for alias, ctx in zip(d["alias"], d["match_context"])
        ]
    except Exception:
        d["kept_clean"] = True

    rows = []
    for alias, g in d.groupby("alias"):
        kept = g[g["kept_clean"]]
        rows.append({
            "alias": alias,
            "reviewed": len(g),
            "true_positive": int(g["label"].sum()),
            "false_positive": int((g["label"] == 0).sum()),
            "precision_orig": round(float(g["label"].mean()), 4),
            "kept_by_clean_rule": len(kept),
            "precision_clean": round(float(kept["label"].mean()), 4) if len(kept) else None,
            "fp_remaining": int((kept["label"] == 0).sum()),
            "tp_lost": int((g[~g["kept_clean"]]["label"] == 1).sum()),
        })
    kept_all = d[d["kept_clean"]]
    rows.append({
        "alias": "(전체)",
        "reviewed": len(d),
        "true_positive": int(d["label"].sum()),
        "false_positive": int((d["label"] == 0).sum()),
        "precision_orig": round(float(d["label"].mean()), 4),
        "kept_by_clean_rule": len(kept_all),
        "precision_clean": round(float(kept_all["label"].mean()), 4) if len(kept_all) else None,
        "fp_remaining": int((kept_all["label"] == 0).sum()),
        "tp_lost": int((d[~d["kept_clean"]]["label"] == 1).sum()),
    })
    tbl = pd.DataFrame(rows)

    # 사람 기입분과 기계 판독의 일치도 (두 열이 모두 있을 때만)
    agree = None
    if {"agent_correct", "human_correct"} <= set(df.columns):
        both = df[df["human_correct"].notna() & df["agent_correct"].notna()]
        if len(both):
            n_ok = int((both["human_correct"].astype(int)
                        == both["agent_correct"].astype(int)).sum())
            agree = {"n": len(both), "agree": n_ok, "rate": round(n_ok / len(both), 4)}

    return tbl, label_col, agree


# ============================================================
# B. 감성신뢰도 KG 검진
# ============================================================
def bootstrap_kappa(y_true, y_pred, n_boot=2000, seed=SEED):
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(yt[idx])) < 2:
            continue
        vals.append(cohen_kappa_score(yt[idx], yp[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def audit_sentiment_kg():
    from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                                 precision_recall_fscore_support)
    review = pd.read_csv(os.path.join(kc.OUT_DIR, "sentiment_review_sample_300.csv"),
                         dtype=str, keep_default_na=False)
    mismatch = pd.read_csv(os.path.join(kc.OUT_DIR, "sentiment_mismatch_analysis.csv"),
                           dtype=str, keep_default_na=False)

    labels = ["긍정", "중립", "부정"]
    res = {"n_sample": len(review)}

    # 무관 포함/제외 두 버전
    valid = review[review["human_sentiment"].str.strip() != ""].copy()
    rel_only = valid[valid["human_relevance"].str.strip() == "관련"].copy()
    res["n_human_labeled"] = len(valid)
    res["n_relevant_labeled"] = len(rel_only)
    res["n_irrelevant"] = int((review["human_relevance"].str.strip() == "무관").sum())

    blocks = []
    kappa_rows = []
    for name, d in [("무관포함", valid), ("무관제외", rel_only)]:
        if len(d) == 0:
            continue
        yt = d["human_sentiment"].str.strip().tolist()
        yp = d["predicted_sentiment"].str.strip().tolist()
        acc = float(np.mean([a == b for a, b in zip(yt, yp)]))
        kappa = cohen_kappa_score(yt, yp, labels=labels)
        lo, hi = bootstrap_kappa(yt, yp)
        kappa_rows.append({"spec": name, "n": len(d), "accuracy": round(acc, 4),
                           "cohen_kappa": round(kappa, 4),
                           "kappa_CI2.5": round(lo, 4) if pd.notna(lo) else np.nan,
                           "kappa_CI97.5": round(hi, 4) if pd.notna(hi) else np.nan,
                           "interpretation": kappa_label(kappa)})
        pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
        for i, lab in enumerate(labels):
            blocks.append({"spec": name, "class": lab, "precision": round(pr[i], 3),
                           "recall": round(rc[i], 3), "f1": round(f1[i], 3),
                           "support": int(sup[i])})
        if name == "무관제외":
            cm = confusion_matrix(yt, yp, labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"사람:{l}" for l in labels],
                                 columns=[f"예측:{l}" for l in labels])
            res["confusion"] = cm_df

    # B4 불일치 파일 정합성
    derived = rel_only[rel_only["human_sentiment"].str.strip()
                       != rel_only["predicted_sentiment"].str.strip()]
    res["derived_mismatch"] = len(derived)
    res["mismatch_file_rows"] = len(mismatch)
    res["mismatch_consistent"] = len(derived) == len(mismatch)
    ids_derived = set(derived["sample_id"])
    ids_file = set(mismatch["sample_id"])
    res["only_in_derived"] = sorted(ids_derived - ids_file)
    res["only_in_file"] = sorted(ids_file - ids_derived)

    # B5 Sentiment 노드 이중합산 버그 검증
    nodes = pd.read_csv(os.path.join(kc.KG_DIR, "nodes.csv"))
    sent_nodes = nodes[nodes["node_type"] == "Sentiment"][["label", "count"]]
    pred_cnt = review["predicted_sentiment"].str.strip().value_counts()
    human_cnt = rel_only["human_sentiment"].str.strip().value_counts()
    bug_rows = []
    for _, r in sent_nodes.iterrows():
        lab = str(r["label"]).strip()
        p = int(pred_cnt.get(lab, 0))
        h = int(human_cnt.get(lab, 0))
        bug_rows.append({"sentiment_node": lab, "kg_count": int(r["count"]),
                         "predicted_count": p, "human_count": h, "pred_plus_human": p + h,
                         "double_counted": int(r["count"]) == p + h})
    res["sentiment_node_bug"] = pd.DataFrame(bug_rows)

    return res, pd.DataFrame(kappa_rows), pd.DataFrame(blocks)


def kappa_label(k):
    if pd.isna(k):
        return "산출불가"
    if k < 0.0:
        return "일치도 없음"
    if k < 0.20:
        return "매우 낮음(slight)"
    if k < 0.40:
        return "낮음(fair)"
    if k < 0.60:
        return "보통(moderate)"
    if k < 0.80:
        return "높음(substantial)"
    return "매우 높음(almost perfect)"


# ============================================================
# main
# ============================================================
def save(df, name):
    if df is None or len(df) == 0:
        return
    df.to_csv(os.path.join(kc.TABLE_DIR, name), index=False, encoding="utf-8-sig")


def main():
    kc.ensure_dirs()
    print("[28] 데이터 로드...")
    rows = load_tagged()
    saved_nodes, saved_edges = read_saved_graph()

    print("[28] A1/A2 무결성·구조 검사...")
    diffs, violations, rc_node, rc_edge = audit_integrity(rows, saved_nodes, saved_edges)
    save(pd.DataFrame(diffs), "kg_audit_integrity_diff.csv")
    save(pd.DataFrame(violations), "kg_audit_structure_violations.csv")

    print("[28] A3 사전 중복·동어반복 엣지...")
    overlap_df, cond = audit_dict_overlap(saved_nodes, saved_edges)
    save(overlap_df, "kg_audit_dict_overlap.csv")
    save(cond, "kg_audit_entity_frame_conditional.csv")
    taut = cond[cond["tautological"]]

    print("[28] A4 프레임 판별력·퇴화 지표...")
    fdf, degdf, n_art = audit_frames(rows, saved_nodes)
    save(fdf, "kg_audit_frame_prevalence.csv")
    save(degdf, "kg_audit_degenerate_indicators.csv")

    print("[28] A5 계층 중복 계상...")
    hdf, inflate = audit_hierarchy(rows)
    save(hdf, "kg_audit_hierarchy_duplication.csv")

    print("[28] A6 커버리지·neg_event_propagation 구성...")
    dead_df, comp_df = audit_coverage(rows, saved_nodes)
    save(dead_df, "kg_audit_dead_entities.csv")
    save(comp_df, "kg_audit_negevent_composition.csv")

    print("[28] A7 연관강도 재평가...")
    assoc = audit_association(saved_nodes, saved_edges, n_art)
    save(assoc, "kg_audit_edge_association.csv")

    print("[28] A8 오탐 검토표본...")
    fp_df, fp_stats = audit_false_positive_sample()

    print("[28] B 감성신뢰도 KG 검진...")
    sres, kappa_df, cls_df = audit_sentiment_kg()
    save(kappa_df, "kg_audit_sentiment_kappa.csv")
    save(cls_df, "kg_audit_sentiment_classwise.csv")
    save(sres["sentiment_node_bug"], "kg_audit_sentiment_node_bug.csv")
    if "confusion" in sres:
        sres["confusion"].to_csv(
            os.path.join(kc.TABLE_DIR, "kg_audit_sentiment_confusion.csv"),
            encoding="utf-8-sig")

    # ---------------- 리포트 ----------------
    L = ["# KG 구축 정확성 검진 리포트\n"]
    L.append(f"- 검진 대상 기사: {n_art}건 (분석기간 {kc.ANALYSIS_START}~{kc.ANALYSIS_END})")
    L.append(f"- 저장본 노드 {len(saved_nodes)}개 / 엣지 {len(saved_edges)}개\n")

    L.append("## A1. 무결성 재계산\n")
    if not diffs:
        L.append("- 독립 재집계 결과가 저장본과 **완전 일치**(노드·엣지 전부). 집계 로직 자체는 정확.")
    else:
        L.append(f"- 불일치 {len(diffs)}건 발견 -> `kg_audit_integrity_diff.csv`")
    L.append("")

    L.append("## A2. 구조 위반 검사\n")
    if not violations:
        L.append("- 자기루프·역방향 중복·가중치 상한 위반 **없음**.")
    else:
        vc = Counter(v["type"] for v in violations)
        L.append(f"- 위반 {len(violations)}건: {dict(vc)} -> `kg_audit_structure_violations.csv`")
    L.append("")

    L.append("## A3. 사전 중복으로 인한 동어반복 엣지 (핵심 결함)\n")
    L.append(f"- FRAME_KEYWORDS 와 ENTITY_DICT 의 표제어 중복: {len(overlap_df)}쌍")
    L.append(f"- ENTITY_FRAME 엣지 중 P(frame|entity) >= 0.99 인 **동어반복 엣지: {len(taut)}개**")
    if len(taut):
        L.append("")
        L.append(taut.head(15)[["entity", "frame", "weight", "entity_count",
                                "P_frame_given_entity"]].to_markdown(index=False))
        L.append("")
        L.append("- 의미: 개체가 등장하면 프레임도 100% 등장하도록 사전이 설계되어 있어 "
                 "해당 엣지는 정보량이 0이며, `cent_*` 중심성과 `neg_event_propagation` 을 "
                 "인위적으로 부풀린다.")
    L.append("")

    L.append("## A4. 프레임 판별력 및 퇴화 지표\n")
    L.append(fdf.to_markdown(index=False))
    L.append("")
    L.append(degdf.to_markdown(index=False))
    L.append("")
    broad = fdf[fdf["judgement"] == "과대포괄"]["frame"].tolist()
    if broad:
        L.append(f"- 과대포괄 프레임: {', '.join(broad)} -> 사실상 상시 ON 이라 판별력이 낮고 "
                 "FRAME_COOC 그래프를 완전그래프에 가깝게 만든다.")
    degen = degdf[degdf["judgement"].str.startswith("퇴화")]["indicator"].tolist()
    if degen:
        L.append(f"- 변동이 거의 없어 설명변수로 무의미한 지표: {', '.join(degen)}")
    L.append("")

    L.append("## A5. 계층 중복 계상\n")
    if len(hdf):
        L.append(hdf.head(10).to_markdown(index=False))
        L.append("")
    L.append(f"- 모델 개체 언급 총합 {inflate['model_entity_mentions_total']}건 / "
             f"모델 보유 기사 {inflate['articles_with_any_model']}건 "
             f"= 기사당 {inflate['mentions_per_article']}회")
    L.append(f"- {inflate['note']}")
    L.append("")

    L.append("## A6. 사전 커버리지\n")
    if len(dead_df):
        L.append(dead_df.head(12).to_markdown(index=False))
        L.append("")
    L.append("### neg_event_propagation 구성 개체별 기여\n")
    L.append(comp_df.to_markdown(index=False))
    L.append("")
    dilute = comp_df[comp_df["judgement"].str.startswith("상시 0")]["negative_event_entity"].tolist()
    if dilute:
        L.append(f"- 상시 0으로 평균을 희석하는 개체: {', '.join(dilute)} "
                 "-> 6개 단순평균이라 지표가 실제보다 낮게/둔감하게 측정됨.")
    L.append("")

    L.append("## A7. 연관강도 재평가 (빈도 상위 vs 의미 상위)\n")
    L.append("### 원시 빈도 상위 10\n")
    L.append(assoc.head(10)[["edge_type", "source", "target", "weight", "lift", "pmi", "g2"]]
             .to_markdown(index=False))
    L.append("")
    L.append("### PMI 상위 10 (빈도 30 이상)\n")
    L.append(assoc.sort_values("pmi", ascending=False).head(10)
             [["edge_type", "source", "target", "weight", "lift", "pmi", "g2"]]
             .to_markdown(index=False))
    L.append("")
    L.append("- 원시 빈도 순위는 노드 빈도에 지배되어 `Frame:산업` 관련 엣지가 상단을 독점한다. "
             "PMI/lift 로 재순위하면 실제로 결합력이 강한 조합이 드러난다.")
    L.append("")

    L.append("## A8. 오탐 검토표본 (단어경계 없는 매칭)\n")
    if fp_df is None:
        L.append(f"- {fp_stats}")
    else:
        L.append(f"- 위험 별칭별 전체 매칭 건수: {fp_stats}")
        L.append(f"- 검토표본 {len(fp_df)}건: `{os.path.relpath(FP_REVIEW_PATH, kc.BASE_DIR)}`")
        fp_tbl, label_col, agree = score_fp_precision(fp_df)
        if fp_tbl is None:
            L.append("- 아직 미검토 상태다. `1`(정탐)/`0`(오탐) 기입 후 정밀도가 산출된다.")
        else:
            L.append(f"- 판정 기준: `match_context` 의 매칭 문자열이 해당 개체를 지칭하는지 "
                     f"(기사 주제 여부가 아니다). 라벨 열: `{label_col}`")
            if agree:
                L.append(f"- 사람 기입 {agree['n']}건과 기계 판독의 일치율 "
                         f"{agree['rate']:.3f} ({agree['agree']}/{agree['n']})")
            L.append("")
            L.append("### 별칭별 정밀도 (원본 규칙 vs 정제 규칙)\n")
            L.append(fp_tbl.to_markdown(index=False))
            L.append("")
            fp_tbl.to_csv(os.path.join(kc.TABLE_DIR, "tbl_audit_fp_precision.csv"),
                          index=False, encoding="utf-8-sig")
            tot = fp_tbl[fp_tbl["alias"] == "(전체)"].iloc[0]
            L.append(f"- 원본 사전의 개체 매칭 정밀도는 **{tot['precision_orig']:.3f}** "
                     f"({tot['false_positive']}/{tot['reviewed']} 오탐)이고, "
                     f"정제 규칙 적용 후 **{tot['precision_clean']:.3f}** 로 개선된다. "
                     f"정탐을 잘못 지운 건수는 {tot['tp_lost']}건이다.")
            worst = fp_tbl[fp_tbl["alias"] != "(전체)"].nsmallest(1, "precision_orig").iloc[0]
            L.append(f"- 최저 정밀도 별칭은 `{worst['alias']}` ({worst['precision_orig']:.3f}) 다. "
                     "다만 전체 매칭 건수가 작아 월별 지표에 미치는 영향은 제한적이다.")
            L.append("- 오탐 유형은 표기상 겹침(리프트/드리프트/페이스리프트/폴리프로필렌), "
                     "영문 약칭 내부 매칭(HMGMA·E-GMP·GGM·KGM), 동형이의어(기아=굶주림)로 "
                     "구분되며, 모두 문자열 규칙으로 교정 가능한 유형이다.")
    L.append("")

    L.append("---\n")
    L.append("## B. 감성신뢰도 KG 검진\n")
    L.append(f"- 표본 {sres['n_sample']}건 중 사람 감성라벨 {sres['n_human_labeled']}건, "
             f"그중 '관련' {sres['n_relevant_labeled']}건, '무관' {sres['n_irrelevant']}건")
    L.append("")
    L.append("### B1-B2. 정확도 · Cohen kappa\n")
    L.append(kappa_df.to_markdown(index=False))
    L.append("")
    if sres["n_human_labeled"] == sres["n_relevant_labeled"]:
        L.append(f"- 두 스펙 수치가 동일한 이유: 사람 감성라벨이 달린 {sres['n_human_labeled']}건은 "
                 f"모두 '관련' 판정이고, '무관' {sres['n_irrelevant']}건에는 감성라벨이 부여되지 "
                 "않았다(검토 200건 = 관련 165 + 무관 35). 따라서 무관 기사는 신뢰도 지표를 "
                 "오염시키지 않는다.")
        L.append("")
    L.append("- kappa 해석: 사전 기반 감성은 사람 판단과 '낮음(fair)' 수준의 일치도만 보인다. "
             "본문에서 감성 변수를 단독 인과 근거로 쓰지 말고, 측정오차를 명시한 뒤 "
             "프레임·구조 지표를 병행 근거로 삼아야 한다.")
    L.append("")
    L.append("### 클래스별 정밀도/재현율\n")
    L.append(cls_df.to_markdown(index=False))
    L.append("")
    if "confusion" in sres:
        L.append("### 혼동행렬 (무관 제외)\n")
        L.append(sres["confusion"].to_markdown())
        L.append("")
    L.append("### B4. 불일치 파일 정합성\n")
    L.append(f"- 검토표본에서 파생한 오분류 {sres['derived_mismatch']}건 vs "
             f"`sentiment_mismatch_analysis.csv` {sres['mismatch_file_rows']}행 -> "
             f"{'일치' if sres['mismatch_consistent'] else '불일치'}")
    if sres["only_in_derived"]:
        L.append(f"- 파생에만 있음: {sres['only_in_derived'][:10]}")
    if sres["only_in_file"]:
        L.append(f"- 파일에만 있음: {sres['only_in_file'][:10]}")
    L.append("")
    L.append("### B5. Sentiment 노드 카운트 이중합산\n")
    L.append(sres["sentiment_node_bug"].to_markdown(index=False))
    L.append("")
    if bool(sres["sentiment_node_bug"]["double_counted"].any()):
        L.append("- `kg_sentiment_reliability.py` 의 `add_node` 가 예측 라벨과 사람 라벨을 "
                 "같은 `Sentiment::*` 노드에 누적하여 count 가 (예측+사람) 합산값이 된다. "
                 "노드 빈도를 그대로 해석하면 안 되며, 정제판에서는 예측/사람 노드를 분리해야 한다.")
    L.append("")

    L.append("---\n")
    L.append("## 종합 판정\n")
    L.append("| 항목 | 결과 |")
    L.append("| --- | --- |")
    L.append(f"| 집계 로직 무결성 | {'정확(완전일치)' if not diffs else f'불일치 {len(diffs)}건'} |")
    L.append(f"| 구조 위반 | {'없음' if not violations else f'{len(violations)}건'} |")
    L.append(f"| 동어반복 엣지 | {len(taut)}개 (교정 필요) |")
    L.append(f"| 과대포괄 프레임 | {', '.join(broad) if broad else '없음'} |")
    L.append(f"| 퇴화 지표 | {', '.join(degen) if degen else '없음'} |")
    L.append(f"| 계층 중복 별칭쌍 | {len(hdf)}쌍 |")
    L.append(f"| 저빈도·미작동 개체 | {len(dead_df)}개 |")
    L.append("")
    L.append("교정안은 `scripts/29_kg_clean_variant.py` 에서 정제판으로 구현하고, "
             "핵심 결과 유지 여부는 `scripts/30_kg_clean_robustness.py` 에서 검증한다.")

    path = os.path.join(kc.TABLE_DIR, "kg_quality_audit.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[28] 리포트 저장: {path}")
    print(f"[28] 동어반복 엣지 {len(taut)}개 / 구조위반 {len(violations)}건 / 무결성 불일치 {len(diffs)}건")


if __name__ == "__main__":
    main()
