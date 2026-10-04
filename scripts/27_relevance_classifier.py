# -*- coding: utf-8 -*-
"""
[27] 관련성 필터 정밀화 + 필터링 코퍼스로 핵심결과 재검증

human_relevance(관련/무관) 200건 라벨로 간단한 분류기(로지스틱회귀, hit-word 특성)를
학습해 키워드 매칭 대비 무관기사 제거 효과를 확인하고, 필터링된 코퍼스로
전국 Granger 핵심 결과가 유지되는지 재검증한다.

출력:
  - outputs/kg/tables/tbl_relevance_classifier.csv
  - outputs/kg/tables/tbl_granger_refiltered.csv
  - outputs/kg/tables/relevance_summary.md
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

CAND = ["news_count", "neg_share", "net_sentiment", "frshare_safety",
        "frshare_newmodel", "salience_newmodel", "neg_event_propagation", "frame_entropy"]


def char_tokenizer(text):
    return list(str(text))


def train_relevance():
    s = pd.read_csv(os.path.join(kc.OUT_DIR, "sentiment_review_sample_300.csv"))
    s = s[s["human_relevance"].notna()].copy()
    # 라벨 정규화: 관련=1, 무관=0
    s["y"] = (s["human_relevance"].astype(str).str.strip() == "관련").astype(int)
    # 전체 코퍼스(news_tagged)에는 Title만 있으므로 공정한 적용을 위해 제목만 사용
    s["text"] = s["Title"].fillna("").astype(str)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=2, max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    # 교차검증 예측으로 성능 추정
    y = s["y"].values
    pred = cross_val_predict(pipe, s["text"], y, cv=5)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    baseline_acc = max(y.mean(), 1 - y.mean())  # 다수클래스 기준
    acc = (pred == y).mean()

    # 전체 라벨로 최종 적합
    pipe.fit(s["text"], y)
    return pipe, {"n_labeled": len(s), "rel_rate": round(y.mean(), 3),
                  "cv_acc": round(acc, 3), "majority_acc": round(baseline_acc, 3),
                  "precision_rel": round(p, 3), "recall_rel": round(r, 3),
                  "f1_rel": round(f, 3)}


def granger_on(df, tag):
    rows = []
    for f in CAND:
        if f not in df.columns:
            continue
        sub = df[["ev_net_add", f]].dropna().diff().dropna()
        if len(sub) < 20 or sub[f].std() == 0:
            continue
        try:
            gc = grangercausalitytests(sub[["ev_net_add", f]], maxlag=3, verbose=False)
            pv = {k: gc[k][0]["ssr_ftest"][1] for k in gc}
            bl = min(pv, key=pv.get)
            rows.append({"feature": f, "best_lag": bl, "p_value": round(pv[bl], 4),
                         "corpus": tag})
        except Exception:
            continue
    return pd.DataFrame(rows)


def rebuild_national_from_corpus(nt_filtered):
    """필터링된 기사로 월별 단순 뉴스지표를 재구성해 기존 등록/KG와 병합."""
    d = nt_filtered.copy()
    d["is_pos"] = (d["sentiment"] == "긍정").astype(int)
    d["is_neg"] = (d["sentiment"] == "부정").astype(int)
    grp = d.groupby("ym")
    monthly = pd.DataFrame({
        "news_count": grp.size(),
        "pos_share": grp["is_pos"].mean(),
        "neg_share": grp["is_neg"].mean(),
    })
    monthly["net_sentiment"] = monthly["pos_share"] - monthly["neg_share"]
    for fr, col in [("안전", "frshare_safety"), ("신차", "frshare_newmodel")]:
        monthly[col] = grp["frames"].apply(
            lambda s: s.fillna("").str.split(";").apply(lambda xs: fr in xs).mean())
    monthly = monthly.reset_index()

    nat = pd.read_csv(kc.NATIONAL_CSV)
    # KG 지표는 기존 것 유지, 단순 뉴스지표만 필터링본으로 교체
    kg_cols = [c for c in ["ym", "ev_net_add", "salience_newmodel",
                           "neg_event_propagation", "frame_entropy"] if c in nat.columns]
    merged = nat[kg_cols].merge(monthly, on="ym", how="inner")
    return merged


def main():
    kc.ensure_dirs()
    pipe, stats = train_relevance()
    pd.DataFrame([stats]).to_csv(
        os.path.join(kc.TABLE_DIR, "tbl_relevance_classifier.csv"),
        index=False, encoding="utf-8-sig")

    # 전체 코퍼스에 적용
    nt = pd.read_csv(kc.NEWS_TAGGED_CSV)
    text = nt["Title"].fillna("").astype(str)
    proba = pipe.predict_proba(text)[:, 1]
    nt["rel_proba"] = proba
    keep = nt[nt["rel_proba"] >= 0.5].copy()
    removed = len(nt) - len(keep)

    # Granger 재검증 (원본 vs 필터링)
    g_orig = granger_on(pd.read_csv(kc.NATIONAL_CSV), "원본")
    g_filt = granger_on(rebuild_national_from_corpus(keep), "필터링")
    g_all = pd.concat([g_orig, g_filt], ignore_index=True)
    g_all.to_csv(os.path.join(kc.TABLE_DIR, "tbl_granger_refiltered.csv"),
                 index=False, encoding="utf-8-sig")

    lines = ["# 관련성 분류기 + 필터링 재검증 요약\n"]
    lines.append("## 1. 관련성 분류기 성능 (5-fold CV)\n")
    lines.append(pd.DataFrame([stats]).to_markdown(index=False))
    lines.append("")
    lines.append(f"- 전체 코퍼스 {len(nt)}건 중 무관 추정 {removed}건 제거 "
                 f"({removed/len(nt)*100:.1f}%), 잔존 {len(keep)}건.")
    lines.append("")
    lines.append("## 2. Granger 재검증 (원본 vs 필터링, 차분 maxlag=3)\n")
    piv = g_all.pivot_table(index="feature", columns="corpus", values="p_value")
    lines.append(piv.round(4).to_markdown())
    lines.append("")
    lines.append("- 두 코퍼스에서 유의패턴이 유지되면 관련성 필터에 결과가 강건함을 의미.")

    path = os.path.join(kc.TABLE_DIR, "relevance_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[27] 저장: {path}")


if __name__ == "__main__":
    main()
