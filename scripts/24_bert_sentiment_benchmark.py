# -*- coding: utf-8 -*-
"""
[24] BERT 감성 벤치마크 (사전기반 대비) + (선택)로버스트 재실행

공개 한국어 감성분류 체크포인트를 로드해 사람 라벨(165건)과 비교, 사전기반 대비
정밀도/재현율/F1 을 산출한다. BERT가 개선되면 --rerun 시 news_tagged.csv 의
sentiment 를 BERT 라벨로 교체한 로버스트 버전(news_tagged_bert.csv)을 생성한다.

사용:
  python scripts/24_bert_sentiment_benchmark.py            # 벤치마크만
  python scripts/24_bert_sentiment_benchmark.py --rerun    # 전체 코퍼스 재라벨(느림)

출력 (outputs/kg/tables/):
  - tbl_bert_benchmark.csv
  - bert_benchmark_summary.md
  - (--rerun 시) data/processed/news_tagged_bert.csv
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

warnings.filterwarnings("ignore")

CANDIDATES = [
    "matthewburke/korean_sentiment",
    "sangrimlee/bert-base-multilingual-cased-nsmc",
    "snunlp/KR-FinBert-SC",
    "WhitePeak/bert-base-cased-Korean-sentiment",
]
LABELS = ["긍정", "부정", "중립"]


def load_pipeline():
    from transformers import pipeline
    for name in CANDIDATES:
        try:
            clf = pipeline("text-classification", model=name, top_k=None,
                           truncation=True, max_length=256, device=-1)
            print(f"  [BERT] 로드 성공: {name}")
            return clf, name
        except Exception as e:
            print(f"  [BERT] 실패 {name}: {str(e)[:80]}")
    return None, None


def label_is_positive(label: str) -> bool:
    l = label.lower()
    if any(k in l for k in ["pos", "positive", "긍정"]):
        return True
    if "label_1" in l or l == "1":
        return True
    return False


def bert_to_3class(score_list):
    """모델 출력(각 라벨 확률)을 3-class 로 매핑. 사전기반과 동일한 임계값(±0.25)."""
    p_pos, p_neg = 0.0, 0.0
    for d in score_list:
        if label_is_positive(d["label"]):
            p_pos += d["score"]
        else:
            p_neg += d["score"]
    diff = p_pos - p_neg
    if diff >= 0.25:
        return "긍정"
    if diff <= -0.25:
        return "부정"
    return "중립"


def metrics_block(y_true, y_pred, name):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    rows = []
    pr, rc, fs, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    for i, lab in enumerate(LABELS):
        rows.append({"method": name, "class": lab, "precision": round(pr[i], 3),
                     "recall": round(rc[i], 3), "f1": round(fs[i], 3), "support": int(sup[i])})
    rows.append({"method": name, "class": "MACRO", "precision": round(p, 3),
                 "recall": round(r, 3), "f1": round(f, 3), "support": len(y_true)})
    return rows, acc, f


def main():
    kc.ensure_dirs()
    rerun = "--rerun" in sys.argv

    sample = pd.read_csv(os.path.join(kc.OUT_DIR, "sentiment_review_sample_300.csv"))
    sample = sample[sample["human_sentiment"].notna()].copy()
    sample["text"] = (sample["Title"].fillna("").astype(str) + " " +
                      sample["Content_excerpt"].fillna("").astype(str)).str.slice(0, 800)

    y_true = sample["human_sentiment"].tolist()
    y_dict = sample["predicted_sentiment"].tolist()

    all_rows = []
    dict_rows, dict_acc, dict_f = metrics_block(y_true, y_dict, "사전기반")
    all_rows += dict_rows

    clf, model_name = load_pipeline()
    bert_f = None
    if clf is not None:
        preds = clf(sample["text"].tolist(), batch_size=16)
        y_bert = [bert_to_3class(p) for p in preds]
        bert_rows, bert_acc, bert_f = metrics_block(y_true, y_bert, f"BERT")
        all_rows += bert_rows
        print(f"  [BERT] 정확도={bert_acc:.3f} macroF1={bert_f:.3f} vs 사전 acc={dict_acc:.3f} F1={dict_f:.3f}")
    else:
        print("  [BERT] 사용 가능한 체크포인트 없음(오프라인?). 사전기반 결과만 저장.")

    tbl = pd.DataFrame(all_rows)
    tbl.to_csv(os.path.join(kc.TABLE_DIR, "tbl_bert_benchmark.csv"),
               index=False, encoding="utf-8-sig")

    lines = ["# BERT 감성 벤치마크 (사람 라벨 165건 기준)\n"]
    lines.append(f"- 사용 모델: {model_name or '없음(오프라인)'}\n")
    lines.append(tbl.to_markdown(index=False))
    lines.append("")
    if bert_f is not None:
        better = bert_f > dict_f
        lines.append(f"- macro-F1: 사전기반 {dict_f:.3f} vs BERT {bert_f:.3f} → "
                     f"{'BERT 우수' if better else '사전기반 우수/동등'}")
        if better:
            lines.append("- 결론: BERT가 우수하므로 로버스트니스 버전(--rerun) 생성을 권장.")
        else:
            lines.append("- 결론: 사전기반이 열등하지 않음 → 본문 사전기반 유지 정당화(부록에 벤치마크 표).")
    else:
        lines.append("- BERT 체크포인트를 받지 못해 벤치마크 미완. 온라인 환경에서 재실행 필요.")

    path = os.path.join(kc.TABLE_DIR, "bert_benchmark_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # 선택: 전체 코퍼스 BERT 재라벨
    if rerun and clf is not None:
        print("[24] 전체 코퍼스 BERT 재라벨 (시간 소요)...")
        nt = pd.read_csv(kc.NEWS_TAGGED_CSV)
        texts = nt["Title"].fillna("").astype(str).str.slice(0, 300).tolist()
        out = clf(texts, batch_size=32)
        nt["sentiment"] = [bert_to_3class(p) for p in out]
        nt.to_csv(os.path.join(kc.DATA_PROC, "news_tagged_bert.csv"),
                  index=False, encoding="utf-8-sig")
        print("[24] 저장: data/processed/news_tagged_bert.csv (16~18 재실행 시 입력 교체)")
    print(f"\n[24] 저장: {path}")


if __name__ == "__main__":
    main()
