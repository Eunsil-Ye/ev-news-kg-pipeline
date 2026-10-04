# -*- coding: utf-8 -*-
"""
[35] 사람 라벨 재검토 (전기차 시장 기준, 블라인드)

34번 벤치마크에서 세 LLM 이 모두 놓친 사람 '부정' 23건을, 같은 수의 무작위 대조 기사와
섞어 전기차 시장 기준으로 다시 라벨링한다. 검토 파일에는 기존 사람 라벨·규칙 라벨·LLM 라벨을
넣지 않아, 어느 기사가 '놓친 기사'인지 검토자가 알 수 없다.
기존 human_sentiment 는 덮어쓰지 않고, 원 라벨과 수정 라벨 두 기준으로 성능을 함께 보고한다.

사용:
  python scripts/35_sentiment_relabel_review.py make      # 검토 파일 생성 (이미 있으면 덮어쓰지 않음)
  python scripts/35_sentiment_relabel_review.py make --all  # 165건 전체를 검토 대상으로
  python scripts/35_sentiment_relabel_review.py report    # 검토 결과 반영 후 성능 재계산

검토 파일: outputs/sentiment_relabel_review.csv
  - human_relevance_ev : 관련 / 비관련  (전기차 시장·보급·구매와 관련된 기사인가)
  - human_sentiment_ev : 긍정 / 부정 / 중립  (비관련이면 비워 두면 중립으로 처리)
  - review_memo_ev     : 판단 메모 (선택)

출력:
  - outputs/kg/tables/tbl_sentiment_relabel_benchmark.csv
  - outputs/kg/tables/sentiment_relabel_summary.md
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

LABELS = ["긍정", "부정", "중립"]
LLM_MODELS = ["gpt-6-luna", "gpt-6-sol", "gpt-6-astra"]
SEED = 42

REVIEW_CSV = os.path.join(kc.OUT_DIR, "sentiment_relabel_review.csv")
KEY_CSV = os.path.join(kc.OUT_DIR, "kg", "llm_cache", "relabel_review_key.csv")
LABELS_CSV = os.path.join(kc.TABLE_DIR, "tbl_llm_sentiment_labels.csv")

GUIDE = """전기차 시장 기준 라벨링 지침
- human_relevance_ev: 기사의 중심 내용이 전기차 시장·보급·구매(차량, 배터리·충전 등 직접 관련 산업 포함)와 관련되면 '관련', 전기차가 한 줄 언급되거나 배경에 불과하면 '비관련'
- human_sentiment_ev (관련 기사만):
  긍정 = 전기차 보급·판매·구매 여건에 유리 (판매 증가, 보조금 확대, 충전 인프라 확충, 신차, 가격 인하, 기술 개선 등)
  부정 = 불리 (화재·리콜·결함, 판매 부진, 보조금 축소, 충전 불편, 가격 부담, 수요 둔화 등)
  중립 = 방향이 뚜렷하지 않거나 양쪽이 비슷하게 섞임
- 기사 전체의 분위기(정치 공방, 파업, 적자 등)가 아니라 전기차 시장에 대한 방향만 판단한다."""


def load_wide() -> pd.DataFrame:
    d = pd.read_csv(LABELS_CSV, encoding="utf-8-sig")
    w = d.pivot(index="sample_id", columns="model", values="llm_label")
    base = d.drop_duplicates("sample_id").set_index("sample_id")[
        ["article_id", "human_label", "rule_label_sample", "rule_label_analysis"]]
    return base.join(w).reset_index()


def make(all_rows: bool) -> None:
    if os.path.exists(REVIEW_CSV):
        raise SystemExit(f"이미 검토 파일이 있다(작업 보호를 위해 덮어쓰지 않음): {REVIEW_CSV}")
    w = load_wide()
    missed = (w["human_label"] == "부정") & (w[LLM_MODELS] != "부정").all(axis=1)
    rng = np.random.default_rng(SEED)
    if all_rows:
        w["group"] = np.where(missed, "missed", "rest")
        sel = w
    else:
        target = w[missed].assign(group="missed")
        ctrl_pool = w[~missed]
        ctrl_idx = rng.choice(ctrl_pool.index, size=len(target), replace=False)
        sel = pd.concat([target, ctrl_pool.loc[ctrl_idx].assign(group="control")])
    sel = sel.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sel["review_id"] = [f"R{i:03d}" for i in range(1, len(sel) + 1)]

    src = pd.read_csv(os.path.join(kc.OUT_DIR, "sentiment_review_sample_300.csv"),
                      encoding="utf-8-sig")[["sample_id", "Date", "newspaper", "Title",
                                             "Content_excerpt", "Url"]]
    review = sel[["review_id", "sample_id"]].merge(src, on="sample_id", how="left")
    review["human_relevance_ev"] = ""
    review["human_sentiment_ev"] = ""
    review["review_memo_ev"] = ""
    review.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")

    os.makedirs(os.path.dirname(KEY_CSV), exist_ok=True)
    sel[["review_id", "sample_id", "group"]].to_csv(KEY_CSV, index=False, encoding="utf-8-sig")
    print(f"[35] 검토 파일 {len(review)}건: {REVIEW_CSV}")
    print(f"     (놓친 기사 {int((sel.group == 'missed').sum())}건 + 대조 "
          f"{int((sel.group != 'missed').sum())}건, 섞어서 배치. 구분 키는 {KEY_CSV})\n")
    print(GUIDE)


def metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                    zero_division=0)
    _, _, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                  average="macro", zero_division=0)
    i = LABELS.index("부정")
    return {"macro_f1": round(mf, 3), "kappa": round(cohen_kappa_score(y_true, y_pred), 3),
            "neg_precision": round(pr[i], 3), "neg_recall": round(rc[i], 3),
            "neg_f1": round(f1[i], 3), "n": len(y_true),
            "n_neg": int((pd.Series(y_true) == "부정").sum())}


def report() -> None:
    if not os.path.exists(REVIEW_CSV):
        raise SystemExit("검토 파일이 없다. 먼저 make 를 실행한다.")
    rv = pd.read_csv(REVIEW_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    for c in ["human_relevance_ev", "human_sentiment_ev"]:
        rv[c] = rv[c].str.strip()
    bad_rel = rv[~rv.human_relevance_ev.isin(["", "관련", "비관련"])]
    bad_sen = rv[~rv.human_sentiment_ev.isin([""] + LABELS)]
    if len(bad_rel) or len(bad_sen):
        raise SystemExit(f"허용되지 않은 값: {bad_rel.review_id.tolist() + bad_sen.review_id.tolist()}")
    need = rv[(rv.human_relevance_ev == "관련") & (rv.human_sentiment_ev == "")]
    if len(need):
        raise SystemExit(f"'관련'인데 감성이 비어 있음: {need.review_id.tolist()}")
    done = rv[rv.human_relevance_ev != ""].copy()
    done["label_ev"] = np.where(done.human_relevance_ev == "비관련", "중립",
                                done.human_sentiment_ev)
    print(f"[35] 검토 완료 {len(done)}/{len(rv)}건")

    w = load_wide().merge(done[["sample_id", "human_relevance_ev", "label_ev"]],
                          on="sample_id", how="left")
    w["human_label_ev"] = w["label_ev"].fillna(w["human_label"])
    key = pd.read_csv(KEY_CSV, encoding="utf-8-sig")
    w = w.merge(key[["sample_id", "group"]], on="sample_id", how="left")

    methods = {"규칙(분석 라벨)": "rule_label_analysis", "규칙(표본 라벨)": "rule_label_sample",
               **{m: m for m in LLM_MODELS if m in w.columns}}
    rel_ok = w["human_relevance_ev"] != "비관련"
    settings = [
        ("원 라벨", w.index == w.index, "human_label"),
        ("수정 라벨(비관련=중립)", w.index == w.index, "human_label_ev"),
        ("수정 라벨(비관련 제외)", rel_ok, "human_label_ev"),
    ]
    rows = []
    for sname, mask, ycol in settings:
        sub = w[mask]
        for mname, pcol in methods.items():
            rows.append({"기준": sname, "방법": mname, **metrics(sub[ycol], sub[pcol])})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(kc.TABLE_DIR, "tbl_sentiment_relabel_benchmark.csv"),
               index=False, encoding="utf-8-sig")

    chg = w[w.label_ev.notna()]
    trans = pd.crosstab(chg["human_label"], chg["human_label_ev"])
    by_group = chg.assign(changed=chg.human_label != chg.human_label_ev) \
        .groupby("group").agg(n=("changed", "size"), changed=("changed", "sum"),
                              irrelevant=("human_relevance_ev", lambda s: (s == "비관련").sum()))
    lines = ["# 사람 라벨 재검토 결과 (전기차 시장 기준, 블라인드)\n",
             f"- 검토 {len(done)}/{len(rv)}건. 검토하지 않은 기사는 원 라벨을 그대로 사용",
             "- 검토 파일에는 원 라벨·규칙·LLM 라벨을 넣지 않았고, 놓친 기사와 무작위 대조 기사를 섞었다.\n",
             "## 그룹별 변경 (missed = 세 LLM 이 모두 놓친 사람 '부정', control = 무작위 대조)\n",
             by_group.to_markdown(), "",
             "## 원 라벨 → 수정 라벨\n", trans.to_markdown(), "",
             "## 기준별 성능\n", tbl.to_markdown(index=False)]
    path = os.path.join(kc.TABLE_DIR, "sentiment_relabel_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[35] 저장: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "report"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    kc.ensure_dirs()
    make(args.all) if args.cmd == "make" else report()


if __name__ == "__main__":
    main()
