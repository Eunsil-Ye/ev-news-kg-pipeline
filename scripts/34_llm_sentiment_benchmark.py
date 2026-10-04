# -*- coding: utf-8 -*-
"""
[34] LLM API 감성 벤치마크 (사람 라벨 165건 전용)

기존 규칙 기반 감성분석은 그대로 두고, OpenAI API 로 같은 165건에 라벨을 붙여
사람 라벨과 비교한다. 전체 코퍼스 실행과 Batch 제출 기능은 두지 않는다.

- API 키: 환경변수 OPENAI_API_KEY
- 비용 상한: 모든 모델·모든 실행을 합산한 누적 사용액이 --max-usd 를 넘지 않는다.
  호출 전마다 (누적 실사용액 + 해당 호출의 최악 비용) 이 상한 이하인지 확인한다.
- 캐시: 같은 모델·같은 기사는 다시 호출하지 않는다.

사용:
  python scripts/34_llm_sentiment_benchmark.py --dry-run                 # 비용 추정만
  python scripts/34_llm_sentiment_benchmark.py --model gpt-6-luna --limit 2
  python scripts/34_llm_sentiment_benchmark.py --model gpt-6-luna
  python scripts/34_llm_sentiment_benchmark.py --report                  # 성능표만 재계산

출력:
  - outputs/kg/tables/tbl_llm_sentiment_labels.csv
  - outputs/kg/tables/tbl_llm_benchmark.csv
  - outputs/kg/tables/llm_benchmark_summary.md
  - outputs/kg/llm_cache/<model>.jsonl, outputs/kg/llm_cache/spend_ledger.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

MODELS = ["gpt-6-luna", "gpt-6-sol", "gpt-6-astra"]
# 일반(비 Batch) 요금, USD / 1M tokens. 캐시 할인은 반영하지 않아 실제보다 크게 잡힌다.
PRICES = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}
MAX_OUTPUT_TOKENS = 1000
DEFAULT_MAX_USD = 3.0
LABELS = ["긍정", "부정", "중립"]
PROMPT_VERSION = "v1-ev_market-zeroshot"

# 비교 기준(결과를 보기 전에 고정): 부정 F1 은 BERT, macro-F1 은 분석 라벨을 넘어야 한다.
BERT_NEG_F1 = 0.678
BERT_MACRO_F1 = 0.421

CACHE_DIR = os.path.join(kc.OUT_DIR, "kg", "llm_cache")
LEDGER = os.path.join(CACHE_DIR, "spend_ledger.json")

SYSTEM_PROMPT = """너는 한국 신문 기사를 분류하는 평가자다.
기사 제목과 본문 앞부분을 읽고, 이 기사가 전기차 시장·보급·구매에 유리한 내용인지 불리한 내용인지 판단하라.

- 긍정: 전기차 보급·판매·구매 여건에 유리한 내용 (예: 판매·보급 증가, 보조금 확대, 충전 인프라 확충, 신차 출시, 가격 인하, 기술 개선)
- 부정: 전기차 보급·판매·구매 여건에 불리한 내용 (예: 화재·리콜·결함, 판매 부진, 보조금 축소, 충전 불편, 가격 부담, 수요 둔화)
- 중립: 유리하거나 불리한 방향이 뚜렷하지 않거나, 두 방향의 내용이 비슷한 비중으로 섞여 있음

판단 근거는 한국어 한 문장, 60자 이내로 쓴다."""

SCHEMA = {
    "type": "json_schema",
    "name": "ev_sentiment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": LABELS},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    },
}


def load_sample() -> pd.DataFrame:
    s = pd.read_csv(os.path.join(kc.OUT_DIR, "sentiment_review_sample_300.csv"),
                    encoding="utf-8-sig")
    s = s[s["human_sentiment"].notna()].copy()
    nt = pd.read_csv(kc.NEWS_TAGGED_CSV, encoding="utf-8-sig",
                     usecols=["article_id", "Url", "sentiment"])
    s = s.merge(nt, on="Url", how="left")
    if s["article_id"].isna().any():
        raise SystemExit("사람 라벨 표본 중 news_tagged.csv 와 연결되지 않는 기사가 있다.")
    # BERT 벤치마크(24번)와 같은 입력
    s["text"] = (s["Title"].fillna("").astype(str) + " " +
                 s["Content_excerpt"].fillna("").astype(str)).str.slice(0, 800)
    return s.rename(columns={
        "Url": "url",
        "human_sentiment": "human_label",
        "predicted_sentiment": "rule_label_sample",
        "sentiment": "rule_label_analysis",
    })[["sample_id", "article_id", "url", "human_label",
        "rule_label_sample", "rule_label_analysis", "text"]]


def est_input_tokens(text: str) -> int:
    # 한국어는 대략 1자 이하 1토큰이므로 글자 수를 토큰 수로 보면 보수적인 추정이 된다.
    return len(SYSTEM_PROMPT) + len(text) + 50


def cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICES[model]
    return in_tok / 1e6 * pin + out_tok / 1e6 * pout


def read_ledger() -> dict:
    if not os.path.exists(LEDGER):
        return {"total_usd": 0.0, "by_model": {}}
    with open(LEDGER, encoding="utf-8") as f:
        return json.load(f)


def write_ledger(led: dict) -> None:
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(led, f, ensure_ascii=False, indent=2)


def cache_path(model: str) -> str:
    return os.path.join(CACHE_DIR, f"{model}.jsonl")


def read_cache(model: str) -> dict:
    out = {}
    p = cache_path(model)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("label") in LABELS:
                    out[rec["sample_id"]] = rec
    return out


def call_api(client, model: str, text: str, state: dict):
    from openai import BadRequestError

    params = dict(state.setdefault("params", {
        "reasoning": {"effort": "low"},
        "temperature": 0,
    }))
    for _ in range(4):
        try:
            return client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=text,
                text={"format": SCHEMA},
                max_output_tokens=MAX_OUTPUT_TOKENS,
                **params,
            )
        except BadRequestError as e:
            msg = str(e).lower()
            dropped = [k for k in list(params) if k in msg]
            if not dropped:
                raise
            for k in dropped:
                params.pop(k)
                print(f"  [{model}] '{k}' 미지원 → 제외하고 재시도")
            state["params"] = params
    raise RuntimeError("파라미터 조정 후에도 요청이 거부되었다.")


def run_model(model: str, sample: pd.DataFrame, max_usd: float, limit: int | None) -> None:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("환경변수 OPENAI_API_KEY 가 없다.")
    client = OpenAI()
    os.makedirs(CACHE_DIR, exist_ok=True)
    led = read_ledger()
    done = read_cache(model)
    todo = sample[~sample["sample_id"].isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    print(f"[34] {model}: 캐시 {len(done)}건, 이번 호출 대상 {len(todo)}건, "
          f"누적 사용액 ${led['total_usd']:.4f} / 상한 ${max_usd:.2f}")

    state: dict = {}
    with open(cache_path(model), "a", encoding="utf-8") as f:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            worst = cost(model, est_input_tokens(row.text), MAX_OUTPUT_TOKENS)
            if led["total_usd"] + worst > max_usd:
                print(f"  [중단] 다음 호출의 최악 비용 ${worst:.4f} 를 더하면 상한 초과. "
                      f"누적 ${led['total_usd']:.4f}")
                break
            resp = call_api(client, model, row.text, state)
            u = resp.usage
            in_tok, out_tok = u.input_tokens, u.output_tokens
            c = cost(model, in_tok, out_tok)
            led["total_usd"] += c
            led["by_model"][model] = led["by_model"].get(model, 0.0) + c
            write_ledger(led)

            label, reason = None, ""
            try:
                obj = json.loads(resp.output_text)
                label, reason = obj.get("label"), obj.get("reason", "")
            except Exception:
                reason = f"[파싱 실패 status={getattr(resp, 'status', '')}]"
            rec = {
                "sample_id": row.sample_id, "model": model,
                "model_snapshot": getattr(resp, "model", model),
                "label": label, "reason": reason,
                "input_tokens": in_tok, "output_tokens": out_tok,
                "cost_usd": round(c, 6), "prompt_version": PROMPT_VERSION,
                "params": state.get("params", {}),
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  누적 ${led['total_usd']:.4f}")
            time.sleep(0.05)


def metrics(y_true, y_pred, method: str) -> list[dict]:
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 precision_recall_fscore_support)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    rows = [{"method": method, "class": lab, "precision": round(pr[i], 3),
             "recall": round(rc[i], 3), "f1": round(f1[i], 3), "support": int(sup[i])}
            for i, lab in enumerate(LABELS)]
    rows.append({"method": method, "class": "MACRO", "precision": round(mp, 3),
                 "recall": round(mr, 3), "f1": round(mf, 3), "support": len(y_true),
                 "accuracy": round(accuracy_score(y_true, y_pred), 3),
                 "kappa": round(cohen_kappa_score(y_true, y_pred), 3)})
    return rows


def report(sample: pd.DataFrame) -> None:
    long_rows, metric_rows = [], []
    metric_rows += metrics(sample["human_label"], sample["rule_label_analysis"], "규칙(분석 라벨)")
    metric_rows += metrics(sample["human_label"], sample["rule_label_sample"], "규칙(표본 라벨)")

    base_cols = ["sample_id", "article_id", "url", "human_label",
                 "rule_label_sample", "rule_label_analysis"]
    for model in MODELS:
        cache = read_cache(model)
        if not cache:
            continue
        lab = pd.DataFrame(cache.values())
        m = sample[base_cols].merge(lab, on="sample_id", how="inner")
        m = m.rename(columns={"label": "llm_label", "reason": "llm_reason"})
        long_rows.append(m[base_cols + ["model", "model_snapshot", "llm_label", "llm_reason",
                                        "input_tokens", "output_tokens", "cost_usd",
                                        "prompt_version"]])
        if len(m) == len(sample):
            metric_rows += metrics(m["human_label"], m["llm_label"], model)
        else:
            print(f"  [{model}] {len(m)}/{len(sample)}건만 있어 성능표에서 제외")

    if long_rows:
        pd.concat(long_rows).to_csv(os.path.join(kc.TABLE_DIR, "tbl_llm_sentiment_labels.csv"),
                                    index=False, encoding="utf-8-sig")
    tbl = pd.DataFrame(metric_rows)
    tbl.to_csv(os.path.join(kc.TABLE_DIR, "tbl_llm_benchmark.csv"),
               index=False, encoding="utf-8-sig")

    macro = tbl[tbl["class"] == "MACRO"].set_index("method")
    neg = tbl[tbl["class"] == "부정"].set_index("method")
    analysis_macro = macro.loc["규칙(분석 라벨)", "f1"]
    led = read_ledger()

    lines = ["# LLM API 감성 벤치마크 (사람 라벨 165건 기준)\n",
             f"- 프롬프트: `{PROMPT_VERSION}` (라벨 기준: 전기차 시장·보급·구매에 유리/불리)",
             f"- 입력: 제목 + 본문 앞부분, 800자 이내 (24번 BERT 벤치마크와 동일)",
             f"- 누적 API 사용액(추정): ${led['total_usd']:.4f}",
             f"- 통과 기준(사전 고정): 부정 F1 > {BERT_NEG_F1}(BERT) 그리고 "
             f"macro-F1 > {analysis_macro}(규칙 분석 라벨)\n",
             "## 요약\n",
             "| 방법 | macro-F1 | κ | 정확도 | 부정 정밀도 | 부정 재현율 | 부정 F1 | 통과 |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for method in macro.index:
        mf, k, acc = macro.loc[method, ["f1", "kappa", "accuracy"]]
        npr, nrc, nf = neg.loc[method, ["precision", "recall", "f1"]]
        verdict = ("통과" if (nf > BERT_NEG_F1 and mf > analysis_macro) else "미통과") \
            if method in MODELS else "—"
        lines.append(f"| {method} | {mf} | {k} | {acc} | {npr} | {nrc} | {nf} | {verdict} |")
    lines.append(f"| BERT (24번, 참고) | {BERT_MACRO_F1} | — | — | 0.621 | 0.745 | {BERT_NEG_F1} | — |")
    lines += ["", "## 클래스별 성능\n", tbl.to_markdown(index=False)]
    path = os.path.join(kc.TABLE_DIR, "llm_benchmark_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[34] 저장: {path}")


def dry_run(sample: pd.DataFrame, max_usd: float) -> None:
    in_tok = sum(est_input_tokens(t) for t in sample["text"])
    led = read_ledger()
    print(f"[34] 대상 {len(sample)}건, 추정 입력 {in_tok:,} 토큰(보수적)")
    print(f"     누적 사용액 ${led['total_usd']:.4f} / 상한 ${max_usd:.2f}\n")
    print("| 모델 | 예상(출력 100토큰/건) | 최악(출력 1,000토큰/건) |")
    print("|---|---:|---:|")
    tot_exp = tot_worst = 0.0
    for m in MODELS:
        exp = cost(m, in_tok, 100 * len(sample))
        worst = cost(m, in_tok, MAX_OUTPUT_TOKENS * len(sample))
        tot_exp += exp
        tot_worst += worst
        print(f"| {m} | ${exp:.3f} | ${worst:.3f} |")
    print(f"| 합계 | ${tot_exp:.3f} | ${tot_worst:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    kc.ensure_dirs()
    sample = load_sample()
    if args.dry_run:
        dry_run(sample, args.max_usd)
        return
    if args.model:
        run_model(args.model, sample, args.max_usd, args.limit)
    report(sample)


if __name__ == "__main__":
    main()
