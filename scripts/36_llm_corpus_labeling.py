# -*- coding: utf-8 -*-
"""
[36] LLM 전체 코퍼스 라벨링 (gpt-6-luna, 일반 API 호출, Batch 미사용)

전기차 기사 29,757건(news_tagged.csv 와 같은 article_id)에 관련성과 전기차 시장 기준 감성을
붙인다. 기존 규칙 기반 라벨(news_tagged.csv)은 수정하지 않고 별도 파일로 저장한다.

- 입력 텍스트: 제목 + 본문 앞 450자 (사람 검토 표본·34번 벤치마크와 같은 구성)
- API 키: 환경변수 OPENAI_API_KEY
- 비용 상한: 34번과 같은 누적 장부(spend_ledger.json)를 쓰며, 누적 사용액 + 진행 중 호출의
  최악 비용이 --max-usd 를 넘지 않을 때만 새 호출을 보낸다.
- 캐시: article_id 별 결과를 JSONL 에 쌓아 중단 후 이어서 실행해도 다시 과금되지 않는다.

작업(--task)은 두 가지이며 각각 따로 호출한다.
  - sentiment: 34번에서 기준을 통과한 v1 프롬프트 (감성만)
  - relevance: 165건으로 관련성 판정을 검증한 v3 프롬프트 (관련성 열만 분석에 사용)

사용:
  python scripts/36_llm_corpus_labeling.py --task sentiment --dry-run
  python scripts/36_llm_corpus_labeling.py --task sentiment --benchmark-only   # 165건만
  python scripts/36_llm_corpus_labeling.py --task sentiment --eval             # 165건 점검
  python scripts/36_llm_corpus_labeling.py --task sentiment                    # 전체 (캐시 건너뜀)
  python scripts/36_llm_corpus_labeling.py --task relevance
  python scripts/36_llm_corpus_labeling.py --export     # data/processed/news_llm_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

MODEL = "gpt-6-luna"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD / 1M tokens, 일반 호출
MAX_OUTPUT_TOKENS = 400
DEFAULT_MAX_USD = 9.0  # 34번 벤치마크 사용액을 포함한 누적 상한
LABELS = ["긍정", "부정", "중립"]

CACHE_DIR = os.path.join(kc.OUT_DIR, "kg", "llm_cache")
LEDGER = os.path.join(CACHE_DIR, "spend_ledger.json")
EXPORT_CSV = os.path.join(kc.DATA_PROC, "news_llm_labels.csv")

REL_VERSION = "v3-relevance-separate"
REL_PROMPT = """너는 한국 신문 기사를 분류하는 평가자다. 기사 제목과 본문 앞부분을 읽고 두 가지를 판단하라.

1) relevance
- 관련: 기사의 중심 내용이 전기차 시장·보급·구매(전기차 차량, 배터리·충전 등 직접 관련 산업 포함)에 관한 것
- 비관련: 전기차가 한두 번 언급되거나 배경에 불과하고, 중심 내용은 다른 주제(정치, 외교, 방역, 일반 경제, 일반 자동차 등)인 것

2) label (전기차 시장 기준, relevance 와 별개로 판단)
- 긍정: 전기차 보급·판매·구매 여건에 유리 (판매·보급 증가, 보조금 확대, 충전 인프라 확충, 신차 출시, 가격 인하, 기술 개선)
- 부정: 전기차 보급·판매·구매 여건에 불리 (화재·리콜·결함, 판매 부진, 보조금 축소, 충전 불편, 가격 부담, 수요 둔화)
- 중립: 방향이 뚜렷하지 않거나 양쪽이 비슷하게 섞임, 또는 전기차 시장에 대한 내용이 없음

기사 전체의 분위기(정치 공방, 파업, 적자 등)가 아니라 전기차 시장에 대한 방향만 판단한다.
reason 은 한국어 한 문장, 40자 이내."""

REL_SCHEMA = {
    "type": "json_schema",
    "name": "ev_relevance_sentiment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "relevance": {"type": "string", "enum": ["관련", "비관련"]},
            "label": {"type": "string", "enum": LABELS},
            "reason": {"type": "string"},
        },
        "required": ["relevance", "label", "reason"],
        "additionalProperties": False,
    },
}


def _bench34():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "34_llm_sentiment_benchmark.py")
    spec = importlib.util.spec_from_file_location("bench34", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_b34 = _bench34()
# 감성: 34번에서 기준을 통과한 v1 프롬프트를 그대로 사용 / 관련성: 165건 검증한 v3 프롬프트
TASKS = {
    "sentiment": (_b34.PROMPT_VERSION, _b34.SYSTEM_PROMPT, _b34.SCHEMA),
    "relevance": (REL_VERSION, REL_PROMPT, REL_SCHEMA),
}
PROMPT_VERSION, SYSTEM_PROMPT, SCHEMA = TASKS["relevance"]


def cache_path(version: str) -> str:
    return os.path.join(CACHE_DIR, f"corpus_{MODEL}_{version}.jsonl")


CACHE = cache_path(PROMPT_VERSION)


def configure(task: str) -> None:
    global PROMPT_VERSION, SYSTEM_PROMPT, SCHEMA, CACHE
    PROMPT_VERSION, SYSTEM_PROMPT, SCHEMA = TASKS[task]
    CACHE = cache_path(PROMPT_VERSION)


def load_corpus() -> pd.DataFrame:
    """14번과 같은 필터·순서로 article_id 를 재현하고 입력 텍스트를 만든다."""
    csv.field_size_limit(10 ** 7)
    rows, n_ev = [], 0
    with open(kc.ALL_NEWS, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            title = kc.norm_text(row.get("Title"))
            content = kc.norm_text(row.get("Content"))
            if not kc.contains_any(f"{title} {content}", kc.EV_KEYWORDS):
                continue
            n_ev += 1
            excerpt = content[:450] + ("..." if len(content) > 450 else "")
            rows.append({"article_id": f"A{n_ev:06d}", "Url": kc.norm_text(row.get("Url")),
                         "text": f"{title} {excerpt}"[:800]})
    corpus = pd.DataFrame(rows)
    nt = pd.read_csv(kc.NEWS_TAGGED_CSV, encoding="utf-8-sig",
                     usecols=["article_id", "Url", "ym", "newspaper"])
    if len(nt) != len(corpus) or not (nt["Url"].fillna("").values ==
                                      corpus["Url"].fillna("").values).all():
        raise SystemExit("All_news.csv 에서 재현한 article_id 가 news_tagged.csv 와 다르다.")
    return corpus.merge(nt[["article_id", "ym", "newspaper"]], on="article_id")


def benchmark_ids() -> pd.DataFrame:
    s = pd.read_csv(os.path.join(kc.OUT_DIR, "sentiment_review_sample_300.csv"),
                    encoding="utf-8-sig")
    s = s[s["human_sentiment"].notna()][["sample_id", "Url", "human_sentiment"]]
    nt = pd.read_csv(kc.NEWS_TAGGED_CSV, encoding="utf-8-sig",
                     usecols=["article_id", "Url", "sentiment"])
    return s.merge(nt, on="Url")


def cost(in_tok: int, out_tok: int) -> float:
    return in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT


def worst_cost(text: str) -> float:
    return cost(len(SYSTEM_PROMPT) + len(text) + 50, MAX_OUTPUT_TOKENS)


def read_ledger() -> dict:
    if not os.path.exists(LEDGER):
        return {"total_usd": 0.0, "by_model": {}}
    with open(LEDGER, encoding="utf-8") as f:
        return json.load(f)


def rebuild_ledger() -> dict:
    """캐시에 남은 호출별 비용을 모두 합산해 장부를 다시 만든다(캐시가 실제 과금의 기준)."""
    import glob
    by = {}
    for p in glob.glob(os.path.join(CACHE_DIR, "*.jsonl")):
        with open(p, encoding="utf-8") as f:
            by[os.path.basename(p)[:-6]] = sum(json.loads(l).get("cost_usd", 0) for l in f)
    led = {"total_usd": sum(by.values()), "by_model": by}
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(led, f, ensure_ascii=False, indent=2)
    return led


LOCK = os.path.join(CACHE_DIR, "run.lock")


def acquire_lock() -> None:
    if os.path.exists(LOCK):
        with open(LOCK) as f:
            pid = f.read().strip()
        raise SystemExit(f"다른 실행(PID {pid})이 진행 중이다. 그 프로세스가 없으면 {LOCK} 를 지운다.")
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))


def read_cache() -> dict:
    out = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("label") in LABELS:
                    out[rec["article_id"]] = rec
    return out


class Runner:
    def __init__(self, max_usd: float, rpm: int):
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("환경변수 OPENAI_API_KEY 가 없다.")
        self.client = OpenAI(max_retries=8, timeout=60)
        self.max_usd = max_usd
        self.lock = threading.Lock()
        self.interval = 60.0 / rpm
        self.next_slot = time.monotonic()
        self.pace_lock = threading.Lock()
        self.led = rebuild_ledger()
        self.reserved = 0.0
        self.stopped = False
        self.n_fail = 0

    def reserve(self, w: float) -> bool:
        with self.lock:
            if self.stopped or self.led["total_usd"] + self.reserved + w > self.max_usd:
                self.stopped = True
                return False
            self.reserved += w
            return True

    def settle(self, w: float, c: float, rec: dict, fout) -> None:
        with self.lock:
            self.reserved -= w
            self.led["total_usd"] += c
            key = os.path.basename(CACHE)[:-6]
            self.led["by_model"][key] = self.led["by_model"].get(key, 0.0) + c
            with open(LEDGER, "w", encoding="utf-8") as f:
                json.dump(self.led, f, ensure_ascii=False, indent=2)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    def one(self, row, fout) -> None:
        w = worst_cost(row.text)
        if not self.reserve(w):
            return
        with self.pace_lock:
            wait = self.next_slot - time.monotonic()
            self.next_slot = max(self.next_slot, time.monotonic()) + self.interval
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self.client.responses.create(
                model=MODEL, instructions=SYSTEM_PROMPT, input=row.text,
                text={"format": SCHEMA}, reasoning={"effort": "low"},
                max_output_tokens=MAX_OUTPUT_TOKENS)
        except Exception as e:
            with self.lock:
                self.reserved -= w
                self.n_fail += 1
            print(f"  [실패] {row.article_id}: {type(e).__name__} {str(e)[:120]}")
            return
        u = resp.usage
        c = cost(u.input_tokens, u.output_tokens)
        rel = label = None
        reason = ""
        try:
            obj = json.loads(resp.output_text)
            rel, label, reason = obj.get("relevance"), obj["label"], obj["reason"]
        except Exception:
            reason = f"[파싱 실패 status={getattr(resp, 'status', '')}]"
            with self.lock:
                self.n_fail += 1
        rec = {"article_id": row.article_id, "relevance": rel, "label": label,
               "reason": reason, "input_tokens": u.input_tokens,
               "output_tokens": u.output_tokens, "cost_usd": round(c, 7),
               "model_snapshot": getattr(resp, "model", MODEL),
               "prompt_version": PROMPT_VERSION, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.settle(w, c, rec, fout)

    def run(self, todo: pd.DataFrame, workers: int) -> None:
        t0, done = time.time(), 0
        with open(CACHE, "a", encoding="utf-8") as fout, \
                ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self.one, r, fout) for r in todo.itertuples(index=False)]
            for fu in as_completed(futs):
                fu.result()
                done += 1
                if done % 500 == 0 or done == len(futs):
                    el = time.time() - t0
                    print(f"  {done}/{len(futs)}  누적 ${self.led['total_usd']:.4f}  "
                          f"실패 {self.n_fail}  {el / 60:.1f}분", flush=True)
        if self.stopped:
            print(f"[중단] 비용 상한 ${self.max_usd:.2f} 도달. 누적 ${self.led['total_usd']:.4f}")


def evaluate() -> None:
    from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
    cache = read_cache()
    b = benchmark_ids()
    b["llm_rel"] = b["article_id"].map(lambda a: cache.get(a, {}).get("relevance"))
    b["llm"] = b["article_id"].map(lambda a: cache.get(a, {}).get("label"))
    if b["llm"].isna().any():
        raise SystemExit(f"165건 중 {b['llm'].isna().sum()}건이 아직 없다.")

    rv = pd.read_csv(os.path.join(kc.OUT_DIR, "sentiment_relabel_review.csv"),
                     encoding="utf-8-sig", dtype=str).fillna("")
    key = pd.read_csv(os.path.join(CACHE_DIR, "relabel_review_key.csv"), encoding="utf-8-sig")
    rv = rv.merge(key, on=["review_id", "sample_id"])
    rv["ev"] = rv.human_sentiment_ev.where(rv.human_relevance_ev == "관련", "중립")
    b = b.merge(rv[["sample_id", "human_relevance_ev", "ev", "group"]], on="sample_id", how="left")
    b["human_ev"] = b["ev"].fillna(b["human_sentiment"])

    old = pd.read_csv(os.path.join(kc.TABLE_DIR, "tbl_llm_sentiment_labels.csv"),
                      encoding="utf-8-sig")
    old = old[old.model == MODEL].set_index("sample_id")["llm_label"]
    b["llm_v1"] = b["sample_id"].map(old)
    b["llm_f"] = b["llm"].where(b["llm_rel"] == "관련", "중립")

    if PROMPT_VERSION == TASKS["sentiment"][0]:
        agree = (b.llm == b.llm_v1).mean()
        ctrl = b[b.group == "control"]
        lines = [f"# 전체 실행용 감성 라벨 재현성 점검 ({MODEL}, {PROMPT_VERSION})\n",
                 f"- 34번 벤치마크 v1 결과와 라벨 일치율(165건): {agree:.3f}",
                 f"- 무작위 대조군 {len(ctrl)}건 정확도(수정 라벨): {(ctrl.llm == ctrl.human_ev).mean():.3f}"]
        from sklearn.metrics import precision_recall_fscore_support as prfs
        pr, rc, f1, _ = prfs(b.human_ev, b.llm, labels=LABELS, zero_division=0)
        mf = prfs(b.human_ev, b.llm, labels=LABELS, average="macro", zero_division=0)[2]
        lines.append(f"- 수정 라벨 macro-F1 {mf:.3f}, 부정 P/R/F {pr[1]:.3f}/{rc[1]:.3f}/{f1[1]:.3f}")
        path = os.path.join(kc.TABLE_DIR, f"llm_corpus_prompt_validation_{PROMPT_VERSION}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("\n".join(lines))
        return

    def m(y, p):
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
        mf = precision_recall_fscore_support(y, p, labels=LABELS, average="macro",
                                             zero_division=0)[2]
        return (f"macro-F1 {mf:.3f}  κ {cohen_kappa_score(y, p):.3f}  "
                f"부정 P/R/F {pr[1]:.3f}/{rc[1]:.3f}/{f1[1]:.3f}")

    rev = b[b.human_relevance_ev.fillna("") != ""]
    rel_agree = (rev.llm_rel == rev.human_relevance_ev).mean()
    ctrl = b[b.group == "control"]
    lines = [f"# LLM 전체 라벨링 프롬프트 검증 ({MODEL}, {PROMPT_VERSION})\n",
             f"- 관련성 일치(사람 재검토 {len(rev)}건): {rel_agree:.3f}",
             f"  - 사람 비관련 {int((rev.human_relevance_ev == '비관련').sum())}건 중 LLM 비관련 "
             f"{int(((rev.human_relevance_ev == '비관련') & (rev.llm_rel == '비관련')).sum())}건",
             f"- 165건 중 LLM 비관련 판정: {int((b.llm_rel == '비관련').sum())}건",
             f"- 무작위 대조군 {len(ctrl)}건 정확도(수정 라벨): 현재 {(ctrl.llm == ctrl.human_ev).mean():.3f}"
             f" / 현재+관련성 필터 {(ctrl.llm_f == ctrl.human_ev).mean():.3f}"
             f" / v1 {(ctrl.llm_v1 == ctrl.human_ev).mean():.3f}"
             f" / 규칙 {(ctrl.sentiment == ctrl.human_ev).mean():.3f}",
             "", "## 165건 성능", "",
             f"- 원 라벨    현재: {m(b.human_sentiment, b.llm)}",
             f"- 원 라벨    v1: {m(b.human_sentiment, b.llm_v1)}",
             f"- 수정 라벨  현재: {m(b.human_ev, b.llm)}",
             f"- 수정 라벨  현재+관련성 필터: {m(b.human_ev, b.llm_f)}",
             f"- 수정 라벨  v1: {m(b.human_ev, b.llm_v1)}",
             f"- 수정 라벨 규칙: {m(b.human_ev, b.sentiment)}",
             "", "사전 기준: 관련성 일치 ≥ 0.80, 대조군 정확도 ≥ v1(0.739), 수정 라벨 부정 F1 ≥ 0.85"]
    path = os.path.join(kc.TABLE_DIR, f"llm_corpus_prompt_validation_{PROMPT_VERSION}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def export(corpus: pd.DataFrame) -> None:
    configure("relevance")
    cols = {"relevance": "llm_relevance", "label": "llm_label", "reason": "llm_reason"}
    lab = pd.DataFrame(read_cache().values())
    out = corpus[["article_id", "ym", "newspaper"]].merge(
        lab[["article_id", *cols, "prompt_version"]].rename(columns=cols),
        on="article_id", how="left")
    out.to_csv(EXPORT_CSV, index=False, encoding="utf-8-sig")
    print(f"[36] 저장 {EXPORT_CSV}: 라벨 {out.llm_label.notna().sum()}/{len(out)}건")
    print(out.llm_relevance.value_counts(dropna=False).to_dict())
    print(out[out.llm_relevance == "관련"].llm_label.value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="sentiment")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=170, help="분당 요청 수 상한 (계정 한도: 500 RPM, 200k TPM)")
    ap.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    args = ap.parse_args()
    kc.ensure_dirs()
    os.makedirs(CACHE_DIR, exist_ok=True)
    configure(args.task)

    if args.eval:
        evaluate()
        return
    corpus = load_corpus()
    if args.export:
        export(corpus)
        return
    if args.benchmark_only:
        corpus = corpus[corpus.article_id.isin(benchmark_ids().article_id)]
    cache = read_cache()
    todo = corpus[~corpus.article_id.isin(cache)]
    if args.limit:
        todo = todo.head(args.limit)
    led = read_ledger()
    exp = sum(cost(len(SYSTEM_PROMPT) + len(t), 60) for t in todo.text)
    worst = sum(worst_cost(t) for t in todo.text)
    print(f"[36] 대상 {len(corpus)}건, 캐시 {len(cache)}건, 이번 호출 {len(todo)}건")
    print(f"     예상 ${exp:.2f} (최악 ${worst:.2f}), 누적 ${led['total_usd']:.4f} / 상한 ${args.max_usd:.2f}")
    if args.dry_run or todo.empty:
        return
    acquire_lock()
    try:
        Runner(args.max_usd, args.rpm).run(todo, args.workers)
    finally:
        os.remove(LOCK)


if __name__ == "__main__":
    main()
