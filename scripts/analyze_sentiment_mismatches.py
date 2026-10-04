# -*- coding: utf-8 -*-
"""
감성 검증 표본에서 규칙 기반 라벨과 사람 라벨이 다른 기사를 추출해 오류 유형을 분류한다.

입력: outputs/sentiment_review_sample_300.csv (사람 라벨 기입본)
출력: outputs/sentiment_mismatch_analysis.csv, outputs/sentiment_mismatch_summary.md
      (kg_sentiment_reliability.py 와 28번 검진 B4 의 입력)
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "sentiment_review_sample_300.csv"
OUT_CSV = ROOT / "outputs" / "sentiment_mismatch_analysis.csv"
OUT_MD = ROOT / "outputs" / "sentiment_mismatch_summary.md"


def clean(value: str | None) -> str:
    return (value or "").strip()


def split_hits(value: str | None) -> list[str]:
    return [item.strip() for item in clean(value).split(",") if item.strip()]


def infer_issue_type(row: dict[str, str]) -> str:
    pred = clean(row.get("predicted_sentiment"))
    human = clean(row.get("human_sentiment"))
    pos_hits = split_hits(row.get("positive_hits"))
    neg_hits = split_hits(row.get("negative_hits"))

    if pred == "중립" and human in {"긍정", "부정"}:
        return "중립 과다 판정"
    if pred == "긍정" and human == "부정":
        return "긍정 과다 판정"
    if pred == "부정" and human == "긍정":
        return "부정 과다 판정"
    if pred in {"긍정", "부정"} and human == "중립":
        return "감성 과다 판정"
    if pos_hits and neg_hits:
        return "긍정/부정 단어 혼재"
    return "기타"


def suggest_rule(row: dict[str, str]) -> str:
    title = clean(row.get("Title"))
    excerpt = clean(row.get("Content_excerpt"))
    text = f"{title} {excerpt}"
    pred = clean(row.get("predicted_sentiment"))
    human = clean(row.get("human_sentiment"))

    if "화재" in text or "리콜" in text or "결함" in text:
        if human == "부정":
            return "화재/리콜/결함 문구는 부정 우선 규칙 후보"
        return "화재/리콜/결함이 예방/대응 맥락인지 확인"
    if "충전" in text and any(word in text for word in ["확대", "확충", "개선", "설치"]):
        return "충전 인프라 확대/확충/개선 문구는 긍정 규칙 후보"
    if "보조금" in text and any(word in text for word in ["축소", "삭감", "중단"]):
        return "보조금 축소/삭감/중단 문구는 부정 규칙 후보"
    if "보조금" in text and any(word in text for word in ["지원", "확대", "지급"]):
        return "보조금 지원/확대/지급 문구는 긍정 규칙 후보"
    if any(word in text for word in ["출시", "신차", "판매 증가", "수출"]):
        return "신차 출시/판매 증가/수출 문구는 긍정 규칙 후보"
    if pred == "중립" and human != "중립":
        return "중립 임계값 완화 또는 감성 문구 추가 후보"
    if human == "중립":
        return "단순 정보/기업 동향은 중립 예외 규칙 후보"
    return "제목과 본문 앞부분을 보고 문구 단위 규칙 추가 검토"


def main() -> None:
    with INPUT.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    reviewed_related = [
        row
        for row in rows
        if clean(row.get("human_relevance")) == "관련" and clean(row.get("human_sentiment"))
    ]
    mismatches = [
        row
        for row in reviewed_related
        if clean(row.get("predicted_sentiment")) != clean(row.get("human_sentiment"))
    ]

    output_rows: list[dict[str, str]] = []
    for row in mismatches:
        pos_hits = split_hits(row.get("positive_hits"))
        neg_hits = split_hits(row.get("negative_hits"))
        output_rows.append(
            {
                "sample_id": clean(row.get("sample_id")),
                "Date": clean(row.get("Date")),
                "newspaper": clean(row.get("newspaper")),
                "Title": clean(row.get("Title")),
                "predicted_sentiment": clean(row.get("predicted_sentiment")),
                "human_sentiment": clean(row.get("human_sentiment")),
                "sentiment_score": clean(row.get("sentiment_score")),
                "issue_type": infer_issue_type(row),
                "suggested_rule": suggest_rule(row),
                "positive_hits": ", ".join(pos_hits),
                "negative_hits": ", ".join(neg_hits),
                "matched_ev_keywords": clean(row.get("matched_ev_keywords")),
                "topic_policy": clean(row.get("topic_policy")),
                "topic_charging": clean(row.get("topic_charging")),
                "topic_safety": clean(row.get("topic_safety")),
                "topic_price": clean(row.get("topic_price")),
                "topic_battery": clean(row.get("topic_battery")),
                "topic_new_model": clean(row.get("topic_new_model")),
                "Url": clean(row.get("Url")),
                "Content_excerpt": clean(row.get("Content_excerpt")),
            }
        )

    fieldnames = [
        "sample_id",
        "Date",
        "newspaper",
        "Title",
        "predicted_sentiment",
        "human_sentiment",
        "sentiment_score",
        "issue_type",
        "suggested_rule",
        "positive_hits",
        "negative_hits",
        "matched_ev_keywords",
        "topic_policy",
        "topic_charging",
        "topic_safety",
        "topic_price",
        "topic_battery",
        "topic_new_model",
        "Url",
        "Content_excerpt",
    ]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    issue_counts = Counter(row["issue_type"] for row in output_rows)
    pair_counts = Counter(
        f"{clean(row.get('predicted_sentiment'))} -> {clean(row.get('human_sentiment'))}"
        for row in mismatches
    )
    positive_hit_counts = Counter()
    negative_hit_counts = Counter()
    for row in mismatches:
        positive_hit_counts.update(split_hits(row.get("positive_hits")))
        negative_hit_counts.update(split_hits(row.get("negative_hits")))

    def table(headers: list[str], body: list[list[str | int]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for item in body:
            lines.append("| " + " | ".join(str(value) for value in item) + " |")
        return "\n".join(lines)

    md = [
        "# 감정분석 오분류 분석",
        "",
        "## 요약",
        "",
        table(
            ["항목", "값"],
            [
                ["수작업 감정 라벨이 있는 관련 기사", len(reviewed_related)],
                ["오분류 기사", len(mismatches)],
                ["자동 라벨 일치 기사", len(reviewed_related) - len(mismatches)],
            ],
        ),
        "",
        "## 자동 라벨 -> 수작업 라벨",
        "",
        table(["오분류 방향", "건수"], [[key, value] for key, value in pair_counts.most_common()]),
        "",
        "## 추정 오류 유형",
        "",
        table(["오류 유형", "건수"], [[key, value] for key, value in issue_counts.most_common()]),
        "",
        "## 오분류 기사에서 자주 등장한 긍정 hit",
        "",
        table(["긍정 hit", "건수"], [[key, value] for key, value in positive_hit_counts.most_common(20)]),
        "",
        "## 오분류 기사에서 자주 등장한 부정 hit",
        "",
        table(["부정 hit", "건수"], [[key, value] for key, value in negative_hit_counts.most_common(20)]),
        "",
        "## 다음 작업",
        "",
        "1. `sentiment_mismatch_analysis.csv`에서 `issue_type`과 `suggested_rule`을 확인한다.",
        "2. 반복적으로 틀리는 문구를 긍정/부정/중립 예외 사전에 추가한다.",
        "3. 보정된 규칙을 전체 기사에 다시 적용한다.",
        "4. 수작업 검토 200건 기준 일치율이 개선되는지 재평가한다.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"reviewed_related={len(reviewed_related)}")
    print(f"mismatches={len(mismatches)}")
    print(f"out_csv={OUT_CSV.resolve()}")
    print(f"out_md={OUT_MD.resolve()}")


if __name__ == "__main__":
    main()
