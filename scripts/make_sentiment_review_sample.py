# -*- coding: utf-8 -*-
"""
감성 분류 사람 검증용 표본 300건 생성 (예측 감성별 100건 층화추출).

출력: outputs/sentiment_review_sample_300.csv
검토자가 human_relevance / human_sentiment 열을 채운 뒤 24·27·28·34·35·36번과
analyze_sentiment_mismatches.py 가 이 파일을 읽는다.
사람 라벨을 지우지 않도록 파일이 이미 있으면 새로 만들지 않는다.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "All_news.csv"
OUT_DIR = ROOT / "outputs"
OUTPUT = OUT_DIR / "sentiment_review_sample_300.csv"
RANDOM_SEED = 20260610
TARGET_TOTAL = 300

EV_KEYWORDS = [
    "전기차",
    "전기자동차",
    "EV",
    "무공해차",
    "친환경차",
    "전동화",
    "아이오닉",
    "아이오닉5",
    "아이오닉6",
    "코나 일렉트릭",
    "코나 전기",
    "테슬라",
    "모델3",
    "모델Y",
    "폴스타",
    "전기 SUV",
    "전기트럭",
    "충전기",
    "충전소",
    "급속충전",
    "완속충전",
    "배터리",
    "전기차 보조금",
]

POSITIVE_WORDS = [
    "성장",
    "증가",
    "확대",
    "확충",
    "개선",
    "호조",
    "인기",
    "수요",
    "판매 증가",
    "판매 호조",
    "출시",
    "신차",
    "수출",
    "투자",
    "협력",
    "개발",
    "상용화",
    "보급",
    "지원",
    "혜택",
    "할인",
    "가격 인하",
    "인하",
    "돌파",
    "최대",
    "선도",
    "친환경",
    "고성능",
    "고효율",
    "충전 편의",
    "충전 인프라 확대",
    "보조금 확대",
    "보조금 지원",
    "수상",
    "흑자",
    "회복",
    "강화",
]

NEGATIVE_WORDS = [
    "화재",
    "폭발",
    "리콜",
    "결함",
    "사고",
    "우려",
    "불안",
    "위험",
    "부족",
    "지연",
    "감소",
    "하락",
    "부진",
    "적자",
    "축소",
    "중단",
    "취소",
    "논란",
    "반발",
    "갈등",
    "고장",
    "침체",
    "불편",
    "부담",
    "비싸",
    "고가",
    "급발진",
    "배터리 결함",
    "충전난",
    "보조금 축소",
    "보조금 삭감",
    "가격 부담",
    "판매 감소",
    "둔화",
    "피해",
    "소송",
    "규제",
]

TOPIC_KEYWORDS = {
    "topic_policy": ["보조금", "환경부", "국토교통부", "정책", "규제", "무공해차", "탄소", "보급사업", "지자체"],
    "topic_charging": ["충전", "충전기", "충전소", "급속충전", "완속충전", "인프라"],
    "topic_safety": ["화재", "폭발", "리콜", "결함", "사고", "안전", "급발진"],
    "topic_price": ["가격", "할인", "인하", "보조금", "리스", "렌트", "구매비"],
    "topic_battery": ["배터리", "전고체", "LFP", "양극재", "음극재", "셀", "ESS"],
    "topic_new_model": ["출시", "신차", "아이오닉", "코나", "테슬라", "모델3", "모델Y", "EV3", "EV6", "EV9"],
}


def norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def contains_any(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    for keyword in keywords:
        if keyword == "EV":
            if re.search(r"(?<![A-Za-z])EV(?![A-Za-z])", text):
                return True
            continue
        if keyword.lower() in low:
            return True
    return False


def hit_words(text: str, keywords: list[str]) -> list[str]:
    low = text.lower()
    hits: list[str] = []
    for keyword in keywords:
        if keyword == "EV":
            if re.search(r"(?<![A-Za-z])EV(?![A-Za-z])", text):
                hits.append(keyword)
            continue
        if keyword.lower() in low:
            hits.append(keyword)
    return hits


def classify(text: str) -> tuple[str, float, list[str], list[str]]:
    pos_hits = hit_words(text, POSITIVE_WORDS)
    neg_hits = hit_words(text, NEGATIVE_WORDS)
    total = len(pos_hits) + len(neg_hits)
    score = 0.0 if total == 0 else (len(pos_hits) - len(neg_hits)) / total
    if score >= 0.25:
        label = "긍정"
    elif score <= -0.25:
        label = "부정"
    else:
        label = "중립"
    return label, score, pos_hits, neg_hits


def load_ev_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with INPUT.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            title = norm_text(row.get("Title"))
            content = norm_text(row.get("Content"))
            text = f"{title} {content}"
            if not contains_any(text, EV_KEYWORDS):
                continue

            matched_ev_keywords = hit_words(text, EV_KEYWORDS)
            label, score, pos_hits, neg_hits = classify(text)
            topics = {name: int(contains_any(text, keywords)) for name, keywords in TOPIC_KEYWORDS.items()}
            content_excerpt = content[:450] + ("..." if len(content) > 450 else "")
            rows.append(
                {
                    "Date": row.get("Date", ""),
                    "Year": row.get("Year", ""),
                    "Month": row.get("Month", ""),
                    "newspaper": row.get("newspaper", ""),
                    "Title": title,
                    "matched_ev_keywords": ", ".join(matched_ev_keywords[:8]),
                    "predicted_sentiment": label,
                    "sentiment_score": f"{score:.3f}",
                    "positive_hits": ", ".join(pos_hits[:8]),
                    "negative_hits": ", ".join(neg_hits[:8]),
                    **topics,
                    "human_relevance": "",
                    "human_sentiment": "",
                    "review_memo": "",
                    "Url": row.get("Url", ""),
                    "Content_excerpt": content_excerpt,
                }
            )
    return rows


def make_sample(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    random.seed(RANDOM_SEED)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["predicted_sentiment"]].append(row)

    sample: list[dict[str, str]] = []
    label_targets = {"긍정": 100, "중립": 100, "부정": 100}
    for label, target in label_targets.items():
        pool = by_label.get(label, [])
        random.shuffle(pool)
        sample.extend(pool[: min(target, len(pool))])

    if len(sample) < TARGET_TOTAL:
        chosen_ids = {id(row) for row in sample}
        remaining = [row for row in rows if id(row) not in chosen_ids]
        random.shuffle(remaining)
        sample.extend(remaining[: TARGET_TOTAL - len(sample)])

    sample = sorted(sample[:TARGET_TOTAL], key=lambda row: (row["Date"], row["newspaper"], row["Title"]))
    for index, row in enumerate(sample, 1):
        row["sample_id"] = f"S{index:03d}"
    return sample


def write_sample(sample: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "sample_id",
        "Date",
        "Year",
        "Month",
        "newspaper",
        "Title",
        "matched_ev_keywords",
        "predicted_sentiment",
        "sentiment_score",
        "positive_hits",
        "negative_hits",
        "topic_policy",
        "topic_charging",
        "topic_safety",
        "topic_price",
        "topic_battery",
        "topic_new_model",
        "human_relevance",
        "human_sentiment",
        "review_memo",
        "Url",
        "Content_excerpt",
    ]
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample)


def main() -> None:
    if OUTPUT.exists():
        print(f"이미 존재하여 새로 만들지 않음: {OUTPUT}")
        return
    rows = load_ev_rows()
    sample = make_sample(rows)
    write_sample(sample)
    print(f"output={OUTPUT.resolve()}")
    print(f"ev_related_rows={len(rows)}")
    print(f"sample_rows={len(sample)}")
    print(f"all_ev_predicted_counts={dict(Counter(row['predicted_sentiment'] for row in rows))}")
    print(f"sample_predicted_counts={dict(Counter(row['predicted_sentiment'] for row in sample))}")
    print(f"sample_year_range={min(row['Date'] for row in sample)}~{max(row['Date'] for row in sample)}")


if __name__ == "__main__":
    main()
