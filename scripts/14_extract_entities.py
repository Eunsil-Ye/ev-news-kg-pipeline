# -*- coding: utf-8 -*-
"""
[14] 전체 코퍼스 개체/프레임/감성 태깅

입력: All_news.csv (30,490건)
처리:
  - 전기차 관련 기사 필터링
  - 기사별 감성(긍정/중립/부정) + 점수
  - 기사별 프레임(9종) 태깅
  - 기사별 개체(기업/모델/정책/사건) 태깅
출력: data/processed/news_tagged.csv

이 태깅 파일이 이후 KG 구축(15), 지표 산출(16), 사건 서브그래프(19)의 공통 입력이다.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc


def main():
    kc.ensure_dirs()
    csv.field_size_limit(10 ** 7)

    n_total = 0
    n_ev = 0
    rows_out = []
    frame_counter = Counter()
    entity_counter = Counter()
    senti_counter = Counter()

    with open(kc.ALL_NEWS, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            title = kc.norm_text(row.get("Title"))
            content = kc.norm_text(row.get("Content"))
            text = f"{title} {content}"
            # 관련성 필터는 전문, 감성/프레임/개체 태깅은 제목+본문 앞 1000자를 쓴다
            text_clf = f"{title} {content[:1000]}"

            if not kc.contains_any(text, kc.EV_KEYWORDS):
                continue
            n_ev += 1

            label, score, n_pos, n_neg = kc.classify_sentiment(text_clf)
            frames = kc.detect_frames(text_clf)
            entities = kc.detect_entities(text_clf)

            senti_counter[label] += 1
            for fr in frames:
                frame_counter[fr] += 1
            for etype, canon in entities:
                entity_counter[f"{etype}:{canon}"] += 1

            year = kc.norm_text(row.get("Year"))
            month = kc.norm_text(row.get("Month"))
            ym = ""
            if year and month:
                ym = f"{int(year):04d}-{int(month):02d}"

            rows_out.append({
                "article_id": f"A{n_ev:06d}",
                "Date": kc.norm_text(row.get("Date")),
                "ym": ym,
                "Year": year,
                "Month": month,
                "newspaper": kc.norm_text(row.get("newspaper")),
                "Title": title,
                "sentiment": label,
                "sentiment_score": f"{score:.4f}",
                "n_pos": n_pos,
                "n_neg": n_neg,
                "frames": ";".join(frames),
                "entities": ";".join(f"{t}:{c}" for t, c in entities),
                "Url": kc.norm_text(row.get("Url")),
            })

    fieldnames = [
        "article_id", "Date", "ym", "Year", "Month", "newspaper", "Title",
        "sentiment", "sentiment_score", "n_pos", "n_neg", "frames", "entities", "Url",
    ]
    with open(kc.NEWS_TAGGED_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[14] 전체 기사 {n_total} / 전기차 관련 {n_ev}")
    print(f"[14] 감성 분포: {dict(senti_counter)}")
    print(f"[14] 프레임 상위: {frame_counter.most_common()}")
    print(f"[14] 개체 상위 15: {entity_counter.most_common(15)}")
    print(f"[14] 저장: {kc.NEWS_TAGGED_CSV}")


if __name__ == "__main__":
    main()
