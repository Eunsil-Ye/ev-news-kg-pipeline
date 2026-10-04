# -*- coding: utf-8 -*-
"""
전체 파이프라인 일괄 실행기

이 파일이 있는 폴더를 프로젝트 루트로 보고, scripts/ 의 단계를 README 의 순서대로 실행한다.
사람 라벨 파일과 LLM 캐시(outputs/kg/llm_cache/)가 있으면 API 호출 없이 모든 표와 그림이 재생성된다.
API 호출 단계(34 라벨링, 36 전수 라벨링)는 포함하지 않으며 README 의 수동 절차를 따른다.

사용:
    python run_pipeline.py                 # 전 단계
    python run_pipeline.py --stage core    # 한 단계만
    python run_pipeline.py --with-bert     # 24번 BERT 비교 포함 (모델 다운로드 필요)
    python run_pipeline.py --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    ("prep", "데이터 준비", [
        ["build_total_vehicle_registration_csv.py"],
        ["38_charger_tables.py"],
    ]),
    ("core", "KG 구축과 월별 지표", [
        ["14_extract_entities.py"],
        ["15_build_temporal_kg.py"],
        ["16_kg_monthly_indicators.py"],
        ["17_merge_kg_into_panel.py"],
    ]),
    ("main", "본 분석", [
        ["18_models_with_kg.py"],
        ["20_granger_multi_correction.py"],
        ["21_national_forecast_lean_kg.py"],
        ["23_event_breakpoint.py"],
    ]),
    ("validation", "측정 타당성", [
        ["make_sentiment_review_sample.py"],
        ["analyze_sentiment_mismatches.py"],
        ["kg_sentiment_reliability.py"],
        ["25_entity_frame_review_sample.py"],
        ["31_enrich_review_context.py"],
        ["34_llm_sentiment_benchmark.py", "--report"],
        ["35_sentiment_relabel_review.py", "report"],
        ["36_llm_corpus_labeling.py", "--eval"],
        ["36_llm_corpus_labeling.py", "--export"],
    ]),
    ("audit", "KG 품질 감사와 정제판", [
        ["fp_agent_labels.py"],
        ["28_kg_quality_audit.py"],
        ["29_kg_clean_variant.py"],
        ["30_kg_clean_robustness.py"],
    ]),
    ("robust", "강건성", [
        ["22_panel_robustness.py"],
        ["26_media_heterogeneity.py"],
        ["27_relevance_classifier.py"],
        ["33_period_sensitivity.py"],
        ["37_llm_h1_retest.py"],
    ]),
    ("figures", "그림", [
        ["19_kg_event_subgraphs.py"],
        ["32_kg_visualizations.py"],
        ["40_thesis_event_figures.py"],
    ]),
]
BERT_STEP = ["24_bert_sentiment_benchmark.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[s[0] for s in STAGES], action="append")
    ap.add_argument("--with-bert", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    stages = [s for s in STAGES if not args.stage or s[0] in args.stage]
    if args.with_bert:
        stages = [(k, n, steps + [BERT_STEP] if k == "validation" else steps) for k, n, steps in stages]
    if args.list:
        for key, name, steps in stages:
            print(f"[{key}] {name}")
            for st in steps:
                print("    " + " ".join(st))
        return

    # 집합 순회 순서를 고정해 동률 정렬과 네트워크 배치가 실행마다 같도록 한다.
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    t0 = time.time()
    for key, name, steps in stages:
        print(f"\n===== [{key}] {name} =====", flush=True)
        for st in steps:
            t = time.time()
            r = subprocess.run([sys.executable, os.path.join("scripts", st[0]), *st[1:]], cwd=ROOT, env=env)
            print(f"--- {' '.join(st)}: {'OK' if r.returncode == 0 else 'FAIL'} ({time.time() - t:.0f}s)", flush=True)
            if r.returncode != 0:
                sys.exit(r.returncode)
    print(f"\n전체 완료 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
