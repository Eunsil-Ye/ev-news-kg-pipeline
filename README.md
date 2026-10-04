# 지식그래프 기반 전기차 뉴스 담론 특성 추출 및 등록 순증가 선행성 평가 — 분석 코드

뉴스 기사에서 규칙 기반으로 개체·프레임·감성을 태깅해 월별 시간 지식그래프를 만들고,
그 그래프에서 담론 특성을 계산해 전기차 등록 순증가에 대한 선행성을 평가하는 파이프라인이다.
`run_pipeline.py` 한 번으로 논문에 쓰인 표 51개와 그림 17개를 다시 만든다.

## 폴더 구성

```
code_submission/
├── README.md
├── requirements.txt
├── run_pipeline.py          일괄 실행기
└── scripts/                 분석 스크립트 31개 + 공용 모듈 2개
```

입력 자료는 저작권·용량 문제로 포함하지 않는다. 아래 경로에 두면 모든 스크립트가 이 폴더를 프로젝트 루트로 인식한다.

## 실행 환경

Python 3.11.6, Windows 11에서 검증했다.

```
pip install -r requirements.txt
python run_pipeline.py            # 전 단계, 약 2분
python run_pipeline.py --list     # 단계와 실행 순서 확인
python run_pipeline.py --stage core --stage main
```

`run_pipeline.py`는 `PYTHONHASHSEED=0`으로 하위 스크립트를 실행한다. 집합 순회 순서가 고정되어
동률 엣지의 정렬과 네트워크 그림 배치가 실행마다 같아지며, 같은 입력이면 산출물이 바이트 단위로 같다.

## 입력 자료

| 경로 | 내용 | 비고 |
|---|---|---|
| `All_news.csv` | 뉴스 기사 원문 30,490건 | 수집 자료 |
| `201801_202605_전기차등록현황.xls` | 시도별 월말 전기차 등록 대수 | 국토교통부 |
| `자동차등록/*.csv` | 시도별 월별 전체 자동차 등록 대수 | 국토교통부 통계누리 |
| `data/processed/ev_charger_region_speed_snapshots.csv` | 권역 × 충전속도 × 기준월 충전기 대수 | 환경부 충전기 현황 xls를 정리한 자료 |
| `outputs/sentiment_review_sample_300.csv` | 감성·관련성 사람 라벨 300건 | 사람 라벨 |
| `outputs/entity_frame_review_sample.csv` | 개체–프레임 연결 사람 판정 | 사람 라벨 |
| `outputs/kg_falsepositive_review_sample.csv` | 개체 오탐 판정 100건 | 판정 라벨 |
| `outputs/sentiment_relabel_review.csv` | LLM–사람 불일치 재검토 46건 | 사람 라벨 |
| `outputs/kg/llm_cache/` | LLM 호출 결과 캐시(jsonl)와 비용 기록 | API 재호출 없이 재현용 |

사람 라벨 파일이 없으면 해당 스크립트가 빈 표본을 새로 만들고 멈춘다(아래 "사람 라벨과 API 단계").

## 스크립트 구성 (분석 스크립트 31개 + 공용 모듈 2개)

번호는 개발 순서에 따른 식별자다. 몇몇 스크립트가 다른 번호 스크립트를 모듈 이름으로 불러 쓰므로
파일 이름을 바꾸지 않는다. 실행 순서는 아래 단계 순서를 따른다.

### 공용 모듈 (2개)

| 파일 | 역할 |
|---|---|
| `kg_common.py` | 경로, 분석기간(2018-01~2025-10), 유의수준(α=0.05), 감성 사전·프레임 사전·개체 별칭 사전, 기사 태깅 함수 |
| `kg_clean_rules.py` | 정제판 KG 규칙: 최장일치 우선(C1), 단어경계(C2), 동어반복 개체–프레임 쌍 21개 제외(C3) |

### 1단계 데이터 준비 (2개) — `--stage prep`

| 파일 | 입력 → 출력 |
|---|---|
| `build_total_vehicle_registration_csv.py` | 자동차등록 원자료 → 월 × 시도 전체 등록 대수 |
| `38_charger_tables.py` | 충전기 스냅샷 → 표 3.4·3.5와 기준월 목록 |

### 2단계 KG 구축과 월별 지표 (4개) — `--stage core`

| 파일 | 입력 → 출력 |
|---|---|
| `14_extract_entities.py` | `All_news.csv` → 관련성 필터와 감성·프레임·개체 태깅 (`news_tagged.csv`) |
| `15_build_temporal_kg.py` | 태깅 결과 → 월별 노드·엣지와 전 기간 통합 그래프, 대화형 HTML |
| `16_kg_monthly_indicators.py` | 월별 그래프 → 담론 특성 18종 (`kg_monthly_features.csv`) |
| `17_merge_kg_into_panel.py` | 담론 특성 + 등록·충전기·전체 등록 → 전국 시계열과 17개 시도 패널 |

### 3단계 본 분석 (4개) — `--stage main`

| 파일 | 내용 |
|---|---|
| `18_models_with_kg.py` | 전국 Granger, 확장창 예측, 패널 고정효과(월 클러스터 SE). 21·22·30·33·37번이 함수를 재사용 |
| `20_granger_multi_correction.py` | 담론 특성 15종 양방향 Granger, 정방향에 BH·Bonferroni 보정, VAR-FEVD·충격반응 |
| `21_national_forecast_lean_kg.py` | 4단계 모형 확장창 예측과 블록 부트스트랩 RMSE 차이 |
| `23_event_breakpoint.py` | OLS-CUSUM과 supF 구조 전환점 탐지 |

### 4단계 측정 타당성 (9개) — `--stage validation`

| 파일 | 내용 |
|---|---|
| `make_sentiment_review_sample.py` | 사람 라벨용 300건 층화 표본 생성 (파일이 있으면 건너뜀) |
| `analyze_sentiment_mismatches.py` | 규칙 감성과 사람 라벨 불일치 유형 분석 |
| `kg_sentiment_reliability.py` | 감성 라벨 신뢰도(κ, 혼동행렬), 오류 패턴 그림 |
| `24_bert_sentiment_benchmark.py` | 공개 한국어 BERT 감성 모형과 비교 (`--with-bert`) |
| `25_entity_frame_review_sample.py` | 개체–프레임 연결 판정 표본 생성과 정밀도 채점 |
| `31_enrich_review_context.py` | 판정 표본에 원문 문맥 추가 |
| `34_llm_sentiment_benchmark.py` | LLM 3종 감성 라벨 비교 (`--report`는 캐시만 사용) |
| `35_sentiment_relabel_review.py` | LLM–사람 불일치 재검토 표본(`make`)과 집계(`report`) |
| `36_llm_corpus_labeling.py` | 전수 LLM 라벨링, 프롬프트 검증(`--eval`), 라벨 내보내기(`--export`) |

### 5단계 KG 품질 감사와 정제판 (4개) — `--stage audit`

| 파일 | 내용 |
|---|---|
| `fp_agent_labels.py` | 개체 오탐 표본 100건의 판정 라벨 기록 |
| `28_kg_quality_audit.py` | 사전 중첩, 동어반복, 이중계상, 오탐 정밀도, 감성 신뢰도 감사 |
| `29_kg_clean_variant.py` | 정제 규칙을 적용한 정제판 KG와 담론 특성 재계산 |
| `30_kg_clean_robustness.py` | 원본 대 정제판 Granger·패널·전환점 재검증 (표 6.2) |

### 6단계 강건성 (5개) — `--stage robust`

| 파일 | 내용 |
|---|---|
| `22_panel_robustness.py` | 패널 모형 대안 사양 |
| `26_media_heterogeneity.py` | 매체별 감성·프레임 구성과 매체 구성 민감도 |
| `27_relevance_classifier.py` | 관련성 필터 정밀도와 재필터 후 Granger |
| `33_period_sensitivity.py` | 분석 시작 2018-01 대 2019-01 비교 (표 6.3) |
| `37_llm_h1_retest.py` | LLM 감성 라벨로 H1 재검정 |

### 7단계 그림 (3개) — `--stage figures`

| 파일 | 내용 |
|---|---|
| `19_kg_event_subgraphs.py` | 사건 월 하위그래프, 담론 변화·지표 추이 그림 (그림 5.1) |
| `32_kg_visualizations.py` | KG 스키마(그림 3.1), 통합 그래프(그림 4.1), 원본 대 정제판 비교 |
| `40_thesis_event_figures.py` | 2020-10 코나 화재·리콜과 2024-08 인천 화재의 전환점·하위그래프 (그림 5.3·5.4) |

## 산출물

| 위치 | 개수 |
|---|---|
| `outputs/kg/tables/*.csv` | 표 51개 (24번 BERT 표 1개 포함) |
| `outputs/kg/tables/*.md` | 단계별 요약 보고서 |
| `outputs/kg/figures/*.png` + `outputs/kg/kg_error_pattern.png` | 그림 17개 |
| `data/processed/*.csv` | 태깅 결과, 담론 특성, 전국 시계열, 지역 패널 등 중간 자료 |

### 논문 표·그림과 산출물 대응

| 논문 | 산출물 | 스크립트 |
|---|---|---|
| 표 3.1 | `tbl_media_sentiment.csv` | 26 |
| 표 3.4, 3.5 | `tbl_charger_national.csv`, `tbl_charger_region.csv` | 38 |
| 표 3.6, 3.7 | 프레임 사전, 개체 별칭 사전 | `kg_common.py` |
| 그림 3.1 | `kg_schema.png` | 32 |
| 그림 4.1 | `kg_corpus_overview.png` | 32 |
| 그림 5.1 | `discourse_evolution.png`, `kg_indicators_timeline.png` | 19 |
| 표 5.2 | `tbl_granger.csv` | 18 |
| 표 5.3 | `tbl_panel_main_effects.csv`, `tbl_panel_lag_joint.csv` | 18 |
| 표 5.4 | `tbl_granger_corrected.csv` | 20 |
| 표 5.5, 그림 5.2 | `tbl_var_fevd.csv`, `var_irf.png` | 20 |
| 표 5.6 | `tbl_national_forecast_4tier.csv`, `tbl_bootstrap_rmse.csv` | 21 |
| 표 5.7 | `tbl_panel_interactions.csv` | 18 |
| 표 5.8 | `tbl_breakpoints.csv` | 23 |
| 그림 5.3, 5.4 | `thesis_breakpoints_two_events.png`, `thesis_event_subgraphs.png` | 40 |
| 표 6.1 | `kg_audit_sentiment_*.csv`, `tbl_entity_frame_precision.csv`, `tbl_audit_fp_precision.csv`, `tbl_relevance_classifier.csv`, `tbl_llm_benchmark.csv`, `tbl_sentiment_relabel_benchmark.csv` | 28, 25, 27, 34, 35 |
| 표 6.2 | `kg_clean_robustness.md`, `tbl_kg_clean_*.csv` | 30 |
| 표 6.3 | `period_sensitivity.md`, `tbl_period_*.csv` | 33 |

## 사람 라벨과 API 단계

다음 단계는 사람의 판정이나 유료 API 호출이 필요해 `run_pipeline.py`가 결과 파일을 읽기만 한다.
처음부터 다시 수행할 때만 아래 순서를 따른다.

1. 감성 라벨: `make_sentiment_review_sample.py`로 300건 표본을 만들고 `human_relevance_ev`, `human_sentiment_ev` 열을 채운다.
2. 개체–프레임 판정: `25_entity_frame_review_sample.py` → `31_enrich_review_context.py` 후 `human_correct` 열(0/1)을 채우고 25번을 다시 실행한다.
3. 오탐 판정: `28_kg_quality_audit.py`가 오탐 표본을 만들면 `fp_agent_labels.py`의 판정을 기록한다.
4. LLM 비교: `OPENAI_API_KEY`를 설정하고 `34_llm_sentiment_benchmark.py --model <모델>`을 실행한다. 비용 상한은 `--max-usd`로 둔다.
5. 불일치 재검토: `35_sentiment_relabel_review.py make`로 표본을 만들고 사람 라벨을 채운다.
6. 전수 LLM 라벨링: `36_llm_corpus_labeling.py --task sentiment`, `--task relevance`를 실행한 뒤 `--export`로 `news_llm_labels.csv`를 만든다.

## 제외한 파일

원본 작업 폴더에는 있으나 논문 결과 산출에 쓰이지 않아 제출본에서 뺀 파일이다.

- `90_md_to_pdf.py`: 집필용 마크다운을 PDF로 변환하는 문서 도구
- `audit_vehicle_registration_files.py`: 자동차등록 원자료 파일 형식 점검용 일회성 스크립트
- `evaluate_sentiment_review.py`: 초기 감성 라벨 평가. `kg_sentiment_reliability.py`와 28번 감사로 대체
