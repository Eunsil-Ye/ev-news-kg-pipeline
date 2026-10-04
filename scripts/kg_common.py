# -*- coding: utf-8 -*-
"""
KG 파이프라인 공용 모듈: 경로, 분석 기간, 유의수준, 사전(감성/프레임/개체), 텍스트 유틸.

모든 분석 스크립트가 이 모듈을 import 한다. 사전과 판정 임계값은 여기서만 정의해
스크립트 간 정의가 갈라지지 않게 한다.
"""

from __future__ import annotations

import os
import re

# ============================================================
# 경로
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_NEWS = os.path.join(BASE_DIR, "All_news.csv")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
KG_DIR = os.path.join(OUT_DIR, "kg")
KG_MONTHLY_DIR = os.path.join(KG_DIR, "monthly")
FIG_DIR = os.path.join(OUT_DIR, "kg", "figures")
TABLE_DIR = os.path.join(OUT_DIR, "kg", "tables")

EV_REGISTRATION_XLS = os.path.join(BASE_DIR, "201801_202605_전기차등록현황.xls")
TOTAL_VEH_REGION_CSV = os.path.join(
    DATA_PROC, "total_vehicle_registration_month_region_201801_202510.csv"
)
CHARGER_CSV = os.path.join(DATA_PROC, "ev_charger_region_speed_snapshots.csv")

NEWS_TAGGED_CSV = os.path.join(DATA_PROC, "news_tagged.csv")
KG_MONTHLY_FEATURES_CSV = os.path.join(DATA_PROC, "kg_monthly_features.csv")
NEWS_MONTHLY_SIMPLE_CSV = os.path.join(DATA_PROC, "news_monthly_simple.csv")
PANEL_CSV = os.path.join(DATA_PROC, "panel_ev_news_kg.csv")
NATIONAL_CSV = os.path.join(DATA_PROC, "national_ev_news_kg.csv")

# 분석 기간
#  - START: 연구 착수 전에 정한 시작 시점. 누적 등록자료를 차분하므로 2018-01 의
#    ev_net_add 는 결측이고, KG 스냅샷은 94개월, 회귀 유효 관측은 93개월이 된다.
#    2019-01 로 제한한 민감도 분석은 33_period_sensitivity.py 가 수행한다.
#  - END: 등록자료의 마지막 수록 월. 원본 xls 파일명은 202605 지만 실제 수록은
#    2025-10 까지이며, 이후 달은 종속변수가 없어 표본에 넣을 수 없다.
ANALYSIS_START = "2018-01"
ANALYSIS_END = "2025-10"

# 유의수준. 선행성 판정은 정방향 p < ALPHA 와 역방향 p >= ALPHA 를 함께 요구한다.
ALPHA = 0.05


# ============================================================
# 전기차 관련 기사 필터 키워드
# ============================================================
EV_KEYWORDS = [
    "전기차", "전기자동차", "EV", "무공해차", "친환경차", "전동화",
    "아이오닉", "코나 일렉트릭", "코나 전기", "테슬라", "모델3", "모델Y",
    "폴스타", "전기 SUV", "전기트럭", "충전기", "충전소", "급속충전",
    "완속충전", "전기차 보조금",
]

# ============================================================
# 감성 사전 (make_sentiment_review_sample.py 의 사전과 같아야 한다)
# ============================================================
POSITIVE_WORDS = [
    "성장", "증가", "확대", "확충", "개선", "호조", "인기", "수요",
    "판매 증가", "판매 호조", "출시", "신차", "수출", "투자", "협력",
    "개발", "상용화", "보급", "지원", "혜택", "할인", "가격 인하",
    "인하", "돌파", "최대", "선도", "친환경", "고성능", "고효율",
    "충전 편의", "충전 인프라 확대", "보조금 확대", "보조금 지원",
    "수상", "흑자", "회복", "강화",
]

NEGATIVE_WORDS = [
    "화재", "폭발", "리콜", "결함", "사고", "우려", "불안", "위험",
    "부족", "지연", "감소", "하락", "부진", "적자", "축소", "중단",
    "취소", "논란", "반발", "갈등", "고장", "침체", "불편", "부담",
    "비싸", "고가", "급발진", "배터리 결함", "충전난", "보조금 축소",
    "보조금 삭감", "가격 부담", "판매 감소", "둔화", "피해", "소송", "규제",
]

# ============================================================
# 프레임(주제) 사전 - 9개 프레임
# ============================================================
FRAME_KEYWORDS = {
    "정책": ["정책", "규제", "국토교통부", "환경부", "지자체", "제도", "법안", "의무화"],
    "보조금": ["보조금", "국고보조금", "보급사업", "세제", "지원금", "예산"],
    "충전": ["충전", "충전기", "충전소", "급속충전", "완속충전", "충전 인프라"],
    "안전": ["화재", "폭발", "리콜", "결함", "사고", "안전", "급발진", "침수"],
    "가격": ["가격", "할인", "인하", "리스", "렌트", "구매비", "가격 인하"],
    "배터리": ["배터리", "전고체", "LFP", "양극재", "음극재", "셀", "ESS", "리튬"],
    "신차": ["출시", "신차", "아이오닉", "코나", "니로", "모델3", "모델Y", "EV3", "EV6", "EV9"],
    "환경": ["무공해차", "탄소중립", "친환경", "미세먼지", "온실가스", "배출가스", "탄소"],
    "산업": ["수출", "실적", "공장", "시장", "점유율", "매출", "생산", "투자"],
}
FRAME_NAMES = list(FRAME_KEYWORDS.keys())

# ============================================================
# 개체(Entity) 사전 - alias -> (type, canonical)
# ============================================================
ENTITY_DICT = {
    # 기업 (company)
    "현대자동차": ("company", "현대차"),
    "현대차": ("company", "현대차"),
    "기아": ("company", "기아"),
    "테슬라": ("company", "테슬라"),
    "LG에너지솔루션": ("company", "LG에너지솔루션"),
    "LG엔솔": ("company", "LG에너지솔루션"),
    "LG화학": ("company", "LG에너지솔루션"),
    "삼성SDI": ("company", "삼성SDI"),
    "SK온": ("company", "SK온"),
    "SK이노베이션": ("company", "SK온"),
    "BYD": ("company", "BYD"),
    "비야디": ("company", "BYD"),
    "한국GM": ("company", "GM"),
    "GM": ("company", "GM"),
    "폭스바겐": ("company", "폭스바겐"),
    "벤츠": ("company", "벤츠"),
    "메르세데스": ("company", "벤츠"),
    "BMW": ("company", "BMW"),
    "포스코퓨처엠": ("company", "포스코"),
    "포스코케미칼": ("company", "포스코"),
    "파나소닉": ("company", "파나소닉"),
    "CATL": ("company", "CATL"),
    "아우디": ("company", "아우디"),
    "볼보": ("company", "볼보"),
    "폴스타": ("company", "폴스타"),
    "르노": ("company", "르노"),
    # 모델 (model)
    "아이오닉6": ("model", "아이오닉6"),
    "아이오닉5": ("model", "아이오닉5"),
    "아이오닉": ("model", "아이오닉"),
    "EV6": ("model", "EV6"),
    "EV9": ("model", "EV9"),
    "EV3": ("model", "EV3"),
    "코나 일렉트릭": ("model", "코나"),
    "코나 전기": ("model", "코나"),
    "니로": ("model", "니로"),
    "모델3": ("model", "모델3"),
    "모델Y": ("model", "모델Y"),
    "모델S": ("model", "모델S"),
    "모델X": ("model", "모델X"),
    "볼트EUV": ("model", "볼트"),
    "볼트EV": ("model", "볼트"),
    "리프": ("model", "리프"),
    # 정책 개체 (policy)
    "보조금": ("policy", "보조금"),
    "무공해차": ("policy", "무공해차"),
    "탄소중립": ("policy", "탄소중립"),
    "친환경차": ("policy", "친환경차"),
    "저공해차": ("policy", "저공해차"),
    # 사건/이슈 개체 (event)
    "급발진": ("event", "급발진"),
    "리콜": ("event", "리콜"),
    "화재": ("event", "화재"),
    "결함": ("event", "결함"),
    "충전난": ("event", "충전난"),
    "보조금 축소": ("event", "보조금축소"),
    "보조금 삭감": ("event", "보조금축소"),
    "가격 인하": ("event", "가격인하"),
}

# 부정 사건 개체 (전파도 계산용)
NEGATIVE_EVENT_ENTITIES = ["화재", "리콜", "결함", "급발진", "충전난", "보조금축소"]


# ============================================================
# 텍스트 유틸
# ============================================================
_EV_RE = re.compile(r"(?<![A-Za-z])EV(?![A-Za-z])")


def norm_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def contains_any(text: str, keywords) -> bool:
    low = text.lower()
    for kw in keywords:
        if kw == "EV":
            if _EV_RE.search(text):
                return True
            continue
        if kw.lower() in low:
            return True
    return False


def hit_words(text: str, keywords) -> list:
    low = text.lower()
    hits = []
    for kw in keywords:
        if kw == "EV":
            if _EV_RE.search(text):
                hits.append(kw)
            continue
        if kw.lower() in low:
            hits.append(kw)
    return hits


def classify_sentiment(text: str):
    """감성 사전 기반 분류. (label, score, n_pos, n_neg) 반환."""
    pos = hit_words(text, POSITIVE_WORDS)
    neg = hit_words(text, NEGATIVE_WORDS)
    total = len(pos) + len(neg)
    score = 0.0 if total == 0 else (len(pos) - len(neg)) / total
    if score >= 0.25:
        label = "긍정"
    elif score <= -0.25:
        label = "부정"
    else:
        label = "중립"
    return label, score, len(pos), len(neg)


def detect_frames(text: str) -> list:
    """텍스트에 등장하는 프레임 목록."""
    return [name for name, kws in FRAME_KEYWORDS.items() if contains_any(text, kws)]


def detect_entities(text: str) -> list:
    """텍스트에서 개체 추출. [(type, canonical), ...] 중복 제거."""
    low = text.lower()
    found = {}
    for alias, (etype, canon) in ENTITY_DICT.items():
        if alias == "EV":
            continue
        if alias.lower() in low:
            found[(etype, canon)] = True
    return list(found.keys())


def ensure_dirs():
    for d in (DATA_PROC, OUT_DIR, KG_DIR, KG_MONTHLY_DIR, FIG_DIR, TABLE_DIR):
        os.makedirs(d, exist_ok=True)


def set_korean_font():
    """matplotlib 한글 폰트 설정."""
    from matplotlib import font_manager, rcParams
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim", "MS Gothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            rcParams["font.sans-serif"] = [name] + list(rcParams.get("font.sans-serif", []))
            break
    rcParams["axes.unicode_minus"] = False
