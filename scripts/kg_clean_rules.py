# -*- coding: utf-8 -*-
"""
KG 정제판(clean) 규칙 모듈

`28_kg_quality_audit.py` 가 찾아낸 결함을 교정하는 규칙을 정의한다.
원본 사전(`kg_common.py`)은 수정하지 않고, 여기서 덮어쓰기 규칙만 제공한다.

교정 항목
  C1 최장일치 우선     "아이오닉5" 매칭 시 "아이오닉" 중복 계상 방지
  C2 단어경계/부정문맥  GM(영문 내부), 리프(리프트/리프레시), 니로(미니로봇) 등 오탐 차단
  C3 동어반복 엣지 제외 별칭이 프레임 키워드를 포함하면 ENTITY_FRAME 엣지가 정보량 0 -> 제외
  C4 죽은 항목 교정     충전난(전체 2건)을 실제 통용 표현으로 확장
  C5 과대포괄 프레임     산업(출현율 77.8%)의 일반어 키워드를 산업 특화어로 축소
  C6 프레임 키워드 오탐  리스(해리스/애널리스트/리스크), 셀(리셀/셀토스/셀프) 등
                       부분문자열 오탐을 정문맥 정규식으로 차단
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kg_common as kc

# ============================================================
# C2. 단어경계 / 부정문맥 정규식
#     여기에 등록된 별칭은 부분문자열 매칭 대신 이 정규식으로 판정한다.
# ============================================================
ALIAS_REGEX = {
    # 영문 약칭: 영문/숫자 내부 매칭 차단 (segment, fragment 의 'gm' 등)
    "GM": re.compile(r"(?<![A-Za-z0-9])GM(?![A-Za-z0-9])"),
    "한국GM": re.compile(r"한국\s*GM(?![A-Za-z0-9])"),
    "BYD": re.compile(r"(?<![A-Za-z])BYD(?![A-Za-z])"),
    "CATL": re.compile(r"(?<![A-Za-z])CATL(?![A-Za-z])"),
    "BMW": re.compile(r"(?<![A-Za-z])BMW(?![A-Za-z])"),
    # 모델명: 뒤에 숫자가 이어지는 오탐(EV300 등) 차단
    "EV3": re.compile(r"(?<![A-Za-z0-9])EV3(?![0-9])"),
    "EV6": re.compile(r"(?<![A-Za-z0-9])EV6(?![0-9])"),
    "EV9": re.compile(r"(?<![A-Za-z0-9])EV9(?![0-9])"),
    # 리프(닛산 Leaf): 리프트/리프레시/리프팅, 클리프·드리프트 차단
    # 폴리프로필렌·엘리프(아파트)·리프모터(중국 Leapmotor)는 검토표본에서 확인된 잔존 오탐
    "리프": re.compile(r"(?<![클그드폴엘])리프(?!트|레|팅|시|로|모터)"),
    # 니로(기아 Niro): 미니로봇 등 차단
    "니로": re.compile(r"(?<![미하니])니로"),
    # 기아: 굶주림 의미의 기아 차단
    "기아": re.compile(r"기아(?!\s*(?:문제|상태|퇴치|인구|난민|사태))"),
    # 볼트: 전압 단위(220볼트) 차단 — 원본은 볼트EV/볼트EUV 별칭만 사용하므로 보강 목적
    "볼트EV": re.compile(r"볼트\s*EV(?![A-Za-z])"),
    "볼트EUV": re.compile(r"볼트\s*EUV"),
}

# ============================================================
# C4. 죽은 사전 항목 교정 (추가 별칭)
#     '충전난'은 전체 코퍼스에서 2건뿐 -> 실제 통용 표현으로 확장
# ============================================================
EXTRA_ENTITY_ALIASES = {
    "충전 대기": ("event", "충전난"),
    "충전 불편": ("event", "충전난"),
    "충전 인프라 부족": ("event", "충전난"),
    "충전기 부족": ("event", "충전난"),
    "충전소 부족": ("event", "충전난"),
}

# ============================================================
# C5. 과대포괄 프레임 정밀화
#     '시장/투자/생산' 같은 일반어를 제거하고 산업 특화어만 남긴다.
# ============================================================
FRAME_KEYWORDS_OVERRIDE = {
    "산업": ["수출", "영업이익", "공장", "점유율", "매출", "양산",
             "설비투자", "생산라인", "수주", "실적"],
}

# ============================================================
# C6. 프레임 키워드 부분문자열 오탐 차단
#     검토표본 130건에서 '리스'는 3/3 오탐, '셀'도 오탐이 확인됐다.
#     코퍼스 전수 점검 결과 '리스'는 해리스(371회)·애널리스트(421회)·
#     넥실리스·스테인리스·크리스마스·리스크·리스트를, '셀'은 리셀·셀토스·
#     셀트리온·액셀러레이터·셀프·베스트셀러를 잡고 있었다.
#     오탐 어휘가 너무 다양해 부정문맥 열거(블랙리스트)로는 누락이 생기므로,
#     두 키워드만 정문맥 열거(화이트리스트)로 판정한다.
#     재현율은 다소 희생되지만 두 키워드 모두 같은 프레임 안에 상위 키워드
#     ('가격', '배터리')가 있어 프레임 자체의 재현율 손실은 작다.
# ============================================================
FRAME_KW_REGEX = {
    # 차량 리스(금융 상품)일 때만 인정.
    # 접미사 분기에는 좌측 경계를 걸어 '팹리스 업체', '넥실리스 사업' 같은
    # 합성어를 배제한다(수식어 분기는 좌측 문맥이 이미 특정돼 불필요).
    "리스": re.compile(
        r"(?:(?:장기|단기|차량|자동차|신차|중고차|전기차|법인)\s*리스)"
        r"|(?<![가-힣A-Za-z])리스\s*(?:료|차|차량|계약|이용|상품|금융|승계|업체|사업)"
        r"|(?:렌트(?:카)?[·,/\s]{1,3}리스)"
        r"|(?<![가-힣A-Za-z])리스[·,/\s]{1,3}렌트"
    ),
    # 배터리 셀(제조 단위)일 때만 인정.
    # 접미사 분기 좌측 경계로 '리셀 가격', '액셀 성능' 등을 배제한다.
    "셀": re.compile(
        r"(?:(?:배터리|이차전지|2차전지|각형|원통형|파우치|리튬|양극|음극)\s*셀)"
        r"|(?<![가-힣A-Za-z0-9])셀\s*(?:생산|제조|공급|업체|공장|가격|단가|성능|용량|기술|설계|수율)"
        r"|(?<![가-힣A-Za-z0-9])셀[·/][\s]?모듈|(?:모듈[·/][\s]?셀)"
    ),
}


def build_entity_dict() -> dict:
    """원본 ENTITY_DICT + 추가 별칭."""
    d = dict(kc.ENTITY_DICT)
    d.update(EXTRA_ENTITY_ALIASES)
    return d


def build_frame_keywords() -> dict:
    """원본 FRAME_KEYWORDS 에 정밀화 오버라이드 적용."""
    d = {k: list(v) for k, v in kc.FRAME_KEYWORDS.items()}
    d.update({k: list(v) for k, v in FRAME_KEYWORDS_OVERRIDE.items()})
    return d


ENTITY_DICT_CLEAN = build_entity_dict()
FRAME_KEYWORDS_CLEAN = build_frame_keywords()
FRAME_NAMES_CLEAN = list(FRAME_KEYWORDS_CLEAN.keys())


# ============================================================
# C3. 동어반복 (canonical, frame) 쌍
#     별칭 문자열이 프레임 키워드를 포함하면, 그 별칭이 매칭된 기사에는
#     해당 프레임도 반드시 태깅된다 -> ENTITY_FRAME 엣지의 정보량이 0.
# ============================================================
def build_tautological_pairs() -> set:
    pairs = set()
    for alias, (etype, canon) in ENTITY_DICT_CLEAN.items():
        for fr, kws in FRAME_KEYWORDS_CLEAN.items():
            for kw in kws:
                if kw and kw in alias:
                    pairs.add((canon, fr))
    return pairs


TAUTOLOGICAL_PAIRS = build_tautological_pairs()


# ============================================================
# C1. 최장일치 우선 개체 추출
# ============================================================
def _alias_spans(text: str, alias: str):
    """별칭이 등장한 (start, end) 목록. 정규식 규칙이 있으면 그것을 사용."""
    pat = ALIAS_REGEX.get(alias)
    if pat is not None:
        return [(m.start(), m.end()) for m in pat.finditer(text)]
    low = text.lower()
    a = alias.lower()
    out = []
    start = 0
    while True:
        i = low.find(a, start)
        if i < 0:
            break
        out.append((i, i + len(a)))
        start = i + 1
    return out


def detect_entities_clean(text: str) -> list:
    """개체 추출. 겹치는 매칭은 더 긴 별칭이 이긴다(최장일치).

    같은 문자열 구간을 두 별칭이 다투는 경우에만 짧은 쪽을 버리므로,
    긴 별칭과 별개 위치에 짧은 별칭이 또 나오면 둘 다 인정된다.
    """
    spans = []
    for alias, (etype, canon) in ENTITY_DICT_CLEAN.items():
        for st, en in _alias_spans(text, alias):
            spans.append((st, en, etype, canon))
    # 긴 구간 우선, 동일 길이면 앞선 위치 우선
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))

    taken = []
    found = {}
    for st, en, etype, canon in spans:
        if any(st < te and en > ts for ts, te in taken):
            continue
        taken.append((st, en))
        found[(etype, canon)] = True
    return list(found.keys())


def frame_hit_words(text: str, kws) -> list:
    """프레임 키워드 중 실제 매칭된 것. C6 정규식이 있으면 그것으로 판정."""
    low = text.lower()
    out = []
    for kw in kws:
        pat = FRAME_KW_REGEX.get(kw)
        if pat is not None:
            if pat.search(text):
                out.append(kw)
        elif kw.lower() in low:
            out.append(kw)
    return out


def detect_frames_clean(text: str) -> list:
    return [name for name, kws in FRAME_KEYWORDS_CLEAN.items()
            if frame_hit_words(text, kws)]


def is_tautological(canon: str, frame: str) -> bool:
    return (canon, frame) in TAUTOLOGICAL_PAIRS
