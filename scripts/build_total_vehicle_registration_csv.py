# -*- coding: utf-8 -*-
"""
국토교통부 자동차등록현황 CSV(시도별 월별, 연도별)를 분석용 CSV 로 정리한다.

원본 CSV 는 천 단위 쉼표가 따옴표 없이 들어 있어 열이 밀린다. 표 안의 소계 관계
(차종별 합계, 소유별 합계, 총계)를 이용해 20개 숫자 열을 복원한다.

입력: 자동차등록/*.csv
출력: data/processed/total_vehicle_registration_month_region_201801_202510.csv (17번의 입력) 외
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "자동차등록"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT = ROOT / "outputs" / "total_vehicle_registration_clean_check.md"

MONTHLY_FILES = [
    SOURCE_DIR / "자동차등록현황보고_자동차등록대수현황 시도별 (201101 ~ 202604).csv",
    SOURCE_DIR / "자동차등록현황보고_자동차등록대수현황 시도별 (201101 ~ 202604) (1).csv",
]

ANNUAL_FILES = [
    SOURCE_DIR / "자동차등록현황보고_자동차등록대수현황_연도별 (1990 ~ 2025).csv",
    SOURCE_DIR / "자동차등록현황보고_자동차등록대수현황_연도별 (1990 ~ 2025) (1).csv",
]

NUMERIC_COLUMNS = [
    "passenger_official",
    "passenger_private",
    "passenger_business",
    "passenger_total",
    "van_official",
    "van_private",
    "van_business",
    "van_total",
    "truck_official",
    "truck_private",
    "truck_business",
    "truck_total",
    "special_official",
    "special_private",
    "special_business",
    "special_total",
    "total_official",
    "total_private",
    "total_business",
    "total_car_stock",
]


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="cp949").splitlines()


def parse_number_group(tokens: list[str]) -> int | None:
    if not tokens:
        return None
    if any(not token.isdigit() for token in tokens):
        return None
    if len(tokens) > 1 and any(len(token) != 3 for token in tokens[1:]):
        return None
    return int("".join(tokens))


def formatted_tokens(value: int) -> list[str]:
    return f"{value:,}".split(",")


def match_formatted_value(tokens: list[str], idx: int, value: int) -> int | None:
    expected = formatted_tokens(value)
    if tokens[idx : idx + len(expected)] == expected:
        return idx + len(expected)
    return None


def valid_vehicle_values(values: list[int]) -> bool:
    if len(values) != 20:
        return False

    # 차종별 합계
    if values[3] != values[0] + values[1] + values[2]:
        return False
    if values[7] != values[4] + values[5] + values[6]:
        return False
    if values[11] != values[8] + values[9] + values[10]:
        return False
    if values[15] != values[12] + values[13] + values[14]:
        return False

    # 소유별(관용/자가용/영업용) 합계
    if values[16] != values[0] + values[4] + values[8] + values[12]:
        return False
    if values[17] != values[1] + values[5] + values[9] + values[13]:
        return False
    if values[18] != values[2] + values[6] + values[10] + values[14]:
        return False

    # 총계를 두 방향(차종 합, 소유 합)으로 모두 확인
    if values[19] != values[16] + values[17] + values[18]:
        return False
    if values[19] != values[3] + values[7] + values[11] + values[15]:
        return False
    return True


def reconstruct_values(tokens: list[str], context: str) -> list[int]:
    matches: list[list[int]] = []

    def parse_three_values(idx: int) -> list[tuple[int, int, int, int]]:
        parsed: list[tuple[int, int, int, int]] = []
        for len_a in (1, 2, 3):
            a = parse_number_group(tokens[idx : idx + len_a])
            if a is None:
                continue
            idx_b = idx + len_a
            for len_b in (1, 2, 3):
                b = parse_number_group(tokens[idx_b : idx_b + len_b])
                if b is None:
                    continue
                idx_c = idx_b + len_b
                for len_c in (1, 2, 3):
                    c = parse_number_group(tokens[idx_c : idx_c + len_c])
                    if c is None:
                        continue
                    parsed.append((a, b, c, idx_c + len_c))
        return parsed

    def parse_vehicle_groups(group_idx: int, idx: int, values: list[int]) -> None:
        if group_idx == 4:
            official = values[0] + values[4] + values[8] + values[12]
            private = values[1] + values[5] + values[9] + values[13]
            business = values[2] + values[6] + values[10] + values[14]
            grand_total = official + private + business
            next_idx = match_formatted_value(tokens, idx, official)
            if next_idx is None:
                return
            next_idx = match_formatted_value(tokens, next_idx, private)
            if next_idx is None:
                return
            next_idx = match_formatted_value(tokens, next_idx, business)
            if next_idx is None:
                return
            next_idx = match_formatted_value(tokens, next_idx, grand_total)
            if next_idx is None:
                return
            candidate = [*values, official, private, business, grand_total]
            if next_idx == len(tokens) and valid_vehicle_values(candidate):
                matches.append(candidate)
            return

        for a, b, c, after_three_idx in parse_three_values(idx):
            subtotal = a + b + c
            next_idx = match_formatted_value(tokens, after_three_idx, subtotal)
            if next_idx is None:
                continue
            parse_vehicle_groups(group_idx + 1, next_idx, [*values, a, b, c, subtotal])

    parse_vehicle_groups(0, 0, [])

    if len(matches) != 1:
        raise ValueError(f"Expected exactly one parse for {context}, got {len(matches)}")
    return matches[0]


def parse_monthly_files() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for path in MONTHLY_FILES:
        for line_no, line in enumerate(read_lines(path)[2:], 3):
            parts = line.split(",")
            if len(parts) < 4:
                continue
            ym, region, district = parts[0], parts[1], parts[2]
            if district != "계":
                continue
            values = reconstruct_values(parts[3:], f"{path.name}:{line_no}:{ym}:{region}:{district}")
            row: dict[str, str | int] = {
                "ym": ym,
                "region": region,
                "source_file": path.name,
            }
            row.update(dict(zip(NUMERIC_COLUMNS, values)))
            rows.append(row)
    rows.sort(key=lambda row: (str(row["ym"]), str(row["region"])))
    return rows


def parse_annual_files() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for path in ANNUAL_FILES:
        for line_no, line in enumerate(read_lines(path)[2:], 3):
            parts = line.split(",")
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            year = parts[0]
            values = reconstruct_values(parts[1:], f"{path.name}:{line_no}:{year}")
            row: dict[str, str | int] = {
                "year": year,
                "source_file": path.name,
            }
            row.update(dict(zip(NUMERIC_COLUMNS, values)))
            rows.append(row)
    rows.sort(key=lambda row: str(row["year"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_monthly_total(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    by_month: dict[str, dict[str, str | int]] = {}
    for row in rows:
        ym = str(row["ym"])
        if ym not in by_month:
            by_month[ym] = {"ym": ym}
            for column in NUMERIC_COLUMNS:
                by_month[ym][column] = 0
        for column in NUMERIC_COLUMNS:
            by_month[ym][column] = int(by_month[ym][column]) + int(row[column])
    return [by_month[ym] for ym in sorted(by_month)]


def build_report(
    monthly_rows: list[dict[str, str | int]],
    analysis_rows: list[dict[str, str | int]],
    monthly_total_rows: list[dict[str, str | int]],
    analysis_total_rows: list[dict[str, str | int]],
    annual_rows: list[dict[str, str | int]],
) -> None:
    months = sorted({str(row["ym"]) for row in monthly_rows})
    analysis_months = sorted({str(row["ym"]) for row in analysis_rows})
    regions = sorted({str(row["region"]) for row in monthly_rows})
    annual_years = sorted({str(row["year"]) for row in annual_rows})

    def table(headers: list[str], body: list[list[str | int]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for item in body:
            lines.append("| " + " | ".join(str(value) for value in item) + " |")
        return "\n".join(lines)

    # 분석 기간의 모든 월에 17개 시도가 있는지 확인
    bad_months = []
    for ym in analysis_months:
        month_regions = {str(row["region"]) for row in analysis_rows if row["ym"] == ym}
        if month_regions != set(regions):
            bad_months.append(ym)

    oldest = analysis_rows[0]
    latest = analysis_rows[-1]

    annual_by_year = {str(row["year"]): int(row["total_car_stock"]) for row in annual_rows}
    dec_checks = []
    for row in monthly_total_rows:
        ym = str(row["ym"])
        if ym.endswith("-12") and ym[:4] in annual_by_year:
            monthly_total = int(row["total_car_stock"])
            annual_total = annual_by_year[ym[:4]]
            dec_checks.append([ym[:4], monthly_total, annual_total, monthly_total - annual_total])

    report = [
        "# 전체 자동차 등록자료 정리 결과",
        "",
        "## 생성 파일",
        "",
        table(
            ["파일", "설명"],
            [
                ["data/processed/total_vehicle_registration_month_region.csv", "전체 확보 기간 월별 × 지역별 전체 자동차 등록대수"],
                ["data/processed/total_vehicle_registration_month_region_201801_202510.csv", "논문 분석 기간 2018-01~2025-10 월별 × 지역별 자료"],
                ["data/processed/total_vehicle_registration_month_total.csv", "전체 확보 기간 월별 전국 합계"],
                ["data/processed/total_vehicle_registration_month_total_201801_202510.csv", "논문 분석 기간 월별 전국 합계"],
                ["data/processed/total_vehicle_registration_annual_total.csv", "2018~2025 연도별 전국 자료"],
            ],
        ),
        "",
        "## 월별 지역별 자료",
        "",
        table(
            ["항목", "값"],
            [
                ["전체 월 수", len(months)],
                ["전체 기간", f"{months[0]} ~ {months[-1]}"],
                ["전체 행 수", len(monthly_rows)],
                ["지역 수", len(regions)],
                ["지역", ", ".join(regions)],
                ["분석 기간 월 수", len(analysis_months)],
                ["분석 기간", f"{analysis_months[0]} ~ {analysis_months[-1]}"],
                ["분석 기간 행 수", len(analysis_rows)],
                ["분석 기간 누락 월/지역", "없음" if not bad_months else ", ".join(bad_months)],
                ["분석 기간 전국 월별 행 수", len(analysis_total_rows)],
            ],
        ),
        "",
        "## 연도별 전국 자료",
        "",
        table(
            ["항목", "값"],
            [
                ["연도 수", len(annual_years)],
                ["기간", f"{annual_years[0]} ~ {annual_years[-1]}"],
                ["행 수", len(annual_rows)],
            ],
        ),
        "",
        "## 연도별 자료와 월별 12월 합계 비교",
        "",
        table(
            ["연도", "월별 12월 전국 합계", "연도별 전국 합계", "차이"],
            [[year, f"{monthly_total:,}", f"{annual_total:,}", f"{diff:,}"] for year, monthly_total, annual_total, diff in dec_checks],
        ),
        "",
        "## 표본 행",
        "",
        table(
            ["구분", "ym", "region", "total_car_stock"],
            [
                ["분석 기간 첫 행", oldest["ym"], oldest["region"], f"{int(oldest['total_car_stock']):,}"],
                ["분석 기간 마지막 행", latest["ym"], latest["region"], f"{int(latest['total_car_stock']):,}"],
            ],
        ),
        "",
        "## 사용 변수",
        "",
        "최소 핵심 변수는 다음과 같다.",
        "",
        "```text",
        "ym",
        "region",
        "total_car_stock",
        "```",
        "",
        "전기차 등록자료와 결합하면 다음 변수를 만들 수 있다.",
        "",
        "```text",
        "ev_share = ev_stock / total_car_stock",
        "```",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    monthly_rows = parse_monthly_files()
    annual_rows = parse_annual_files()
    analysis_rows = [
        row for row in monthly_rows if "2018-01" <= str(row["ym"]) <= "2025-10"
    ]
    monthly_total_rows = aggregate_monthly_total(monthly_rows)
    analysis_total_rows = aggregate_monthly_total(analysis_rows)

    monthly_fields = ["ym", "region", *NUMERIC_COLUMNS, "source_file"]
    monthly_total_fields = ["ym", *NUMERIC_COLUMNS]
    annual_fields = ["year", *NUMERIC_COLUMNS, "source_file"]

    write_csv(OUTPUT_DIR / "total_vehicle_registration_month_region.csv", monthly_rows, monthly_fields)
    write_csv(
        OUTPUT_DIR / "total_vehicle_registration_month_region_201801_202510.csv",
        analysis_rows,
        monthly_fields,
    )
    write_csv(OUTPUT_DIR / "total_vehicle_registration_month_total.csv", monthly_total_rows, monthly_total_fields)
    write_csv(
        OUTPUT_DIR / "total_vehicle_registration_month_total_201801_202510.csv",
        analysis_total_rows,
        monthly_total_fields,
    )
    write_csv(OUTPUT_DIR / "total_vehicle_registration_annual_total.csv", annual_rows, annual_fields)
    build_report(monthly_rows, analysis_rows, monthly_total_rows, analysis_total_rows, annual_rows)

    print(f"monthly_rows={len(monthly_rows)}")
    print(f"analysis_rows={len(analysis_rows)}")
    print(f"annual_rows={len(annual_rows)}")
    print(f"monthly_total_rows={len(monthly_total_rows)}")
    print(f"analysis_total_rows={len(analysis_total_rows)}")
    print(f"monthly_output={(OUTPUT_DIR / 'total_vehicle_registration_month_region.csv').resolve()}")
    print(f"analysis_output={(OUTPUT_DIR / 'total_vehicle_registration_month_region_201801_202510.csv').resolve()}")
    print(f"monthly_total_output={(OUTPUT_DIR / 'total_vehicle_registration_month_total.csv').resolve()}")
    print(f"analysis_total_output={(OUTPUT_DIR / 'total_vehicle_registration_month_total_201801_202510.csv').resolve()}")
    print(f"annual_output={(OUTPUT_DIR / 'total_vehicle_registration_annual_total.csv').resolve()}")
    print(f"report={REPORT.resolve()}")


if __name__ == "__main__":
    main()
