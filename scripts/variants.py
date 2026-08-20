"""전략 변형 정의 — 그림자 모드.

매일 여러 전략을 동시에 계산해 후보를 기록하고, 다음 실행에서 전부 채점한다.
발행(대시보드 메인)은 primary 변형 하나만 하고 나머지는 성과 비교용으로만 쌓는다.

왜 필요한가
  2026-07-29 시점 39거래일 백테스트에서 역추세는 추세추종을 크게 이겼지만
  (누적 -26.5% vs -53.1%), 시장 중립 구조의 t값은 0.7 수준이었다. 샤프 1.8짜리
  전략을 t=2로 확인하려면 약 290거래일이 필요하다. 백테스트를 더 돌려봐야
  같은 39일을 다시 보는 것이므로, 실제 out-of-sample 데이터를 쌓는 수밖에 없다.

각 변형은 gate(탈락 규칙)와 score(순위 규칙)를 갖는다. 팩터 계산은 factors.py가
공통으로 하고, 여기서는 어떤 팩터를 어떤 부호로 쓸지만 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

import factors as F

z = F.zscore


# --- 공통 헬퍼 ---------------------------------------------------------


def _liquidity_gate(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """모든 변형이 공유하는 최소 조건. 변형 간 비교가 유동성 차이 때문에
    흐려지지 않도록 여기서 동일하게 건다."""
    rules = cfg["universe"]
    counts = {}

    atr = frame["atr_pct"]
    frame = frame[atr.between(rules["atr_hard_min"], rules["atr_hard_max"]) | atr.isna()]
    counts[f"ATR {rules['atr_hard_min']}~{rules['atr_hard_max']}% (안전 상하한)"] = len(frame)
    return frame, counts


def _atr_relative_band(frame: pd.DataFrame, cfg: dict, counts: dict) -> pd.DataFrame:
    rules = cfg["universe"]
    atr = frame["atr_pct"].dropna()
    if len(atr) >= rules["atr_band_min_pool"]:
        low_pct, high_pct = rules["atr_pct_band"]
        lo, hi = atr.quantile(low_pct / 100), atr.quantile(high_pct / 100)
        frame = frame[frame["atr_pct"].between(lo, hi) | frame["atr_pct"].isna()]
        counts[f"ATR 풀 내 {low_pct}~{high_pct}분위 ({lo:.1f}~{hi:.1f}%)"] = len(frame)
    else:
        counts[f"ATR 상대밴드 (풀 {len(atr)}종목 < {rules['atr_band_min_pool']}, 미적용)"] = len(frame)
    return frame


# --- 1. 역추세 (production) --------------------------------------------


def reversal_gate(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """낙폭 과다 + 저평가 + 우량성. falling knife 방어가 핵심."""
    rules, gates = cfg["universe"], cfg["quality_gates"]
    frame, counts = _liquidity_gate(frame, cfg)

    frame = frame[frame["ret_20d"].between(rules["min_ret_20d"], rules["max_ret_20d"])]
    counts[f"20일 수익률 {rules['min_ret_20d']}~{rules['max_ret_20d']}%"] = len(frame)

    frame = frame[frame["drawdown_52w"] <= rules["max_drawdown_52w"]]
    counts[f"52주 고점 대비 {rules['max_drawdown_52w']}% 이상 하락"] = len(frame)

    low, high = rules["rsi_range"]
    frame = frame[frame["rsi14"].between(low, high)]
    counts[f"RSI {low}~{high} (과매도 구간)"] = len(frame)

    if gates["require_profitable"]:
        frame = frame[frame["is_profitable"]]
        counts["흑자 (EPS > 0)"] = len(frame)
    # require_coverage 제거 — 목표주가 커버리지가 시총 Q1 0.4% -> Q5 69.2%로
    # 퀄리티가 아니라 규모 필터로 작동한다.

    frame = frame[frame["roe"] >= gates["min_roe"]]
    counts[f"ROE {gates['min_roe']}% 이상"] = len(frame)
    frame = frame[frame["pbr"] <= gates["max_pbr"]]
    counts[f"PBR {gates['max_pbr']}배 이하"] = len(frame)
    # min_target_upside 제거 — 217종목 중 187개(86.2%)를 통과시켜 무변별이었다.

    threshold = gates.get("min_oc_after_down")
    if threshold is not None:
        enough = frame["down_day_count"] >= gates["min_oc_after_down_samples"]
        frame = frame[~enough | (frame["oc_after_down"] >= threshold)]
        counts[f"하락 다음날 일중수익률 {threshold}% 이상"] = len(frame)

    frame = _atr_relative_band(frame, cfg, counts)
    return frame.reset_index(drop=True), counts


def reversal_score(frame: pd.DataFrame, cfg: dict, regime: str) -> pd.DataFrame:
    """5개 블록을 국면별 가중치로 합산. block_* 컬럼도 남겨 리포트에 쓴다."""
    s = cfg["scoring"]

    drawdown_fit = -(frame["drawdown_52w"] - s["drawdown_sweet_spot"]).abs()
    rsi_fit = -(frame["rsi14"] - s["rsi_sweet_spot"]).abs()
    bb_fit = -(frame["bb_percent_b"] - s["bb_sweet_spot"]).abs()

    frame["block_drawdown"] = (
        0.30 * z(drawdown_fit) + 0.28 * z(rsi_fit)
        + 0.22 * z(bb_fit) - 0.20 * z(frame["ma20_disparity"])
    )

    reliable = frame["down_day_count"] >= s["min_down_day_samples"]
    frame["block_reversal"] = (
        0.34 * z(frame["oc_after_down"].where(reliable))
        + 0.22 * z(frame["oc_after_down_winrate"].where(reliable))
        + 0.18 * z(frame["lower_shadow"])
        + 0.14 * z(frame["capitulation"].clip(upper=s["capitulation_cap"]))
        + 0.12 * z(frame["oc_mean"])
    )

    frame["block_flow"] = (
        0.40 * z(frame["foreign_net_5d_pct"]) + 0.30 * z(frame["organ_net_5d_pct"])
        + 0.15 * z(frame["foreign_ratio_change"])
        + 0.15 * z(frame["foreign_buy_days"] + frame["organ_buy_days"])
    )
    frame.loc[frame["dual_buying"], "block_flow"] += 0.30

    effective_per = frame["forward_per"].fillna(frame["per"])
    frame["block_value"] = (
        0.40 * z(frame["target_upside"]) - 0.25 * z(frame["pbr"])
        - 0.20 * z(effective_per) + 0.15 * z(frame["dividend_yield"])
    )

    frame["block_quality"] = (
        0.45 * z(frame["roe"]) + 0.30 * z(frame["recomm_score"])
        + 0.25 * z(frame["oc_sharpe"])
    )

    weights = cfg["weights"][regime]
    frame["score"] = sum(weights[b] * frame[f"block_{b}"] for b in weights)
    return frame


# --- 2. 추세추종 (원래 전략, 대조군) ------------------------------------


def momentum_gate(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """전환 전 규칙. 낙폭 조건이 없고 과매수만 배제한다."""
    frame, counts = _liquidity_gate(frame, cfg)
    frame = frame[(frame["rsi14"] <= 78) | frame["rsi14"].isna()]
    counts["RSI 78 이하"] = len(frame)
    frame = _atr_relative_band(frame, cfg, counts)
    return frame.reset_index(drop=True), counts


def momentum_score(frame: pd.DataFrame, cfg: dict, regime: str) -> pd.DataFrame:
    """부호가 역추세와 정반대. 오른 종목·강한 RSI가 상위로 온다."""
    rsi_fit = -(frame["rsi14"] - 57.0).abs()
    bb_fit = -(frame["bb_percent_b"] - 0.62).abs()

    momentum = (
        0.30 * z(frame["ret_5d"]) + 0.25 * z(frame["ret_20d"])
        + 0.25 * z(frame["relative_strength"]) + 0.20 * z(frame["ma20_disparity"])
    )
    intraday = (
        0.40 * z(frame["oc_mean"]) + 0.30 * z(frame["oc_winrate"])
        + 0.30 * z(frame["oc_sharpe"])
    )
    technical = (
        0.35 * z(rsi_fit) + 0.35 * z(frame["macd_hist"])
        + 0.30 * z(bb_fit)
    )
    flow = (
        0.55 * z(frame["foreign_net_5d_pct"]) + 0.45 * z(frame["organ_net_5d_pct"])
    )
    quality = 0.60 * z(frame["target_upside"]) + 0.40 * z(frame["recomm_score"])

    frame["score"] = (
        0.28 * flow + 0.26 * momentum + 0.22 * intraday
        + 0.14 * technical + 0.10 * quality
    )
    return frame


# --- 3. 과매도 단일 팩터 (단순 기준선) ----------------------------------


def oversold_gate(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """유동성 공통 게이트 + 두 가지만 얹는다.

    이 전략의 정의는 "점수를 RSI 하나로만 매긴다"이지 "아무것도 안 거른다"가
    아니다. 발행 전략이 된 이상 최소한의 안전 게이트는 필요하고, 아래 둘은
    점수가 아니라 탈락 조건이므로 단일팩터 원칙을 깨지 않는다.

    1) RSI 하한 — 극단 과매도는 되돌림이 아니라 추세적 붕괴인 경우가 많다.
       12년 패널에서 모멘텀 십분위가 역U자였다(D1 -7.3 / D5 +6.9 / D10 -5.5).
    2) 변동성 최상위 분위 배제 — 12년 생존편향 제거 패널에서 적대적 검증
       3개 렌즈를 모두 통과한 유일한 양성 규칙(유니버스 EW 대비 연 +3.60%p,
       NW t=4.76). 수익을 늘리는 규칙이 아니라 재앙을 피하는 규칙이다.
    """
    rules = cfg.get("oversold", {})
    frame, counts = _liquidity_gate(frame, cfg)

    floor = rules.get("rsi_floor")
    if floor is not None:
        frame = frame[frame["rsi14"] >= floor]
        counts[f"RSI {floor} 이상 (추세적 붕괴 제외)"] = len(frame)

    # 변동성 상위 분위 배제. 절대 임계가 아니라 그날 풀 안에서의 상대 위치로
    # 건다 — 시장 전체 변동성이 국면에 따라 몇 배씩 달라지기 때문이다.
    top_pct = rules.get("exclude_top_vol_pct")
    atr = frame["atr_pct"].dropna()
    if top_pct and len(atr) >= rules.get("vol_screen_min_pool", 40):
        cutoff = atr.quantile(1 - top_pct / 100)
        frame = frame[(frame["atr_pct"] <= cutoff) | frame["atr_pct"].isna()]
        counts[f"ATR 상위 {top_pct}% 배제 (>{cutoff:.1f}%)"] = len(frame)
    elif top_pct:
        counts[f"ATR 상위 배제 (풀 {len(atr)}종목 부족, 미적용)"] = len(frame)

    return frame.reset_index(drop=True), counts


def oversold_score(frame: pd.DataFrame, cfg: dict, regime: str) -> pd.DataFrame:
    """RSI(14) 하나. 국면별 가중치도 없고 블록도 없다.

    팩터를 더하지 않는 것이 이 전략의 정의다. 5블록 모델이 단일 RSI를
    이기지 못했다는 관찰이 전환의 근거이므로, 여기에 무언가를 얹는 순간
    검증하려던 가설 자체가 사라진다.
    """
    frame["score"] = -z(frame["rsi14"])
    return frame


# --- 4. 역추세 + 시장 타이밍 --------------------------------------------


def reversal_timed_gate(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    return reversal_gate(frame, cfg)


def reversal_timed_score(frame: pd.DataFrame, cfg: dict, regime: str) -> pd.DataFrame:
    """종목 선정은 역추세와 동일. 차이는 '오늘 매매하는가'에만 있다.

    39일 표본에서 코스피 전일 일중 흐름이 마이너스인 날의 역추세 성과가
    +0.995%/일, 플러스인 날이 -2.458%/일이었다. 다만 예측변수 4개 x 전략 2개를
    비교한 뒤 고른 것이라 다중비교 보정 전 t=2.4, 보정 후엔 유의하지 않다.
    가설로만 두고 실제 데이터로 검증한다. 매매 여부는 screen.py가 판단한다.
    """
    return reversal_score(frame, cfg, regime)


# --- 5. 우량 과매도 2단계 (발행본) --------------------------------------


def quality_oversold_gate(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """oversold_gate를 **직접 호출**한 뒤 펀더멘털만 얹는다.

    ■ 복사해서 다시 쓰면 안 된다. 이 스펙에서 가장 틀리기 쉬운 지점이다.
      oversold_gate의 마지막 단계 'ATR 상위 20% 배제'는 절대 임계가 아니라
      그날 풀 안에서의 상대 분위다. 펀더멘털을 먼저 걸면 남은 풀의 변동성
      분포가 달라져(우량주는 저변동 편향) 컷오프 자체가 이동하고, 그러면 두
      변형의 차이가 '펀더멘털'이 아니라 '펀더멘털 + 서로 다른 변동성 컷'이
      되어 그림자 비교가 무효화된다.
      순서를 지키면 컷이 양쪽 모두 동일하고, quality_oversold 통과 종목은
      항상 oversold 풀의 부분집합이 된다.
    """
    frame, counts = oversold_gate(frame, cfg)
    rules = cfg.get("quality_growth", {})

    # (0) 데이터 가용성 가드. 네이버 재무 장애 시 3종목짜리 리스트를 발행하지
    #     않기 위해, 수집률이 낮으면 펀더멘털 게이트를 통째로 건너뛴다.
    have = frame["fin_has_data"]
    coverage = float(have.mean()) if len(frame) else 0.0
    floor = rules.get("min_data_coverage", 0.80)
    if coverage < floor:
        counts[f"재무 수집률 {coverage:.0%} < {floor:.0%} — 펀더멘털 게이트 미적용"] = len(frame)
        return frame.reset_index(drop=True), counts

    frame = frame[have]
    counts["재무제표 수집 성공"] = len(frame)

    # (1) 수익성. 결측=탈락 (커버리지 100%라 비용이 사실상 0이다).
    #     NaN 비교는 False가 되어 자동으로 탈락한다.
    min_roe = rules.get("min_roe_reported")
    if min_roe is not None:
        frame = frame[frame["roe_reported"] >= min_roe]
        counts[f"보고 ROE {min_roe}% 이상 (TTM 우선)"] = len(frame)

    # (2) 성장. 결측=탈락. 결측 종목은 최근 분할·재상장이라 성장을 평가할
    #     이력이 실제로 없으므로 탈락이 옳다.
    min_cagr = rules.get("min_rev_cagr")
    if min_cagr is not None:
        frame = frame[frame["rev_cagr"] >= min_cagr]
        counts[f"매출 CAGR {min_cagr}% 이상 (실적 3개 연도, 2년 환산)"] = len(frame)

    return frame.reset_index(drop=True), counts


@dataclass
class Variant:
    name: str
    label: str
    description: str
    gate: Callable[[pd.DataFrame, dict], tuple[pd.DataFrame, dict]]
    score: Callable[[pd.DataFrame, dict, str], pd.DataFrame]
    primary: bool = False
    # 시장 상태에 따라 그날 매매를 건너뛰는 변형인지
    market_timed: bool = False
    # 점수를 팩터 하나로만 매기는 변형인지. True면 5블록 분해가 없으므로
    # 리포트·대시보드가 블록 막대 대신 단일팩터 표시로 전환한다.
    single_factor: bool = False
    tags: list[str] = field(default_factory=list)


VARIANTS: list[Variant] = [
    Variant(
        name="reversal",
        label="역추세 (대조군)",
        description="낙폭 과다 + 저평가 + 우량성 5블록, 국면별 가중치. 2026-08-20까지 발행본이었다",
        gate=reversal_gate, score=reversal_score,
    ),
    Variant(
        name="momentum",
        label="추세추종 (대조군)",
        description="전환 전 규칙. 수익률·이격도·RSI가 높을수록 상위",
        gate=momentum_gate, score=momentum_score,
    ),
    Variant(
        name="oversold",
        label="과매도 단일팩터 (대조군)",
        description="RSI(14)만으로 순위. 펀더멘털 게이트 없음 — quality_oversold의 순수 대조군",
        gate=oversold_gate, score=oversold_score, single_factor=True,
    ),
    Variant(
        name="quality_oversold",
        label="우량 과매도 (2단계, 발행본)",
        description=(
            "1단계 펀더멘털 게이트(보고 ROE 5% 이상 · 매출 CAGR 0% 이상)로 기업을 "
            "거른 뒤, 2단계로 RSI(14) 단일 순위. 점수 규칙은 oversold와 완전히 "
            "동일하고 차이는 게이트뿐이다. 이 게이트에는 검증된 수익 근거가 없다 "
            "— 12년 패널 대용 검정에서 10종목 기준 증분 -2.16bp/일(t=-1.39)"
        ),
        gate=quality_oversold_gate, score=oversold_score,
        primary=True, single_factor=True,
    ),
    Variant(
        name="reversal_timed",
        label="역추세 + 시장 타이밍",
        description="종목은 역추세와 동일. 코스피 전일 일중이 플러스면 매매 안 함",
        gate=reversal_timed_gate, score=reversal_timed_score, market_timed=True,
    ),
]

PRIMARY = next(v for v in VARIANTS if v.primary)
