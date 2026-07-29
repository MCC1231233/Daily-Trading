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
    if gates["require_coverage"]:
        frame = frame[frame["has_coverage"]]
        counts["애널리스트 커버리지 존재"] = len(frame)

    frame = frame[frame["roe"] >= gates["min_roe"]]
    counts[f"ROE {gates['min_roe']}% 이상"] = len(frame)
    frame = frame[frame["pbr"] <= gates["max_pbr"]]
    counts[f"PBR {gates['max_pbr']}배 이하"] = len(frame)
    frame = frame[frame["target_upside"] >= gates["min_target_upside"]]
    counts[f"목표주가 상승여력 {gates['min_target_upside']}% 이상"] = len(frame)

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
    """유동성 외에는 거의 아무것도 안 거른다.

    백테스트에서 RSI 하나만 쓴 버전이 5블록 모델과 거의 같은 성과를 냈다.
    정교함이 실제로 값을 하는지 확인하려면 이 단순 기준선이 있어야 한다.
    """
    frame, counts = _liquidity_gate(frame, cfg)
    return frame.reset_index(drop=True), counts


def oversold_score(frame: pd.DataFrame, cfg: dict, regime: str) -> pd.DataFrame:
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
    tags: list[str] = field(default_factory=list)


VARIANTS: list[Variant] = [
    Variant(
        name="reversal",
        label="역추세 (발행본)",
        description="낙폭 과다 + 저평가 + 우량성 5블록, 국면별 가중치",
        gate=reversal_gate, score=reversal_score, primary=True,
    ),
    Variant(
        name="momentum",
        label="추세추종 (대조군)",
        description="전환 전 규칙. 수익률·이격도·RSI가 높을수록 상위",
        gate=momentum_gate, score=momentum_score,
    ),
    Variant(
        name="oversold",
        label="과매도 단일팩터",
        description="RSI만 보고 가장 낮은 종목. 정교함이 값을 하는지 확인하는 기준선",
        gate=oversold_gate, score=oversold_score,
    ),
    Variant(
        name="reversal_timed",
        label="역추세 + 시장 타이밍",
        description="종목은 역추세와 동일. 코스피 전일 일중이 플러스면 매매 안 함",
        gate=reversal_timed_gate, score=reversal_timed_score, market_timed=True,
    ),
]

PRIMARY = next(v for v in VARIANTS if v.primary)
