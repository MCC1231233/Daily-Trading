"""팩터 계산.

전략이 "9시 시가 매수 → 15시 종가 매도"이므로, 계산하는 모든 값은
**전 거래일 종가까지의 정보만** 사용한다. 당일 시가조차 쓰지 않는다.
그래야 실제로 장 시작 전에 나올 수 있는 후보군이 된다.

가장 중요한 블록은 intraday_profile이다. 보유 기간이 하루 6시간이라
"이 종목이 원래 시가 대비 종가가 오르는 성향인가"가 20일 모멘텀보다
직접적으로 전략에 맞는 질문이기 때문이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe(value) -> float | None:
    """NaN/inf를 JSON에 넣을 수 있는 None으로 바꾼다."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(value) or np.isinf(value)) else round(value, 4)


# --- 기술적 지표 --------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = gain.iloc[-1] / last_loss
    return _safe(100 - 100 / (1 + rs))


def macd_histogram(close: pd.Series) -> float | None:
    """MACD 히스토그램을 종가 대비 %로 정규화 (종목 간 비교 가능하게)."""
    if len(close) < 35:
        return None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return _safe((macd_line.iloc[-1] - signal.iloc[-1]) / close.iloc[-1] * 100)


def bollinger_percent_b(close: pd.Series, period: int = 20) -> float | None:
    """볼린저 밴드 내 위치. 0=하단, 1=상단."""
    if len(close) < period:
        return None
    window = close.tail(period)
    mean, std = window.mean(), window.std()
    if std == 0:
        return 0.5
    lower, upper = mean - 2 * std, mean + 2 * std
    return _safe((close.iloc[-1] - lower) / (upper - lower))


def atr_percent(frame: pd.DataFrame, period: int = 14) -> float | None:
    """ATR을 종가 대비 %로. 일중 매매에서 기대 변동폭 = 수익 여지이자 리스크."""
    if len(frame) < period + 1:
        return None
    high, low, prev_close = frame["high"], frame["low"], frame["close"].shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.tail(period).mean()
    return _safe(atr / frame["close"].iloc[-1] * 100)


def ma_disparity(close: pd.Series, period: int = 20) -> float | None:
    """이격도. 종가가 20일선 대비 몇 % 위/아래인지."""
    if len(close) < period:
        return None
    ma = close.tail(period).mean()
    return _safe((close.iloc[-1] / ma - 1) * 100) if ma else None


def volume_surge(volume: pd.Series, period: int = 20) -> float | None:
    """직전일 거래량 / 20일 평균 거래량."""
    if len(volume) < period + 1:
        return None
    baseline = volume.tail(period).mean()
    return _safe(volume.iloc[-1] / baseline) if baseline else None


# --- 일중 성향 (이 전략의 핵심 블록) -------------------------------------


def reversal_profile(frame: pd.DataFrame, lookback: int = 20) -> dict:
    """낙폭 반등 전략에 특화된 지표들.

    "많이 빠진 종목을 산다"의 최대 위험은 하락에 실제 이유가 있는 경우다.
    아래 지표들은 "팔 사람이 대체로 다 팔았는가"를 가늠하려는 것이지,
    낙폭 자체를 좋게 보려는 게 아니다.

    lower_shadow    전일 캔들에서 저가 대비 종가가 되돌린 비율. 1에 가까우면
                    장중 저점에서 매수세가 받쳤다는 뜻.
    down_streak     연속 하락 마감 일수. 길수록 되돌림 여지가 크지만
                    동시에 추세적 악재일 가능성도 커진다.
    capitulation    하락일에 거래량이 터졌는지 (항복 매도 신호).
    oc_after_down   **이 전략의 핵심 검증 지표.** 전일 하락한 다음 날의
                    시가→종가 수익률 평균. 실제 매매 조건과 정확히 같은
                    상황에서 이 종목이 어떻게 움직였는지를 본다.
    down_day_count  표본 수. oc_after_down의 신뢰도를 판단하는 데 쓴다.
    """
    if len(frame) < 25:
        return {}

    result: dict = {}
    last = frame.iloc[-1]

    span = last["high"] - last["low"]
    if span > 0:
        result["lower_shadow"] = _safe((last["close"] - last["low"]) / span)

    # 연속 하락 마감 일수
    changes = frame["close"].diff().tail(lookback)
    streak = 0
    for value in reversed(changes.tolist()):
        if value is None or np.isnan(value) or value >= 0:
            break
        streak += 1
    result["down_streak"] = streak

    volumes = frame["volume"]
    if len(volumes) >= 21:
        baseline = volumes.iloc[-21:-1].mean()
        if baseline:
            ratio = volumes.iloc[-1] / baseline
            fell = last["close"] < frame["close"].iloc[-2]
            result["capitulation"] = _safe(ratio if fell else 0.0)

    # 전일이 하락이었던 날들만 골라 그날의 시가→종가를 본다.
    work = frame.copy()
    work["prev_change"] = work["close"].pct_change().shift(1)
    work["intraday"] = (work["close"] / work["open"] - 1) * 100
    after_down = work[work["prev_change"] < 0]["intraday"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    result["down_day_count"] = int(len(after_down))
    if len(after_down) >= 5:
        result["oc_after_down"] = _safe(after_down.mean())
        result["oc_after_down_winrate"] = _safe((after_down > 0).mean() * 100)

    return result


def valuation_profile(snapshot: dict, last_close: float | None) -> dict:
    """저평가·수익성 지표. '싸다'와 '망가졌다'를 구분하는 데 쓴다."""
    result: dict = {}

    eps, bps = snapshot.get("eps"), snapshot.get("bps")
    if eps is not None and bps:
        # ROE = EPS / BPS. 자기자본 대비 얼마나 벌고 있는지.
        result["roe"] = _safe(eps / bps * 100)
    result["eps"] = _safe(eps)
    result["is_profitable"] = bool(eps is not None and eps > 0)

    high_52w = snapshot.get("high_52w")
    if high_52w and last_close:
        # 52주 고점 대비 낙폭. 이 전략에서 '낙폭 과다'의 가장 직접적인 척도.
        result["drawdown_52w"] = _safe((last_close / high_52w - 1) * 100)

    # 애널리스트가 커버하는 종목인지. 목표주가가 있다는 건 최소한
    # 기관이 들여다보고 있다는 뜻이라, 정보 공백 리스크가 낮다.
    result["has_coverage"] = bool(snapshot.get("target_price"))

    return result


def intraday_profile(frame: pd.DataFrame, lookback: int = 20) -> dict:
    """과거 시가→종가 수익률의 분포.

    oc_mean    평균 일중 수익률 (%)
    oc_winrate 시가보다 종가가 높았던 날의 비율
    oc_sharpe  평균 / 표준편차 — 일관성. 한 방에 번 것과 꾸준히 번 것을 구분한다.
    gap_follow 갭 상승일의 일중 수익률 평균. 양수면 갭을 이어가고,
               음수면 갭이 메워지는(페이드) 성향이라 9시 매수에 불리하다.
    """
    if len(frame) < lookback + 1:
        return {}

    recent = frame.tail(lookback).copy()
    open_to_close = (recent["close"] / recent["open"] - 1) * 100
    open_to_close = open_to_close.replace([np.inf, -np.inf], np.nan).dropna()
    if open_to_close.empty:
        return {}

    std = open_to_close.std()

    # 갭: 당일 시가 vs 전일 종가
    with_prev = frame.tail(lookback + 1).copy()
    with_prev["prev_close"] = with_prev["close"].shift(1)
    with_prev = with_prev.dropna(subset=["prev_close"])
    gap_pct = (with_prev["open"] / with_prev["prev_close"] - 1) * 100
    intraday_pct = (with_prev["close"] / with_prev["open"] - 1) * 100
    gap_up_days = intraday_pct[gap_pct > 0]

    return {
        "oc_mean": _safe(open_to_close.mean()),
        "oc_winrate": _safe((open_to_close > 0).mean() * 100),
        "oc_sharpe": _safe(open_to_close.mean() / std) if std else None,
        "oc_best": _safe(open_to_close.max()),
        "oc_worst": _safe(open_to_close.min()),
        "gap_follow": _safe(gap_up_days.mean()) if len(gap_up_days) >= 3 else None,
        "avg_range_pct": _safe(
            ((recent["high"] - recent["low"]) / recent["open"] * 100).mean()
        ),
    }


# --- 수급 --------------------------------------------------------------


def flow_profile(trend: pd.DataFrame, avg_volume: float | None) -> dict:
    """외국인/기관 순매수를 평균 거래량으로 나눠 종목 규모와 무관하게 비교.

    절대 주식 수로 보면 대형주가 항상 이기므로, 회전율 기준으로 정규화한다.
    """
    if trend.empty or not avg_volume:
        return {}

    recent5 = trend.tail(5)
    result = {
        "foreign_net_5d_pct": _safe(
            recent5["foreign_net"].sum() / (avg_volume * 5) * 100
        ),
        "organ_net_5d_pct": _safe(recent5["organ_net"].sum() / (avg_volume * 5) * 100),
        "foreign_buy_days": int((recent5["foreign_net"] > 0).sum()),
        "organ_buy_days": int((recent5["organ_net"] > 0).sum()),
    }

    ratios = trend["foreign_ratio"].dropna()
    if len(ratios) >= 2:
        result["foreign_ratio"] = _safe(ratios.iloc[-1])
        result["foreign_ratio_change"] = _safe(ratios.iloc[-1] - ratios.iloc[0])

    # 외국인과 기관이 같은 방향이면 신뢰도가 올라간다.
    foreign_5d = recent5["foreign_net"].sum()
    organ_5d = recent5["organ_net"].sum()
    result["dual_buying"] = bool(foreign_5d > 0 and organ_5d > 0)

    return result


# --- 종합 --------------------------------------------------------------


def growth_profile(annual: dict, quarter: dict, snapshot: dict) -> dict:
    """1단계(기업 선별) 팩터.

    게이트에 쓰는 것은 roe_reported, rev_cagr 둘뿐이고 나머지는 전부 화면
    표시 전용이다. 설계 원칙 셋:

      1) 실적과 컨센서스를 절대 한 계열로 잇지 않는다. 실적으로만 추세를
         만들고 컨센서스는 별도 컬럼으로 표시만 한다. 컨센서스 보유 여부가
         퀄리티가 아니라 시총 그 자체이기 때문이다(시총 Q1 보유율 0.8% vs
         Q5 65.8%, rank-biserial +0.771).
      2) 비율 팩터는 분모가 0 근방에서 폭발한다. 실측 컨센 영업이익 성장률
         최대 +3,150%(NC, 직전 영업이익 ~0). 전부 ±300% 클리핑하고 분모가
         양수일 때만 계산한다 — 적자→흑자 전환은 비율로 표현할 수 없다.
      3) 기존 roe(=EPS/BPS) 컬럼을 절대 덮어쓰지 않는다. reversal 변형이
         게이트 입력으로 쓰고 있어 덮어쓰면 대조군 계열이 끊긴다.
    """
    result: dict = {"fin_has_data": False}
    if not annual:
        return result
    result["fin_has_data"] = True

    def last(values):
        for value in reversed(values or []):
            if value is not None:
                return value
        return None

    def clip(x):
        return None if x is None else _safe(max(min(x, 300.0), -300.0))

    revenue = [v for v in (annual.get("revenue_actual") or []) if v is not None]
    op = annual.get("op_actual") or []
    op_clean = [v for v in op if v is not None]

    # --- 성장: 매출 ---
    # 실적 연도가 항상 3개뿐이라 CAGR 지수는 2다. 화면에 '3년'이라고 적으면
    # 거짓이 되므로 rev_cagr_years를 함께 내보낸다.
    if len(revenue) >= 2 and revenue[0] > 0:
        years = len(revenue) - 1
        result["rev_cagr"] = _safe(((revenue[-1] / revenue[0]) ** (1 / years) - 1) * 100)
        result["rev_cagr_years"] = int(years)
    if len(revenue) >= 2 and revenue[-2] > 0:
        result["rev_yoy"] = _safe((revenue[-1] / revenue[-2] - 1) * 100)

    # --- 성장: 영업이익 (표시 전용) ---
    if len(op_clean) >= 2 and op_clean[-2] > 0:
        result["op_yoy"] = clip((op_clean[-1] / op_clean[-2] - 1) * 100)
    result["op_positive_years"] = int(sum(1 for v in op if v is not None and v > 0))
    result["fin_years"] = int(annual.get("n_actual") or 0)

    # --- 수익성 ---
    # TTM(분기표) 우선, 없으면 연간표로 폴백. 어느 쪽을 썼는지 반드시 남긴다.
    roe_annual = last(annual.get("roe_actual"))
    roe_ttm = last((quarter or {}).get("roe_actual"))
    result["roe_annual"] = _safe(roe_annual)
    result["roe_ttm"] = _safe(roe_ttm)
    result["roe_reported"] = _safe(roe_ttm if roe_ttm is not None else roe_annual)
    result["roe_source"] = (
        "TTM" if roe_ttm is not None
        else ("연간" if roe_annual is not None else None)
    )

    result["op_margin"] = _safe(last(annual.get("op_margin_actual")))
    # 부채비율은 표시 전용이다. 200% 컷은 금융계열을 통째로 배제한다
    # (금융 부채비율 중앙 1,029% vs 비금융 79%) — 업종 배제로 작동한다.
    result["debt_ratio"] = _safe(last(annual.get("debt_ratio_actual")))

    # --- 이익 개선 방향 (표시 전용, 게이트 금지) ---
    op_cons = last(annual.get("op_consensus"))
    op_last = last(op)
    if op_cons is not None and op_last is not None and op_last > 0:
        result["cons_op_growth"] = clip((op_cons / op_last - 1) * 100)

    eps, forward_eps = snapshot.get("eps"), snapshot.get("forward_eps")
    if eps and forward_eps and eps > 0:
        result["eps_growth_fwd"] = clip((forward_eps / eps - 1) * 100)

    return result


def compute_all(
    history: pd.DataFrame,
    trend: pd.DataFrame,
    snapshot: dict,
    index_return_20d: float | None = None,
    annual: dict | None = None,
    quarter: dict | None = None,
) -> dict:
    """한 종목의 전체 팩터 딕셔너리."""
    if history.empty or len(history) < 25:
        return {}

    close = history["close"]
    volume = history["volume"]
    avg_volume_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else None

    factors: dict = {
        "last_close": _safe(close.iloc[-1]),
        "last_date": history["date"].iloc[-1],
        "rsi14": rsi(close),
        "macd_hist": macd_histogram(close),
        "bb_percent_b": bollinger_percent_b(close),
        "atr_pct": atr_percent(history),
        "ma20_disparity": ma_disparity(close, 20),
        "ma5_disparity": ma_disparity(close, 5),
        "volume_surge": volume_surge(volume),
        "ret_5d": _safe((close.iloc[-1] / close.iloc[-6] - 1) * 100)
        if len(close) > 6
        else None,
        "ret_20d": _safe((close.iloc[-1] / close.iloc[-21] - 1) * 100)
        if len(close) > 21
        else None,
    }

    factors.update(intraday_profile(history))
    factors.update(reversal_profile(history))
    factors.update(flow_profile(trend, avg_volume_20))
    factors.update(valuation_profile(snapshot, factors.get("last_close")))
    factors.update(growth_profile(annual or {}, quarter or {}, snapshot))

    # 지수 대비 상대강도 — 시장 전체가 오른 것과 종목이 강한 것을 구분한다.
    if index_return_20d is not None and factors.get("ret_20d") is not None:
        factors["relative_strength"] = _safe(factors["ret_20d"] - index_return_20d)

    # 밸류에이션과 컨센서스
    factors["per"] = _safe(snapshot.get("per"))
    factors["forward_per"] = _safe(snapshot.get("forward_per"))
    factors["pbr"] = _safe(snapshot.get("pbr"))
    factors["dividend_yield"] = _safe(snapshot.get("dividend_yield"))
    factors["recomm_score"] = _safe(snapshot.get("recomm_score"))

    target = snapshot.get("target_price")
    if target and factors.get("last_close"):
        factors["target_upside"] = _safe(
            (target / factors["last_close"] - 1) * 100
        )
        factors["target_price"] = _safe(target)

    high_52w, low_52w = snapshot.get("high_52w"), snapshot.get("low_52w")
    if high_52w and low_52w and high_52w > low_52w:
        factors["pos_52w"] = _safe(
            (factors["last_close"] - low_52w) / (high_52w - low_52w) * 100
        )

    return factors


def zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """결측은 0(중립)으로, 이상치는 ±3σ로 자른다.

    클리핑을 하는 이유: 한 종목이 특정 팩터에서 극단값이면 z-score가 10을 넘어
    다른 모든 팩터를 압도해버린다. 랭킹이 사실상 그 한 종목의 한 지표로 결정된다.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std()
    if not std or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return ((numeric - numeric.mean()) / std).clip(-clip, clip).fillna(0.0)
