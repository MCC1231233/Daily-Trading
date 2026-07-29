"""추세 추종 vs 역추세, 어느 쪽이 일중(시가→종가) 수익을 예측하는가.

방법
  유동성 상위 종목의 60거래일 시세를 받아 (종목, 날짜) 패널을 만든다.
  t-1 종가까지의 정보로 팩터를 계산하고, t일의 시가→종가 수익률을 결과로 둔다.
  팩터를 5분위로 나눠 각 분위의 평균 일중 수익률과 승률을 비교한다.

주의
  - 팩터는 전부 t-1 종가 시점 정보만 사용한다 (lookahead 없음).
  - 표본은 최근 60거래일 한 국면뿐이다. 다른 국면으로 일반화되지 않는다.
  - 수수료·세금·슬리피지 미반영.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasource import NaverClient, fetch_universe  # noqa: E402

ROUND_TRIP_COST = 0.18  # 거래세 0.15% + 왕복 수수료 약 0.03%


def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def build_panel(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for code, h in histories.items():
        if h.empty or len(h) < 30:
            continue
        h = h.sort_values("date").reset_index(drop=True)

        close = h["close"]
        feat = pd.DataFrame({
            "code": code,
            "date": h["date"],
            # --- t-1 종가 시점 팩터 (shift(1)로 하루 밀어 lookahead 제거) ---
            "ret_20d": (close / close.shift(20) - 1).shift(1) * 100,
            "ret_5d": (close / close.shift(5) - 1).shift(1) * 100,
            "ret_1d": close.pct_change().shift(1) * 100,
            "rsi14": rsi_series(close).shift(1),
            "ma20_disp": (close / close.rolling(20).mean() - 1).shift(1) * 100,
            # --- t일 결과: 시가 매수 → 종가 매도 ---
            "intraday": (h["close"] / h["open"] - 1) * 100,
        })
        rows.append(feat)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
    # 극단값이 평균을 지배하지 않도록 상하 0.5% 절사
    lo, hi = panel["intraday"].quantile([0.005, 0.995])
    return panel[panel["intraday"].between(lo, hi)]


def quintile_report(panel: pd.DataFrame, factor: str, label: str) -> None:
    try:
        panel = panel.copy()
        panel["q"] = pd.qcut(panel[factor], 5, labels=["Q1(최저)", "Q2", "Q3", "Q4", "Q5(최고)"])
    except ValueError:
        print(f"  {label}: 분위 생성 실패")
        return

    grouped = panel.groupby("q", observed=True)["intraday"]
    print(f"\n[{label}]")
    print(f"  {'분위':<9} {'평균':>8} {'중앙값':>8} {'승률':>7} {'표본':>7}  {'비용차감':>9}")
    for name, series in grouped:
        print(f"  {name:<9} {series.mean():>7.3f}% {series.median():>7.3f}% "
              f"{(series > 0).mean() * 100:>6.1f}% {len(series):>7,}  "
              f"{series.mean() - ROUND_TRIP_COST:>8.3f}%")

    q1, q5 = grouped.get_group("Q1(최저)"), grouped.get_group("Q5(최고)")
    spread = q1.mean() - q5.mean()
    # 두 표본 평균 차이의 t값. |t| > 2면 우연으로 보기 어렵다.
    se = np.sqrt(q1.var() / len(q1) + q5.var() / len(q5))
    tstat = spread / se if se else 0
    direction = "역추세 우세 (낮을수록 좋음)" if spread > 0 else "추세추종 우세 (높을수록 좋음)"
    print(f"  → Q1 - Q5 = {spread:+.3f}%p (t={tstat:+.2f})  {direction}")


def main() -> int:
    print("유니버스 수집 중...")
    universe = fetch_universe()
    universe = universe[
        (universe["close"] >= 2000)
        & (universe["marcap"] >= 3e11)
        & (universe["amount"] >= 5e9)
    ].sort_values("amount", ascending=False).head(300)
    codes = universe["code"].tolist()
    print(f"대상 {len(codes)}종목, 60거래일 시세 수집 중...")

    client = NaverClient(workers=8)
    histories = client.bulk(codes, "price_history", days=60)

    panel = build_panel(histories)
    print(f"\n패널: {panel['code'].nunique()}종목 x {panel['date'].nunique()}일 "
          f"= {len(panel):,} 관측치")
    print(f"기간: {panel['date'].min()} ~ {panel['date'].max()}")
    print(f"전체 평균 일중 수익률: {panel['intraday'].mean():+.3f}% "
          f"(승률 {(panel['intraday'] > 0).mean() * 100:.1f}%)")
    print(f"왕복 거래비용 가정: {ROUND_TRIP_COST}%")

    for factor, label in [
        ("ret_20d", "20일 수익률 (추세추종=Q5 / 역추세=Q1)"),
        ("ret_5d", "5일 수익률"),
        ("ret_1d", "전일 수익률"),
        ("rsi14", "RSI(14)"),
        ("ma20_disp", "20일선 이격도"),
    ]:
        quintile_report(panel, factor, label)

    # 두 팩터를 교차: 장기 추세 위 / 단기 과매도 조합이 실제로 유효한가
    print("\n\n[교차 분석] 20일 수익률 × RSI — 각 칸의 평균 일중 수익률(%)")
    panel = panel.copy()
    panel["mom_bucket"] = pd.qcut(panel["ret_20d"], 3, labels=["추세↓", "추세→", "추세↑"])
    panel["rsi_bucket"] = pd.qcut(panel["rsi14"], 3, labels=["과매도", "중립", "과매수"])
    pivot = panel.pivot_table(
        index="mom_bucket", columns="rsi_bucket", values="intraday",
        aggfunc="mean", observed=True,
    )
    counts = panel.pivot_table(
        index="mom_bucket", columns="rsi_bucket", values="intraday",
        aggfunc="size", observed=True,
    )
    print(pivot.round(3).to_string())
    print("\n표본 수")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
