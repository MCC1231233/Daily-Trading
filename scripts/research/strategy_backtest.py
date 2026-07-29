"""추세추종 / 역추세 / 혼합 — 실제로 매일 10종목 뽑아 돌려본다.

매일 t-1 종가 시점 정보로 종목을 고르고, t일 시가 매수 / 종가 매도 수익률을
기록한다. 후보 선정에 쓰는 정보는 전부 shift(1) 처리해 lookahead가 없다.

비교 대상
  momentum  추세추종: 최근 수익률·이격도·RSI가 높을수록 상위
  reversal  역추세  : 낙폭·과매도일수록 상위
  hybrid_5  절반씩 : 각 전략 상위 5종목
  hybrid_mix 점수합 : 두 점수를 0.5씩 섞어 상위 10
  oversold  단일팩터: RSI만 보고 가장 낮은 10종목
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasource import NaverClient, fetch_universe  # noqa: E402

COST = 0.18
PICKS = 10


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
        close, volume = h["close"], h["volume"]
        rows.append(pd.DataFrame({
            "code": code,
            "date": h["date"],
            "ret_20d": (close / close.shift(20) - 1).shift(1) * 100,
            "ret_5d": (close / close.shift(5) - 1).shift(1) * 100,
            "rsi14": rsi_series(close).shift(1),
            "ma20_disp": (close / close.rolling(20).mean() - 1).shift(1) * 100,
            "amount": (close * volume).shift(1),
            "intraday": (h["close"] / h["open"] - 1) * 100,
        }))
    panel = pd.concat(rows, ignore_index=True)
    return panel.replace([np.inf, -np.inf], np.nan).dropna()


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if not std or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return ((series - series.mean()) / std).clip(-3, 3).fillna(0)


def run(panel: pd.DataFrame) -> pd.DataFrame:
    daily = []
    for date, day in panel.groupby("date"):
        # 그날 거래대금 상위 200종목만 후보로 (실제 운용 가능성 반영)
        day = day.nlargest(min(200, len(day)), "amount").copy()
        if len(day) < 40:
            continue

        # 두 전략의 점수는 같은 팩터에 부호만 반대다.
        day["score_mom"] = (
            0.35 * zscore(day["ret_5d"]) + 0.30 * zscore(day["ret_20d"])
            + 0.20 * zscore(day["ma20_disp"]) + 0.15 * zscore(day["rsi14"])
        )
        day["score_rev"] = -day["score_mom"]

        picks = {
            "momentum": day.nlargest(PICKS, "score_mom"),
            "reversal": day.nlargest(PICKS, "score_rev"),
            "oversold": day.nsmallest(PICKS, "rsi14"),
        }
        picks["hybrid_5"] = pd.concat([
            day.nlargest(PICKS // 2, "score_mom"),
            day.nlargest(PICKS // 2, "score_rev"),
        ])
        day["score_mix"] = 0.5 * day["score_mom"] + 0.5 * day["score_rev"]
        picks["hybrid_mix"] = day.nlargest(PICKS, "score_mix")

        record = {"date": date, "universe": day["intraday"].mean()}
        for name, frame in picks.items():
            record[name] = frame["intraday"].mean()
        daily.append(record)

    return pd.DataFrame(daily).sort_values("date").reset_index(drop=True)


def summarize(results: pd.DataFrame) -> None:
    strategies = ["momentum", "reversal", "hybrid_5", "hybrid_mix", "oversold", "universe"]
    print(f"\n기간: {results['date'].min()} ~ {results['date'].max()} "
          f"({len(results)}거래일)\n")
    print(f"{'전략':<12} {'일평균':>9} {'비용차감':>9} {'누적':>10} {'승률':>7} {'표준편차':>9} {'샤프':>7} {'최악일':>9}")

    for name in strategies:
        series = results[name]
        net = series - COST
        cumulative = (1 + net / 100).prod() - 1
        sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() else 0
        print(f"{name:<12} {series.mean():>8.3f}% {net.mean():>8.3f}% "
              f"{cumulative * 100:>9.1f}% {(net > 0).mean() * 100:>6.1f}% "
              f"{series.std():>8.2f}% {sharpe:>7.2f} {series.min():>8.2f}%")

    print("\n일별 수익률 (비용 차감 전, %)")
    print(f"{'날짜':<12} " + " ".join(f"{s[:8]:>9}" for s in strategies))
    for _, row in results.tail(15).iterrows():
        print(f"{row['date']:<12} " + " ".join(f"{row[s]:>9.2f}" for s in strategies))


def main() -> int:
    print("데이터 수집 중...")
    universe = fetch_universe()
    universe = universe[
        (universe["close"] >= 2000)
        & (universe["marcap"] >= 3e11)
        & (universe["amount"] >= 3e9)
    ].sort_values("amount", ascending=False).head(400)

    client = NaverClient(workers=8)
    histories = client.bulk(universe["code"].tolist(), "price_history", days=60)

    panel = build_panel(histories)
    print(f"패널: {panel['code'].nunique()}종목 x {panel['date'].nunique()}일")

    results = run(panel)
    summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
