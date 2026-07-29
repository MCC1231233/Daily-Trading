"""하루 수익률을 오버나이트와 일중으로 쪼개 어디서 돈이 나오는지 본다.

close(t-1) --오버나이트--> open(t) --일중--> close(t)

일중 구간이 -1%/일이었다면, 같은 기간 종가 기준 수익률이 그만큼 나쁘지 않은 한
오버나이트 구간이 그 손실을 메우고 있다는 뜻이다. 어느 구간에 매매를 붙일지가
팩터를 다듬는 것보다 훨씬 큰 결정이므로 먼저 확인한다.

주의: 모든 팩터는 t-1 종가 시점 정보만 사용한다.
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
        close, prev_close = h["close"], h["close"].shift(1)
        rows.append(pd.DataFrame({
            "code": code,
            "date": h["date"],
            "ret_5d": (close / close.shift(5) - 1).shift(1) * 100,
            "ret_20d": (close / close.shift(20) - 1).shift(1) * 100,
            "rsi14": rsi_series(close).shift(1),
            "ma20_disp": (close / close.rolling(20).mean() - 1).shift(1) * 100,
            "amount": (close * h["volume"]).shift(1),
            # 세 구간으로 분해
            "overnight": (h["open"] / prev_close - 1) * 100,
            "intraday": (close / h["open"] - 1) * 100,
            "full_day": (close / prev_close - 1) * 100,
        }))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = panel["full_day"].quantile([0.005, 0.995])
    return panel[panel["full_day"].between(lo, hi)]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if not std or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return ((series - series.mean()) / std).clip(-3, 3).fillna(0)


def decomposition(panel: pd.DataFrame) -> None:
    print("=" * 68)
    print("1. 구간별 수익률 분해 (전 종목 평균)")
    print("=" * 68)
    print(f"  {'구간':<22} {'평균':>9} {'중앙값':>9} {'승률':>8} {'표준편차':>9}")
    for column, label in [
        ("overnight", "오버나이트 (종가→시가)"),
        ("intraday", "일중 (시가→종가)"),
        ("full_day", "하루 전체 (종가→종가)"),
    ]:
        s = panel[column]
        print(f"  {label:<22} {s.mean():>8.3f}% {s.median():>8.3f}% "
              f"{(s > 0).mean() * 100:>7.1f}% {s.std():>8.2f}%")


def factor_by_window(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 68)
    print("2. 팩터별 5분위 — 구간마다 방향이 같은가")
    print("=" * 68)
    for factor, label in [("rsi14", "RSI(14)"), ("ret_5d", "5일 수익률")]:
        work = panel.copy()
        work["q"] = pd.qcut(work[factor], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        print(f"\n[{label}]  Q1=최저 ... Q5=최고")
        print(f"  {'분위':<5} {'오버나이트':>11} {'일중':>10} {'하루전체':>11}")
        for name, sub in work.groupby("q", observed=True):
            print(f"  {name:<5} {sub['overnight'].mean():>10.3f}% "
                  f"{sub['intraday'].mean():>9.3f}% {sub['full_day'].mean():>10.3f}%")
        q1 = work[work["q"] == "Q1"]
        q5 = work[work["q"] == "Q5"]
        print("  Q1-Q5 " + "  ".join(
            f"{c}:{q1[c].mean() - q5[c].mean():+.3f}%p"
            for c in ("overnight", "intraday", "full_day")))


def strategy_by_window(panel: pd.DataFrame) -> None:
    """매일 상위 10종목을 뽑아 세 구간 각각에서 돌려본다."""
    print("\n" + "=" * 68)
    print("3. 매일 10종목 선정 — 어느 구간에 붙일 때 가장 좋은가")
    print("=" * 68)

    daily = []
    for date, day in panel.groupby("date"):
        day = day.nlargest(min(200, len(day)), "amount").copy()
        if len(day) < 40:
            continue
        mom = (0.35 * zscore(day["ret_5d"]) + 0.30 * zscore(day["ret_20d"])
               + 0.20 * zscore(day["ma20_disp"]) + 0.15 * zscore(day["rsi14"]))
        day["mom"] = mom

        selections = {
            "역추세": day.nsmallest(PICKS, "mom"),
            "추세추종": day.nlargest(PICKS, "mom"),
            "과매도(RSI)": day.nsmallest(PICKS, "rsi14"),
            "전체시장": day,
        }
        record = {"date": date}
        for name, sel in selections.items():
            for window in ("overnight", "intraday", "full_day"):
                record[f"{name}|{window}"] = sel[window].mean()
        daily.append(record)

    results = pd.DataFrame(daily)
    print(f"\n{len(results)}거래일, 왕복비용 {COST}% 차감 후 누적 수익률")
    print(f"  {'전략':<14} {'오버나이트':>12} {'일중':>12} {'하루전체':>12}")
    for name in ("역추세", "추세추종", "과매도(RSI)", "전체시장"):
        cells = []
        for window in ("overnight", "intraday", "full_day"):
            net = results[f"{name}|{window}"] - COST
            cumulative = ((1 + net / 100).prod() - 1) * 100
            cells.append(f"{cumulative:>11.1f}%")
        print(f"  {name:<14} " + " ".join(cells))

    print(f"\n  일평균 (비용차감)")
    print(f"  {'전략':<14} {'오버나이트':>12} {'일중':>12} {'하루전체':>12}")
    for name in ("역추세", "추세추종", "과매도(RSI)", "전체시장"):
        cells = []
        for window in ("overnight", "intraday", "full_day"):
            net = results[f"{name}|{window}"] - COST
            cells.append(f"{net.mean():>10.3f}% ")
        print(f"  {name:<14} " + " ".join(cells))

    print(f"\n  승률")
    print(f"  {'전략':<14} {'오버나이트':>12} {'일중':>12} {'하루전체':>12}")
    for name in ("역추세", "추세추종", "과매도(RSI)", "전체시장"):
        cells = []
        for window in ("overnight", "intraday", "full_day"):
            net = results[f"{name}|{window}"] - COST
            cells.append(f"{(net > 0).mean() * 100:>10.1f}% ")
        print(f"  {name:<14} " + " ".join(cells))


def main() -> int:
    print("데이터 수집 중...")
    universe = fetch_universe()
    universe = universe[
        (universe["close"] >= 2000) & (universe["marcap"] >= 3e11)
        & (universe["amount"] >= 3e9)
    ].sort_values("amount", ascending=False).head(400)

    client = NaverClient(workers=8)
    histories = client.bulk(universe["code"].tolist(), "price_history", days=60)
    panel = build_panel(histories)
    print(f"패널: {panel['code'].nunique()}종목 x {panel['date'].nunique()}일 "
          f"= {len(panel):,} 관측치")
    print(f"기간: {panel['date'].min()} ~ {panel['date'].max()}\n")

    decomposition(panel)
    factor_by_window(panel)
    strategy_by_window(panel)

    print("\n" + "=" * 68)
    print("주의: 표본 39거래일, 폭락 국면 하나. 다른 국면으로 일반화되지 않음.")
    print("오버나이트 전략은 갭 리스크를 안고 자며, 장 마감 동시호가 체결을 가정함.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
