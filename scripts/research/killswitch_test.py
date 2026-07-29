"""매매를 쉬는 날을 고를 수 있는가.

앞선 백테스트에서 모든 전략이 손실이었고, 시장 전체의 일중 드리프트가
-1.02%/일이었다. 팩터를 아무리 다듬어도 시장 자체가 장중에 빠지는 날에는
답이 없다. 그렇다면 '오늘은 안 한다'를 t-1 정보로 판단할 수 있는지가
팩터 개선보다 훨씬 큰 문제다.

t-1 종가 시점에 알 수 있는 시장 상태로 t일의 일중 드리프트를 예측해본다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasource import NaverClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy_backtest import build_panel, run  # noqa: E402
from datasource import fetch_universe  # noqa: E402

COST = 0.18


def kospi_frame(client: NaverClient) -> pd.DataFrame:
    rows = client._get("index/KOSPI/price?pageSize=60&page=1")
    frame = pd.DataFrame({
        "date": [str(r["localTradedAt"])[:10] for r in rows],
        "close": [float(str(r["closePrice"]).replace(",", "")) for r in rows],
        "open": [float(str(r["openPrice"]).replace(",", "")) for r in rows],
    }).sort_values("date").reset_index(drop=True)

    frame["kospi_prev_ret"] = frame["close"].pct_change().shift(1) * 100
    frame["kospi_prev_intraday"] = ((frame["close"] / frame["open"] - 1) * 100).shift(1)
    ma20 = frame["close"].rolling(20).mean()
    frame["below_ma20"] = (frame["close"] < ma20).shift(1)
    frame["ma20_gap"] = ((frame["close"] / ma20 - 1) * 100).shift(1)
    return frame[["date", "kospi_prev_ret", "kospi_prev_intraday", "below_ma20", "ma20_gap"]]


def report(merged: pd.DataFrame, column: str, label: str, strategy: str) -> None:
    print(f"\n[{label}] — {strategy} 전략 기준")
    valid = merged.dropna(subset=[column, strategy])
    if valid[column].dtype == bool:
        groups = [(f"{column}=True", valid[valid[column]]),
                  (f"{column}=False", valid[~valid[column]])]
    else:
        med = valid[column].median()
        groups = [(f"{column} <= {med:.2f} (약세)", valid[valid[column] <= med]),
                  (f"{column} >  {med:.2f} (강세)", valid[valid[column] > med])]

    for name, sub in groups:
        if sub.empty:
            continue
        net = sub[strategy] - COST
        print(f"  {name:<28} 일평균 {sub[strategy].mean():+7.3f}%  "
              f"비용차감 {net.mean():+7.3f}%  승률 {(net > 0).mean() * 100:5.1f}%  "
              f"일수 {len(sub):>3}")


def main() -> int:
    client = NaverClient(workers=8)
    universe = fetch_universe()
    universe = universe[
        (universe["close"] >= 2000) & (universe["marcap"] >= 3e11)
        & (universe["amount"] >= 3e9)
    ].sort_values("amount", ascending=False).head(400)
    histories = client.bulk(universe["code"].tolist(), "price_history", days=60)

    results = run(build_panel(histories))
    merged = results.merge(kospi_frame(client), on="date", how="left")

    print(f"표본 {len(merged)}거래일")
    print(f"전체: reversal {merged['reversal'].mean():+.3f}%  "
          f"universe {merged['universe'].mean():+.3f}%")

    for column, label in [
        ("below_ma20", "코스피가 20일선 아래인가"),
        ("kospi_prev_ret", "코스피 전일 등락률"),
        ("kospi_prev_intraday", "코스피 전일 일중(시가→종가) 흐름"),
        ("ma20_gap", "코스피 20일선 이격도"),
    ]:
        for strategy in ("reversal", "universe"):
            report(merged, column, label, strategy)

    # 전일 시장이 장중에 밀린 날을 건너뛰면?
    print("\n\n[규칙 검증] 코스피 전일 일중 흐름이 마이너스면 당일 매매 쉼")
    ok = merged.dropna(subset=["kospi_prev_intraday"])
    traded = ok[ok["kospi_prev_intraday"] > 0]
    skipped = ok[ok["kospi_prev_intraday"] <= 0]
    for name, sub in (("매매한 날", traded), ("쉰 날", skipped)):
        if sub.empty:
            continue
        net = sub["reversal"] - COST
        cumulative = (1 + net / 100).prod() - 1
        print(f"  {name:<8} {len(sub):>3}일  일평균 {net.mean():+7.3f}%  "
              f"누적 {cumulative * 100:+7.1f}%  승률 {(net > 0).mean() * 100:5.1f}%")
    all_net = ok["reversal"] - COST
    print(f"  {'전일 매매':<8} {len(ok):>3}일  일평균 {all_net.mean():+7.3f}%  "
          f"누적 {((1 + all_net / 100).prod() - 1) * 100:+7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
