"""신호에 알파가 있는가, 아니면 그냥 시장을 따라간 것인가.

앞선 분석에서 과매도 종목의 장중 수익률은 -0.591%/일, 시장 전체는 -1.217%/일이었다.
차이 +0.63%p가 진짜 알파라면 시장 노출을 제거했을 때 양수가 남아야 한다.

세 가지를 테스트한다.
  1. 롱온리 초과수익  선정 종목 - 그날 유니버스 평균 (베타 1 헤지와 동등)
  2. 롱숏 스프레드    과매도 상위 10 롱 / 과매수 상위 10 숏
  3. 지수 헤지        선정 종목 롱 / 코스피 지수 숏

실행 가능성 주의
  한국 시장에서 개인의 공매도는 대주 물량과 업틱룰 제약이 크다. 2번은
  이론적 스프레드이고, 실무에서는 인버스 ETF나 지수선물 매도(3번)가 대안이다.
  숏 레그에도 왕복 비용이 붙는다는 점을 반영했다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasource import NaverClient, fetch_universe  # noqa: E402

COST = 0.18       # 롱 레그 왕복
SHORT_COST = 0.20 # 숏 레그 왕복 (대차수수료 포함 가정)
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
        close = h["close"]
        rows.append(pd.DataFrame({
            "code": code, "date": h["date"],
            "ret_5d": (close / close.shift(5) - 1).shift(1) * 100,
            "ret_20d": (close / close.shift(20) - 1).shift(1) * 100,
            "rsi14": rsi_series(close).shift(1),
            "ma20_disp": (close / close.rolling(20).mean() - 1).shift(1) * 100,
            "amount": (close * h["volume"]).shift(1),
            "intraday": (close / h["open"] - 1) * 100,
        }))
    panel = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = panel["intraday"].quantile([0.005, 0.995])
    return panel[panel["intraday"].between(lo, hi)]


def kospi_intraday(client: NaverClient) -> pd.DataFrame:
    rows = client._get("index/KOSPI/price?pageSize=60&page=1")
    num = lambda v: float(str(v).replace(",", ""))  # noqa: E731
    return pd.DataFrame({
        "date": [str(r["localTradedAt"])[:10] for r in rows],
        "kospi_intraday": [num(r["closePrice"]) / num(r["openPrice"]) * 100 - 100 for r in rows],
    })


def stats(series: pd.Series, label: str) -> None:
    cumulative = ((1 + series / 100).prod() - 1) * 100
    sharpe = series.mean() / series.std() * np.sqrt(252) if series.std() else 0
    tstat = series.mean() / (series.std() / np.sqrt(len(series))) if series.std() else 0
    print(f"  {label:<34} {series.mean():>7.3f}% {cumulative:>9.1f}% "
          f"{(series > 0).mean() * 100:>6.1f}% {series.std():>7.2f}% "
          f"{sharpe:>7.2f} {tstat:>6.2f}")


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

    daily = []
    for date, day in panel.groupby("date"):
        day = day.nlargest(min(200, len(day)), "amount").copy()
        if len(day) < 40:
            continue
        z = lambda s: ((s - s.mean()) / s.std()).clip(-3, 3).fillna(0)  # noqa: E731
        day["mom"] = (0.35 * z(day["ret_5d"]) + 0.30 * z(day["ret_20d"])
                      + 0.20 * z(day["ma20_disp"]) + 0.15 * z(day["rsi14"]))

        oversold = day.nsmallest(PICKS, "rsi14")["intraday"].mean()
        reversal = day.nsmallest(PICKS, "mom")["intraday"].mean()
        overbought = day.nlargest(PICKS, "rsi14")["intraday"].mean()
        momentum = day.nlargest(PICKS, "mom")["intraday"].mean()
        market = day["intraday"].mean()

        daily.append({
            "date": date, "oversold": oversold, "reversal": reversal,
            "overbought": overbought, "momentum": momentum, "market": market,
        })

    results = pd.DataFrame(daily).merge(kospi_intraday(client), on="date", how="left")
    print(f"패널 {panel['code'].nunique()}종목 x {len(results)}거래일\n")

    header = f"  {'전략':<34} {'일평균':>8} {'누적':>9} {'승률':>7} {'표준편차':>8} {'샤프':>7} {'t값':>6}"

    print("=" * 88)
    print("1. 롱온리 (비용 차감)")
    print("=" * 88)
    print(header)
    stats(results["oversold"] - COST, "과매도 10종목")
    stats(results["reversal"] - COST, "역추세 10종목")
    stats(results["market"] - COST, "유니버스 전체")

    print("\n" + "=" * 88)
    print("2. 초과수익 — 선정 종목 − 유니버스 평균 (베타 1 헤지 등가, 비용 차감)")
    print("=" * 88)
    print(header)
    stats(results["oversold"] - results["market"] - COST - SHORT_COST, "과매도 − 유니버스")
    stats(results["reversal"] - results["market"] - COST - SHORT_COST, "역추세 − 유니버스")

    print("\n" + "=" * 88)
    print("3. 롱숏 스프레드 (양 레그 비용 차감)")
    print("=" * 88)
    print(header)
    stats(results["oversold"] - results["overbought"] - COST - SHORT_COST,
          "과매도 롱 / 과매수 숏")
    stats(results["reversal"] - results["momentum"] - COST - SHORT_COST,
          "역추세 롱 / 추세추종 숏")

    print("\n" + "=" * 88)
    print("4. 지수 헤지 — 종목 롱 / 코스피 숏 (실무상 인버스 ETF·선물)")
    print("=" * 88)
    print(header)
    stats(results["oversold"] - results["kospi_intraday"] - COST - SHORT_COST,
          "과매도 롱 / 코스피 숏")
    stats(results["reversal"] - results["kospi_intraday"] - COST - SHORT_COST,
          "역추세 롱 / 코스피 숏")

    print("\n" + "=" * 88)
    print("주의: 39거래일, 폭락 국면 하나. t값 절대치 2 미만은 우연과 구분되지 않음.")
    print("한국 개인 공매도는 대주 물량·업틱룰 제약이 큼. 2·3번은 이론적 상한.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
