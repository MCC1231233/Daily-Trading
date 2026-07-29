"""일중 매매 후보 스크리닝 (9시 시가 매수 → 15시 종가 매도 가정).

실행 순서
  1. 코스피/코스닥 전 종목에서 유동성·시총·가격 하드 필터
  2. 거래대금 상위 candidate_pool개만 상세 데이터 수집 (API 호출량 통제)
  3. 팩터 5블록 계산 → z-score → 국면별 가중합
  4. 업종 분산 제약을 걸고 상위 N개 선정
  5. 직전 리포트 후보들의 실제 시가→종가 수익률을 채점해서 누적 성과에 반영

출력은 docs/data/YYYY-MM-DD.json 하나. HTML은 render.py가 이 JSON을 읽는다.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

import factors as F
from datasource import NaverClient, fetch_macro, fetch_universe

KST = ZoneInfo("Asia/Seoul")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "data"

# 장 마감(15:30) + 데이터 반영 여유. 이 시각 이전이면 당일 봉은 미완성으로 본다.
SESSION_COMPLETE_HOUR = 16


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_trading_day(day: date) -> bool:
    """한국 증시 개장일 여부.

    주말 + 법정공휴일(대체공휴일·음력 명절 포함) + 연말 휴장일 12/31을 제외한다.
    임시공휴일이나 특별 휴장은 라이브러리 갱신에 의존하므로,
    잘못 판단해도 리포트가 하루 더 생기는 정도로만 영향이 남게 설계했다.
    """
    if day.weekday() >= 5:
        return False
    if (day.month, day.day) == (12, 31):
        return False
    try:
        from holidayskr import is_holiday

        return not is_holiday(day.strftime("%Y-%m-%d"))
    except Exception:
        return True  # 판단 불가면 개장일로 보고 진행


def next_trading_day(day: date) -> date:
    candidate = day + timedelta(days=1)
    for _ in range(15):
        if is_trading_day(candidate):
            return candidate
        candidate += timedelta(days=1)
    return candidate


def trim_incomplete_session(frame: pd.DataFrame, today: str, complete: bool):
    """장중에 돌리면 당일 봉이 미완성이라 시가→종가 통계가 오염된다.

    마감 전이면 오늘 날짜 행을 잘라내고 전일까지의 정보만 쓴다.
    이렇게 해야 08:20 정규 실행이든 낮에 수동 실행이든 같은 결과가 나온다.
    """
    if frame.empty or complete:
        return frame
    return frame[frame["date"] < today].reset_index(drop=True)


# --- 1단계: 유니버스 필터 ------------------------------------------------


def prefilter(universe: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    rules = cfg["universe"]
    counts = {"전체": len(universe)}

    filtered = universe.copy()
    if "dept" in filtered.columns:
        pattern = "|".join(rules["exclude_dept_keywords"])
        flagged = filtered["dept"].fillna("").str.contains(pattern, na=False)
        filtered = filtered[~flagged]
        counts["관리·경고종목 제외"] = len(filtered)

    filtered = filtered[filtered["close"] >= rules["min_price"]]
    counts[f"주가 {rules['min_price']:,}원 이상"] = len(filtered)

    filtered = filtered[filtered["marcap"] >= rules["min_marcap_krw"]]
    counts[f"시총 {rules['min_marcap_krw'] / 1e8:,.0f}억 이상"] = len(filtered)

    filtered = filtered[filtered["amount"] >= rules["min_amount_krw"]]
    counts[f"거래대금 {rules['min_amount_krw'] / 1e8:,.0f}억 이상"] = len(filtered)

    if "change_pct" in filtered.columns:
        filtered = filtered[
            filtered["change_pct"].between(
                rules["min_prev_change_pct"], rules["max_prev_change_pct"]
            )
        ]
        counts["전일 등락률 ±15% 이내"] = len(filtered)

    filtered = filtered.sort_values("amount", ascending=False)
    filtered = filtered.head(rules["candidate_pool"]).reset_index(drop=True)
    counts[f"거래대금 상위 {rules['candidate_pool']}"] = len(filtered)

    return filtered, counts


# --- 2단계: 시장 국면 ---------------------------------------------------


def classify_regime(macro: dict) -> dict:
    """국내 두 지수의 20일선 위치로 국면을 나눈다.

    지수가 20일선 위에 있을 때와 아래에 있을 때는 같은 모멘텀 신호라도
    이후 하루 수익률의 분포가 다르다. 그래서 가중치를 바꾼다.
    """
    domestic = macro.get("domestic", {})
    above = [
        info.get("above_ma20")
        for info in domestic.values()
        if info.get("above_ma20") is not None
    ]

    if not above:
        regime, label = "neutral", "판단 불가 (지수 데이터 결측)"
    elif all(above):
        regime, label = "risk_on", "위험선호 — 코스피·코스닥 모두 20일선 위"
    elif not any(above):
        regime, label = "risk_off", "위험회피 — 코스피·코스닥 모두 20일선 아래"
    else:
        regime, label = "neutral", "혼조 — 두 지수 방향 불일치"

    overseas = macro.get("global", {})
    notes = []
    for symbol in ("US500", "IXIC"):
        info = overseas.get(symbol)
        if info:
            direction = "상승" if info["change_pct"] > 0 else "하락"
            notes.append(f"{info['label']} {info['change_pct']:+.2f}% {direction}")
    fx = overseas.get("USD/KRW")
    if fx:
        notes.append(f"원/달러 {fx['close']:,.1f} ({fx['change_pct']:+.2f}%)")

    return {"regime": regime, "label": label, "notes": notes}


# --- 3단계: 팩터 → 블록 점수 --------------------------------------------


# 블록 계산에 쓰이는 컬럼. 한 종목도 값을 못 받으면 컬럼 자체가 없어서
# KeyError가 나므로, 계산 전에 결측 컬럼을 만들어 둔다.
REQUIRED_FACTOR_COLUMNS = (
    "foreign_net_5d_pct", "organ_net_5d_pct", "foreign_ratio_change",
    "foreign_buy_days", "organ_buy_days", "dual_buying",
    "ret_5d", "ret_20d", "relative_strength", "ma20_disparity",
    "oc_mean", "oc_winrate", "oc_sharpe", "gap_follow",
    "rsi14", "macd_hist", "bb_percent_b", "volume_surge",
    "target_upside", "recomm_score", "pbr", "per", "forward_per", "atr_pct",
)


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """결측 컬럼 생성 + 숫자형 강제.

    네이버가 값을 안 주면 None이 섞여 컬럼이 object dtype이 되고,
    그 상태로 산술 연산을 하면 TypeError가 난다. 여기서 한 번에 정리한다.
    """
    for column in REQUIRED_FACTOR_COLUMNS:
        if column not in frame.columns:
            frame[column] = False if column == "dual_buying" else float("nan")
        elif column != "dual_buying":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["dual_buying"] = frame["dual_buying"].fillna(False).astype(bool)
    return frame


def build_block_scores(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """각 팩터를 z-score로 표준화한 뒤 5개 블록으로 합산한다."""
    tech_cfg = cfg["technical"]
    z = F.zscore

    # 수급: 외국인·기관이 실제로 얼마나 사갔는지
    frame["block_flow"] = (
        0.40 * z(frame["foreign_net_5d_pct"])
        + 0.30 * z(frame["organ_net_5d_pct"])
        + 0.15 * z(frame["foreign_ratio_change"])
        + 0.15 * z(frame["foreign_buy_days"] + frame["organ_buy_days"])
    )
    frame.loc[frame["dual_buying"] == True, "block_flow"] += 0.25  # noqa: E712

    # 모멘텀: 최근 추세와 시장 대비 상대강도
    frame["block_momentum"] = (
        0.30 * z(frame["ret_5d"])
        + 0.25 * z(frame["ret_20d"])
        + 0.25 * z(frame["relative_strength"])
        + 0.20 * z(frame["ma20_disparity"])
    )

    # 일중 성향: 이 전략의 보유 구간(9시~15시)에서 실제로 어떻게 움직였는가
    frame["block_intraday"] = (
        0.35 * z(frame["oc_mean"])
        + 0.25 * z(frame["oc_winrate"])
        + 0.25 * z(frame["oc_sharpe"])
        + 0.15 * z(frame["gap_follow"])
    )

    # 기술적: 중심값에서 벗어난 정도를 감점으로 환산한다.
    # 단순히 "RSI 높을수록 좋다"로 두면 과매수 종목만 뽑히기 때문이다.
    rsi_fit = -(frame["rsi14"] - tech_cfg["rsi_sweet_spot"]).abs()
    bb_fit = -(frame["bb_percent_b"] - tech_cfg["bb_sweet_spot"]).abs()
    capped_surge = frame["volume_surge"].clip(upper=tech_cfg["volume_surge_cap"])
    frame["block_technical"] = (
        0.30 * z(rsi_fit)
        + 0.30 * z(frame["macd_hist"])
        + 0.20 * z(bb_fit)
        + 0.20 * z(capped_surge)
    )

    # 펀더멘털: 애널리스트 컨센서스와 밸류에이션.
    # PER/PBR은 낮을수록 좋으므로 z-score를 빼는 방향으로 넣는다.
    # 추정PER이 있으면 그쪽을 쓰고, 없는 종목만 실적 PER로 메운다.
    effective_per = frame["forward_per"].fillna(frame["per"])
    frame["block_quality"] = (
        0.40 * z(frame["target_upside"])
        + 0.25 * z(frame["recomm_score"])
        - 0.20 * z(frame["pbr"])
        - 0.15 * z(effective_per)
    )

    return frame


def apply_factor_gates(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """히스토리를 받아봐야 알 수 있는 조건들 (변동성·과매수)."""
    rules = cfg["universe"]
    counts = {}

    # 안전 상·하한 먼저 (국면과 무관하게 적용)
    atr = frame["atr_pct"]
    frame = frame[
        atr.between(rules["atr_hard_min"], rules["atr_hard_max"]) | atr.isna()
    ]
    counts[f"ATR {rules['atr_hard_min']}~{rules['atr_hard_max']}% (안전 상하한)"] = len(frame)

    # 그다음 오늘 풀 기준 상대 위치로 양 끝을 잘라낸다
    atr = frame["atr_pct"].dropna()
    if len(atr) >= 20:
        low_pct, high_pct = rules["atr_pct_band"]
        low, high = atr.quantile(low_pct / 100), atr.quantile(high_pct / 100)
        frame = frame[frame["atr_pct"].between(low, high) | frame["atr_pct"].isna()]
        counts[f"ATR 풀 내 {low_pct}~{high_pct}분위 ({low:.1f}~{high:.1f}%)"] = len(frame)

    frame = frame[(frame["rsi14"] <= rules["max_rsi"]) | frame["rsi14"].isna()]
    counts[f"RSI {rules['max_rsi']} 이하"] = len(frame)

    return frame.reset_index(drop=True), counts


def select_with_diversification(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """점수 순으로 뽑되 한 업종이 max_per_sector를 넘지 않게 한다.

    같은 업종 10종목은 사실상 한 종목에 10배 베팅한 것과 같다.
    업종 정보가 없는 종목은 제약 없이 통과시킨다.
    """
    limit = cfg["risk"]["max_per_sector"]
    target = cfg["universe"]["picks"]

    selected, sector_count = [], {}
    for _, row in frame.sort_values("score", ascending=False).iterrows():
        sector = row.get("industry_code") or ""
        if sector and sector_count.get(sector, 0) >= limit:
            continue
        selected.append(row)
        if sector:
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) >= target:
            break

    return pd.DataFrame(selected)


# --- 5단계: 직전 리포트 채점 --------------------------------------------


def score_pending_reports(client: NaverClient, today: str) -> tuple[list[dict], list[str]]:
    """아직 채점 안 된 리포트를 모두 채점한다.

    한 건만 처리하면, 휴장으로 채점 불가한 리포트가 큐 앞을 막아 그 뒤 것들이
    영원히 안 채점된다. 그래서 전부 훑고, 매매일이 3일 넘게 지났는데도
    시세가 없는 건은 휴장으로 보고 skipped 처리해 큐에서 뺀다.
    """
    index_path = DATA_DIR / "index.json"
    if not index_path.exists():
        return [], []

    entries = json.loads(index_path.read_text(encoding="utf-8")).get("reports", [])
    scored, skipped = [], []

    for entry in entries:
        if entry.get("scored") or entry.get("skipped"):
            continue
        if entry["date"] == today:
            continue

        result = _score_one(client, entry["date"])
        if result:
            scored.append(result)
            continue

        trade_date = entry.get("trade_date") or entry["date"]
        age = (datetime.strptime(today, "%Y-%m-%d")
               - datetime.strptime(trade_date, "%Y-%m-%d")).days
        if age >= 3:
            skipped.append(entry["date"])

    return scored, skipped


def _score_one(client: NaverClient, report_date: str) -> dict | None:
    """한 리포트의 후보들이 실제로 어떻게 됐는지 확인한다.

    시가 매수 / 종가 매도를 가정하고 (종가/시가 - 1)을 실현 수익률로 본다.
    수수료·세금·슬리피지는 반영하지 않은 총수익률이다.
    """
    report_path = DATA_DIR / f"{report_date}.json"
    if not report_path.exists():
        return None

    report = json.loads(report_path.read_text(encoding="utf-8"))
    trade_date = report.get("trade_date") or report["date"]

    results = []
    for pick in report.get("picks", []):
        history = client.price_history(pick["code"], days=10)
        if history.empty:
            continue
        day = history[history["date"] == trade_date]
        if day.empty:
            continue
        row = day.iloc[0]
        if not row["open"]:
            continue
        results.append(
            {
                "code": pick["code"],
                "name": pick["name"],
                "open": row["open"],
                "close": row["close"],
                "high": row["high"],
                "low": row["low"],
                "return_pct": round((row["close"] / row["open"] - 1) * 100, 2),
                "max_gain_pct": round((row["high"] / row["open"] - 1) * 100, 2),
                "max_loss_pct": round((row["low"] / row["open"] - 1) * 100, 2),
            }
        )

    if not results:
        return None

    returns = [r["return_pct"] for r in results]
    return {
        "date": report_date,
        "trade_date": trade_date,
        "results": results,
        "avg_return_pct": round(sum(returns) / len(returns), 2),
        "win_count": sum(1 for r in returns if r > 0),
        "total_count": len(returns),
        "best": max(results, key=lambda r: r["return_pct"]),
        "worst": min(results, key=lambda r: r["return_pct"]),
    }


# --- 메인 --------------------------------------------------------------


def main() -> int:
    cfg = load_config()
    now = datetime.now(KST)
    today = now.strftime("%Y-%m-%d")
    session_complete = now.hour >= SESSION_COMPLETE_HOUR

    # 장 시작 전에 돌면 오늘이 매매일, 마감 후에 돌면 다음 개장일이 매매일이다.
    trade_day = next_trading_day(now.date()) if session_complete else now.date()

    print(f"[1/6] 실행 시각 {now:%Y-%m-%d %H:%M} KST "
          f"(당일 세션 {'완료' if session_complete else '미완료 → 전일 종가 기준'})")
    print(f"      대상 매매일: {trade_day}")

    if not is_trading_day(trade_day):
        print(f"      {trade_day}은(는) 휴장일 — 리포트를 발행하지 않고 종료")
        return 0

    client = NaverClient(workers=8)

    print("[2/6] 매크로 수집")
    macro = fetch_macro(client)
    regime = classify_regime(macro)
    print(f"      국면: {regime['label']}")

    print("[3/6] 유니버스 필터")
    universe = fetch_universe()
    pool, filter_counts = prefilter(universe, cfg)
    print(f"      {filter_counts['전체']}종목 → 후보 {len(pool)}종목")

    codes = pool["code"].tolist()
    print(f"[4/6] 상세 데이터 수집 ({len(codes)}종목 x 3개 엔드포인트)")
    histories = client.bulk(codes, "price_history", days=60)
    trends = client.bulk(codes, "investor_trend", days=20)
    snapshots = client.bulk(codes, "snapshot")

    # 지수 20일 수익률 — 상대강도 계산 기준
    index_return_20d = None
    index_rows = client._get("index/KOSPI/price?pageSize=30&page=1")
    if index_rows:
        closes = pd.Series(
            [float(str(r.get("closePrice", "0")).replace(",", "")) for r in index_rows]
        ).iloc[::-1].reset_index(drop=True)
        if len(closes) > 21:
            index_return_20d = float((closes.iloc[-1] / closes.iloc[-21] - 1) * 100)

    print("[5/6] 팩터 계산 및 스코어링")
    records = []
    for _, row in pool.iterrows():
        code = row["code"]
        history = trim_incomplete_session(
            histories.get(code, pd.DataFrame()), today, session_complete
        )
        trend = trim_incomplete_session(
            trends.get(code, pd.DataFrame()), today, session_complete
        )
        snapshot = snapshots.get(code) or {}

        computed = F.compute_all(history, trend, snapshot, index_return_20d)
        if not computed:
            continue

        records.append(
            {
                "code": code,
                "name": row["name"],
                "market": row["market"],
                "marcap": float(row["marcap"]),
                "amount": float(row["amount"]),
                "industry_code": snapshot.get("industry_code"),
                **computed,
            }
        )

    if not records:
        print("      후보 없음 — 데이터 수집 실패로 판단하고 중단")
        return 1

    frame = ensure_columns(pd.DataFrame(records))
    frame, gate_counts = apply_factor_gates(frame, cfg)
    filter_counts.update(gate_counts)
    frame = build_block_scores(frame, cfg)

    weights = cfg["weights"][regime["regime"]]
    frame["score"] = sum(
        weights[block] * frame[f"block_{block}"] for block in weights
    )
    frame["rank_pct"] = frame["score"].rank(pct=True) * 100

    picks = select_with_diversification(frame, cfg)
    print(f"      {len(frame)}종목 스코어링 → 상위 {len(picks)}종목 선정")

    print("[6/6] 미채점 리포트 정산")
    scored_list, skipped = score_pending_reports(client, today)
    for item in scored_list:
        print(f"      {item['date']} 후보 평균 {item['avg_return_pct']:+.2f}% "
              f"({item['win_count']}/{item['total_count']} 상승)")
    for skip in skipped:
        print(f"      {skip} 채점 불가 (휴장 추정) — 통계에서 제외")
    if not scored_list and not skipped:
        print("      정산 대상 없음")

    payload = build_payload(
        today, trade_day, now, picks, frame, macro, regime, filter_counts,
        weights, cfg, client, scored_list[0] if scored_list else None,
        session_complete,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / f"{today}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    update_index(today, payload, scored_list, skipped)
    print(f"\n완료: docs/data/{today}.json ({len(picks)}종목)")
    return 0


def build_payload(
    today, trade_day, now, picks, frame, macro, regime, filter_counts,
    weights, cfg, client, scored, session_complete,
) -> dict:
    """HTML이 그대로 읽어 쓰는 리포트 JSON."""
    asof = frame["last_date"].mode()
    asof_date = asof.iloc[0] if not asof.empty else today
    trade_date = trade_day.strftime("%Y-%m-%d")

    pick_list = []
    for order, (_, row) in enumerate(picks.iterrows(), start=1):
        pick_list.append(
            {
                "rank": order,
                "code": row["code"],
                "name": row["name"],
                "market": row["market"],
                "industry": client.industry_name(row.get("industry_code")),
                "score": round(float(row["score"]), 3),
                "percentile": round(float(row["rank_pct"]), 1),
                "last_close": row.get("last_close"),
                "marcap_bn": round(float(row["marcap"]) / 1e8, 0),
                "amount_bn": round(float(row["amount"]) / 1e8, 0),
                "blocks": {
                    "flow": round(float(row["block_flow"]), 2),
                    "momentum": round(float(row["block_momentum"]), 2),
                    "intraday": round(float(row["block_intraday"]), 2),
                    "technical": round(float(row["block_technical"]), 2),
                    "quality": round(float(row["block_quality"]), 2),
                },
                "factors": {
                    key: (None if pd.isna(row.get(key)) else row.get(key))
                    for key in (
                        "rsi14", "macd_hist", "bb_percent_b", "atr_pct",
                        "ma20_disparity", "ma5_disparity", "volume_surge",
                        "ret_5d", "ret_20d", "relative_strength",
                        "oc_mean", "oc_winrate", "oc_sharpe", "gap_follow",
                        "avg_range_pct", "foreign_net_5d_pct", "organ_net_5d_pct",
                        "foreign_buy_days", "organ_buy_days", "foreign_ratio",
                        "foreign_ratio_change", "per", "forward_per", "pbr",
                        "dividend_yield",
                        "target_upside", "target_price", "recomm_score", "pos_52w",
                    )
                    if key in row.index
                },
                "rationale": explain(row),
            }
        )

    return {
        "date": today,
        "trade_date": trade_date,
        "asof_date": asof_date,
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S KST"),
        "session_complete": session_complete,
        "regime": regime,
        "macro": macro,
        "weights": weights,
        "universe_funnel": filter_counts,
        "scored_count": len(frame),
        "picks": pick_list,
        "previous_result": scored,
        "data_warnings": client.failures[:20],
        "config_snapshot": {
            "min_amount_krw": cfg["universe"]["min_amount_krw"],
            "min_marcap_krw": cfg["universe"]["min_marcap_krw"],
            "atr_pct_band": cfg["universe"]["atr_pct_band"],
            "atr_hard_max": cfg["universe"]["atr_hard_max"],
            "max_rsi": cfg["universe"]["max_rsi"],
            "max_per_sector": cfg["risk"]["max_per_sector"],
        },
    }


def explain(row) -> list[str]:
    """왜 이 종목이 올라왔는지 사람이 읽는 문장으로.

    점수만 보여주면 룰을 검증할 수 없다. 어떤 팩터가 기여했는지 드러내야
    사용자가 "이 근거는 납득이 안 된다"고 판단하고 걸러낼 수 있다.
    """
    lines = []

    def has(key):
        value = row.get(key)
        return value is not None and not pd.isna(value)

    if has("foreign_net_5d_pct") and row["foreign_net_5d_pct"] > 0:
        lines.append(
            f"외국인 5일 순매수 (평균거래량 대비 {row['foreign_net_5d_pct']:+.1f}%, "
            f"{int(row.get('foreign_buy_days', 0))}/5일 매수 우위)"
        )
    if has("organ_net_5d_pct") and row["organ_net_5d_pct"] > 0:
        lines.append(f"기관 5일 순매수 {row['organ_net_5d_pct']:+.1f}%")
    if row.get("dual_buying"):
        lines.append("외국인·기관 동반 순매수")

    if has("oc_mean") and has("oc_winrate"):
        lines.append(
            f"최근 20일 시가→종가 평균 {row['oc_mean']:+.2f}%, "
            f"상승 마감 {row['oc_winrate']:.0f}%"
        )
    if has("gap_follow") and row["gap_follow"] > 0:
        lines.append(f"갭 상승일에도 일중 평균 {row['gap_follow']:+.2f}%로 추세 유지")

    if has("relative_strength") and row["relative_strength"] > 0:
        lines.append(f"20일 상대강도 지수 대비 {row['relative_strength']:+.1f}%p")
    if has("ma20_disparity"):
        lines.append(f"20일선 이격도 {row['ma20_disparity']:+.1f}%")

    if has("rsi14"):
        lines.append(f"RSI {row['rsi14']:.0f}")
    if has("atr_pct"):
        lines.append(f"ATR {row['atr_pct']:.1f}% (기대 일중 변동폭)")

    if has("target_upside") and row["target_upside"] > 0:
        lines.append(f"컨센서스 목표주가 대비 {row['target_upside']:+.0f}% 여력")
    if has("per") and row["per"] and row["per"] > 0:
        lines.append(f"PER {row['per']:.1f}배")

    return lines


def update_index(
    today: str, payload: dict, scored_list: list[dict], skipped: list[str]
) -> None:
    """사이드바가 읽는 날짜 목록. 최신순 유지."""
    index_path = DATA_DIR / "index.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists()
        else {"reports": []}
    )
    reports = [r for r in index.get("reports", []) if r["date"] != today]

    # 이번 실행에서 채점한 과거 리포트에 결과를 붙인다.
    by_date = {s["date"]: s for s in scored_list}
    for report in reports:
        scored = by_date.get(report["date"])
        if scored:
            report["scored"] = True
            report["avg_return_pct"] = scored["avg_return_pct"]
            report["win_count"] = scored["win_count"]
            report["total_count"] = scored["total_count"]
            _persist_score(scored)
        elif report["date"] in skipped:
            report["skipped"] = True

    reports.insert(
        0,
        {
            "date": today,
            "trade_date": payload["trade_date"],
            "regime": payload["regime"]["regime"],
            "pick_count": len(payload["picks"]),
            "top_names": [p["name"] for p in payload["picks"][:3]],
            "scored": False,
        },
    )
    reports.sort(key=lambda r: r["date"], reverse=True)

    index["reports"] = reports
    index["updated_at"] = payload["generated_at"]
    index["performance"] = summarize_performance(reports)
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _persist_score(scored: dict) -> None:
    """채점 결과를 해당 날짜 리포트 파일에도 써넣는다."""
    path = DATA_DIR / f"{scored['date']}.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    report["realized"] = scored
    by_code = {r["code"]: r for r in scored["results"]}
    for pick in report.get("picks", []):
        if pick["code"] in by_code:
            pick["realized"] = by_code[pick["code"]]
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_performance(reports: list[dict]) -> dict:
    """채점 완료된 리포트들의 누적 통계."""
    done = [r for r in reports if r.get("scored") and r.get("avg_return_pct") is not None]
    if not done:
        return {"sessions": 0}

    returns = [r["avg_return_pct"] for r in done]
    cumulative = 1.0
    for value in reversed(returns):
        cumulative *= 1 + value / 100

    return {
        "sessions": len(done),
        "avg_daily_return_pct": round(sum(returns) / len(returns), 3),
        "cumulative_return_pct": round((cumulative - 1) * 100, 2),
        "positive_sessions": sum(1 for value in returns if value > 0),
        "session_win_rate_pct": round(
            sum(1 for value in returns if value > 0) / len(returns) * 100, 1
        ),
        "best_session_pct": round(max(returns), 2),
        "worst_session_pct": round(min(returns), 2),
        "pick_win_rate_pct": round(
            sum(r.get("win_count", 0) for r in done)
            / max(sum(r.get("total_count", 0) for r in done), 1)
            * 100,
            1,
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
