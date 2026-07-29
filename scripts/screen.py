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
    "ret_5d", "ret_20d", "relative_strength", "ma20_disparity", "ma5_disparity",
    "oc_mean", "oc_winrate", "oc_sharpe",
    "rsi14", "macd_hist", "bb_percent_b", "volume_surge",
    "target_upside", "recomm_score", "pbr", "per", "forward_per", "atr_pct",
    # 낙폭 반등 전략용
    "drawdown_52w", "lower_shadow", "down_streak", "capitulation",
    "oc_after_down", "oc_after_down_winrate", "down_day_count",
    "roe", "eps", "dividend_yield",
)

BOOLEAN_FACTOR_COLUMNS = ("dual_buying", "is_profitable", "has_coverage")


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """결측 컬럼 생성 + 숫자형 강제.

    네이버가 값을 안 주면 None이 섞여 컬럼이 object dtype이 되고,
    그 상태로 산술 연산을 하면 TypeError가 난다. 여기서 한 번에 정리한다.
    """
    for column in REQUIRED_FACTOR_COLUMNS:
        if column not in frame.columns:
            frame[column] = float("nan")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOLEAN_FACTOR_COLUMNS:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def build_block_scores(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """각 팩터를 z-score로 표준화한 뒤 5개 블록으로 합산한다.

    전략이 역추세(낙폭 반등)이므로 부호 방향이 추세 전략과 반대인 팩터가 많다.
    수익률·이격도·RSI는 낮을수록 점수가 올라간다.
    """
    cfg_score = cfg["scoring"]
    z = F.zscore

    # --- 낙폭: 얼마나 눌렸는가 -----------------------------------------
    # 단조 증가로 두지 않고 sweet spot에서 멀수록 감점하는 이유는,
    # -70% 같은 극단 낙폭은 되돌림이 아니라 훼손일 확률이 높기 때문이다.
    drawdown_fit = -(
        frame["drawdown_52w"] - cfg_score["drawdown_sweet_spot"]
    ).abs()
    rsi_fit = -(frame["rsi14"] - cfg_score["rsi_sweet_spot"]).abs()
    bb_fit = -(frame["bb_percent_b"] - cfg_score["bb_sweet_spot"]).abs()

    frame["block_drawdown"] = (
        0.30 * z(drawdown_fit)
        + 0.28 * z(rsi_fit)
        + 0.22 * z(bb_fit)
        - 0.20 * z(frame["ma20_disparity"])   # 20일선 아래일수록 가점
    )

    # --- 반등: 팔 사람이 다 팔았다는 흔적이 있는가 ----------------------
    # oc_after_down이 이 블록의 핵심이다. 전일 하락한 다음 날의 시가→종가
    # 수익률로, 실제 매매 조건과 정확히 같은 상황의 과거 성적이다.
    # 표본이 적으면 우연일 수 있어 신뢰 구간을 못 채우면 중립 처리한다.
    reliable = frame["down_day_count"] >= cfg_score["min_down_day_samples"]
    oc_down = frame["oc_after_down"].where(reliable)
    oc_down_wr = frame["oc_after_down_winrate"].where(reliable)
    capped_capitulation = frame["capitulation"].clip(
        upper=cfg_score["capitulation_cap"]
    )

    frame["block_reversal"] = (
        0.34 * z(oc_down)
        + 0.22 * z(oc_down_wr)
        + 0.18 * z(frame["lower_shadow"])
        + 0.14 * z(capped_capitulation)
        + 0.12 * z(frame["oc_mean"])
    )

    # --- 수급: 낙폭 구간에서도 기관·외국인이 받고 있는가 -----------------
    # 이 전략에서 수급은 단순 선호가 아니라 검증 장치다. 빠지는 종목을
    # 외국인·기관이 사고 있다면 시장이 저점 매집으로 보고 있다는 뜻이다.
    frame["block_flow"] = (
        0.40 * z(frame["foreign_net_5d_pct"])
        + 0.30 * z(frame["organ_net_5d_pct"])
        + 0.15 * z(frame["foreign_ratio_change"])
        + 0.15 * z(frame["foreign_buy_days"] + frame["organ_buy_days"])
    )
    frame.loc[frame["dual_buying"], "block_flow"] += 0.30

    # --- 밸류: 싼가 -----------------------------------------------------
    # 추정PER이 있으면 그쪽을 쓰고, 없는 종목만 실적 PER로 메운다.
    effective_per = frame["forward_per"].fillna(frame["per"])
    frame["block_value"] = (
        0.40 * z(frame["target_upside"])
        - 0.25 * z(frame["pbr"])
        - 0.20 * z(effective_per)
        + 0.15 * z(frame["dividend_yield"])
    )

    # --- 퀄리티: 싼 데 이유가 없는가 ------------------------------------
    frame["block_quality"] = (
        0.45 * z(frame["roe"])
        + 0.30 * z(frame["recomm_score"])
        + 0.25 * z(frame["oc_sharpe"])   # 일중 움직임의 일관성
    )

    return frame


def apply_factor_gates(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """히스토리를 받아봐야 알 수 있는 조건들 (변동성·과매수)."""
    rules = cfg["universe"]
    gates = cfg["quality_gates"]
    counts = {}

    # --- 낙폭 조건: 이 전략의 대상인지 ---------------------------------
    # 결측을 통과시키지 않는다. 낙폭 여부를 확인 못 한 종목은 대상이 아니다.
    frame = frame[frame["ret_20d"].between(rules["min_ret_20d"], rules["max_ret_20d"])]
    counts[f"20일 수익률 {rules['min_ret_20d']}~{rules['max_ret_20d']}%"] = len(frame)

    frame = frame[frame["drawdown_52w"] <= rules["max_drawdown_52w"]]
    counts[f"52주 고점 대비 {rules['max_drawdown_52w']}% 이상 하락"] = len(frame)

    low, high = rules["rsi_range"]
    frame = frame[frame["rsi14"].between(low, high)]
    counts[f"RSI {low}~{high} (과매도 구간)"] = len(frame)

    # --- 퀄리티 게이트: falling knife 방어 -----------------------------
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

    # 이 전략의 매매 조건과 같은 상황에서 과거에 어떻게 됐는지.
    # 표본이 모자란 종목은 판단 불가로 보고 통과시킨다 — 없는 근거로
    # 탈락시키면 신규 상장·거래 재개 종목이 영구히 배제된다.
    threshold = gates.get("min_oc_after_down")
    if threshold is not None:
        enough = frame["down_day_count"] >= gates["min_oc_after_down_samples"]
        frame = frame[~enough | (frame["oc_after_down"] >= threshold)]
        counts[f"하락 다음날 일중수익률 {threshold}% 이상"] = len(frame)

    # --- 변동성: 안전 상하한 먼저, 그다음 풀 내 상대 위치 ---------------
    atr = frame["atr_pct"]
    frame = frame[
        atr.between(rules["atr_hard_min"], rules["atr_hard_max"]) | atr.isna()
    ]
    counts[f"ATR {rules['atr_hard_min']}~{rules['atr_hard_max']}% (안전 상하한)"] = len(frame)

    # 상대 변동성 밴드는 모집단이 충분할 때만 건다. 앞선 게이트를 통과한
    # 종목이 얼마 안 되는 날에 양 끝을 25%나 더 잘라내면, 걸러낸 이유가
    # "위험해서"가 아니라 "그날 표본이 작아서"가 되어버린다.
    atr = frame["atr_pct"].dropna()
    if len(atr) >= rules["atr_band_min_pool"]:
        low_pct, high_pct = rules["atr_pct_band"]
        lo, hi = atr.quantile(low_pct / 100), atr.quantile(high_pct / 100)
        frame = frame[frame["atr_pct"].between(lo, hi) | frame["atr_pct"].isna()]
        counts[f"ATR 풀 내 {low_pct}~{high_pct}분위 ({lo:.1f}~{hi:.1f}%)"] = len(frame)
    else:
        counts[f"ATR 상대밴드 (풀 {len(atr)}종목 < {rules['atr_band_min_pool']}, 미적용)"] = len(frame)

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
                    block: round(float(row[f"block_{block}"]), 2)
                    for block in ("drawdown", "reversal", "flow", "value", "quality")
                },
                "factors": {
                    key: (None if pd.isna(row.get(key)) else row.get(key))
                    for key in (
                        "rsi14", "macd_hist", "bb_percent_b", "atr_pct",
                        "ma20_disparity", "ma5_disparity", "volume_surge",
                        "ret_5d", "ret_20d", "relative_strength",
                        "drawdown_52w", "lower_shadow", "down_streak",
                        "capitulation", "oc_after_down", "oc_after_down_winrate",
                        "down_day_count",
                        "oc_mean", "oc_winrate", "oc_sharpe", "avg_range_pct",
                        "foreign_net_5d_pct", "organ_net_5d_pct",
                        "foreign_buy_days", "organ_buy_days", "foreign_ratio",
                        "foreign_ratio_change", "per", "forward_per", "pbr",
                        "roe", "eps", "dividend_yield",
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
        "strategy": "낙폭 과다 저평가 우량주 일중 반등",
        "pool_health": pool_health(len(frame), len(picks)),
        "config_snapshot": {
            "min_amount_krw": cfg["universe"]["min_amount_krw"],
            "min_marcap_krw": cfg["universe"]["min_marcap_krw"],
            "atr_pct_band": cfg["universe"]["atr_pct_band"],
            "atr_hard_max": cfg["universe"]["atr_hard_max"],
            "rsi_range": cfg["universe"]["rsi_range"],
            "max_ret_20d": cfg["universe"]["max_ret_20d"],
            "max_drawdown_52w": cfg["universe"]["max_drawdown_52w"],
            "quality_gates": cfg["quality_gates"],
            "max_per_sector": cfg["risk"]["max_per_sector"],
        },
    }


def pool_health(scored: int, picked: int) -> dict:
    """살아남은 후보가 몇 개였는지, 그래서 이 선정이 얼마나 선별적인지.

    이 전략은 모든 게이트를 통과하는 종목이 시장 국면에 따라 크게 줄어든다.
    후보가 12개인 날의 '상위 10종목'과 120개인 날의 '상위 10종목'은 의미가
    전혀 다른데, 화면에는 똑같이 10개가 보인다. 그 차이를 숫자로 남긴다.
    """
    ratio = scored / picked if picked else 0

    if ratio >= 5:
        level, message = "good", None
    elif ratio >= 2.5:
        level, message = "fair", (
            f"게이트를 통과한 종목이 {scored}개로 넉넉하지 않습니다. "
            "하위권 후보는 상위권과 근거의 질이 상당히 다를 수 있습니다."
        )
    else:
        level, message = "poor", (
            f"게이트를 통과한 종목이 {scored}개뿐이라 사실상 통과 종목 대부분을 "
            "그대로 싣고 있습니다. 순위 간 변별력이 낮고, 이 전략에 맞는 "
            "기회가 지금 시장에 드물다는 신호로 읽는 편이 안전합니다."
        )

    return {
        "scored": scored,
        "picked": picked,
        "ratio": round(ratio, 1),
        "level": level,
        "message": message,
    }


def explain(row) -> list[str]:
    """왜 이 종목이 올라왔는지 사람이 읽는 문장으로.

    점수만 보여주면 룰을 검증할 수 없다. 어떤 팩터가 기여했는지 드러내야
    사용자가 "이 근거는 납득이 안 된다"고 판단하고 걸러낼 수 있다.
    낙폭 → 반등 근거 → 수급 검증 → 밸류/퀄리티 순으로 읽히게 배열한다.
    """
    lines = []

    def has(key):
        value = row.get(key)
        return value is not None and not pd.isna(value)

    # 얼마나 빠졌나
    if has("drawdown_52w"):
        lines.append(f"52주 고점 대비 {row['drawdown_52w']:.1f}%")
    if has("ret_20d"):
        lines.append(f"20일 수익률 {row['ret_20d']:+.1f}% · 5일 {row.get('ret_5d', float('nan')):+.1f}%")
    if has("rsi14"):
        lines.append(f"RSI {row['rsi14']:.0f} (과매도 구간)")
    if has("bb_percent_b"):
        lines.append(f"볼린저 밴드 내 위치 {row['bb_percent_b']:.2f} (0=하단)")

    # 반등 근거
    if has("oc_after_down") and has("down_day_count"):
        lines.append(
            f"과거 하락 다음날 시가→종가 평균 {row['oc_after_down']:+.2f}% "
            f"(승률 {row.get('oc_after_down_winrate', float('nan')):.0f}%, 표본 {int(row['down_day_count'])}일)"
        )
    if has("lower_shadow") and row["lower_shadow"] > 0.5:
        lines.append(f"전일 저가 대비 종가 회복률 {row['lower_shadow']:.0%} (장중 매수세 유입)")
    if has("capitulation") and row["capitulation"] > 1.5:
        lines.append(f"하락일 거래량 평소의 {row['capitulation']:.1f}배 (항복 매도 가능성)")
    if has("down_streak") and row["down_streak"] >= 2:
        lines.append(f"{int(row['down_streak'])}일 연속 하락 마감")

    # 수급 검증
    if has("foreign_net_5d_pct") and row["foreign_net_5d_pct"] > 0:
        lines.append(
            f"낙폭에도 외국인 5일 순매수 (평균거래량 대비 {row['foreign_net_5d_pct']:+.1f}%, "
            f"{int(row.get('foreign_buy_days', 0))}/5일 매수 우위)"
        )
    if has("organ_net_5d_pct") and row["organ_net_5d_pct"] > 0:
        lines.append(f"기관 5일 순매수 {row['organ_net_5d_pct']:+.1f}%")
    if row.get("dual_buying"):
        lines.append("외국인·기관 동반 순매수 — 저점 매집 신호")

    # 저평가·퀄리티
    if has("target_upside"):
        lines.append(f"컨센서스 목표주가 대비 {row['target_upside']:+.0f}% 여력")
    if has("roe"):
        lines.append(f"ROE {row['roe']:.1f}%")
    if has("pbr"):
        lines.append(f"PBR {row['pbr']:.2f}배")
    if has("forward_per") or has("per"):
        per = row["forward_per"] if has("forward_per") else row["per"]
        label = "추정PER" if has("forward_per") else "PER"
        lines.append(f"{label} {per:.1f}배")

    if has("atr_pct"):
        lines.append(f"ATR {row['atr_pct']:.1f}% (기대 일중 변동폭)")

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
