"""주문 실행기 — 스크리닝 리포트를 실제 주문으로 옮긴다.

    python scripts/trade.py preflight   자격증명·계좌·시세 연결 점검 (주문 없음)
    python scripts/trade.py prepare     08:30 리포트 없으면 스크리너 직접 실행
    python scripts/trade.py entry       08:50 진입 (지정가 매수)
    python scripts/trade.py sweep       09:05 미체결 취소
    python scripts/trade.py exit        15:20 청산 (시장가 → 종가 단일가 체결)
    python scripts/trade.py settle      15:40 체결 대조 + 성과 기록
    python scripts/trade.py status      계좌 현황 출력

■ 왜 상주 데몬이 아니라 일회성 명령인가
데몬은 죽어도 아무도 모른다. 진입은 됐는데 프로세스가 죽어 청산이 안 되면
오버나잇 노출이 생기고, 그건 이 전략의 정의를 위반한다. OS 스케줄러(cron /
작업 스케줄러)는 매번 새 프로세스를 띄우고 실패를 종료코드로 남긴다.

■ 왜 GitHub Actions 로 안 하는가
예약 워크플로는 러너 부하에 따라 10~30분씩 밀린다. 스크리닝(07:40)은 80분
여유가 있어 괜찮지만, 08:50 진입과 15:20 청산은 분 단위로 맞아야 한다.

■ 시간 설계 — 채점과 실행을 일치시키는 것이 핵심이다
그림자 모드는 일봉의 시가→종가로 채점한다. 실행이 그와 다르면 측정한
수익률과 실제 수익률이 매일 어긋나고, 몇 달 쌓은 표본이 무의미해진다.
  진입 08:50 지정가  -> 장시작 동시호가(08:30~09:00) 참여 -> 09:00 시가 체결
  청산 15:20 시장가  -> 종가 단일가(15:20~15:30) 참여   -> 15:30 종가 체결
청산에만 시장가를 쓰는 이유는 단일가 매매에서 시장가가 가격 결정에 최우선
참여해 사실상 종가 체결을 보장하기 때문이다. 반대로 진입에 시장가를 쓰면
장시작 동시호가에서 호가가 얇을 때 상한가까지 긁으므로 절대 쓰지 않는다.

■ 멱등성
entry 는 저널과 잔고를 모두 확인하고, 둘 중 하나라도 비어 있지 않으면 거부한다.
exit 는 정반대로 저널을 보지 않고 **실제 잔고**만 보고 판다. 저널이 깨져도
청산은 되어야 하기 때문이다. 그래서 exit 는 몇 번을 돌려도 안전하다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))

from broker.kis import ORD_LIMIT, ORD_MARKET, KisError, from_env
from screen import is_trading_day, load_config

KST = ZoneInfo("Asia/Seoul")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "data"
# 체결 기록은 저장소에 커밋하지 않는다 (.gitignore). 계좌 잔고와 주문번호가
# 담기는데 대시보드는 공개돼 있다. 공개용 요약은 settle 이 따로 만든다.
JOURNAL_DIR = ROOT / "live"

# 한국거래소 호가가격단위 (2023-01-25 개편, 유가·코스닥 동일).
# 지정가를 여기 맞추지 않으면 주문이 거부된다.
TICK_TABLE = [
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
]
TICK_ABOVE = 1_000


def tick_size(price: float) -> int:
    for ceiling, tick in TICK_TABLE:
        if price < ceiling:
            return tick
    return TICK_ABOVE


def round_to_tick(price: float, mode: str = "down") -> int:
    """호가단위로 정렬. 경계에서 단위가 바뀌므로 정렬 후 한 번 더 확인한다."""
    tick = tick_size(price)
    snapped = math.floor(price / tick) * tick if mode == "down" else math.ceil(price / tick) * tick
    # 예: 49,990원을 올림하면 50,000원이 되는데 그 구간의 단위는 50이 아니라 100이다.
    if tick_size(snapped) != tick:
        tick = tick_size(snapped)
        snapped = math.floor(snapped / tick) * tick if mode == "down" else math.ceil(snapped / tick) * tick
    return int(snapped)


def now_kst() -> datetime:
    return datetime.now(KST)


def parse_hhmm(text: str) -> dtime:
    hour, minute = text.split(":")
    return dtime(int(hour), int(minute))


def in_window(window: list[str], moment: datetime) -> bool:
    start, end = parse_hhmm(window[0]), parse_hhmm(window[1])
    return start <= moment.time() <= end


def journal_path(day: date) -> Path:
    return JOURNAL_DIR / ("%s.json" % day.isoformat())


def load_journal(day: date) -> dict:
    path = journal_path(day)
    if not path.exists():
        return {"date": day.isoformat(), "entry": None, "exit": None, "settle": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_journal(day: date, journal: dict) -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    journal_path(day).write_text(
        json.dumps(journal, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_report(day: date) -> dict | None:
    """그날 매매할 리포트. trade_date 가 오늘이 아니면 쓰지 않는다.

    07:40 스크리닝이 실패해 리포트가 없거나 어제 것만 남은 날에 어제 후보를
    사는 것이 가장 흔한 사고 유형이다. 날짜를 반드시 대조한다.
    """
    path = DATA_DIR / ("%s.json" % day.isoformat())
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("trade_date") != day.isoformat():
        return None
    return report


def cfg_trading(config: dict) -> dict:
    section = config.get("trading") or {}
    if not section.get("enabled"):
        raise SystemExit("config.yaml 의 trading.enabled 가 false 다. 실행하지 않는다.")
    return section


def log(*parts) -> None:
    print(*parts, flush=True)


# --------------------------------------------------------------------- 명령


def cmd_preflight(args, config) -> int:
    """주문을 내지 않고 배관만 점검한다. 처음 세팅했을 때 이걸 먼저 돌려라."""
    client = from_env()
    log("환경         :", client.env, "(demo = 모의투자)")
    log("계좌         :", client.cano + "-" + client.acnt_prdt_cd)

    client.token()
    log("접근토큰     : 발급/캐시 정상")

    holdings, summary = client.balance()
    log("예수금       : %d원" % int(summary["cash"]))
    log("D+2 정산     : %d원" % int(summary["settled_cash"]))
    log("보유종목     : %d개" % len(holdings))
    for row in holdings:
        log("   - %s %s %d주 (평단 %d, 현재 %d, %+.2f%%)"
            % (row["code"], row["name"], row["qty"], row["avg_price"], row["current"], row["pnl_pct"]))

    today = now_kst().date()
    report = load_report(today)
    if report:
        log("오늘 리포트  : %s (%d종목, %s)"
            % (report["trade_date"], len(report["picks"]), report.get("strategy", "")))
        sample = report["picks"][0]
        quote = client.price(sample["code"])
        log("시세 조회    : %s %s 현재 %d / 전일종가 %d / 상한 %d"
            % (sample["code"], sample["name"], quote["price"], quote["prev_close"], quote["upper_limit"]))
    else:
        log("오늘 리포트  : 없음 (휴장이거나 스크리닝 미실행)")

    log("")
    log("점검 완료 — 주문은 내지 않았다.")
    return 0


def cmd_status(args, config) -> int:
    client = from_env()
    holdings, summary = client.balance()
    total = summary["total_eval"]
    log("[%s] %s 계좌" % (now_kst().strftime("%Y-%m-%d %H:%M"), "모의" if client.env == "demo" else "실전"))
    log("  총평가 %d원 = 예수금 %d + 주식 %d"
        % (int(total), int(summary["cash"]), int(summary["stock_eval"])))
    if not holdings:
        log("  보유 없음 (플랫)")
    for row in holdings:
        log("  %s %-12s %5d주  평단 %8d  현재 %8d  %+7.2f%%  평가 %10d"
            % (row["code"], row["name"][:12], row["qty"], row["avg_price"],
               row["current"], row["pnl_pct"], int(row["eval_amount"])))
    journal = load_journal(now_kst().date())
    for phase in ("entry", "exit", "settle"):
        record = journal.get(phase)
        log("  저널 %-6s: %s" % (phase, record["at"] if record else "-"))
    return 0


def cmd_prepare(args, config) -> int:
    """오늘 리포트가 없으면 스크리너를 직접 돌린다. 진입 20분 전에 실행한다.

    ■ 왜 필요한가
    리포트는 원래 GitHub Actions 가 새벽에 만들어 커밋한다. 그런데 예약
    워크플로 지연은 통제할 수 없다 — 2026-08-31~09-07 실측에서 6/6 일이
    98~162분 밀려 장 시작 뒤에 도착했다. 그 상태로는 08:50 진입이 매일
    빈손으로 끝난다.

    그래서 매매 서버가 스스로 만들 수 있게 해둔다. 리포트가 이미 있으면
    아무것도 하지 않으므로(스크리너는 2~4분 걸린다) 매일 걸어둬도 된다.
    """
    today = now_kst().date()

    if not is_trading_day(today):
        log("휴장일 — 준비할 것이 없다.")
        return 0

    if load_report(today):
        log("오늘(%s) 리포트가 이미 있다 — 스크리너를 돌리지 않는다." % today)
        return 0

    log("오늘(%s) 리포트가 없다 — 스크리너를 직접 실행한다." % today)
    # 서브프로세스로 띄운다. screen.py 는 전역 상태를 꽤 쓰고 실행이 길어서
    # 같은 프로세스에서 import 해 돌리면 실패 시 정리가 어렵다.
    import subprocess

    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "screen.py")],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        log("스크리너 실패 (종료코드 %d) — 오늘은 진입할 수 없다." % result.returncode)
        return 1

    if not load_report(today):
        # 휴장일 판정이나 데이터 부족으로 리포트를 만들지 않고 끝난 경우다.
        log("스크리너는 성공했지만 오늘자 리포트가 생기지 않았다.")
        return 1

    log("리포트 준비 완료.")
    return 0


def cmd_entry(args, config) -> int:
    trading = cfg_trading(config)
    moment = now_kst()
    today = moment.date()

    if not is_trading_day(today):
        log("휴장일 — 진입하지 않는다.")
        return 0

    window = trading["entry"]["window"]
    if not args.force and not in_window(window, moment):
        log("진입 시간대(%s~%s)가 아니다. 현재 %s. 테스트라면 --force."
            % (window[0], window[1], moment.strftime("%H:%M")))
        return 1

    report = load_report(today)
    if not report:
        log("오늘(%s) 리포트가 없다 — 진입하지 않는다. 스크리닝 실패 여부를 확인해라." % today)
        return 1

    journal = load_journal(today)
    if journal.get("entry") and not args.force:
        log("이미 진입한 날이다 (%s). 중복 주문을 막기 위해 종료한다." % journal["entry"]["at"])
        return 0

    client = from_env()
    holdings, summary = client.balance()
    if holdings:
        # 전날 청산이 실패했다는 뜻이다. 여기서 더 사면 노출이 두 배가 된다.
        log("보유 잔고가 남아 있다 (%d종목). 전날 청산 실패로 보인다." % len(holdings))
        for row in holdings:
            log("   - %s %s %d주" % (row["code"], row["name"], row["qty"]))
        log("먼저 `python scripts/trade.py exit --force` 로 정리한 뒤 다시 실행해라.")
        return 1

    dry = trading.get("dry_run", True) or args.dry_run
    picks = report["picks"]

    # 배분 기준은 예수금이 아니라 주문가능현금이다. 첫 종목 기준으로 한 번만
    # 조회하고 종목 수로 나눈다 — 매수마다 다시 조회하면 앞선 주문이 반영돼
    # 뒤로 갈수록 배분액이 줄어드는 불균등 배분이 된다.
    probe = picks[0]
    probe_price = round_to_tick(probe["last_close"], "up")
    available = client.orderable_cash(probe["code"], probe_price)
    budget = available * float(trading.get("capital_fraction", 0.95))
    cap = trading.get("max_notional_krw")
    if cap:
        budget = min(budget, float(cap))
    per_name = budget / max(len(picks), 1)
    order_cap = float(trading.get("max_order_krw", 0) or 0)
    if order_cap:
        per_name = min(per_name, order_cap)

    log("[%s] 진입 %s / %s"
        % (moment.strftime("%H:%M:%S"), "DRY-RUN" if dry else "실주문", client.env))
    log("  주문가능현금 %d원 → 종목당 배분 %d원 (%d종목)"
        % (int(available), int(per_name), len(picks)))

    band = float(trading["entry"].get("limit_band_pct", 2.0)) / 100.0
    placed, skipped = [], []

    for pick in picks:
        code, name = pick["code"], pick["name"]
        try:
            quote = client.price(code)
        except KisError as exc:
            skipped.append({"code": code, "name": name, "reason": "시세 조회 실패: %s" % exc})
            continue

        if quote["halted"]:
            skipped.append({"code": code, "name": name, "reason": "거래정지"})
            continue

        # 기준가는 리포트의 전일종가가 아니라 시세 API 의 기준가를 쓴다.
        # 그 사이 권리락·액면분할이 있으면 리포트 값이 실제 호가와 어긋난다.
        base = quote["prev_close"] or pick["last_close"]
        limit = round_to_tick(base * (1 + band), "up")
        if quote["upper_limit"]:
            limit = min(limit, int(quote["upper_limit"]))

        qty = int(per_name // limit)
        if qty < int(trading["entry"].get("min_qty", 1)):
            skipped.append({"code": code, "name": name,
                            "reason": "배분액 %d원으로 %d원짜리를 살 수 없다" % (int(per_name), limit)})
            continue

        if dry:
            log("  [DRY] 매수 %s %-12s %4d주 @ %d (약 %d원)"
                % (code, name[:12], qty, limit, qty * limit))
            placed.append({"code": code, "name": name, "qty": qty, "price": limit,
                           "order_no": "", "branch": "", "dry_run": True})
            continue

        try:
            result = client.order(code, "buy", qty, limit, ord_dvsn=ORD_LIMIT)
        except KisError as exc:
            skipped.append({"code": code, "name": name, "reason": "주문 실패: %s" % exc})
            log("  [실패] %s %s — %s" % (code, name, exc))
            continue

        log("  매수 접수 %s %-12s %4d주 @ %d  주문번호 %s"
            % (code, name[:12], qty, limit, result["order_no"]))
        placed.append({"code": code, "name": name, "qty": qty, "price": limit,
                       "order_no": result["order_no"], "branch": result["branch"], "dry_run": False})

    journal["entry"] = {
        "at": moment.isoformat(timespec="seconds"),
        "env": client.env,
        "dry_run": dry,
        "budget_per_name": int(per_name),
        "orders": placed,
        "skipped": skipped,
    }
    save_journal(today, journal)

    log("  접수 %d종목 / 제외 %d종목" % (len(placed), len(skipped)))
    for row in skipped:
        log("    제외 %s %s — %s" % (row["code"], row["name"], row["reason"]))
    if not placed:
        log("  접수된 주문이 없다.")
        return 1
    return 0


def cmd_sweep(args, config) -> int:
    """장 시작 후 미체결 잔량 취소.

    시가가 지정가 위에서 형성되면 체결되지 않고 주문이 장중 내내 살아 있다.
    그 상태로 두면 오후에 가격이 내려왔을 때 체결돼 '시가 매수'가 아니라
    '오후 매수'가 되고, 청산 시각에는 이미 늦다. 그래서 09:05 에 정리한다.
    """
    trading = cfg_trading(config)
    today = now_kst().date()
    journal = load_journal(today)
    entry = journal.get("entry")
    if not entry or entry.get("dry_run"):
        log("취소할 실주문이 없다.")
        return 0

    client = from_env()
    holdings, _ = client.balance()
    filled = {row["code"]: row["qty"] for row in holdings}

    cancelled = []
    for order in entry["orders"]:
        remaining = order["qty"] - filled.get(order["code"], 0)
        if remaining <= 0 or not order.get("order_no"):
            continue
        try:
            client.cancel(order["order_no"], order["branch"])
            cancelled.append({"code": order["code"], "name": order["name"], "qty": remaining})
            log("  취소 %s %s 잔량 %d주" % (order["code"], order["name"], remaining))
        except KisError as exc:
            # 이미 전량 체결됐거나 장이 끝난 주문은 취소가 거부된다. 정상이다.
            log("  취소 불가 %s %s — %s" % (order["code"], order["name"], exc))

    journal["sweep"] = {"at": now_kst().isoformat(timespec="seconds"), "cancelled": cancelled}
    save_journal(today, journal)
    return 0


def cmd_exit(args, config) -> int:
    """청산. 저널이 아니라 실제 잔고를 진실로 삼는다 — 몇 번을 돌려도 안전하다."""
    trading = cfg_trading(config)
    moment = now_kst()
    today = moment.date()

    window = trading["exit"]["window"]
    if not args.force and not in_window(window, moment):
        log("청산 시간대(%s~%s)가 아니다. 현재 %s. 강제로 정리하려면 --force."
            % (window[0], window[1], moment.strftime("%H:%M")))
        return 1

    client = from_env()
    holdings, _ = client.balance()
    if not holdings:
        log("보유 종목이 없다 — 청산할 것이 없다.")
        return 0

    dry = trading.get("dry_run", True) or args.dry_run
    market = str(trading["exit"].get("order_type", "market")).lower() == "market"
    log("[%s] 청산 %s / %d종목"
        % (moment.strftime("%H:%M:%S"), "DRY-RUN" if dry else "실주문", len(holdings)))

    sold, failed = [], []
    for row in holdings:
        qty = row["sellable"]
        if qty <= 0:
            failed.append({"code": row["code"], "name": row["name"], "reason": "매도가능수량 0 (미결제)"})
            continue

        if dry:
            log("  [DRY] 매도 %s %-12s %4d주 시장가" % (row["code"], row["name"][:12], qty))
            sold.append({"code": row["code"], "name": row["name"], "qty": qty, "order_no": "", "dry_run": True})
            continue

        try:
            if market:
                result = client.order(row["code"], "sell", qty, 0, ord_dvsn=ORD_MARKET)
            else:
                # 지정가 청산은 미체결 위험이 있다. 하한가로 넣어 체결을 보장한다.
                quote = client.price(row["code"])
                result = client.order(row["code"], "sell", qty,
                                      round_to_tick(quote["lower_limit"] or quote["price"] * 0.9, "up"),
                                      ord_dvsn=ORD_LIMIT)
        except KisError as exc:
            failed.append({"code": row["code"], "name": row["name"], "reason": str(exc)})
            log("  [실패] %s %s — %s" % (row["code"], row["name"], exc))
            continue

        log("  매도 접수 %s %-12s %4d주  주문번호 %s"
            % (row["code"], row["name"][:12], qty, result["order_no"]))
        sold.append({"code": row["code"], "name": row["name"], "qty": qty,
                     "order_no": result["order_no"], "dry_run": False})

    journal = load_journal(today)
    journal["exit"] = {
        "at": moment.isoformat(timespec="seconds"),
        "dry_run": dry,
        "orders": sold,
        "failed": failed,
    }
    save_journal(today, journal)

    if failed:
        # 청산 실패는 오버나잇 노출이다. 종료코드로 스케줄러에 알린다.
        log("  청산 실패 %d종목 — 오버나잇 노출이 생긴다. 확인이 필요하다." % len(failed))
        return 1
    return 0


def cmd_settle(args, config) -> int:
    """체결 대조. 이 단계가 모의투자를 돌리는 진짜 이유다.

    그림자 모드는 '시가에 사서 종가에 판다'는 가정으로 채점한다. 여기서는
    실제 체결가로 다시 계산해서 둘의 차이 = 슬리피지를 남긴다. 백테스트로는
    절대 알 수 없는 값이고, 비용을 넣은 뒤에도 수익이 남는지를 결정한다.
    """
    trading = cfg_trading(config)
    today = args.date or now_kst().date()
    day_key = today.strftime("%Y%m%d")

    client = from_env()
    fills = client.executions(day_key)
    if not fills:
        log("%s 체결 내역이 없다." % today)
        return 0

    # 종목별로 매수/매도를 각각 수량가중 평균한다. 분할 체결이면 여러 건이 온다.
    legs: dict[str, dict] = {}
    for fill in fills:
        leg = legs.setdefault(fill["code"], {"name": fill["name"], "buy_qty": 0, "buy_amt": 0.0,
                                             "sell_qty": 0, "sell_amt": 0.0})
        if fill["side"] == "buy":
            leg["buy_qty"] += fill["qty"]
            leg["buy_amt"] += fill["amount"]
        else:
            leg["sell_qty"] += fill["qty"]
            leg["sell_amt"] += fill["amount"]

    costs = trading.get("costs", {})
    tax = float(costs.get("tax_pct", 0.20)) / 100.0
    fee = float(costs.get("commission_pct", 0.0036)) / 100.0

    report = load_report(today) or {}
    assumed = {p["code"]: p for p in report.get("picks", [])}

    rows, gross_pnl, net_pnl, invested = [], 0.0, 0.0, 0.0
    for code, leg in sorted(legs.items()):
        if leg["buy_qty"] <= 0 or leg["sell_qty"] <= 0:
            log("  미완결 %s %s (매수 %d / 매도 %d)"
                % (code, leg["name"], leg["buy_qty"], leg["sell_qty"]))
            continue
        buy_px = leg["buy_amt"] / leg["buy_qty"]
        sell_px = leg["sell_amt"] / leg["sell_qty"]
        qty = min(leg["buy_qty"], leg["sell_qty"])
        gross = (sell_px - buy_px) * qty
        cost = leg["sell_amt"] * tax + (leg["buy_amt"] + leg["sell_amt"]) * fee
        net = gross - cost

        rows.append({
            "code": code,
            "name": leg["name"],
            "qty": qty,
            "buy_price": round(buy_px, 1),
            "sell_price": round(sell_px, 1),
            "gross_pct": round((sell_px / buy_px - 1) * 100, 3) if buy_px else 0.0,
            "net_pct": round(net / leg["buy_amt"] * 100, 3) if leg["buy_amt"] else 0.0,
            "cost_pct": round(cost / leg["buy_amt"] * 100, 3) if leg["buy_amt"] else 0.0,
            "assumed_close": assumed.get(code, {}).get("last_close"),
        })
        gross_pnl += gross
        net_pnl += net
        invested += leg["buy_amt"]

    if not rows:
        log("완결된 왕복 거래가 없다.")
        return 0

    gross_pct = gross_pnl / invested * 100 if invested else 0.0
    net_pct = net_pnl / invested * 100 if invested else 0.0

    log("[%s] 체결 기준 정산 — %d종목" % (today, len(rows)))
    log("  %-8s %-12s %6s %9s %9s %8s %8s" % ("종목", "이름", "수량", "매수", "매도", "총수익", "세후"))
    for row in rows:
        log("  %-8s %-12s %6d %9.0f %9.0f %+7.2f%% %+7.2f%%"
            % (row["code"], row["name"][:12], row["qty"], row["buy_price"],
               row["sell_price"], row["gross_pct"], row["net_pct"]))
    log("  ─────")
    log("  투입 %d원 · 총수익 %+.3f%% · 비용 %.3f%%p · **세후 %+.3f%%**"
        % (int(invested), gross_pct, gross_pct - net_pct, net_pct))

    journal = load_journal(today)
    journal["settle"] = {
        "at": now_kst().isoformat(timespec="seconds"),
        "env": client.env,
        "invested": int(invested),
        "gross_pct": round(gross_pct, 3),
        "net_pct": round(net_pct, 3),
        "cost_pct": round(gross_pct - net_pct, 3),
        "legs": rows,
    }
    save_journal(today, journal)

    if trading.get("publish_summary"):
        # 공개 대시보드에는 수익률만 올린다. 금액·수량·주문번호는 남기지 않는다.
        _publish_summary(today, client.env, gross_pct, net_pct, len(rows))
    return 0


def _publish_summary(day: date, env: str, gross_pct: float, net_pct: float, count: int) -> None:
    path = DATA_DIR / "live_summary.json"
    payload = {"env": env, "sessions": []}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["env"] = env
    payload["sessions"] = [s for s in payload.get("sessions", []) if s["date"] != day.isoformat()]
    payload["sessions"].append({
        "date": day.isoformat(),
        "names": count,
        "gross_pct": round(gross_pct, 3),
        "net_pct": round(net_pct, 3),
    })
    payload["sessions"].sort(key=lambda s: s["date"])
    series = [s["net_pct"] for s in payload["sessions"]]
    compounded = 1.0
    for value in series:
        compounded *= 1 + value / 100
    payload["cumulative_net_pct"] = round((compounded - 1) * 100, 3)
    payload["updated_at"] = now_kst().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


COMMANDS = {
    "preflight": cmd_preflight,
    "status": cmd_status,
    "prepare": cmd_prepare,
    "entry": cmd_entry,
    "sweep": cmd_sweep,
    "exit": cmd_exit,
    "settle": cmd_settle,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="스크리닝 리포트를 주문으로 실행한다")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--force", action="store_true",
                        help="시간대·중복 검사를 건너뛴다 (테스트/수동 정리용)")
    parser.add_argument("--dry-run", action="store_true",
                        help="config 설정과 무관하게 주문을 내지 않는다")
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="settle 전용. 과거 날짜를 다시 정산한다 (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_config()
    try:
        return COMMANDS[args.command](args, config)
    except KisError as exc:
        log("API 오류: %s" % exc)
        return 2
    except (RuntimeError, ValueError) as exc:
        # 자격증명 누락·계좌번호 형식 오류가 대부분이다. 스택트레이스를
        # 그대로 보이면 원인이 묻힌다 — 한 줄로 말해준다.
        log("설정 오류: %s" % exc)
        log("  .env 의 KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT 를 확인해라.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
