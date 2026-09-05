"""주문 로직 셀프테스트 — 네트워크도 자격증명도 없이 돈다.

    python scripts/selftest_trade.py

가짜 브로커를 끼워 넣고 실제 리포트 JSON으로 진입·청산·정산을 통과시킨다.
검증 대상은 '무엇을 살까'가 아니라 '주문이 제대로 만들어지는가'다.

이걸 만든 이유: 주문 코드의 버그는 모의투자에서도 하루에 한 번밖에 못 만난다.
호가단위가 틀리면 주문이 거부되고, 그걸 다음 날 아침 08:50에 알게 된다.
여기서 미리 다 밟아본다.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import trade
from broker.kis import ORD_LIMIT, ORD_MARKET, KisError

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print("  %s %s%s" % ("OK  " if condition else "FAIL", name, (" — " + detail) if detail else ""))


class FakeBroker:
    """KisClient 와 같은 표면을 가진 가짜. 주문을 메모리에 쌓기만 한다."""

    def __init__(self, cash=100_000_000.0, holdings=None, reject=()):
        self.env = "demo"
        self.cano, self.acnt_prdt_cd = "12345678", "01"
        self.cash = cash
        self._holdings = list(holdings or [])
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self.reject = set(reject)

    def token(self):
        return "fake"

    def price(self, code):
        # 호가단위 경계를 일부러 밟는 가격들. 49,990 은 올림하면 구간이 바뀐다.
        table = {
            "069960": 94_600, "247540": 32_900, "138930": 14_370,
            "035900": 38_900, "015760": 32_150,
        }
        base = float(table.get(code, 49_990))
        return {
            "code": code, "price": base, "open": base, "high": base, "low": base,
            "prev_close": base, "upper_limit": base * 1.3, "lower_limit": base * 0.7,
            "change_pct": 0.0, "halted": code in self.reject,
        }

    def balance(self):
        return list(self._holdings), {
            "cash": self.cash, "settled_cash": self.cash,
            "total_eval": self.cash, "stock_eval": 0.0,
        }

    def orderable_cash(self, code, price):
        return self.cash

    def order(self, code, side, qty, price=0, ord_dvsn=ORD_LIMIT):
        if code in self.reject:
            raise KisError("40580000", "주문 거부(테스트)")
        record = {"code": code, "side": side, "qty": qty, "price": price, "ord_dvsn": ord_dvsn,
                  "order_no": "%06d" % (len(self.orders) + 1), "branch": "00950", "time": "085000"}
        self.orders.append(record)
        return record

    def cancel(self, order_no, branch, qty=0):
        self.cancelled.append(order_no)
        return {"order_no": order_no, "time": "090500"}

    def executions(self, day, code=""):
        rows = []
        for record in self.orders:
            px = record["price"] or 50_000
            # 매도는 1.5% 높은 값에 체결됐다고 가정 — 정산 산식 검증용
            fill = px if record["side"] == "buy" else px * 1.015
            rows.append({
                "order_no": record["order_no"], "code": record["code"], "name": "테스트",
                "side": record["side"], "qty": record["qty"], "price": fill,
                "amount": fill * record["qty"], "time": record["time"],
            })
        return rows


class Args:
    def __init__(self, **kw):
        self.force = kw.get("force", False)
        self.dry_run = kw.get("dry_run", False)
        self.date = kw.get("date")


def main() -> int:
    config = trade.load_config()
    trading = dict(config.get("trading") or {})
    trading["enabled"] = True
    trading["dry_run"] = False       # 가짜 브로커라 실주문이 나가지 않는다
    config = dict(config, trading=trading)

    # ---------------------------------------------------------- 호가단위
    print("\n[1] 호가가격단위")
    cases = [
        (1_995, "up", 1_995, "2천원 미만 = 1원"),
        (4_321, "up", 4_325, "2천~5천 = 5원"),
        (14_370, "up", 14_370, "5천~2만 = 10원"),
        (32_901, "up", 32_950, "2만~5만 = 50원"),
        (49_990, "up", 50_000, "경계 넘김 — 50원이 아니라 100원 단위여야 한다"),
        (94_600, "up", 94_600, "5만~20만 = 100원"),
        (234_567, "up", 235_000, "20만~50만 = 500원"),
        (612_345, "up", 613_000, "50만 이상 = 1천원"),
        (32_949, "down", 32_900, "내림"),
    ]
    for value, mode, expected, note in cases:
        got = trade.round_to_tick(value, mode)
        check("%d %s -> %d (%s)" % (value, mode, expected, note), got == expected, "받은 값 %d" % got)

    # ---------------------------------------------------------- 시간 창
    print("\n[2] 실행 시간대")
    def at(hhmm):
        return datetime(2026, 9, 7, int(hhmm[:2]), int(hhmm[3:]), tzinfo=trade.KST)
    check("08:50 은 진입 시간대", trade.in_window(["08:00", "09:00"], at("08:50")))
    check("09:30 은 진입 시간대 아님", not trade.in_window(["08:00", "09:00"], at("09:30")))
    check("15:20 은 청산 시간대", trade.in_window(["15:00", "15:30"], at("15:20")))
    check("14:00 은 청산 시간대 아님", not trade.in_window(["15:00", "15:30"], at("14:00")))

    # ------------------------------------------------- 진입/청산/정산 통합
    print("\n[3] 진입 → 청산 → 정산")
    reports = sorted(p for p in trade.DATA_DIR.glob("*.json") if p.stem[0].isdigit())
    if not reports:
        print("  리포트가 없어 통합 테스트를 건너뛴다.")
        return 1
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    day = date.fromisoformat(report["trade_date"])

    tmp = Path(tempfile.mkdtemp(prefix="trade-selftest-"))
    original_journal, original_now, original_from_env, original_trading_day = (
        trade.JOURNAL_DIR, trade.now_kst, trade.from_env, trade.is_trading_day)
    trade.JOURNAL_DIR = tmp
    trade.is_trading_day = lambda d: True
    try:
        broker = FakeBroker()
        trade.from_env = lambda: broker
        trade.now_kst = lambda: datetime.combine(day, datetime.min.time(), trade.KST).replace(hour=8, minute=50)

        rc = trade.cmd_entry(Args(), config)
        check("진입 종료코드 0", rc == 0, "받은 값 %s" % rc)
        check("10종목 매수 접수", len(broker.orders) == len(report["picks"]),
              "%d건 / 후보 %d개" % (len(broker.orders), len(report["picks"])))
        check("전부 지정가", all(o["ord_dvsn"] == ORD_LIMIT for o in broker.orders))
        check("전부 매수", all(o["side"] == "buy" for o in broker.orders))
        check("수량 모두 1주 이상", all(o["qty"] >= 1 for o in broker.orders))
        check("지정가가 호가단위에 정렬됨",
              all(o["price"] % trade.tick_size(o["price"]) == 0 for o in broker.orders))

        notional = sum(o["qty"] * o["price"] for o in broker.orders)
        cap = broker.cash * trading["capital_fraction"]
        check("총 주문금액이 배분 예산 이하", notional <= cap + 1,
              "%d원 <= %d원" % (notional, int(cap)))
        per_cap = trading.get("max_order_krw") or float("inf")
        check("종목당 상한 준수", all(o["qty"] * o["price"] <= per_cap for o in broker.orders))

        # 중복 진입 차단
        before = len(broker.orders)
        trade.cmd_entry(Args(), config)
        check("두 번째 진입은 거부됨 (저널)", len(broker.orders) == before)

        # 잔고가 남아 있으면 진입 거부
        dirty = FakeBroker(holdings=[{"code": "005930", "name": "삼성전자", "qty": 10,
                                      "sellable": 10, "avg_price": 70000, "current": 71000,
                                      "eval_amount": 710000, "pnl_pct": 1.4}])
        trade.from_env = lambda: dirty
        (tmp / ("%s.json" % day.isoformat())).unlink()
        rc = trade.cmd_entry(Args(), config)
        check("전날 잔고가 남으면 진입 거부", rc == 1 and not dirty.orders,
              "종료코드 %s, 주문 %d건" % (rc, len(dirty.orders)))

        # 청산 — 잔고 기준
        trade.from_env = lambda: dirty
        trade.now_kst = lambda: datetime.combine(day, datetime.min.time(), trade.KST).replace(hour=15, minute=20)
        rc = trade.cmd_exit(Args(), config)
        sells = [o for o in dirty.orders if o["side"] == "sell"]
        check("청산 종료코드 0", rc == 0, "받은 값 %s" % rc)
        check("보유 1종목을 시장가 매도", len(sells) == 1 and sells[0]["ord_dvsn"] == ORD_MARKET)
        check("매도가능수량으로 주문", sells[0]["qty"] == 10)

        # 청산 멱등성 — 잔고가 비면 아무 일도 없어야 한다
        flat = FakeBroker(holdings=[])
        trade.from_env = lambda: flat
        rc = trade.cmd_exit(Args(), config)
        check("잔고가 없으면 청산은 no-op", rc == 0 and not flat.orders)

        # 거래정지 종목은 건너뛴다
        halted_code = report["picks"][0]["code"]
        blocked = FakeBroker(reject=[halted_code])
        trade.from_env = lambda: blocked
        trade.now_kst = lambda: datetime.combine(day, datetime.min.time(), trade.KST).replace(hour=8, minute=50)
        (tmp / ("%s.json" % day.isoformat())).unlink()
        trade.cmd_entry(Args(), config)
        check("거래정지 종목은 제외", all(o["code"] != halted_code for o in blocked.orders),
              "%d종목 접수" % len(blocked.orders))

        # 정산 — 매도가 1.5% 높게 체결됐다고 가정했으므로 총수익 +1.5%,
        # 비용(세 0.20 + 수수료 왕복 0.0072)을 빼면 세후는 그보다 낮아야 한다.
        trade.now_kst = lambda: datetime.combine(day, datetime.min.time(), trade.KST).replace(hour=15, minute=40)
        settle_broker = FakeBroker()
        trade.from_env = lambda: settle_broker
        for pick in report["picks"][:3]:
            settle_broker.order(pick["code"], "buy", 10, trade.round_to_tick(pick["last_close"], "up"))
            settle_broker.order(pick["code"], "sell", 10, trade.round_to_tick(pick["last_close"], "up"))
        rc = trade.cmd_settle(Args(date=day), config)
        journal = json.loads((tmp / ("%s.json" % day.isoformat())).read_text(encoding="utf-8"))
        settle = journal.get("settle") or {}
        check("정산 종료코드 0", rc == 0)
        check("총수익 +1.5% 로 계산됨", abs(settle.get("gross_pct", 0) - 1.5) < 0.02,
              "받은 값 %s" % settle.get("gross_pct"))
        check("세후 < 총수익 (비용 차감)", settle.get("net_pct", 9) < settle.get("gross_pct", 0))
        expected_cost = 0.20 * 1.015 + 0.0036 * (1 + 1.015)  # 매도금액 기준 세 + 왕복 수수료
        check("비용이 예상 범위 (%.3f%%p 내외)" % expected_cost,
              abs(settle.get("cost_pct", 0) - expected_cost) < 0.02,
              "받은 값 %s" % settle.get("cost_pct"))
        check("저널에 체결 기록 3종목", len(settle.get("legs", [])) == 3)

    finally:
        trade.JOURNAL_DIR, trade.now_kst = original_journal, original_now
        trade.from_env, trade.is_trading_day = original_from_env, original_trading_day
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n통과 %d / 실패 %d" % (len(PASS), len(FAIL)))
    for name in FAIL:
        print("  실패: %s" % name)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
