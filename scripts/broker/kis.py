"""한국투자증권(KIS) Open API REST 클라이언트.

이 모듈은 주문을 '어떻게 보내는가'만 책임진다. '무엇을 살까'는 screen.py,
'언제 얼마나 살까'는 trade.py다. 셋을 섞지 않는 이유는 실계좌 전환 시
사람이 다시 읽어야 할 코드를 이 파일 하나로 좁히기 위해서다.

■ 스펙 출처
공식 저장소 koreainvestment/open-trading-api 의 examples_llm 샘플 기준이다.
웹에 널리 퍼진 tr_id 는 구버전이다 — 현금주문은 TTTC0802U/TTTC0801U 가 아니라
TTTC0012U(매수)/TTTC0011U(매도)이고, EXCG_ID_DVSN_CD 필드가 추가됐다.
구버전 값으로 보내면 주문이 통째로 거부된다. 스펙을 고칠 일이 있으면
검색 결과가 아니라 위 저장소를 확인해라.

■ 토큰
접근토큰은 24시간 유효한데 발급은 분당 1회로 제한된다. 매 실행마다 새로
받으면 하루 몇 번만 돌려도 제한에 걸린다. 그래서 파일에 캐시하고 만료
10분 전에만 재발급한다. 캐시 파일은 저장소 바깥(기본 ~/.kis)에 둔다 —
토큰이 커밋되면 그 자체로 사고다.

■ 유량
모의투자는 실전보다 호출 제한이 훨씬 빡빡하다(EGW00201). 호출 간 최소
간격을 강제해서 10종목 루프가 제한에 걸리지 않게 한다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# 실전과 모의는 호스트가 다르다. 같은 앱키를 양쪽에 쓸 수 없고 계좌번호도 다르다.
DOMAINS = {
    "real": "https://openapi.koreainvestment.com:9443",
    "demo": "https://openapivts.koreainvestment.com:29443",
}

# tr_id 표. 여기 값이 곧 계약이라 상수로 모아둔다.
TR = {
    ("order", "real", "buy"): "TTTC0012U",
    ("order", "real", "sell"): "TTTC0011U",
    ("order", "demo", "buy"): "VTTC0012U",
    ("order", "demo", "sell"): "VTTC0011U",
    ("cancel", "real"): "TTTC0013U",
    ("cancel", "demo"): "VTTC0013U",
    ("balance", "real"): "TTTC8434R",
    ("balance", "demo"): "VTTC8434R",
    ("psbl", "real"): "TTTC8908R",
    ("psbl", "demo"): "VTTC8908R",
    ("ccld", "real"): "TTTC0081R",
    ("ccld", "demo"): "VTTC0081R",
    ("price", "real"): "FHKST01010100",
    ("price", "demo"): "FHKST01010100",
}

# 주문구분. 시장가(01)를 진입에 쓰면 호가가 얇을 때 상한가까지 긁는다.
# 진입은 반드시 지정가(00), 청산만 시장가를 쓴다 (이유는 trade.py 주석).
ORD_LIMIT = "00"
ORD_MARKET = "01"

_TOKEN_DIR = Path(os.environ.get("KIS_TOKEN_DIR", Path.home() / ".kis"))


class KisError(RuntimeError):
    """API 가 rt_cd != '0' 을 돌려준 경우. msg_cd 로 분기할 수 있게 담아둔다."""

    def __init__(self, code: str, message: str, tr_id: str = ""):
        super().__init__(f"[{code}] {message}" + (f" (tr_id={tr_id})" if tr_id else ""))
        self.code = code
        self.message = message


def _num(value, default: float = 0.0) -> float:
    """KIS 응답은 숫자도 전부 문자열이고 빈 값은 '' 로 온다."""
    try:
        text = str(value).strip().replace(",", "")
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


class KisClient:
    """한 계좌 = 한 클라이언트. 스레드 간 공유하지 않는 것을 전제로 한다."""

    def __init__(
        self,
        env: str,
        app_key: str,
        app_secret: str,
        account: str,
        *,
        timeout: float = 10.0,
        min_interval: float | None = None,
    ):
        if env not in DOMAINS:
            raise ValueError("env 는 real 또는 demo 여야 한다 — 받은 값: %r" % env)
        if not app_key or not app_secret:
            raise ValueError("앱키/앱시크릿이 비어 있다. 환경변수를 확인해라.")

        # 계좌번호는 12345678-01 과 1234567801 둘 다 받는다.
        digits = account.replace("-", "").strip()
        if len(digits) != 10 or not digits.isdigit():
            raise ValueError("계좌번호는 종합8자리+상품2자리 = 10자리 — 받은 값: %r" % account)
        self.cano, self.acnt_prdt_cd = digits[:8], digits[8:]

        self.env = env
        self.base = DOMAINS[env]
        self.app_key = app_key
        self.app_secret = app_secret
        self.timeout = timeout
        # 모의투자는 호출 제한이 낮다. 초당 2건 근처에서 EGW00201 이 난다.
        self.min_interval = min_interval if min_interval is not None else (0.6 if env == "demo" else 0.12)

        self._session = requests.Session()
        self._token: str | None = None
        self._token_expires: datetime | None = None
        self._last_call = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 인증

    @property
    def _token_path(self) -> Path:
        # 앱키 앞 8자리로 파일을 나눈다. 실전/모의 앱키가 달라 서로 덮어쓰지 않는다.
        return _TOKEN_DIR / ("token_%s_%s.json" % (self.env, self.app_key[:8]))

    def _load_cached_token(self) -> bool:
        try:
            raw = json.loads(self._token_path.read_text(encoding="utf-8"))
            expires = datetime.fromisoformat(raw["expires"])
        except (OSError, ValueError, KeyError):
            return False
        # 만료 10분 전부터는 새로 받는다. 장중에 토큰이 끊기면 청산이 막힌다.
        if expires - timedelta(minutes=10) <= datetime.now():
            return False
        self._token, self._token_expires = raw["token"], expires
        return True

    def _issue_token(self) -> None:
        response = self._session.post(
            self.base + "/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            },
            timeout=self.timeout,
        )
        payload = response.json() if response.content else {}
        token = payload.get("access_token")
        if not token:
            # 분당 1회 제한에 걸리면 여기로 온다. 캐시가 있는데도 왔다면 캐시 로직 버그다.
            raise KisError(
                payload.get("error_code", "TOKEN"),
                payload.get("error_description") or ("토큰 발급 실패 (HTTP %s)" % response.status_code),
            )

        self._token = token
        expired_at = payload.get("access_token_token_expired")
        if expired_at:
            self._token_expires = datetime.strptime(expired_at, "%Y-%m-%d %H:%M:%S")
        else:
            self._token_expires = datetime.now() + timedelta(seconds=_num(payload.get("expires_in"), 86400))

        try:
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(
                json.dumps({"token": token, "expires": self._token_expires.isoformat()}),
                encoding="utf-8",
            )
            os.chmod(self._token_path, 0o600)
        except OSError:
            pass  # 캐시 실패는 치명적이지 않다 — 다음 실행에서 한 번 더 받으면 된다

    def token(self) -> str:
        with self._lock:
            if self._token and self._token_expires and self._token_expires - timedelta(minutes=10) > datetime.now():
                return self._token
            if not self._load_cached_token():
                self._issue_token()
            return self._token  # type: ignore[return-value]

    # ------------------------------------------------------------- 호출 공통

    def _headers(self, tr_id: str, hashkey: str = "") -> dict:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": "Bearer " + self.token(),
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",  # 개인
        }
        if hashkey:
            headers["hashkey"] = hashkey
        return headers

    def _hashkey(self, body: dict) -> str:
        """POST 바디 위변조 검증용. 실패해도 주문은 나가지만 붙이는 편이 안전하다."""
        try:
            response = self._session.post(
                self.base + "/uapi/hashkey",
                json=body,
                headers={
                    "content-type": "application/json; charset=utf-8",
                    "appkey": self.app_key,
                    "appsecret": self.app_secret,
                },
                timeout=self.timeout,
            )
            return response.json().get("HASH", "")
        except Exception:
            return ""

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.monotonic()

    def _call(self, method: str, path: str, tr_id: str, payload: dict, retries: int = 3) -> dict:
        last: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            try:
                if method == "GET":
                    response = self._session.get(
                        self.base + path, headers=self._headers(tr_id), params=payload, timeout=self.timeout
                    )
                else:
                    response = self._session.post(
                        self.base + path,
                        headers=self._headers(tr_id, self._hashkey(payload)),
                        json=payload,
                        timeout=self.timeout,
                    )
                body = response.json() if response.content else {}

                if body.get("rt_cd") == "0":
                    return body

                code = body.get("msg_cd", str(response.status_code))
                message = str(body.get("msg1", response.text[:200])).strip()

                # 유량 초과와 토큰 만료만 재시도한다. 나머지(잔고부족·종목오류)는
                # 재시도해도 같은 답이 오고, 주문에서는 중복 주문 위험이 된다.
                if code in ("EGW00201", "EGW00133"):
                    last = KisError(code, message, tr_id)
                    time.sleep(1.0 + attempt)
                    continue
                if code in ("EGW00121", "EGW00123"):  # 토큰 만료 / 유효하지 않음
                    self._token = None
                    try:
                        self._token_path.unlink()
                    except OSError:
                        pass
                    last = KisError(code, message, tr_id)
                    continue
                raise KisError(code, message, tr_id)

            except requests.RequestException as exc:
                # 통신 예외는 주문이 나갔는지 알 수 없다. POST 는 절대 재시도하지 않는다.
                last = exc
                if method == "POST":
                    raise KisError(
                        "NETWORK", "주문 전송 중 통신 오류 — 체결 여부를 직접 확인해야 한다: %s" % exc, tr_id
                    ) from exc
                time.sleep(1.0 + attempt)

        if isinstance(last, KisError):
            raise last
        raise KisError("RETRY", "%d회 재시도 실패: %s" % (retries, last), tr_id)

    # ------------------------------------------------------------------ 조회

    def price(self, code: str) -> dict:
        """현재가 조회. 시장코드 J = 주식/ETF/ETN."""
        body = self._call(
            "GET",
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            TR[("price", self.env)],
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        out = body.get("output") or {}
        return {
            "code": code,
            "price": _num(out.get("stck_prpr")),
            "open": _num(out.get("stck_oprc")),
            "high": _num(out.get("stck_hgpr")),
            "low": _num(out.get("stck_lwpr")),
            "prev_close": _num(out.get("stck_sdpr")),  # 기준가 = 전일 종가
            "upper_limit": _num(out.get("stck_mxpr")),
            "lower_limit": _num(out.get("stck_llam")),
            "change_pct": _num(out.get("prdy_ctrt")),
            "halted": str(out.get("temp_stop_yn", "N")).upper() == "Y",
        }

    def balance(self) -> tuple[list[dict], dict]:
        """(보유종목, 계좌요약). 청산은 저널이 아니라 이 함수를 진실로 삼는다."""
        body = self._call(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            TR[("balance", self.env)],
            {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",            # 종목별
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",            # 전일매매 포함
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )
        holdings = []
        for row in body.get("output1") or []:
            qty = int(_num(row.get("hldg_qty")))
            if qty <= 0:
                continue
            holdings.append(
                {
                    "code": str(row.get("pdno", "")).strip(),
                    "name": str(row.get("prdt_name", "")).strip(),
                    "qty": qty,
                    # 매도가능 수량은 보유 수량과 다르다(미결제분 등). 청산은 이쪽을 쓴다.
                    "sellable": int(_num(row.get("ord_psbl_qty"), qty)),
                    "avg_price": _num(row.get("pchs_avg_pric")),
                    "current": _num(row.get("prpr")),
                    "eval_amount": _num(row.get("evlu_amt")),
                    "pnl_pct": _num(row.get("evlu_pfls_rt")),
                }
            )
        rows = body.get("output2") or [{}]
        summary = rows[0] if rows else {}
        return holdings, {
            "cash": _num(summary.get("dnca_tot_amt")),                # 예수금 총액
            "settled_cash": _num(summary.get("prvs_rcdl_excc_amt")),  # D+2 정산 예수금
            "total_eval": _num(summary.get("tot_evlu_amt")),
            "stock_eval": _num(summary.get("scts_evlu_amt")),
        }

    def orderable_cash(self, code: str, price: float) -> float:
        """해당 종목을 그 가격에 지정가로 살 때의 주문가능현금.

        예수금과 다르다 — 증거금율과 기주문 금액이 반영된 값이라
        수량 산정에는 반드시 이쪽을 써야 한다.
        """
        body = self._call(
            "GET",
            "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            TR[("psbl", self.env)],
            {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "PDNO": code,
                "ORD_UNPR": str(int(price)),
                "ORD_DVSN": ORD_LIMIT,
                "CMA_EVLU_AMT_ICLD_YN": "N",
                "OVRS_ICLD_YN": "N",
            },
        )
        return _num((body.get("output") or {}).get("ord_psbl_cash"))

    def executions(self, day: str, code: str = "") -> list[dict]:
        """당일 체결 내역. day 는 YYYYMMDD.

        실제 체결단가를 여기서 얻는다. 백테스트의 '시가/종가 가정'과 실제
        체결가의 차이 = 슬리피지이고, 그걸 재는 게 모의투자 단계의 목적이다.
        """
        rows: list[dict] = []
        fk, nk = "", ""
        for _ in range(20):  # 연속조회 상한. 10종목 왕복이면 1~2페이지면 끝난다.
            body = self._call(
                "GET",
                "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
                TR[("ccld", self.env)],
                {
                    "CANO": self.cano,
                    "ACNT_PRDT_CD": self.acnt_prdt_cd,
                    "INQR_STRT_DT": day,
                    "INQR_END_DT": day,
                    "SLL_BUY_DVSN_CD": "00",  # 전체
                    "INQR_DVSN": "00",
                    "PDNO": code,
                    "CCLD_DVSN": "01",        # 체결분만
                    "ORD_GNO_BRNO": "",
                    "ODNO": "",
                    "INQR_DVSN_3": "00",
                    "INQR_DVSN_1": "",
                    "CTX_AREA_FK100": fk,
                    "CTX_AREA_NK100": nk,
                },
            )
            for row in body.get("output1") or []:
                filled = int(_num(row.get("tot_ccld_qty")))
                if filled <= 0:
                    continue
                rows.append(
                    {
                        "order_no": str(row.get("odno", "")).strip(),
                        "code": str(row.get("pdno", "")).strip(),
                        "name": str(row.get("prdt_name", "")).strip(),
                        "side": "sell" if str(row.get("sll_buy_dvsn_cd")) == "01" else "buy",
                        "qty": filled,
                        "price": _num(row.get("avg_prvs")),  # 체결 평균가
                        "amount": _num(row.get("tot_ccld_amt")),
                        "time": str(row.get("ord_tmd", "")).strip(),
                    }
                )
            fk = str(body.get("ctx_area_fk100", "")).strip()
            nk = str(body.get("ctx_area_nk100", "")).strip()
            if str(body.get("tr_cont", "")) not in ("F", "M") or not nk:
                break
        return rows

    # ------------------------------------------------------------------ 주문

    def order(self, code: str, side: str, qty: int, price: int = 0, ord_dvsn: str = ORD_LIMIT) -> dict:
        """현금 주문. side 는 buy 또는 sell.

        시장가는 ORD_UNPR 을 0 으로 보낸다. 반대로 지정가에 0 을 보내면
        상한가로 주문금액이 잡히므로(공식 문서 주의사항) 가격을 반드시 채운다.
        """
        if side not in ("buy", "sell"):
            raise ValueError("side 는 buy 또는 sell")
        if qty <= 0:
            raise ValueError("주문 수량이 0 이하: %s" % qty)
        if ord_dvsn == ORD_LIMIT and price <= 0:
            raise ValueError("지정가 주문에 가격이 없다")

        body = self._call(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-cash",
            TR[("order", self.env, side)],
            {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "PDNO": code,
                "ORD_DVSN": ord_dvsn,
                "ORD_QTY": str(int(qty)),
                "ORD_UNPR": str(int(price)) if ord_dvsn == ORD_LIMIT else "0",
                "EXCG_ID_DVSN_CD": "KRX",
                "SLL_TYPE": "01" if side == "sell" else "",  # 01 = 일반매도
                "CNDT_PRIC": "",
            },
            retries=1,  # 주문은 재시도하지 않는다. 중복 주문이 실패보다 나쁘다.
        )
        out = body.get("output") or {}
        return {
            "order_no": str(out.get("ODNO", "")).strip(),
            "branch": str(out.get("KRX_FWDG_ORD_ORGNO", "")).strip(),
            "time": str(out.get("ORD_TMD", "")).strip(),
            "code": code,
            "side": side,
            "qty": int(qty),
            "price": int(price),
            "ord_dvsn": ord_dvsn,
        }

    def cancel(self, order_no: str, branch: str, qty: int = 0) -> dict:
        """미체결 주문 취소. qty=0 이면 잔량 전부."""
        body = self._call(
            "POST",
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            TR[("cancel", self.env)],
            {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "KRX_FWDG_ORD_ORGNO": branch,
                "ORGN_ODNO": order_no,
                "ORD_DVSN": ORD_LIMIT,
                "RVSE_CNCL_DVSN_CD": "02",  # 02 = 취소
                "ORD_QTY": str(int(qty)),
                "ORD_UNPR": "0",
                "QTY_ALL_ORD_YN": "Y" if qty == 0 else "N",
                "EXCG_ID_DVSN_CD": "KRX",
            },
            retries=1,
        )
        out = body.get("output") or {}
        return {"order_no": str(out.get("ODNO", "")).strip(), "time": str(out.get("ORD_TMD", "")).strip()}


def from_env() -> KisClient:
    """환경변수에서 클라이언트를 만든다. 자격증명은 절대 저장소에 두지 않는다.

    KIS_ENV          demo(기본) | real
    KIS_APP_KEY      앱키
    KIS_APP_SECRET   앱시크릿
    KIS_ACCOUNT      계좌번호 10자리 (12345678-01 형식도 허용)

    실전 전환은 KIS_ENV=real 만으로는 안 되고 KIS_ALLOW_REAL 을 함께 요구한다.
    환경변수 하나를 잘못 넣어 실계좌에 주문이 나가는 사고를 막는 이중잠금이다.
    """
    env = os.environ.get("KIS_ENV", "demo").strip().lower()
    if env == "real" and os.environ.get("KIS_ALLOW_REAL") != "I_UNDERSTAND":
        raise RuntimeError(
            "실계좌 모드에는 KIS_ALLOW_REAL=I_UNDERSTAND 가 함께 필요하다. "
            "모의투자에서 세후 초과수익이 유의해지기 전에는 켜지 마라."
        )
    return KisClient(
        env=env,
        app_key=os.environ.get("KIS_APP_KEY", ""),
        app_secret=os.environ.get("KIS_APP_SECRET", ""),
        account=os.environ.get("KIS_ACCOUNT", ""),
    )
