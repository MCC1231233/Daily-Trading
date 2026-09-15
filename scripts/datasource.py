"""데이터 수집 계층.

KRX 정보데이터시스템 엔드포인트는 2026년부터 로그인이 필요해져서 사용하지 않는다.
대신 두 개의 공개 소스만 쓴다.

  1. FinanceDataReader.StockListing  -> 전체 상장종목 스냅샷 (종가/거래대금/시총)
  2. 네이버 금융 모바일 API          -> 일별 시세, 투자자별 수급, 밸류에이션, 컨센서스

모든 네트워크 호출은 실패해도 예외를 밖으로 던지지 않고 None/빈 값을 돌려준다.
한 종목의 수급 데이터가 빠졌다고 전체 스크리닝이 죽으면 안 되기 때문이다.
빠진 팩터는 screen.py에서 중립값으로 처리하고 리포트에 결측으로 표기한다.
"""

from __future__ import annotations

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

NAVER_API = "https://m.stock.naver.com/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Referer": "https://m.stock.naver.com/",
    "Accept": "application/json",
}

# 네이버 API는 pageSize 60까지만 허용한다 (100은 400을 돌려준다).
MAX_PAGE_SIZE = 60

KST = ZoneInfo("Asia/Seoul")

# FinanceDataReader.StockListing 이 내부적으로 읽는 캐시 저장소.
#
# ■ 왜 fdr.StockListing 을 직접 쓰지 않는가
# FDR 은 (1) KRX 에 max_work_dt(최종 거래일)를 묻고 (2) 그 날짜의 CSV 를 이
# 저장소에서 받는다. 그런데 KRX 는 장 시작 전에도 '오늘'을 최종 거래일로
# 답하는 반면, 캐시 저장소는 장이 끝난 뒤에야 그날 파일을 올린다. 두 시점이
# 어긋나는 날 fdr.StockListing 은 404 로 죽고, 유니버스가 통째로 비어
# 스크리닝 전체가 실패한다. 실제로 2026-09-05 와 2026-09-08 두 번 그렇게
# 리포트가 누락됐다 (Actions run #10, #33).
#
# 그래서 max_work_dt 를 묻지 않고 오늘부터 거꾸로 걸어 존재하는 최신 파일을
# 찾는다. 스크리너는 어차피 **전일 종가**를 쓰므로 최신 가용 파일이 정확히
# 필요한 데이터다 — 품질 저하가 아니라 오히려 정상 동작이다.
KRX_LISTING_CACHE = (
    "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache"
    "/refs/heads/master/data/listing/krx/%s.csv"
)

# 캐시 CSV 의 Market 값은 4종이다. KOSDAQ GLOBAL 을 KOSDAQ 으로 합쳐야
# 기존 fdr.StockListing("KOSDAQ") 결과와 종목 수가 정확히 일치한다
# (2026-09-04 기준 2,527종목으로 실측 대조 완료). KONEX 는 제외한다.
LISTING_MARKETS = ("KOSPI", "KOSDAQ", "KOSDAQ GLOBAL")

# 마지막으로 성공한 상장목록의 기준일. screen.py 가 신선도 경고에 쓴다.
LISTING_ASOF: str | None = None


# 장 마감 + 데이터 반영 여유. 이 시각 이전에는 '오늘' 파일을 쓰지 않는다.
# screen.py 의 SESSION_COMPLETE_HOUR 와 같은 값이지만, datasource 가 screen 을
# import 하면 순환이 되므로 여기 따로 둔다.
LISTING_SESSION_COMPLETE_HOUR = 16


def _listing_is_usable(frame: pd.DataFrame) -> bool:
    """가격이 실제로 채워진 파일인지.

    ■ 이 검사가 없으면 무슨 일이 나는가 (2026-09-14, 09-15 실측)
    캐시 저장소는 장 마감 전에 **종목 목록만 담긴 껍데기 파일**을 먼저 올린다.
    2026-09-15.csv 는 2,871행이 전부 있는데 Close/Amount/Marcap 이 100% 결측
    이었다. 파일 존재만 확인하고 집어오면 prefilter 의 dropna 에서 전 종목이
    날아가 '0종목'이 되고 스크리닝이 통째로 실패한다. 이틀 연속 그렇게 죽었다
    (Actions run #38, #39).

    정상 파일은 Close>0 비율이 100% 다. 껍데기는 0% 다. 0.5 는 넉넉한 해자다.
    """
    if frame.empty or "Close" not in frame.columns:
        return False
    close = pd.to_numeric(frame["Close"], errors="coerce")
    return bool((close > 0).mean() >= 0.5)


def _fetch_listing(max_lookback: int = 10) -> tuple[pd.DataFrame, str]:
    """상장종목 스냅샷을 캐시에서 받는다. (프레임, 기준일) 을 돌려준다.

    하루씩 거슬러 올라가며 **가격이 채워진** 첫 파일을 쓴다. 주말·연휴는
    자연히 건너뛰어지므로 휴장일 계산이 따로 필요 없다.

    시작점은 오늘이 아니라 '마지막으로 마감된 세션'이다. 장중(16시 이전)에
    돌리면 오늘 파일은 미완성이라, 있더라도 당일 누적 거래대금으로 유동성
    필터가 걸려 후보가 과소 선정된다. 스크리너는 전일 종가 기준이므로
    애초에 오늘 파일이 필요 없다.
    """
    global LISTING_ASOF

    now = datetime.now(KST)
    day = now.date()
    if now.hour < LISTING_SESSION_COMPLETE_HOUR:
        day -= timedelta(days=1)

    skipped: list[str] = []
    for _ in range(max_lookback + 1):
        url = KRX_LISTING_CACHE % day.isoformat()
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            day -= timedelta(days=1)
            continue
        if response.status_code == 200:
            frame = pd.read_csv(io.StringIO(response.text), dtype={"Code": str})
            if _listing_is_usable(frame):
                LISTING_ASOF = day.isoformat()
                if skipped:
                    print(f"      상장목록: 가격 미반영 파일 건너뜀 ({', '.join(skipped)})")
                return frame, LISTING_ASOF
            skipped.append(day.isoformat())
        day -= timedelta(days=1)

    # 캐시 경로가 통째로 바뀐 경우를 대비한 최후 수단. FDR 이 소스를 옮겼다면
    # 이쪽이 살아 있을 수 있다.
    import FinanceDataReader as fdr

    frames = []
    for market in ("KOSPI", "KOSDAQ"):
        listing = fdr.StockListing(market)
        listing["Market"] = market
        frames.append(listing)
    LISTING_ASOF = None
    return pd.concat(frames, ignore_index=True), "unknown"


def _to_num(value: Any) -> float | None:
    """네이버 문자열을 float으로.

    값에 단위가 붙어서 온다 ('16.85배', '12,372원', '0.80%', '-1,500').
    앞쪽의 부호+숫자 부분만 뽑아내고 나머지는 버린다.
    '22조 578억' 같은 복합 단위는 여기서 다루지 않는다 (쓰지 않는 필드).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if text in ("", "-", "N/A"):
        return None
    match = re.match(r"^[+-]?\d*\.?\d+", text)
    return float(match.group()) if match else None


def _fin_num(value) -> float | None:
    """재무 표의 셀 값을 float으로.

    _to_num과 분리하는 이유가 둘 있다.
      1) 선행 부호를 반드시 살려야 한다. 적자 기업의 '-1,234'를 양수로 읽으면
         흑자·성장 게이트가 정반대로 작동한다.
      2) 결측 표기 '-'와 음수 부호를 구분해야 한다.
    실측 확인된 표기: 천단위 쉼표('3,336,059'), 마이너스 접두('-91,124'),
    결측 단일 하이픈('-'), 퍼센트('13.07'). 괄호음수·△는 관측되지 않았으나
    방어 코드를 둔다.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in ("", "-", "--", "N/A"):
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    text = text.replace("△", "-")
    match = re.match(r"^[+-]?\d*\.?\d+", text)
    return float(match.group()) if match else None


class NaverClient:
    """네이버 금융 모바일 API 클라이언트. 세션 재사용 + 지수 백오프 재시도."""

    def __init__(self, workers: int = 8, retries: int = 3, timeout: int = 15):
        self.workers = workers
        self.retries = retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self._industry_cache: dict[str, str | None] = {}
        self.failures: list[str] = []

    def _get(self, path: str) -> Any:
        last_error = ""
        for attempt in range(self.retries):
            try:
                response = self._session.get(
                    f"{NAVER_API}/{path}", timeout=self.timeout
                )
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:  # 네트워크/JSON 오류 모두 재시도 대상
                last_error = type(exc).__name__
            time.sleep(0.4 * (2**attempt))
        self.failures.append(f"{path} ({last_error})")
        return None

    # --- 개별 종목 ------------------------------------------------------

    def price_history(self, code: str, days: int = MAX_PAGE_SIZE) -> pd.DataFrame:
        """일별 OHLCV. 오래된 날짜가 위로 오도록 정렬해서 돌려준다."""
        rows = self._get(f"stock/{code}/price?pageSize={min(days, MAX_PAGE_SIZE)}&page=1")
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "date": [str(r.get("localTradedAt", ""))[:10] for r in rows],
                "open": [_to_num(r.get("openPrice")) for r in rows],
                "high": [_to_num(r.get("highPrice")) for r in rows],
                "low": [_to_num(r.get("lowPrice")) for r in rows],
                "close": [_to_num(r.get("closePrice")) for r in rows],
                "volume": [_to_num(r.get("accumulatedTradingVolume")) for r in rows],
            }
        )
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        return frame.sort_values("date").reset_index(drop=True)

    def investor_trend(self, code: str, days: int = 20) -> pd.DataFrame:
        """투자자별 순매수 수량 (외국인/기관/개인) + 외국인 보유비율."""
        rows = self._get(f"stock/{code}/trend?pageSize={min(days, MAX_PAGE_SIZE)}&page=1")
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "date": [str(r.get("bizdate", ""))[:10] for r in rows],
                "foreign_net": [_to_num(r.get("foreignerPureBuyQuant")) for r in rows],
                "organ_net": [_to_num(r.get("organPureBuyQuant")) for r in rows],
                "indiv_net": [_to_num(r.get("individualPureBuyQuant")) for r in rows],
                "foreign_ratio": [_to_num(r.get("foreignerHoldRatio")) for r in rows],
                "close": [_to_num(r.get("closePrice")) for r in rows],
            }
        )
        return frame.sort_values("date").reset_index(drop=True)

    def snapshot(self, code: str) -> dict:
        """밸류에이션 + 애널리스트 컨센서스. totalInfos는 key/value 리스트로 온다."""
        payload = self._get(f"stock/{code}/integration")
        if not payload:
            return {}

        infos = {
            item.get("code"): item.get("value")
            for item in (payload.get("totalInfos") or [])
            if isinstance(item, dict)
        }
        consensus = payload.get("consensusInfo") or {}

        return {
            "per": _to_num(infos.get("per")),
            # cnsPer는 컨센서스 기준 추정 PER. 일중 매매라도 밸류 블록은
            # 과거 실적보다 향후 이익 전망을 보는 쪽이 맞아 함께 담아둔다.
            "forward_per": _to_num(infos.get("cnsPer")),
            "pbr": _to_num(infos.get("pbr")),
            "eps": _to_num(infos.get("eps")),
            # BPS는 ROE(=EPS/BPS) 계산용. "낙폭 과다 + 저평가"를 뽑을 때
            # 싼 이유가 실적 훼손이면 안 되므로 수익성 지표가 반드시 필요하다.
            "bps": _to_num(infos.get("bps")),
            "forward_eps": _to_num(infos.get("cnsEps")),
            "dividend_yield": _to_num(infos.get("dividendYieldRatio")),
            "foreign_rate": _to_num(infos.get("foreignRate")),
            "high_52w": _to_num(infos.get("highPriceOf52Weeks")),
            "low_52w": _to_num(infos.get("lowPriceOf52Weeks")),
            "target_price": _to_num(consensus.get("priceTargetMean")),
            "recomm_score": _to_num(consensus.get("recommMean")),
            "industry_code": payload.get("industryCode"),
        }

    def industry_name(self, industry_code: str) -> str | None:
        """업종 코드 → 업종명.

        모바일 API에는 업종명이 없고 코드('313')만 온다. 코드만 보여주면
        분산이 제대로 됐는지 사람이 확인할 수 없어서, 웹 금융의 업종 페이지
        <title>에서 이름을 가져온다. 코드 수가 적어 호출 부담은 없다.
        """
        if not industry_code:
            return None
        if industry_code in self._industry_cache:
            return self._industry_cache[industry_code]

        name = None
        try:
            response = self._session.get(
                "https://finance.naver.com/sise/sise_group_detail.naver",
                params={"type": "upjong", "no": industry_code},
                timeout=self.timeout,
            )
            response.encoding = "euc-kr"
            match = re.search(r"<title>(.*?)</title>", response.text, re.S)
            if match:
                title = match.group(1).strip()
                name = title.split(":")[0].strip() or None
        except Exception:
            name = None

        self._industry_cache[industry_code] = name
        return name


    # --- 재무제표 -------------------------------------------------------

    # 네이버 재무표의 한글 행 제목 → 내부 키. 12종목 전수 대조에서 annual과
    # quarter 모두 16개 행이 고정 스키마로 왔고, 그중 쓰는 것만 담는다.
    _FIN_ROWS = {
        "매출액": "revenue",
        "영업이익": "op",
        "ROE": "roe",
        "영업이익률": "op_margin",
        "부채비율": "debt_ratio",
    }

    @staticmethod
    def _parse_finance(payload: dict) -> dict:
        """finance/annual · finance/quarter 공통 파서.

        실측으로 확인한 함정 셋을 전부 여기서 처리한다.
          1) trTitleList의 key 순서와 rowList의 columns 키 순서가 둘 다 정렬돼
             있지 않다(삼성전자 columns 순서 = 202512, 202612, 202312, 202412).
             반드시 직접 sort한다.
          2) isConsensus는 'Y'/'N' 문자열이다. 실적과 절대 섞지 않는다.
          3) 결산월이 12월이 아닌 종목이 있다(950210 = 6월 결산). 연도 키를
             그대로 쓰지 않고 정렬된 리스트로 평탄화해 비교 가능하게 만든다.
        존재하지 않는 종목코드도 HTTP 200에 빈 financeInfo를 돌려주므로
        상태코드가 아니라 trTitleList/rowList 유무로 판정한다.
        """
        info = (payload or {}).get("financeInfo") or {}
        titles = info.get("trTitleList") or []
        rows = info.get("rowList") or []
        if not titles or not rows:
            return {}

        actual, consensus = [], []
        for title in titles:
            key = title.get("key")
            if not key:
                continue
            bucket = (consensus if str(title.get("isConsensus", "N")).upper() == "Y"
                      else actual)
            bucket.append(key)
        actual.sort()
        consensus.sort()

        by_title = {r.get("title"): (r.get("columns") or {}) for r in rows}

        def series(korean: str, keys: list[str]) -> list:
            columns = by_title.get(korean) or {}
            return [_fin_num((columns.get(k) or {}).get("value")) for k in keys]

        result = {"n_actual": len(actual), "n_consensus": len(consensus)}
        for korean, key in NaverClient._FIN_ROWS.items():
            result[f"{key}_actual"] = series(korean, actual)
            result[f"{key}_consensus"] = series(korean, consensus)
        return result

    def finance_annual(self, code: str) -> dict:
        """연간 재무제표 3개 연도 + 컨센서스 1개 연도.

        실측 커버리지: 후보풀 271종목 HTTP+파싱 성공 100.0%, 3개 연도 보유
        99.3%, 매출+영업이익 둘 다 보유 100.0%. 8스레드 271종목 0.9초.
        """
        return self._parse_finance(self._get(f"stock/{code}/finance/annual"))

    def finance_quarter(self, code: str) -> dict:
        """분기 재무제표 5개 분기 + 컨센서스 1개 분기.

        연간표만 쓰면 최신 실적이 8개월 묵는다. 분기표의 ROE 칼럼은 TTM 값이라
        훨씬 신선하고, 어느 쪽을 쓰느냐가 ROE>=5 판정을 217중 23종목(11%)에서
        뒤집는다(S-Oil 2.01→10.35, 펄어비스 -1.05→16.85). 수집 비용 +2.5초.
        """
        return self._parse_finance(self._get(f"stock/{code}/finance/quarter"))

    # --- 병렬 수집 ------------------------------------------------------

    def bulk(self, codes: list[str], method: str, **kwargs) -> dict[str, Any]:
        """여러 종목에 대해 위 메서드 중 하나를 병렬 실행."""
        fetch = getattr(self, method)

        def one(code: str):
            return code, fetch(code, **kwargs)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return dict(pool.map(one, codes))


def fetch_universe() -> pd.DataFrame:
    """코스피 + 코스닥 전 종목 스냅샷.

    보통주가 아닌 것(우선주/스팩/리츠/ETF)은 여기서 걸러낸다.
    일중 매매 대상으로 성격이 다르고, 유동성·공시 구조도 다르기 때문이다.
    """
    listing, asof = _fetch_listing()

    if "Market" in listing.columns:
        universe = listing[listing["Market"].isin(LISTING_MARKETS)].copy()
        # KOSDAQ GLOBAL 은 별도 시장 라벨이지만 매매 성격은 코스닥과 같다.
        universe["Market"] = universe["Market"].where(
            universe["Market"] != "KOSDAQ GLOBAL", "KOSDAQ"
        )
    else:
        universe = listing.copy()  # FDR 폴백 경로 (이미 시장별로 라벨돼 있다)
    universe = universe.reset_index(drop=True)
    print(f"      상장목록 기준일 {asof} ({len(universe)}종목)")

    universe = universe.rename(
        columns={
            "Code": "code",
            "Name": "name",
            "Market": "market",
            "Dept": "dept",
            "Close": "close",
            "ChagesRatio": "change_pct",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Amount": "amount",
            "Marcap": "marcap",
            "Stocks": "shares",
        }
    )
    keep = [
        "code", "name", "market", "dept", "close", "change_pct",
        "open", "high", "low", "volume", "amount", "marcap", "shares",
    ]
    universe = universe[[c for c in keep if c in universe.columns]].copy()

    # 우선주는 종목코드 끝자리가 0이 아니다. 스팩/리츠/ETF는 종목명으로 거른다.
    universe = universe[universe["code"].str.match(r"^\d{6}$", na=False)]
    universe = universe[universe["code"].str.endswith("0")]
    drop_pattern = r"스팩|리츠|ETN|ETF|인프라|우선주|배당우|사모|유동화"
    universe = universe[~universe["name"].str.contains(drop_pattern, na=False)]

    for column in ("close", "amount", "marcap", "volume", "change_pct"):
        if column in universe.columns:
            universe[column] = pd.to_numeric(universe[column], errors="coerce")

    return universe.dropna(subset=["close", "amount", "marcap"]).reset_index(drop=True)


def fetch_macro(client: NaverClient) -> dict:
    """시장 국면 판단용 매크로 지표.

    국내 지수는 네이버, 해외 지수/환율은 FinanceDataReader를 쓴다.
    어느 하나가 실패해도 나머지로 국면 판단은 가능하도록 개별 try로 감싼다.
    """
    macro: dict[str, Any] = {"domestic": {}, "global": {}, "errors": []}

    for index_code, label in (("KOSPI", "코스피"), ("KOSDAQ", "코스닥")):
        payload = client._get(f"index/{index_code}/basic")
        if not payload:
            macro["errors"].append(f"{label} 지수")
            continue
        macro["domestic"][index_code] = {
            "label": label,
            "close": _to_num(payload.get("closePrice")),
            "change_pct": _to_num(payload.get("fluctuationsRatio")),
        }

    # 지수 20일 추세: 종가가 20일 이동평균 위인지로 위험선호 판단
    for index_code in ("KOSPI", "KOSDAQ"):
        rows = client._get(f"index/{index_code}/price?pageSize=30&page=1")
        if not rows or index_code not in macro["domestic"]:
            continue
        closes = pd.Series(
            [_to_num(r.get("closePrice")) for r in rows]
        ).dropna().iloc[::-1].reset_index(drop=True)
        if len(closes) >= 20:
            ma20 = closes.tail(20).mean()
            macro["domestic"][index_code]["ma20"] = round(float(ma20), 2)
            macro["domestic"][index_code]["above_ma20"] = bool(closes.iloc[-1] > ma20)
            macro["domestic"][index_code]["ma20_gap_pct"] = round(
                float((closes.iloc[-1] / ma20 - 1) * 100), 2
            )

    try:
        import FinanceDataReader as fdr

        # 국내 지수는 위에서 네이버로 받으므로 여기서는 해외만.
        # 코스피 야간 흐름을 선행하는 지표들 — 미국 지수, 환율, 변동성.
        for symbol, label in (
            ("US500", "S&P 500"),
            ("IXIC", "나스닥"),
            ("USD/KRW", "원/달러"),
            ("VIX", "VIX"),
        ):
            try:
                # 환율 시계열은 휴장일이 NaN으로 채워져 오는 경우가 있어
                # 결측을 먼저 털어내야 전일 대비가 NaN이 되지 않는다.
                closes = fdr.DataReader(symbol).tail(40)["Close"].dropna()
                if closes.empty:
                    continue
                last = float(closes.iloc[-1])
                prev = float(closes.iloc[-2]) if len(closes) > 1 else last
                ma20 = float(closes.tail(20).mean())
                macro["global"][symbol] = {
                    "label": label,
                    "close": round(last, 2),
                    "change_pct": round((last / prev - 1) * 100, 2) if prev else 0.0,
                    "above_ma20": bool(last > ma20),
                }
            except Exception:
                macro["errors"].append(label)
    except Exception as exc:
        macro["errors"].append(f"FDR 해외지표 ({type(exc).__name__})")

    return macro
