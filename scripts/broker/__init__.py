"""증권사 주문 API 어댑터.

현재는 한국투자증권(KIS)만 있다. 다른 증권사를 붙일 일이 생기면
KisClient 와 같은 메서드 이름(price/balance/orderable_cash/order/cancel/
executions)을 갖는 클래스를 여기 추가하고 trade.py 의 클라이언트 생성부만
바꾸면 된다. trade.py 는 증권사별 tr_id 나 필드명을 알지 못한다.
"""
