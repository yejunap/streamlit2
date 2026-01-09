"""
Polymarket CLOB "risk-free set" opportunity scanner
- Looks for markets where buying ALL outcomes (a "set") costs < $1 per set.
- Uses:
  - Gamma API to list markets
  - CLOB /books (batch) to fetch orderbooks

pip install requests
Python 3.9+
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

HEADERS = {"User-Agent": "set-arb-scanner/2.0", "Accept": "application/json"}


# -----------------------------
# Tunables (보수적으로 잡는 게 중요)
# -----------------------------
MAX_MARKETS = 500          # 훑을 시장 수
PAGE_LIMIT = 100           # Gamma 페이지 크기
QTY_PER_OUTCOME = 10.0     # outcome마다 살 수량(share). "세트" 10개 만들기
MIN_EDGE = 0.02            # 최소 기대이익($). 0.02 = 2센트 이상만 표시

FEE_BPS = 25               # (가정) 총비용에 0.25% 가산
EXTRA_COST = 0.01          # (가정) 고정비용(달러). 실행비/슬리피지/기타를 보수적으로 반영

REQUIRE_FULL_LIQUIDITY = True  # qty만큼 못 사면 제외
CHUNK_TOKENS = 200              # /books 배치 크기

# 너무 촘촘한 기회(합계 0.9999 같은)는 체결/슬리피지로 깨지기 쉬움
# 그래서 MIN_EDGE를 1~3센트 정도로 두는 게 실전적으로 낫습니다.


# -----------------------------
# HTTP helpers
# -----------------------------
def get_json(url: str, params: Optional[dict] = None, timeout: int = 20) -> Any:
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def post_json(url: str, payload: Any, timeout: int = 20) -> Any:
    r = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def f(x: Any, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------
# Models
# -----------------------------
@dataclass
class OutcomeFill:
    token_id: str
    avg_price: float
    filled: float
    notional: float

@dataclass
class SetOpp:
    market_id: str
    slug: str
    question: str
    total_cost: float
    payout: float
    edge: float
    fills: List[OutcomeFill]
    ts: float


# -----------------------------
# Orderbook fill math
# -----------------------------
def weighted_fill_from_asks(asks: List[Dict[str, Any]], qty: float) -> Tuple[float, float, float]:
    """
    Fill qty shares by walking asks from best price upward.
    Returns (avg_price, filled_qty, notional).
    """
    if qty <= 0:
        return math.nan, 0.0, 0.0

    levels: List[Tuple[float, float]] = []
    for lvl in asks or []:
        p = f(lvl.get("price"))
        s = f(lvl.get("size"))
        if math.isfinite(p) and math.isfinite(s) and p > 0 and s > 0:
            levels.append((p, s))
    levels.sort(key=lambda x: x[0])

    remaining = qty
    notional = 0.0
    filled = 0.0

    for price, size in levels:
        if remaining <= 1e-12:
            break
        take = min(size, remaining)
        notional += take * price
        filled += take
        remaining -= take

    avg = notional / filled if filled > 0 else math.nan
    return avg, filled, notional


# -----------------------------
# Gamma: markets
# -----------------------------
def fetch_markets(max_markets: int, page_limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0

    while len(out) < max_markets:
        limit = min(page_limit, max_markets - len(out))
        params = {
            "limit": limit,
            "offset": offset,
            "active": "true",
            "closed": "false",
        }
        data = get_json(f"{GAMMA_BASE}/markets", params=params)

        if isinstance(data, dict) and isinstance(data.get("data"), list):
            batch = data["data"]
        elif isinstance(data, list):
            batch = data
        else:
            break

        if not batch:
            break

        out.extend(batch)
        offset += len(batch)
        time.sleep(0.05)

    return out[:max_markets]


def extract_tokens(m: Dict[str, Any]) -> List[str]:
    ids = m.get("clobTokenIds")
    if isinstance(ids, list):
        return [str(x) for x in ids if x is not None]
    return []


# -----------------------------
# CLOB: batch orderbooks
# -----------------------------
def fetch_books(token_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not token_ids:
        return {}
    payload = [{"token_id": tid} for tid in token_ids]
    data = post_json(f"{CLOB_BASE}/books", payload)

    books: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for b in data:
            asset_id = str(b.get("asset_id") or "")
            if asset_id:
                books[asset_id] = b
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                books[str(k)] = v
    return books


# -----------------------------
# Scanner
# -----------------------------
def scan_risk_free_sets(
    qty_per_outcome: float,
    min_edge: float,
    fee_bps: int,
    extra_cost: float,
    max_markets: int,
) -> List[SetOpp]:
    markets = fetch_markets(max_markets=max_markets, page_limit=PAGE_LIMIT)

    rows: List[Tuple[str, str, str, List[str]]] = []
    all_tokens: List[str] = []

    for m in markets:
        mid = str(m.get("id") or "")
        slug = str(m.get("slug") or "")
        question = str(m.get("question") or m.get("title") or "")

        token_ids = extract_tokens(m)
        if not mid or len(token_ids) < 2:
            continue

        rows.append((mid, slug, question, token_ids))
        all_tokens.extend(token_ids)

    # Pull books in chunks
    books: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(all_tokens), CHUNK_TOKENS):
        chunk = all_tokens[i:i + CHUNK_TOKENS]
        books.update(fetch_books(chunk))
        time.sleep(0.05)

    fee_mult = 1.0 + fee_bps / 10_000.0
    payout = 1.0 * qty_per_outcome  # set qty 만큼이면 어떤 결과든 $1 * qty 정산 가정

    opps: List[SetOpp] = []

    for mid, slug, question, token_ids in rows:
        fills: List[OutcomeFill] = []
        total = 0.0
        ok = True

        for tid in token_ids:
            book = books.get(tid)
            if not book:
                ok = False
                break

            asks = book.get("asks") or []
            avg, filled, notional = weighted_fill_from_asks(asks, qty_per_outcome)

            if not math.isfinite(avg) or filled <= 0:
                ok = False
                break

            if REQUIRE_FULL_LIQUIDITY and filled + 1e-9 < qty_per_outcome:
                ok = False
                break

            fills.append(OutcomeFill(token_id=tid, avg_price=avg, filled=filled, notional=notional))
            total += notional

        if not ok:
            continue

        total_adj = total * fee_mult + extra_cost
        edge = payout - total_adj

        if edge >= min_edge:
            opps.append(SetOpp(
                market_id=mid,
                slug=slug,
                question=question,
                total_cost=total_adj,
                payout=payout,
                edge=edge,
                fills=fills,
                ts=time.time(),
            ))

    opps.sort(key=lambda x: x.edge, reverse=True)
    return opps


def main():
    opps = scan_risk_free_sets(
        qty_per_outcome=QTY_PER_OUTCOME,
        min_edge=MIN_EDGE,
        fee_bps=FEE_BPS,
        extra_cost=EXTRA_COST,
        max_markets=MAX_MARKETS,
    )

    print(f"Found {len(opps)} opportunities (qty={QTY_PER_OUTCOME}, min_edge=${MIN_EDGE:.3f})\n")

    for o in opps[:30]:
        print("=" * 90)
        print(f"Market {o.market_id} | {o.slug}")
        print(f"Q: {o.question}")
        print(f"Cost: ${o.total_cost:.6f}  Payout: ${o.payout:.6f}  Edge: ${o.edge:.6f}")
        for ff in o.fills:
            print(f"  - token {ff.token_id}: avg={ff.avg_price:.6f} notional={ff.notional:.6f} filled={ff.filled:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
