"""아비 계산 커널 (streamlit non-dependant).

패턴 4 가지를 계산한다.  (a_book = Jita, b_book = 두 번째 허브)
  A  Amarr sellers → Jita buyers      (운송 O)
  B  Jita  sellers → Amarr buyers     (운송 O)
  C  Amarr sellers → Jita sellers     (운송 O, 매도창 ↔ 매도창)
  D  Jita  sellers → Amarr sellers    (운송 O, 매도창 ↔ 매도창)

C/D는 한 스테이션 안에서 매수호가가 매도호가를 뒤집는 일(예전의 C/D)이 아니라,
한 허브의 매도호가에 사서 다른 허브의 매도호가에 파는 차익을 센다. 네 패턴 다 화물을 싣고 간다.
"""
from __future__ import annotations

import json
import math

from datetime import datetime, timezone

# ---- 🚫 낚시물 차단 -------------------------------------------------------------------------------
# 악성 판매자가 걸어놓은 낀(new)-딜 물건은 사면 안 된다. 판별은 사람 몫 — 여기는 보관과 판정뿐.
# 기준 목록은 저장소의 blocklist.json (다시 켜도 살아남는 유일한 곳), 그날의 추가는 사이드에서.
def load_blocklist(path: str) -> dict:
    """저장소의 blocklist.json — {"ids": [int…], "names": ["이름"…]} 두 칸만 본다."""
    ids, names = set(), set()
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        ids = {int(x) for x in raw.get("ids", [])}
        names = {str(x).strip().lower() for x in raw.get("names", []) if str(x).strip()}
    except FileNotFoundError:
        pass
    return {"ids": ids, "names": names}


def parse_block_add(text: str) -> tuple[set, set]:
    """쉼터 치기 — 숫자는 타입 id, 나머지는 이름(대소문자 안 가린다)."""
    ids, names = set(), set()
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        if part.isdigit():
            ids.add(int(part))
        else:
            names.add(part.lower())
    return ids, names


def is_blocked(tid, name: str, bl: dict) -> bool:
    return tid in bl["ids"] or (name or "").strip().lower() in bl["names"]


def commit_blocklist(content: str, token: str, repo: str,
                     path: str = "eve_arb_dashboard/blocklist.json") -> str:
    """저장소에 새 기준을 알린다 — GitHub Contents API PUT 한 발. 한 줄 메시지를 준다."""
    import base64
    import requests
    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    hdrs = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        g = requests.get(api, headers=hdrs, timeout=15)
        sha = g.json().get("sha") if g.status_code == 200 else None
        body = {"message": "blocklist: 저장", "content": base64.b64encode(content.encode()).decode()}
        if sha:
            body["sha"] = sha
        r = requests.put(api, headers=hdrs, json=body, timeout=15)
        return "깃에 올림 — 다시 펴도 그대로" if r.status_code in (200, 201) \
            else f"깃 HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return f"깃 고장 — {e}"


# ---- 🧹 장바구니 비우기 --------------------------------------------------------------------------
# 장바구니(pinned)와 표의 체크는 저장소가 두 곳이다: 체크는 st.dataframe 위젯 상태(tbl_ 키)에 산다.
# pinned만 비우면 화면의 체크는 그대로 남고, 더 나쁘게 그 다음 체크 소리에
# 찍혀 있던 묵은 체크가 빈 장바구니로 되돌아온다. 그래서 비울 때 체크도 함께 비운다.
def clear_basket_state(ss) -> None:
    """pinned을 비우고 tbl_ 표의 체크를 모두 푼다 — 버튼 콜백에서 부른다."""
    ss["pinned"] = set()
    for k in list(ss):
        if k.startswith("tbl_"):
            ss[k] = {"selection": {"rows": []}}
        elif k.startswith(("seeded_", "selhit_")):
            del ss[k]


# 수수료 (% 비례). 프리셋 값:
#   buy          — 살 때 원금에 얹히는 요율 (브로커+세 합산)
#   sell_tax     — 팔 때 판매세 (시장가/지정가 무관 발생)
#   broker_sell  — 팔 때 브로커 수수료. **시장가(즉시판매)면 안 붙는다** — 앱에서 토글
# 수수료 (% 비례). 프리셋 값 (2026-08 EVE University Wiki 기준):
#   sell_tax    — 판매세. 팔면 무조건 발생. 2025-03 패치로 기본 7.5%,
#                 Accounting Lv5면 3.375% (레벨당 -11%)
#   broker_sell — 브로커 수수료. **'immediate'가 아닌 지정가를 걸 때만** 내는 돈
#                 (NPC 스테이션 기본 3%, 스킬/평판으로 ~1.9%까지) — 즉시판매엔 없음.
#                 단 **C/D는 예외** — 파는 편이 늘 매도 지정가이므로 브로커 수수료가 있다
#                 (요즘 장당 4%. 사이드바 「C/D 매도 브로커 수수료」, --broker-cd로 고친다)
#   buy         — 살 때 원금에 얹히는 요율. 즉시구매(매도호가 소진)면 0이 맞고,
#                 매수지정가를 걸어두면 그 물량 기준의 브로커 수수료가 생긴다
PRESETS = {
    "Accounting5 (세 3.4%)": {"buy": 0.0, "sell_tax": 0.03375, "broker_sell": 0.0188},
    "무스킬 (세 7.5% / 브로커 3%)": {"buy": 0.03, "sell_tax": 0.075, "broker_sell": 0.03},
    "검토 (수수료 0% — 호가 역전만)": {"buy": 0.0, "sell_tax": 0.0, "broker_sell": 0.0},
}


def patterns_for(hub_b: str) -> dict:
    """두 번째 허브 이름에 맞춘 패턴 라벨 — 쌍이 바뀌면 라벨만 달라진다.

    A/B 는 사고파는 편이 뒤집힌 평범한 운송 아비요,
    C/D 는 **두 허브의 매도창끼리** 민다 — 한 데서 사서 다른 데 매도창에 팝니다.
    그래서 네 패턴 모두 화물을 싣고 간다 (부피·ISK/m³ 유효).
    """
    return {
        "A": {"label": f"{hub_b}→Jita", "transport": True},
        "B": {"label": f"Jita→{hub_b}", "transport": True},
        # C: {hub_b} 매도창에서 사서 → Jita 매도창에 팔기
        "C": {"label": f"{hub_b}→Jita (매도→매도)", "transport": True},
        # D: Jita 매도창에서 사서 → {hub_b} 매도창에 팔기
        "D": {"label": f"Jita→{hub_b} (매도→매도)", "transport": True},
    }


# 하위 호환 — single-hub 스크립트(archive의 census 등)는 Amarr 쌍을 쓴다.
PATTERNS = patterns_for("Amarr")


def _now():
    return datetime.now(timezone.utc)


def side_isk(side) -> float:
    return sum(p * q for p, q, _ in side)


def depth_score(book: dict) -> float:
    """한 스테이션에서의 창 두께(ISK) — 적게 얇은 쪽을 따른다."""
    if not book or not book.get("sells") or not book.get("buys"):
        return 0.0
    return min(side_isk(book["sells"]), side_isk(book["buys"]))


def candidate_scores(books_a: dict, books_b: dict) -> dict[int, float]:
    """교차 종목의 후보 점수 = 네 방향(A/B/C/D)에 벌 수 있는 최고 조마진 금액(ISK).

    예전엔 창 두께를 점수로 썼는데, 그 순서가 시장과 정반다이므로
    조마진이 큰 것을 자른다. 거래량이 큰 것이 이득의 순이라 금액으로 한다.
    C/D(매도창 ↔ 매도창)도 한 갈퀴 돈다 — 한 데 매도창이 다른 데보다 싼 종목은
    A/B가 아무것도 안 내는데도 이득이 나므로 후보에서 새면 안 된다.
    """
    out = {}
    for t in set(books_a) & set(books_b):
        best = 0.0
        for asks, bids, walker in (
            (books_b[t].get("sells", []), books_a[t].get("buys", []), match_ladder),        # A
            (books_a[t].get("sells", []), books_b[t].get("buys", []), match_ladder),        # B
            (books_b[t].get("sells", []), books_a[t].get("sells", []), match_sell_ladder),  # C
            (books_a[t].get("sells", []), books_b[t].get("sells", []), match_sell_ladder),  # D
        ):
            best = max(best, walker(asks, bids, 0.0, 0.0)["profit"])
        if best > 0:
            out[t] = best
    return out


def internal_inversions(books: dict) -> set[int]:
    """한 스테이션 안에서 사자 > 팔기인 종목 — C/D 패턴은 교차 종목이 아니어도 산다."""
    return {t for t, b in books.items()
            if b.get("sells") and b.get("buys") and b["buys"][0][0] > b["sells"][0][0]}


def match_ladder(asks: list, bids: list, buy_fee: float, sell_fee: float) -> dict:
    """매도 호가와 매수 호가를 한 칸씩 채운다.

    각 칸의 한계이익은 감소한다며 단조로우므로, 한계이익이 0 이하는 바로 멈춘다.
    """
    ai = bi = aq = bq = 0
    qty = cost = rev = 0.0
    n_asks, n_bids = len(asks), len(bids)
    while ai < n_asks and bi < n_bids:
        pa, qa, _ = asks[ai]
        pb, qb, _ = bids[bi]
        if pb * (1 - sell_fee) <= pa * (1 + buy_fee):
            break
        take = min(qa - aq, qb - bq)
        if take <= 0:
            if aq >= qa:
                ai += 1
                aq = 0
            if bq >= qb:
                bi += 1
                bq = 0
            continue
        qty += take
        cost += take * pa
        rev += take * pb
        aq += take
        bq += take
        if aq >= qa:
            ai += 1
            aq = 0
        if bq >= qb:
            bi += 1
            bq = 0

    if qty <= 0:
        return {"qty": 0.0, "buy_avg": 0.0, "sell_avg": 0.0, "cost": 0.0,
                "revenue": 0.0, "profit": 0.0, "net_pct": 0.0, "gross_pct": 0.0}

    buy_avg, sell_avg = cost / qty, rev / qty
    net_rev, net_cost = rev * (1 - sell_fee), cost * (1 + buy_fee)
    gross = (sell_avg / buy_avg - 1) * 100 if buy_avg else 0.0
    net = (net_rev / net_cost - 1) * 100 if net_cost else 0.0
    return {
        "qty": qty, "buy_avg": buy_avg, "sell_avg": sell_avg,
        "cost": net_cost, "revenue": net_rev, "profit": net_rev - net_cost,
        "gross_pct": gross, "net_pct": net,
    }


def match_sell_ladder(buy_asks: list, sell_asks: list,
                      buy_fee: float, sell_fee: float) -> dict:
    """매도 ↔ 매도 — 한 창에서 사서(매도호가 소진) 다른 창에 파는(매도호가 소진) 계산.

    두 편 다 **매도 호가 창**이다. 전의 C/D(한 스테이션 호가 뒤집힘)와 달라,
    창을 맞물게 기어간다 — 파는 편 창가의 한 칸은 그 칸에 물린 물량만큼만 받아 준다.
    (그 이상은 창에 없는 가격에 거는 일 — 지정가대기가 되어 계산 밖으로 버린다)
    한 칸의 한계이익이 0 이하로 떨어지면 멈춘다(호가 거리가 크지 않으므로 단조롭다).
    """
    ai = bi = aq = bq = 0
    qty = cost = rev = 0.0
    n_buy, n_sell = len(buy_asks), len(sell_asks)
    while ai < n_buy and bi < n_sell:
        pa, qa, _ = buy_asks[ai]
        ps, qs, _ = sell_asks[bi]
        if ps * (1 - sell_fee) <= pa * (1 + buy_fee):
            break
        take = min(qa - aq, qs - bq)
        if take <= 0:
            if aq >= qa:
                ai += 1
                aq = 0
            if bq >= qs:
                bi += 1
                bq = 0
            continue
        qty += take
        cost += take * pa
        rev += take * ps
        aq += take
        bq += take
        if aq >= qa:
            ai += 1
            aq = 0
        if bq >= qs:
            bi += 1
            bq = 0

    if qty <= 0:
        return {"qty": 0.0, "buy_avg": 0.0, "sell_avg": 0.0, "cost": 0.0,
                "revenue": 0.0, "profit": 0.0, "net_pct": 0.0, "gross_pct": 0.0}

    buy_avg, sell_avg = cost / qty, rev / qty
    net_rev, net_cost = rev * (1 - sell_fee), cost * (1 + buy_fee)
    gross = (sell_avg / buy_avg - 1) * 100 if buy_avg else 0.0
    net = (net_rev / net_cost - 1) * 100 if net_cost else 0.0
    return {
        "qty": qty, "buy_avg": buy_avg, "sell_avg": sell_avg,
        "cost": net_cost, "revenue": net_rev, "profit": net_rev - net_cost,
        "gross_pct": gross, "net_pct": net,
    }


def _age_h(orders) -> float | None:
    if not orders or orders[0][2] is None:
        return None
    return (_now() - orders[0][2]).total_seconds() / 3600


def analyze_type(tid: int, name: str, unit_volume: float, a_book: dict, b_book: dict,
                 buy_fee: float, sell_fee: float, daily_vol: float,
                 cargo_m3: float, hub_b: str = "Amarr",
                sell_fee_cd: float | None = None) -> list[dict]:
    """한 종목의 패턴 A/B/C/D 4 행을 만든다. hub_b = Jita와 쌍을 이루는 두 번째 허브.

    sell_fee   — A/B가 파는 편(다른 허브의 **매수창**)에 들는 비용 — 즉시 접하면 브로커 없음
    sell_fee_cd— C/D가 파는 편(**매도창에 이름을 건다**)에 들는 비용. 지정가이므로 브로커 수수료가
                 무조건 있다. 없으면(=None) sell_fee를 그대로 쓴다.
    """
    pats = patterns_for(hub_b)
    pair = f"Jita↔{hub_b}"
    fee_cd = sell_fee if sell_fee_cd is None else sell_fee_cd
    # (살 창, 팔 창, 계산기, 팔 곳에서 멀리는 창, 팔 비용)
    legs = {
        "A": (b_book.get("sells", []), a_book.get("buys", []), match_ladder, a_book, sell_fee),
        "B": (a_book.get("sells", []), b_book.get("buys", []), match_ladder, b_book, sell_fee),
        # C/D 는 매도창 ↔ 매도창 — 한 데서 사서 저잣거리의 매도창에 팔는다
        "C": (b_book.get("sells", []), a_book.get("sells", []), match_sell_ladder, a_book, fee_cd),
        "D": (a_book.get("sells", []), b_book.get("sells", []), match_sell_ladder, b_book, fee_cd),
    }
    rows = []
    for key, (asks, bids, walker, dest, fee) in legs.items():
        m = walker(asks, bids, buy_fee, fee)
        if m["qty"] <= 0:
            continue
        vol_m3 = m["qty"] * unit_volume
        # 팔 곳에 포개져 있는 매도주문 총량 — 내가 여기 섞여 있을 때 언제 내 팔물이 되는가
        sell_queue = sum(q for _, q, _ in dest.get("sells", [])) if dest else 0.0
        row = {
            "type_id": tid, "item": name, "pattern": key,
            "pair": pair, "hub": hub_b,
            "direction": pats[key]["label"],
            "transport": pats[key]["transport"],
            "buy_avg": m["buy_avg"], "sell_avg": m["sell_avg"],
            "gross_pct": m["gross_pct"], "net_pct": m["net_pct"],
            "qty": m["qty"], "capital": m["cost"], "profit": m["profit"],
            "unit_volume": unit_volume, "volume_m3": vol_m3,
            "isk_per_m3": (m["profit"] / vol_m3) if vol_m3 > 0 else None,
            "trips": (-(-vol_m3 // cargo_m3)) if vol_m3 > 0 else None,
            "profit_per_trip": (m["profit"] / max(1, -(-vol_m3 // cargo_m3))) if vol_m3 > 0 else None,
            "ask_age_h": _age_h(asks), "bid_age_h": _age_h(bids),
            "daily_volume": daily_vol,
            "days_to_sell": (m["qty"] / daily_vol) if daily_vol else None,
            # 팔 창 총량을 하루 평균 거래로 며칠이면 다 태우나 — 작으면 줄이 금 지난다.
            "sell_queue_qty": sell_queue,
            "sell_queue_days": (sell_queue / daily_vol) if daily_vol else None,
        }
        rows.append(row)
    return rows


RANKERS = {
    "ISK/m³": lambda d: d["isk_per_m3"] if d["isk_per_m3"] is not None else -1,
    "순이익 ISK": lambda d: d["profit"],
    "순이익 %": lambda d: d["net_pct"],
    "1회 운항 이익": lambda d: d["profit_per_trip"] if d["profit_per_trip"] is not None else -1,
    # 규모 자체가 큰 것도 순서대로 볼 수 있게
    "총부피 m³": lambda d: d["volume_m3"] if d["volume_m3"] is not None else -1,
    "체결량": lambda d: d["qty"],
    "셀오더/거래량": lambda d: d["sell_queue_days"] if d["sell_queue_days"] is not None else 1e18,
}

# ---- 표 기본 정렬 -----------------------------------------------------------------------------------
# ① 순이익 ② 묶인 돈 ③ 부피 ④ 셀오더 대비 거래량(10일 평균 기준)
# 앞의 것은 클수록, 뒤의 셋은 작을수록 좋다. 모르면(없으면) 맨 뒤로 미룬다.
def _worse(x):
    """정렬에서 모르는 값은 뒤로 — 작을수록 앞인 칸에서 쓴다."""
    return float("inf") if x is None else x


def _best(x):
    """정렬에서 모르는 값은 뒤로 — 내림(클수록 앞) 칸에서 쓴다."""
    return float("-inf") if x is None else x


def _profit_bucket(p: float, tol: float) -> float:
    """순이익을 tol% 간격으로 묶는다 — 한 칸이면 다음 자리(묶인 돈·부피·셀오더)로 가린다.

    간격을 0이면 있는 그대로(따닥이 같은 것만 묶음). 비율로 자르므로 큰돈과 작은 돈이 같은 칸에 안 갇힌다.
    """
    if p is None or p <= 0:
        return float("-inf")
    if tol <= 0:
        return p
    return math.floor(math.log(p) / math.log(1 + tol))


def rank_key(row: dict, tol: float = 0.0):
    """표 한 행의 순위 — 네 가지를 순서대로 재고, 숫자가 작을수록 앞이다."""
    return (
        -_profit_bucket(row["profit"], tol),   # ① 순이익은 클수록 (tol% 안쪽은 같이라 본다)
        row["capital"],                        # ② 묶인 돈은 작을수록
        _worse(row["volume_m3"]),              # ③ 부피는 작을수록
        _worse(row["sell_queue_days"]),        # ④ 셀오더 대비 거래량(10일)은 작을수록
    )


def rank_volume_key(row: dict):
    """예전 순서 — 부피 효율이 앞이다."""
    return (-_best(row["isk_per_m3"]), -row["profit"], row["capital"])


SORTS = {
    "① 순이익↓ ② 묶인 돈↓ ③ 부피↓ ④ 셀오더/거래량↓": rank_key,
    "① ISK/m³↓ ② 순이익↓ ③ 묶인 자본↓": rank_volume_key,
}


def sort_rows(rows: list[dict], key: str | None = None, tol: float = 0.0) -> list[dict]:
    """표와 리스트(arb_list)가 같은 순서를 쓴다."""
    if key not in SORTS or SORTS[key] is rank_key:
        return sorted(rows, key=lambda r: rank_key(r, tol))
    return sorted(rows, key=SORTS[key])




def filter_rows(rows: list[dict], min_net_pct: float, min_profit: float,
                min_capital: float, min_daily_volume: float,
                patterns: set[str]) -> list[dict]:
    out = []
    for r in rows:
        if r["pattern"] not in patterns:
            continue
        if r["net_pct"] < min_net_pct or r["profit"] < min_profit:
            continue
        if r["capital"] < min_capital:
            continue
        if r["daily_volume"] < min_daily_volume:
            continue
        out.append(r)
    return out


def pair_summary(rows: list[dict]) -> dict[str, dict]:
    """쌍별 총합 — 세 쌍을 한 화면에서 견주어 어느 쌍이 남는지 가린다."""
    out: dict[str, dict] = {}
    for r in rows:
        s = out.setdefault(r["pair"], {
            "행": 0, "합순이익": 0.0, "묶인자본": 0.0,
            "최고_ISK_m3": 0.0, "최고_종목": "—",
            "최고_순익": 0.0, "최고_순익종목": "—",
        })
        s["행"] += 1
        s["묶인자본"] += r["capital"]
        if (r["isk_per_m3"] or 0) > s["최고_ISK_m3"]:
            s["최고_ISK_m3"], s["최고_종목"] = r["isk_per_m3"], r["item"]
        # 네 패턴(A/B/C/D) 다 두 허브를 오간다 — C/D는 매도창 ↔ 매도창이라 같이 센다
        if r["pattern"] in ("A", "B", "C", "D"):
            s["합순이익"] += r["profit"]
            if r["profit"] > s["최고_순익"]:
                s["최고_순익"], s["최고_순익종목"] = r["profit"], r["item"]
    return out

