"""매도창 ↔ 매도창 (패턴 C/D) 목록 — 화면 없이 돈다.

Jita 매도창과 허브(Amarr·Dodixie·Rens) 매도창을 맞물게 돌려
**싸게 사서 비싸 파는** 종목만 적는다.

  C = 허브 매도창에서 사서 → Jita 매도창에 팔기
  D = Jita 매도창에서 사서 → 허브 매도창에 팔기

실행:
  python3 arb_list.py                          # 세 쌍 다 (주문판은 ETag 캐시를 맞는다)
  python3 arb_list.py --hubs Amarr Rens         # 짝을 고르기
  python3 arb_list.py --min-net-pct 1 --max-cand 800
  python3 arb_list.py --listing                 # 팔기를 지정가(브로커 수수료 포함)로 가정
  python3 arb_list.py --hist                     # 일평균 거래량도 붙인다 (더 느다)

주의 — 두 편 다 **물려 있는 매도호가**를 세는 계산이라, 그 가격에 살 손님이 있다는 보장이
없다. 파는 창이 얇고 그 자리에 손님이 없으면 팔려지지도 없이 버티는 물량이 된다. 그러므로
일평균 거래량(daily_volume)을 꼭 함께 보고 걸러야 한다.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import esi
from arb_core import (PRESETS, SORTS, analyze_type, candidate_scores, is_blocked,
                      load_blocklist, sort_rows)

BLOCK_FILE = os.path.join(_HERE, "blocklist.json")


def fetch_books(hubs: list[str], workers: int, log) -> dict:
    books: dict[str, dict] = {}
    for h in hubs:
        hb = esi.HUBS[h]
        log(f"{h} 주문판 받는 중…")
        books[h] = esi.fetch_hub_orders(hb["region"], hb["station"], workers=workers, log=log)
        log(f"{h}: {len(books[h])}종목")
    return books


def pick_candidates(books: dict, hubs: list[str], max_cand: int, log) -> list[int]:
    """쌍별로 조마진 금액 상위 max_cand를 모은다 — A/B가 아무것도 아닌 종목도 C/D면 산다."""
    keep: list[int] = []
    for h in hubs:
        sc = candidate_scores(books["Jita"], books[h])
        top = sorted(((s, t) for t, s in sc.items()), reverse=True)[:max_cand]
        keep += [t for _, t in top]
        log(f"Jita↔{h}: 조마진 나는 종목 {len(sc)} → 상위 {len(top)}")
    return list(dict.fromkeys(keep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubs", nargs="*", default=list(esi.SECONDARIES),
                    choices=list(esi.SECONDARIES), help="Jita와 짝할 허브")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cand", type=int, default=1500, help="쌍별 후보 상한")
    ap.add_argument("--preset", default="Accounting5 (세 3.4%)", choices=list(PRESETS))
    ap.add_argument("--listing", action="store_true",
                    help="팔기를 지정가로 가정 — 브로커 수수료를 얹는다")
    ap.add_argument("--hist", action="store_true", help="일평균 거래량(ESI history)도 붙인다")
    ap.add_argument("--hist-days", type=int, default=10,
                    help="거래량을 며칠 새 평균으로 재나 (기본 10일)")
    ap.add_argument("--min-net-pct", type=float, default=0.5)
    ap.add_argument("--max-net-pct", type=float, default=100.0,
                    help="이 율 이상 나는 행은 접어둔다 — 창에 박힌 안 팔리는 호가라 본다")
    ap.add_argument("--min-qty", type=float, default=1, help="최소 체결량(개)")
    ap.add_argument("--min-isk", type=float, default=1_000_000, help="최소 순이익(ISK)")
    ap.add_argument("--max-queue-days", type=float, default=0.0,
                    help="셀오더 총량 / 하루 평균 거래가 이 값 이하만 남긴다 (0은 끄기)")
    ap.add_argument("--broker-cd", type=float, default=4.0,
                    help="C/D 매지정 브로커 수수료 %% — 파는 창에 이름을 걸니 무조건 물린다 (기본 4)")
    ap.add_argument("--profit-tol", type=float, default=0.0,
                    help="이 비율 안쪽인 순이익은 같은 것으로 보고 다음 자리로 가린다 (예 10 = 10%%)")
    ap.add_argument("--sort", default=next(iter(SORTS)), choices=list(SORTS),
                    help="표에 올리는 순서")
    ap.add_argument("--cargo-m3", type=float, default=534_000)
    ap.add_argument("--only", default="C,D", help="붙잡을 패턴 (기본 C,D)")
    ap.add_argument("--out", default=None, help="CSV 받을 곳 (기본 /tmp/arb_cd)")
    args = ap.parse_args()

    out_csv = args.out or f"/tmp/arb_cd_{datetime.now():%m%d_%H%M}.csv"
    log_lines: list[str] = []

    def log(m: str) -> None:
        print(m, flush=True)
        log_lines.append(m)

    hubs = ["Jita"] + list(args.hubs)
    books = fetch_books(hubs, args.workers, log)
    tids = pick_candidates(books, list(args.hubs), args.max_cand, log)
    log(f"후보 {len(tids)}종목 — 부피·이름을 지킨다")

    meta = esi.fetch_type_meta(tids, progress=None)
    hist: dict[int, float] = {}
    if args.hist:
        log(f"일평균 거래량 — {args.hist_days}일 평균)")
        hist = esi.fetch_histories(esi.JITA_REGION, tids, days=args.hist_days)

    blocked = load_blocklist(BLOCK_FILE)
    fees = PRESETS[args.preset]
    sell_fee = fees["sell_tax"] if not args.listing else fees["sell_tax"] + fees["broker_sell"]
    # C/D는 파는 편이 늘 매도 지정가 — 브로커 수수료를 무조건 물린다 (기본 4%)
    sell_fee_cd = fees["sell_tax"] + args.broker_cd / 100
    log(f"수수료 — {args.preset}: 사기 {fees['buy']:.1%} / 팔기 {sell_fee:.1%} "
        f"/ C/D 팔기 {sell_fee_cd:.1%}(브로커 {args.broker_cd:.1f}% 포함)")

    want = {p.strip() for p in args.only.split(",") if p.strip()}
    rows: list[dict] = []
    for tid in tids:
        m = meta.get(tid)
        if not m or not m.get("volume"):
            continue
        if is_blocked(tid, m["name"], blocked):
            continue
        for h in args.hubs:
            b_hub = books[h].get(tid, {"sells": [], "buys": []})
            a_hub = books["Jita"].get(tid, {"sells": [], "buys": []})
            if not b_hub["sells"] or not a_hub["sells"]:
                continue
            rows += analyze_type(tid, m["name"], m["volume"], a_hub, b_hub,
                                 fees["buy"], sell_fee, hist.get(tid, 0.0),
                                 args.cargo_m3, hub_b=h, sell_fee_cd=sell_fee_cd)
    rows = [r for r in rows if r["pattern"] in want]
    keep_rows = [r for r in rows
                 if args.min_net_pct <= r["net_pct"] <= args.max_net_pct
                 and r["profit"] >= args.min_isk and r["qty"] >= args.min_qty]
    if args.max_queue_days > 0:      # 셀오더 대비 거래량이 걸은 것만 남긴다
        keep_rows = [r for r in keep_rows if r["sell_queue_days"] is not None
                     and r["sell_queue_days"] <= args.max_queue_days]
    keep_rows = sort_rows(keep_rows, args.sort, args.profit_tol / 100)
    junk = sum(1 for r in rows if r["net_pct"] > args.max_net_pct)
    log(f"매도↔매도 {len(rows)}행 → 조건(순익 {args.min_net_pct}~{args.max_net_pct}% · "
        f"{args.min_isk:,.0f}ISK · {args.min_qty:,.0f}개 · 셀오더/거래량 ≤{args.max_queue_days:g}일 "
        f"또는 모름) {len(keep_rows)}행 (조마진 {args.max_net_pct:.0f}%↑ 거름 {junk}행)")

    cols = ["item", "pair", "pattern", "direction", "buy_avg", "sell_avg", "gross_pct",
            "net_pct", "qty", "profit", "capital", "volume_m3", "isk_per_m3",
            "trips", "profit_per_trip", "daily_volume", "sell_queue_qty", "sell_queue_days",
            "days_to_sell", "type_id"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep_rows)
    log(f"CSV → {out_csv}")

    def fmt(x, n=0):
        return "-" if x is None else f"{x:,.{n}f}"

    print(f"\n정렬: {args.sort or next(iter(SORTS))} — {args.hist_days}일 평균 거래량 기준")
    print(f"\n{'종목':<28} {'쌍':<14} {'패턴':<3} {'매집가':>10} {'청산가':>10} "
          f"{'순이%':>6} {'체결량':>10} {'순이익':>14} {'셀오더':>9} {'/거래량':>8}")
    for r in keep_rows[:40]:
        print(f"{r['item'][:27]:<28} {r['pair']:<14} {r['pattern']:<3} "
              f"{fmt(r['buy_avg'], 1):>10} {fmt(r['sell_avg'], 1):>10} "
              f"{fmt(r['net_pct'], 1):>6} {fmt(r['qty']):>10} {fmt(r['profit']):>14} "
              f"{fmt(r['sell_queue_qty']):>9} {fmt(r['sell_queue_days'], 1):>8}")
    print(f"\n합계 {len(keep_rows)}행 — 순이익 다더하기 "
          f"{sum(r['profit'] for r in keep_rows):,.0f} ISK")
    by_pair: dict[str, list] = {}
    for r in keep_rows:
        by_pair.setdefault(r["pattern"] + " " + r["pair"], []).append(r)
    for k, v in sorted(by_pair.items(), key=lambda kv: -sum(r["profit"] for r in kv[1])):
        print(f"  {k:<28} {len(v):>4}행  {sum(r['profit'] for r in v):>16,.0f} ISK")


if __name__ == "__main__":
    main()
