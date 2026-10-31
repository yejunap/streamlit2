#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8421 포털에 통합하는 앱들의 계산 백단 — 포털이 직접 그리고 포털 안에서 산다.

기존 스트림릿 앱 파일은 건드리지 않는다 — 그 애들의 계산부를 UI 시작
지점까지만 잘아 떠쓴다(앱 원본 무변경). /api/apps/<이름> 으로 온다.
"""
import os
import sys
import time

APPS_DIR = os.environ.get(
    "EVE_APPS_DIR",
    "/home/slreflex2/.cline/data/workspaces/chat/streamlit2/eve-industry/apps")

for _p in (os.path.join(APPS_DIR, "eve_arb_dashboard"),):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _q(qs, key, default=""):
    v = qs.get(key, [default])[0]
    return v if v != "" else default


class _FakeSt:
    class _Ctx:
        def __call__(self, *a, **k): return self
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __getattr__(self, k): return self

    def __getattr__(self, k):
        if k in ("cache_data", "cache_resource"):
            return lambda *a, **k: (lambda f: f)
        return self._Ctx()


def _esi():
    import esi
    return esi


class _EsiRequests:
    """requests.get(url, params=..) 생김새를 esi.get(etag·ttl 캐시)로 중계."""
    def __init__(self):
        self.esi = _esi()

    class _Resp:
        status_code = 200
        def __init__(self, body): self._b = body
        def json(self): return self._b

    def get(self, url, params=None, timeout=None, **kw):
        import hashlib
        from urllib.parse import urlencode
        path = url.split("esi.evetech.net", 1)[-1]
        if not path.startswith("/"):
            raise ValueError("ESI url 아님: %s" % url[:80])
        sig = (path + "?" + urlencode(
            sorted(((k, v) for k, v in (params or {}).items())))).encode("utf-8")
        key = "req_" + hashlib.md5(sig).hexdigest()[:20]
        body, _ = self.esi.get(path, params=params or {}, cache_key=key, ttl=900)
        return self._Resp(body)


_DEFS = {}


def _defs(app_file):
    if app_file in _DEFS:
        return _DEFS[app_file]
    path = os.path.join(APPS_DIR, app_file)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for m in ("st.title(", "\n# UI\n"):
        i = src.find(m)
        if i > 0:
            src = src[:i]
            break
    ns = {"__name__": "portal_defs", "__file__": path}
    old = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _FakeSt()
    try:
        exec(compile(src, path, "exec"), ns)
    finally:
        if old:
            sys.modules["streamlit"] = old
        else:
            sys.modules.pop("streamlit", None)
    ns["requests"] = _EsiRequests()
    _DEFS[app_file] = ns
    return ns


def _price_of(ns, tid, hub_id=None):
    kw = {"station_id": hub_id} if hub_id else {}
    try:
        return ns["get_market_price"](tid, **kw)
    except Exception:
        return None


def _num(v, nd):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


def crystal(qs):
    """크로스탈·라그 이익 (/api/apps/crystal) — 원판 계산과 동일."""
    import re
    ns = _defs("eve_crystal_profit.py")
    broker = float(_q(qs, "broker", "1.5"))
    tax = float(_q(qs, "tax", "3.4"))
    install = float(_q(qs, "install", "3"))
    sizes = set(_q(qs, "sizes", "S,M,R").split(","))
    t0 = time.time()
    material_prices = {}
    for mname, tid in ns["MATERIALS"].items():
        p = _price_of(ns, tid)
        if p:
            material_prices[mname] = p
    if not material_prices:
        return {"error": "재료 시세를 못 받았다 — ESI 확인"}
    rows = []
    for cname, cdata in ns["CRYSTALS"].items():
        info = ns["calculate_profit"](cname, cdata, material_prices)
        if not info:
            continue
        total_cost = info["material_cost"] * (1 + install / 100)
        net = info["sell_price"] * (1 - (broker + tax) / 100)
        paf = net - total_cost
        margin = (paf / total_cost * 100) if total_cost else None
        dv, sv = info["avg_daily_volume"], info["sell_volume"]
        rows.append({
            "Item": cname,
            "Material Cost (per unit)": _num(info["material_cost"], 0),
            "Sell Order Price": _num(info["sell_price"], 0),
            "Profit per unit (after fees)": _num(paf, 0),
            "Margin %": _num(margin, 2),
            "Profit (10 BPC)": _num(paf * info["output_10_bpc"], 0),
            "Avg Daily Volume (100d)": _num(dv, 0),
            "Sell Volume": _num(sv, 0),
            "Output per BPC": _num(info["output_per_bpc"], 0),
            "Days to Sell": _num(dv / sv, 2) if dv and sv else 0,
        })

    def keep(name):
        m = re.search(r" [SML]$", name)
        if m:
            return m.group()[-1] in sizes
        return "R" in sizes
    rows = [r for r in rows if keep(r["Item"])]
    # 원판 순서: Margin % 내음으로 정리해 최고를 고기고, 표는 Profit (10 BPC) 내림
    rows.sort(key=lambda r: -(r["Margin %"] or 0))
    best = rows[0] if rows else None
    rows.sort(key=lambda r: -(r["Profit (10 BPC)"] or 0))
    marg = [r["Margin %"] for r in rows if r["Margin %"] is not None]
    mat_groups = [
        ("Crystal/Ammunition Materials:", [["Morphite", "Rare Mineral"],
                                           ["Tungsten Carbide", "For Crystals"],
                                           ["Fullerides", "Base Material"],
                                           ["Crystalline Carbonide", "For Ammo"]]),
        ("R.A.M. (Robotic Assembly Modules):", [["R.A.M.- Ammunition Tech", "Ammo/Crystal"],
                                                ["R.A.M.- Electronics", "For Rigs"]]),
        ("Rig Materials (Salvage/PI):", [["Miniature Electronics", "PI Material"],
                                         ["Artificial Neural Network", "Salvage"],
                                         ["Micro Circuit", "Salvage"],
                                         ["Logic Circuit", "Salvage"]]),
        ("Shield Rig Materials:", [["R.A.M.- Shield Tech", "For Shield Rigs"],
                                   ["Power Circuit", "Salvage"],
                                   ["Enhanced Ward Console", "Salvage"]]),
        ("Armor Rig Materials:", [["R.A.M.- Armor/Hull Tech", "For Armor Rigs"],
                                  ["Nanite Compound", "Salvage"],
                                  ["Interface Circuit", "Salvage"],
                                  ["Intact Armor Plates", "Salvage"]]),
    ]
    groups = [{"title": g, "items": [[m, _num(material_prices[m]["lowest_sell"], 2), d]
                                     for m, d in ms if m in material_prices]}
              for g, ms in mat_groups]
    pos = sum(1 for r in rows if (r["Profit per unit (after fees)"] or 0) > 0)
    return {
        "cols": list(rows[0].keys()) if rows else ["Item"],
        "rows": rows,
        "mat_groups": groups,
        "summary": {
            "avg_margin": _num(sum(marg) / len(marg), 2) if marg else 0,
            "best": best["Item"] if best else "—",
            "best_margin": best["Margin %"] if best else None,
            "best_profit": best["Profit per unit (after fees)"] if best else None,
            "profitable": "%d/%d" % (pos, len(rows)),
        },
        "ms": int((time.time() - t0) * 1000),
    }


def fighter(qs):
    """전투기 이익 (/api/apps/fighter) — T1/T2 표 둘."""
    ns = _defs("eve_fighter_profit.py")
    hubs = ns["TRADE_HUBS"]
    hub = _q(qs, "hub", "Jita")
    if hub not in hubs:
        return {"error": "모르는 허브: %s" % hub}
    hub_id = hubs[hub]
    fees = (float(_q(qs, "broker", "1.5")), float(_q(qs, "tax", "3.4")))
    install_rate = float(_q(qs, "install", "3")) / 100
    cats = set(_q(qs, "cats", "Light,Support,Heavy").split(","))
    t0 = time.time()
    mp = {}
    for name, tid in {**ns["MINERALS"], **ns["COMPONENTS"]}.items():
        p = _price_of(ns, tid, hub_id)
        if p:
            mp[name] = p
    t1_prices, t2_prices = {}, {}
    for name, data in ns["T1_FIGHTERS"].items():
        p = _price_of(ns, data["type_id"], hub_id)
        if p:
            mp[name] = t1_prices[name] = p
    for name, data in ns["T2_FIGHTERS"].items():
        p = _price_of(ns, data["type_id"], hub_id)
        if p:
            t2_prices[name] = p
    mineral_prices = {m: p for m, p in mp.items() if m in ns["MINERALS"]}
    component_prices = {c: p for c, p in mp.items() if c in ns["COMPONENTS"]}
    t1_cost_by_mfg = {}
    for t1_name in ns["T1_FIGHTERS"]:
        cost, _unused = ns["t1_manufacturing_cost"](t1_name, mineral_prices, install_rate)
        if cost is not None:
            t1_cost_by_mfg[t1_name] = cost

    def add_vol(row):
        try:
            v = ns["get_market_history"](row["type_id"])
        except Exception:
            v = 0
        row["일평균거래량(100일)"] = _num(v, 0)
        row["소요일"] = _num(row["sell_volume"] / v, 3) if v else 0
        return row

    def mats_list(row):
        return [[m, _num(b["price"], 2), b["qty"], _num(b["total"], 0)]
                for m, b in (row.get("breakdown") or {}).items()]

    t1_rows = []
    for t1_name in ns["T1_FIGHTERS"]:
        row = ns["calc_t1_profit"](t1_name, mineral_prices, t1_prices, fees, install_rate)
        if row:
            add_vol(row)
            t1_rows.append({
                "전투기": row["fighter"], "처방": row.get("recipe", ""),
                "제조원가": _num(row["material_cost"], 0),
                "판매호가": _num(row["sell_price"], 0),
                "단위이익": _num(row["profit"], 0),
                "마진(%)": _num(row["margin"], 2),
                "ISK/시간": _num(row["isk_per_hr"], 0),
                "BPC이익(50기)": _num(row["per_bpc"], 0),
                "일평균거래량(100일)": row["일평균거래량(100일)"],
                "소요일": row["소요일"],
                "breakdown": mats_list(row)})
    t1_rows.sort(key=lambda r: -(r["단위이익"] or 0))

    t2_rows = []
    for name, fdata in ns["T2_FIGHTERS"].items():
        if fdata["category"] not in cats or fdata.get("materials") is None:
            continue          # Heavy 실측치는 스트림릿 앱에 그대로
        row = ns["calc_t2_profit"](name, fdata, mineral_prices, component_prices,
                                   t2_prices, t1_prices, t1_cost_by_mfg, fees,
                                   install_rate)
        if not row:
            continue
        add_vol(row)
        t2_rows.append({
            "T2 전투기": row["fighter"], "T1": row.get("t1", ""), "분류": row["category"],
            "원가A": _num(row["cost_A"], 0), "원가B": _num(row["cost_B"], 0),
            "판매호가": _num(row["sell_price"], 0),
            "T1 시가": _num(row["t1_market_price"], 0),
            "T1 제조원가": _num(row["t1_mfg_cost"], 0),
            "단위이익A": _num(row["profit_A"], 0),
            "단위이익B": _num(row["profit_B"], 0),
            "마진A(%)": _num(row["margin_A"], 2),
            "마진B(%)": _num(row["margin_B"], 2),
            "ISK/h A": _num(row["isk_hr_A"], 0), "ISK/h B": _num(row["isk_hr_B"], 0),
            "BPC이익 A(10기)": _num(row["per_bpc_A"], 0),
            "BPC이익 B(10기)": _num(row["per_bpc_B"], 0),
            "최적": "B(T1제조)" if row["profit_B"] > row["profit_A"] else "A(T1구매)",
            "일평균거래량(100일)": row["일평균거래량(100일)"],
            "소요일": row["소요일"],
            "breakdown": [[m, _num(b["price"], 2), b["qty"], _num(b["total"], 0)]
                          for m, b in (row.get("breakdown") or {}).items()]})
    t2_rows.sort(key=lambda r: -(r["단위이익B"] or 0))
    best = t2_rows[0] if t2_rows else None
    return {
        "hub": hub,
        "n_prices": len(mp) + len(t1_prices) + len(t2_prices),
        "t2_summary": {
            "best": best["T2 전투기"] if best else "—",
            "best_profit": best["단위이익B"] if best else None,
            "best_margin": best["마진B(%)"] if best else None,
            "profitable": "%d/%d" % (
                sum(1 for r in t2_rows if (r["단위이익A"] or 0) > 0 or (r["단위이익B"] or 0) > 0),
                len(t2_rows)),
            "t1_better": "%d/%d" % (
                sum(1 for r in t2_rows if (r["단위이익B"] or 0) > (r["단위이익A"] or 0)),
                len(t2_rows)),
        },
        "t1": {"cols": list(t1_rows[0].keys()) if t1_rows else [], "rows": t1_rows},
        "t2": {"cols": list(t2_rows[0].keys()) if t2_rows else [], "rows": t2_rows},
        "ms": int((time.time() - t0) * 1000),
    }


def arb(qs):
    """어비 (/api/apps/arb) — app.py run_scan/enrich/build_rows를 예산가 없이 옮긴다."""
    import csv as _csv
    from datetime import datetime, timezone
    import esi
    import arb_core as AC

    log = []
    L = lambda m: log.append(str(m))
    hubs_sel = [h for h in _q(qs, "hubs", "Amarr").split(",") if h in esi.SECONDARIES]
    if not hubs_sel:
        hubs_sel = list(esi.SECONDARIES)
    workers = int(_q(qs, "workers", "8"))
    max_cand = (10 ** 9) if _q(qs, "all", "1") == "1" else int(_q(qs, "max_cand", "1000"))
    want_hist = _q(qs, "hist", "1") == "1"
    patterns = {c for c in _q(qs, "patterns", "ABCD") if c in "ABCD"}
    market_sell = _q(qs, "market_sell", "1") == "1"
    broker_cd = float(_q(qs, "broker_cd", "4"))
    cargo = float(_q(qs, "cargo", "534000"))
    min_pct = float(_q(qs, "min_pct", "1"))
    min_profit = float(_q(qs, "min_profit", "0"))
    min_cap = float(_q(qs, "min_cap", "0"))
    min_dv = float(_q(qs, "min_dv", "100" if want_hist else "0"))
    max_qd = float(_q(qs, "max_qd", "10"))
    _s = _q(qs, "sort", "")
    sort_key = _s if _s in AC.SORTS else (
        list(AC.SORTS)[int(_s)] if _s in ("0", "1") else list(AC.SORTS)[0])
    tol = float(_q(qs, "tol", "0")) / 100
    fee_names = _q(qs, "presets", "")
    presets = [p for p in fee_names.split("|") if p in AC.PRESETS] or \
              [next(k for k in AC.PRESETS if k.startswith("Accounting"))]
    if _q(qs, "block_off", "0") == "1":
        blocked = {"ids": set(), "names": set()}
    else:
        blocked = AC.load_blocklist(os.path.join(
            os.path.dirname(os.path.abspath(esi.__file__)), "blocklist.json"))
    want_hist = want_hist or min_dv > 0 or max_qd > 0
    t0 = time.time()

    here = os.path.dirname(os.path.abspath(esi.__file__))
    try:
        with open(os.path.join(here, "data", "fighters.csv"), encoding="utf-8") as f:
            FIGHTER_TIDS = [int(r["type_id"]) for r in _csv.DictReader(f)]
    except FileNotFoundError:
        FIGHTER_TIDS = []
    names = ["Jita"] + [h for h in hubs_sel if h != "Jita"]
    books = {}
    for h in names:
        hb = esi.HUBS[h]
        L("%s 주문판…" % h)
        try:
            books[h] = esi.fetch_hub_orders(hb["region"], hb["station"],
                                            workers=workers, log=L)
        except esi.ESIError as e:
            if h == "Jita":
                return {"error": "Jita 주문판 실패: %s" % e, "log": log}
            L("⚠ %s 주문판 실패(%s) — 이 쌍 접는다" % (h, e))
    names = [h for h in names if h in books]
    if len(names) < 2:
        return {"error": "짝이 없다 — Jita만으로는 어비 못 센다", "log": log}
    keep, scores = [], {}
    jita_inv = AC.internal_inversions(books["Jita"])
    for h in names[1:]:
        sc = AC.candidate_scores(books["Jita"], books[h])
        top = [t for _, t in sorted(((s, t) for t, s in sc.items()),
                                    reverse=True)][:max_cand]
        keep += top
        for t, s in sc.items():
            scores[t] = max(scores.get(t, 0.0), s)
        keep += list(jita_inv | AC.internal_inversions(books[h]))
        L("Jita↔%s: 교차 %d종" % (h, len(sc)))
    keep += FIGHTER_TIDS
    keep = list(dict.fromkeys(keep))
    scan = {"hubs": names,
            "books": {t: {h: books[h].get(t, {"sells": [], "buys": []}) for h in names}
                      for t in keep},
            "scores": scores, "at": datetime.now(timezone.utc)}
    del books
    ids = list(scan["books"])
    meta = esi.fetch_type_meta(ids, workers=10)
    keep2 = []
    for t in ids:
        m = meta.get(t, {})
        if AC.is_blocked(t, m.get("name", ""), blocked) or AC.is_apparel(m):
            continue
        keep2.append(t)
    for t in [t for t in ids if t not in set(keep2)]:
        scan["books"].pop(t, None)
    if len(ids) != len(keep2):
        L("🚫 걸러냄 %d종목" % (len(ids) - len(keep2)))
    hist = esi.fetch_histories(esi.JITA_REGION, list(scan["books"]), workers=10) \
        if want_hist else {}
    # ---- build_rows (app.py:583) -------------------------------------------
    cf = {p: AC.PRESETS[p] for p in presets}
    sell_fee_of = {p: (cf[p]["sell_tax"] if market_sell
                       else cf[p]["sell_tax"] + cf[p]["broker_sell"]) for p in cf}
    sell_fee_cd = {p: AC.sell_fee_cd_of(cf[p], broker_cd) for p in cf}
    rows = {p: [] for p in cf}
    for tid, bks in scan["books"].items():
        m = meta.get(tid)
        if not m or not m.get("volume"):
            continue
        for hub in scan["hubs"][1:]:
            b_hub = bks.get(hub, {"sells": [], "buys": []})
            for pname in cf:
                rows[pname] += AC.analyze_type(
                    tid, m["name"], m["volume"], bks["Jita"], b_hub,
                    cf[pname]["buy"], sell_fee_of[pname],
                    hist.get(tid, 0.0), cargo, hub_b=hub,
                    sell_fee_cd=sell_fee_cd[pname])
    # ---- 필터·정렬·한 표 (app.py filt + to_table) ----------------------------
    COLUMNS = ["종목", "쌍", "패턴", "방향", "매집가", "청산가", "조마진%", "순이익%",
               "체결량", "순이익 ISK", "묶인 자본", "단위부피 m³", "총부피 m³", "ISK/m³",
               "필요 운항", "1회 운항 이익", "셀오더 총량", "하루 소비량",
               "셀오더 재고/소비량", "소요일", "매도호가 유지 h", "매수호가 유지 h"]
    SRC = ["item", "pair", "pattern", "direction", "buy_avg", "sell_avg",
           "gross_pct", "net_pct", "qty", "profit", "capital", "unit_volume",
           "volume_m3", "isk_per_m3", "trips", "profit_per_trip", "sell_queue_qty",
           "daily_volume", "sell_queue_days", "days_to_sell", "ask_age_h", "bid_age_h"]
    DEC = {"매집가": 2, "청산가": 2, "조마진%": 2, "순이익%": 2, "순이익 ISK": 0,
           "묶인 자본": 0, "단위부피 m³": 3, "총부피 m³": 0, "ISK/m³": 0,
           "셀오더 재고/소비량": 2, "소요일": 2}
    out = []
    for pname, rr in rows.items():
        kept = AC.filter_rows(rr, min_pct, min_profit, min_cap, min_dv, patterns)
        if max_qd > 0:
            kept = [r for r in kept if r["sell_queue_days"] is not None
                    and r["sell_queue_days"] <= max_qd]
        for r in AC.sort_rows(kept, sort_key, tol):
            d = {"수수료": pname}
            for k, col in zip(SRC, COLUMNS):
                v = r[k]
                d[col] = v if isinstance(v, str) else _num(v, DEC.get(col, 9))
            out.append(d)
    # 원판은 프리셋별로 sort_rows 순서를 지킨다 — 여기서 뒤집으면 고른 정렬이 다 무린다
    L("표 %d행" % len(out))
    return {"hubs": names, "candidates": len(scan["books"]),
            "cols": ["수수료"] + COLUMNS, "rows": out[:2000],
            "truncated": len(out) > 2000, "total_rows": len(out),
            "log": log[-40:], "ms": int((time.time() - t0) * 1000)}



def station(qs):
    """스테이션 트레이딩 (/api/apps/station) — 한 스테이션 안에서 사고팔기.

    순간 차익이 아니라 스프레드를 판다: 매수호가를 최고 매도호가 못 미치게 걸면
    그 사이 간격(스프레드)만큼의 회수를 내 호가가 채운다. 매도호가 최솟값을 사고
    매수호가 최대값에 파는 순간 수익은 째다(ac reversal) — 이 판이 재는 것은
    스프레드 회수가능치: 스프레드 − 내 판매수수료, 그걸 하루에 몇 회 접을 수 있는지.
    """
    import esi
    import arb_core as AC

    log = []
    L = lambda m: log.append(str(m))
    hub = _q(qs, "hub", "Jita")
    if hub not in esi.HUBS:
        return {"error": "모르는 허브: %s (있음: %s)" % (hub, ", ".join(esi.HUBS))}
    workers = int(_q(qs, "workers", "8"))
    max_items = int(_q(qs, "max_items", "300"))
    min_spread = float(_q(qs, "min_spread", "1"))       # 최소 스프레드 %
    fee_pct = float(_q(qs, "fee", "4.9"))               # 내 판매 수수료 % (세+브로커)
    min_dv = float(_q(qs, "min_dv", "50"))              # 최소 하루 체결량
    min_capture = float(_q(qs, "min_capture", "0"))     # 최소 하루 회수 ISK
    t0 = time.time()

    hb = esi.HUBS[hub]
    L("%s — 전체 주문을 내린다" % hub)
    try:
        book = esi.fetch_hub_orders(hb["region"], hb["station"],
                                    workers=workers, log=L)
    except esi.ESIError as e:
        return {"error": "%s 주문판 실패: %s" % (hub, e), "log": log}

    # candidate: 양편이 걸린 종목 — 스프레드 창을 내 호가가 채운다
    cand = {}
    for tid, b in book.items():
        if not (b["sells"] and b["buys"]):
            continue
        ask = min(p for p, _, _ in b["sells"])
        bid = max(p for p, _, _ in b["buys"])
        if bid <= 0 or ask <= bid:
            continue                      # 뒤집힌 놈은 어비 판의 몫
        aq = sum(q for p, q, _ in b["sells"] if p == ask)
        bq = sum(q for p, q, _ in b["buys"] if p == bid)
        cand[tid] = (ask, bid, aq, bq)
    del book
    # candidates ranked by book thickness (how much sits at best prices)
    keep = sorted(cand, key=lambda t: min(cand[t][2], cand[t][3]),
                  reverse=True)[:max_items]
    L("양편 걸림 %d종 중 창두께 상위 %d종 판다" % (len(cand), len(keep)))

    meta = esi.fetch_type_meta(keep, workers=workers)
    blocked = {"ids": set(), "names": set()} if _q(qs, "block_off", "0") == "1" \
        else AC.load_blocklist(os.path.join(
            os.path.dirname(os.path.abspath(esi.__file__)), "blocklist.json"))
    keep = [t for t in keep
            if not AC.is_blocked(t, meta.get(t, {}).get("name", ""), blocked)
            and not AC.is_apparel(meta.get(t, {}))]
    hist = esi.fetch_histories(hb["region"], keep, workers=workers)

    rows = []
    for tid in keep:
        m = meta.get(tid)
        if not m or not m.get("name"):
            continue
        ask, bid, aq, bq = cand[tid]
        dv = hist.get(tid, 0.0)
        spread = ask - bid
        spread_pct = spread / bid * 100
        capture = ask * (1 - fee_pct / 100) - bid      # 내 몫의 단위에
        net_pct = capture / bid * 100
        trade = min(aq, bq)
        if dv > 0:
            trade = min(trade, dv)
        cap = capture * trade
        if spread_pct < min_spread or net_pct <= 0 or dv < min_dv or cap < min_capture:
            continue
        rows.append({"종목": m["name"], "매도호가": _num(ask, 2), "매수호가": _num(bid, 2),
                     "스프레드 ISK": _num(spread, 0), "스프레드 %": _num(spread_pct, 2),
                     "매도 창두께": _num(aq, 0), "매수 창두께": _num(bq, 0),
                     "수수료 차감 %": _num(net_pct, 2), "하루 체결량": _num(dv, 0),
                     "하루 회수 ISK": _num(cap, 0)})
    rows.sort(key=lambda r: (r["하루 회수 ISK"] or 0, r["스프레드 %"] or 0), reverse=True)
    L("표 %d행" % len(rows))
    cols = ["종목", "매도호가", "매수호가", "스프레드 ISK", "스프레드 %", "매도 창두께",
            "매수 창두께", "수수료 차감 %", "하루 체결량", "하루 회수 ISK"]
    return {"hub": hub, "candidates": len(cand), "cols": cols, "rows": rows[:2000],
            "truncated": len(rows) > 2000, "total_rows": len(rows),
            "log": log[-40:], "ms": int((time.time() - t0) * 1000)}


HANDLERS = {"crystal": crystal, "fighter": fighter, "arb": arb, "station": station}


def handle(path, qs):
    """/api/apps/<name> 의 <name> 과 쿼리를 받는다. 항상 dict 로 준다."""
    name = path.strip("/")
    fn = HANDLERS.get(name)
    if not fn:
        return {"error": "앱 없음: %s (있음: %s)" % (name, ", ".join(HANDLERS))}
    try:
        return fn(qs)
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}
