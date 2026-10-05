import streamlit as st
import pandas as pd
import requests
import time

# =============================================================
# Config
# =============================================================
st.set_page_config(
    page_title="EVE Online - Fighter Profit Calculator",
    layout="wide",
    page_icon="✈️",
)

# =============================================================
# Password Protection
# =============================================================
PASSWORD = "5767"

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Login Required")
    password = st.text_input("Enter password:", type="password")
    if st.button("Login"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# =============================================================
# Manufacturing data (verified via Fuzzwork blueprint API)
# =============================================================

# --- Minerals (T1 fighter inputs) ---
MINERALS = {
    'Tritanium': 34,
    'Pyerite': 35,
    'Mexallon': 36,
    'Isogen': 37,
    'Nocxium': 38,
    'Zydrine': 39,
    'Megacyte': 40,
}

# Two distinct T1 recipes (per fighter, per run)
T1_RECIPE = {
    # Light attack fighters (8 ships)
    'light': {
        'Tritanium': 165000, 'Pyerite': 45000, 'Mexallon': 15000,
        'Isogen': 6000, 'Nocxium': 900, 'Zydrine': 500, 'Megacyte': 400,
    },
    # Support fighters (4 ships)
    'support': {
        'Tritanium': 400000, 'Pyerite': 120000, 'Mexallon': 35000,
        'Isogen': 15000, 'Nocxium': 2500, 'Zydrine': 600, 'Megacyte': 1000,
    },
    # Heavy fighters — real values pending; see HEAVY_T2_MATERIALS note below.
    'heavy': None,
}

T1_RUNS_PER_BPC = 50   # maxProductionLimit of T1 fighter blueprints
T2_RUNS_PER_BPC = 10   # maxProductionLimit of T2 fighter blueprints
BUILD_TIME_SEC = 9000  # 2.5 hours per run (T1 and T2 alike)

# --- Heavyfighter recipes -------------------------------------------------------------
# 이 네트워크는 Fuzzwork 등 블루프린트 데이터 접근이 막혀 있어 Heavy 재료 실측치를
# 자동으로 적을 수 없다. 게임 제조 화면에서 보고 채우면 그 즉시 수익표에 반영된다.
# 값이 None이면 그 항목은 표에 안 나옵니다 (없는 숫자를 지레 계산하지 않음).
HEAVY_T2_MATERIALS = {   # (component: qty) — 예: {R.A.M.- Starship Tech: 3, ...}
    'Ametat II': None, 'Antaeus II': None, 'Cyclops II': None, 'Gungnir II': None,
    'Malleus II': None, 'Mantis II': None, 'Termite II': None, 'Tyrfing II': None,
}
# --- T1 fighters (also the T2 precursor item) ---
T1_FIGHTERS = {
    'Templar I':   {'type_id': 23055, 'recipe': 'light'},
    'Dragonfly I': {'type_id': 23057, 'recipe': 'light'},
    'Firbolg I':   {'type_id': 23059, 'recipe': 'light'},
    'Einherji I':  {'type_id': 23061, 'recipe': 'light'},
    'Equite I':    {'type_id': 40358, 'recipe': 'light'},
    'Locust I':    {'type_id': 40359, 'recipe': 'light'},
    'Satyr I':     {'type_id': 40360, 'recipe': 'light'},
    'Gram I':      {'type_id': 40361, 'recipe': 'light'},
    'Scarab I':    {'type_id': 40345, 'recipe': 'support'},
    'Siren I':     {'type_id': 40346, 'recipe': 'support'},
    'Dromi I':     {'type_id': 40347, 'recipe': 'support'},
    'Cenobite I':  {'type_id': 37599, 'recipe': 'support'},
    # Heavy fighters (T2 precursor items) — types.csv 대조 확정 tid
    'Ametat I':    {'type_id': 40362, 'recipe': 'heavy'},
    'Antaeus I':   {'type_id': 40364, 'recipe': 'heavy'},
    'Cyclops I':   {'type_id': 32325, 'recipe': 'heavy'},
    'Gungnir I':   {'type_id': 40365, 'recipe': 'heavy'},
    'Malleus I':   {'type_id': 32340, 'recipe': 'heavy'},
    'Mantis I':    {'type_id': 32344, 'recipe': 'heavy'},
    'Termite I':   {'type_id': 40363, 'recipe': 'heavy'},
    'Tyrfing I':   {'type_id': 32342, 'recipe': 'heavy'},
}

# --- Component type IDs ---
COMPONENTS = {
    'R.A.M.- Starship Tech': 11478,
    'Guidance Systems': 9834,
    'Morphite': 11399,
    'Phenolic Composites': 16680,
    'Fusion Thruster': 11532,
    'Plasma Thruster': 11530,
    'Magpulse Thruster': 11533,
    'Ion Thruster': 11531,
    'Radar Sensor Cluster': 11537,
    'Ladar Sensor Cluster': 11536,
    'Gravimetric Sensor Cluster': 11534,
    'Magnetometric Sensor Cluster': 11535,
    'Antimatter Reactor Unit': 11549,
    'Nuclear Reactor Unit': 11548,
    'Graviton Reactor Unit': 11550,
    'Fusion Reactor Unit': 11547,
    'Tesseract Capacitor Unit': 11554,
    'Scalar Capacitor Unit': 11552,
    'Oscillator Capacitor Unit': 11553,
    'Electrolytic Capacitor Unit': 11551,
    'Nanoelectrical Microprocessor': 11539,
    'Quantum Microprocessor': 11540,
    'Photon Microprocessor': 11541,
    'Nanomechanical Microprocessor': 11538,
    'Laser Focusing Crystals': 11689,
    'Superconductor Rails': 11690,
    'Particle Accelerator Unit': 11688,
    'Thermonuclear Trigger Unit': 11691,
}

# --- T2 fighters: precursor T1 + component quantities (per unit) ---
T2_FIGHTERS = {
    'Templar II': {
        'type_id': 40556, 'category': 'Light', 't1': 'Templar I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Guidance Systems': 12, 'Morphite': 12,
            'Fusion Thruster': 12, 'Radar Sensor Cluster': 12,
            'Laser Focusing Crystals': 12, 'Antimatter Reactor Unit': 16,
        },
    },
    'Dragonfly II': {
        'type_id': 40557, 'category': 'Light', 't1': 'Dragonfly I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Guidance Systems': 12, 'Morphite': 12,
            'Magpulse Thruster': 12, 'Gravimetric Sensor Cluster': 12,
            'Superconductor Rails': 12, 'Graviton Reactor Unit': 16,
        },
    },
    'Firbolg II': {
        'type_id': 40558, 'category': 'Light', 't1': 'Firbolg I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Guidance Systems': 12, 'Morphite': 12,
            'Ion Thruster': 12, 'Magnetometric Sensor Cluster': 12,
            'Particle Accelerator Unit': 12, 'Fusion Reactor Unit': 16,
        },
    },
    'Einherji II': {
        'type_id': 40559, 'category': 'Light', 't1': 'Einherji I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Guidance Systems': 12, 'Morphite': 12,
            'Plasma Thruster': 12, 'Ladar Sensor Cluster': 12,
            'Thermonuclear Trigger Unit': 12, 'Nuclear Reactor Unit': 16,
        },
    },
    'Equite II': {
        'type_id': 40552, 'category': 'Light', 't1': 'Equite I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Guidance Systems': 8,
            'Radar Sensor Cluster': 8, 'Morphite': 10,
            'Antimatter Reactor Unit': 12, 'Fusion Thruster': 20,
            'Phenolic Composites': 40,
        },
    },
    'Locust II': {
        'type_id': 40554, 'category': 'Light', 't1': 'Locust I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Guidance Systems': 8,
            'Gravimetric Sensor Cluster': 8, 'Morphite': 10,
            'Graviton Reactor Unit': 12, 'Magpulse Thruster': 20,
            'Phenolic Composites': 40,
        },
    },
    'Satyr II': {
        'type_id': 40555, 'category': 'Light', 't1': 'Satyr I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Guidance Systems': 8,
            'Magnetometric Sensor Cluster': 8, 'Morphite': 10,
            'Fusion Reactor Unit': 12, 'Ion Thruster': 20,
            'Phenolic Composites': 40,
        },
    },
    'Gram II': {
        'type_id': 40553, 'category': 'Light', 't1': 'Gram I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Guidance Systems': 8,
            'Ladar Sensor Cluster': 8, 'Morphite': 10,
            'Nuclear Reactor Unit': 12, 'Plasma Thruster': 20,
            'Phenolic Composites': 40,
        },
    },
    'Cenobite II': {
        'type_id': 40568, 'category': 'Support', 't1': 'Cenobite I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Fusion Thruster': 22, 'Morphite': 24,
            'Radar Sensor Cluster': 28, 'Tesseract Capacitor Unit': 36,
            'Antimatter Reactor Unit': 38, 'Nanoelectrical Microprocessor': 40,
        },
    },
    'Scarab II': {
        'type_id': 40569, 'category': 'Support', 't1': 'Scarab I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Magpulse Thruster': 22, 'Morphite': 24,
            'Gravimetric Sensor Cluster': 28, 'Scalar Capacitor Unit': 36,
            'Graviton Reactor Unit': 38, 'Quantum Microprocessor': 40,
        },
    },
    'Siren II': {
        'type_id': 40570, 'category': 'Support', 't1': 'Siren I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Ion Thruster': 22, 'Morphite': 24,
            'Magnetometric Sensor Cluster': 28, 'Oscillator Capacitor Unit': 36,
            'Fusion Reactor Unit': 38, 'Photon Microprocessor': 40,
        },
    },
    'Dromi II': {
        'type_id': 40571, 'category': 'Support', 't1': 'Dromi I',
        'materials': {
            'R.A.M.- Starship Tech': 3, 'Plasma Thruster': 22, 'Morphite': 24,
            'Ladar Sensor Cluster': 28, 'Electrolytic Capacitor Unit': 36,
            'Nuclear Reactor Unit': 38, 'Nanomechanical Microprocessor': 40,
        },
    },
    # --- Heavy fighters — 재료는 HEAVY_T2_MATERIALS에 실측 채표(현재 None) ---
    'Ametat II': {
        'type_id': 40560, 'category': 'Heavy', 't1': 'Ametat I',
        'materials': HEAVY_T2_MATERIALS['Ametat II'],
    },
    'Antaeus II': {
        'type_id': 40562, 'category': 'Heavy', 't1': 'Antaeus I',
        'materials': HEAVY_T2_MATERIALS['Antaeus II'],
    },
    'Cyclops II': {
        'type_id': 40563, 'category': 'Heavy', 't1': 'Cyclops I',
        'materials': HEAVY_T2_MATERIALS['Cyclops II'],
    },
    'Gungnir II': {
        'type_id': 40564, 'category': 'Heavy', 't1': 'Gungnir I',
        'materials': HEAVY_T2_MATERIALS['Gungnir II'],
    },
    'Malleus II': {
        'type_id': 40561, 'category': 'Heavy', 't1': 'Malleus I',
        'materials': HEAVY_T2_MATERIALS['Malleus II'],
    },
    'Mantis II': {
        'type_id': 40567, 'category': 'Heavy', 't1': 'Mantis I',
        'materials': HEAVY_T2_MATERIALS['Mantis II'],
    },
    'Termite II': {
        'type_id': 40566, 'category': 'Heavy', 't1': 'Termite I',
        'materials': HEAVY_T2_MATERIALS['Termite II'],
    },
    'Tyrfing II': {
        'type_id': 40565, 'category': 'Heavy', 't1': 'Tyrfing I',
        'materials': HEAVY_T2_MATERIALS['Tyrfing II'],
    },
}

TRADE_HUBS = {
    'Jita': 60003760,
    'Amarr': 60008494,
    'Dodixie': 60011866,
    'Rens': 60004588,
    'Hek': 60005686,
}

# =============================================================
# ESI API functions
# =============================================================

@st.cache_data(ttl=600)
def get_market_price(type_id, region_id=10000002, station_id=60003760):
    """Region orders filtered to a single station (default Jita 4-4)."""
    try:
        url = f"https://esi.evetech.net/latest/markets/{region_id}/orders/"
        params = {'datasource': 'tranquility', 'type_id': type_id}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            orders = response.json()
            orders = [o for o in orders if o.get('location_id') == station_id]
            buy_orders = [o for o in orders if o['is_buy_order']]
            sell_orders = [o for o in orders if not o['is_buy_order']]
            highest_buy = max([o['price'] for o in buy_orders]) if buy_orders else 0
            lowest_sell = min([o['price'] for o in sell_orders]) if sell_orders else 0
            return {
                'highest_buy': highest_buy,
                'lowest_sell': lowest_sell,
                'buy_volume': sum([o['volume_remain'] for o in buy_orders]),
                'sell_volume': sum([o['volume_remain'] for o in sell_orders]),
            }
        return None
    except Exception as e:
        st.warning(f"Failed to get price for Type ID {type_id}: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def get_market_history(type_id, region_id=10000002, days=100):
    """Average daily traded volume over the last `days` days."""
    try:
        url = f"https://esi.evetech.net/latest/markets/{region_id}/history/"
        params = {'datasource': 'tranquility', 'type_id': type_id}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            history = response.json()
            recent = history[-days:] if len(history) > days else history
            if recent:
                return sum(h['volume'] for h in recent) / len(recent)
            return 0
        return 0
    except Exception:
        return 0

# =============================================================
# Profit calculations
# =============================================================

def apply_fees(sell_price, broker_fee, sales_tax):
    """Sell price net of broker fee and sales tax."""
    return sell_price * (1 - (broker_fee + sales_tax) / 100)


def t1_manufacturing_cost(t1_name, mineral_prices, install_rate):
    """Total cost to manufacture one T1 fighter from minerals.

    Returns (total_cost, mat_breakdown) or (None, missing_list).
    """
    recipe = T1_RECIPE[T1_FIGHTERS[t1_name]['recipe']]
    if recipe is None:           # Heavy — 사이드바 실측 입력 우선, 없으면 표에서 빠진다
        recipe = st.session_state.get("heavy_mats", {}).get("T1 레시피 (전체 공유)")
    if not recipe:
        return None, []
    mat_cost = 0
    breakdown = {}
    for mineral, qty in recipe.items():
        if mineral not in mineral_prices or mineral_prices[mineral]['lowest_sell'] <= 0:
            return None, [mineral]
        price = mineral_prices[mineral]['lowest_sell']
        mat_cost += price * qty
        breakdown[mineral] = {'price': price, 'qty': qty, 'total': price * qty}
    total = mat_cost * (1 + install_rate)
    return total, breakdown


def calc_t1_profit(t1_name, mineral_prices, product_prices, fees, install_rate):
    """Profit for manufacturing and selling one T1 fighter."""
    broker_fee, sales_tax = fees
    t1_total, breakdown = t1_manufacturing_cost(t1_name, mineral_prices, install_rate)
    if t1_total is None:
        return None

    price = product_prices.get(t1_name)
    if not price or price['lowest_sell'] <= 0:
        return None

    sell = price['lowest_sell']
    net_sell = apply_fees(sell, broker_fee, sales_tax)
    profit = net_sell - t1_total
    margin = profit / t1_total * 100 if t1_total > 0 else 0
    hours = BUILD_TIME_SEC / 3600
    return {
        'fighter': t1_name,
        'type_id': T1_FIGHTERS[t1_name]['type_id'],
        'recipe': T1_FIGHTERS[t1_name]['recipe'],
        'material_cost': t1_total,
        'sell_price': sell,
        'net_sell': net_sell,
        'profit': profit,
        'margin': margin,
        'isk_per_hr': profit / hours,
        'per_bpc': profit * T1_RUNS_PER_BPC,
        'sell_volume': price['sell_volume'],
        'breakdown': breakdown,
    }


def calc_t2_profit(name, fighter, mineral_prices, component_prices,
                   product_prices, t1_prices, t1_cost_by_mfg,
                   fees, install_rate):
    """Profit for manufacturing and selling one T2 fighter.

    Scenario A: T1 precursor bought on market.
    Scenario B: T1 precursor manufactured from minerals.
    """
    broker_fee, sales_tax = fees
    t1_name = fighter['t1']

    # T2 component cost (excluding the T1 precursor)
    if fighter['materials'] is None:     # Heavy 실측 대기 — 숫자 지레 없음
        return None
    comp_cost = 0
    breakdown = {}
    for comp, qty in fighter['materials'].items():
        if comp not in component_prices or component_prices[comp]['lowest_sell'] <= 0:
            return None
        price = component_prices[comp]['lowest_sell']
        comp_cost += price * qty
        breakdown[comp] = {'price': price, 'qty': qty, 'total': price * qty}

    prod = product_prices.get(name)
    t1_prod = t1_prices.get(t1_name)
    if not prod or prod['lowest_sell'] <= 0 or not t1_prod or t1_prod['lowest_sell'] <= 0:
        return None

    # Scenario A: buy T1 at market sell price
    t1_mat_A = t1_prod['lowest_sell'] + comp_cost
    total_A = t1_mat_A * (1 + install_rate)

    # Scenario B: manufacture T1 from minerals (its install cost included)
    if t1_cost_by_mfg is None or t1_name not in t1_cost_by_mfg:
        return None
    t1_mat_B = t1_cost_by_mfg[t1_name] + comp_cost
    total_B = t1_mat_B * (1 + install_rate)

    net_sell = apply_fees(prod['lowest_sell'], broker_fee, sales_tax)
    profit_A = net_sell - total_A
    profit_B = net_sell - total_B
    margin_A = profit_A / total_A * 100 if total_A > 0 else 0
    margin_B = profit_B / total_B * 100 if total_B > 0 else 0

    hours_A = BUILD_TIME_SEC / 3600
    hours_B = 2 * BUILD_TIME_SEC / 3600  # T1 run + T2 run

    return {
        'fighter': name,
        'type_id': fighter['type_id'],
        'category': fighter['category'],
        't1': t1_name,
        'cost_A': total_A,
        'cost_B': total_B,
        'sell_price': prod['lowest_sell'],
        'net_sell': net_sell,
        'profit_A': profit_A,
        'profit_B': profit_B,
        'margin_A': margin_A,
        'margin_B': margin_B,
        'isk_hr_A': profit_A / hours_A,
        'isk_hr_B': profit_B / hours_B,
        'per_bpc_A': profit_A * T2_RUNS_PER_BPC,
        'per_bpc_B': profit_B * T2_RUNS_PER_BPC,
        'sell_volume': prod['sell_volume'],
        't1_market_price': t1_prod['lowest_sell'],
        't1_mfg_cost': t1_cost_by_mfg[t1_name],
        'breakdown': breakdown,
    }

# =============================================================
# UI
# =============================================================

def _has_matplotlib():
    """pandas Styler gradients need matplotlib >= 3.9.3."""
    try:
        import matplotlib
        parts = tuple(int(p) for p in matplotlib.__version__.split('.')[:3] if p.isdigit())
        return parts >= (3, 9, 3)
    except Exception:
        return False


def styled_table(df, fmt_map, gradient_col=None, vmin=-10, vmax=30):
    """Return a styled view of the DataFrame when possible.

    background_gradient requires matplotlib >= 3.9.3; skip the gradient
    (keep plain number formatting) when it is unavailable or too old.
    """
    try:
        if gradient_col and _has_matplotlib():
            return df.style.format(fmt_map).background_gradient(
                subset=[gradient_col], cmap='RdYlGn', vmin=vmin, vmax=vmax)
        return df.style.format(fmt_map)
    except Exception:
        return df


st.title("✈️ EVE Online - Fighter Manufacturing Profit Calculator")
st.caption("T1/T2 전투기 제조 수익 분석 — 전투기 12종 + 병행 T1 전투기 12종")

with st.sidebar:
    st.header("⚙️ Settings")

    hub_name = st.selectbox("Trading Hub", list(TRADE_HUBS.keys()), index=0)
    hub_id = TRADE_HUBS[hub_name]

    st.info(
        "재료 조달: 시장 팔 호가(lowest sell) 전량 구매 기준\n\n"
        "T2 시나리오 A: T1 전투기 시장 구매\n\n"
        "T2 시나리오 B: T1 전투기 자체 제조(광물 구매)\n\n"
        f"BPC당 생산량: T1 {T1_RUNS_PER_BPC}기 / T2 {T2_RUNS_PER_BPC}기, 런당 {BUILD_TIME_SEC/3600:.1f}시간"
    )

    st.markdown("---")

    st.subheader("Fee Settings")

    broker_fee = st.slider("Broker Fee (%)", min_value=0.0, max_value=5.0,
                           value=1.5, step=0.1, help="스테이션 중개 수수료")
    sales_tax = st.slider("Sales Tax (%)", min_value=0.0, max_value=5.0,
                          value=3.4, step=0.1, help="판매세")
    installation_cost = st.slider("Installation Cost (%)", min_value=0.0, max_value=10.0,
                                  value=3.0, step=0.1, help="재료비 대비 설치비 (%)")

    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Data Source: EVE Online ESI API")
    st.caption("Material price source: Fuzzwork blueprint data")

# =============================================================
# Data loading
# =============================================================

install_rate = installation_cost / 100
fees = (broker_fee, sales_tax)

st.header("📊 Loading Market Data")

all_material_tids = {**MINERALS, **COMPONENTS}

with st.spinner("시장 시세 로딩 중..."):
    progress = st.progress(0)
    status = st.empty()

    material_prices = {}
    t1_prices = {}
    t2_prices = {}

    items = [(k, tid) for k, tid in all_material_tids.items()]
    total = len(items)
    for i, (name, tid) in enumerate(items):
        status.text(f"Loading: {name} ({i+1}/{total})")
        price = get_market_price(tid, station_id=hub_id)
        if price:
            material_prices[name] = price
        progress.progress((i + 1) / total)
        time.sleep(0.15)

    # Fighters are materials too (T1 = material for T2)
    for name, data in T1_FIGHTERS.items():
        price = get_market_price(data['type_id'], station_id=hub_id)
        if price:
            material_prices[name] = price
            t1_prices[name] = price
    for name, data in T2_FIGHTERS.items():
        price = get_market_price(data['type_id'], station_id=hub_id)
        if price:
            t2_prices[name] = price

    progress.empty()
    status.empty()

# Mineral prices keyed for T1 recipe lookup
mineral_prices = {m: p for m, p in material_prices.items() if m in MINERALS}
component_prices = {c: p for c, p in material_prices.items() if c in COMPONENTS}

# T1 manufacturing costs (needed for T2 scenario B and T1 profit table)
t1_cost_by_mfg = {}
for t1_name in T1_FIGHTERS:
    total_cost, _ = t1_manufacturing_cost(t1_name, mineral_prices, install_rate)
    if total_cost is not None:
        t1_cost_by_mfg[t1_name] = total_cost

st.success(f"시세 로딩 완료 — {len(material_prices)}개 항목 (허브: {hub_name})")

# =============================================================
# T1 fighter profit
# =============================================================
st.divider()
st.header("🛩️ T1 전투기 제조 수익 (광물 → T1)")

t1_rows = []
for t1_name in T1_FIGHTERS:
    row = calc_t1_profit(t1_name, mineral_prices, t1_prices, fees, install_rate)
    if row:
        avg_vol = get_market_history(row['type_id'])
        row['avg_daily_volume'] = avg_vol
        # days to sell = my intended qty vs market capacity; use sell_volume as market supply
        row['days_to_sell'] = (row['sell_volume'] / avg_vol) if avg_vol else 0
        t1_rows.append(row)

if t1_rows:
    t1_df = pd.DataFrame(t1_rows).sort_values('profit', ascending=False)
    display = t1_df[[
        'fighter', 'recipe', 'material_cost', 'sell_price',
        'profit', 'margin', 'isk_per_hr', 'per_bpc', 'avg_daily_volume', 'days_to_sell',
    ]].copy()
    display.columns = [
        '전투기', '처방', '제조원가', '판매호가', '단위이익',
        '마진(%)', 'ISK/시간', 'BPC이익(50기)', '일평균거래량(100일)', '소요일',
    ]
    st.dataframe(
        styled_table(
            display, {
                '제조원가': '{:,.0f}', '판매호가': '{:,.0f}', '단위이익': '{:,.0f}',
                '마진(%)': '{:.2f}', 'ISK/시간': '{:,.0f}', 'BPC이익(50기)': '{:,.0f}',
                '일평균거래량(100일)': '{:,.0f}', '소요일': '{:.3f}',
            },
            gradient_col='마진(%)',
        ),
        width="stretch", height=400,
    )

    with st.expander("T1 제조 재료 상세 (광물 내역)"):
        for r in sorted(t1_rows, key=lambda x: x['fighter']):
            inner = pd.DataFrame([
                {'재료': m, '단가': b['price'], '수량': b['qty'], '합계': b['total']}
                for m, b in r['breakdown'].items()
            ])
            st.markdown(f"**{r['fighter']}**")
            st.dataframe(
                styled_table(inner, {'단가': '{:,.2f}', '수량': '{:,.0f}', '합계': '{:,.0f}'}),
                width="stretch", hide_index=True,
            )
else:
    st.error("T1 수익 계산 불가 — 광물 시세를 확인하세요.")

# =============================================================
# T2 fighter profit (A/B comparison)
# =============================================================
st.divider()
st.header("⚔️ T2 전투기 제조 수익 (T1 + 부품 → T2)")

cat_filter = st.multiselect(
    "카테고리 필터", options=['Light', 'Support', 'Heavy'],
    default=['Light', 'Support', 'Heavy'],
)

_pending = [n for n, v in HEAVY_T2_MATERIALS.items() if v is None]
if _pending:
    st.info(f"⚠️ Heavy {_pending and len(_pending)}종은 재료 실측치가 없어 수익표에서 빠져 있습니다. "
            "게임 제조창의 부품 목록(예: 'Morphite 12' 8줄)을 주세요 — "
            "app.py 상단 HEAVY_T2_MATERIALS에 바로 박아 넣습니다.")

t2_rows = []
for name, fighter in T2_FIGHTERS.items():
    if fighter['category'] not in cat_filter:
        continue
    if fighter['materials'] is None:     # Heavy — 사이드바 실측 입력이 먼저
        _m = st.session_state.get("heavy_mats", {}).get(name)
        if _m:
            fighter = {**fighter, 'materials': _m}
    row = calc_t2_profit(name, fighter, mineral_prices, component_prices,
                         t2_prices, t1_prices, t1_cost_by_mfg, fees, install_rate)
    if row:
        avg_vol = get_market_history(row['type_id'])
        row['avg_daily_volume'] = avg_vol
        row['days_to_sell'] = (row['sell_volume'] / avg_vol) if avg_vol else 0
        row['best'] = 'B(T1제조)' if row['profit_B'] > row['profit_A'] else 'A(T1구매)'
        t2_rows.append(row)

if t2_rows:
    t2_df = pd.DataFrame(t2_rows).sort_values('profit_B', ascending=False)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    best_row = t2_df.iloc[0]
    m1.metric("최고 수익 (B)", f"{best_row['profit_B']:,.0f} ISK", f"{best_row['fighter']}")
    m2.metric("최고 마진 (B)", f"{best_row['margin_B']:.1f}%", f"{best_row['fighter']}")
    profitable = len(t2_df[(t2_df['profit_A'] > 0) | (t2_df['profit_B'] > 0)])
    m3.metric("수익 있는 전투기", f"{profitable}/{len(t2_df)}")
    t1_better = (t2_df['profit_B'] > t2_df['profit_A']).sum()
    m4.metric("T1 제조가 이로운 경우", f"{t1_better}/{len(t2_df)}")

    display = t2_df[[
        'fighter', 'category', 'cost_A', 'cost_B', 'sell_price', 't1_market_price',
        't1_mfg_cost', 'profit_A', 'profit_B', 'margin_A', 'margin_B',
        'isk_hr_A', 'isk_hr_B', 'per_bpc_A', 'per_bpc_B', 'best',
        'avg_daily_volume', 'days_to_sell',
    ]].copy()
    display.columns = [
        'T2 전투기', '분류', '원가A', '원가B', '판매호가', 'T1 시가',
        'T1 제조원가', '단위이익A', '단위이익B', '마진A(%)', '마진B(%)',
        'ISK/h A', 'ISK/h B', 'BPC이익 A(10기)', 'BPC이익 B(10기)', '최적',
        '일평균거래량(100일)', '소요일',
    ]
    st.dataframe(
        styled_table(
            display, {
                '원가A': '{:,.0f}', '원가B': '{:,.0f}', '판매호가': '{:,.0f}',
                'T1 시가': '{:,.0f}', 'T1 제조원가': '{:,.0f}',
                '단위이익A': '{:,.0f}', '단위이익B': '{:,.0f}',
                '마진A(%)': '{:.2f}', '마진B(%)': '{:.2f}',
                'ISK/h A': '{:,.0f}', 'ISK/h B': '{:,.0f}',
                'BPC이익 A(10기)': '{:,.0f}', 'BPC이익 B(10기)': '{:,.0f}',
                '일평균거래량(100일)': '{:,.0f}', '소요일': '{:.2f}',
            },
            gradient_col='마진B(%)',
        ),
        width="stretch", height=500,
    )

    with st.expander("T2 제조 부품 상세 (T2 전투기별)"):
        for r in sorted(t2_rows, key=lambda x: x['fighter']):
            inner = pd.DataFrame([
                {'부품': m, '단가': b['price'], '수량': b['qty'], '합계': b['total']}
                for m, b in r['breakdown'].items()
            ])
            st.markdown(
                f"**{r['fighter']}** — T1: {r['t1']} (시가 {r['t1_market_price']:,.0f} / "
                f"제조원가 {r['t1_mfg_cost']:,.0f})"
            )
            st.dataframe(
                styled_table(inner, {'단가': '{:,.2f}', '수량': '{:,.0f}', '합계': '{:,.0f}'}),
                width="stretch", hide_index=True,
            )
else:
    st.error("T2 수익 계산 불가 — 부품/완성품 시세를 확인하세요.")

st.divider()
st.caption("""
주의:
- 재료는 전부 시장 팔 호가 전량 구매 기준, 판매는 팔 호가 기준입니다.
- A: T1 전투기 시장 구매 / B: T1 전투기 자체 제조(광물 구매). B는 제조 시간 2배(5시간).
- 발명(디크립터/데이터코어) 비용은 제외된 제조 원가 기준입니다.
- ME/TE 연구, 팩토리 보너스, 운송 비용은 반영되지 않았습니다.
- 비인기 종목은 호가가 없어 빠지거나 수익이 크게 부정적일 수 있습니다.

Data Source: EVE Online ESI API (CCP Games) / 제조 처방: Fuzzwork
""")
