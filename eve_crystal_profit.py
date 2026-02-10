import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime
import time

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="EVE Online - Crystal Profit Calculator", layout="wide", page_icon="💎")

# -----------------------------
# Password Protection
# -----------------------------
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

# -----------------------------
# EVE ESI API Functions
# -----------------------------

# Laser Crystal Type ID Mapping (Advanced Frequency Crystals - Tech 2)
# Note: blueprint 10 runs = 40 units (4 per run × 10 runs)
# Void/Null: 1 run = 5,000 units (10 runs = 50,000 units)
# Materials are per 1 run × 10
CRYSTALS = {
    # Conflagration (Advanced X-Ray)
    'Conflagration S': {'type_id': 12565, 'materials': {'Morphite': 10, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 1080, 'Fullerides': 450}, 'runs': 10, 'output_per_run': 4},
    'Conflagration M': {'type_id': 12814, 'materials': {'Morphite': 60, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 4320, 'Fullerides': 1800}, 'runs': 10, 'output_per_run': 4},
    'Conflagration L': {'type_id': 12816, 'materials': {'Morphite': 150, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 10800, 'Fullerides': 4500}, 'runs': 10, 'output_per_run': 4},

    # Scorch (Advanced Multifrequency)
    'Scorch S': {'type_id': 12563, 'materials': {'Morphite': 10, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 1080, 'Fullerides': 450}, 'runs': 10, 'output_per_run': 4},
    'Scorch M': {'type_id': 12818, 'materials': {'Morphite': 60, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 4320, 'Fullerides': 1800}, 'runs': 10, 'output_per_run': 4},
    'Scorch L': {'type_id': 12820, 'materials': {'Morphite': 150, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 10800, 'Fullerides': 4500}, 'runs': 10, 'output_per_run': 4},

    # Aurora (Advanced Radio) - Requires more Tungsten Carbide
    'Aurora S': {'type_id': 12559, 'materials': {'Morphite': 10, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 2610, 'Fullerides': 450}, 'runs': 10, 'output_per_run': 4},
    'Aurora M': {'type_id': 12822, 'materials': {'Morphite': 60, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 10440, 'Fullerides': 1800}, 'runs': 10, 'output_per_run': 4},
    'Aurora L': {'type_id': 12824, 'materials': {'Morphite': 150, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 26100, 'Fullerides': 4500}, 'runs': 10, 'output_per_run': 4},

    # Gleam (Advanced Infrared) - Requires more Tungsten Carbide
    'Gleam S': {'type_id': 12557, 'materials': {'Morphite': 10, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 2610, 'Fullerides': 450}, 'runs': 10, 'output_per_run': 4},
    'Gleam M': {'type_id': 12826, 'materials': {'Morphite': 60, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 10440, 'Fullerides': 1800}, 'runs': 10, 'output_per_run': 4},
    'Gleam L': {'type_id': 12828, 'materials': {'Morphite': 150, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 26100, 'Fullerides': 4500}, 'runs': 10, 'output_per_run': 4},

    # Small Ionic Field Projector II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small Ionic Field Projector II': {'type_id': 31280, 'materials': {'Miniature Electronics': 24, 'R.A.M.- Electronics': 4, 'Artificial Neural Network': 4, 'Micro Circuit': 4, 'Logic Circuit': 4}, 'runs': 4, 'output_per_run': 1},

    # Medium Core Defense Field Extender II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Medium Core Defense Field Extender II': {'type_id': 31796, 'materials': {'R.A.M.- Shield Tech': 4, 'Power Circuit': 24, 'Logic Circuit': 24, 'Enhanced Ward Console': 12}, 'runs': 4, 'output_per_run': 1},

    # Small Core Defense Field Extender II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small Core Defense Field Extender II': {'type_id': 31794, 'materials': {'R.A.M.- Shield Tech': 4, 'Power Circuit': 4, 'Logic Circuit': 4, 'Enhanced Ward Console': 4}, 'runs': 4, 'output_per_run': 1},

    # Large Core Defense Field Extender II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large Core Defense Field Extender II': {'type_id': 26448, 'materials': {'R.A.M.- Shield Tech': 4, 'Power Circuit': 120, 'Logic Circuit': 120, 'Enhanced Ward Console': 48}, 'runs': 4, 'output_per_run': 1},

    # Medium Trimark Armor Pump II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Medium Trimark Armor Pump II': {'type_id': 31059, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 12, 'Interface Circuit': 20, 'Intact Armor Plates': 12}, 'runs': 4, 'output_per_run': 1},

    # Large Trimark Armor Pump II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large Trimark Armor Pump II': {'type_id': 26302, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 60, 'Interface Circuit': 92, 'Intact Armor Plates': 80}, 'runs': 4, 'output_per_run': 1},

    # Small EM Armor Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small EM Armor Reinforcer II': {'type_id': 31003, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 4, 'Interface Circuit': 4, 'Intact Armor Plates': 4}, 'runs': 4, 'output_per_run': 1},

    # Small Thermal Armor Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small Thermal Armor Reinforcer II': {'type_id': 31039, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 4, 'Interface Circuit': 4, 'Intact Armor Plates': 4}, 'runs': 4, 'output_per_run': 1},

    # Medium EM Armor Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Medium EM Armor Reinforcer II': {'type_id': 31005, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 4, 'Interface Circuit': 20, 'Intact Armor Plates': 12}, 'runs': 4, 'output_per_run': 1},

    # Medium Thermal Armor Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Medium Thermal Armor Reinforcer II': {'type_id': 31041, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 4, 'Interface Circuit': 20, 'Intact Armor Plates': 12}, 'runs': 4, 'output_per_run': 1},

    # Large EM Armor Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large EM Armor Reinforcer II': {'type_id': 26286, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 48, 'Interface Circuit': 92, 'Intact Armor Plates': 80}, 'runs': 4, 'output_per_run': 1},

    # Large Thermal Armor Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large Thermal Armor Reinforcer II': {'type_id': 26292, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 48, 'Interface Circuit': 92, 'Intact Armor Plates': 80}, 'runs': 4, 'output_per_run': 1},

    # Large Core Defense Field Purger II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large Core Defense Field Purger II': {'type_id': 26450, 'materials': {'R.A.M.- Shield Tech': 4, 'Power Circuit': 120, 'Logic Circuit': 120, 'Enhanced Ward Console': 48}, 'runs': 4, 'output_per_run': 1},

    # Large Capacitor Control Circuit II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large Capacitor Control Circuit II': {'type_id': 26374, 'materials': {'R.A.M.- Energy Tech': 4, 'Power Circuit': 80, 'Logic Circuit': 92, 'Capacitor Console': 24}, 'runs': 4, 'output_per_run': 1},

    # Medium Capacitor Control Circuit II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Medium Capacitor Control Circuit II': {'type_id': 31378, 'materials': {'R.A.M.- Energy Tech': 4, 'Power Circuit': 20, 'Logic Circuit': 20, 'Capacitor Console': 4}, 'runs': 4, 'output_per_run': 1},

    # Small Capacitor Control Circuit II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small Capacitor Control Circuit II': {'type_id': 31376, 'materials': {'R.A.M.- Energy Tech': 4, 'Power Circuit': 4, 'Logic Circuit': 4, 'Capacitor Console': 4}, 'runs': 4, 'output_per_run': 1},

    # Large Transverse Bulkhead II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large Transverse Bulkhead II': {'type_id': 33900, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 92, 'Single-crystal Superalloy I-beam': 80, 'Interface Circuit': 60}, 'runs': 4, 'output_per_run': 1},

    # Medium Transverse Bulkhead II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Medium Transverse Bulkhead II': {'type_id': 33896, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 20, 'Single-crystal Superalloy I-beam': 12, 'Interface Circuit': 12}, 'runs': 4, 'output_per_run': 1},

    # Small Transverse Bulkhead II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small Transverse Bulkhead II': {'type_id': 33892, 'materials': {'R.A.M.- Armor/Hull Tech': 4, 'Nanite Compound': 4, 'Single-crystal Superalloy I-beam': 4, 'Interface Circuit': 4}, 'runs': 4, 'output_per_run': 1},

    # Small Ancillary Current Router II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small Ancillary Current Router II': {'type_id': 31364, 'materials': {'R.A.M.- Energy Tech': 4, 'Power Conduit': 4, 'Power Circuit': 4, 'Logic Circuit': 4}, 'runs': 4, 'output_per_run': 1},

    # Large EM Shield Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large EM Shield Reinforcer II': {'type_id': 26436, 'materials': {'R.A.M.- Shield Tech': 4, 'Intact Shield Emitter': 48, 'Micro Circuit': 80, 'Interface Circuit': 92}, 'runs': 4, 'output_per_run': 1},

    # Medium EM Shield Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Medium EM Shield Reinforcer II': {'type_id': 31724, 'materials': {'R.A.M.- Shield Tech': 4, 'Intact Shield Emitter': 12, 'Micro Circuit': 20, 'Interface Circuit': 20}, 'runs': 4, 'output_per_run': 1},

    # Small EM Shield Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small EM Shield Reinforcer II': {'type_id': 31722, 'materials': {'R.A.M.- Shield Tech': 4, 'Intact Shield Emitter': 4, 'Micro Circuit': 4, 'Interface Circuit': 4}, 'runs': 4, 'output_per_run': 1},

    # Large Thermal Shield Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Large Thermal Shield Reinforcer II': {'type_id': 26442, 'materials': {'R.A.M.- Shield Tech': 4, 'Intact Shield Emitter': 48, 'Micro Circuit': 80, 'Interface Circuit': 92}, 'runs': 4, 'output_per_run': 1},

    # Medium Thermal Shield Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Medium Thermal Shield Reinforcer II': {'type_id': 31760, 'materials': {'R.A.M.- Shield Tech': 4, 'Intact Shield Emitter': 12, 'Micro Circuit': 20, 'Interface Circuit': 20}, 'runs': 4, 'output_per_run': 1},

    # Small Thermal Shield Reinforcer II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small Thermal Shield Reinforcer II': {'type_id': 31758, 'materials': {'R.A.M.- Shield Tech': 4, 'Intact Shield Emitter': 4, 'Micro Circuit': 4, 'Interface Circuit': 4}, 'runs': 4, 'output_per_run': 1},
}

# Material Type IDs (Materials needed for Advanced Crystal manufacturing)
MATERIALS = {
    # Base materials
    'Morphite': 11399,
    'R.A.M.- Ammunition Tech': 11476,
    'R.A.M.- Electronics': 11483,
    'Tungsten Carbide': 16672,
    'Fullerides': 16679,
    'Crystalline Carbonide': 16670,
    # Salvage Materials (For Rig manufacturing)
    'Miniature Electronics': 9842,
    'Artificial Neural Network': 25616,
    'Micro Circuit': 25618,
    'Logic Circuit': 25619,
    # Shield Rig Materials
    'R.A.M.- Shield Tech': 11484,
    'Power Circuit': 25617,
    'Enhanced Ward Console': 25625,
    # Armor Rig Materials
    'R.A.M.- Armor/Hull Tech': 11475,
    'Nanite Compound': 25609,
    'Interface Circuit': 25620,
    'Intact Armor Plates': 25624,
    # Capacitor Rig Materials
    'R.A.M.- Energy Tech': 11482,
    'Capacitor Console': 25622,
    # Engineering Rig Materials
    'Power Conduit': 25613,
    # Hull/Bulkhead Rig Materials
    'Single-crystal Superalloy I-beam': 25614,
    # Shield Resistance Rig Materials
    'Intact Shield Emitter': 25608,
}

# Major trading hubs
TRADE_HUBS = {
    'Jita': 60003760,
    'Amarr': 60008494,
    'Dodixie': 60011866,
    'Rens': 60004588,
    'Hek': 60005686
}

@st.cache_data(ttl=600)
def get_market_price(type_id, region_id=10000002, station_id=60003760):
    """Get market prices from ESI API filtered by Jita station"""
    try:
        url = f"https://esi.evetech.net/latest/markets/{region_id}/orders/"
        params = {'datasource': 'tranquility', 'type_id': type_id}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            orders = response.json()

            # Filter orders by Jita station only
            orders = [o for o in orders if o.get('location_id') == station_id]

            # Separate buy/sell orders
            buy_orders = [o for o in orders if o['is_buy_order']]
            sell_orders = [o for o in orders if not o['is_buy_order']]

            # Highest buy, lowest sell
            highest_buy = max([o['price'] for o in buy_orders]) if buy_orders else 0
            lowest_sell = min([o['price'] for o in sell_orders]) if sell_orders else 0

            return {
                'highest_buy': highest_buy,
                'lowest_sell': lowest_sell,
                'buy_volume': sum([o['volume_remain'] for o in buy_orders]),
                'sell_volume': sum([o['volume_remain'] for o in sell_orders])
            }
        return None
    except Exception as e:
        st.warning(f"Failed to get price for Type ID {type_id}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_market_history(type_id, region_id=10000002, days=100):
    """Get market history from ESI API (100-day average volume)"""
    try:
        url = f"https://esi.evetech.net/latest/markets/{region_id}/history/"
        params = {'datasource': 'tranquility', 'type_id': type_id}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            history = response.json()

            # Use only last 100 days
            recent_history = history[-days:] if len(history) > days else history

            if recent_history:
                avg_volume = sum([h['volume'] for h in recent_history]) / len(recent_history)
                return avg_volume
            return 0
        return 0
    except Exception as e:
        return 0

@st.cache_data(ttl=600)
def get_station_price(type_id, station_id):
    """Market price at specific station"""
    try:
        url = f"https://esi.evetech.net/latest/markets/structures/{station_id}/"
        params = {'datasource': 'tranquility'}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            orders = response.json()
            orders = [o for o in orders if o['type_id'] == type_id]

            buy_orders = [o for o in orders if o['is_buy_order']]
            sell_orders = [o for o in orders if not o['is_buy_order']]

            highest_buy = max([o['price'] for o in buy_orders]) if buy_orders else 0
            lowest_sell = min([o['price'] for o in sell_orders]) if sell_orders else 0

            return {'highest_buy': highest_buy, 'lowest_sell': lowest_sell}
        return None
    except:
        return None

def calculate_profit(crystal_name, crystal_data, material_prices):
    """Calculate crystal manufacturing profit"""
    material_cost_total = 0
    material_breakdown = {}

    for material, quantity in crystal_data['materials'].items():
        if material in material_prices:
            price = material_prices[material]['lowest_sell']
            cost = price * quantity
            material_cost_total += cost
            material_breakdown[material] = {'price': price, 'quantity': quantity, 'total': cost}
        else:
            return None

    runs = crystal_data.get('runs', 10)
    output_per_run = crystal_data.get('output_per_run', 4)
    total_output = runs * output_per_run

    material_cost_per_unit = material_cost_total / total_output

    crystal_price_data = get_market_price(crystal_data['type_id'])

    if not crystal_price_data:
        return None

    sell_price = crystal_price_data['lowest_sell']
    buy_order_price = crystal_price_data['highest_buy']

    avg_daily_volume = get_market_history(crystal_data['type_id'])

    profit_per_unit = sell_price - material_cost_per_unit
    profit_margin = (profit_per_unit / material_cost_per_unit * 100) if material_cost_per_unit > 0 else 0

    total_profit = profit_per_unit * total_output
    total_revenue = sell_price * total_output

    bpc_count = 10
    output_10_bpc = bpc_count * total_output
    material_cost_10_bpc = material_cost_per_unit * output_10_bpc
    profit_10_bpc = profit_per_unit * output_10_bpc

    return {
        'crystal_name': crystal_name,
        'material_cost': material_cost_per_unit,
        'material_cost_total': material_cost_total,
        'material_breakdown': material_breakdown,
        'output_count': total_output,
        'sell_price': sell_price,
        'buy_order_price': buy_order_price,
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'profit': profit_per_unit,
        'profit_margin': profit_margin,
        'buy_volume': crystal_price_data['buy_volume'],
        'sell_volume': crystal_price_data['sell_volume'],
        'lowest_sell': crystal_price_data['lowest_sell'],
        'avg_daily_volume': avg_daily_volume,
        'profit_10_bpc': profit_10_bpc,
        'material_cost_10_bpc': material_cost_10_bpc,
        'output_10_bpc': output_10_bpc,
        'output_per_bpc': total_output
    }

# UI
st.title("🏭 EVE Online - Manufacturing Profit Calculator")
st.caption("T2 Crystals, Ammo, Rigs manufacturing profitability analysis")

with st.sidebar:
    st.header("⚙️ Settings")

    st.info("Material Purchase: Jita Sell Order (lowest price)\n\nProduction: 1 BPC = 40 units / Rig: 4 units")

    st.markdown("---")

    st.subheader("Fee Settings")

    broker_fee = st.slider(
        "Broker Fee (%)",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Station trading broker fee"
    )

    sales_tax = st.slider(
        "Sales Tax (%)",
        min_value=0.0,
        max_value=5.0,
        value=3.4,
        step=0.1,
        help="Sales tax"
    )

    installation_cost = st.slider(
        "Installation Cost (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Manufacturing installation cost (% of material cost)"
    )

    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Data Source: EVE Online ESI API")
    st.caption("Refresh: Every 10 minutes")

# Data Loading
st.header("📊 Loading Market Data")

with st.spinner("Loading material prices..."):
    material_prices = {}
    for material_name, type_id in MATERIALS.items():
        price_data = get_market_price(type_id)
        if price_data:
            material_prices[material_name] = price_data
        time.sleep(0.2)

if material_prices:
    st.subheader("🔧 Material Prices (Jita Sell Order - Lowest)")

    st.write("**Crystal/Ammunition Materials:**")
    mat_col1, mat_col2, mat_col3, mat_col4 = st.columns(4)

    with mat_col1:
        if 'Morphite' in material_prices:
            st.metric(
                "Morphite",
                f"{material_prices['Morphite']['lowest_sell']:,.2f} ISK",
                delta="Rare Mineral"
            )

    with mat_col2:
        if 'Tungsten Carbide' in material_prices:
            st.metric(
                "Tungsten Carbide",
                f"{material_prices['Tungsten Carbide']['lowest_sell']:,.2f} ISK",
                delta="For Crystals"
            )

    with mat_col3:
        if 'Fullerides' in material_prices:
            st.metric(
                "Fullerides",
                f"{material_prices['Fullerides']['lowest_sell']:,.2f} ISK",
                delta="Base Material"
            )

    with mat_col4:
        if 'Crystalline Carbonide' in material_prices:
            st.metric(
                "Crystalline Carbonide",
                f"{material_prices['Crystalline Carbonide']['lowest_sell']:,.2f} ISK",
                delta="For Ammo"
            )

    st.write("**R.A.M. (Robotic Assembly Modules):**")
    ram_col1, ram_col2 = st.columns(2)

    with ram_col1:
        if 'R.A.M.- Ammunition Tech' in material_prices:
            st.metric(
                "R.A.M.- Ammunition Tech",
                f"{material_prices['R.A.M.- Ammunition Tech']['lowest_sell']:,.2f} ISK",
                delta="Ammo/Crystal"
            )

    with ram_col2:
        if 'R.A.M.- Electronics' in material_prices:
            st.metric(
                "R.A.M.- Electronics",
                f"{material_prices['R.A.M.- Electronics']['lowest_sell']:,.2f} ISK",
                delta="For Rigs"
            )

    st.write("**Rig Materials (Salvage/PI):**")
    rig_col1, rig_col2, rig_col3, rig_col4 = st.columns(4)

    with rig_col1:
        if 'Miniature Electronics' in material_prices:
            st.metric(
                "Miniature Electronics",
                f"{material_prices['Miniature Electronics']['lowest_sell']:,.2f} ISK",
                delta="PI Material"
            )

    with rig_col2:
        if 'Artificial Neural Network' in material_prices:
            st.metric(
                "Artificial Neural Network",
                f"{material_prices['Artificial Neural Network']['lowest_sell']:,.2f} ISK",
                delta="Salvage"
            )

    with rig_col3:
        if 'Micro Circuit' in material_prices:
            st.metric(
                "Micro Circuit",
                f"{material_prices['Micro Circuit']['lowest_sell']:,.2f} ISK",
                delta="Salvage"
            )

    with rig_col4:
        if 'Logic Circuit' in material_prices:
            st.metric(
                "Logic Circuit",
                f"{material_prices['Logic Circuit']['lowest_sell']:,.2f} ISK",
                delta="Salvage"
            )

    st.write("**Shield Rig Materials:**")
    shield_col1, shield_col2, shield_col3 = st.columns(3)

    with shield_col1:
        if 'R.A.M.- Shield Tech' in material_prices:
            st.metric(
                "R.A.M.- Shield Tech",
                f"{material_prices['R.A.M.- Shield Tech']['lowest_sell']:,.2f} ISK",
                delta="For Shield Rigs"
            )

    with shield_col2:
        if 'Power Circuit' in material_prices:
            st.metric(
                "Power Circuit",
                f"{material_prices['Power Circuit']['lowest_sell']:,.2f} ISK",
                delta="Salvage"
            )

    with shield_col3:
        if 'Enhanced Ward Console' in material_prices:
            st.metric(
                "Enhanced Ward Console",
                f"{material_prices['Enhanced Ward Console']['lowest_sell']:,.2f} ISK",
                delta="Salvage"
            )

    st.write("**Armor Rig Materials:**")
    armor_col1, armor_col2, armor_col3, armor_col4 = st.columns(4)

    with armor_col1:
        if 'R.A.M.- Armor/Hull Tech' in material_prices:
            st.metric(
                "R.A.M.- Armor/Hull Tech",
                f"{material_prices['R.A.M.- Armor/Hull Tech']['lowest_sell']:,.2f} ISK",
                delta="For Armor Rigs"
            )

    with armor_col2:
        if 'Nanite Compound' in material_prices:
            st.metric(
                "Nanite Compound",
                f"{material_prices['Nanite Compound']['lowest_sell']:,.2f} ISK",
                delta="Salvage"
            )

    with armor_col3:
        if 'Interface Circuit' in material_prices:
            st.metric(
                "Interface Circuit",
                f"{material_prices['Interface Circuit']['lowest_sell']:,.2f} ISK",
                delta="Salvage"
            )

    with armor_col4:
        if 'Intact Armor Plates' in material_prices:
            st.metric(
                "Intact Armor Plates",
                f"{material_prices['Intact Armor Plates']['lowest_sell']:,.2f} ISK",
                delta="Salvage"
            )

st.divider()

st.header("💰 Crystal Manufacturing Profitability")

with st.spinner("Loading crystal market data..."):
    profit_data = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (crystal_name, crystal_data) in enumerate(CRYSTALS.items()):
        status_text.text(f"Loading: {crystal_name} ({idx+1}/{len(CRYSTALS)})")

        profit_info = calculate_profit(crystal_name, crystal_data, material_prices)
        if profit_info:
            # Calculate installation cost
            install_cost_rate = installation_cost / 100
            install_cost_per_unit = profit_info['material_cost'] * install_cost_rate
            total_cost_per_unit = profit_info['material_cost'] + install_cost_per_unit

            # Calculate profit after all fees (broker fee + sales tax applied to sell price)
            total_fees = (broker_fee + sales_tax) / 100
            net_sell_price = profit_info['sell_price'] * (1 - total_fees)
            profit_info['profit_after_fees'] = net_sell_price - total_cost_per_unit
            profit_info['profit_margin_after_fees'] = (profit_info['profit_after_fees'] / total_cost_per_unit * 100) if total_cost_per_unit > 0 else 0
            profit_info['installation_cost'] = install_cost_per_unit
            profit_info['total_cost'] = total_cost_per_unit
            profit_info['net_sell_price'] = net_sell_price

            # Update profit_10_bpc with installation cost and fees
            profit_info['profit_10_bpc'] = profit_info['profit_after_fees'] * profit_info['output_10_bpc']

            profit_data.append(profit_info)

        progress_bar.progress((idx + 1) / len(CRYSTALS))
        time.sleep(0.3)

    progress_bar.empty()
    status_text.empty()

if profit_data:
    df = pd.DataFrame(profit_data)
    df = df.sort_values('profit_margin_after_fees', ascending=False)

    st.subheader("📈 Profitability Summary")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric(
            "Average Profit Margin",
            f"{df['profit_margin_after_fees'].mean():.2f}%"
        )

    with summary_col2:
        best_crystal = df.iloc[0]['crystal_name']
        best_margin = df.iloc[0]['profit_margin_after_fees']
        st.metric(
            "Highest Margin Crystal",
            best_crystal,
            f"{best_margin:.2f}%"
        )

    with summary_col3:
        best_profit = df.iloc[0]['profit_after_fees']
        st.metric(
            "Highest Unit Profit",
            f"{best_profit:,.0f} ISK"
        )

    with summary_col4:
        profitable_count = len(df[df['profit_after_fees'] > 0])
        st.metric(
            "Profitable Crystals",
            f"{profitable_count}/{len(df)}"
        )

    st.divider()

    st.subheader("📋 Detailed Profitability Data")

    size_filter = st.multiselect(
        "Size Filter",
        options=['S (Small)', 'M (Medium)', 'L (Large)', 'Rig / Other'],
        default=['S (Small)', 'M (Medium)', 'Rig / Other']
    )

    filtered_df = df.copy()
    size_codes = []
    if 'S (Small)' in size_filter:
        size_codes.append('S')
    if 'M (Medium)' in size_filter:
        size_codes.append('M')
    if 'L (Large)' in size_filter:
        size_codes.append('L')

    if size_codes:
        pattern = '|'.join([f' {s}$' for s in size_codes])
        if 'Rig / Other' in size_filter:
            filtered_df = filtered_df[
                filtered_df['crystal_name'].str.contains(pattern) |
                ~filtered_df['crystal_name'].str.contains(r' [SML]$')
            ]
        else:
            filtered_df = filtered_df[filtered_df['crystal_name'].str.contains(pattern)]
    elif 'Rig / Other' in size_filter:
        filtered_df = filtered_df[~filtered_df['crystal_name'].str.contains(r' [SML]$')]

    display_df = filtered_df[[
        'crystal_name', 'material_cost', 'sell_price',
        'profit_after_fees', 'profit_margin_after_fees', 'profit_10_bpc',
        'avg_daily_volume', 'sell_volume', 'output_per_bpc'
    ]].copy()

    display_df['days_to_sell'] = display_df['avg_daily_volume'] / display_df['sell_volume']
    display_df['days_to_sell'] = display_df['days_to_sell'].replace([float('inf'), -float('inf')], 0)

    display_df = display_df.sort_values(by='profit_10_bpc', ascending=False)

    display_df.columns = [
        'Item', 'Material Cost (per unit)', 'Sell Order Price',
        'Profit per unit (after fees)', 'Margin %', 'Profit (10 BPC)',
        'Avg Daily Volume (100d)', 'Sell Volume', 'Output per BPC', 'Days to Sell'
    ]

    st.dataframe(
        display_df.style.format({
            'Material Cost (per unit)': '{:,.0f}',
            'Sell Order Price': '{:,.0f}',
            'Profit per unit (after fees)': '{:,.0f}',
            'Margin %': '{:.2f}%',
            'Profit (10 BPC)': '{:,.0f}',
            'Avg Daily Volume (100d)': '{:,.0f}',
            'Sell Volume': '{:,.0f}',
            'Output per BPC': '{:,.0f}',
            'Days to Sell': '{:.2f}'
        }).background_gradient(subset=['Margin %'], cmap='RdYlGn', vmin=-10, vmax=50),
        use_container_width=True,
        height=600
    )

else:
    st.error("Unable to load data. Please check ESI API status.")

st.divider()
st.caption("""
Important Notes:
- Real-time market conditions constantly change
- Manufacturing uses 10 runs basis (40 units produced)
- Consider manufacturing time, blueprint research, facility bonuses
- Large trades may move market prices
- Fees vary by skills and standings

Data Source: EVE Online ESI API (CCP Games)
""")
