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

    # Void (Hybrid Ammo) - 5,000 units per run (materials: 1 run × 10)
    'Void M': {'type_id': 12789, 'materials': {'Morphite': 6, 'R.A.M.- Ammunition Tech': 1, 'Crystalline Carbonide': 240, 'Fullerides': 240}, 'runs': 10, 'output_per_run': 5000},

    # Null (Hybrid Ammo) - 5,000 units per run (materials: 1 run × 10)
    'Null M': {'type_id': 12785, 'materials': {'Morphite': 6, 'R.A.M.- Ammunition Tech': 1, 'Crystalline Carbonide': 240, 'Fullerides': 240}, 'runs': 10, 'output_per_run': 5000},

    # Small Ionic Field Projector II (Rig) - 1 unit per run, 4 runs = 4 units (materials: 1 run × 4)
    'Small Ionic Field Projector II': {'type_id': 31280, 'materials': {'Miniature Electronics': 24, 'R.A.M.- Electronics': 4, 'Artificial Neural Network': 4, 'Micro Circuit': 4, 'Logic Circuit': 4}, 'runs': 4, 'output_per_run': 1},
}

# Material Type IDs (Materials needed for Advanced Crystal manufacturing)
MATERIALS = {
    # Base materials
    'Morphite': 11399,  # 수정: 16670 -> 11399 (verified on evemarketbrowser.com)
    'R.A.M.- Ammunition Tech': 11476,  # 수정: 11538 -> 11476
    'R.A.M.- Electronics': 11483,  # For Small Ionic Field Projector II manufacturing
    'Tungsten Carbide': 16672,
    'Fullerides': 16679,  # 수정: 16673 -> 16679 (verified on evemarketbrowser.com)
    'Crystalline Carbonide': 16670,  # For Void/Null manufacturing
    # Salvage Materials (For Rig manufacturing)
    'Miniature Electronics': 9842,  # PI material
    'Artificial Neural Network': 25616,
    'Micro Circuit': 25618,
    'Logic Circuit': 25619,
}

# Major trading hubs
TRADE_HUBS = {
    'Jita': 60003760,
    'Amarr': 60008494,
    'Dodixie': 60011866,
    'Rens': 60004588,
    'Hek': 60005686
}

@st.cache_data(ttl=600)  # 10분 캐시
def get_market_price(type_id, region_id=10000002):
    """Get market prices from ESI API (default: The Forge - Jita)"""
    try:
        # Get market orders
        url = f"https://esi.evetech.net/latest/markets/{region_id}/orders/"
        params = {'datasource': 'tranquility', 'type_id': type_id}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            orders = response.json()

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

@st.cache_data(ttl=3600)  # 1시간 캐시 (히스토리는 자주 변하지 않음)
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
    """Calculate crystal manufacturing profit (10 runs basis, materials always use Jita Sell lowest price)"""
    # Calculate material cost (always use Sell Order lowest price)
    # 10 runs = 40 units (4 per run × 10)
    material_cost_total = 0
    material_breakdown = {}

    for material, quantity in crystal_data['materials'].items():
        if material in material_prices:
            price = material_prices[material]['lowest_sell']  # Always use Sell Order lowest
            cost = price * quantity
            material_cost_total += cost
            material_breakdown[material] = {'price': price, 'quantity': quantity, 'total': cost}
        else:
            # Return None if material price not found
            return None

    # Total units produced in 10 runs
    runs = crystal_data.get('runs', 10)
    output_per_run = crystal_data.get('output_per_run', 4)
    total_output = runs * output_per_run  # 10 × 4 = 40 units

    # Material cost per unit
    material_cost_per_unit = material_cost_total / total_output

    # Crystal market price (per unit)
    crystal_price_data = get_market_price(crystal_data['type_id'])

    if not crystal_price_data:
        return None

    sell_price = crystal_price_data['lowest_sell']  # Sell at Jita Sell Order price
    buy_order_price = crystal_price_data['highest_buy']  # Instant sell price (reference)

    # 100-day average volume
    avg_daily_volume = get_market_history(crystal_data['type_id'])

    # Profit calculation (per unit - sell via Sell Order)
    profit_per_unit = sell_price - material_cost_per_unit
    profit_margin = (profit_per_unit / material_cost_per_unit * 100) if material_cost_per_unit > 0 else 0

    # Total profit (40 units - 10 runs)
    total_profit = profit_per_unit * total_output
    total_revenue = sell_price * total_output

    # Profit calculation for 10 BPC (Blueprint Copy)
    # Crystal: 1 BPC = 10 runs = 40 units → 10 BPC = 100 runs = 400 units
    # Rig: 1 BPC = 4 runs = 4 units → 10 BPC = 40 runs = 40 units
    bpc_count = 10
    output_10_bpc = bpc_count * total_output  # 10 BPC = 10 × (runs × output_per_run)
    material_cost_10_bpc = material_cost_per_unit * output_10_bpc
    profit_10_bpc = profit_per_unit * output_10_bpc

    return {
        'crystal_name': crystal_name,
        'material_cost': material_cost_per_unit,
        'material_cost_total': material_cost_total,
        'material_breakdown': material_breakdown,
        'output_count': total_output,
        'sell_price': sell_price,  # Sell Order price (per unit)
        'buy_order_price': buy_order_price,  # Buy Order price (instant sell, per unit)
        'total_revenue': total_revenue,  # Total revenue (40 units)
        'total_profit': total_profit,  # Total profit (40 units)
        'profit': profit_per_unit,
        'profit_margin': profit_margin,
        'buy_volume': crystal_price_data['buy_volume'],
        'sell_volume': crystal_price_data['sell_volume'],
        'lowest_sell': crystal_price_data['lowest_sell'],
        'avg_daily_volume': avg_daily_volume,  # 100-day average volume
        'profit_10_bpc': profit_10_bpc,  # 10 BPC profit (Crystal: 400 units, Rig: 40 units)
        'material_cost_10_bpc': material_cost_10_bpc,  # 10 BPC material cost
        'output_10_bpc': output_10_bpc,  # 10 BPC output
        'output_per_bpc': total_output  # Output per BPC (Crystal: 40, Rig: 4)
    }

# -----------------------------
# UI
# -----------------------------
st.title("💎 EVE Online - Advanced Crystal Manufacturing Profit Calculator")
st.caption("Advanced Frequency Crystal manufacturing profitability analysis - Material prices from Jita Sell Orders")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    st.info("**Material Purchase:**\nJita Sell Order (lowest price)\n\n**Production:**\n1 BPC = Crystal: 40 units / Rig: 4 units")

    st.markdown("---")

    # Fee Settings
    st.subheader("Fee Settings")

    broker_fee = st.slider(
        "Broker Fee (%)",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Station trading broker fee"
    )

    sales_tax = st.slider(
        "Sales Tax (%)",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1,
        help="Sales tax"
    )

    st.markdown("---")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("**Data Source:**")
    st.caption("EVE Online ESI API")
    st.caption("Refresh: Every 10 minutes")

# -----------------------------
# Data Loading
# -----------------------------
st.header("📊 Loading Market Data")

with st.spinner("Loading material prices..."):
    material_prices = {}
    for material_name, type_id in MATERIALS.items():
        price_data = get_market_price(type_id)
        if price_data:
            material_prices[material_name] = price_data
        time.sleep(0.2)  # API rate limit

# Material Prices Display
if material_prices:
    st.subheader("🔧 Material Prices (Jita Sell Order - Lowest)")

    # Base materials 표시
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

    # R.A.M. materials
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

    # Rig manufacturing materials
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

st.divider()

# Crystal Profit Calculation
st.header("💰 Crystal Manufacturing Profitability")

with st.spinner("Loading crystal market data... (this may take a while)"):
    profit_data = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (crystal_name, crystal_data) in enumerate(CRYSTALS.items()):
        status_text.text(f"Loading: {crystal_name} ({idx+1}/{len(CRYSTALS)})")

        profit_info = calculate_profit(crystal_name, crystal_data, material_prices)
        if profit_info:
            # Apply fees
            total_fees = (broker_fee + sales_tax) / 100
            profit_info['profit_after_fees'] = profit_info['profit'] * (1 - total_fees)
            profit_info['profit_margin_after_fees'] = (profit_info['profit_after_fees'] / profit_info['material_cost'] * 100) if profit_info['material_cost'] > 0 else 0

            profit_data.append(profit_info)

        progress_bar.progress((idx + 1) / len(CRYSTALS))
        time.sleep(0.3)  # API rate limit

    progress_bar.empty()
    status_text.empty()

if profit_data:
    df = pd.DataFrame(profit_data)
    df = df.sort_values('profit_margin_after_fees', ascending=False)

    # Summary statistics
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

    # Detailed table
    st.subheader("📋 Detailed Profitability Data")

    # Size filter
    size_filter = st.multiselect(
        "Size Filter",
        options=['S (Small)', 'M (Medium)', 'L (Large)', 'Rig / Other'],
        default=['S (Small)', 'M (Medium)', 'Rig / Other']
    )

    # Apply filter
    filtered_df = df.copy()
    size_codes = []
    if 'S (Small)' in size_filter:
        size_codes.append('S')
    if 'M (Medium)' in size_filter:
        size_codes.append('M')
    if 'L (Large)' in size_filter:
        size_codes.append('L')

    if size_codes:
        # Advanced crystals format: "Conflagration S", "Scorch M", "Aurora L"
        # Rig/Other are items without size codes
        pattern = '|'.join([f' {s}$' for s in size_codes])
        if 'Rig / Other' in size_filter:
            # Items with size codes OR without size codes (Rig/Other)
            filtered_df = filtered_df[
                filtered_df['crystal_name'].str.contains(pattern) |
                ~filtered_df['crystal_name'].str.contains(r' [SML]$')
            ]
        else:
            filtered_df = filtered_df[filtered_df['crystal_name'].str.contains(pattern)]
    elif 'Rig / Other' in size_filter:
        # Only Rig/Other selected
        filtered_df = filtered_df[~filtered_df['crystal_name'].str.contains(r' [SML]$')]

    # Display table
    display_df = filtered_df[[
        'crystal_name', 'material_cost', 'sell_price',
        'profit_after_fees', 'profit_margin_after_fees', 'profit_10_bpc',
        'avg_daily_volume', 'sell_volume', 'output_per_bpc'
    ]].copy()

    # Calculate Days to Sell (Avg Daily Volume / Sell Volume)
    display_df['days_to_sell'] = display_df['avg_daily_volume'] / display_df['sell_volume']
    display_df['days_to_sell'] = display_df['days_to_sell'].replace([float('inf'), -float('inf')], 0)
    
    # Sort by profit_10_bpc descending
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

# Footer
st.divider()
st.caption("""
**Important Notes:**
- This data reflects real-time market conditions and can change rapidly
- Manufacturing calculated on 10 runs basis (40 units produced)
- Actual manufacturing should consider manufacturing time, blueprint research level, facility bonuses, etc.
- Large trades may move market prices - use caution
- Fees may vary based on skills and standings

**Data Source:** EVE Online ESI API (CCP Games)
""")
