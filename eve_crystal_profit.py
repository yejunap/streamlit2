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

# 레이저 크리스탈 타입 ID 매핑 (Advanced Frequency Crystals - Tech 2)
# 주의: blueprint 10 runs = 40개 생산 (1 run당 4개 × 10 runs)
# Void/Null은 1 run당 5,000개 생산 (10 runs = 50,000개)
# 재료는 1 run 기준 × 10
CRYSTALS = {
    # Conflagration (Advanced X-Ray)
    'Conflagration S': {'type_id': 12565, 'materials': {'Morphite': 10, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 1080, 'Fullerides': 450}, 'runs': 10, 'output_per_run': 4},
    'Conflagration M': {'type_id': 12814, 'materials': {'Morphite': 60, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 4320, 'Fullerides': 1800}, 'runs': 10, 'output_per_run': 4},
    'Conflagration L': {'type_id': 12816, 'materials': {'Morphite': 150, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 10800, 'Fullerides': 4500}, 'runs': 10, 'output_per_run': 4},

    # Scorch (Advanced Multifrequency)
    'Scorch S': {'type_id': 12563, 'materials': {'Morphite': 10, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 1080, 'Fullerides': 450}, 'runs': 10, 'output_per_run': 4},
    'Scorch M': {'type_id': 12818, 'materials': {'Morphite': 60, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 4320, 'Fullerides': 1800}, 'runs': 10, 'output_per_run': 4},
    'Scorch L': {'type_id': 12820, 'materials': {'Morphite': 150, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 10800, 'Fullerides': 4500}, 'runs': 10, 'output_per_run': 4},

    # Aurora (Advanced Radio) - Tungsten Carbide가 더 많이 필요
    'Aurora S': {'type_id': 12559, 'materials': {'Morphite': 10, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 2610, 'Fullerides': 450}, 'runs': 10, 'output_per_run': 4},
    'Aurora M': {'type_id': 12822, 'materials': {'Morphite': 60, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 10440, 'Fullerides': 1800}, 'runs': 10, 'output_per_run': 4},
    'Aurora L': {'type_id': 12824, 'materials': {'Morphite': 150, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 26100, 'Fullerides': 4500}, 'runs': 10, 'output_per_run': 4},

    # Gleam (Advanced Infrared) - Tungsten Carbide가 더 많이 필요
    'Gleam S': {'type_id': 12557, 'materials': {'Morphite': 10, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 2610, 'Fullerides': 450}, 'runs': 10, 'output_per_run': 4},
    'Gleam M': {'type_id': 12826, 'materials': {'Morphite': 60, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 10440, 'Fullerides': 1800}, 'runs': 10, 'output_per_run': 4},
    'Gleam L': {'type_id': 12828, 'materials': {'Morphite': 150, 'R.A.M.- Ammunition Tech': 10, 'Tungsten Carbide': 26100, 'Fullerides': 4500}, 'runs': 10, 'output_per_run': 4},

    # Void (Hybrid Ammo) - 1 run당 5,000개 생산 (재료는 1 run 기준 × 10)
    'Void M': {'type_id': 12789, 'materials': {'Morphite': 6, 'R.A.M.- Ammunition Tech': 1, 'Crystalline Carbonide': 240, 'Fullerides': 240}, 'runs': 10, 'output_per_run': 5000},

    # Null (Hybrid Ammo) - 1 run당 5,000개 생산 (재료는 1 run 기준 × 10)
    'Null M': {'type_id': 12785, 'materials': {'Morphite': 6, 'R.A.M.- Ammunition Tech': 1, 'Crystalline Carbonide': 240, 'Fullerides': 240}, 'runs': 10, 'output_per_run': 5000},
}

# 재료 타입 ID (Advanced Crystal 제조에 필요한 재료들)
MATERIALS = {
    # 기본 재료
    'Morphite': 11399,  # 수정: 16670 -> 11399 (evemarketbrowser.com 확인)
    'R.A.M.- Ammunition Tech': 11476,  # 수정: 11538 -> 11476
    'Tungsten Carbide': 16672,
    'Fullerides': 16679,  # 수정: 16673 -> 16679 (evemarketbrowser.com 확인)
    'Crystalline Carbonide': 16670,  # Void/Null 제조용
}

# 주요 거래 허브
TRADE_HUBS = {
    'Jita': 60003760,
    'Amarr': 60008494,
    'Dodixie': 60011866,
    'Rens': 60004588,
    'Hek': 60005686
}

@st.cache_data(ttl=600)  # 10분 캐시
def get_market_price(type_id, region_id=10000002):
    """ESI API로 시장 가격 가져오기 (기본: The Forge - Jita)"""
    try:
        # 시장 주문 가져오기
        url = f"https://esi.evetech.net/latest/markets/{region_id}/orders/"
        params = {'datasource': 'tranquility', 'type_id': type_id}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            orders = response.json()

            # 매수/매도 주문 분리
            buy_orders = [o for o in orders if o['is_buy_order']]
            sell_orders = [o for o in orders if not o['is_buy_order']]

            # 최고 매수가, 최저 매도가
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
        st.warning(f"Type ID {type_id} 가격 조회 실패: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # 1시간 캐시 (히스토리는 자주 변하지 않음)
def get_market_history(type_id, region_id=10000002, days=100):
    """ESI API로 시장 거래 히스토리 가져오기 (100일 평균 거래량)"""
    try:
        url = f"https://esi.evetech.net/latest/markets/{region_id}/history/"
        params = {'datasource': 'tranquility', 'type_id': type_id}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            history = response.json()

            # 최근 100일 데이터만 사용
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
    """특정 스테이션의 시장 가격"""
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
    """크리스탈 제조 수익 계산 (10 runs 기준, 재료는 항상 Jita Sell 최저가 사용)"""
    # 재료 비용 계산 (항상 Sell Order 최저가 사용)
    # 10 runs = 40개 생산 (1 run당 4개 × 10)
    material_cost_total = 0
    material_breakdown = {}

    for material, quantity in crystal_data['materials'].items():
        if material in material_prices:
            price = material_prices[material]['lowest_sell']  # 항상 Sell Order 최저가
            cost = price * quantity
            material_cost_total += cost
            material_breakdown[material] = {'price': price, 'quantity': quantity, 'total': cost}
        else:
            # 재료 가격을 찾을 수 없으면 None 반환
            return None

    # 10 runs로 생산되는 총 개수
    runs = crystal_data.get('runs', 10)
    output_per_run = crystal_data.get('output_per_run', 4)
    total_output = runs * output_per_run  # 10 × 4 = 40개

    # 1개당 재료 비용
    material_cost_per_unit = material_cost_total / total_output

    # 크리스탈 시장 가격 (1개 가격)
    crystal_price_data = get_market_price(crystal_data['type_id'])

    if not crystal_price_data:
        return None

    sell_price = crystal_price_data['lowest_sell']  # Jita Sell Order 가격으로 판매
    buy_order_price = crystal_price_data['highest_buy']  # 즉시 판매 가격 (참고용)

    # 100일 평균 거래량
    avg_daily_volume = get_market_history(crystal_data['type_id'])

    # 수익 계산 (1개 기준 - Sell Order로 판매)
    profit_per_unit = sell_price - material_cost_per_unit
    profit_margin = (profit_per_unit / material_cost_per_unit * 100) if material_cost_per_unit > 0 else 0

    # 총 수익 (40개 기준 - 10 runs)
    total_profit = profit_per_unit * total_output
    total_revenue = sell_price * total_output

    # 100 runs (10 생산 라인) 수익 = 400개
    runs_100 = 100
    output_100_runs = runs_100 * output_per_run  # 100 × 4 = 400개
    material_cost_100_runs = material_cost_per_unit * output_100_runs
    profit_100_runs = profit_per_unit * output_100_runs

    return {
        'crystal_name': crystal_name,
        'material_cost': material_cost_per_unit,
        'material_cost_total': material_cost_total,
        'material_breakdown': material_breakdown,
        'output_count': total_output,
        'sell_price': sell_price,  # Sell Order 가격 (1개)
        'buy_order_price': buy_order_price,  # Buy Order 가격 (즉시 판매, 1개)
        'total_revenue': total_revenue,  # 총 수익 (40개)
        'total_profit': total_profit,  # 총 이익 (40개)
        'profit': profit_per_unit,
        'profit_margin': profit_margin,
        'buy_volume': crystal_price_data['buy_volume'],
        'sell_volume': crystal_price_data['sell_volume'],
        'lowest_sell': crystal_price_data['lowest_sell'],
        'avg_daily_volume': avg_daily_volume,  # 100일 평균 거래량
        'profit_100_runs': profit_100_runs,  # 100 runs (400개) 수익
        'material_cost_100_runs': material_cost_100_runs  # 100 runs 재료 비용
    }

# -----------------------------
# UI
# -----------------------------
st.title("💎 EVE Online - Advanced Crystal Manufacturing Profit Calculator")
st.caption("Advanced Frequency Crystal 제조 수익성 분석 (10 runs = 40개 생산) - 재료는 Jita Sell 최저가")

# Sidebar
with st.sidebar:
    st.header("⚙️ 설정")

    st.info("**재료 구매 방식:**\nJita Sell Order 최저가 사용\n\n**생산 방식:**\n10 runs = 40개 생산")

    st.markdown("---")

    # 수수료 설정
    st.subheader("수수료 설정")

    broker_fee = st.slider(
        "Broker Fee (%)",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="스테이션 거래 수수료"
    )

    sales_tax = st.slider(
        "Sales Tax (%)",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1,
        help="판매 세금"
    )

    st.markdown("---")

    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("**데이터 출처:**")
    st.caption("EVE Online ESI API")
    st.caption("갱신 주기: 10분")

# -----------------------------
# 데이터 로딩
# -----------------------------
st.header("📊 시장 데이터 로딩")

with st.spinner("재료 가격 조회 중..."):
    material_prices = {}
    for material_name, type_id in MATERIALS.items():
        price_data = get_market_price(type_id)
        if price_data:
            material_prices[material_name] = price_data
        time.sleep(0.2)  # API rate limit

# 재료 가격 표시
if material_prices:
    st.subheader("🔧 재료 가격 (Jita Sell Order 최저가)")

    # 기본 재료 표시
    mat_col1, mat_col2, mat_col3, mat_col4 = st.columns(4)

    with mat_col1:
        if 'Morphite' in material_prices:
            st.metric(
                "Morphite",
                f"{material_prices['Morphite']['lowest_sell']:,.2f} ISK",
                delta="희귀 광물"
            )

    with mat_col2:
        if 'Tungsten Carbide' in material_prices:
            st.metric(
                "Tungsten Carbide",
                f"{material_prices['Tungsten Carbide']['lowest_sell']:,.2f} ISK",
                delta="기본 재료"
            )

    with mat_col3:
        if 'Fullerides' in material_prices:
            st.metric(
                "Fullerides",
                f"{material_prices['Fullerides']['lowest_sell']:,.2f} ISK",
                delta="기본 재료"
            )

    with mat_col4:
        if 'R.A.M.- Ammunition Tech' in material_prices:
            st.metric(
                "R.A.M.- Ammo Tech",
                f"{material_prices['R.A.M.- Ammunition Tech']['lowest_sell']:,.2f} ISK",
                delta="Tech 2 재료"
            )

    st.write("**XL 크리스탈 제조용 Tech 1 XL 크리스탈:**")
    xl_col1, xl_col2 = st.columns(2)
    with xl_col1:
        for crystal in ['X-Ray XL', 'Multifrequency XL']:
            if crystal in material_prices:
                st.text(f"{crystal}: {material_prices[crystal]['lowest_sell']:,.0f} ISK")
    with xl_col2:
        for crystal in ['Radio XL', 'Infrared XL']:
            if crystal in material_prices:
                st.text(f"{crystal}: {material_prices[crystal]['lowest_sell']:,.0f} ISK")

st.divider()

# 크리스탈 수익 계산
st.header("💰 크리스탈 제조 수익성")

with st.spinner("크리스탈 시장 데이터 조회 중... (시간이 걸릴 수 있습니다)"):
    profit_data = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (crystal_name, crystal_data) in enumerate(CRYSTALS.items()):
        status_text.text(f"조회 중: {crystal_name} ({idx+1}/{len(CRYSTALS)})")

        profit_info = calculate_profit(crystal_name, crystal_data, material_prices)
        if profit_info:
            # 수수료 적용
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

    # 요약 통계
    st.subheader("📈 수익성 요약")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric(
            "평균 수익률",
            f"{df['profit_margin_after_fees'].mean():.2f}%"
        )

    with summary_col2:
        best_crystal = df.iloc[0]['crystal_name']
        best_margin = df.iloc[0]['profit_margin_after_fees']
        st.metric(
            "최고 수익률 크리스탈",
            best_crystal,
            f"{best_margin:.2f}%"
        )

    with summary_col3:
        best_profit = df.iloc[0]['profit_after_fees']
        st.metric(
            "최고 단위 수익",
            f"{best_profit:,.0f} ISK"
        )

    with summary_col4:
        profitable_count = len(df[df['profit_after_fees'] > 0])
        st.metric(
            "수익 가능 크리스탈",
            f"{profitable_count}/{len(df)}"
        )

    st.divider()

    # 수익률 차트
    st.subheader("📊 크리스탈별 수익률 비교")

    fig_margin = go.Figure()

    colors = ['green' if x > 0 else 'red' for x in df['profit_margin_after_fees']]

    fig_margin.add_trace(go.Bar(
        x=df['crystal_name'],
        y=df['profit_margin_after_fees'],
        text=df['profit_margin_after_fees'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        marker_color=colors,
        name='Profit Margin'
    ))

    fig_margin.update_layout(
        title="수익률 비교 (수수료 적용 후)",
        xaxis_title="Crystal Type",
        yaxis_title="Profit Margin (%)",
        height=500,
        margin=dict(t=80, b=120, l=60, r=60),
        xaxis_tickangle=-45
    )

    fig_margin.add_hline(y=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig_margin, use_container_width=True)

    # 절대 수익 차트
    st.subheader("💵 크리스탈별 절대 수익 비교 (10 runs = 40개)")

    fig_profit = go.Figure()

    fig_profit.add_trace(go.Bar(
        x=df['crystal_name'],
        y=df['total_profit'],
        text=df['total_profit'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside',
        marker_color='lightblue',
        name='Total Profit (40개)'
    ))

    fig_profit.update_layout(
        title="총 수익 비교 (10 runs = 40개, 수수료 적용 전)",
        xaxis_title="Crystal Type",
        yaxis_title="Total Profit (ISK)",
        height=500,
        margin=dict(t=80, b=120, l=60, r=60),
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_profit, use_container_width=True)

    # 상세 테이블
    st.subheader("📋 상세 수익성 데이터")

    # 크기별 필터
    size_filter = st.multiselect(
        "크기 필터",
        options=['S (Small)', 'M (Medium)', 'L (Large)', 'XL (Extra Large)'],
        default=['S (Small)', 'M (Medium)', 'L (Large)', 'XL (Extra Large)']
    )

    # 필터 적용
    filtered_df = df.copy()
    size_codes = []
    if 'S (Small)' in size_filter:
        size_codes.append('S')
    if 'M (Medium)' in size_filter:
        size_codes.append('M')
    if 'L (Large)' in size_filter:
        size_codes.append('L')
    if 'XL (Extra Large)' in size_filter:
        size_codes.append('XL')

    if size_codes:
        # Advanced 크리스탈은 "Conflagration S", "Scorch M", "Aurora L", "Gleam XL" 형식
        filtered_df = filtered_df[filtered_df['crystal_name'].str.contains('|'.join([f' {s}$' for s in size_codes]))]

    # 테이블 표시
    display_df = filtered_df[[
        'crystal_name', 'material_cost', 'sell_price',
        'profit_after_fees', 'profit_margin_after_fees', 'profit_100_runs',
        'avg_daily_volume', 'sell_volume'
    ]].copy()

    display_df.columns = [
        'Crystal', 'Material Cost (1개)', 'Sell Order Price',
        'Profit per unit (after fees)', 'Margin %', 'Profit (100 runs = 400개)',
        '100일 평균 거래량/일', 'Sell Volume'
    ]

    st.dataframe(
        display_df.style.format({
            'Material Cost (1개)': '{:,.0f}',
            'Sell Order Price': '{:,.0f}',
            'Profit per unit (after fees)': '{:,.0f}',
            'Margin %': '{:.2f}%',
            'Profit (100 runs = 400개)': '{:,.0f}',
            '100일 평균 거래량/일': '{:,.0f}',
            'Sell Volume': '{:,.0f}'
        }).background_gradient(subset=['Margin %'], cmap='RdYlGn', vmin=-10, vmax=50),
        use_container_width=True,
        height=600
    )

    # 크기별 비교
    st.subheader("📏 크기별 수익률 비교")

    # Advanced 크리스탈은 이름 끝에 "S", "M", "L", "XL"이 있음
    df['size'] = df['crystal_name'].str.extract(r' (S|M|L|XL)$')[0]
    size_comparison = df.groupby('size').agg({
        'profit_margin_after_fees': 'mean',
        'profit_after_fees': 'mean',
        'material_cost': 'mean'
    }).reset_index()

    # 크기 순서 정렬 (S, M, L, XL)
    size_order = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
    size_comparison['size_order'] = size_comparison['size'].map(size_order)
    size_comparison = size_comparison.sort_values('size_order').drop('size_order', axis=1)

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        fig_size_margin = px.bar(
            size_comparison,
            x='size',
            y='profit_margin_after_fees',
            title='크기별 평균 수익률',
            labels={'size': 'Crystal Size', 'profit_margin_after_fees': 'Avg Profit Margin (%)'},
            color='profit_margin_after_fees',
            color_continuous_scale='RdYlGn'
        )
        fig_size_margin.update_layout(height=400)
        st.plotly_chart(fig_size_margin, use_container_width=True)

    with comp_col2:
        fig_size_profit = px.bar(
            size_comparison,
            x='size',
            y='profit_after_fees',
            title='크기별 평균 단위 수익',
            labels={'size': 'Crystal Size', 'profit_after_fees': 'Avg Profit (ISK)'},
            color='profit_after_fees',
            color_continuous_scale='Blues'
        )
        fig_size_profit.update_layout(height=400)
        st.plotly_chart(fig_size_profit, use_container_width=True)

    # 추천
    st.divider()
    st.header("💡 제조 추천")

    top_5 = df.head(5)

    st.write("**수익률 기준 Top 5 추천 Advanced 크리스탈:**")
    for idx, row in top_5.iterrows():
        with st.expander(f"#{top_5.index.get_loc(idx)+1}: {row['crystal_name']} - {row['profit_margin_after_fees']:.2f}% 수익률"):
            rec_col1, rec_col2, rec_col3 = st.columns(3)

            with rec_col1:
                st.write("**비용 정보**")
                st.metric("10 runs 재료 비용", f"{row['material_cost_total']:,.0f} ISK")
                st.metric("100 runs 재료 비용", f"{row['material_cost_100_runs']:,.0f} ISK")
                st.metric("1개당 재료 비용", f"{row['material_cost']:,.0f} ISK")
                st.metric("판매 가격 (Sell Order, 1개)", f"{row['sell_price']:,.0f} ISK")

                # 재료 상세
                if 'material_breakdown' in row and row['material_breakdown']:
                    st.write("**재료 상세 (10 runs):**")
                    for mat, details in row['material_breakdown'].items():
                        st.caption(f"• {mat}: {details['quantity']}개 × {details['price']:,.0f} = {details['total']:,.0f} ISK")

            with rec_col2:
                st.write("**수익 정보 (Sell Order 기준)**")
                st.metric("1개당 수익 (수수료 후)", f"{row['profit_after_fees']:,.0f} ISK")
                st.metric("수익률", f"{row['profit_margin_after_fees']:.2f}%")
                st.metric("10 runs (40개) 총 수익", f"{row['total_profit']:,.0f} ISK")

                # 100 runs 수익 (수수료 적용)
                profit_100_runs_after_fees = row['profit_100_runs'] * (1 - (broker_fee + sales_tax) / 100)
                st.metric(
                    "100 runs (400개) 총 수익",
                    f"{profit_100_runs_after_fees:,.0f} ISK",
                    delta="10 생산 라인"
                )

            with rec_col3:
                st.write("**시장 유동성**")
                st.metric("100일 평균 거래량/일", f"{row['avg_daily_volume']:,.0f}")
                st.metric("현재 매도 주문량", f"{row['sell_volume']:,.0f}")

                # 시장 유동성 평가 (평균 거래량 기준)
                if row['avg_daily_volume'] > 1000:
                    st.success("✅ 높은 유동성 (빠른 판매 가능)")
                elif row['avg_daily_volume'] > 500:
                    st.info("📊 중간 유동성")
                else:
                    st.warning("⚠️ 낮은 유동성 (판매에 시간 소요)")

else:
    st.error("데이터를 불러올 수 없습니다. ESI API 상태를 확인해주세요.")

# Footer
st.divider()
st.caption("""
**주의사항:**
- 이 데이터는 실시간 시장 상황을 반영하며, 빠르게 변동될 수 있습니다
- 제조는 10 runs 기준 (40개 생산)으로 계산됩니다
- 실제 제조 시 제조 시간, 블루프린트 연구 레벨, 시설 보너스 등을 고려해야 합니다
- 대량 거래 시 시장 가격이 움직일 수 있으므로 주의하세요
- 수수료는 스킬과 스탠딩에 따라 달라질 수 있습니다

**데이터 출처:** EVE Online ESI API (CCP Games)
""")
