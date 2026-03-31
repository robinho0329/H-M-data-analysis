"""
상품 세그먼트 분석 페이지 - 실데이터 기반
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_dir))


@st.cache_data
def load_product_segment_data():
    """상품 세그먼트 분석용 실데이터 로드"""
    from app.utils.cache_manager import load_raw_data
    merged = load_raw_data()
    if merged is None:
        return None

    # 색상별 통계
    color_stats = merged.groupby('colour_group_name').agg(
        total_revenue=('price', 'sum'),
        transaction_count=('price', 'count'),
        avg_price=('price', 'mean'),
        unique_customers=('customer_id', 'nunique')
    ).reset_index().sort_values('total_revenue', ascending=False)

    # 제품 타입별 통계
    product_stats = merged.groupby('product_type_name').agg(
        total_revenue=('price', 'sum'),
        transaction_count=('price', 'count'),
        avg_price=('price', 'mean'),
        unique_customers=('customer_id', 'nunique')
    ).reset_index().sort_values('total_revenue', ascending=False)

    # 의류 그룹별 통계
    garment_stats = merged.groupby('garment_group_name').agg(
        total_revenue=('price', 'sum'),
        transaction_count=('price', 'count'),
        avg_price=('price', 'mean'),
        unique_customers=('customer_id', 'nunique')
    ).reset_index().sort_values('total_revenue', ascending=False)

    # 월별 시계열 (상위 항목만)
    merged['month'] = merged['t_dat'].dt.to_period('M').astype(str)
    color_ts = merged.groupby(['month', 'colour_group_name'])['price'].sum().reset_index()
    product_ts = merged.groupby(['month', 'product_type_name'])['price'].sum().reset_index()
    garment_ts = merged.groupby(['month', 'garment_group_name'])['price'].sum().reset_index()

    # 가격대 구분
    merged['price_tier'] = pd.cut(
        merged['price'],
        bins=[0, 0.02, 0.05, 0.10, float('inf')],
        labels=['저가 (～₩0.02)', '중가 (₩0.02～0.05)', '고가 (₩0.05～0.10)', '프리미엄 (₩0.10+)']
    )
    price_stats = merged.groupby('price_tier', observed=True).agg(
        total_revenue=('price', 'sum'),
        transaction_count=('price', 'count'),
        avg_price=('price', 'mean'),
        unique_customers=('customer_id', 'nunique')
    ).reset_index().rename(columns={'price_tier': 'segment'})
    price_ts = merged.groupby(['month', 'price_tier'], observed=True)['price'].sum().reset_index()

    return {
        'color': color_stats,
        'product': product_stats,
        'garment': garment_stats,
        'price': price_stats,
        'color_ts': color_ts,
        'product_ts': product_ts,
        'garment_ts': garment_ts,
        'price_ts': price_ts,
    }


def show():
    st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:11px; color:#767676; margin-bottom:12px; letter-spacing:0.3px;'>
        홈 / 분석 / <span style='color:#000000; font-weight:600;'>상품 분석</span>
    </div>
    <h1 style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;
               font-size:1.8rem; font-weight:700; color:#000000;
               margin:0 0 4px 0; letter-spacing:0.5px; text-transform:uppercase;'>
        상품 세그먼트 분석
    </h1>
    <div style='height:2px; background:#000000; width:40px; margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)

    data = load_product_segment_data()
    if data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    col_sel, col_top = st.columns([2, 1])
    with col_sel:
        segment_type = st.selectbox(
            "세그먼트 유형 선택",
            ["색상 (Color)", "제품 타입", "의류 그룹", "가격대"]
        )
    with col_top:
        top_n = st.slider("상위 N개 표시", 4, 15, 8)

    st.markdown("---")

    # 유형별 데이터 선택
    if segment_type == "색상 (Color)":
        stats_df = data['color'].head(top_n).copy()
        ts_df = data['color_ts']
        name_col = 'colour_group_name'
        label = '색상'
    elif segment_type == "제품 타입":
        stats_df = data['product'].head(top_n).copy()
        ts_df = data['product_ts']
        name_col = 'product_type_name'
        label = '제품 타입'
    elif segment_type == "의류 그룹":
        stats_df = data['garment'].head(top_n).copy()
        ts_df = data['garment_ts']
        name_col = 'garment_group_name'
        label = '의류 그룹'
    else:  # 가격대
        stats_df = data['price'].copy()
        ts_df = data['price_ts']
        name_col = 'segment'
        label = '가격대'

    top_items = stats_df[name_col].tolist()

    # KPI
    cols = st.columns(min(4, len(top_items)))
    for i, (_, row) in enumerate(stats_df.head(4).iterrows()):
        with cols[i]:
            rev_share = row['total_revenue'] / stats_df['total_revenue'].sum() * 100
            st.metric(
                label=row[name_col],
                value=f"₩{row['total_revenue']:,.2f}",
                delta=f"비중 {rev_share:.1f}%"
            )

    st.markdown("---")

    # 차트 2개
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{label}별 월별 매출 추이")

        ts_filtered = ts_df[ts_df[name_col].isin(top_items)].copy()

        fig = go.Figure()
        colors_palette = px.colors.qualitative.Plotly
        for i, item in enumerate(top_items):
            item_ts = ts_filtered[ts_filtered[name_col] == item].sort_values('month')
            fig.add_trace(go.Scatter(
                x=item_ts['month'],
                y=item_ts['price'],
                name=item,
                mode='lines+markers',
                line=dict(color=colors_palette[i % len(colors_palette)])
            ))

        fig.update_layout(
            xaxis_title="월",
            yaxis_title="매출액 (₩)",
            hovermode='x unified',
            height=420,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=-0.3)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("매출 비중")
        fig = go.Figure(data=[go.Pie(
            labels=stats_df[name_col],
            values=stats_df['total_revenue'],
            hole=0.35,
        )])
        fig.update_layout(height=420, showlegend=True,
                          legend=dict(orientation='h', yanchor='bottom', y=-0.3))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 수익성 차트
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"거래 건수 vs 매출")
        fig = go.Figure(data=[go.Bar(
            x=stats_df[name_col],
            y=stats_df['transaction_count'],
            name='거래건수',
            marker_color='#E50010',
            yaxis='y'
        )])
        fig.add_trace(go.Scatter(
            x=stats_df[name_col],
            y=stats_df['total_revenue'],
            name='총매출(₩)',
            mode='lines+markers',
            marker_color='#222222',
            yaxis='y2'
        ))
        fig.update_layout(
            yaxis=dict(title='거래건수', side='left'),
            yaxis2=dict(title='총매출(₩)', side='right', overlaying='y'),
            height=380, template='plotly_white',
            legend=dict(x=0.7, y=1.1, orientation='h')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"건당 평균 매출")
        fig = go.Figure(data=[go.Bar(
            x=stats_df[name_col],
            y=stats_df['avg_price'],
            marker=dict(
                color=stats_df['avg_price'],
                colorscale='Reds',
                showscale=True
            )
        )])
        fig.update_layout(
            yaxis_title="건당 평균(₩)",
            height=380,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 상세 테이블
    st.markdown("---")
    st.subheader("📊 상품별 상세 통계")
    display_df = stats_df[[name_col, 'total_revenue', 'transaction_count', 'avg_price', 'unique_customers']].copy()
    display_df.columns = [label, '총매출(₩)', '거래건수', '건당평균(₩)', '고객수']
    display_df['총매출(₩)'] = display_df['총매출(₩)'].map(lambda x: f"₩{x:,.4f}")
    display_df['건당평균(₩)'] = display_df['건당평균(₩)'].map(lambda x: f"₩{x:,.4f}")
    display_df['거래건수'] = display_df['거래건수'].map(lambda x: f"{x:,}건")
    display_df['고객수'] = display_df['고객수'].map(lambda x: f"{x:,}명")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 동적 인사이트
    st.markdown("---")
    st.subheader("💡 인사이트")

    total_rev = stats_df['total_revenue'].sum()
    top1 = stats_df.iloc[0]
    top1_share = top1['total_revenue'] / total_rev * 100
    top3_share = stats_df.head(3)['total_revenue'].sum() / total_rev * 100
    highest_avg = stats_df.loc[stats_df['avg_price'].idxmax()]
    lowest_rev = stats_df.iloc[-1]

    col_i1, col_i2, col_i3 = st.columns(3)

    with col_i1:
        st.success(
            f"🏆 **1위 {label}: {top1[name_col]}**\n\n"
            f"- 총 매출: **₩{top1['total_revenue']:,.4f}**\n"
            f"- 매출 비중: **{top1_share:.1f}%**\n"
            f"- 거래: **{int(top1['transaction_count']):,}건**\n\n"
            f"상위 3개가 전체의 **{top3_share:.1f}%** 차지"
        )

    with col_i2:
        st.info(
            f"💎 **건당 최고 수익: {highest_avg[name_col]}**\n\n"
            f"- 건당 평균: **₩{highest_avg['avg_price']:,.4f}**\n"
            f"- 고객 수: **{int(highest_avg['unique_customers']):,}명**\n\n"
            f"프리미엄 마케팅 및 업셀링 적합"
        )

    with col_i3:
        if segment_type == "가격대":
            low_price = stats_df[stats_df[name_col].astype(str).str.contains('저가')]
            if len(low_price) > 0:
                low_cnt = int(low_price.iloc[0]['transaction_count'])
                low_share = low_cnt / stats_df['transaction_count'].sum() * 100
                st.warning(
                    f"📦 **볼륨 vs 마진 전략**\n\n"
                    f"- 저가대 거래 비중: **{low_share:.1f}%**\n"
                    f"- 건당 최고가 대비 가격 차이 큼\n\n"
                    f"마진 개선을 위해 중고가 전환 유도 필요"
                )
            else:
                st.warning(f"📈 **성장 여력**\n\n최하위 {label} {lowest_rev[name_col]} 집중 분석 필요")
        else:
            st.warning(
                f"📈 **성장 여력: {lowest_rev[name_col]}**\n\n"
                f"- 현재 매출: **₩{lowest_rev['total_revenue']:,.4f}**\n"
                f"- 전체 비중: **{lowest_rev['total_revenue']/total_rev*100:.1f}%**\n\n"
                f"타겟 캠페인으로 볼륨 증대 가능"
            )
