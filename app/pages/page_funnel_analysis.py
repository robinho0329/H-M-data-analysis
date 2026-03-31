"""
퍼널 분석 페이지
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# 프로젝트 경로 설정
project_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_dir))

from config import PROCESSED_DIR

@st.cache_data
def load_funnel_data():
    """퍼널 분석 결과 로드"""
    results_dir = PROCESSED_DIR / "analysis"

    try:
        data = {
            'lifecycle': pd.read_csv(results_dir / "funnel_lifecycle.csv"),
            'frequency': pd.read_csv(results_dir / "funnel_frequency.csv"),
            'category': pd.read_csv(results_dir / "funnel_category.csv"),
            'channel': pd.read_csv(results_dir / "funnel_channel.csv")
        }
    except FileNotFoundError:
        st.warning("⚠️ 퍼널 분석 데이터를 찾을 수 없습니다. 분석 파이프라인을 먼저 실행해주세요.")
        return None

    return data


def display_funnel_chart(data, x_col, y_col, title, color_col=None):
    """퍼널 차트 표시"""
    fig = go.Figure(data=[go.Funnel(
        y=data[y_col],
        x=data[x_col],
        marker=dict(color=data[color_col] if color_col else None),
        textposition="inside",
        textinfo="value+percent previous"
    )])

    fig.update_layout(
        title=title,
        height=500,
        font=dict(size=12),
        showlegend=False
    )

    return fig


def show():
    st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:11px; color:#767676; margin-bottom:12px; letter-spacing:0.3px;'>
        Home / Analytics / <span style='color:#000000; font-weight:600;'>FUNNEL ANALYSIS</span>
    </div>
    <h1 style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;
               font-size:1.8rem; font-weight:700; color:#000000;
               margin:0 0 4px 0; letter-spacing:0.5px; text-transform:uppercase;'>
        FUNNEL ANALYSIS
    </h1>
    <div style='height:2px; background:#000000; width:40px; margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)

    # 데이터 로드
    funnel_data = load_funnel_data()
    if funnel_data is None:
        return

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 라이프사이클",
        "🔄 구매 빈도",
        "📦 제품 다양성",
        "🛍️ 판매 채널"
    ])

    # ==================== Tab 1: 라이프사이클 ====================
    with tab1:
        st.subheader("고객 라이프사이클 퍼널")

        lifecycle_df = funnel_data['lifecycle'].copy()
        lifecycle_df = lifecycle_df.sort_values('customer_count', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            # 퍼널 차트
            fig = go.Figure(data=[go.Funnel(
                y=lifecycle_df['lifecycle'],
                x=lifecycle_df['customer_count'],
                textposition="inside",
                textinfo="value+percent previous",
                marker=dict(color=['#E50010', '#222222', '#CC0000', '#FF6B6B'])
            )])
            fig.update_layout(title="고객 라이프사이클 단계", height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 지표
            st.markdown("### 📊 주요 지표")

            for _, row in lifecycle_df.iterrows():
                with st.container():
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric(row['lifecycle'], f"{int(row['customer_count']):,}명")
                    with col_b:
                        st.metric("", f"{row['percentage']:.1f}%")
                    with col_c:
                        st.metric("", f"₩{row['total_revenue']:,.0f}")

        # 상세 테이블
        st.markdown("### 📋 상세 현황")
        st.dataframe(
            lifecycle_df[[
                'lifecycle', 'customer_count', 'percentage',
                'total_revenue', 'avg_customer_value', 'avg_purchases'
            ]].rename(columns={
                'lifecycle': '라이프사이클',
                'customer_count': '고객수',
                'percentage': '비율(%)',
                'total_revenue': '총매출',
                'avg_customer_value': '고객당평균값',
                'avg_purchases': '고객당평균구매'
            }),
            use_container_width=True,
            hide_index=True
        )

        # 인사이트
        st.markdown("### 💡 인사이트")
        active_pct = lifecycle_df[lifecycle_df['lifecycle'] == 'Active']['percentage'].values[0]
        new_pct = lifecycle_df[lifecycle_df['lifecycle'] == 'New']['percentage'].values[0]

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.info(f"🆕 신규 고객 비율\n{new_pct:.1f}%\n\n30일 이내 첫 구매 고객이 대부분")
        with col_i2:
            st.warning(f"⚠️ 활성 고객 비율\n{active_pct:.1f}%\n\n최근 30일 구매 고객이 매우 적음")
        with col_i3:
            st.error(f"❌ 재활성화 필요\n{100-active_pct:.1f}%\n\n휴면/비활성 고객의 재구매 유도 필요")

    # ==================== Tab 2: 구매 빈도 ====================
    with tab2:
        st.subheader("고객 구매 빈도 분포")

        frequency_df = funnel_data['frequency'].copy()
        frequency_df = frequency_df.sort_values('customer_count', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            # 퍼널 차트
            fig = go.Figure(data=[go.Funnel(
                y=frequency_df['frequency'],
                x=frequency_df['customer_count'],
                textposition="inside",
                textinfo="value+percent previous",
                marker=dict(color=['#E50010', '#222222', '#CC0000', '#FF6B6B'])
            )])
            fig.update_layout(title="구매 빈도 퍼널", height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 누적 비율 차트
            fig = go.Figure(data=[go.Bar(
                y=frequency_df['frequency'],
                x=frequency_df['cumulative_pct'],
                orientation='h',
                marker=dict(color='#E50010')
            )])
            fig.update_layout(
                title="누적 고객 비율",
                xaxis_title="누적 %",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # 상세 테이블
        st.markdown("### 📋 구매 빈도 현황")
        st.dataframe(
            frequency_df[[
                'frequency', 'customer_count', 'percentage', 'cumulative_pct',
                'avg_purchase_count', 'total_revenue', 'avg_customer_value'
            ]].rename(columns={
                'frequency': '구매빈도',
                'customer_count': '고객수',
                'percentage': '비율(%)',
                'cumulative_pct': '누적(%)',
                'avg_purchase_count': '평균구매수',
                'total_revenue': '총매출',
                'avg_customer_value': '고객당평균값'
            }),
            use_container_width=True,
            hide_index=True
        )

        # 인사이트
        st.markdown("### 💡 인사이트")
        one_time_pct = frequency_df[frequency_df['frequency'] == 'One-time']['percentage'].values[0]
        frequent_pct = frequency_df[frequency_df['frequency'] == 'Frequent']['percentage'].values[0]

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.error(f"🚨 1회 구매 고객\n{one_time_pct:.1f}%\n\n거의 절반이 한 번만 구매")
        with col_i2:
            st.info(f"✅ 충성도 높은 고객\n{frequent_pct:.1f}%\n\n정기 구매자는 극소수")
        with col_i3:
            st.warning(f"⚡ 재구매 유도 필요\n{100-one_time_pct:.1f}%\n\n2회 이상 구매 전환율 개선 필수")

    # ==================== Tab 3: 제품 다양성 ====================
    with tab3:
        st.subheader("고객 제품 다양성 분포")

        category_df = funnel_data['category'].copy()
        category_df = category_df.sort_values('customer_count', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            # 원형 차트
            fig = go.Figure(data=[go.Pie(
                labels=category_df['diversity'],
                values=category_df['customer_count'],
                hole=0.4,
                marker=dict(colors=['#E50010', '#222222', '#CC0000'])
            )])
            fig.update_layout(title="제품 다양성 분포", height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 막대 차트
            fig = go.Figure(data=[go.Bar(
                x=category_df['diversity'],
                y=category_df['avg_customer_value'],
                marker=dict(color=['#E50010', '#222222', '#CC0000'])
            )])
            fig.update_layout(
                title="다양성별 평균 고객가치",
                xaxis_title="다양성",
                yaxis_title="고객당 평균값",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # 상세 테이블
        st.markdown("### 📋 제품 다양성 분석")
        st.dataframe(
            category_df[[
                'diversity', 'customer_count', 'percentage',
                'avg_product_groups', 'avg_product_types', 'total_revenue', 'avg_customer_value'
            ]].rename(columns={
                'diversity': '다양성',
                'customer_count': '고객수',
                'percentage': '비율(%)',
                'avg_product_groups': '평균상품군수',
                'avg_product_types': '평균상품타입수',
                'total_revenue': '총매출',
                'avg_customer_value': '고객당평균값'
            }),
            use_container_width=True,
            hide_index=True
        )

        # 인사이트
        st.markdown("### 💡 인사이트")
        low_pct = category_df[category_df['diversity'] == 'Low']['percentage'].values[0]
        high_value = category_df[category_df['diversity'] == 'High']['avg_customer_value'].values[0]
        low_value = category_df[category_df['diversity'] == 'Low']['avg_customer_value'].values[0]

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.warning(f"📉 낮은 다양성\n{low_pct:.1f}%\n\n대부분 한두 카테고리만 구매")
        with col_i2:
            st.success(f"⬆️ 가치 격차\n{high_value/low_value:.1f}배\n\n다양한 제품 구매 시 가치 UP")
        with col_i3:
            st.info(f"🎯 크로스셀 기회\n제품 추천으로\n카테고리 확대 가능")

    # ==================== Tab 4: 판매 채널 ====================
    with tab4:
        st.subheader("판매 채널 비교")

        channel_df = funnel_data['channel'].copy()

        col1, col2 = st.columns(2)

        with col1:
            # 매출 비교
            fig = go.Figure(data=[go.Bar(
                x=channel_df['channel'],
                y=channel_df['total_revenue'],
                marker=dict(color=['#E50010', '#222222'])
            )])
            fig.update_layout(
                title="채널별 총 매출",
                yaxis_title="매출(원)",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 고객 비교
            fig = go.Figure(data=[go.Bar(
                x=channel_df['channel'],
                y=channel_df['unique_customers'],
                marker=dict(color=['#E50010', '#222222'])
            )])
            fig.update_layout(
                title="채널별 독립 고객 수",
                yaxis_title="고객수",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # 상세 테이블
        st.markdown("### 📋 채널별 상세 현황")
        st.dataframe(
            channel_df[[
                'channel', 'unique_customers', 'total_revenue',
                'avg_order_value', 'avg_transaction_value', 'transaction_count'
            ]].rename(columns={
                'channel': '채널',
                'unique_customers': '독립고객수',
                'total_revenue': '총매출',
                'avg_order_value': '고객당평균값',
                'avg_transaction_value': '거래당평균값',
                'transaction_count': '총거래수'
            }),
            use_container_width=True,
            hide_index=True
        )

        # 동적 인사이트
        st.markdown("### 💡 인사이트")
        online_row = channel_df[channel_df['channel'] == 'Online']
        offline_row = channel_df[channel_df['channel'] == 'Offline']
        total_rev = channel_df['total_revenue'].sum()
        total_cnt = channel_df['transaction_count'].sum()
        total_cust = channel_df['unique_customers'].sum()

        online_rev_pct = (online_row['total_revenue'].values[0] / total_rev * 100) if len(online_row) > 0 else 0
        offline_rev_pct = 100 - online_rev_pct
        online_cnt_pct = (online_row['transaction_count'].values[0] / total_cnt * 100) if len(online_row) > 0 else 0
        offline_cnt_pct = 100 - online_cnt_pct

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            if len(online_row) > 0:
                on_cust = int(online_row['unique_customers'].values[0])
                on_cnt = int(online_row['transaction_count'].values[0])
                on_avg = online_row['avg_transaction_value'].values[0]
                st.success(f"💻 **온라인 채널 (주도)**\n\n"
                           f"- 거래 비중: **{online_cnt_pct:.1f}%** ({on_cnt:,}건)\n"
                           f"- 매출 비중: **{online_rev_pct:.1f}%**\n"
                           f"- 고객 수: **{on_cust:,}명**\n"
                           f"- 건당 평균: **₩{on_avg:.4f}**\n\n"
                           f"거래 건수 기준 온라인이 압도적 우위")
        with col_i2:
            if len(offline_row) > 0:
                off_cust = int(offline_row['unique_customers'].values[0])
                off_cnt = int(offline_row['transaction_count'].values[0])
                off_avg = offline_row['avg_transaction_value'].values[0]
                st.warning(f"🏬 **오프라인 채널**\n\n"
                           f"- 거래 비중: **{offline_cnt_pct:.1f}%** ({off_cnt:,}건)\n"
                           f"- 매출 비중: **{offline_rev_pct:.1f}%**\n"
                           f"- 고객 수: **{off_cust:,}명**\n"
                           f"- 건당 평균: **₩{off_avg:.4f}**\n\n"
                           f"매출 비중은 낮지만 고단가 구매 채널")
        with col_i3:
            if len(online_row) > 0 and len(offline_row) > 0:
                off_avg2 = offline_row['avg_transaction_value'].values[0]
                on_avg2 = online_row['avg_transaction_value'].values[0]
                online_cust_pct = int(online_row['unique_customers'].values[0] / total_cust * 100)
                if on_avg2 > 0 and off_avg2 > 0:
                    if on_avg2 >= off_avg2:
                        leader = "온라인"
                        ratio = on_avg2 / off_avg2
                        action = "온라인 채널 객단가 강점 활용\n오프라인 고객 온라인 유입 유도"
                    else:
                        leader = "오프라인"
                        ratio = off_avg2 / on_avg2
                        action = "오프라인 고단가 강점 활용\n옴니채널로 온라인 객단가 향상"
                    st.info(f"🔀 **채널 전략 인사이트**\n\n"
                            f"- **{leader}** 건당 가치: **{ratio:.2f}배** 높음\n"
                            f"- 온라인 고객 비중: **{online_cust_pct}%**\n\n"
                            f"{action}")
