"""
비즈니스 요약 대시보드 - 전체 분석 결과 한눈에 보기
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_dir))

from config import PROCESSED_DIR


@st.cache_data
def load_summary_data():
    """요약 대시보드용 데이터 로드"""
    from app.utils.cache_manager import load_raw_data
    results_dir = PROCESSED_DIR / "analysis"

    merged = load_raw_data()
    try:
        funnel = {
            'lifecycle': pd.read_csv(results_dir / "funnel_lifecycle.csv"),
            'frequency': pd.read_csv(results_dir / "funnel_frequency.csv"),
            'channel': pd.read_csv(results_dir / "funnel_channel.csv"),
        }
        cohort = {
            'retention': pd.read_csv(results_dir / "cohort_retention.csv", index_col=0),
            'club_status': pd.read_csv(results_dir / "cohort_club_status.csv"),
            'newsletter': pd.read_csv(results_dir / "cohort_newsletter.csv"),
        }
    except FileNotFoundError:
        st.warning("⚠️ 비즈니스 요약 데이터를 찾을 수 없습니다. 분석 파이프라인을 먼저 실행해주세요.")
        return None, None, None
    return merged, funnel, cohort


def show():
    st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:11px; color:#767676; margin-bottom:12px; letter-spacing:0.3px;'>
        홈 / 분석 / <span style='color:#000000; font-weight:600;'>비즈니스 요약</span>
    </div>
    <h1 style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;
               font-size:1.8rem; font-weight:700; color:#000000;
               margin:0 0 4px 0; letter-spacing:0.5px; text-transform:uppercase;'>
        비즈니스 요약
    </h1>
    <div style='height:2px; background:#000000; width:40px; margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)

    merged, funnel, cohort = load_summary_data()
    if merged is None or funnel is None or cohort is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    # ==================== 섹션 1: 핵심 KPI ====================
    st.markdown("## 🎯 핵심 KPI")

    total_customers = merged['customer_id'].nunique()
    total_transactions = len(merged)
    total_revenue = merged['price'].sum()
    avg_order_value = merged['price'].mean()
    online_rev = funnel['channel'][funnel['channel']['channel'] == 'Online']['total_revenue'].values[0]
    offline_rev = funnel['channel'][funnel['channel']['channel'] == 'Offline']['total_revenue'].values[0]
    online_pct = online_rev / (online_rev + offline_rev) * 100

    lc = funnel['lifecycle']
    active_pct = lc[lc['lifecycle'] == 'Active']['percentage'].values[0] if 'Active' in lc['lifecycle'].values else 0
    new_pct = lc[lc['lifecycle'] == 'New']['percentage'].values[0] if 'New' in lc['lifecycle'].values else 0

    freq = funnel['frequency']
    one_time_pct = freq[freq['frequency'] == 'One-time']['percentage'].values[0] if 'One-time' in freq['frequency'].values else 0

    avg_retention = cohort['retention'].iloc[:, 1].mean() if cohort['retention'].shape[1] > 1 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 전체 고객 수", f"{total_customers:,}명")
    with col2:
        st.metric("🛒 총 거래 건수", f"{total_transactions:,}건")
    with col3:
        st.metric("💰 총 매출", f"₩{total_revenue:,.2f}")
    with col4:
        st.metric("📦 건당 평균 매출", f"₩{avg_order_value:.4f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("🟢 활성 고객 비율", f"{active_pct:.1f}%",
                  delta=f"-{100 - active_pct:.1f}% 비활성", delta_color="inverse")
    with col6:
        st.metric("🆕 신규 고객 비율", f"{new_pct:.1f}%")
    with col7:
        st.metric("💻 온라인 매출 비중", f"{online_pct:.1f}%")
    with col8:
        st.metric("🔄 평균 리텐션율", f"{avg_retention:.1f}%",
                  delta=f"목표 20% 대비 {avg_retention - 20:.1f}%", delta_color="normal")

    st.markdown("---")

    # ==================== 섹션 2: 매출 트렌드 ====================
    st.markdown("## 📈 월별 매출 트렌드")

    col1, col2 = st.columns([3, 1])

    with col1:
        monthly_rev = merged.groupby(merged['t_dat'].dt.to_period('M').astype(str))['price'].sum().reset_index()
        monthly_rev.columns = ['month', 'revenue']
        monthly_cnt = merged.groupby(merged['t_dat'].dt.to_period('M').astype(str))['price'].count().reset_index()
        monthly_cnt.columns = ['month', 'count']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_rev['month'],
            y=monthly_rev['revenue'],
            name='월별 매출',
            marker_color='#E50010',
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=monthly_cnt['month'],
            y=monthly_cnt['count'],
            name='거래 건수',
            mode='lines+markers',
            marker_color='#222222',
            yaxis='y2'
        ))
        fig.update_layout(
            yaxis=dict(title='매출(₩)', side='left'),
            yaxis2=dict(title='거래건수', side='right', overlaying='y'),
            height=350, template='plotly_white',
            legend=dict(orientation='h', y=1.1),
            hovermode='x unified',
            font=dict(family="Inter, Helvetica Neue, Arial, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        peak_month = monthly_rev.loc[monthly_rev['revenue'].idxmax(), 'month']
        low_month = monthly_rev.loc[monthly_rev['revenue'].idxmin(), 'month']
        growth = (monthly_rev['revenue'].iloc[-1] - monthly_rev['revenue'].iloc[0]) / monthly_rev['revenue'].iloc[0] * 100

        st.markdown("### 📊 트렌드 요약")
        st.metric("최고 매출 월", peak_month)
        st.metric("최저 매출 월", low_month)
        st.metric("연간 성장률", f"{growth:+.1f}%")
        peak_rev = monthly_rev['revenue'].max()
        low_rev = monthly_rev['revenue'].min()
        st.metric("최고/최저 배율", f"{peak_rev / low_rev:.2f}배")

    st.markdown("---")

    # ==================== 섹션 3: 고객 분석 요약 ====================
    st.markdown("## 👥 고객 분석 요약")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 라이프사이클")
        lc_sorted = funnel['lifecycle'].sort_values('customer_count', ascending=False)
        fig = go.Figure(data=[go.Pie(
            labels=lc_sorted['lifecycle'],
            values=lc_sorted['customer_count'],
            hole=0.4,
            marker=dict(colors=['#E50010', '#222222', '#FF6B6B', '#CC0000'])
        )])
        fig.update_layout(height=280, showlegend=True,
                          legend=dict(orientation='h', y=-0.15),
                          margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 구매 빈도")
        freq_sorted = funnel['frequency'].sort_values('customer_count', ascending=False)
        fig = go.Figure(data=[go.Bar(
            x=freq_sorted['frequency'],
            y=freq_sorted['customer_count'],
            marker=dict(color=['#E50010', '#222222', '#FF6B6B', '#CC0000'])
        )])
        fig.update_layout(height=280, template='plotly_white',
                          showlegend=False, margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### 클럽 상태")
        club_stats = cohort['club_status'].groupby('club_status').agg(
            customers=('unique_customers', 'sum')
        ).reset_index().sort_values('customers', ascending=False)
        fig = go.Figure(data=[go.Pie(
            labels=club_stats['club_status'],
            values=club_stats['customers'],
            hole=0.4,
            marker=dict(colors=['#E50010', '#222222', '#AAAAAA'])
        )])
        fig.update_layout(height=280, showlegend=True,
                          legend=dict(orientation='h', y=-0.15),
                          margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==================== 섹션 4: 상품 분석 요약 ====================
    st.markdown("## 🛍️ 상품 분석 요약")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top 10 색상별 매출")
        color_rev = merged.groupby('colour_group_name')['price'].sum().sort_values(ascending=False).head(10)
        fig = go.Figure(data=[go.Bar(
            x=color_rev.values,
            y=color_rev.index,
            orientation='h',
            marker=dict(color=color_rev.values, colorscale='Reds', showscale=False)
        )])
        fig.update_layout(height=320, template='plotly_white',
                          xaxis_title='매출(₩)', yaxis_autorange='reversed',
                          margin=dict(l=120))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top 10 제품 타입별 매출")
        product_rev = merged.groupby('product_type_name')['price'].sum().sort_values(ascending=False).head(10)
        fig = go.Figure(data=[go.Bar(
            x=product_rev.values,
            y=product_rev.index,
            orientation='h',
            marker=dict(color=product_rev.values, colorscale='Reds', showscale=False)
        )])
        fig.update_layout(height=320, template='plotly_white',
                          xaxis_title='매출(₩)', yaxis_autorange='reversed',
                          margin=dict(l=120))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==================== 섹션 5: 전략 권고사항 ====================
    st.markdown("## 🎯 전략 권고사항")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚨 즉시 조치 필요")

        urgent_items = []
        if one_time_pct > 40:
            urgent_items.append(f"**1회 구매 고객 {one_time_pct:.1f}%** → 재구매 유도 캠페인 시급")
        if active_pct < 15:
            urgent_items.append(f"**활성 고객 {active_pct:.1f}%** → 휴면 고객 재활성화 프로그램 필요")
        if avg_retention < 15:
            urgent_items.append(f"**리텐션 {avg_retention:.1f}%** → 구매 후 팔로우업 시스템 구축 필요")
        if online_pct < 50:
            urgent_items.append(f"**온라인 비중 {online_pct:.1f}%** → 디지털 채널 강화 전략 필요")

        if urgent_items:
            for item in urgent_items:
                st.error(f"🔴 {item}")
        else:
            st.success("현재 긴급 이슈 없음")

    with col2:
        st.markdown("### ✅ 성장 기회")

        # 뉴스레터 효과 계산
        news = cohort['newsletter']
        news_agg = news.groupby('newsletter_freq')['avg_spending_per_customer'].mean()
        none_val = news_agg.get('NONE', 0)
        reg_val = news_agg.get('Regularly', none_val)
        news_ratio = reg_val / none_val if none_val > 0 else 1

        # Black 색상 집중도
        black_rev = merged[merged['colour_group_name'] == 'Black']['price'].sum()
        black_share = black_rev / total_revenue * 100

        opportunities = [
            f"**뉴스레터 구독 전환**: 정기구독자 {news_ratio:.2f}배 높은 지출 → 미구독 고객 구독 유도",
            f"**Black 색상 집중**: 전체 매출 {black_share:.1f}% 차지 → 신규 컬러 라인 다양화",
            f"**ACTIVE 클럽 확장**: 가장 높은 충성도 → 클럽 멤버십 혜택 강화로 PRE-CREATE 전환",
            f"**온라인 채널 강화**: 거래 비중 {online_pct:.1f}% → 모바일 UX 개선 및 디지털 마케팅 투자",
        ]
        for item in opportunities:
            st.success(f"🟢 {item}")
