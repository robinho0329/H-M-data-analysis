"""
코호트 분석 페이지
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import numpy as np

# 프로젝트 경로 설정
project_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_dir))

from config import PROCESSED_DIR


@st.cache_data
def load_cohort_data():
    """코호트 분석 결과 로드"""
    results_dir = PROCESSED_DIR / "analysis"

    data = {
        'retention': pd.read_csv(results_dir / "cohort_retention.csv", index_col=0),
        'revenue': pd.read_csv(results_dir / "cohort_revenue.csv", index_col=0),
        'avg_spending': pd.read_csv(results_dir / "cohort_avg_spending.csv", index_col=0),
        'age_group': pd.read_csv(results_dir / "cohort_age_group.csv"),
        'club_status': pd.read_csv(results_dir / "cohort_club_status.csv"),
        'newsletter': pd.read_csv(results_dir / "cohort_newsletter.csv")
    }

    return data


def create_heatmap(data, title, colorscale='RdYlGn'):
    """히트맵 생성"""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=colorscale,
        text=np.round(data.values, 1),
        texttemplate='%{text}',
        textfont={"size": 9},
        hoverongaps=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="코호트 나이(개월)",
        yaxis_title="코호트 (첫 구매월)",
        height=600,
        xaxis={'side': 'bottom'}
    )

    return fig


def show():
    st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:11px; color:#767676; margin-bottom:12px; letter-spacing:0.3px;'>
        Home / Analytics / <span style='color:#000000; font-weight:600;'>COHORT ANALYSIS</span>
    </div>
    <h1 style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;
               font-size:1.8rem; font-weight:700; color:#000000;
               margin:0 0 4px 0; letter-spacing:0.5px; text-transform:uppercase;'>
        COHORT ANALYSIS
    </h1>
    <div style='height:2px; background:#000000; width:40px; margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)

    # 데이터 로드
    cohort_data = load_cohort_data()

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 리텐션율",
        "💰 매출",
        "💸 평균지출",
        "👥 나이대",
        "🏢 클럽상태",
        "📧 뉴스레터"
    ])

    # ==================== Tab 1: 리텐션 ====================
    with tab1:
        st.subheader("고객 리텐션율 분석")
        st.markdown("같은 월에 첫 구매한 고객들의 재구매율 추이")

        retention_df = cohort_data['retention'].iloc[:, :8]  # 처음 8개월만 표시

        col1, col2 = st.columns([3, 1])

        with col1:
            # 히트맵
            fig = create_heatmap(retention_df, "월별 코호트 리텐션율 (%)", 'RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 해석")
            st.markdown("""
            - **100%**: 초기 구매 고객
            - **높을수록**: 재구매율 높음
            - **낮을수록**: 고객 이탈 높음

            ### 🔍 관찰
            """)

            avg_retention = retention_df.iloc[:, 1].mean()
            st.metric("평균 1개월 리텐션", f"{avg_retention:.1f}%", delta=f"{avg_retention - 50:.1f}%",
                     delta_color="inverse")

        # 상세 통계
        st.markdown("### 📋 코호트별 리텐션 통계")

        retention_stats = pd.DataFrame({
            '코호트': retention_df.index,
            '초기고객': retention_df.iloc[:, 0].values.astype(int),
            '1개월': retention_df.iloc[:, 1].values,
            '2개월': retention_df.iloc[:, 2].values if len(retention_df.columns) > 2 else np.nan,
            '3개월': retention_df.iloc[:, 3].values if len(retention_df.columns) > 3 else np.nan
        })

        st.dataframe(retention_stats, use_container_width=True, hide_index=True)

        # 인사이트
        st.markdown("### 💡 인사이트")
        col_i1, col_i2, col_i3 = st.columns(3)

        with col_i1:
            st.error(f"⚠️ 극도로 낮은 리텐션\n평균: {avg_retention:.1f}%\n\n거의 모든 고객이 재구매 안 함")

        with col_i2:
            st.warning(f"🔄 조기 이탈 심각\n초기 리텐션 \n개선 필수")

        with col_i3:
            st.info(f"🎯 액션 필요\n- 구매 후 팔로우업\n- 맞춤형 추천\n- 로열티 프로그램")

    # ==================== Tab 2: 매출 ====================
    with tab2:
        st.subheader("코호트별 매출 분석")
        st.markdown("각 코호트의 월별 매출 추이")

        revenue_df = cohort_data['revenue'].iloc[:, :8]

        col1, col2 = st.columns([3, 1])

        with col1:
            fig = create_heatmap(revenue_df, "월별 코호트 매출(원)", 'Reds')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 통계")
            total_revenue = revenue_df.iloc[:, 0].sum()
            avg_cohort_revenue = revenue_df.iloc[:, 0].mean()

            st.metric("초기 월 총매출", f"₩{total_revenue:,.0f}")
            st.metric("코호트당 평균 초기매출", f"₩{avg_cohort_revenue:,.0f}")

        # 상세 분석
        st.markdown("### 📋 코호트별 매출")
        revenue_stats = pd.DataFrame({
            '코호트': revenue_df.index,
            '초기월': revenue_df.iloc[:, 0].values.astype(int),
            '1개월후': revenue_df.iloc[:, 1].values.astype(int) if len(revenue_df.columns) > 1 else 0,
            '2개월후': revenue_df.iloc[:, 2].values.astype(int) if len(revenue_df.columns) > 2 else 0
        })

        st.dataframe(revenue_stats, use_container_width=True, hide_index=True)

        # 인사이트
        st.markdown("### 💡 인사이트")
        col_i1, col_i2, col_i3 = st.columns(3)

        with col_i1:
            decline_rate = ((revenue_df.iloc[:, 0].mean() - revenue_df.iloc[:, 1].mean()) /
                           revenue_df.iloc[:, 0].mean() * 100) if revenue_df.iloc[:, 0].mean() > 0 else 0
            st.warning(f"📉 **월 매출 급락**\n\n"
                       f"- 초기 대비 **{decline_rate:.0f}%** 감소\n"
                       f"- 총 초기월 매출: **₩{revenue_df.iloc[:, 0].sum():,.2f}**\n\n"
                       f"재구매율 개선 없이는 매출 유지 불가")

        with col_i2:
            best_cohort = revenue_df.iloc[:, 0].idxmax()
            best_rev = revenue_df.iloc[:, 0].max()
            worst_cohort = revenue_df.iloc[:, 0].idxmin()
            worst_rev = revenue_df.iloc[:, 0].min()
            st.success(f"🏆 **최고 초기 매출 코호트**\n\n"
                       f"- 코호트: **{best_cohort}**\n"
                       f"- 초기월 매출: **₩{best_rev:,.2f}**\n\n"
                       f"최저 코호트 ({worst_cohort}): ₩{worst_rev:,.2f}")

        with col_i3:
            total_initial = revenue_df.iloc[:, 0].sum()
            total_month1 = revenue_df.iloc[:, 1].sum() if len(revenue_df.columns) > 1 else 0
            retention_rev_rate = total_month1 / total_initial * 100 if total_initial > 0 else 0
            st.info(f"🎯 **매출 리텐션율**\n\n"
                    f"- 1개월 매출 유지율: **{retention_rev_rate:.1f}%**\n"
                    f"- 유지 목표: **20% 이상**\n\n"
                    f"초기 경험 최적화 → 재방문율 향상 필요")

    # ==================== Tab 3: 평균 지출액 ====================
    with tab3:
        st.subheader("코호트별 고객당 평균 지출액")

        avg_df = cohort_data['avg_spending'].iloc[:, :8]

        col1, col2 = st.columns([3, 1])

        with col1:
            fig = create_heatmap(avg_df, "월별 코호트 고객당 평균 지출액(원)", 'Reds')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 통계")
            avg_initial = avg_df.iloc[:, 0].mean()
            avg_month1 = avg_df.iloc[:, 1].mean()

            st.metric("초기월 평균값", f"₩{avg_initial:,.0f}")
            st.metric("1개월 평균값", f"₩{avg_month1:,.0f}")

        st.markdown("### 💡 인사이트")
        col_i1, col_i2, col_i3 = st.columns(3)

        with col_i1:
            best_spend_cohort = avg_df.iloc[:, 0].idxmax()
            best_spend_val = avg_df.iloc[:, 0].max()
            overall_avg = avg_df.iloc[:, 0].mean()
            premium_ratio = best_spend_val / overall_avg if overall_avg > 0 else 1
            st.success(f"💎 **최고 지출 코호트**\n\n"
                       f"- 코호트: **{best_spend_cohort}**\n"
                       f"- 평균 지출: **₩{best_spend_val:.4f}**\n"
                       f"- 전체 평균 대비: **{premium_ratio:.2f}배**\n\n"
                       f"해당 시기 마케팅 전략 분석 권장")

        with col_i2:
            if len(avg_df.columns) > 1:
                m0 = avg_df.iloc[:, 0].mean()
                m1 = avg_df.iloc[:, 1].mean()
                change = (m1 - m0) / m0 * 100 if m0 > 0 else 0
                direction = "증가" if change > 0 else "감소"
                color_fn = st.success if change > 0 else st.warning
                color_fn(f"📊 **1개월 후 지출 변화**\n\n"
                         f"- 초기: **₩{m0:.4f}**\n"
                         f"- 1개월 후: **₩{m1:.4f}**\n"
                         f"- 변화율: **{abs(change):.1f}% {direction}**\n\n"
                         f"재구매 고객의 지출 패턴 모니터링 필요")
            else:
                st.info("1개월 이후 데이터 없음")

        with col_i3:
            st.info(f"🎯 **LTV 증대 전략**\n\n"
                    f"- 전체 평균 지출: **₩{avg_df.iloc[:, 0].mean():.4f}**\n"
                    f"- 재구매 유도 시 LTV 개선\n\n"
                    f"프리미엄 제품 추천 + 로열티 프로그램 연계 권장")

    # ==================== Tab 4: 나이대별 ====================
    with tab4:
        st.subheader("나이대별 월별 구매 행동")

        age_df = cohort_data['age_group'].copy()
        age_df['month'] = age_df['month'].astype(str)

        col1, col2 = st.columns(2)

        with col1:
            # 나이대별 고객 수
            age_summary = age_df.groupby('age_group')['unique_customers'].sum().sort_values(ascending=False)

            fig = go.Figure(data=[go.Bar(
                x=age_summary.index,
                y=age_summary.values,
                marker=dict(color='#E50010')
            )])
            fig.update_layout(
                title="나이대별 총 구매고객",
                xaxis_title="나이대",
                yaxis_title="고객수",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 나이대별 평균 지출
            age_spending = age_df.groupby('age_group')['avg_spending_per_customer'].mean().sort_values(ascending=False)

            fig = go.Figure(data=[go.Bar(
                x=age_spending.index,
                y=age_spending.values,
                marker=dict(color='#CC0000')
            )])
            fig.update_layout(
                title="나이대별 고객당 평균 지출",
                xaxis_title="나이대",
                yaxis_title="평균 지출액(원)",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # 상세 분석
        st.markdown("### 📋 나이대별 상세 현황")
        age_detailed = age_df.groupby('age_group').agg({
            'unique_customers': 'sum',
            'total_revenue': 'sum',
            'transaction_count': 'sum',
            'avg_spending_per_customer': 'mean'
        }).round(2)

        st.dataframe(
            age_detailed.rename(columns={
                'unique_customers': '고객수',
                'total_revenue': '총매출',
                'transaction_count': '거래수',
                'avg_spending_per_customer': '평균지출'
            }),
            use_container_width=True
        )

        # 동적 인사이트
        st.markdown("### 💡 인사이트")
        if len(age_detailed) > 0:
            top_age = age_detailed['total_revenue'].idxmax()
            top_spending_age = age_detailed['avg_spending_per_customer'].idxmax()
            top_count_age = age_detailed['unique_customers'].idxmax()
            top_rev = age_detailed.loc[top_age, 'total_revenue']
            total_rev = age_detailed['total_revenue'].sum()
            top_pct = top_rev / total_rev * 100 if total_rev > 0 else 0

            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.success(f"🏆 **핵심 매출 연령대: {top_age}**\n\n"
                           f"- 총 매출: **₩{top_rev:,.0f}** ({top_pct:.1f}%)\n"
                           f"- 고객 수: **{int(age_detailed.loc[top_age, 'unique_customers']):,}명**\n\n"
                           f"이 연령대에 마케팅 예산 집중 권장")
            with col_i2:
                top_sp_val = age_detailed.loc[top_spending_age, 'avg_spending_per_customer']
                st.info(f"💎 **건당 지출 최고: {top_spending_age}**\n\n"
                        f"- 평균 지출: **₩{top_sp_val:.4f}**\n"
                        f"- 고객 수: **{int(age_detailed.loc[top_spending_age, 'unique_customers']):,}명**\n\n"
                        f"프리미엄 제품 라인 타겟에 적합")
            with col_i3:
                top_cnt_val = int(age_detailed.loc[top_count_age, 'unique_customers'])
                cnt_pct = top_cnt_val / age_detailed['unique_customers'].sum() * 100
                st.warning(f"👥 **최대 고객층: {top_count_age}**\n\n"
                           f"- 고객 수: **{top_cnt_val:,}명** ({cnt_pct:.1f}%)\n\n"
                           f"볼륨 기반 캠페인 효과 극대화 가능")

    # ==================== Tab 5: 클럽 상태별 ====================
    with tab5:
        st.subheader("클럽 상태별 월별 구매 행동")

        club_df = cohort_data['club_status'].copy()

        col1, col2 = st.columns(2)

        with col1:
            # 클럽 상태별 고객 수
            club_summary = club_df.groupby('club_status')['unique_customers'].sum().sort_values(ascending=False)

            fig = go.Figure(data=[go.Bar(
                x=club_summary.index,
                y=club_summary.values,
                marker=dict(color=['#E50010', '#222222', '#CC0000'])
            )])
            fig.update_layout(
                title="클럽 상태별 총 구매고객",
                xaxis_title="클럽 상태",
                yaxis_title="고객수",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 클럽 상태별 평균 지출
            club_spending = club_df.groupby('club_status')['avg_spending_per_customer'].mean().sort_values(ascending=False)

            fig = go.Figure(data=[go.Bar(
                x=club_spending.index,
                y=club_spending.values,
                marker=dict(color=['#E50010', '#222222', '#CC0000'])
            )])
            fig.update_layout(
                title="클럽 상태별 고객당 평균 지출",
                xaxis_title="클럽 상태",
                yaxis_title="평균 지출액(원)",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # 상세 분석
        st.markdown("### 📋 클럽 상태별 상세 현황")
        club_detailed = club_df.groupby('club_status').agg({
            'unique_customers': 'sum',
            'total_revenue': 'sum',
            'transaction_count': 'sum',
            'avg_spending_per_customer': 'mean'
        }).round(2)

        st.dataframe(
            club_detailed.rename(columns={
                'unique_customers': '고객수',
                'total_revenue': '총매출',
                'transaction_count': '거래수',
                'avg_spending_per_customer': '평균지출'
            }),
            use_container_width=True
        )

        # 동적 인사이트
        st.markdown("### 💡 인사이트")
        col_i1, col_i2, col_i3 = st.columns(3)

        with col_i1:
            if 'ACTIVE' in club_detailed.index:
                active_rev = club_detailed.loc['ACTIVE', 'total_revenue']
                active_cnt = int(club_detailed.loc['ACTIVE', 'unique_customers'])
                active_pct = active_cnt / club_detailed['unique_customers'].sum() * 100
                active_avg = club_detailed.loc['ACTIVE', 'avg_spending_per_customer']
                st.success(f"🏆 **ACTIVE 클럽 회원**\n\n"
                           f"- 고객 수: **{active_cnt:,}명** ({active_pct:.1f}%)\n"
                           f"- 총 매출: **₩{active_rev:,.0f}**\n"
                           f"- 평균 지출: **₩{active_avg:.4f}**\n\n"
                           f"핵심 충성 고객, 혜택 강화 필요")
        with col_i2:
            if 'PRE-CREATE' in club_detailed.index:
                pre_cnt = int(club_detailed.loc['PRE-CREATE', 'unique_customers'])
                pre_avg = club_detailed.loc['PRE-CREATE', 'avg_spending_per_customer']
                pre_pct = pre_cnt / club_detailed['unique_customers'].sum() * 100
                st.warning(f"⏳ **PRE-CREATE 고객**\n\n"
                           f"- 고객 수: **{pre_cnt:,}명** ({pre_pct:.1f}%)\n"
                           f"- 평균 지출: **₩{pre_avg:.4f}**\n\n"
                           f"정회원 전환 시 매출 증대 잠재력 큼")
        with col_i3:
            if 'LEFT CLUB' in club_detailed.index:
                left_cnt = int(club_detailed.loc['LEFT CLUB', 'unique_customers'])
                left_avg = club_detailed.loc['LEFT CLUB', 'avg_spending_per_customer']
                if 'ACTIVE' in club_detailed.index:
                    active_avg2 = club_detailed.loc['ACTIVE', 'avg_spending_per_customer']
                    gap = active_avg2 - left_avg
                    st.error(f"🚪 **LEFT CLUB 고객**\n\n"
                             f"- 고객 수: **{left_cnt:,}명**\n"
                             f"- ACTIVE 대비 지출 차이: **₩{gap:.4f}**\n\n"
                             f"재가입 유도로 이탈 매출 복구 가능")
                else:
                    st.error(f"🚪 **LEFT CLUB 고객**\n\n"
                             f"- 고객 수: **{left_cnt:,}명**\n\n"
                             f"재가입 캠페인 필요")

    # ==================== Tab 6: 뉴스레터 ====================
    with tab6:
        st.subheader("뉴스레터 구독 상태별 구매 행동")

        newsletter_df = cohort_data['newsletter'].copy()

        col1, col2 = st.columns(2)

        with col1:
            # 뉴스레터 상태별 고객 수
            news_summary = newsletter_df.groupby('newsletter_freq')['unique_customers'].sum().sort_values(ascending=False)

            fig = go.Figure(data=[go.Bar(
                x=news_summary.index,
                y=news_summary.values,
                marker=dict(color=['#E50010', '#222222', '#CC0000'])
            )])
            fig.update_layout(
                title="뉴스레터 상태별 총 구매고객",
                xaxis_title="뉴스레터 구독",
                yaxis_title="고객수",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 뉴스레터 상태별 평균 지출
            news_spending = newsletter_df.groupby('newsletter_freq')['avg_spending_per_customer'].mean().sort_values(ascending=False)

            fig = go.Figure(data=[go.Bar(
                x=news_spending.index,
                y=news_spending.values,
                marker=dict(color=['#E50010', '#222222', '#CC0000'])
            )])
            fig.update_layout(
                title="뉴스레터 상태별 고객당 평균 지출",
                xaxis_title="뉴스레터 구독",
                yaxis_title="평균 지출액(원)",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # 상세 분석
        st.markdown("### 📋 뉴스레터 상태별 상세 현황")
        news_detailed = newsletter_df.groupby('newsletter_freq').agg({
            'unique_customers': 'sum',
            'total_revenue': 'sum',
            'transaction_count': 'sum',
            'avg_spending_per_customer': 'mean'
        }).round(2)

        st.dataframe(
            news_detailed.rename(columns={
                'unique_customers': '고객수',
                'total_revenue': '총매출',
                'transaction_count': '거래수',
                'avg_spending_per_customer': '평균지출'
            }),
            use_container_width=True
        )

        # 동적 인사이트
        st.markdown("### 💡 인사이트")
        col_i1, col_i2, col_i3 = st.columns(3)

        with col_i1:
            if 'NONE' in news_detailed.index:
                none_cnt = int(news_detailed.loc['NONE', 'unique_customers'])
                none_pct = none_cnt / news_detailed['unique_customers'].sum() * 100
                none_avg = news_detailed.loc['NONE', 'avg_spending_per_customer']
                none_rev = news_detailed.loc['NONE', 'total_revenue']
                st.warning(f"📭 **미구독 고객 (NONE)**\n\n"
                           f"- 고객 수: **{none_cnt:,}명** ({none_pct:.1f}%)\n"
                           f"- 총 매출: **₩{none_rev:,.0f}**\n"
                           f"- 평균 지출: **₩{none_avg:.4f}**\n\n"
                           f"가장 많은 고객층, 구독 전환 시 큰 효과")

        with col_i2:
            if 'Regularly' in news_detailed.index:
                reg_cnt = int(news_detailed.loc['Regularly', 'unique_customers'])
                reg_avg = news_detailed.loc['Regularly', 'avg_spending_per_customer']
                reg_pct = reg_cnt / news_detailed['unique_customers'].sum() * 100
                reg_rev = news_detailed.loc['Regularly', 'total_revenue']
                st.success(f"📬 **정기 구독자 (Regularly)**\n\n"
                           f"- 고객 수: **{reg_cnt:,}명** ({reg_pct:.1f}%)\n"
                           f"- 총 매출: **₩{reg_rev:,.0f}**\n"
                           f"- 평균 지출: **₩{reg_avg:.4f}**\n\n"
                           f"가장 높은 참여도, 우선 관리 대상")

        with col_i3:
            none_val = (news_detailed.loc['NONE', 'avg_spending_per_customer']
                       if 'NONE' in news_detailed.index else 0)
            reg_val = (news_detailed.loc['Regularly', 'avg_spending_per_customer']
                      if 'Regularly' in news_detailed.index else 0)
            mon_val = (news_detailed.loc['Monthly', 'avg_spending_per_customer']
                      if 'Monthly' in news_detailed.index else 0)

            if none_val > 0 and reg_val > 0:
                ratio = reg_val / none_val
                mon_ratio = mon_val / none_val if mon_val > 0 else 0
                st.info(f"📊 **구독 효과 비교**\n\n"
                        f"- Regularly vs NONE: **{ratio:.2f}배**\n"
                        f"- Monthly vs NONE: **{mon_ratio:.2f}배**\n\n"
                        f"구독 유도만으로 고객 가치 {ratio:.1f}배 증대 가능")
