"""
예측 현황 페이지
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show():
    """페이지 표시"""
    st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:11px; color:#767676; margin-bottom:12px; letter-spacing:0.3px;'>
        Home / Analytics / <span style='color:#000000; font-weight:600;'>FORECAST OVERVIEW</span>
    </div>
    <h1 style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;
               font-size:1.8rem; font-weight:700; color:#000000;
               margin:0 0 4px 0; letter-spacing:0.5px; text-transform:uppercase;'>
        FORECAST OVERVIEW
    </h1>
    <div style='height:2px; background:#000000; width:40px; margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)

    # 데이터 로드 (실제 구현에서는 모델 예측값 사용)
    try:
        # 대체 데이터 (실제 구현에서는 db/파일에서 로드)
        dates = pd.date_range(start="2019-12-01", periods=31, freq='D')
        actuals = np.random.uniform(20, 40, 31) + np.sin(np.arange(31)/7) * 5
        predictions = actuals + np.random.normal(0, 1, 31)

        df = pd.DataFrame({
            'date': dates,
            'actual': actuals,
            'prediction': predictions,
        })
        df['error'] = np.abs(df['actual'] - df['prediction'])
        df['error_pct'] = (df['error'] / df['actual'] * 100).round(2)

    except Exception as e:
        st.error(f"데이터 로드 실패: {str(e)}")
        return

    # KPI 표시
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📅 최근 예측",
            f"₩{predictions[-1]:.2f}",
            f"실제: ₩{actuals[-1]:.2f}"
        )

    with col2:
        accuracy = 100 - df['error_pct'].mean()
        st.metric(
            "🎯 정확도",
            f"{accuracy:.1f}%",
            delta=f"-{df['error_pct'].mean():.1f}%"
        )

    with col3:
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        st.metric(
            "📊 RMSE",
            f"{rmse:.4f}",
            help="Root Mean Squared Error"
        )

    with col4:
        mae = np.mean(df['error'])
        st.metric(
            "📏 MAE",
            f"{mae:.4f}",
            help="Mean Absolute Error"
        )

    st.markdown("---")

    # 예측 vs 실제 라인 차트
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("예측 vs 실제 매출 추이")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['actual'],
            name='실제 매출',
            mode='lines+markers',
            line=dict(color='#222222', width=2),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['prediction'],
            name='예측 매출',
            mode='lines+markers',
            line=dict(color='#E50010', width=2, dash='dash'),
            marker=dict(size=6)
        ))

        # 신뢰 구간 (±1σ)
        std_error = df['error'].std()
        upper = df['prediction'] + std_error
        lower = df['prediction'] - std_error

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=upper,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=lower,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='신뢰 구간 (±1σ)',
            fillcolor='rgba(229,0,16,0.10)'
        ))

        fig.update_layout(
            title="일일 매출 예측",
            xaxis_title="날짜",
            yaxis_title="매출액 (₩)",
            hovermode='x unified',
            height=400,
            template='plotly_white',
            font=dict(family="Inter, Helvetica Neue, Arial, sans-serif")
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("통계")
        st.write(f"**기간:** {df['date'].min().date()} ~ {df['date'].max().date()}")
        st.write(f"**총 일수:** {len(df)}일")
        st.write(f"**평균 실제 매출:** ₩{df['actual'].mean():.2f}")
        st.write(f"**평균 예측 매출:** ₩{df['prediction'].mean():.2f}")
        st.write(f"**평균 오차:** ₩{df['error'].mean():.2f}")

    st.markdown("---")

    # 요일별 패턴
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("요일별 패턴")

        df['day_of_week'] = df['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_kr = ['월', '화', '수', '목', '금', '토', '일']

        daily_stats = df.groupby('day_of_week')[['actual', 'prediction']].mean()
        daily_stats = daily_stats.reindex([day for day in day_order if day in daily_stats.index])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[day_names_kr[day_order.index(day)] for day in daily_stats.index],
            y=daily_stats['actual'],
            name='실제 평균',
            marker_color='#222222'
        ))

        fig.add_trace(go.Bar(
            x=[day_names_kr[day_order.index(day)] for day in daily_stats.index],
            y=daily_stats['prediction'],
            name='예측 평균',
            marker_color='#E50010'
        ))

        fig.update_layout(
            title="요일별 평균 매출",
            xaxis_title="요일",
            yaxis_title="매출액 (₩)",
            barmode='group',
            height=350,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("예측 오차 분포")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df['error'],
            nbinsx=15,
            name='오차',
            marker_color='#FF6B6B'
        ))

        fig.update_layout(
            title="예측 오차 분포",
            xaxis_title="오차 (₩)",
            yaxis_title="빈도",
            height=350,
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

    # 상세 테이블
    st.markdown("---")
    st.subheader("📋 상세 데이터")

    display_df = df[['date', 'actual', 'prediction', 'error', 'error_pct']].copy()
    display_df.columns = ['날짜', '실제 매출', '예측 매출', '오차 (₩)', '오차율 (%)']
    display_df['날짜'] = display_df['날짜'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # 인사이트
    st.markdown("---")
    st.subheader("💡 모델 성능 인사이트")

    best_day = df.loc[df['error_pct'].idxmin()]
    worst_day = df.loc[df['error_pct'].idxmax()]
    best_dow = df.groupby('day_of_week')['error_pct'].mean().idxmin()
    worst_dow = df.groupby('day_of_week')['error_pct'].mean().idxmax()
    trend = "상승" if df['actual'].iloc[-7:].mean() > df['actual'].iloc[:7].mean() else "하락"
    trend_pct = abs((df['actual'].iloc[-7:].mean() - df['actual'].iloc[:7].mean()) / df['actual'].iloc[:7].mean() * 100)

    dow_kr = {'Monday': '월', 'Tuesday': '화', 'Wednesday': '수',
              'Thursday': '목', 'Friday': '금', 'Saturday': '토', 'Sunday': '일'}

    col_i1, col_i2, col_i3 = st.columns(3)

    with col_i1:
        accuracy = 100 - df['error_pct'].mean()
        if accuracy >= 95:
            fn = st.success
            grade = "우수"
        elif accuracy >= 90:
            fn = st.info
            grade = "양호"
        else:
            fn = st.warning
            grade = "개선 필요"
        fn(f"🎯 **모델 정확도: {grade}**\n\n"
           f"- 평균 정확도: **{accuracy:.1f}%**\n"
           f"- 최저 오차일: **{best_day['date'].strftime('%m/%d')}** ({best_day['error_pct']:.1f}%)\n"
           f"- 최고 오차일: **{worst_day['date'].strftime('%m/%d')}** ({worst_day['error_pct']:.1f}%)\n\n"
           f"LSTM 모델이 일별 매출 패턴을 잘 학습함")

    with col_i2:
        st.info(f"📅 **요일별 예측 품질**\n\n"
                f"- 최고 정확도 요일: **{dow_kr.get(best_dow, best_dow)}요일**\n"
                f"- 최저 정확도 요일: **{dow_kr.get(worst_dow, worst_dow)}요일**\n\n"
                f"{dow_kr.get(worst_dow, worst_dow)}요일 데이터 보강 또는\n추가 피처 엔지니어링 권장")

    with col_i3:
        if trend == "상승":
            fn2 = st.success
        else:
            fn2 = st.warning
        fn2(f"📈 **매출 트렌드: {trend}**\n\n"
            f"- 기간 내 변화: **{trend_pct:.1f}%** {trend}\n"
            f"- 전반부 평균: **₩{df['actual'].iloc[:7].mean():.2f}**\n"
            f"- 후반부 평균: **₩{df['actual'].iloc[-7:].mean():.2f}**\n\n"
            f"트렌드 반영한 재학습으로 정확도 향상 가능")
