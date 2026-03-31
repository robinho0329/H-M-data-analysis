"""
예측 현황 페이지
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
import sys
import logging
from pathlib import Path

project_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "results"


@st.cache_data
def load_daily_sales():
    """일별 매출 데이터 로드"""
    path = DATA_DIR / "daily_sales.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data
def load_forecast_test():
    """예측 테스트 데이터 로드 (정규화된 값)"""
    path = DATA_DIR / "forecast_test.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data
def build_combined_df():
    """
    daily_sales와 forecast_test를 결합하여 combined DataFrame을 생성.
    - actual: daily_sales의 total_sales (실제 스케일)
    - prediction: 예측 기간(12/16~12/31)에 대해 실제 스케일로 변환
    - error, error_pct: 예측 기간에만 계산
    """
    daily = load_daily_sales()[["date", "total_sales", "day_of_week"]].copy()
    daily = daily.rename(columns={"total_sales": "actual"})

    forecast = load_forecast_test().copy()

    # 실제 스케일 변환: predicted_real = daily_total_sales * (prediction_norm / actual_norm)
    # Dec 25는 actual_norm == 0 이므로 0으로 처리
    forecast = forecast.merge(
        daily[["date", "actual"]].rename(columns={"actual": "daily_actual"}),
        on="date",
        how="left"
    )

    def compute_predicted_real(row):
        if row["actual"] == 0:
            return np.nan
        return row["daily_actual"] * (row["prediction"] / row["actual"])

    forecast["prediction_real"] = forecast.apply(compute_predicted_real, axis=1)

    # error는 실제 스케일 기준
    forecast["error_real"] = np.abs(forecast["daily_actual"] - forecast["prediction_real"])
    # error_pct는 이미 percentage 단위 (3.97 == 3.97%)
    forecast_slim = forecast[["date", "prediction_real", "error_real", "error_pct"]].rename(
        columns={
            "prediction_real": "prediction",
            "error_real": "error",
        }
    )

    # 전체 기간 merge
    combined = daily.merge(forecast_slim, on="date", how="left")

    return combined


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

    # 데이터 로드
    try:
        combined = build_combined_df()
    except Exception as e:
        st.error(f"데이터 로드 실패: {str(e)}")
        return

    # 날짜 범위 선택
    st.markdown("**📅 기간 선택**")
    date_range = st.date_input(
        "기간 선택",
        value=(date(2019, 12, 1), date(2019, 12, 31)),
        min_value=date(2019, 1, 1),
        max_value=date(2019, 12, 31),
        label_visibility="collapsed"
    )

    # 튜플이 아닐 경우(단일 날짜 선택 중간 상태) 처리
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("날짜 범위를 선택해주세요 (시작일과 종료일).")
        return

    start_date, end_date = date_range

    # 필터링
    mask = (combined["date"].dt.date >= start_date) & (combined["date"].dt.date <= end_date)
    df = combined[mask].copy()

    if df.empty:
        st.warning("선택 기간에 데이터가 없습니다.")
        return

    # 예측 데이터가 있는 행 (Dec 16-31 중 범위에 포함된 날)
    df_forecast = df[df["prediction"].notna() & (df["prediction"] > 0)].copy()
    has_forecast = len(df_forecast) > 0

    # ─────────────────────────────────────────────
    # KPI 메트릭
    # ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    if has_forecast:
        last_forecast = df_forecast.iloc[-1]
        accuracy = 100 - df_forecast["error_pct"].mean()
        rmse = np.sqrt(np.mean((df_forecast["actual"] - df_forecast["prediction"]) ** 2))
        mae = df_forecast["error"].mean()

        with col1:
            st.metric(
                "📅 최근 예측",
                f"₩{last_forecast['prediction']:.2f}",
                f"실제: ₩{last_forecast['actual']:.2f}"
            )

        with col2:
            st.metric(
                "🎯 정확도",
                f"{accuracy:.1f}%",
                delta=f"-{df_forecast['error_pct'].mean():.1f}%"
            )

        with col3:
            st.metric(
                "📊 RMSE",
                f"{rmse:.4f}",
                help="Root Mean Squared Error"
            )

        with col4:
            st.metric(
                "📏 MAE",
                f"{mae:.4f}",
                help="Mean Absolute Error"
            )
    else:
        st.warning("선택 기간에 예측 데이터 없음 (예측 데이터는 2019-12-16 ~ 2019-12-31 기간에만 존재합니다)")
        with col1:
            st.metric("📅 최근 예측", "N/A")
        with col2:
            st.metric("🎯 정확도", "N/A")
        with col3:
            st.metric("📊 RMSE", "N/A")
        with col4:
            st.metric("📏 MAE", "N/A")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # 예측 vs 실제 라인 차트  |  통계
    # ─────────────────────────────────────────────
    col1, col2 = st.columns([3, 1.2])

    with col1:
        st.subheader("예측 vs 실제 매출 추이")

        fig = go.Figure()

        # 실제 매출 (항상 표시)
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["actual"],
            name="실제 매출",
            mode="lines+markers",
            line=dict(color="#222222", width=2.5),
            marker=dict(size=5)
        ))

        # 예측 매출 (예측 기간만)
        if has_forecast:
            fig.add_trace(go.Scatter(
                x=df_forecast["date"],
                y=df_forecast["prediction"],
                name="예측 매출",
                mode="lines+markers",
                line=dict(color="#E50010", width=2, dash="dash"),
                marker=dict(size=6)
            ))

            # 신뢰 구간 (±1σ)
            std_error = df_forecast["error"].std()
            upper = df_forecast["prediction"] + std_error
            lower = df_forecast["prediction"] - std_error

            fig.add_trace(go.Scatter(
                x=df_forecast["date"],
                y=upper,
                fill=None,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
            ))

            fig.add_trace(go.Scatter(
                x=df_forecast["date"],
                y=lower,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                name="신뢰 구간 (±1σ)",
                fillcolor="rgba(229,0,16,0.10)"
            ))

        # Y축 범위 계산 (변동이 잘 보이도록)
        all_vals = df["actual"].dropna().tolist()
        if has_forecast:
            all_vals += df_forecast["prediction"].dropna().tolist()
        y_min = max(0, min(all_vals) - 10)
        y_max = max(all_vals) + 10

        # 예측 시작 지점 표시 (shapes로 추가 — 데이터 범위에 영향 안 줌)
        shapes = []
        if has_forecast:
            forecast_start = df_forecast["date"].min()
            shapes.append(dict(
                type="line",
                x0=forecast_start, x1=forecast_start,
                y0=0, y1=1, yref="paper",
                line=dict(width=1, dash="dot", color="rgba(150,150,150,0.5)"),
            ))

        fig.update_layout(
            shapes=shapes,
            title="일일 매출 예측",
            xaxis_title="날짜",
            yaxis_title="매출액 (₩)",
            hovermode="x unified",
            height=480,
            template="plotly_white",
            font=dict(family="Inter, Helvetica Neue, Arial, sans-serif"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            yaxis=dict(
                tickformat=",.0f",
                tickprefix="₩",
                range=[y_min, y_max],
                dtick=10,
                gridcolor="rgba(0,0,0,0.06)",
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0.06)",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("통계")
        st.write(f"**기간:** {df['date'].min().date()} ~ {df['date'].max().date()}")
        st.write(f"**총 일수:** {len(df)}일")
        st.write(f"**평균 실제 매출:** ₩{df['actual'].mean():.2f}")
        if has_forecast:
            st.write(f"**평균 예측 매출:** ₩{df_forecast['prediction'].mean():.2f}")
            st.write(f"**평균 오차:** ₩{df_forecast['error'].mean():.2f}")
        else:
            st.write("**평균 예측 매출:** N/A")
            st.write("**평균 오차:** N/A")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # 요일별 패턴  |  오차 분포
    # ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("요일별 패턴")

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_names_kr = ["월", "화", "수", "목", "금", "토", "일"]

        # day_of_week 컬럼 생성 (daily_sales에서 가져온 값 사용 or 직접 계산)
        if "day_of_week" in df.columns:
            df_dow = df.copy()
        else:
            df_dow = df.copy()
            df_dow["day_of_week"] = df_dow["date"].dt.day_name()

        daily_stats_actual = df_dow.groupby("day_of_week")["actual"].mean()
        daily_stats_actual = daily_stats_actual.reindex(
            [d for d in day_order if d in daily_stats_actual.index]
        )

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[day_names_kr[day_order.index(d)] for d in daily_stats_actual.index],
            y=daily_stats_actual.values,
            name="실제 평균",
            marker_color="#222222"
        ))

        if has_forecast:
            df_forecast_dow = df_forecast.copy()
            if "day_of_week" not in df_forecast_dow.columns:
                df_forecast_dow["day_of_week"] = df_forecast_dow["date"].dt.day_name()

            daily_stats_pred = df_forecast_dow.groupby("day_of_week")["prediction"].mean()
            daily_stats_pred = daily_stats_pred.reindex(
                [d for d in day_order if d in daily_stats_pred.index]
            )

            fig.add_trace(go.Bar(
                x=[day_names_kr[day_order.index(d)] for d in daily_stats_pred.index],
                y=daily_stats_pred.values,
                name="예측 평균",
                marker_color="#E50010"
            ))

        fig.update_layout(
            title="요일별 평균 매출",
            xaxis_title="요일",
            yaxis_title="매출액 (₩)",
            barmode="group",
            height=380,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("예측 오차 분포")

        if has_forecast:
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=df_forecast["error"],
                nbinsx=15,
                name="오차",
                marker_color="#E50010"
            ))

            fig.update_layout(
                title="예측 오차 분포",
                xaxis_title="오차 (₩)",
                yaxis_title="빈도",
                height=350,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("선택 기간에 예측 데이터가 없어 오차 분포를 표시할 수 없습니다.")

    # ─────────────────────────────────────────────
    # 상세 테이블
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 상세 데이터")

    display_df = df[["date", "actual", "prediction", "error", "error_pct"]].copy()
    display_df.columns = ["날짜", "실제 매출", "예측 매출", "오차 (₩)", "오차율 (%)"]
    display_df["날짜"] = display_df["날짜"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ─────────────────────────────────────────────
    # 인사이트
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 모델 성능 인사이트")

    dow_kr = {
        "Monday": "월", "Tuesday": "화", "Wednesday": "수",
        "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"
    }

    col_i1, col_i2, col_i3 = st.columns(3)

    with col_i1:
        if has_forecast:
            accuracy = 100 - df_forecast["error_pct"].mean()
            best_day = df_forecast.loc[df_forecast["error_pct"].idxmin()]
            worst_day = df_forecast.loc[df_forecast["error_pct"].idxmax()]
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
        else:
            st.info("🎯 **모델 정확도**\n\n선택 기간에 예측 데이터가 없습니다.")

    with col_i2:
        if has_forecast:
            df_forecast_dow2 = df_forecast.copy()
            if "day_of_week" not in df_forecast_dow2.columns:
                df_forecast_dow2["day_of_week"] = df_forecast_dow2["date"].dt.day_name()

            dow_error = df_forecast_dow2.groupby("day_of_week")["error_pct"].mean()
            best_dow = dow_error.idxmin()
            worst_dow = dow_error.idxmax()

            st.info(f"📅 **요일별 예측 품질**\n\n"
                    f"- 최고 정확도 요일: **{dow_kr.get(best_dow, best_dow)}요일**\n"
                    f"- 최저 정확도 요일: **{dow_kr.get(worst_dow, worst_dow)}요일**\n\n"
                    f"{dow_kr.get(worst_dow, worst_dow)}요일 데이터 보강 또는\n추가 피처 엔지니어링 권장")
        else:
            st.info("📅 **요일별 예측 품질**\n\n선택 기간에 예측 데이터가 없습니다.")

    with col_i3:
        if len(df) >= 2:
            half = max(1, len(df) // 2)
            first_half_mean = df["actual"].iloc[:half].mean()
            second_half_mean = df["actual"].iloc[-half:].mean()
            trend = "상승" if second_half_mean > first_half_mean else "하락"
            trend_pct = abs((second_half_mean - first_half_mean) / first_half_mean * 100) if first_half_mean != 0 else 0

            if trend == "상승":
                fn2 = st.success
            else:
                fn2 = st.warning
            fn2(f"📈 **매출 트렌드: {trend}**\n\n"
                f"- 기간 내 변화: **{trend_pct:.1f}%** {trend}\n"
                f"- 전반부 평균: **₩{first_half_mean:.2f}**\n"
                f"- 후반부 평균: **₩{second_half_mean:.2f}**\n\n"
                f"트렌드 반영한 재학습으로 정확도 향상 가능")
        else:
            st.info("📈 **매출 트렌드**\n\n데이터가 부족하여 트렌드를 계산할 수 없습니다.")
