"""
고객 세그먼트 분석 페이지
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def map_segment_names(segment_key: str) -> str:
    """세그먼트 키를 UI용 이름으로 변환"""
    if segment_key == 'active':
        return 'Active'
    elif segment_key == 'inactive':
        return 'Inactive'
    elif segment_key.startswith('age_'):
        return segment_key.replace('age_', '')
    elif segment_key.startswith('club_'):
        return segment_key.replace('club_', '')
    elif segment_key.startswith('news_'):
        return segment_key.replace('news_', '')
    return segment_key


def get_segment_type(segment_key: str) -> str:
    """세그먼트 키로부터 타입 판단"""
    if segment_key in ['active', 'inactive']:
        return '활동도'
    elif segment_key.startswith('age_'):
        return '연령대'
    elif segment_key.startswith('club_'):
        return '클럽 상태'
    elif segment_key.startswith('news_'):
        return '뉴스레터'
    return None


def show():
    """페이지 표시"""
    st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:11px; color:#767676; margin-bottom:12px; letter-spacing:0.3px;'>
        홈 / 분석 / <span style='color:#000000; font-weight:600;'>고객 분석</span>
    </div>
    <h1 style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;
               font-size:1.8rem; font-weight:700; color:#000000;
               margin:0 0 4px 0; letter-spacing:0.5px; text-transform:uppercase;'>
        고객 세그먼트 분석
    </h1>
    <div style='height:2px; background:#000000; width:40px; margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)

    # 데이터 로드
    project_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_dir))

    from app.utils.cache_manager import create_segment_analyzer, load_raw_data

    analyzer, ts_gen = create_segment_analyzer()
    merged_data = load_raw_data()

    if analyzer is None or merged_data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    # 모든 세그먼트 생성
    all_segments = analyzer.create_customer_segments()

    # 세그먼트를 타입별로 분류
    segment_types = {}
    for seg_key in all_segments.keys():
        seg_type = get_segment_type(seg_key)
        if seg_type not in segment_types:
            segment_types[seg_type] = []
        segment_types[seg_type].append(seg_key)

    # 세그먼트 선택
    col1, col2, col3 = st.columns(3)

    with col1:
        segment_type = st.selectbox(
            "세그먼트 유형 선택",
            list(segment_types.keys())
        )

    # 선택된 타입의 세그먼트 목록
    available_segments = segment_types.get(segment_type, [])
    display_names = {seg: map_segment_names(seg) for seg in available_segments}

    with col2:
        selected_display = st.multiselect(
            "세그먼트 선택",
            list(display_names.values()),
            default=list(display_names.values())[:1]
        )
        # UI이름을 다시 키로 변환
        selected_segments = [
            seg for seg, display_name in display_names.items()
            if display_name in selected_display
        ]

    with col3:
        date_range = st.date_input(
            "기간 선택",
            value=(pd.Timestamp("2019-12-01"), pd.Timestamp("2019-12-31")),
            key="date_range"
        )

    st.markdown("---")

    # 날짜 범위 유효성 검사
    if isinstance(date_range, tuple) and len(date_range) == 2:
        pass  # 정상
    else:
        st.info("날짜 범위를 선택해주세요.")
        return

    # 선택된 세그먼트의 시계열 데이터 생성
    segments_data = {}
    segment_stats = {}

    for segment_key in selected_segments:
        segment_data = all_segments[segment_key]
        if len(segment_data) > 0:
            # 시계열 생성
            ts = ts_gen.create_segment_timeseries(
                segment_data, segment_key
            )

            # 기간 필터링
            ts_filtered = ts[
                (ts['date'] >= pd.Timestamp(date_range[0])) &
                (ts['date'] <= pd.Timestamp(date_range[1]))
            ]

            if len(ts_filtered) > 0:
                segments_data[segment_key] = ts_filtered
                segment_stats[segment_key] = {
                    'count': len(segment_data),
                    'total_sales': segment_data['price'].sum(),
                    'avg_price': segment_data['price'].mean(),
                }

    # KPI 표시
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for i, segment_key in enumerate(selected_segments[:4]):
        if segment_key in segment_stats:
            with cols[i]:
                stats = segment_stats[segment_key]
                st.metric(
                    f"{map_segment_names(segment_key)}",
                    f"₩{stats['total_sales']:,.0f}",
                    f"{stats['count']:,}명"
                )

    st.markdown("---")

    # 세그먼트별 매출 추이
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{segment_type}별 매출 추이")

        if len(segments_data) > 0:
            fig = go.Figure()

            for segment_key in selected_segments:
                if segment_key in segments_data:
                    ts_filtered = segments_data[segment_key]
                    fig.add_trace(go.Scatter(
                        x=ts_filtered['date'],
                        y=ts_filtered['sales'],
                        name=map_segment_names(segment_key),
                        mode='lines+markers'
                    ))

            fig.update_layout(
                title=f"{segment_type}별 일일 매출",
                xaxis_title="날짜",
                yaxis_title="매출액 (₩)",
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("해당 기간에 데이터가 없습니다.")

    with col2:
        st.subheader("세그먼트별 비중")

        if len(segment_stats) > 0:
            sales_values = [
                segment_stats[seg]['total_sales']
                for seg in selected_segments if seg in segment_stats
            ]
            labels = [
                map_segment_names(seg)
                for seg in selected_segments if seg in segment_stats
            ]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=sales_values,
                marker=dict(line=dict(color='white', width=2))
            )])

            fig.update_layout(
                title="전체 매출 비중",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("데이터가 없습니다.")

    st.markdown("---")

    # 세그먼트별 통계
    st.subheader("📊 세그먼트별 통계")

    if len(segments_data) > 0:
        stats_data = []
        for segment_key in selected_segments:
            if segment_key in segments_data:
                ts = segments_data[segment_key]
                stats = segment_stats[segment_key]
                sales = ts['sales'].values
                stats_data.append({
                    "세그먼트": map_segment_names(segment_key),
                    "고객 수": f"{stats['count']:,}명",
                    "총 매출": f"₩{stats['total_sales']:,.0f}",
                    "평균 매출": f"₩{sales.mean():,.0f}",
                    "최대 매출": f"₩{sales.max():,.0f}",
                    "최소 매출": f"₩{sales.min():,.0f}",
                    "표준편차": f"₩{sales.std():,.0f}",
                })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("통계 데이터가 없습니다.")

    st.markdown("---")

    # ==================== 동적 인사이트 ====================
    st.subheader("💡 세그먼트 인사이트")

    if len(segment_stats) >= 1:
        # 선택된 세그먼트들의 실제 데이터로 인사이트 계산
        stats_list = [(k, v) for k, v in segment_stats.items() if k in selected_segments]

        if segment_type == '활동도':
            active = segment_stats.get('active', {})
            inactive = segment_stats.get('inactive', {})
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                if active:
                    st.success(f"✅ **활성 고객**\n\n"
                               f"- 고객 수: **{active['count']:,}명**\n"
                               f"- 총 매출: **₩{active['total_sales']:,.0f}**\n"
                               f"- 평균 거래액: **₩{active['avg_price']:.4f}**\n\n"
                               f"구매 패턴이 규칙적이며 충성도 높음")
            with col_i2:
                if inactive:
                    st.error(f"⚠️ **비활성 고객**\n\n"
                             f"- 고객 수: **{inactive['count']:,}명**\n"
                             f"- 총 매출: **₩{inactive['total_sales']:,.0f}**\n"
                             f"- 평균 거래액: **₩{inactive['avg_price']:.4f}**\n\n"
                             f"재활성화 캠페인이 시급함")
            with col_i3:
                if active and inactive:
                    ratio = active['avg_price'] / inactive['avg_price'] if inactive['avg_price'] > 0 else 1
                    st.info(f"📊 **활성 vs 비활성 비교**\n\n"
                            f"- 평균 거래액 차이: **{ratio:.2f}배**\n"
                            f"- 활성 고객 비율: **{active['count']/(active['count']+inactive['count'])*100:.1f}%**\n\n"
                            f"활성 고객이 비활성 대비 더 높은 가치")

        elif segment_type == '연령대':
            sorted_by_sales = sorted(stats_list, key=lambda x: x[1]['total_sales'], reverse=True)
            sorted_by_price = sorted(stats_list, key=lambda x: x[1]['avg_price'], reverse=True)
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                if sorted_by_sales:
                    top = sorted_by_sales[0]
                    st.success(f"🏆 **최고 매출 연령대**\n\n"
                               f"- 연령대: **{map_segment_names(top[0])}**\n"
                               f"- 총 매출: **₩{top[1]['total_sales']:,.0f}**\n"
                               f"- 고객 수: **{top[1]['count']:,}명**\n\n"
                               f"핵심 타겟 연령대로 집중 마케팅 필요")
            with col_i2:
                if sorted_by_price:
                    top_price = sorted_by_price[0]
                    st.info(f"💎 **건당 최고 지출 연령대**\n\n"
                            f"- 연령대: **{map_segment_names(top_price[0])}**\n"
                            f"- 평균 거래액: **₩{top_price[1]['avg_price']:.4f}**\n\n"
                            f"프리미엄 라인 마케팅 적합")
            with col_i3:
                if len(sorted_by_sales) > 1:
                    bottom = sorted_by_sales[-1]
                    st.warning(f"📈 **성장 가능 연령대**\n\n"
                               f"- 연령대: **{map_segment_names(bottom[0])}**\n"
                               f"- 현재 매출: **₩{bottom[1]['total_sales']:,.0f}**\n\n"
                               f"맞춤형 콘텐츠로 구매 유도 가능")

        elif segment_type == '클럽 상태':
            active_club = segment_stats.get('club_ACTIVE', {})
            pre_create = segment_stats.get('club_PRE-CREATE', {})
            left_club = segment_stats.get('club_LEFT CLUB', {})
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                if active_club:
                    st.success(f"🏆 **ACTIVE 클럽 회원**\n\n"
                               f"- 고객 수: **{active_club['count']:,}명**\n"
                               f"- 총 매출: **₩{active_club['total_sales']:,.0f}**\n"
                               f"- 평균 거래액: **₩{active_club['avg_price']:.4f}**\n\n"
                               f"가장 충성도 높은 핵심 고객층")
            with col_i2:
                if pre_create:
                    pct = pre_create['count'] / (sum(v['count'] for v in segment_stats.values())) * 100
                    st.warning(f"⏳ **PRE-CREATE 고객**\n\n"
                               f"- 고객 수: **{pre_create['count']:,}명** ({pct:.1f}%)\n"
                               f"- 평균 거래액: **₩{pre_create['avg_price']:.4f}**\n\n"
                               f"정회원 전환 유도 전략 필요")
            with col_i3:
                if left_club:
                    if active_club:
                        ratio = active_club['avg_price'] / left_club['avg_price'] if left_club['avg_price'] > 0 else 1
                        st.error(f"🚪 **LEFT CLUB 고객**\n\n"
                                 f"- 고객 수: **{left_club['count']:,}명**\n"
                                 f"- ACTIVE 대비: **{ratio:.1f}배 낮은** 거래액\n\n"
                                 f"재가입 캠페인으로 가치 복구 가능")
                    else:
                        st.error(f"🚪 **LEFT CLUB 고객**\n\n"
                                 f"- 고객 수: **{left_club['count']:,}명**\n\n"
                                 f"재가입 캠페인 필요")

        elif segment_type == '뉴스레터':
            none_stats = segment_stats.get('news_NONE', {})
            monthly_stats = segment_stats.get('news_Monthly', {})
            regularly_stats = segment_stats.get('news_Regularly', {})
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                if none_stats:
                    total = sum(v['count'] for v in segment_stats.values())
                    pct = none_stats['count'] / total * 100
                    st.warning(f"📭 **미구독 고객 (NONE)**\n\n"
                               f"- 고객 수: **{none_stats['count']:,}명** ({pct:.1f}%)\n"
                               f"- 평균 거래액: **₩{none_stats['avg_price']:.4f}**\n\n"
                               f"구독 전환 시 매출 증대 잠재력 높음")
            with col_i2:
                if regularly_stats:
                    st.success(f"📬 **정기 구독자 (Regularly)**\n\n"
                               f"- 고객 수: **{regularly_stats['count']:,}명**\n"
                               f"- 평균 거래액: **₩{regularly_stats['avg_price']:.4f}**\n\n"
                               f"가장 높은 구매 참여도 보유")
            with col_i3:
                if regularly_stats and none_stats:
                    ratio = regularly_stats['avg_price'] / none_stats['avg_price'] if none_stats['avg_price'] > 0 else 1
                    st.info(f"📊 **구독 효과 분석**\n\n"
                            f"- 정기구독 vs 미구독: **{ratio:.2f}배**\n"
                            f"- 구독 유도 ROI 높음\n\n"
                            f"이메일 마케팅 강화 권장")
    else:
        st.info("세그먼트를 선택하면 인사이트가 표시됩니다.")

    st.markdown("---")

    # 고객 특성 분석
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("세그먼트별 고객 수")

        if len(segment_stats) > 0:
            counts = [
                segment_stats[seg]['count']
                for seg in selected_segments if seg in segment_stats
            ]
            labels = [
                map_segment_names(seg)
                for seg in selected_segments if seg in segment_stats
            ]

            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=counts,
                marker_color='#E50010'
            )])

            fig.update_layout(
                title="세그먼트별 고객 수",
                xaxis_title="세그먼트",
                yaxis_title="고객 수",
                height=350,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("데이터가 없습니다.")

    with col2:
        st.subheader("평균 거래액")

        if len(segment_stats) > 0:
            avg_prices = [
                segment_stats[seg]['avg_price']
                for seg in selected_segments if seg in segment_stats
            ]
            labels = [
                map_segment_names(seg)
                for seg in selected_segments if seg in segment_stats
            ]

            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=avg_prices,
                marker_color='#222222'
            )])

            fig.update_layout(
                title="세그먼트별 평균 거래액",
                xaxis_title="세그먼트",
                yaxis_title="평균 거래액 (₩)",
                height=350,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("데이터가 없습니다.")
