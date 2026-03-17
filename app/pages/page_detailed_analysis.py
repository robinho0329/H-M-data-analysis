"""
상세 분석 페이지 - 실데이터 기반
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
def load_detailed_data():
    """상세 분석용 실데이터 로드 및 집계"""
    from app.utils.cache_manager import load_raw_data
    merged = load_raw_data()
    if merged is None:
        return None

    # 색상별 통계
    color_stats = merged.groupby('colour_group_name').agg(
        revenue=('price', 'sum'),
        count=('price', 'count'),
        avg_price=('price', 'mean'),
        unique_customers=('customer_id', 'nunique')
    ).reset_index().sort_values('revenue', ascending=False).head(10)

    # 제품 타입별 통계
    product_stats = merged.groupby('product_type_name').agg(
        revenue=('price', 'sum'),
        count=('price', 'count'),
        avg_price=('price', 'mean')
    ).reset_index().sort_values('revenue', ascending=False).head(10)

    # 의류 그룹별 통계
    garment_stats = merged.groupby('garment_group_name').agg(
        revenue=('price', 'sum'),
        count=('price', 'count')
    ).reset_index().sort_values('revenue', ascending=False)

    # 월별 매출
    monthly = merged.copy()
    monthly['month'] = monthly['t_dat'].dt.month
    monthly['month_name'] = monthly['t_dat'].dt.strftime('%m월')
    monthly_stats = monthly.groupby('month').agg(
        revenue=('price', 'sum'),
        count=('price', 'count'),
        avg_price=('price', 'mean'),
        month_name=('month_name', 'first')
    ).reset_index().sort_values('month')

    # 요일별 매출
    dow = merged.copy()
    dow['day_of_week'] = dow['t_dat'].dt.dayofweek
    dow_stats = dow.groupby('day_of_week').agg(
        avg_revenue=('price', 'mean'),
        total_revenue=('price', 'sum'),
        count=('price', 'count')
    ).reset_index()

    # 분기별 매출
    quarterly = merged.copy()
    quarterly['quarter'] = quarterly['t_dat'].dt.quarter
    q_stats = quarterly.groupby('quarter').agg(
        revenue=('price', 'sum'),
        count=('price', 'count')
    ).reset_index()

    # 연령대별 색상 선호도 (Top 4 색상 vs 연령대)
    age_color_cross = None
    top4_colors = []
    if 'age' in merged.columns:
        cross = merged.copy()
        cross['age_group'] = pd.cut(cross['age'],
            bins=[0, 20, 30, 40, 50, 60, 120],
            labels=['10대', '20대', '30대', '40대', '50대', '60대+'])
        top4_colors = merged.groupby('colour_group_name')['price'].count().nlargest(4).index.tolist()
        cross_filtered = cross[cross['colour_group_name'].isin(top4_colors)]
        age_color_cross = cross_filtered.groupby(
            ['age_group', 'colour_group_name'], observed=True
        )['price'].count().unstack(fill_value=0)
        # 정규화 (비율)
        row_sums = age_color_cross.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # 0으로 나누기 방지
        age_color_cross = age_color_cross.div(row_sums, axis=0)

    # 활동 고객 vs 비활동 고객 선호 제품 (Top 5)
    active_product = None
    if 'Active' in merged.columns:
        top5_products = merged.groupby('product_type_name')['price'].count().nlargest(5).index.tolist()
        ap = merged[merged['product_type_name'].isin(top5_products)]
        active_product = ap.groupby(['Active', 'product_type_name'])['price'].agg('count').unstack(fill_value=0)

    # 뉴스레터별 평균 지출
    news_stats = None
    if 'fashion_news_frequency' in merged.columns:
        news_stats = merged.groupby('fashion_news_frequency').agg(
            avg_revenue=('price', 'mean'),
            total_revenue=('price', 'sum'),
            count=('price', 'count'),
            unique_customers=('customer_id', 'nunique')
        ).reset_index()

    # 색상 월별 트렌드 (Top 3 색상)
    top3_colors = merged.groupby('colour_group_name')['price'].count().nlargest(3).index.tolist()
    color_monthly = merged[merged['colour_group_name'].isin(top3_colors)].copy()
    color_monthly['month'] = color_monthly['t_dat'].dt.to_period('M').astype(str)
    color_monthly_stats = color_monthly.groupby(['month', 'colour_group_name'])['price'].sum().reset_index()

    return {
        'color_stats': color_stats,
        'product_stats': product_stats,
        'garment_stats': garment_stats,
        'monthly_stats': monthly_stats,
        'dow_stats': dow_stats,
        'q_stats': q_stats,
        'age_color_cross': age_color_cross,
        'top4_colors': top4_colors,
        'active_product': active_product,
        'news_stats': news_stats,
        'color_monthly_stats': color_monthly_stats,
        'top3_colors': top3_colors,
    }


def show():
    """페이지 표시"""
    st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:11px; color:#767676; margin-bottom:12px; letter-spacing:0.3px;'>
        Home / Analytics / <span style='color:#000000; font-weight:600;'>DETAILED ANALYSIS</span>
    </div>
    <h1 style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;
               font-size:1.8rem; font-weight:700; color:#000000;
               margin:0 0 4px 0; letter-spacing:0.5px; text-transform:uppercase;'>
        DETAILED ANALYSIS
    </h1>
    <div style='height:2px; background:#000000; width:40px; margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)

    data = load_detailed_data()
    if data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎨 색상 분석", "👕 제품 분석", "📅 계절성 분석", "🔗 고객-상품 교차분석"]
    )

    # ── Tab 1: 색상 분석 ─────────────────────────────────────────────────────
    with tab1:
        st.subheader("🎨 색상별 심층 분석")

        color_stats = data['color_stats']
        color_monthly_stats = data['color_monthly_stats']
        top3_colors = data['top3_colors']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 색상별 총매출**")
            fig = go.Figure(data=[go.Bar(
                x=color_stats['revenue'],
                y=color_stats['colour_group_name'],
                orientation='h',
                marker_color='#E50010',
                text=color_stats['revenue'].apply(lambda v: f"{v:,.0f}"),
                textposition='outside'
            )])
            fig.update_layout(
                title="색상별 총매출 (Top 10)",
                xaxis_title="총매출 (SEK)",
                yaxis_title="색상",
                yaxis=dict(autorange='reversed'),
                height=380,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Top 3 색상 월별 매출 트렌드**")
            fig = go.Figure()
            palette = ['#E50010', '#222222', '#FF6B6B']
            for i, color_name in enumerate(top3_colors):
                subset = color_monthly_stats[color_monthly_stats['colour_group_name'] == color_name]
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset['month'],
                        y=subset['price'],
                        name=color_name,
                        mode='lines+markers',
                        line=dict(color=palette[i % len(palette)])
                    ))
            fig.update_layout(
                title="Top 3 색상 월별 매출 추이",
                xaxis_title="월",
                yaxis_title="매출 (SEK)",
                height=380,
                template='plotly_white',
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig, use_container_width=True)

        # 버블차트
        st.markdown("**Top 10 색상 거래건수 vs 건당평균가격 버블차트**")
        fig = go.Figure(data=[go.Scatter(
            x=color_stats['count'],
            y=color_stats['avg_price'],
            mode='markers+text',
            marker=dict(
                size=color_stats['revenue'] / color_stats['revenue'].max() * 60 + 10,
                color=list(range(len(color_stats))),
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="순위")
            ),
            text=color_stats['colour_group_name'],
            textposition='top center'
        )])
        fig.update_layout(
            title="색상별 거래건수 vs 건당평균가격 (버블 크기 = 총매출)",
            xaxis_title="거래건수",
            yaxis_title="건당평균가격 (SEK)",
            height=380,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        # 인사이트
        if len(color_stats) >= 2:
            top1 = color_stats.iloc[0]
            top2 = color_stats.iloc[1]
            diff_pct = (top1['revenue'] - top2['revenue']) / top2['revenue'] * 100
            st.success(
                f"인사이트: 1위 색상 **{top1['colour_group_name']}** "
                f"(총매출: {top1['revenue']:,.0f} SEK) 은(는) "
                f"2위 **{top2['colour_group_name']}** 대비 **{diff_pct:.1f}%** 높은 매출을 기록했습니다."
            )

    # ── Tab 2: 제품 분석 ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("👕 제품 타입 심층 분석")

        garment_stats = data['garment_stats']
        product_stats = data['product_stats']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**의류 그룹별 매출 비중 (도넛)**")
            fig = go.Figure(data=[go.Pie(
                labels=garment_stats['garment_group_name'],
                values=garment_stats['revenue'],
                hole=0.4,
                textinfo='label+percent'
            )])
            fig.update_layout(
                title="의류 그룹별 매출 비중",
                height=380,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Top 10 제품 타입 거래건수**")
            fig = go.Figure(data=[go.Bar(
                x=product_stats['count'],
                y=product_stats['product_type_name'],
                orientation='h',
                marker_color='#222222',
                text=product_stats['count'].apply(lambda v: f"{v:,}"),
                textposition='outside'
            )])
            fig.update_layout(
                title="제품 타입별 거래건수 (Top 10)",
                xaxis_title="거래건수",
                yaxis_title="제품 타입",
                yaxis=dict(autorange='reversed'),
                height=380,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        # 산점도
        st.markdown("**제품 타입별 거래량 vs 평균가격 산점도**")
        fig = go.Figure(data=[go.Scatter(
            x=product_stats['count'],
            y=product_stats['avg_price'],
            mode='markers+text',
            marker=dict(
                size=product_stats['revenue'] / product_stats['revenue'].max() * 50 + 10,
                color=product_stats['revenue'],
                colorscale='OrRd',
                showscale=True,
                colorbar=dict(title="총매출")
            ),
            text=product_stats['product_type_name'],
            textposition='top center'
        )])
        fig.update_layout(
            title="제품 타입별 거래량 vs 평균가격 (버블 크기 = 총매출)",
            xaxis_title="거래건수",
            yaxis_title="평균가격 (SEK)",
            height=380,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        # 인사이트
        top_revenue_prod = product_stats.sort_values('revenue', ascending=False).iloc[0]
        top_price_prod = product_stats.sort_values('avg_price', ascending=False).iloc[0]
        st.success(
            f"인사이트: 최고 수익 제품 타입은 **{top_revenue_prod['product_type_name']}** "
            f"(총매출: {top_revenue_prod['revenue']:,.0f} SEK), "
            f"건당 최고 평균가격 제품 타입은 **{top_price_prod['product_type_name']}** "
            f"({top_price_prod['avg_price']:.2f} SEK/건) 입니다."
        )

    # ── Tab 3: 계절성 분석 ───────────────────────────────────────────────────
    with tab3:
        st.subheader("📅 계절성 분석")

        monthly_stats = data['monthly_stats']
        dow_stats = data['dow_stats']
        q_stats = data['q_stats']

        # 월별 복합차트 (Bar + Line + 계절 배경)
        st.markdown("**월별 매출 추이 (계절 배경 포함)**")
        month_labels = monthly_stats['month_name'].tolist()

        fig = go.Figure()

        # 계절 배경색 shading
        season_shapes = [
            dict(type='rect', x0=2.5, x1=5.5, y0=0, y1=1, yref='paper',
                 fillcolor='rgba(144,238,144,0.15)', line_width=0, layer='below'),  # 봄=연초록
            dict(type='rect', x0=5.5, x1=8.5, y0=0, y1=1, yref='paper',
                 fillcolor='rgba(255,255,153,0.20)', line_width=0, layer='below'),  # 여름=연노랑
            dict(type='rect', x0=8.5, x1=11.5, y0=0, y1=1, yref='paper',
                 fillcolor='rgba(255,165,0,0.12)', line_width=0, layer='below'),   # 가을=연주황
        ]
        # 겨울 (1~2월 + 12월) - 두 구간으로 나눔
        season_shapes.append(
            dict(type='rect', x0=0.5, x1=2.5, y0=0, y1=1, yref='paper',
                 fillcolor='rgba(135,206,235,0.20)', line_width=0, layer='below')  # 겨울=연파랑
        )
        season_shapes.append(
            dict(type='rect', x0=11.5, x1=12.5, y0=0, y1=1, yref='paper',
                 fillcolor='rgba(135,206,235,0.20)', line_width=0, layer='below')
        )

        fig.add_trace(go.Bar(
            x=month_labels,
            y=monthly_stats['revenue'],
            name='월별 총매출',
            marker_color='#E50010',
            opacity=0.8
        ))
        fig.add_trace(go.Scatter(
            x=month_labels,
            y=monthly_stats['revenue'],
            name='추세선',
            mode='lines+markers',
            line=dict(color='#CC0000', width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="월별 총매출 추이 (봄=초록 / 여름=노랑 / 가을=주황 / 겨울=파랑)",
            xaxis_title="월",
            yaxis_title="총매출 (SEK)",
            height=420,
            template='plotly_white',
            shapes=season_shapes,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**분기별 매출 비교**")
            q_labels = [f"Q{int(q)}" for q in q_stats['quarter']]
            fig = go.Figure(data=[go.Bar(
                x=q_labels,
                y=q_stats['revenue'],
                marker_color=['#E50010', '#222222', '#FF6B6B', '#CC0000'],
                text=q_stats['revenue'].apply(lambda v: f"{v:,.0f}"),
                textposition='outside'
            )])
            fig.update_layout(
                title="분기별 총매출",
                xaxis_title="분기",
                yaxis_title="총매출 (SEK)",
                height=380,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**요일별 평균 매출**")
            day_names = ['월', '화', '수', '목', '금', '토', '일']
            dow_stats_sorted = dow_stats.sort_values('day_of_week')
            day_labels = [day_names[int(d)] for d in dow_stats_sorted['day_of_week']]
            # 주말(5, 6)은 주황, 평일은 파랑
            bar_colors = [
                '#222222' if int(d) >= 5 else '#E50010'
                for d in dow_stats_sorted['day_of_week']
            ]
            fig = go.Figure(data=[go.Bar(
                x=day_labels,
                y=dow_stats_sorted['avg_revenue'],
                marker_color=bar_colors,
                text=dow_stats_sorted['avg_revenue'].apply(lambda v: f"{v:.4f}"),
                textposition='outside'
            )])
            fig.update_layout(
                title="요일별 평균 매출 (빨강=평일 / 검정=주말)",
                xaxis_title="요일",
                yaxis_title="평균 매출 (SEK)",
                height=380,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        # 인사이트
        if not monthly_stats.empty:
            max_month = monthly_stats.loc[monthly_stats['revenue'].idxmax()]
            min_month = monthly_stats.loc[monthly_stats['revenue'].idxmin()]

            weekend_mask = dow_stats_sorted['day_of_week'] >= 5
            weekday_avg = dow_stats_sorted[~weekend_mask]['avg_revenue'].mean()
            weekend_avg = dow_stats_sorted[weekend_mask]['avg_revenue'].mean()

            st.info(
                f"인사이트: 최고 매출월은 **{max_month['month_name']}** "
                f"({max_month['revenue']:,.0f} SEK), "
                f"최저 매출월은 **{min_month['month_name']}** "
                f"({min_month['revenue']:,.0f} SEK)입니다. "
                f"주말 평균 매출({weekend_avg:.4f} SEK)은 "
                f"평일 평균 매출({weekday_avg:.4f} SEK) 대비 "
                f"**{(weekend_avg / weekday_avg - 1) * 100:.1f}%** "
                f"{'높습니다' if weekend_avg > weekday_avg else '낮습니다'}."
            )

    # ── Tab 4: 고객-상품 교차분석 ────────────────────────────────────────────
    with tab4:
        st.subheader("🔗 고객-상품 교차 분석")

        age_color_cross = data['age_color_cross']
        active_product = data['active_product']
        news_stats = data['news_stats']

        # 연령대별 Top 4 색상 선호도 Heatmap
        st.markdown("**연령대별 Top 4 색상 선호도 Heatmap**")
        if age_color_cross is not None and not age_color_cross.empty:
            z_values = age_color_cross.values.tolist()
            y_labels = [str(idx) for idx in age_color_cross.index.tolist()]
            x_labels = age_color_cross.columns.tolist()

            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                colorscale='Reds',
                text=[[f"{v:.2%}" for v in row] for row in z_values],
                texttemplate="%{text}",
                hovertemplate="연령대: %{y}<br>색상: %{x}<br>비율: %{text}<extra></extra>"
            ))
            fig.update_layout(
                title="연령대별 색상 선호도 (비율)",
                xaxis_title="색상",
                yaxis_title="연령대",
                height=380,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("연령 데이터가 없어 히트맵을 표시할 수 없습니다.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**활동/비활동 고객 Top 5 제품 선호 비교**")
            if active_product is not None and not active_product.empty:
                fig = go.Figure()
                product_cols = active_product.columns.tolist()
                for idx_val, color_bar, label in [(0, '#CC0000', '비활동 고객'), (1, '#FF6B6B', '활동 고객')]:
                    if idx_val in active_product.index:
                        fig.add_trace(go.Bar(
                            name=label,
                            x=product_cols,
                            y=active_product.loc[idx_val].values,
                            marker_color=color_bar
                        ))
                fig.update_layout(
                    title="활동/비활동 고객 제품 선호 비교",
                    xaxis_title="제품 타입",
                    yaxis_title="거래건수",
                    barmode='group',
                    height=380,
                    template='plotly_white',
                    xaxis=dict(tickangle=30)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("활동 고객 데이터가 없습니다.")

        with col2:
            st.markdown("**뉴스레터 구독 빈도별 평균 지출 + 고객수**")
            if news_stats is not None and not news_stats.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=news_stats['fashion_news_frequency'],
                    y=news_stats['avg_revenue'],
                    name='평균 지출 (SEK)',
                    marker_color='#E50010',
                    yaxis='y1'
                ))
                fig.add_trace(go.Scatter(
                    x=news_stats['fashion_news_frequency'],
                    y=news_stats['unique_customers'],
                    name='고객수',
                    mode='lines+markers',
                    line=dict(color='#222222', width=2),
                    marker=dict(size=8),
                    yaxis='y2'
                ))
                fig.update_layout(
                    title="뉴스레터 구독 빈도별 평균 지출 및 고객수",
                    xaxis_title="구독 빈도",
                    yaxis=dict(title="평균 지출 (SEK)", side='left'),
                    yaxis2=dict(title="고객수", side='right', overlaying='y'),
                    height=380,
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("뉴스레터 데이터가 없습니다.")

        # 인사이트
        if news_stats is not None and not news_stats.empty:
            ns = news_stats.set_index('fashion_news_frequency')
            regularly_avg = ns.loc['Regularly', 'avg_revenue'] if 'Regularly' in ns.index else None
            none_avg = ns.loc['NONE', 'avg_revenue'] if 'NONE' in ns.index else None

            if regularly_avg is not None and none_avg is not None and none_avg > 0:
                ratio = regularly_avg / none_avg
                st.success(
                    f"인사이트: 뉴스레터 'Regularly' 구독자의 평균 지출({regularly_avg:.4f} SEK)은 "
                    f"'NONE' 비구독자({none_avg:.4f} SEK)의 **{ratio:.2f}배** 수준입니다."
                )
            elif regularly_avg is not None or none_avg is not None:
                avail = regularly_avg if regularly_avg is not None else none_avg
                label = 'Regularly' if regularly_avg is not None else 'NONE'
                st.info(f"인사이트: '{label}' 그룹 평균 지출 = {avail:.4f} SEK")
