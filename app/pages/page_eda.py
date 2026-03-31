"""
데이터 탐색 (EDA) 페이지 - 실데이터 기반
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
def load_eda_data():
    """EDA 페이지용 실데이터 로드 및 집계"""
    from app.utils.cache_manager import load_raw_data
    merged = load_raw_data()
    if merged is None:
        return None

    # 고객 정보 (unique by customer_id)
    cust_cols = ['customer_id']
    for c in ['age', 'club_member_status', 'fashion_news_frequency', 'Active']:
        if c in merged.columns:
            cust_cols.append(c)
    customers = merged[cust_cols].drop_duplicates('customer_id')

    # 일별 집계
    daily = merged.groupby(merged['t_dat'].dt.date).agg(
        total_revenue=('price', 'sum'),
        transaction_count=('price', 'count'),
        unique_customers=('customer_id', 'nunique'),
        unique_articles=('article_id', 'nunique'),
        avg_price=('price', 'mean'),
    ).reset_index()
    daily.rename(columns={daily.columns[0]: 'date'}, inplace=True)
    daily['date'] = pd.to_datetime(daily['date'])

    # 색상별 판매량 Top 15
    color_stats = merged.groupby('colour_group_name')['price'].agg(
        count='count', revenue='sum'
    ).reset_index().sort_values('count', ascending=False).head(15)

    # 제품 타입별 판매량 Top 15
    product_stats = merged.groupby('product_type_name')['price'].agg(
        count='count', revenue='sum'
    ).reset_index().sort_values('count', ascending=False).head(15)

    # 의류 그룹별
    garment_stats = merged.groupby('garment_group_name')['price'].agg(
        count='count', revenue='sum'
    ).reset_index().sort_values('count', ascending=False)

    # 가격 분포 샘플
    price_sample = merged['price'].sample(min(5000, len(merged)), random_state=42).values

    # 실제 통계
    stats = {
        'total_transactions': len(merged),
        'total_revenue': float(merged['price'].sum()),
        'unique_customers': int(merged['customer_id'].nunique()),
        'unique_articles': int(merged['article_id'].nunique()),
        'date_min': merged['t_dat'].min().strftime('%Y-%m-%d'),
        'date_max': merged['t_dat'].max().strftime('%Y-%m-%d'),
        'avg_price': float(merged['price'].mean()),
        'median_price': float(merged['price'].median()),
        'min_price': float(merged['price'].min()),
        'max_price': float(merged['price'].max()),
        'missing_values': int(merged.isnull().sum().sum()),
        'outliers': int(((merged['price'] - merged['price'].mean()).abs() > 3 * merged['price'].std()).sum()),
    }

    return {
        'customers': customers,
        'daily': daily,
        'color_stats': color_stats,
        'product_stats': product_stats,
        'garment_stats': garment_stats,
        'price_sample': price_sample,
        'stats': stats,
    }


def show():
    """페이지 표시"""
    st.markdown("""
<div style='margin-bottom:24px;'>
    <div style='font-size:11px; color:#767676; margin-bottom:12px; letter-spacing:0.3px;'>
        홈 / 분석 / <span style='color:#000000; font-weight:600;'>데이터 탐색</span>
    </div>
    <h1 style='font-family:Helvetica Neue,Helvetica,Arial,sans-serif;
               font-size:1.8rem; font-weight:700; color:#000000;
               margin:0 0 4px 0; letter-spacing:0.5px; text-transform:uppercase;'>
        데이터 탐색
    </h1>
    <div style='height:2px; background:#000000; width:40px; margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)

    data = load_eda_data()
    if data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    customers = data['customers']
    daily = data['daily']
    color_stats = data['color_stats']
    product_stats = data['product_stats']
    garment_stats = data['garment_stats']
    price_sample = data['price_sample']
    stats = data['stats']

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["👥 고객 분석", "🛍️ 상품 분석", "📊 시계열", "🔗 상관관계", "📈 요약 통계"]
    )

    # ==================== Tab1: 고객 분석 ====================
    with tab1:
        st.subheader("👥 고객 데이터 분석")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**연령대 분포**")
            if 'age' in customers.columns:
                age_data = customers['age'].dropna()
                fig = go.Figure(data=[go.Histogram(
                    x=age_data, nbinsx=35,
                    marker_color='#E50010', opacity=0.85
                )])
                fig.update_layout(
                    title=f"고객 연령 분포 (n={len(age_data):,}명)",
                    xaxis_title="연령", yaxis_title="고객 수",
                    height=360, template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

                age_bins = pd.cut(age_data,
                                  bins=[0, 20, 30, 40, 50, 60, 70, 120],
                                  labels=['10대', '20대', '30대', '40대', '50대', '60대', '70대+'])
                age_dist = age_bins.value_counts().sort_index()
                dominant = age_dist.idxmax()
                st.caption(f"최다 연령대: **{dominant}** ({age_dist.max():,}명, {age_dist.max()/len(age_data)*100:.1f}%)")
            else:
                st.info("연령 데이터 없음")

        with col2:
            st.markdown("**활동도별 고객 구성**")
            if 'Active' in customers.columns:
                active_counts = customers['Active'].value_counts()
                labels_map = {1: '활동', 0: '비활동'}
                labels = [labels_map.get(k, str(k)) for k in active_counts.index]
                fig = go.Figure(data=[go.Pie(
                    labels=labels, values=active_counts.values,
                    marker=dict(colors=['#FF6B6B', '#CC0000']),
                    hole=0.35
                )])
                fig.update_layout(title="활동도별 고객 비중", height=360)
                st.plotly_chart(fig, use_container_width=True)
                active_n = int(active_counts.get(1, 0))
                total_n = len(customers)
                st.caption(f"활동 고객: **{active_n:,}명** ({active_n/total_n*100:.1f}%)")
            else:
                st.info("활동도 데이터 없음")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**클럽 회원 상태**")
            if 'club_member_status' in customers.columns:
                club_counts = customers['club_member_status'].value_counts()
                colors_list = ['#E50010', '#222222', '#CC0000']
                fig = go.Figure(data=[go.Bar(
                    x=club_counts.index.tolist(),
                    y=club_counts.values,
                    marker_color=colors_list[:len(club_counts)],
                    text=club_counts.values,
                    textposition='outside'
                )])
                fig.update_layout(
                    title="클럽 회원 상태 분포",
                    xaxis_title="상태", yaxis_title="고객 수",
                    height=360, template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("클럽 상태 데이터 없음")

        with col2:
            st.markdown("**뉴스레터 구독 빈도**")
            if 'fashion_news_frequency' in customers.columns:
                news_counts = customers['fashion_news_frequency'].value_counts()
                color_map = {'NONE': '#CC0000', 'Monthly': '#222222', 'Regularly': '#FF6B6B'}
                bar_colors = [color_map.get(str(k), '#E50010') for k in news_counts.index]
                fig = go.Figure(data=[go.Bar(
                    x=news_counts.index.tolist(),
                    y=news_counts.values,
                    marker_color=bar_colors,
                    text=news_counts.values,
                    textposition='outside'
                )])
                fig.update_layout(
                    title="뉴스레터 구독 분포",
                    xaxis_title="구독 빈도", yaxis_title="고객 수",
                    height=360, template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("뉴스레터 데이터 없음")

    # ==================== Tab2: 상품 분석 ====================
    with tab2:
        st.subheader("🛍️ 상품 데이터 분석")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 15 색상별 판매량**")
            fig = go.Figure(data=[go.Bar(
                x=color_stats['colour_group_name'],
                y=color_stats['count'],
                marker_color='#E50010',
                text=color_stats['count'].apply(lambda x: f"{x:,}"),
                textposition='outside'
            )])
            fig.update_layout(
                title="색상별 판매 건수 (Top 15)",
                xaxis_title="색상", yaxis_title="판매량",
                height=380, template='plotly_white',
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**의류 그룹 구성**")
            fig = go.Figure(data=[go.Pie(
                labels=garment_stats['garment_group_name'],
                values=garment_stats['count'],
                hole=0.35
            )])
            fig.update_layout(
                title="의류 그룹별 판매 비중",
                height=380,
                legend=dict(orientation='h', yanchor='bottom', y=-0.45)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Top 15 제품 타입별 판매량**")
        fig = go.Figure(data=[go.Bar(
            x=product_stats['product_type_name'],
            y=product_stats['count'],
            marker_color='#222222',
            text=product_stats['count'].apply(lambda x: f"{x:,}"),
            textposition='outside'
        )])
        fig.update_layout(
            title="제품 타입별 판매 건수 (Top 15)",
            xaxis_title="제품 타입", yaxis_title="판매량",
            height=380, template='plotly_white',
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**가격 분포**")
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[go.Histogram(
                x=price_sample, nbinsx=50,
                marker_color='#888888', opacity=0.85
            )])
            fig.update_layout(
                title=f"가격 분포 (샘플 {min(5000, len(price_sample)):,}건)",
                xaxis_title="가격 (₩)", yaxis_title="빈도",
                height=320, template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**가격 통계 요약**")
            price_summary = pd.DataFrame({
                '통계': ['최솟값', '25%', '중앙값', '평균', '75%', '최댓값'],
                '가격 (₩)': [
                    f"₩{stats['min_price']:.6f}",
                    f"₩{float(np.percentile(price_sample, 25)):.4f}",
                    f"₩{stats['median_price']:.4f}",
                    f"₩{stats['avg_price']:.4f}",
                    f"₩{float(np.percentile(price_sample, 75)):.4f}",
                    f"₩{stats['max_price']:.2f}",
                ]
            })
            st.dataframe(price_summary, use_container_width=True, hide_index=True)
            st.metric("이상치 (3σ 초과)", f"{stats['outliers']:,}건",
                      f"전체의 {stats['outliers']/stats['total_transactions']*100:.3f}%")

    # ==================== Tab3: 시계열 ====================
    with tab3:
        st.subheader("📊 시계열 특성")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**일별 매출 추이 (7일 이동평균 포함)**")
            ma7 = daily['total_revenue'].rolling(7, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily['date'], y=daily['total_revenue'],
                name='일별 매출', mode='lines',
                fill='tozeroy',
                line=dict(color='#E50010', width=1.2),
                fillcolor='rgba(229, 0, 16, 0.10)'
            ))
            fig.add_trace(go.Scatter(
                x=daily['date'], y=ma7,
                name='7일 이동평균', mode='lines',
                line=dict(color='#222222', width=2.5)
            ))
            fig.update_layout(
                title="2019년 일별 매출",
                xaxis_title="날짜", yaxis_title="매출 (₩)",
                height=380, template='plotly_white',
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**일별 거래 건수**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily['date'], y=daily['transaction_count'],
                name='거래건수', mode='lines',
                line=dict(color='#222222', width=1.5)
            ))
            ma7_cnt = daily['transaction_count'].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=daily['date'], y=ma7_cnt,
                name='7일 이동평균', mode='lines',
                line=dict(color='#FF6B6B', width=2.5)
            ))
            fig.update_layout(
                title="2019년 일별 거래 건수",
                xaxis_title="날짜", yaxis_title="거래 건수",
                height=380, template='plotly_white',
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**월별 매출 추이**")
            daily_m = daily.copy()
            daily_m['month'] = daily_m['date'].dt.to_period('M').astype(str)
            monthly_agg = daily_m.groupby('month')['total_revenue'].sum().reset_index()
            fig = go.Figure(data=[go.Bar(
                x=monthly_agg['month'],
                y=monthly_agg['total_revenue'],
                marker_color='#FF6B6B',
                text=monthly_agg['total_revenue'].apply(lambda x: f"₩{x:,.0f}"),
                textposition='outside', textfont=dict(size=9)
            )])
            fig.update_layout(
                title="월별 총 매출",
                xaxis_title="월", yaxis_title="매출 (₩)",
                height=360, template='plotly_white',
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**요일별 평균 매출**")
            daily_d = daily.copy()
            daily_d['day_of_week'] = daily_d['date'].dt.dayofweek
            dow_stats = daily_d.groupby('day_of_week')['total_revenue'].mean()
            day_names = ['월', '화', '수', '목', '금', '토', '일']
            colors_dow = ['#E50010' if d < 5 else '#222222' for d in range(7)]
            fig = go.Figure(data=[go.Bar(
                x=day_names,
                y=[dow_stats.get(d, 0) for d in range(7)],
                marker_color=colors_dow,
                text=[f"₩{dow_stats.get(d, 0):,.2f}" for d in range(7)],
                textposition='outside', textfont=dict(size=9)
            )])
            fig.update_layout(
                title="요일별 평균 매출 (빨강=평일, 검정=주말)",
                xaxis_title="요일", yaxis_title="평균 매출 (₩)",
                height=360, template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

    # ==================== Tab4: 상관관계 ====================
    with tab4:
        st.subheader("🔗 변수 간 상관관계")

        corr_features = ['total_revenue', 'transaction_count', 'unique_customers',
                         'unique_articles', 'avg_price']
        corr_labels = ['매출액', '거래건수', '고객수', '상품수', '평균가격']

        corr_matrix = daily[corr_features].corr().values

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_labels,
            y=corr_labels,
            colorscale='RdGy',
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 13}
        ))
        fig.update_layout(
            title="일별 집계 변수 간 피어슨 상관계수 (실데이터)",
            height=440
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**주요 상관관계 해석**")

        pairs = []
        for i in range(len(corr_labels)):
            for j in range(i + 1, len(corr_labels)):
                pairs.append((corr_labels[i], corr_labels[j], corr_matrix[i][j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        for a, b, r in pairs[:6]:
            direction = "양의" if r > 0 else "음의"
            strength = "강한" if abs(r) >= 0.7 else ("중간" if abs(r) >= 0.4 else "약한")
            emoji = "🔴" if abs(r) >= 0.7 else ("🟡" if abs(r) >= 0.4 else "🟢")
            st.write(f"{emoji} **{a} ↔ {b}**: {r:.3f} ({strength} {direction} 상관)")

    # ==================== Tab5: 요약 통계 ====================
    with tab5:
        st.subheader("📈 요약 통계")

        summary_data = {
            "통계": [
                "총 거래 건수", "총 거래액", "고객 수", "상품 수",
                "기간", "일평균 거래 건수", "일평균 매출",
                "평균 거래액", "중앙값 거래액", "거래액 범위",
            ],
            "값": [
                f"{stats['total_transactions']:,}건",
                f"₩{stats['total_revenue']:,.2f}",
                f"{stats['unique_customers']:,}명",
                f"{stats['unique_articles']:,}개",
                f"{stats['date_min']} ~ {stats['date_max']} (365일)",
                f"{stats['total_transactions'] / 365:.0f}건/일",
                f"₩{stats['total_revenue'] / 365:.2f}/일",
                f"₩{stats['avg_price']:.4f}",
                f"₩{stats['median_price']:.4f}",
                f"₩{stats['min_price']:.6f} ~ ₩{stats['max_price']:.2f}",
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("결측값", f"{stats['missing_values']:,}개",
                      "데이터 품질 우수" if stats['missing_values'] == 0 else "처리 필요")
        with col2:
            st.metric("가격 이상치 (3σ)", f"{stats['outliers']:,}건",
                      f"전체의 {stats['outliers'] / stats['total_transactions'] * 100:.3f}%")
        with col3:
            st.metric("데이터 커버리지", "365일", "2019년 전체")

        st.markdown("---")
        st.markdown("**월별 거래 통계**")
        daily_m2 = daily.copy()
        daily_m2['month'] = daily_m2['date'].dt.strftime('%Y-%m')
        monthly_table = daily_m2.groupby('month').agg(
            거래건수=('transaction_count', 'sum'),
            총매출=('total_revenue', 'sum'),
            일평균매출=('total_revenue', 'mean'),
        ).reset_index()
        monthly_table.columns = ['월', '거래건수', '총매출(₩)', '일평균매출(₩)']
        monthly_table['총매출(₩)'] = monthly_table['총매출(₩)'].map(lambda x: f"₩{x:,.2f}")
        monthly_table['일평균매출(₩)'] = monthly_table['일평균매출(₩)'].map(lambda x: f"₩{x:,.2f}")
        monthly_table['거래건수'] = monthly_table['거래건수'].map(lambda x: f"{x:,}건")
        st.dataframe(monthly_table, use_container_width=True, hide_index=True)
