"""
퍼널 분석 (Funnel Analysis)
고객 라이프사이클 단계별 분석
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FunnelAnalyzer:
    """고객 퍼널 분석"""

    def __init__(self, merged_data):
        """
        Parameters
        ----------
        merged_data : pd.DataFrame
            거래, 고객, 상품 병합 데이터
        """
        self.data = merged_data.copy()
        self.data['t_dat'] = pd.to_datetime(self.data['t_dat'])

    def create_customer_lifecycle_funnel(self):
        """
        고객 라이프사이클 퍼널
        - New (신규): 첫 구매 후 30일 이내
        - Active (활성): 최근 30일 이내 구매
        - Dormant (휴면): 30~60일 구매 없음
        - Inactive (비활성): 60일 이상 구매 없음
        """
        cutoff_date = self.data['t_dat'].max()

        # 고객별 첫 구매일, 마지막 구매일
        customer_activity = self.data.groupby('customer_id').agg({
            't_dat': ['min', 'max'],
            'price': 'sum',
            'customer_id': 'size'
        }).reset_index(drop=True)

        customer_activity.columns = ['first_purchase', 'last_purchase', 'total_spent', 'purchase_count']
        customer_activity.index.name = 'customer_id'
        customer_activity = customer_activity.reset_index()

        # 라이프사이클 분류
        def classify_lifecycle(row):
            days_since_first = (row['last_purchase'] - row['first_purchase']).days
            days_since_last = (cutoff_date - row['last_purchase']).days

            if days_since_first <= 30:
                return 'New'
            elif days_since_last <= 30:
                return 'Active'
            elif days_since_last <= 60:
                return 'Dormant'
            else:
                return 'Inactive'

        customer_activity['lifecycle'] = customer_activity.apply(classify_lifecycle, axis=1)

        # 퍼널 데이터
        funnel = customer_activity.groupby('lifecycle').agg({
            'customer_id': 'count',
            'total_spent': ['sum', 'mean'],
            'purchase_count': ['mean', 'sum']
        }).round(2)

        funnel.columns = ['customer_count', 'total_revenue', 'avg_customer_value', 'total_purchases', 'avg_purchases']
        funnel = funnel.reset_index()

        # 퍼널 순서
        lifecycle_order = ['New', 'Active', 'Dormant', 'Inactive']
        funnel['lifecycle'] = pd.Categorical(funnel['lifecycle'], categories=lifecycle_order, ordered=True)
        funnel = funnel.sort_values('lifecycle')

        # 비율 계산
        total_customers = funnel['customer_count'].sum()
        funnel['percentage'] = (funnel['customer_count'] / total_customers * 100).round(2)
        funnel['conversion_rate'] = (funnel['customer_count'] / total_customers).round(4)

        return funnel, customer_activity

    def create_purchase_frequency_funnel(self):
        """
        구매 빈도 퍼널
        - One-time: 1회
        - Few: 2-5회
        - Regular: 6-20회
        - Frequent: 20회 이상
        """
        purchase_count = self.data.groupby('customer_id').size().reset_index(name='purchase_count')

        def classify_frequency(count):
            if count == 1:
                return 'One-time'
            elif count <= 5:
                return 'Few'
            elif count <= 20:
                return 'Regular'
            else:
                return 'Frequent'

        purchase_count['frequency'] = purchase_count['purchase_count'].apply(classify_frequency)

        # 고객별 총 지출액
        customer_spending = self.data.groupby('customer_id')['price'].sum().reset_index()
        customer_spending.columns = ['customer_id', 'total_spent']

        purchase_count = purchase_count.merge(customer_spending, on='customer_id')

        # 퍼널 데이터
        funnel = purchase_count.groupby('frequency').agg({
            'customer_id': 'count',
            'purchase_count': 'mean',
            'total_spent': ['sum', 'mean']
        }).round(2)

        funnel.columns = ['customer_count', 'avg_purchase_count', 'total_revenue', 'avg_customer_value']
        funnel = funnel.reset_index()

        # 퍼널 순서
        frequency_order = ['One-time', 'Few', 'Regular', 'Frequent']
        funnel['frequency'] = pd.Categorical(funnel['frequency'], categories=frequency_order, ordered=True)
        funnel = funnel.sort_values('frequency')

        # 비율 계산
        total_customers = funnel['customer_count'].sum()
        funnel['percentage'] = (funnel['customer_count'] / total_customers * 100).round(2)
        funnel['cumulative_pct'] = funnel['percentage'].cumsum().round(2)

        return funnel

    def create_product_category_funnel(self):
        """
        제품 카테고리별 퍼널
        구매한 고객의 제품 카테고리 다양성
        """
        # 고객당 구매한 고유 카테고리 수
        customer_category_diversity = self.data.groupby('customer_id').agg({
            'product_group_name': 'nunique',
            'product_type_name': 'nunique',
            'colour_group_name': 'nunique'
        }).reset_index()

        customer_category_diversity.columns = ['customer_id', 'product_groups', 'product_types', 'colors']

        # 고객별 총 지출액
        customer_spending = self.data.groupby('customer_id')['price'].sum().reset_index()
        customer_spending.columns = ['customer_id', 'total_spent']

        customer_category_diversity = customer_category_diversity.merge(customer_spending, on='customer_id')

        # 제품군 다양성 분류
        def classify_diversity(row):
            diversity_score = row['product_groups'] + row['product_types']
            if diversity_score <= 5:
                return 'Low'
            elif diversity_score <= 15:
                return 'Medium'
            else:
                return 'High'

        customer_category_diversity['diversity'] = customer_category_diversity.apply(classify_diversity, axis=1)

        # 퍼널 데이터
        funnel = customer_category_diversity.groupby('diversity').agg({
            'customer_id': 'count',
            'product_groups': 'mean',
            'product_types': 'mean',
            'total_spent': ['sum', 'mean']
        }).round(2)

        funnel.columns = ['customer_count', 'avg_product_groups', 'avg_product_types', 'total_revenue', 'avg_customer_value']
        funnel = funnel.reset_index()

        # 퍼널 순서
        diversity_order = ['Low', 'Medium', 'High']
        funnel['diversity'] = pd.Categorical(funnel['diversity'], categories=diversity_order, ordered=True)
        funnel = funnel.sort_values('diversity')

        # 비율 계산
        total_customers = funnel['customer_count'].sum()
        funnel['percentage'] = (funnel['customer_count'] / total_customers * 100).round(2)

        return funnel

    def create_sales_channel_funnel(self):
        """
        판매 채널별 퍼널
        (온라인 vs 오프라인)
        """
        channel_data = self.data.groupby('sales_channel_id').agg({
            'customer_id': 'nunique',
            'price': ['sum', 'mean'],
            't_dat': 'nunique'
        }).reset_index()

        channel_data.columns = ['sales_channel_id', 'unique_customers', 'total_revenue', 'avg_order_value', 'trading_days']

        # 채널명 (1=오프라인, 2=온라인)
        channel_names = {1: 'Offline', 2: 'Online'}
        channel_data['channel'] = channel_data['sales_channel_id'].map(channel_names)

        # 평균 거래액
        transaction_count = self.data.groupby('sales_channel_id').size().reset_index(name='transaction_count')
        channel_data = channel_data.merge(transaction_count, on='sales_channel_id')
        channel_data['avg_transaction_value'] = (channel_data['total_revenue'] / channel_data['transaction_count']).round(2)

        return channel_data[['channel', 'unique_customers', 'total_revenue', 'avg_order_value', 'avg_transaction_value', 'transaction_count']].round(2)

    def get_funnel_insights(self):
        """전체 퍼널 인사이트 생성"""
        lifecycle_funnel, customer_activity = self.create_customer_lifecycle_funnel()
        frequency_funnel = self.create_purchase_frequency_funnel()
        category_funnel = self.create_product_category_funnel()
        channel_funnel = self.create_sales_channel_funnel()

        return {
            'lifecycle': lifecycle_funnel,
            'frequency': frequency_funnel,
            'category': category_funnel,
            'channel': channel_funnel,
            'customer_activity': customer_activity
        }
