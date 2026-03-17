"""
코호트 분석 (Cohort Analysis)
고객 취득 시점 기준으로 시간 경과에 따른 행동 분석
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class CohortAnalyzer:
    """코호트 분석"""

    def __init__(self, merged_data):
        """
        Parameters
        ----------
        merged_data : pd.DataFrame
            거래, 고객, 상품 병합 데이터
        """
        self.data = merged_data.copy()
        self.data['t_dat'] = pd.to_datetime(self.data['t_dat'])

    def create_monthly_cohort_retention(self):
        """
        월별 코호트 기반 리텐션 분석
        - X축: 코호트(고객 첫 구매 월)
        - Y축: 코호트 나이(개월 경과)
        - 값: 리텐션 비율(%)
        """
        # 고객별 첫 구매 월
        customer_first_purchase = self.data.groupby('customer_id')['t_dat'].min().reset_index()
        customer_first_purchase.columns = ['customer_id', 'first_purchase_date']
        customer_first_purchase['cohort_month'] = customer_first_purchase['first_purchase_date'].dt.to_period('M')

        # 거래 데이터에 코호트 정보 추가
        data_with_cohort = self.data.merge(customer_first_purchase, on='customer_id')
        data_with_cohort['transaction_month'] = data_with_cohort['t_dat'].dt.to_period('M')

        # 코호트 나이(개월) 계산
        data_with_cohort['cohort_age'] = (data_with_cohort['transaction_month'] - data_with_cohort['cohort_month']).apply(lambda x: x.n)

        # 코호트별 월별 고객 수
        cohort_data = data_with_cohort.groupby(['cohort_month', 'cohort_age'])['customer_id'].nunique().reset_index()
        cohort_data.columns = ['cohort_month', 'cohort_age', 'customers']

        # 피벗 테이블 (행: 코호트, 열: 코호트 나이)
        cohort_pivot = cohort_data.pivot_table(index='cohort_month', columns='cohort_age', values='customers')

        # 각 코호트의 초기 고객 수로 정규화 (리텐션 비율)
        cohort_size = cohort_pivot.iloc[:, 0]
        retention_table = cohort_pivot.divide(cohort_size, axis=0) * 100
        retention_table = retention_table.round(1)

        return retention_table

    def create_monthly_cohort_revenue(self):
        """
        월별 코호트 기반 매출 분석
        """
        # 고객별 첫 구매 월
        customer_first_purchase = self.data.groupby('customer_id')['t_dat'].min().reset_index()
        customer_first_purchase.columns = ['customer_id', 'first_purchase_date']
        customer_first_purchase['cohort_month'] = customer_first_purchase['first_purchase_date'].dt.to_period('M')

        # 거래 데이터에 코호트 정보 추가
        data_with_cohort = self.data.merge(customer_first_purchase, on='customer_id')
        data_with_cohort['transaction_month'] = data_with_cohort['t_dat'].dt.to_period('M')

        # 코호트 나이(개월) 계산
        data_with_cohort['cohort_age'] = (data_with_cohort['transaction_month'] - data_with_cohort['cohort_month']).apply(lambda x: x.n)

        # 코호트별 월별 매출
        cohort_data = data_with_cohort.groupby(['cohort_month', 'cohort_age'])['price'].sum().reset_index()
        cohort_data.columns = ['cohort_month', 'cohort_age', 'revenue']

        # 피벗 테이블
        cohort_pivot = cohort_data.pivot_table(index='cohort_month', columns='cohort_age', values='revenue')
        cohort_pivot = cohort_pivot.round(0)

        return cohort_pivot

    def create_monthly_cohort_avg_spending(self):
        """
        월별 코호트 기반 고객당 평균 지출액
        """
        # 고객별 첫 구매 월
        customer_first_purchase = self.data.groupby('customer_id')['t_dat'].min().reset_index()
        customer_first_purchase.columns = ['customer_id', 'first_purchase_date']
        customer_first_purchase['cohort_month'] = customer_first_purchase['first_purchase_date'].dt.to_period('M')

        # 거래 데이터에 코호트 정보 추가
        data_with_cohort = self.data.merge(customer_first_purchase, on='customer_id')
        data_with_cohort['transaction_month'] = data_with_cohort['t_dat'].dt.to_period('M')

        # 코호트 나이(개월) 계산
        data_with_cohort['cohort_age'] = (data_with_cohort['transaction_month'] - data_with_cohort['cohort_month']).apply(lambda x: x.n)

        # 코호트별 월별 평균 지출액
        cohort_data = data_with_cohort.groupby(['cohort_month', 'cohort_age']).agg({
            'price': 'sum',
            'customer_id': 'nunique'
        }).reset_index()

        cohort_data['avg_spending'] = (cohort_data['price'] / cohort_data['customer_id']).round(2)
        cohort_data = cohort_data[['cohort_month', 'cohort_age', 'avg_spending']]

        # 피벗 테이블
        cohort_pivot = cohort_data.pivot_table(index='cohort_month', columns='cohort_age', values='avg_spending')
        cohort_pivot = cohort_pivot.round(2)

        return cohort_pivot

    def create_age_group_cohort(self):
        """
        나이대별 코호트 분석
        같은 나이대 고객들의 월별 구매 행동
        """
        # 나이대 분류
        self.data['age_group'] = pd.cut(self.data['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 100],
                                        labels=['10s', '20s', '30s', '40s', '50s', '60s', '70s', '80+'])

        # 월별 데이터
        self.data['transaction_month'] = self.data['t_dat'].dt.to_period('M')

        # 나이대별 월별 구매 분석
        age_cohort = self.data.groupby(['age_group', 'transaction_month']).agg({
            'customer_id': 'nunique',
            'price': ['sum', 'mean'],
            't_dat': 'count'
        }).reset_index()

        age_cohort.columns = ['age_group', 'month', 'unique_customers', 'total_revenue', 'avg_order_value', 'transaction_count']
        age_cohort['avg_spending_per_customer'] = (age_cohort['total_revenue'] / age_cohort['unique_customers']).round(2)

        return age_cohort

    def create_club_status_cohort(self):
        """
        클럽 상태별 코호트 분석
        """
        # 월별 데이터
        self.data['transaction_month'] = self.data['t_dat'].dt.to_period('M')

        # 클럽 상태별 월별 구매 분석
        club_cohort = self.data.groupby(['club_member_status', 'transaction_month']).agg({
            'customer_id': 'nunique',
            'price': ['sum', 'mean'],
            't_dat': 'count'
        }).reset_index()

        club_cohort.columns = ['club_status', 'month', 'unique_customers', 'total_revenue', 'avg_order_value', 'transaction_count']
        club_cohort['avg_spending_per_customer'] = (club_cohort['total_revenue'] / club_cohort['unique_customers']).round(2)

        return club_cohort

    def create_newsletter_cohort(self):
        """
        뉴스레터 구독 상태별 코호트 분석
        """
        # 월별 데이터
        self.data['transaction_month'] = self.data['t_dat'].dt.to_period('M')

        # 뉴스레터 상태별 월별 구매 분석
        newsletter_cohort = self.data.groupby(['fashion_news_frequency', 'transaction_month']).agg({
            'customer_id': 'nunique',
            'price': ['sum', 'mean'],
            't_dat': 'count'
        }).reset_index()

        newsletter_cohort.columns = ['newsletter_freq', 'month', 'unique_customers', 'total_revenue', 'avg_order_value', 'transaction_count']
        newsletter_cohort['avg_spending_per_customer'] = (newsletter_cohort['total_revenue'] / newsletter_cohort['unique_customers']).round(2)

        return newsletter_cohort

    def get_cohort_insights(self):
        """전체 코호트 인사이트 생성"""
        return {
            'retention': self.create_monthly_cohort_retention(),
            'revenue': self.create_monthly_cohort_revenue(),
            'avg_spending': self.create_monthly_cohort_avg_spending(),
            'age_group': self.create_age_group_cohort(),
            'club_status': self.create_club_status_cohort(),
            'newsletter': self.create_newsletter_cohort()
        }
