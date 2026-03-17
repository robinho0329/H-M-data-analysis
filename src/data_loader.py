"""
데이터 로드 및 병합 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """H&M 거래 데이터 로더"""

    def __init__(self, transactions_path: str, customers_path: str, articles_path: str):
        """
        Args:
            transactions_path: 거래 데이터 경로
            customers_path: 고객 데이터 경로
            articles_path: 상품 데이터 경로
        """
        self.transactions_path = transactions_path
        self.customers_path = customers_path
        self.articles_path = articles_path

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """원본 데이터 로드"""
        logger.info("데이터 로딩 중...")

        transactions = pd.read_csv(self.transactions_path)
        customers = pd.read_csv(self.customers_path)
        articles = pd.read_csv(self.articles_path)

        logger.info(f"✓ Transactions: {transactions.shape}")
        logger.info(f"✓ Customers: {customers.shape}")
        logger.info(f"✓ Articles: {articles.shape}")

        return transactions, customers, articles

    def merge_data(self,
                   transactions: pd.DataFrame,
                   customers: pd.DataFrame,
                   articles: pd.DataFrame) -> pd.DataFrame:
        """데이터 병합"""
        logger.info("데이터 병합 중...")

        # 트랜잭션과 고객 정보 병합
        merged = transactions.merge(
            customers,
            on='customer_id',
            how='left'
        )

        # 상품 정보 병합
        merged = merged.merge(
            articles,
            on='article_id',
            how='left'
        )

        logger.info(f"✓ Merged data shape: {merged.shape}")

        return merged

    def prepare_data(self) -> pd.DataFrame:
        """전체 데이터 준비"""
        transactions, customers, articles = self.load_raw_data()
        merged = self.merge_data(transactions, customers, articles)

        # 날짜 변환
        merged['t_dat'] = pd.to_datetime(merged['t_dat'])

        # 기본 정보 출력
        logger.info(f"\n데이터 정보:")
        logger.info(f"- 기간: {merged['t_dat'].min()} ~ {merged['t_dat'].max()}")
        logger.info(f"- 총 거래 건수: {len(merged):,}")
        logger.info(f"- 고객 수: {merged['customer_id'].nunique():,}")
        logger.info(f"- 상품 수: {merged['article_id'].nunique():,}")
        logger.info(f"- 결측값: {merged.isnull().sum().sum()}")

        return merged


class SegmentAnalyzer:
    """세그먼트 분석 클래스"""

    def __init__(self, merged_data: pd.DataFrame):
        self.data = merged_data

    def create_customer_segments(self) -> Dict[str, pd.DataFrame]:
        """고객 세그먼트 생성"""
        segments = {}

        # 활동도
        segments['active'] = self.data[self.data['Active'] == 1]
        segments['inactive'] = self.data[self.data['Active'] == 0]

        # 연령대
        age_groups = {
            '10s': (10, 19),
            '20s': (20, 29),
            '30s': (30, 39),
            '40s': (40, 49),
            '50s': (50, 59),
            '60s': (60, 69),
            '70+': (70, 100),
        }
        for name, (low, high) in age_groups.items():
            segments[f'age_{name}'] = self.data[
                (self.data['age'] >= low) & (self.data['age'] <= high)
            ]

        # 클럽 상태
        for status in self.data['club_member_status'].unique():
            if pd.notna(status):
                segments[f'club_{status.replace(" ", "_")}'] = self.data[
                    self.data['club_member_status'] == status
                ]

        # 뉴스레터
        for freq in self.data['fashion_news_frequency'].unique():
            if pd.notna(freq):
                segments[f'news_{freq.replace(" ", "_")}'] = self.data[
                    self.data['fashion_news_frequency'] == freq
                ]

        logger.info(f"생성된 고객 세그먼트: {len(segments)}개")

        return segments

    def get_top_colors(self, n: int = 15) -> list:
        """인기 색상 Top N"""
        return self.data['colour_group_name'].value_counts().head(n).index.tolist()

    def get_top_product_types(self, n: int = 15) -> list:
        """인기 제품 타입 Top N"""
        return self.data['product_type_name'].value_counts().head(n).index.tolist()

    def get_garment_groups(self) -> list:
        """모든 의류 그룹"""
        return self.data['garment_group_name'].unique().tolist()

    def get_segment_stats(self) -> Dict:
        """세그먼트별 통계"""
        stats = {
            'total_sales': self.data['price'].sum(),
            'total_transactions': len(self.data),
            'avg_transaction': self.data['price'].mean(),
            'unique_customers': self.data['customer_id'].nunique(),
            'unique_products': self.data['article_id'].nunique(),
        }
        return stats


if __name__ == "__main__":
    from config import TRANSACTIONS_PATH, CUSTOMERS_PATH, ARTICLES_PATH

    # 테스트
    loader = DataLoader(TRANSACTIONS_PATH, CUSTOMERS_PATH, ARTICLES_PATH)
    merged = loader.prepare_data()

    analyzer = SegmentAnalyzer(merged)
    print("\n세그먼트 분석:")
    print(f"- 인기 색상: {analyzer.get_top_colors(5)}")
    print(f"- 인기 제품: {analyzer.get_top_product_types(5)}")
    print(f"- 의류 그룹 수: {len(analyzer.get_garment_groups())}")
    print(f"\n통계:\n{analyzer.get_segment_stats()}")
