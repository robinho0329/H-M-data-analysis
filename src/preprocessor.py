"""
데이터 전처리 및 시계열 생성 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesGenerator:
    """시계열 데이터 생성기"""

    def __init__(self, merged_data: pd.DataFrame, date_column: str = 't_dat'):
        """
        Args:
            merged_data: 병합된 데이터
            date_column: 날짜 컬럼명
        """
        self.data = merged_data.copy()
        self.date_column = date_column

        # 날짜 변환
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])

    def create_daily_sales(self) -> pd.DataFrame:
        """일일 매출 시계열 생성"""
        logger.info("일일 매출 시계열 생성 중...")

        daily = self.data.groupby(self.date_column).agg({
            'price': ['sum', 'count', 'mean'],
            'customer_id': 'nunique',
            'article_id': 'nunique',
        }).reset_index()

        daily.columns = ['date', 'total_sales', 'transaction_count',
                        'avg_transaction', 'unique_customers', 'unique_products']

        # 날짜 범위 확인
        date_range = pd.date_range(
            start=daily['date'].min(),
            end=daily['date'].max(),
            freq='D'
        )

        # 모든 날짜 포함 (거래 없는 날도)
        daily = daily.set_index('date').reindex(date_range).reset_index()
        daily.columns = ['date'] + daily.columns[1:].tolist()

        # 결측값 처리 (0으로)
        daily[['transaction_count', 'avg_transaction', 'unique_customers', 'unique_products']] = \
            daily[['transaction_count', 'avg_transaction', 'unique_customers', 'unique_products']].fillna(0)

        # total_sales 선형 보간
        daily['total_sales'] = daily['total_sales'].interpolate(method='linear')
        daily['total_sales'] = daily['total_sales'].fillna(0)

        logger.info(f"✓ 생성된 시계열: {len(daily)}일")
        logger.info(f"  - 기간: {daily['date'].min()} ~ {daily['date'].max()}")
        logger.info(f"  - 총 매출: {daily['total_sales'].sum():.2f}")
        logger.info(f"  - 일평균: {daily['total_sales'].mean():.4f}")

        return daily

    def create_segment_timeseries(self, segment_data: pd.DataFrame,
                                 segment_name: str) -> pd.DataFrame:
        """세그먼트별 시계열 생성"""
        daily = segment_data.groupby(self.date_column).agg({
            'price': ['sum', 'count', 'mean'],
        }).reset_index()

        daily.columns = ['date', 'sales', 'count', 'avg_price']

        # 날짜 범위 맞추기
        date_range = pd.date_range(
            start=segment_data[self.date_column].min(),
            end=segment_data[self.date_column].max(),
            freq='D'
        )

        daily = daily.set_index('date').reindex(date_range).reset_index()
        daily.columns = ['date'] + daily.columns[1:].tolist()

        # 결측값 처리
        daily[['count', 'avg_price']] = daily[['count', 'avg_price']].fillna(0)
        daily['sales'] = daily['sales'].interpolate(method='linear').fillna(0)

        daily['segment'] = segment_name

        return daily

    def create_multiple_segment_timeseries(self, segments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """여러 세그먼트 시계열 생성"""
        logger.info(f"세그먼트별 시계열 생성 중... ({len(segments)}개)")

        timeseries_list = []
        for segment_name, segment_data in segments.items():
            ts = self.create_segment_timeseries(segment_data, segment_name)
            timeseries_list.append(ts)

        result = pd.concat(timeseries_list, ignore_index=True)
        logger.info(f"✓ 생성된 세그먼트 시계열: {len(segments)}개")

        return result


class DataNormalizer:
    """데이터 정규화"""

    def __init__(self):
        self.scalers = {}

    def fit(self, data: pd.DataFrame, columns: list) -> None:
        """정규화 학습"""
        for col in columns:
            scaler = MinMaxScaler()
            scaler.fit(data[[col]])
            self.scalers[col] = scaler

        logger.info(f"✓ {len(columns)}개 컬럼 정규화 학습 완료")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """정규화 적용"""
        result = data.copy()
        for col, scaler in self.scalers.items():
            if col in result.columns:
                result[col] = scaler.transform(result[[col]])
        return result

    def inverse_transform(self, data: pd.DataFrame, column: str) -> np.ndarray:
        """역정규화"""
        if column in self.scalers:
            return self.scalers[column].inverse_transform(data[[column]])
        return data[column].values

    def save(self, path: str) -> None:
        """정규화 기준 저장"""
        with open(path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"✓ 정규화 기준 저장: {path}")

    def load(self, path: str) -> None:
        """정규화 기준 로드"""
        with open(path, 'rb') as f:
            self.scalers = pickle.load(f)
        logger.info(f"✓ 정규화 기준 로드: {path}")


class TrainValTestSplit:
    """시계열 데이터 분할"""

    def __init__(self, data: pd.DataFrame, date_column: str = 'date'):
        self.data = data
        self.date_column = date_column

    def split_by_date(self, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """날짜 기준으로 분할"""
        logger.info("Train/Val/Test 분할 중...")

        train = self.data[self.data[self.date_column] <= train_end].copy()
        val = self.data[
            (self.data[self.date_column] > train_end) &
            (self.data[self.date_column] <= val_end)
        ].copy()
        test = self.data[self.data[self.date_column] > val_end].copy()

        logger.info(f"✓ Train: {len(train)}일 ({train[self.date_column].min()} ~ {train[self.date_column].max()})")
        logger.info(f"✓ Val: {len(val)}일 ({val[self.date_column].min()} ~ {val[self.date_column].max()})")
        logger.info(f"✓ Test: {len(test)}일 ({test[self.date_column].min()} ~ {test[self.date_column].max()})")

        return train, val, test


if __name__ == "__main__":
    from config import TRANSACTIONS_PATH, CUSTOMERS_PATH, ARTICLES_PATH, TRAIN_END_DATE, VAL_END_DATE
    from data_loader import DataLoader

    # 데이터 로드
    loader = DataLoader(TRANSACTIONS_PATH, CUSTOMERS_PATH, ARTICLES_PATH)
    merged = loader.prepare_data()

    # 시계열 생성
    ts_gen = TimeSeriesGenerator(merged)
    daily_sales = ts_gen.create_daily_sales()

    # 정규화
    normalizer = DataNormalizer()
    normalizer.fit(daily_sales, ['total_sales'])
    normalized = normalizer.transform(daily_sales)

    # 분할
    splitter = TrainValTestSplit(normalized)
    train, val, test = splitter.split_by_date(TRAIN_END_DATE, VAL_END_DATE)

    print("\n정규화된 데이터 샘플:")
    print(train[['date', 'total_sales']].head())
