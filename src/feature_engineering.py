"""
파생변수 생성 모듈
"""
import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """파생변수 생성"""

    def __init__(self, data: pd.DataFrame, date_column: str = 'date',
                 target_column: str = 'total_sales'):
        """
        Args:
            data: 시계열 데이터
            date_column: 날짜 컬럼명
            target_column: 타겟 컬럼명
        """
        self.data = data.copy()
        self.date_column = date_column
        self.target_column = target_column

    def add_time_features(self) -> pd.DataFrame:
        """시간 특성 추가"""
        logger.info("시간 특성 추가 중...")

        # 날짜 변환
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])

        # 시간 특성
        self.data['day_of_week'] = self.data[self.date_column].dt.dayofweek  # 0-6
        self.data['day_of_month'] = self.data[self.date_column].dt.day
        self.data['month'] = self.data[self.date_column].dt.month
        self.data['quarter'] = self.data[self.date_column].dt.quarter
        self.data['week_of_year'] = self.data[self.date_column].dt.isocalendar().week
        self.data['year'] = self.data[self.date_column].dt.year

        # 파생 특성
        self.data['is_weekend'] = (self.data['day_of_week'] >= 5).astype(int)
        self.data['is_month_start'] = (self.data['day_of_month'] <= 7).astype(int)
        self.data['is_month_end'] = (self.data['day_of_month'] >= 24).astype(int)

        logger.info("✓ 시간 특성 추가 완료")
        return self.data

    def add_lag_features(self, lags: List[int]) -> pd.DataFrame:
        """래그 특성 추가"""
        logger.info(f"래그 특성 추가 중... (lag: {lags})")

        for lag in lags:
            self.data[f'{self.target_column}_lag_{lag}'] = \
                self.data[self.target_column].shift(lag)

        # 결측값 처리
        self.data = self.data.dropna()

        logger.info(f"✓ {len(lags)}개 래그 특성 추가 완료")
        return self.data

    def add_rolling_features(self, windows: List[int]) -> pd.DataFrame:
        """이동평균 및 변동성 특성 추가"""
        logger.info(f"이동평균 특성 추가 중... (windows: {windows})")

        for window in windows:
            # 이동평균
            self.data[f'{self.target_column}_ma_{window}'] = \
                self.data[self.target_column].rolling(window=window, min_periods=1).mean()

            # 이동 표준편차 (변동성)
            self.data[f'{self.target_column}_std_{window}'] = \
                self.data[self.target_column].rolling(window=window, min_periods=1).std()

            # 계절성 지수
            self.data[f'{self.target_column}_seasonal_{window}'] = \
                self.data[self.target_column] / (self.data[f'{self.target_column}_ma_{window}'] + 1e-8)

        logger.info(f"✓ {len(windows)}개 이동평균 특성 추가 완료")
        return self.data

    def add_growth_features(self) -> pd.DataFrame:
        """성장률 특성 추가"""
        logger.info("성장률 특성 추가 중...")

        # 주간 성장률 (WoW)
        self.data[f'{self.target_column}_wow'] = \
            (self.data[self.target_column] - self.data[self.target_column].shift(7)) / \
            (self.data[self.target_column].shift(7) + 1e-8)

        # 월간 성장률 (MoM)
        self.data[f'{self.target_column}_mom'] = \
            (self.data[self.target_column] - self.data[self.target_column].shift(30)) / \
            (self.data[self.target_column].shift(30) + 1e-8)

        # 로그 수익률 (Log Return)
        self.data[f'{self.target_column}_log_return'] = \
            np.log(self.data[self.target_column] + 1e-8) - \
            np.log(self.data[self.target_column].shift(1) + 1e-8)

        # 결측값 처리
        self.data = self.data.fillna(0)

        logger.info("✓ 성장률 특성 추가 완료")
        return self.data

    def add_trend_features(self, window: int = 7) -> pd.DataFrame:
        """추세 특성 추가"""
        logger.info(f"추세 특성 추가 중... (window: {window})")

        # 기울기 (최근 window일의 선형 회귀 기울기)
        slopes = []
        for i in range(len(self.data)):
            if i < window:
                slopes.append(0)
            else:
                x = np.arange(window)
                y = self.data[self.target_column].iloc[i-window:i].values
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)

        self.data[f'{self.target_column}_trend'] = slopes

        logger.info("✓ 추세 특성 추가 완료")
        return self.data

    def add_fourier_features(self, periods: List[int] = None) -> pd.DataFrame:
        """Fourier 계절성 피처 추가"""
        if periods is None:
            periods = [7, 30, 365]

        logger.info(f"Fourier 계절성 피처 추가 중... (periods: {periods})")

        # 날짜 변환
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])

        # 연중 일수 (1월 1일로부터 경과 일수)
        day_of_year = self.data[self.date_column].dt.dayofyear

        for period in periods:
            # sin 변환: 주기적 패턴의 사인 성분
            self.data[f'fourier_sin_{period}'] = np.sin(2 * np.pi * day_of_year / period)
            # cos 변환: 주기적 패턴의 코사인 성분
            self.data[f'fourier_cos_{period}'] = np.cos(2 * np.pi * day_of_year / period)

        logger.info(f"✓ Fourier 피처 {len(periods) * 2}개 추가 완료 (sin/cos 각 {len(periods)}개)")
        return self.data

    def add_event_features(self) -> pd.DataFrame:
        """H&M 주요 이벤트/세일 피처 추가"""
        logger.info("H&M 이벤트 피처 추가 중...")

        # 날짜 변환
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])

        # H&M 주요 이벤트 목록 (2019 기준, 월-일 형식)
        event_dates_mmdd = [
            (1, 1),    # 신정
            (2, 14),   # 발렌타인
            (3, 1),    # 봄세일시작
            (4, 20),   # 부활절
            (5, 1),    # 어린이날
            (6, 1),    # 여름세일시작
            (11, 29),  # 블랙프라이데이
            (12, 13),  # 크리스마스세일
            (12, 25),  # 크리스마스
        ]

        dates = self.data[self.date_column]

        # 각 날짜에 대해 가장 가까운 이벤트까지의 절대 일수 계산
        def calc_min_days_to_event(date):
            """해당 날짜에서 가장 가까운 이벤트까지의 절대 일수 반환"""
            year = date.year
            min_days = float('inf')
            for month, day in event_dates_mmdd:
                # 해당 연도, 전년도, 다음 연도의 이벤트 날짜 모두 고려
                for y in [year - 1, year, year + 1]:
                    try:
                        event_date = pd.Timestamp(year=y, month=month, day=day)
                        diff = abs((date - event_date).days)
                        if diff < min_days:
                            min_days = diff
                    except Exception:
                        pass
            return min_days

        # 가장 가까운 이벤트까지의 절대 일수
        self.data['days_to_nearest_event'] = dates.apply(calc_min_days_to_event)

        # 어떤 이벤트로부터 ±7일 이내면 is_event_week = 1
        self.data['is_event_week'] = (self.data['days_to_nearest_event'] <= 7).astype(int)

        logger.info(f"✓ H&M 이벤트 피처 추가 완료 (이벤트 수: {len(event_dates_mmdd)}개)")
        return self.data

    def add_season_features(self) -> pd.DataFrame:
        """계절 피처 추가 (봄/여름/가을/겨울 원-핫 인코딩 + 숫자형)"""
        logger.info("계절 피처 추가 중...")

        # 날짜 변환
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])

        month = self.data[self.date_column].dt.month

        # 원-핫 인코딩: 봄(3-5월), 여름(6-8월), 가을(9-11월), 겨울(12-2월)
        self.data['is_spring'] = month.isin([3, 4, 5]).astype(int)
        self.data['is_summer'] = month.isin([6, 7, 8]).astype(int)
        self.data['is_fall']   = month.isin([9, 10, 11]).astype(int)
        self.data['is_winter'] = month.isin([12, 1, 2]).astype(int)

        # 숫자형 계절: 1(봄), 2(여름), 3(가을), 4(겨울)
        season_num = np.select(
            [
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
                month.isin([12, 1, 2]),
            ],
            [1, 2, 3, 4],
            default=0
        )
        self.data['season_num'] = season_num

        logger.info("✓ 계절 피처 추가 완료 (is_spring, is_summer, is_fall, is_winter, season_num)")
        return self.data

    def create_engineered_features(self, lags: List[int] = None,
                                  windows: List[int] = None) -> pd.DataFrame:
        """모든 파생변수 한번에 생성"""
        if lags is None:
            lags = [1, 2, 3, 7, 14, 30]
        if windows is None:
            windows = [7, 14, 30]

        logger.info("\n" + "="*60)
        logger.info("파생변수 생성 시작")
        logger.info("="*60)

        self.add_time_features()
        self.add_lag_features(lags)
        self.add_rolling_features(windows)
        self.add_growth_features()
        self.add_trend_features()
        self.add_fourier_features()
        self.add_event_features()
        self.add_season_features()

        logger.info("\n생성된 특성:")
        feature_cols = [col for col in self.data.columns if col not in [self.date_column, self.target_column]]
        logger.info(f"✓ 총 {len(feature_cols)}개 특성")
        logger.info(f"  - 시간 특성: day_of_week, month, quarter 등")
        logger.info(f"  - 래그 특성: {len(lags)}개")
        logger.info(f"  - 이동평균: {len(windows)*3}개 (MA + STD + Seasonal)")
        logger.info(f"  - 성장률: WoW, MoM, Log Return")
        logger.info(f"  - 추세: Trend")
        logger.info(f"  - Fourier 계절성: sin/cos 각 3개 (period=7, 30, 365)")
        logger.info(f"  - H&M 이벤트: days_to_nearest_event, is_event_week")
        logger.info(f"  - 계절: is_spring, is_summer, is_fall, is_winter, season_num")

        return self.data

    def get_feature_columns(self) -> List[str]:
        """특성 컬럼명 반환"""
        exclude = [self.date_column, self.target_column, 'segment']
        return [col for col in self.data.columns if col not in exclude]

    def get_engineered_data(self) -> pd.DataFrame:
        """최종 데이터 반환"""
        return self.data


if __name__ == "__main__":
    from config import TRANSACTIONS_PATH, CUSTOMERS_PATH, ARTICLES_PATH, TRAIN_END_DATE, VAL_END_DATE
    from data_loader import DataLoader
    from preprocessor import TimeSeriesGenerator, DataNormalizer, TrainValTestSplit

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

    # 특성 공학
    fe = FeatureEngineer(normalized)
    engineered = fe.create_engineered_features()

    print("\n엔지니어링된 데이터 샘플:")
    print(engineered[['date', 'total_sales', 'day_of_week', 'month']].head(10))
    print(f"\n총 컬럼 수: {len(engineered.columns)}")
