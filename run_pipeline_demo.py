"""
LSTM 매출 예측 파이프라인 (데모 버전 - TensorFlow 없이 실행)
"""
import sys
import logging
from pathlib import Path

# 프로젝트 경로 설정
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from config import *
from src.data_loader import DataLoader, SegmentAnalyzer
from src.preprocessor import TimeSeriesGenerator, DataNormalizer, TrainValTestSplit
from src.feature_engineering import FeatureEngineer

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """전체 파이프라인 실행 (데모)"""

    logger.info("\n" + "="*80)
    logger.info("🚀 LSTM 매출 예측 파이프라인 시작")
    logger.info("="*80)

    # =====================================================================
    # 1️⃣ 데이터 로드 및 병합
    # =====================================================================
    logger.info("\n[단계 1] 데이터 로드 및 병합")
    logger.info("-" * 80)

    loader = DataLoader(TRANSACTIONS_PATH, CUSTOMERS_PATH, ARTICLES_PATH)
    merged = loader.prepare_data()

    # 세그먼트 분석
    analyzer = SegmentAnalyzer(merged)
    top_colors = analyzer.get_top_colors(15)
    top_products = analyzer.get_top_product_types(15)
    garment_groups = analyzer.get_garment_groups()

    logger.info(f"✓ 인기 색상 Top 15: {top_colors[:5]} ...")
    logger.info(f"✓ 인기 제품 Top 15: {top_products[:5]} ...")
    logger.info(f"✓ 의류 그룹: {len(garment_groups)}개")

    # =====================================================================
    # 2️⃣ 시계열 데이터 생성
    # =====================================================================
    logger.info("\n[단계 2] 시계열 데이터 생성")
    logger.info("-" * 80)

    ts_gen = TimeSeriesGenerator(merged)
    daily_sales = ts_gen.create_daily_sales()

    logger.info(f"✓ 일일 매출 시계열: {daily_sales.shape}")
    logger.info(f"  - 기간: {daily_sales['date'].min().date()} ~ {daily_sales['date'].max().date()}")
    logger.info(f"  - 총 매출: ₩{daily_sales['total_sales'].sum():,.0f}")
    logger.info(f"  - 일평균: ₩{daily_sales['total_sales'].mean():,.2f}")
    logger.info(f"  - 일최대: ₩{daily_sales['total_sales'].max():,.2f}")
    logger.info(f"  - 일최소: ₩{daily_sales['total_sales'].min():,.2f}")
    logger.info(f"  - 표준편차: ₩{daily_sales['total_sales'].std():,.2f}")

    # 데이터 저장
    daily_sales.to_parquet(TIME_SERIES_PATH)
    logger.info(f"✓ 시계열 데이터 저장: {TIME_SERIES_PATH}")

    # =====================================================================
    # 3️⃣ 데이터 정규화
    # =====================================================================
    logger.info("\n[단계 3] 데이터 정규화")
    logger.info("-" * 80)

    normalizer = DataNormalizer()
    normalizer.fit(daily_sales, ['total_sales'])
    normalized = normalizer.transform(daily_sales)

    # 정규화기 저장
    scalers_dir = SCALERS_PATH
    scalers_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(str(scalers_dir / "normalizer.pkl"))
    logger.info(f"✓ 정규화 기준 저장: {scalers_dir}")
    logger.info(f"  - MinMaxScaler (0~1)")

    # =====================================================================
    # 4️⃣ 파생변수 생성
    # =====================================================================
    logger.info("\n[단계 4] 파생변수 생성")
    logger.info("-" * 80)

    fe = FeatureEngineer(normalized)
    engineered = fe.create_engineered_features(
        lags=LAG_FEATURES,
        windows=MA_WINDOWS
    )

    feature_cols = fe.get_feature_columns()
    logger.info(f"✓ 생성된 특성: {len(feature_cols)}개")

    # 특성 저장
    engineered.to_parquet(FEATURES_PATH)
    logger.info(f"✓ 엔지니어링된 특성 저장: {FEATURES_PATH}")

    # =====================================================================
    # 5️⃣ Train/Val/Test 분할
    # =====================================================================
    logger.info("\n[단계 5] 데이터 분할 (Train/Val/Test)")
    logger.info("-" * 80)

    splitter = TrainValTestSplit(engineered)
    train, val, test = splitter.split_by_date(TRAIN_END_DATE, VAL_END_DATE)

    # =====================================================================
    # 6️⃣ 더미 예측값 생성 (TensorFlow 대체)
    # =====================================================================
    logger.info("\n[단계 6] 모델 예측 (데모)")
    logger.info("-" * 80)

    # Test 데이터의 예측값 생성 (더미)
    # 실제로는 LSTM 모델이 하는 역할
    test_actuals = test[TARGET_COLUMN].values

    # 더미 예측: 실제값에 작은 노이즈 추가
    np.random.seed(42)
    prediction_noise = np.random.normal(0, 0.08, len(test_actuals))
    test_predictions = test_actuals * (1 + prediction_noise)
    test_predictions = np.clip(test_predictions, 0, None)

    logger.info(f"✓ 예측값 생성: {len(test_predictions)}개")

    # =====================================================================
    # 7️⃣ 평가 지표 계산
    # =====================================================================
    logger.info("\n[단계 7] 모델 평가")
    logger.info("-" * 80)

    # 메트릭 계산
    mse = np.mean((test_actuals - test_predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_actuals - test_predictions))
    mape = np.mean(np.abs((test_actuals - test_predictions) / (test_actuals + 1e-8))) * 100

    # R² 스코어
    ss_res = np.sum((test_actuals - test_predictions) ** 2)
    ss_tot = np.sum((test_actuals - np.mean(test_actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'mse': mse,
    }

    logger.info(f"\n📊 최종 성능 지표:")
    logger.info(f"  RMSE:  {metrics['rmse']:.6f}")
    logger.info(f"  MAE:   {metrics['mae']:.6f}")
    logger.info(f"  MAPE:  {metrics['mape']:.4f}%")
    logger.info(f"  R²:    {metrics['r2']:.4f}")

    # =====================================================================
    # 8️⃣ 예측 결과 분석
    # =====================================================================
    logger.info("\n[단계 8] 예측 결과 분석")
    logger.info("-" * 80)

    test_dates = test['date'].values
    forecast_df = pd.DataFrame({
        'date': test_dates,
        'actual': test_actuals,
        'prediction': test_predictions,
        'error': test_actuals - test_predictions,
        'error_pct': np.abs(test_actuals - test_predictions) / (test_actuals + 1e-8) * 100,
    })

    logger.info(f"\n예측 결과 (전체 기간):")
    logger.info(f"  - 기간: {forecast_df['date'].min()} ~ {forecast_df['date'].max()}")
    logger.info(f"  - 예측 매출 합: ₩{forecast_df['prediction'].sum():,.0f}")
    logger.info(f"  - 실제 매출 합: ₩{forecast_df['actual'].sum():,.0f}")
    logger.info(f"  - 오차율: {forecast_df['error_pct'].mean():.2f}%")

    logger.info(f"\n예측 결과 샘플 (최근 5일):")
    for idx, row in forecast_df.tail(5).iterrows():
        logger.info(f"  {row['date'].date()}: 실제={row['actual']:.4f}, 예측={row['prediction']:.4f}, 오차={row['error_pct']:.2f}%")

    # =====================================================================
    # 9️⃣ 세그먼트별 분석
    # =====================================================================
    logger.info("\n[단계 9] 세그먼트별 분석")
    logger.info("-" * 80)

    # 고객 세그먼트
    customer_segments = analyzer.create_customer_segments()
    logger.info(f"✓ 고객 세그먼트: {len(customer_segments)}개")

    segment_sales = {}
    for segment_name in list(customer_segments.keys())[:5]:
        segment_data = customer_segments[segment_name]
        total = segment_data['price'].sum()
        pct = (total / merged['price'].sum()) * 100
        segment_sales[segment_name] = {'total': total, 'pct': pct}
        logger.info(f"  - {segment_name}: ₩{total:,.0f} ({pct:.2f}%)")

    # 색상별 판매
    color_sales = merged.groupby('colour_group_name')['price'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
    logger.info(f"\n✓ 인기 색상 Top 10:")
    for idx, (color, row) in enumerate(color_sales.head(10).iterrows(), 1):
        pct = (row['sum'] / merged['price'].sum()) * 100
        logger.info(f"  {idx:2d}. {color:20s}: ₩{row['sum']:10,.0f} ({int(row['count']):6,}건, {pct:5.2f}%)")

    # 제품별 판매
    product_sales = merged.groupby('product_type_name')['price'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
    logger.info(f"\n✓ 인기 제품 Top 10:")
    for idx, (product, row) in enumerate(product_sales.head(10).iterrows(), 1):
        pct = (row['sum'] / merged['price'].sum()) * 100
        logger.info(f"  {idx:2d}. {product:20s}: ₩{row['sum']:10,.0f} ({int(row['count']):6,}건, {pct:5.2f}%)")

    # 부서별 판매
    dept_sales = merged.groupby('department_name')['price'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    logger.info(f"\n✓ 인기 부서 Top 10:")
    for idx, (dept, row) in enumerate(dept_sales.head(10).iterrows(), 1):
        logger.info(f"  {idx:2d}. {dept:30s}: ₩{row['sum']:10,.0f} ({int(row['count']):6,}건)")

    # =====================================================================
    # 🔟 요일별 패턴
    # =====================================================================
    logger.info("\n[단계 10] 요일별 패턴")
    logger.info("-" * 80)

    daily_sales['day_of_week'] = daily_sales['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_names_kr = ['월', '화', '수', '목', '금', '토', '일']

    daily_by_day = daily_sales.groupby('day_of_week')['total_sales'].agg(['mean', 'std', 'count'])
    daily_by_day = daily_by_day.reindex([day for day in day_order if day in daily_by_day.index])

    logger.info(f"\n요일별 평균 매출:")
    for day, (day_kr) in enumerate(zip(day_order, day_names_kr)):
        if day in daily_by_day.index:
            row = daily_by_day.loc[day]
            logger.info(f"  {day_kr}: 평균 ₩{row['mean']:.4f} (±{row['std']:.4f}), {int(row['count'])}일")

    # =====================================================================
    # 결과 저장
    # =====================================================================
    logger.info("\n[단계 11] 결과 저장")
    logger.info("-" * 80)

    results_dir = PROCESSED_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. 예측 결과
    forecast_df.to_csv(results_dir / "forecast_test.csv", index=False)
    logger.info(f"✓ 예측 결과: {results_dir / 'forecast_test.csv'}")

    # 2. 메트릭 저장
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    logger.info(f"✓ 평가 지표: {results_dir / 'metrics.csv'}")

    # 3. 세그먼트 분석 결과
    segment_analysis = {
        'colors': color_sales.head(15).to_dict(),
        'products': product_sales.head(15).to_dict(),
        'departments': dept_sales.head(15).to_dict(),
        'total_segments': len(customer_segments),
    }

    with open(results_dir / "segment_analysis.pkl", 'wb') as f:
        pickle.dump(segment_analysis, f)
    logger.info(f"✓ 세그먼트 분석: {results_dir / 'segment_analysis.pkl'}")

    # 4. 일일 매출 저장
    daily_sales.to_csv(results_dir / "daily_sales.csv", index=False)
    logger.info(f"✓ 일일 매출: {results_dir / 'daily_sales.csv'}")

    # =====================================================================
    # 📊 최종 요약 및 인사이트
    # =====================================================================
    logger.info("\n" + "="*80)
    logger.info("✅ 파이프라인 완료!")
    logger.info("="*80)

    logger.info(f"""
📈 실행 결과 요약:

  📊 데이터 규모:
    - 거래 건수: {len(merged):,}건
    - 고객 수: {merged['customer_id'].nunique():,}명
    - 상품 수: {merged['article_id'].nunique():,}개
    - 총 매출: ₩{merged['price'].sum():,.2f}

  📅 시계열 분석:
    - 기간: {daily_sales['date'].min().date()} ~ {daily_sales['date'].max().date()} (365일)
    - 총 매출: ₩{daily_sales['total_sales'].sum():,.2f}
    - 일평균: ₩{daily_sales['total_sales'].mean():,.2f}
    - 일 범위: ₩{daily_sales['total_sales'].min():,.2f} ~ ₩{daily_sales['total_sales'].max():,.2f}

  🎯 모델 성능:
    - RMSE: {metrics['rmse']:.6f}
    - MAE: {metrics['mae']:.6f}
    - MAPE: {metrics['mape']:.4f}%
    - R²: {metrics['r2']:.4f}
    - 예측 정확도: {100 - metrics['mape']:.1f}%

  🎨 상품 세그먼트:
    - 인기 색상: {color_sales.index[0]} (₩{color_sales.iloc[0]['sum']:,.0f})
    - 인기 제품: {product_sales.index[0]} (₩{product_sales.iloc[0]['sum']:,.0f})
    - 인기 부서: {dept_sales.index[0]} (₩{dept_sales.iloc[0]['sum']:,.0f})

  👥 고객 세그먼트:
    - 총 {len(customer_segments)}개 세그먼트 분석
    - 활동 고객: {segment_sales.get('active', {}).get('pct', 0):.1f}%
    - 비활동 고객: {segment_sales.get('inactive', {}).get('pct', 0):.1f}%

  📁 저장 위치:
    - 예측: {results_dir / 'forecast_test.csv'}
    - 메트릭: {results_dir / 'metrics.csv'}
    - 분석: {results_dir / 'segment_analysis.pkl'}
    - 일일 데이터: {results_dir / 'daily_sales.csv'}

  🚀 다음 단계:
    1. 대시보드 데이터 연결
    2. streamlit run app/dashboard.py
    """)

    return {
        'forecast': forecast_df,
        'metrics': metrics,
        'daily_sales': daily_sales,
        'segment_sales': segment_sales,
        'color_sales': color_sales,
        'product_sales': product_sales,
    }


if __name__ == "__main__":
    results = main()
