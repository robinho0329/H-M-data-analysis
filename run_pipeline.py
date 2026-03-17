"""
LSTM 매출 예측 전체 파이프라인 실행
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
from src.models.lstm_model import LSTMModel, SequenceDataGenerator
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator, PredictionAnalyzer

import pandas as pd
import numpy as np
import pickle

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """전체 파이프라인 실행"""

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

    logger.info(f"✓ Train 데이터: {len(train)}일 ({train['date'].min().date()} ~ {train['date'].max().date()})")
    logger.info(f"✓ Val 데이터: {len(val)}일 ({val['date'].min().date()} ~ {val['date'].max().date()})")
    logger.info(f"✓ Test 데이터: {len(test)}일 ({test['date'].min().date()} ~ {test['date'].max().date()})")

    # =====================================================================
    # 6️⃣ 시퀀스 데이터 생성
    # =====================================================================
    logger.info("\n[단계 6] 시퀀스 데이터 생성")
    logger.info("-" * 80)

    seq_gen = SequenceDataGenerator(LOOKBACK_WINDOW)

    X_train, y_train = seq_gen.create_sequences(
        train[feature_cols].values,
        train[TARGET_COLUMN].values
    )

    X_val, y_val = seq_gen.create_sequences(
        val[feature_cols].values,
        val[TARGET_COLUMN].values
    )

    X_test, y_test = seq_gen.create_sequences(
        test[feature_cols].values,
        test[TARGET_COLUMN].values
    )

    logger.info(f"✓ Train 시퀀스: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"✓ Val 시퀀스: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"✓ Test 시퀀스: X={X_test.shape}, y={y_test.shape}")

    # =====================================================================
    # 7️⃣ LSTM 모델 구축 및 학습
    # =====================================================================
    logger.info("\n[단계 7] LSTM 모델 구축 및 학습")
    logger.info("-" * 80)

    lstm_model = LSTMModel(LOOKBACK_WINDOW, X_train.shape[2], "global_lstm")
    lstm_model.build_model()

    history = lstm_model.train(
        X_train, y_train,
        X_val, y_val,
        **LSTM_PARAMS
    )

    logger.info("✓ 모델 학습 완료")

    # 모델 저장
    GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    lstm_model.save(str(GLOBAL_MODEL_PATH))
    logger.info(f"✓ 모델 저장: {GLOBAL_MODEL_PATH}")

    # =====================================================================
    # 8️⃣ 모델 평가
    # =====================================================================
    logger.info("\n[단계 8] 모델 평가")
    logger.info("-" * 80)

    predictions_test = lstm_model.predict(X_test)

    evaluator = ModelEvaluator(predictions_test, y_test)
    metrics = evaluator.calculate_metrics()

    logger.info("\n📊 최종 성능 지표:")
    logger.info(f"  RMSE:  {metrics['rmse']:.6f}")
    logger.info(f"  MAE:   {metrics['mae']:.6f}")
    logger.info(f"  MAPE:  {metrics['mape']:.4f}%")
    logger.info(f"  R²:    {metrics['r2']:.4f}")

    # =====================================================================
    # 9️⃣ 예측 결과 분석
    # =====================================================================
    logger.info("\n[단계 9] 예측 결과 분석")
    logger.info("-" * 80)

    # 역정규화
    predictions_inv = normalizer.inverse_transform(
        pd.DataFrame({TARGET_COLUMN: predictions_test.flatten()})
    ).flatten()

    test_dates = test['date'].iloc[LOOKBACK_WINDOW:].reset_index(drop=True)
    y_test_inv = y_test.flatten()

    # 예측 데이터프레임
    forecast_df = pd.DataFrame({
        'date': test_dates,
        'actual': y_test_inv,
        'prediction': predictions_inv,
        'error': y_test_inv - predictions_inv,
        'error_pct': np.abs(y_test_inv - predictions_inv) / (y_test_inv + 1e-8) * 100,
    })

    logger.info(f"\n예측 결과 샘플 (최근 5일):")
    logger.info(forecast_df.tail(5).to_string(index=False))

    # =====================================================================
    # 🔟 세그먼트별 분석
    # =====================================================================
    logger.info("\n[단계 10] 세그먼트별 분석")
    logger.info("-" * 80)

    # 고객 세그먼트별 시계열 생성
    customer_segments = analyzer.create_customer_segments()
    logger.info(f"✓ 고객 세그먼트: {len(customer_segments)}개")

    for segment_name in list(customer_segments.keys())[:5]:
        segment_data = customer_segments[segment_name]
        total = segment_data['price'].sum()
        pct = (total / merged['price'].sum()) * 100
        logger.info(f"  - {segment_name}: ₩{total:,.0f} ({pct:.2f}%)")

    # 색상별 판매
    color_sales = merged.groupby('colour_group_name')['price'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
    logger.info(f"\n✓ 인기 색상 Top 5:")
    for idx, (color, row) in enumerate(color_sales.head(5).iterrows(), 1):
        logger.info(f"  {idx}. {color}: ₩{row['sum']:,.0f} ({int(row['count'])}건)")

    # 제품별 판매
    product_sales = merged.groupby('product_type_name')['price'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
    logger.info(f"\n✓ 인기 제품 Top 5:")
    for idx, (product, row) in enumerate(product_sales.head(5).iterrows(), 1):
        logger.info(f"  {idx}. {product}: ₩{row['sum']:,.0f} ({int(row['count'])}건)")

    # =====================================================================
    # 결과 저장
    # =====================================================================
    logger.info("\n[단계 11] 결과 저장")
    logger.info("-" * 80)

    # 예측 결과 저장
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
        'total_segments': len(customer_segments),
    }

    with open(results_dir / "segment_analysis.pkl", 'wb') as f:
        pickle.dump(segment_analysis, f)
    logger.info(f"✓ 세그먼트 분석: {results_dir / 'segment_analysis.pkl'}")

    # 4. 학습 이력 저장
    history_dict = lstm_model.get_history()
    if history_dict:
        history_df = pd.DataFrame(history_dict)
        history_df.to_csv(results_dir / "training_history.csv", index=False)
        logger.info(f"✓ 학습 이력: {results_dir / 'training_history.csv'}")

    # =====================================================================
    # 📊 최종 요약
    # =====================================================================
    logger.info("\n" + "="*80)
    logger.info("✅ 파이프라인 완료!")
    logger.info("="*80)

    logger.info(f"""
📈 실행 결과 요약:

  📊 데이터:
    - 거래 건수: {len(merged):,}건
    - 고객 수: {merged['customer_id'].nunique():,}명
    - 상품 수: {merged['article_id'].nunique():,}개

  📅 시계열:
    - 기간: {daily_sales['date'].min().date()} ~ {daily_sales['date'].max().date()}
    - 총 매출: ₩{daily_sales['total_sales'].sum():,.0f}
    - 일평균: ₩{daily_sales['total_sales'].mean():,.2f}

  🧠 모델:
    - Lookback: {LOOKBACK_WINDOW}일
    - 입력 특성: {X_train.shape[2]}개
    - 학습 샘플: {len(X_train)}개

  📊 성능:
    - RMSE: {metrics['rmse']:.6f}
    - MAE: {metrics['mae']:.6f}
    - MAPE: {metrics['mape']:.4f}%
    - R²: {metrics['r2']:.4f}

  🎨 세그먼트:
    - 고객 세그먼트: {len(customer_segments)}개
    - 색상: {len(top_colors)}개 분석
    - 제품: {len(top_products)}개 분석
    - 의류 그룹: {len(garment_groups)}개

  💾 저장 위치:
    - 모델: {GLOBAL_MODEL_PATH}
    - 예측: {results_dir / 'forecast_test.csv'}
    - 메트릭: {results_dir / 'metrics.csv'}

  🚀 다음 단계:
    streamlit run app/dashboard.py
    """)

    return forecast_df, metrics, daily_sales


if __name__ == "__main__":
    forecast_df, metrics, daily_sales = main()
