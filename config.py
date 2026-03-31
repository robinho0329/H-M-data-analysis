"""
LSTM Sales Forecasting Project Configuration
"""
from pathlib import Path

# 프로젝트 경로
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "Raw data"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURES_DIR = PROJECT_DIR / "data" / "features"
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"

# 데이터 파일
TRANSACTIONS_PATH = DATA_DIR / "transactions_hm.csv"
CUSTOMERS_PATH = DATA_DIR / "customer_hm.csv"
ARTICLES_PATH = DATA_DIR / "articles_hm.csv"

# 처리된 데이터
PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_data.parquet"
TIME_SERIES_PATH = PROCESSED_DIR / "time_series.parquet"
FEATURES_PATH = FEATURES_DIR / "engineered_features.parquet"

# 모델 파일
GLOBAL_MODEL_PATH = MODELS_DIR / "weights" / "global_lstm.h5"
SCALERS_PATH = MODELS_DIR / "scalers"

# 데이터 설정
DATE_COLUMN = "t_dat"
DATE_FORMAT = "%Y-%m-%d"

# 시계열 설정
LOOKBACK_WINDOW = 30  # 과거 30일로 다음날 예측
TARGET_COLUMN = "total_sales"

# 모델 설정
LSTM_PARAMS = {
    "lookback": LOOKBACK_WINDOW,
    "batch_size": 32,
    "epochs": 150,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "early_stopping_patience": 15,
}

# 시계열 분할
TRAIN_END_DATE = "2019-11-30"  # 70%
VAL_END_DATE = "2019-12-15"    # 15%
# Test: 2019-12-16 ~ 2019-12-31 (15%)

# 고객 세그먼트
CUSTOMER_SEGMENTS = {
    "active": ["Active"],
    "age_groups": ["10s", "20s", "30s", "40s", "50s", "60s", "70+"],
    "club_status": ["ACTIVE", "PRE-CREATE", "LEFT CLUB"],
    "news_frequency": ["NONE", "Monthly", "Regularly"],
}

# 상품 세그먼트
PRODUCT_SEGMENTS = {
    "top_colors": 15,  # Top 15 색상
    "top_product_types": 15,  # Top 15 제품타입
    "all_garment_groups": True,  # 모든 의류그룹 (21개)
}

# 파생변수 설정
LAG_FEATURES = [1, 2, 3, 7, 14, 30]
MA_WINDOWS = [7, 14, 30]

# 평가 지표
EVAL_METRICS = ["rmse", "mae", "mape", "r2"]

# 시각화
COLORS = {
    "primary": "#E50010",
    "secondary": "#222222",
    "success": "#2ca02c",
    "danger": "#E50010",
    "warning": "#FF6B6B",
}
