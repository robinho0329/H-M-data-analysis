# H&M LSTM 매출 예측 대시보드

H&M 거래 데이터를 활용한 LSTM 기반 시계열 매출 예측 시스템

## 📋 프로젝트 개요

- **목표**: LSTM을 이용한 일일 매출 예측
- **데이터**: H&M 트랜잭션 데이터 (2019년)
- **기간**: 2019년 1월 1일 ~ 2019년 12월 31일
- **거래건수**: 약 105만 건
- **고객수**: 약 45만 명
- **상품수**: 약 5만 개

## 📁 프로젝트 구조

```
basic_project/
├── Raw data/                          # 원본 데이터
│   ├── transactions_hm.csv           # 거래 데이터
│   ├── customer_hm.csv               # 고객 정보
│   └── articles_hm.csv               # 상품 정보
│
├── notebooks/                         # 분석 및 실험 노트북
│   └── [분석 결과]
│
├── src/                              # 핵심 모듈
│   ├── data_loader.py               # 데이터 로드 및 병합
│   ├── preprocessor.py              # 데이터 전처리 및 시계열 생성
│   ├── feature_engineering.py       # 파생변수 생성
│   ├── trainer.py                   # 모델 학습 관리
│   ├── evaluator.py                 # 모델 평가 및 시각화
│   ├── models/
│   │   ├── lstm_model.py           # LSTM 모델 정의
│   │   └── __init__.py
│   ├── utils/                       # 유틸 함수
│   │   └── __init__.py
│   └── __init__.py
│
├── app/                             # Streamlit 대시보드
│   ├── dashboard.py                 # 메인 대시보드
│   ├── pages/                       # 대시보드 페이지
│   │   ├── page_overview.py        # 1. 예측 현황
│   │   ├── page_customer_segments.py # 2. 고객 세그먼트
│   │   ├── page_product_segments.py # 3. 상품 세그먼트
│   │   ├── page_detailed_analysis.py # 4. 상세 분석
│   │   ├── page_performance.py     # 5. 성능 & 모델
│   │   ├── page_eda.py             # 6. 데이터 탐색
│   │   └── __init__.py
│   ├── utils/                      # 대시보드 유틸
│   │   ├── cache_manager.py       # 캐싱
│   │   └── __init__.py
│   └── __init__.py
│
├── data/                           # 처리된 데이터
│   ├── processed/                 # 전처리된 데이터
│   │   ├── processed_data.parquet
│   │   └── time_series.parquet
│   ├── features/                  # 엔지니어링된 특성
│   │   └── engineered_features.parquet
│   └── [중간 결과물]
│
├── models/                        # 학습된 모델
│   ├── weights/                  # 모델 가중치
│   │   ├── global_lstm.h5
│   │   └── [세그먼트별 모델]
│   └── scalers/                  # 정규화 기준
│
├── config.py                      # 프로젝트 설정
├── requirements.txt               # 의존성
├── README.md                      # 프로젝트 문서
└── .gitignore
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 디렉토리 이동
cd basic_project

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 전처리

```bash
# 데이터 로드 및 시계열 생성
python src/preprocessor.py

# 파생변수 생성
python src/feature_engineering.py
```

### 3. 모델 학습

```bash
# LSTM 모델 학습 (노트북)
jupyter notebook notebooks/01_LSTM_Pipeline.ipynb
```

### 4. 대시보드 실행

```bash
# Streamlit 대시보드 시작
streamlit run app/dashboard.py
```

## 📊 주요 기능

### 데이터 처리 (src/)

| 모듈 | 설명 |
|------|------|
| `data_loader.py` | H&M 데이터 로드 및 병합 |
| `preprocessor.py` | 시계열 생성, 정규화, 데이터 분할 |
| `feature_engineering.py` | 시간/래그/이동평균/성장률 특성 |
| `trainer.py` | 모델 학습 및 관리 |
| `evaluator.py` | 성능 평가 및 시각화 |

### LSTM 모델 (src/models/)

```python
# 모델 아키텍처
LSTM(128) → LSTM(64) → LSTM(32) → Dense(32) → Dense(16) → Dense(1)
```

**특성:**
- Lookback: 30일
- 정규화: MinMaxScaler
- 최적화: Adam (lr=0.001)
- 손실함수: MSE
- Early Stopping: patience=15

### 대시보드 페이지 (app/)

1. **📈 예측 현황** - 실제 vs 예측 매출 비교
2. **👥 고객 세그먼트** - 활동도, 연령대, 클럽 상태별 분석
3. **🛍️ 상품 세그먼트** - 색상, 제품타입, 의류그룹별 분석
4. **🔍 상세 분석** - 색상/제품/계절성/교차분석
5. **📊 성능 & 모델** - 성능 지표, 학습 곡선, 모델 설정
6. **📉 데이터 탐색** - 원본 데이터 EDA

## 📈 예상 성능

| 지표 | 목표 |
|------|------|
| RMSE | < 2.0 |
| MAE | < 1.5 |
| MAPE | < 5% |
| R² Score | > 0.80 |

## 🔧 설정

`config.py`에서 주요 설정을 변경할 수 있습니다:

```python
# 시계열 설정
LOOKBACK_WINDOW = 30           # 입력 시퀀스 길이
TARGET_COLUMN = "total_sales"  # 예측 타겟

# 모델 설정
LSTM_PARAMS = {
    "batch_size": 32,
    "epochs": 150,
    "learning_rate": 0.001,
    "dropout": 0.2,
}

# 세그먼트 설정
PRODUCT_SEGMENTS = {
    "top_colors": 15,
    "top_product_types": 15,
    "all_garment_groups": True,
}
```

## 📚 데이터 설명

### Transactions (거래 데이터)
- `t_dat`: 거래 날짜
- `customer_id`: 고객 ID
- `article_id`: 상품 ID
- `price`: 정규화된 가격 (0~1)
- `sales_channel_id`: 판매 채널 (1: 온라인, 2: 오프라인)

### Customers (고객 정보)
- `customer_id`: 고객 ID
- `FN`: 레코드 번호
- `Active`: 활동 여부 (0/1)
- `club_member_status`: 클럽 상태
- `fashion_news_frequency`: 뉴스레터 구독 빈도
- `age`: 나이

### Articles (상품 정보)
- `article_id`: 상품 ID
- `product_type_name`: 제품 타입
- `colour_group_name`: 색상
- `department_name`: 부서
- `section_name`: 섹션
- `garment_group_name`: 의류 그룹
- 기타 25개 속성

## 🎯 주요 분석 포인트

### 세그먼트 분석

**고객 세그먼트:**
- 활동도 (Active/Inactive)
- 연령대 (10s~70+)
- 클럽 상태 (3가지)
- 뉴스레터 (3단계)

**상품 세그먼트:**
- 색상: Top 15 + Others
- 제품타입: Top 15 + Others
- 의류그룹: 21개 모두
- 가격대: 4단계

**교차 분석:**
- 연령대별 인기 색상
- 활동도별 선호 상품
- 뉴스레터 구독자의 구매력
- 계절별 트렌드

## 🔄 데이터 파이프라인

```
Raw Data
   ↓
[데이터 로드 & 병합]
   ↓
[전처리]
   ├─ 결측값 처리
   ├─ 정규화
   └─ 시계열 생성
   ↓
[특성공학]
   ├─ 시간 특성
   ├─ 래그 특성
   ├─ 이동평균
   └─ 성장률
   ↓
[Train/Val/Test 분할]
   ├─ Train: 70% (2019-01~11)
   ├─ Val: 15% (2019-12-01~15)
   └─ Test: 15% (2019-12-16~31)
   ↓
[LSTM 모델 학습]
   ↓
[평가 & 예측]
   ↓
[대시보드 시각화]
```

## 📝 사용 예제

### 데이터 로드
```python
from src.data_loader import DataLoader

loader = DataLoader(transactions_path, customers_path, articles_path)
merged_data = loader.prepare_data()
```

### 시계열 생성
```python
from src.preprocessor import TimeSeriesGenerator, DataNormalizer

ts_gen = TimeSeriesGenerator(merged_data)
daily_sales = ts_gen.create_daily_sales()

normalizer = DataNormalizer()
normalizer.fit(daily_sales, ['total_sales'])
normalized = normalizer.transform(daily_sales)
```

### LSTM 모델 학습
```python
from src.models.lstm_model import LSTMModel

model = LSTMModel(lookback=30, n_features=25)
model.build_model()
history = model.train(X_train, y_train, X_val, y_val, epochs=150)
predictions = model.predict(X_test)
```

## 📊 성능 지표 계산

```python
from src.evaluator import ModelEvaluator

evaluator = ModelEvaluator(predictions, actuals)
metrics = evaluator.calculate_metrics()

# 출력
# RMSE: 1.2345
# MAE: 0.8765
# MAPE: 3.45%
# R²: 0.8234
```

## 🤝 기여

이슈 및 개선 사항은 언제든 환영합니다.

## 📄 라이선스

MIT License

## 📞 연락

프로젝트 관련 문의는 이메일로 연락해주세요.

---

**마지막 업데이트**: 2024년 1월 15일
