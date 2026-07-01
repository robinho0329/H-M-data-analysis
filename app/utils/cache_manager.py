"""
데이터 캐싱 관리
"""
import streamlit as st
import pandas as pd
import logging
from pathlib import Path
import pickle
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)
def load_processed_data(path: str) -> pd.DataFrame:
    """처리된 데이터 로드 (캐시됨)"""
    try:
        data = pd.read_parquet(path)
        logger.info(f"✓ 캐시에서 데이터 로드: {path}")
        return data
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {path}")
        return None


@st.cache_resource
def load_normalizer(path: str):
    """정규화기 로드 (캐시됨)"""
    try:
        with open(path, 'rb') as f:
            normalizer = pickle.load(f)
        logger.info(f"✓ 정규화기 로드: {path}")
        return normalizer
    except Exception as e:
        logger.error(f"정규화기 로드 실패: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def load_raw_data():
    """원본 데이터 로드 (캐시됨).

    - 로컬: `Raw data/` 의 원본 CSV 3종(transactions·customer·articles)을 병합.
    - 배포(클라우드): 원본 CSV는 대용량이라 gitignore → 커밋된
      `data/processed/raw_sample.parquet`(고객 10% 대표표본)로 자동 폴백.
    """
    try:
        from pathlib import Path
        project_dir = Path(__file__).parent.parent.parent
        raw_dir = project_dir / "Raw data"
        tx_path = raw_dir / "transactions_hm.csv"

        if tx_path.exists():
            transactions = pd.read_csv(tx_path)
            customers = pd.read_csv(raw_dir / "customer_hm.csv")
            articles = pd.read_csv(raw_dir / "articles_hm.csv")
            merged = transactions.merge(customers, on='customer_id', how='left').merge(
                articles, on='article_id', how='left'
            )
            logger.info("✓ 원본 CSV 병합 로드")
        else:
            # 클라우드 폴백: 커밋된 대표표본 사용
            sample_path = project_dir / "data" / "processed" / "raw_sample.parquet"
            merged = pd.read_parquet(sample_path)
            logger.info("✓ 원본 CSV 미존재 → 대표표본(raw_sample.parquet) 사용")

        merged['t_dat'] = pd.to_datetime(merged['t_dat'])
        logger.info(f"✓ 원본 데이터 로드 완료: {merged.shape}")
        return merged
    except Exception as e:
        logger.error(f"원본 데이터 로드 실패: {str(e)}")
        return None


@st.cache_resource
def create_segment_analyzer():
    """세그먼트 분석기 생성 (캐시됨)"""
    try:
        project_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_dir))

        from src.data_loader import SegmentAnalyzer
        from src.preprocessor import TimeSeriesGenerator

        merged = load_raw_data()
        if merged is None:
            return None, None

        analyzer = SegmentAnalyzer(merged)
        ts_gen = TimeSeriesGenerator(merged)

        logger.info("✓ 세그먼트 분석기 생성 완료")
        return analyzer, ts_gen
    except Exception as e:
        logger.error(f"세그먼트 분석기 생성 실패: {str(e)}")
        return None, None


def clear_cache():
    """캐시 초기화"""
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("✓ 캐시 초기화 완료")
