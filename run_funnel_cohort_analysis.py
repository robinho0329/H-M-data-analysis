"""
퍼널 분석 및 코호트 분석 실행
"""
import sys
import logging
import pickle
from pathlib import Path

# 프로젝트 경로 설정
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from config import *
from src.data_loader import DataLoader
from src.funnel_analyzer import FunnelAnalyzer
from src.cohort_analyzer import CohortAnalyzer

import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """퍼널 분석 및 코호트 분석 실행"""

    logger.info("\n" + "="*80)
    logger.info("🔍 퍼널 분석 및 코호트 분석 시작")
    logger.info("="*80)

    # =====================================================================
    # 1️⃣ 데이터 로드
    # =====================================================================
    logger.info("\n[단계 1] 데이터 로드")
    logger.info("-" * 80)

    loader = DataLoader(TRANSACTIONS_PATH, CUSTOMERS_PATH, ARTICLES_PATH)
    merged = loader.prepare_data()

    logger.info(f"✓ 병합 데이터 로드 완료")
    logger.info(f"  - 거래: {len(merged):,}건")
    logger.info(f"  - 고객: {merged['customer_id'].nunique():,}명")
    logger.info(f"  - 상품: {merged['article_id'].nunique():,}종")
    logger.info(f"  - 기간: {merged['t_dat'].min().date()} ~ {merged['t_dat'].max().date()}")

    # =====================================================================
    # 2️⃣ 퍼널 분석
    # =====================================================================
    logger.info("\n[단계 2] 퍼널 분석")
    logger.info("-" * 80)

    funnel_analyzer = FunnelAnalyzer(merged)

    # 라이프사이클 퍼널
    logger.info("\n📊 고객 라이프사이클 퍼널")
    lifecycle_funnel, _ = funnel_analyzer.create_customer_lifecycle_funnel()
    logger.info("\n" + str(lifecycle_funnel))

    # 구매 빈도 퍼널
    logger.info("\n📊 구매 빈도 퍼널")
    frequency_funnel = funnel_analyzer.create_purchase_frequency_funnel()
    logger.info("\n" + str(frequency_funnel))

    # 제품 다양성 퍼널
    logger.info("\n📊 제품 다양성 퍼널")
    category_funnel = funnel_analyzer.create_product_category_funnel()
    logger.info("\n" + str(category_funnel))

    # 판매 채널 퍼널
    logger.info("\n📊 판매 채널 퍼널")
    channel_funnel = funnel_analyzer.create_sales_channel_funnel()
    logger.info("\n" + str(channel_funnel))

    # =====================================================================
    # 3️⃣ 코호트 분석
    # =====================================================================
    logger.info("\n[단계 3] 코호트 분석")
    logger.info("-" * 80)

    cohort_analyzer = CohortAnalyzer(merged)

    # 리텐션 코호트
    logger.info("\n📈 월별 코호트 리텐션율 (상위 10개)")
    retention_cohort = cohort_analyzer.create_monthly_cohort_retention()
    logger.info("\n" + str(retention_cohort.head(10)))

    # 매출 코호트
    logger.info("\n💰 월별 코호트 매출 (상위 10개)")
    revenue_cohort = cohort_analyzer.create_monthly_cohort_revenue()
    logger.info("\n" + str(revenue_cohort.head(10)))

    # 평균 지출액 코호트
    logger.info("\n💸 월별 코호트 평균 지출액 (상위 10개)")
    avg_spending_cohort = cohort_analyzer.create_monthly_cohort_avg_spending()
    logger.info("\n" + str(avg_spending_cohort.head(10)))

    # 나이대별 코호트
    logger.info("\n👥 나이대별 월별 구매 (상위 15개)")
    age_cohort = cohort_analyzer.create_age_group_cohort()
    logger.info("\n" + str(age_cohort.head(15)))

    # 클럽 상태별 코호트
    logger.info("\n🏢 클럽 상태별 월별 구매")
    club_cohort = cohort_analyzer.create_club_status_cohort()
    logger.info("\n" + str(club_cohort.head(20)))

    # 뉴스레터별 코호트
    logger.info("\n📧 뉴스레터 상태별 월별 구매")
    newsletter_cohort = cohort_analyzer.create_newsletter_cohort()
    logger.info("\n" + str(newsletter_cohort.head(20)))

    # =====================================================================
    # 4️⃣ 결과 저장
    # =====================================================================
    logger.info("\n[단계 4] 결과 저장")
    logger.info("-" * 80)

    results_dir = PROCESSED_DIR / "analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 퍼널 분석 결과
    funnel_results = {
        'lifecycle': lifecycle_funnel,
        'frequency': frequency_funnel,
        'category': category_funnel,
        'channel': channel_funnel
    }

    # 코호트 분석 결과
    cohort_results = {
        'retention': retention_cohort,
        'revenue': revenue_cohort,
        'avg_spending': avg_spending_cohort,
        'age_group': age_cohort,
        'club_status': club_cohort,
        'newsletter': newsletter_cohort
    }

    # CSV로 저장
    for key, df in funnel_results.items():
        filepath = results_dir / f"funnel_{key}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"✓ {filepath.name} 저장 완료")

    for key, df in cohort_results.items():
        filepath = results_dir / f"cohort_{key}.csv"
        if isinstance(df, pd.DataFrame) and df.index.name:
            df.to_csv(filepath)
        else:
            df.to_csv(filepath, index=False)
        logger.info(f"✓ {filepath.name} 저장 완료")

    # Pickle로도 저장 (나중에 대시보드에서 로드)
    all_results = {
        'funnel': funnel_results,
        'cohort': cohort_results
    }

    pickle_path = results_dir / "analysis_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"✓ {pickle_path.name} 저장 완료")

    # =====================================================================
    # 5️⃣ 주요 인사이트
    # =====================================================================
    logger.info("\n[단계 5] 주요 인사이트")
    logger.info("-" * 80)

    # 라이프사이클 분석
    active_pct = lifecycle_funnel[lifecycle_funnel['lifecycle'] == 'Active']['percentage'].values[0]
    logger.info(f"\n🎯 고객 라이프사이클:")
    logger.info(f"  - 활성 고객 비율: {active_pct:.1f}%")
    logger.info(f"  - 신규 고객: {lifecycle_funnel[lifecycle_funnel['lifecycle'] == 'New']['customer_count'].values[0]:,}명")
    logger.info(f"  - 비활성 고객: {lifecycle_funnel[lifecycle_funnel['lifecycle'] == 'Inactive']['customer_count'].values[0]:,}명")

    # 구매 빈도 분석
    repeat_customers = frequency_funnel[frequency_funnel['frequency'].isin(['Regular', 'Frequent'])]['customer_count'].sum()
    repeat_pct = (repeat_customers / frequency_funnel['customer_count'].sum() * 100)
    logger.info(f"\n🔄 재구매 고객:")
    logger.info(f"  - 정기 이상 구매: {repeat_pct:.1f}%")
    logger.info(f"  - 1회 구매: {frequency_funnel[frequency_funnel['frequency'] == 'One-time']['percentage'].values[0]:.1f}%")

    # 채널 분석
    online_revenue = channel_funnel[channel_funnel['channel'] == 'Online']['total_revenue'].values[0]
    offline_revenue = channel_funnel[channel_funnel['channel'] == 'Offline']['total_revenue'].values[0]
    total_revenue = online_revenue + offline_revenue
    logger.info(f"\n🛍️ 판매 채널:")
    logger.info(f"  - 온라인 매출: {online_revenue:,.0f}원 ({online_revenue/total_revenue*100:.1f}%)")
    logger.info(f"  - 오프라인 매출: {offline_revenue:,.0f}원 ({offline_revenue/total_revenue*100:.1f}%)")

    # 코호트 리텐션
    first_month_retained = (retention_cohort.iloc[:, 1] > 0).sum()
    logger.info(f"\n📊 코호트 리텐션:")
    logger.info(f"  - 2개월차 유지율이 있는 코호트: {first_month_retained}개")
    logger.info(f"  - 평균 1개월 리텐션: {retention_cohort.iloc[:, 1].mean():.1f}%")

    logger.info("\n" + "="*80)
    logger.info("✅ 퍼널 분석 및 코호트 분석 완료!")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
