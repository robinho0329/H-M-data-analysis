"""
대시보드 완전성 테스트
"""
import sys
from pathlib import Path

# 프로젝트 경로 설정
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from config import PROCESSED_DIR
import pandas as pd
import traceback


def test_all_pages():
    """모든 대시보드 페이지 테스트"""
    print("\n" + "="*80)
    print("🧪 대시보드 완전성 테스트 시작")
    print("="*80 + "\n")

    test_results = {}

    # =====================================================================
    # 1️⃣ 퍼널 분석 페이지 테스트
    # =====================================================================
    print("[1️⃣] 퍼널 분석 페이지 테스트")
    print("-" * 80)

    try:
        # 모듈 임포트
        from app.pages.page_funnel_analysis import load_funnel_data, show

        # 데이터 로드
        funnel_data = load_funnel_data()
        assert isinstance(funnel_data, dict), "funnel_data는 딕셔너리여야 함"
        assert 'lifecycle' in funnel_data, "lifecycle 키 필요"
        assert 'frequency' in funnel_data, "frequency 키 필요"
        assert 'category' in funnel_data, "category 키 필요"
        assert 'channel' in funnel_data, "channel 키 필요"

        # 데이터 검증
        for key, df in funnel_data.items():
            assert isinstance(df, pd.DataFrame), f"{key}는 DataFrame이어야 함"
            assert len(df) > 0, f"{key}가 비어있음"
            print(f"  ✓ {key}: {df.shape[0]}행 {df.shape[1]}열")

        print("✅ 퍼널 분석 페이지: 통과\n")
        test_results['퍼널 분석'] = True

    except Exception as e:
        print(f"❌ 퍼널 분석 페이지 에러: {e}\n")
        traceback.print_exc()
        test_results['퍼널 분석'] = False

    # =====================================================================
    # 2️⃣ 코호트 분석 페이지 테스트
    # =====================================================================
    print("[2️⃣] 코호트 분석 페이지 테스트")
    print("-" * 80)

    try:
        # 모듈 임포트
        from app.pages.page_cohort_analysis import load_cohort_data, show

        # 데이터 로드
        cohort_data = load_cohort_data()
        assert isinstance(cohort_data, dict), "cohort_data는 딕셔너리여야 함"
        assert 'retention' in cohort_data, "retention 키 필요"
        assert 'revenue' in cohort_data, "revenue 키 필요"
        assert 'avg_spending' in cohort_data, "avg_spending 키 필요"
        assert 'age_group' in cohort_data, "age_group 키 필요"
        assert 'club_status' in cohort_data, "club_status 키 필요"
        assert 'newsletter' in cohort_data, "newsletter 키 필요"

        # 데이터 검증
        for key, df in cohort_data.items():
            assert isinstance(df, pd.DataFrame), f"{key}는 DataFrame이어야 함"
            assert len(df) > 0, f"{key}가 비어있음"
            print(f"  ✓ {key}: {df.shape[0]}행 {df.shape[1]}열")

        print("✅ 코호트 분석 페이지: 통과\n")
        test_results['코호트 분석'] = True

    except Exception as e:
        print(f"❌ 코호트 분석 페이지 에러: {e}\n")
        traceback.print_exc()
        test_results['코호트 분석'] = False

    # =====================================================================
    # 3️⃣ 기존 페이지들 호환성 테스트
    # =====================================================================
    print("[3️⃣] 기존 대시보드 페이지 호환성 테스트")
    print("-" * 80)

    existing_pages = {
        'page_overview': 'Overview',
        'page_customer_segments': 'Customer Segments',
        'page_product_segments': 'Product Segments',
        'page_detailed_analysis': 'Detailed Analysis',
        'page_performance': 'Performance',
        'page_eda': 'EDA'
    }

    for page_name, display_name in existing_pages.items():
        try:
            module = __import__(f'app.pages.{page_name}', fromlist=['show'])
            assert hasattr(module, 'show'), f"{page_name}에 show() 함수 없음"
            print(f"  ✓ {display_name}: show() 함수 확인")
            test_results[display_name] = True
        except Exception as e:
            print(f"  ⚠️ {display_name}: {e}")
            test_results[display_name] = False

    print()

    # =====================================================================
    # 4️⃣ 대시보드 메인 모듈 테스트
    # =====================================================================
    print("[4️⃣] 대시보드 메인 모듈 테스트")
    print("-" * 80)

    try:
        from app.dashboard import main
        print("  ✓ dashboard.main() 함수 확인")
        print("✅ 대시보드 메인: 통과\n")
        test_results['대시보드 메인'] = True
    except Exception as e:
        print(f"❌ 대시보드 메인 에러: {e}\n")
        test_results['대시보드 메인'] = False

    # =====================================================================
    # 5️⃣ 데이터 무결성 테스트
    # =====================================================================
    print("[5️⃣] 데이터 무결성 테스트")
    print("-" * 80)

    try:
        results_dir = PROCESSED_DIR / "analysis"
        required_files = [
            'funnel_lifecycle.csv',
            'funnel_frequency.csv',
            'funnel_category.csv',
            'funnel_channel.csv',
            'cohort_retention.csv',
            'cohort_revenue.csv',
            'cohort_avg_spending.csv',
            'cohort_age_group.csv',
            'cohort_club_status.csv',
            'cohort_newsletter.csv',
            'analysis_results.pkl'
        ]

        for filename in required_files:
            filepath = results_dir / filename
            assert filepath.exists(), f"{filename} 파일 없음"
            print(f"  ✓ {filename}")

        print("✅ 데이터 무결성: 통과\n")
        test_results['데이터 무결성'] = True

    except Exception as e:
        print(f"❌ 데이터 무결성 에러: {e}\n")
        test_results['데이터 무결성'] = False

    # =====================================================================
    # 최종 결과
    # =====================================================================
    print("="*80)
    print("📊 테스트 결과 요약")
    print("="*80 + "\n")

    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)

    for name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"총점: {passed}/{total} ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 모든 테스트 통과! 대시보드를 실행할 준비가 되었습니다.")
        return True
    else:
        print(f"\n⚠️ {total-passed}개의 테스트 실패")
        return False


if __name__ == "__main__":
    success = test_all_pages()
    sys.exit(0 if success else 1)
