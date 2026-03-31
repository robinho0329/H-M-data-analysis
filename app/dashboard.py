"""
H&M LSTM 매출 예측 대시보드 메인
"""
import streamlit as st
import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _logo(size=26, color="#CC0000", ls="-1px"):
    """H&M 로고 - SVG 없이 styled b 태그 사용 (Streamlit 완전 호환)"""
    return (
        f'<b style="font-family:\'Arial Black\',Arial,sans-serif;'
        f'font-size:{size}px;font-weight:900;color:{color};'
        f'letter-spacing:{ls};line-height:1;display:inline-block;">H&amp;M</b>'
    )


def main():
    st.set_page_config(
        page_title="H&M Sales Analytics",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
<style>
html, body, [class*="css"], .stApp {
    font-family: 'Helvetica Neue LT','Helvetica Neue',Helvetica,Arial,sans-serif !important;
    background-color: #FFFFFF !important;
    color: #000000 !important;
}
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E8E8E8 !important;
    padding-top: 0 !important;
    min-width: 240px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * {
    color: #000000 !important;
    font-family: 'Helvetica Neue LT','Helvetica Neue',Helvetica,Arial,sans-serif !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #555555 !important; font-size: 12px !important; }
[data-testid="stSidebar"] hr { border-color: #F0F0F0 !important; margin: 0 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap: 0 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #000000 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    padding: 13px 20px !important;
    border-bottom: 1px solid #F0F0F0 !important;
    border-radius: 0 !important;
    width: 100% !important;
    transition: background 0.15s !important;
    background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #F7F7F7 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] div:has(input:checked) label {
    color: #000000 !important;
    font-weight: 700 !important;
    background: #F5F5F5 !important;
    border-left: 3px solid #000000 !important;
    padding-left: 17px !important;
}
.main .block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1280px !important;
}
.main > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
h1 {
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    color: #000000 !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
    border: none !important;
    padding-left: 0 !important;
    margin-bottom: 0.5rem !important;
}
h2 {
    font-weight: 700 !important;
    color: #000000 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
}
h3 { font-weight: 600 !important; color: #333 !important; font-size: 0.95rem !important; }
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E8E8E8 !important;
    border-top: 2px solid #E50010 !important;
    border-radius: 0 !important;
    padding: 16px 20px !important;
    box-shadow: none !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: #888888 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #000000 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: 1px solid #E8E8E8 !important;
    gap: 0 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #767676 !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    padding: 12px 16px !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #000000 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {
    background: transparent !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #000000 !important;
}
.stButton > button {
    background: #000000 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #E50010 !important; }
.stAlert { border-radius: 0 !important; font-size: 13px !important; }
[data-testid="stSuccess"] { background: #F9F9F9 !important; border-left: 3px solid #E50010 !important; border-radius: 0 !important; }
[data-testid="stInfo"] { background: #F9F9F9 !important; border-left: 3px solid #000000 !important; border-radius: 0 !important; }
[data-testid="stWarning"] { border-left: 3px solid #E8A000 !important; border-radius: 0 !important; }
[data-testid="stDataFrame"] th {
    background: #000000 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stExpander"] { border: 1px solid #E8E8E8 !important; border-radius: 0 !important; }
[data-testid="stProgress"] > div > div { background: #000000 !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { border-radius: 0 !important; border-color: #E8E8E8 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #F5F5F5; }
::-webkit-scrollbar-thumb { background: #CCCCCC; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #000000; }
hr { border-color: #E8E8E8 !important; margin: 1rem 0 !important; }
footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"], .stAppHeader { display: none !important; }
div#hm-promo {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 1000100 !important;
    margin: 0 !important;
    width: 100vw !important;
    box-sizing: border-box !important;
}
.main .block-container {
    padding-top: 36px !important;
}
</style>
""", unsafe_allow_html=True)

    # ── SIDEBAR ──
    st.sidebar.markdown(
        '<div style="background:#FFFFFF;padding:42px 20px 14px 20px;border-bottom:1px solid #E8E8E8;">'
        + _logo(26) +
        '<div style="font-size:9px;color:#AAAAAA;letter-spacing:2.5px;'
        'text-transform:uppercase;margin-top:5px;font-weight:600;">SALES ANALYTICS</div>'
        '</div>',
        unsafe_allow_html=True
    )

    try:
        from app.utils.cache_manager import load_raw_data

        @st.cache_data(ttl=3600)
        def _get_sidebar_summary():
            _data = load_raw_data()
            if _data is None:
                return None
            return {
                'customers': int(_data['customer_id'].nunique()),
                'transactions': int(len(_data)),
                'start': _data['t_dat'].min().strftime('%Y-%m-%d'),
                'end': _data['t_dat'].max().strftime('%Y-%m-%d'),
            }

        _summary = _get_sidebar_summary()
        if _summary:
            st.sidebar.markdown(
                '<div style="padding:12px 20px;border-bottom:1px solid #F0F0F0;">'
                '<div style="font-size:9px;font-weight:700;color:#E50010;letter-spacing:1.5px;'
                'text-transform:uppercase;margin-bottom:8px;">DATASET OVERVIEW</div>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">'
                '<div style="text-align:center;">'
                f'<div style="font-size:15px;font-weight:800;color:#000;">{_summary["customers"]//1000}K</div>'
                '<div style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:0.8px;">Customers</div>'
                '</div>'
                '<div style="text-align:center;">'
                f'<div style="font-size:15px;font-weight:800;color:#000;">{_summary["transactions"]//10000}만</div>'
                '<div style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:0.8px;">Transactions</div>'
                '</div>'
                '</div>'
                f'<div style="font-size:10px;color:#999;text-align:center;margin-top:8px;letter-spacing:0.2px;">'
                f'{_summary["start"]} — {_summary["end"]}</div>'
                '</div>',
                unsafe_allow_html=True
            )
    except Exception:
        pass

    st.sidebar.markdown(
        '<div style="padding:10px 20px 4px 20px;">'
        '<div style="font-size:9px;font-weight:700;color:#BBBBBB;letter-spacing:2px;text-transform:uppercase;">MENU</div>'
        '</div>',
        unsafe_allow_html=True
    )

    pages = {
        "비즈니스 요약": "business_summary",
        "매출 예측": "overview",
        "고객 분석": "customer_segments",
        "상품 분석": "product_segments",
        "상세 분석": "detailed_analysis",
        "데이터 탐색": "eda",
        "퍼널 분석": "funnel_analysis",
        "코호트 분석": "cohort_analysis",
        "모델 성능": "performance",
    }

    selected_page = st.sidebar.radio("", list(pages.keys()), label_visibility="collapsed")

    st.sidebar.markdown(
        '<div style="padding:12px 20px;border-top:1px solid #E8E8E8;margin-top:8px;">'
        '<div style="font-size:9px;color:#CCCCCC;letter-spacing:1px;text-transform:uppercase;text-align:center;line-height:1.8;">'
        'H&amp;M Group · 2019 Fashion Data<br>LSTM Forecasting Model</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── MAIN HEADER ──
    st.markdown(
        '<div id="hm-promo" style="background:#FFFFFF;border-bottom:1px solid #EEEEEE;padding:8px 24px;'
        'margin:-1rem -2rem 0 -2rem;display:flex;align-items:center;justify-content:space-between;">'
        '<span style="font-size:11px;color:#E50010;font-weight:600;letter-spacing:0.2px;">'
        '미드 시즌 세일 시작! 최대 50% OFF</span>'
        '<span style="color:#E50010;font-size:16px;cursor:pointer;font-weight:300;line-height:1;">+</span>'
        '</div>',
        unsafe_allow_html=True
    )

    nav_labels = ["예측", "고객", "상품", "분석"]

    nav_items_html = ""
    for nav_label in nav_labels:
        nav_items_html += (
            '<span style="display:inline-flex;align-items:center;height:100%;padding:0 18px;'
            'font-size:13px;font-weight:500;'
            'letter-spacing:0.3px;'
            'color:#000000;white-space:nowrap;'
            'border-bottom:2px solid transparent;'
            f'box-sizing:border-box;">{nav_label}</span>'
        )

    st.markdown(
        '<div style="background:#FFFFFF;border-bottom:1px solid #E8E8E8;padding:0;margin:0 -2rem;">'
        '<div style="display:flex;align-items:center;padding:0 32px;height:60px;">'
        '<div style="display:flex;align-items:stretch;height:100%;">'
        '<div style="display:flex;align-items:center;margin-right:28px;flex-shrink:0;">'
        + _logo(26) +
        '</div>'
        f'<nav style="display:flex;align-items:stretch;height:100%;">{nav_items_html}</nav>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # ── PAGE CONTENT ──
    if selected_page == "비즈니스 요약":
        from pages.page_business_summary import show
        show()
    elif selected_page == "매출 예측":
        from pages.page_overview import show
        show()
    elif selected_page == "고객 분석":
        from pages.page_customer_segments import show
        show()
    elif selected_page == "상품 분석":
        from pages.page_product_segments import show
        show()
    elif selected_page == "상세 분석":
        from pages.page_detailed_analysis import show
        show()
    elif selected_page == "데이터 탐색":
        from pages.page_eda import show
        show()
    elif selected_page == "퍼널 분석":
        from pages.page_funnel_analysis import show
        show()
    elif selected_page == "코호트 분석":
        from pages.page_cohort_analysis import show
        show()
    elif selected_page == "모델 성능":
        from pages.page_performance import show
        show()

    # ── FOOTER ──
    st.markdown(
        '<div style="margin-top:40px;border-top:1px solid #E8E8E8;padding:20px 0 14px 0;text-align:center;">'
        + _logo(20) +
        '<div style="font-size:10px;color:#AAAAAA;letter-spacing:2px;text-transform:uppercase;'
        'font-weight:600;margin-top:8px;">HENNES &amp; MAURITZ · SALES ANALYTICS</div>'
        '<div style="font-size:10px;color:#CCCCCC;margin-top:4px;letter-spacing:0.5px;">'
        'Powered by TensorFlow · Streamlit · 2019 Fashion Retail Dataset</div>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
