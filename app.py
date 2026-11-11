import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 페이지 기본 설정
st.set_page_config(
    page_title="교통사고 사망자 수 예측 및 안전 모니터",
    page_icon="🚦",
    layout="centered"
)

# 1. 모델 로드 함수 (캐싱 적용)
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# 모델 파일 로드
MODEL_PATH = "traffic_casualty_model.pkcls"

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"모델 파일('{MODEL_PATH}')을 찾을 수 없습니다. 동일한 디렉토리에 파일이 있는지 확인해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"모델을 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# 앱 타이틀
st.title("🚦 교통사고 예측 및 어린이 안전 모니터")
st.caption("시도별 교통사고 데이터 기반 사망자 수 예측 웹 애플리케이션")
st.markdown("---")

# 2. 사용자 입력 (사이드바 또는 메인 화면)
st.sidebar.header("📊 입력 조건 설정")

# 시도 목록 (Orange3 모델에 학습된 범주 입력)
sido_list = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", 
    "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", 
    "경기도", "강원특별자치도", "충청북도", "충청남도", 
    "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"
]

selected_sido = st.sidebar.selectbox("시·도 선택", sido_list)
accident_count = st.sidebar.slider("사고 건수", min_value=0, max_value=500, value=50, step=1)
minor_injured = st.sidebar.slider("경상자 수", min_value=0, max_value=500, value=30, step=1)

# 모델 예측 실행
# Orange3 모델 구조에 맞는 DataFrame 생성
input_df = pd.DataFrame([{
    "시도": selected_sido,
    "사고건수": accident_count,
    "경상자수": minor_injured
}])

try:
    # Orange3 Model 객체인 경우 predict() 사용
    prediction = model(input_df)
    predicted_deaths = float(prediction[0])
except Exception:
    try:
        # standard sklearn API fallback
        prediction = model.predict(input_df)
        predicted_deaths = float(prediction[0])
    except Exception as e:
        st.error(f"예측 도중 오류가 발생했습니다: {e}")
        predicted_deaths = 0.0

# 예측 결과 보정 (음수 방지 및 반올림)
predicted_deaths = max(0.0, predicted_deaths)

# 3. 실시간 결과 및 신호등 UI
st.subheader("🔮 실시간 예측 결과")
col1, col2 = st.columns([1, 2])

with col1:
    st.metric(label="예상 사망자 수", value=f"{predicted_deaths:.2f} 명")

# 기준에 따른 위험도 구분 (예상 사망자 수 기준)
if predicted_deaths < 1.0:
    status_color = "🟢"
    status_text = "안전"
    card_bg = "#E8F5E9"
    border_color = "#2E7D32"
elif predicted_deaths < 3.0:
    status_color = "🟡"
    status_text = "주의"
    card_bg = "#FFFDE7"
    border_color = "#FBC02D"
else:
    status_color = "🔴"
    status_text = "위험"
    card_bg = "#FFEBEE"
    border_color = "#C62828"

with col2:
    # 대형 신호등 표시
    st.markdown(
        f"""
        <div style="
            background-color: {card_bg}; 
            border: 2px solid {border_color}; 
            border-radius: 12px; 
            padding: 15px; 
            text-align: center;
        ">
            <h2 style="margin:0; font-size: 2.5rem;">{status_color} {status_text}</h2>
            <p style="margin:5px 0 0 0; color: #555;">현재 설정된 조건의 위험도 수준입니다.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown("---")

# 4. 어린이 보호구역 안전 수칙 카드
st.subheader("🎒 어린이 보호구역 안전 수칙")

st.info("""
**🚗 운전자 준수 사항**
1. **시속 30km 이하 서행**: 어린이 보호구역 내에서는 무조건 서행하세요.
2. **주정차 금지**: 어린이 보호구역 내 주정차는 법적으로 엄격히 금지됩니다.
3. **횡단보도 앞 일단정지**: 신호등이 없는 횡단보도에서도 일단 정지 후 출발하세요.

**🚶 어린이/보행자 지도 사항**
1. **서다·보다·걷다 3원칙**: 길을 건널 땐 먼저 멈추고, 차를 확인한 뒤 건너세요.
2. **급제동/갑자기 뛰어들기 금지**: 차량 운전자의 시야 확보를 위해 갑자기 튀어나가지 않도록 주의합니다.
""")
