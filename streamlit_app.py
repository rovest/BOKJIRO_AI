import streamlit as st
import time
from app.chatbot import WelfareChatbot

# 1. st.set_page_config()는 반드시 한 번만, 가장 먼저 호출되어야 합니다.
# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="복지로AI",
    page_icon="🧭",
    layout="centered"
)

# 2. 여러 CSS 코드를 하나의 st.markdown으로 통합하고, 버튼 테두리 제거 속성을 추가합니다.
# --- CSS 스타일 ---
# --- CSS 스타일 ---
st.markdown("""
<style>
/* --- 사이드바 스타일 --- */
/* 토글 스위치 라벨 스타일 지정 */
div[data-testid="stSidebar"] div[data-testid="stToggle"] label {
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* 사이드바의 버튼, 마크다운 텍스트의 스타일을 통일합니다 */
div[data-testid="stSidebar"] button,
div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 15px !important;
    font-weight: 600 !important;
    text-align: left !important;
}

/* [수정] 버튼을 더 구체적으로 선택하여 테두리와 배경을 확실히 제거합니다 */
div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    border: none !important;
    background-color: transparent !important;
    padding: 0 !important; /* 모든 내부 여백 제거 */
    margin: 0.25rem 0; /* 상하 마진 추가로 간격 조정 */
}

/* --- 메인 콘텐츠 스타일 --- */
/* (기존 메인 콘텐츠 스타일은 그대로 유지) */
.subtitle {
    text-align: center;
    font-size: 2.0em;
    color: #4A4A4A;
    margin-top: 0rem !important;
    margin-bottom: 2.5em;
    line-height: 1.6;
}
div[data-testid="stChatInput"] > div {
    border: 1px solid transparent !important;
    border-radius: 1rem;
}
div[data-testid="stChatInput"] textarea {
    padding: 1rem !important;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# --- 페이지 상단 제목 및 설명 ---

st.markdown(
    '<p style="text-align: center; font-weight: bold; margin-bottom: 0rem; padding-bottom: 0rem;"><span style="color: #4A4A4A; font-size: 2.3rem;">나에게 힘이되는 </span><span style="color: #1E90FF; font-size: 2.9rem;">복지로AI</span></p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align: center; font-size: 1.1em; color: #777777; line-height: 1.3; margin-bottom: 2.5em;">수백 가지 복잡한 복지 혜택의 미로, 더 이상 헤매지 마세요.<br>어떤 상황에 처해있든, <strong>당신의 AI 비서</strong>가 가장 밝은 길을 비춰드립니다.</p>',
    unsafe_allow_html=True
)

# --- 챗봇 클래스 로딩 (캐시 사용) ---
@st.cache_resource
def load_chatbot_instance(llm_name):
    """선택된 LLM에 맞춰 챗봇 인스턴스를 로드합니다."""
    print(f"DEBUG: '{llm_name}' 모델로 챗봇 인스턴스를 새로 로드합니다.")
    return WelfareChatbot(user_id="streamlit_user", llm_choice=llm_name)

# --- 헬퍼 함수 ---
def get_initial_message():
    """초기 인사 메시지를 반환하는 함수"""
    return [{
        "role": "assistant",
        "content": """
        <div style="display: flex; align-items: flex-start; gap: 12px;">
          <span style="font-size: 1.8em; line-height: 1.3;">💡</span>
          <div style="flex: 1; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
            <strong>어떤 도움이 필요하신가요?</strong><br>
            복지서비스가 필요한 분의 <strong>상황</strong>과 원하시는 <strong>지원 내용</strong>을 함께 알려주세요.<br><br>
            <small style="color: #555;">
              <strong>예시)</strong><br>
              • <strong>상황:</strong> 30대, 아이 둘을 키우는 한부모가정<br>
              • <strong>지원 내용:</strong> 저렴한 주거 공간과 양육비 지원
            </small>
          </div>
        </div>
        """
    }]

# --- 세션 상태 초기화 ---
def initialize_session_state():
    """웹 페이지가 처음 로드되거나 새로고침될 때 세션 상태를 초기화합니다."""
    if "messages" not in st.session_state:
        st.session_state.messages = get_initial_message()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm" not in st.session_state:
        st.session_state.llm = "gemini"
    if "dialogue_mode" not in st.session_state:
        st.session_state.dialogue_mode = "NORMAL"
    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions = []
    if "llm_selector_visible" not in st.session_state:
        st.session_state.llm_selector_visible = False

initialize_session_state()

# --- 사이드바 ---
with st.sidebar:
    st.toggle("LLM 엔진 변경", key="llm_selector_visible")

    if st.session_state.llm_selector_visible:
        llm_options = ["gemini", "gemma", "exaone"]
        current_llm_index = llm_options.index(st.session_state.llm)
        selected_llm = st.selectbox(
            "엔진 선택:",
            options=llm_options,
            index=current_llm_index,
            key='llm_selector'
        )
    else:
        selected_llm = st.session_state.llm

    # [수정] "새 대화 시작" 앞에 아이콘 추가
    if st.button("➕ 새 대화 시작"):
        if len(st.session_state.messages) > 1:
            chat_title = st.session_state.messages[1]['content'][:30] + "..."
            st.session_state.chat_history.insert(0, {"title": chat_title, "messages": st.session_state.messages})
        st.session_state.messages = get_initial_message()
        st.session_state.dialogue_mode = "NORMAL"
        st.session_state.asked_questions = []
        st.rerun()

    # [수정] "대화 기록" 앞에 아이콘 추가
    st.markdown("📜 대화 기록")

    for i, chat in enumerate(st.session_state.chat_history):
        if st.button(chat["title"], key=f"history_{i}"):
            st.session_state.messages = chat["messages"]
            st.rerun()

# --- 모델 변경 및 챗봇 로드 ---
# 모델 변경 시 캐시된 챗봇 인스턴스 및 세션 상태 초기화
if st.session_state.llm != selected_llm:
    st.session_state.llm = selected_llm
    load_chatbot_instance.clear()
    st.session_state.messages = [{"role": "assistant", "content": f"AI 엔진을 '{selected_llm}'(으)로 변경했습니다. 무엇을 도와드릴까요?"}]
    st.session_state.dialogue_mode = "NORMAL"
    st.session_state.asked_questions = []

chatbot = load_chatbot_instance(st.session_state.llm)

# --- 채팅 기록 표시 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- 사용자 입력 및 대화 로직 ---
if prompt := st.chat_input("..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("AI가 분석 중입니다..."):
            response_content, _ = chatbot.chat(st.session_state)
            st.markdown(response_content, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response_content})