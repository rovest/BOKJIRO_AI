# 복지로AI - 복지 서비스 챗봇

## 프로젝트 소개
한국의 복지 서비스 정보를 효과적으로 제공하는 RAG 기반 AI 챗봇입니다.

## 주요 기능
- 🤖 Google Gemini 2.0 Flash 기반 자연어 처리
- 🔍 FAISS 벡터 검색을 통한 정확한 복지 정보 검색
- ⚡ Fast Track 키워드 매칭으로 빠른 응답
- 💬 Context-aware 대화 (이전 대화 기록 고려)
- 📊 대분류-중분류-사업명 계층 구조 지원

## 기술 스택
- **Frontend**: Streamlit
- **AI/ML**: LangChain, Google Gemini API, FAISS
- **Language**: Python 3.10+
- **Data**: 46,260개 복지 서비스 정보

## 설치 및 실행

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. API 키 설정
```bash
# .env 파일 생성
GOOGLE_API_KEY=your-google-api-key-here
```

### 3. 실행
```bash
streamlit run streamlit_app.py
```

## 프로젝트 구조
```
├── streamlit_app.py      # 메인 Streamlit 앱
├── app/                  # 핵심 로직
│   ├── chatbot.py       # 챗봇 메인 클래스
│   ├── db_service.py    # 데이터베이스 서비스
│   └── llm_service.py   # LLM 모델 관리
├── data/                # 복지 정보 데이터
└── db/                  # FAISS 벡터 데이터베이스
```

## 데모
[배포된 웹사이트 URL 추가 예정]

## 개발자
- 개발자 정보 추가