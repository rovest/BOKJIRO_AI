# scripts/build_databases.py

import json
import os
from pathlib import Path

# Google Embedding 모델 및 FAISS 벡터 저장소, Document 객체를 사용하기 위해 라이브러리 임포트
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 상수 정의 ---
# 이 스크립트 파일의 위치를 기준으로 기본 경로를 설정합니다.
CURRENT_DIR = Path(__file__).parent
BASE_DIR = CURRENT_DIR.parent
# 원본 데이터 파일 경로
DATA_PATH = BASE_DIR / "data" / "vd_base_v2_refined.json"
# 벡터 DB를 저장할 디렉토리 경로
DB_DIR = BASE_DIR / "db"
FAISS_PATH = str(DB_DIR / "faiss_index")


def create_enriched_content(item_data: dict) -> str:
    """
    메타데이터를 활용하여 검색 품질을 높이기 위한 '의미 보강 텍스트'를 생성합니다.
    이 텍스트는 벡터로 변환될 때 사용되어, 문맥적 의미를 풍부하게 담게 됩니다.
    """
    metadata = item_data.get("metadata", {})
    text = item_data.get("text", "")
    
    # 메타데이터의 주요 필드를 추출합니다. 값이 없는 경우 '정보 없음'으로 처리합니다.
    dae_bulryu = metadata.get('대분류', '정보 없음')
    jung_bulryu = metadata.get('중분류', '정보 없음')
    service_name = metadata.get('사업명', '정보 없음')
    item_name = metadata.get('항목', '정보 없음')

    # 템플릿을 사용하여 자연스러운 문장 형태로 콘텐츠를 재구성합니다.
    # 이를 통해 각 데이터 조각이 어떤 맥락에 속하는지 AI가 더 잘 이해할 수 있습니다.
    enriched_content = (
        f"이 서비스의 대분류는 '{dae_bulryu}', 중분류는 '{jung_bulryu}'이며, 사업명은 '{service_name}'입니다. "
        f"세부 항목 '{item_name}'에 대한 내용은 다음과 같습니다: {text}"
    )
    return enriched_content


def main():
    """데이터베이스 구축 메인 함수"""
    print("데이터베이스 구축을 시작합니다.")

    # 1. 원본 데이터 파일 로드 및 예외 처리
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 데이터 파일을 찾을 수 없습니다 - {DATA_PATH}")
        return
    except json.JSONDecodeError:
        print(f"오류: JSON 파일 파싱에 실패했습니다 - {DATA_PATH}")
        return

    # 2. LangChain이 사용하기 좋은 Document 객체 리스트로 데이터 가공
    documents = []
    print("데이터 전처리를 시작합니다...")
    for item in original_data:
        # 벡터화를 위한 '의미 보강 텍스트'를 생성합니다.
        enriched_content = create_enriched_content(item)
        
        # 원본 메타데이터를 그대로 유지합니다.
        metadata = item.get("metadata", {})
        
        # LangChain의 Document 객체 생성
        # page_content에는 검색 시 의미 비교에 사용될 '의미 보강 텍스트'를 넣습니다.
        # metadata에는 원본 메타데이터와 원본 텍스트를 모두 저장하여,
        # 나중에 검색 결과로 활용할 수 있게 합니다.
        doc = Document(
            page_content=enriched_content, 
            metadata={
                **metadata,  # 원본 메타데이터를 모두 포함
                "original_text": item.get("text", "") # LLM 답변 생성 시 활용할 원본 텍스트 저장
            }
        )
        documents.append(doc)
    
    print(f"총 {len(documents)}개의 문서를 처리합니다.")

    # 3. Google의 최신 임베딩 모델 로드
    print("Google 최신 임베딩 모델(text-embedding-004)을 로드합니다...")
    try:
        # GOOGLE_API_KEY 환경변수를 사용하여 임베딩 모델을 초기화합니다.
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception as e:
        print(f"!!! Google Embedding 모델 로딩 실패: {e}")
        print("!!! .env 파일에 GOOGLE_API_KEY가 올바르게 설정되었는지 확인해주세요.")
        return

    # 4. FAISS 벡터 데이터베이스 생성 및 저장
    # from_documents 함수는 문서 리스트를 받아 임베딩하고 FAISS 인덱스를 생성합니다.
    print("FAISS 벡터 DB 생성을 시작합니다...")
    vector_db = FAISS.from_documents(documents, embeddings)
    
    # 저장할 디렉토리가 없으면 생성합니다.
    DB_DIR.mkdir(exist_ok=True)
    
    # 생성된 벡터 DB를 로컬 파일 시스템에 저장합니다.
    vector_db.save_local(folder_path=FAISS_PATH)
    
    print(f"'{FAISS_PATH}'에 Vector DB가 성공적으로 저장되었습니다.")
    print("\n✅ 모든 데이터베이스 구축이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()


