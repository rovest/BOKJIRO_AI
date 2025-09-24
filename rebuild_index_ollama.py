#!/usr/bin/env python3
"""
Ollama nomic-embed-text 임베딩으로 FAISS 인덱스를 재생성하는 스크립트
"""
import json
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.ollama_embeddings import get_ollama_embeddings

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_data(data_path: Path):
    """JSON 데이터 파일 로드"""
    logging.info(f"JSON 데이터 파일을 로드합니다: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_documents_from_json(json_data, max_docs=None):
    """JSON 데이터를 LangChain Document 객체로 변환 (테스트용 문서 수 제한 가능)"""
    documents = []
    data_to_process = json_data[:max_docs] if max_docs else json_data
    logging.info(f"총 {len(data_to_process)}개 항목을 Document 객체로 변환 중...")

    for i, item in enumerate(data_to_process):
        if i % 1000 == 0:
            logging.info(f"진행률: {i}/{len(data_to_process)} ({i/len(data_to_process)*100:.1f}%)")

        doc = Document(
            page_content=item.get('text', ''),
            metadata=item.get('metadata', {})
        )
        documents.append(doc)

    logging.info(f"총 {len(documents)}개 Document 생성 완료")
    return documents

def build_faiss_index(documents, embeddings, output_path: Path):
    """FAISS 인덱스 생성 및 저장"""
    logging.info("FAISS 벡터 인덱스를 생성합니다...")

    # 작은 배치로 처리하여 안정성 확보
    batch_size = 100
    vector_db = None

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        logging.info(f"배치 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} 처리 중... ({len(batch)}개 문서)")

        try:
            if vector_db is None:
                vector_db = FAISS.from_documents(batch, embeddings)
            else:
                batch_db = FAISS.from_documents(batch, embeddings)
                vector_db.merge_from(batch_db)
        except Exception as e:
            logging.error(f"배치 처리 중 오류 발생: {e}")
            continue

    # 인덱스 저장
    logging.info(f"FAISS 인덱스를 저장합니다: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(output_path))

    logging.info("FAISS 인덱스 생성 및 저장 완료!")
    return vector_db

def main():
    # 경로 설정
    base_dir = Path(__file__).parent
    data_file = base_dir / "data" / "vd_base_v2_refined.json"
    output_dir = base_dir / "db" / "faiss_index"

    if not data_file.exists():
        logging.error(f"데이터 파일을 찾을 수 없습니다: {data_file}")
        return

    try:
        # Ollama 임베딩 모델 로드
        logging.info("Ollama nomic-embed-text 임베딩 모델을 로드합니다...")
        embeddings = get_ollama_embeddings()
        logging.info("Ollama 임베딩 모델 로드 완료")

        # 테스트를 위해 처음 5000개 문서만 처리
        logging.info("⚠️ 테스트 모드: 처음 5000개 문서만 처리합니다.")

        # 데이터 로드
        json_data = load_json_data(data_file)

        # Document 객체 생성 (테스트용으로 5000개 제한)
        documents = create_documents_from_json(json_data, max_docs=5000)

        # FAISS 인덱스 생성
        build_faiss_index(documents, embeddings, output_dir)

        logging.info("✅ Ollama 기반 FAISS 인덱스 생성이 완료되었습니다!")
        logging.info(f"📁 저장 위치: {output_dir}")
        logging.info("🔄 전체 데이터 처리를 원하시면 max_docs=None으로 수정하세요.")

    except Exception as e:
        logging.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()