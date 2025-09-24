#!/usr/bin/env python3
"""
BGE-M3 임베딩으로 FAISS 인덱스를 재생성하는 스크립트
"""
import json
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.local_embeddings import get_local_embeddings

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_data(data_path: Path):
    """JSON 데이터 파일 로드"""
    logging.info(f"JSON 데이터 파일을 로드합니다: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_documents_from_json(json_data):
    """JSON 데이터를 LangChain Document 객체로 변환"""
    documents = []
    logging.info(f"총 {len(json_data)}개 항목을 Document 객체로 변환 중...")

    for i, item in enumerate(json_data):
        if i % 5000 == 0:
            logging.info(f"진행률: {i}/{len(json_data)} ({i/len(json_data)*100:.1f}%)")

        doc = Document(
            page_content=item.get('text', ''),
            metadata=item.get('metadata', {})
        )
        documents.append(doc)

    logging.info(f"총 {len(documents)}개 Document 생성 완료")
    return documents

def build_faiss_index(documents, embeddings, output_path: Path):
    """FAISS 인덱스 생성 및 저장"""
    logging.info("FAISS 벡터 인덱스를 생성합니다... (시간이 다소 소요될 수 있습니다)")

    # 배치 처리로 메모리 효율성 개선
    batch_size = 1000
    vector_db = None

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        logging.info(f"배치 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} 처리 중... ({len(batch)}개 문서)")

        if vector_db is None:
            vector_db = FAISS.from_documents(batch, embeddings)
        else:
            batch_db = FAISS.from_documents(batch, embeddings)
            vector_db.merge_from(batch_db)

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
    output_dir = base_dir / "db" / "faiss_index_bge"

    if not data_file.exists():
        logging.error(f"데이터 파일을 찾을 수 없습니다: {data_file}")
        return

    try:
        # BGE-M3 임베딩 모델 로드
        logging.info("BGE-M3 임베딩 모델을 로드합니다...")
        embeddings = get_local_embeddings()
        logging.info("BGE-M3 임베딩 모델 로드 완료")

        # 데이터 로드
        json_data = load_json_data(data_file)

        # Document 객체 생성
        documents = create_documents_from_json(json_data)

        # FAISS 인덱스 생성
        build_faiss_index(documents, embeddings, output_dir)

        logging.info("✅ BGE-M3 기반 FAISS 인덱스 재생성이 완료되었습니다!")
        logging.info(f"📁 저장 위치: {output_dir}")

    except Exception as e:
        logging.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()