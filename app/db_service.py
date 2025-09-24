# app/db_service.py

import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict, OrderedDict

# .env 파일에서 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# LangChain 관련 라이브러리 임포트
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .local_embeddings import get_local_embeddings  # BGE-M3 활성화
from .ollama_embeddings import get_ollama_embeddings


# --- 상수 정의 ---
try:
    CURRENT_DIR = Path(__file__).parent
    BASE_DIR = CURRENT_DIR.parent
    DB_DIR = BASE_DIR / "db"
    # BGE-M3 전용 인덱스 경로
    FAISS_PATH = str(DB_DIR / "faiss_index_bge")
    if not (DB_DIR / "faiss_index_bge").exists():
        # 기존 Ollama 인덱스가 있으면 사용
        if (DB_DIR / "faiss_index").exists():
            FAISS_PATH = str(DB_DIR / "faiss_index")
        else:
            raise FileNotFoundError("FAISS index directory not found at any path.")
except NameError:
    BASE_DIR = Path.cwd()
    DB_DIR = BASE_DIR / "db"
    FAISS_PATH = str(DB_DIR / "faiss_index_bge")


class DBService:
    """
    FAISS 벡터 데이터베이스와 상호작용하며,
    목차 기반 검색을 핵심 전략으로 사용하는 서비스 클래스.
    """
    def __init__(self, faiss_path=FAISS_PATH, embedding_type="ollama"):
        logging.info("DEBUG: DBService 인스턴스 초기화 시작...")

        try:
            if embedding_type == "ollama":
                # Ollama 임베딩 모델 사용 (nomic-embed-text)
                logging.info("Ollama nomic-embed-text 임베딩 모델을 로딩합니다...")
                self.embeddings = get_ollama_embeddings()
                logging.info("DEBUG: Ollama 임베딩 모델 로딩 성공.")
            elif embedding_type == "bge":
                # BGE-M3 로컬 임베딩 모델 사용
                logging.info("로컬 BGE-M3 임베딩 모델을 로딩합니다...")
                self.embeddings = get_local_embeddings()
                logging.info("DEBUG: BGE-M3 로컬 임베딩 모델 로딩 성공.")
            else:
                # Google 임베딩 모델 사용 (기존 방식)
                self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                logging.info("DEBUG: Google 최신 임베딩 모델 로딩 성공.")
        except Exception as e:
            logging.error(f"!!! 임베딩 모델 초기화 실패: {e}")
            if embedding_type == "ollama":
                raise ConnectionError(
                    f"Ollama 임베딩 모델 초기화에 실패했습니다.\n"
                    f"Ollama 서버가 실행 중인지, nomic-embed-text 모델이 설치되었는지 확인하세요.\n"
                    f"상세 오류: {e}"
                )
            elif embedding_type == "bge":
                raise ConnectionError(
                    f"BGE-M3 로컬 임베딩 모델 초기화에 실패했습니다.\n"
                    f"다음 패키지가 설치되었는지 확인하세요: sentence-transformers, FlagEmbedding\n"
                    f"상세 오류: {e}"
                )
            else:
                raise ConnectionError(
                    f"Google Embedding 모델 초기화에 실패했습니다. 다음을 확인하세요:\n"
                    f"- 'GOOGLE_API_KEY' 환경 변수가 .env 파일에 올바르게 설정되었는지.\n"
                    f"- Google Cloud 프로젝트에서 'Generative Language API'가 활성화되었는지.\n"
                    f"상세 오류: {e}"
                )

        try:
            absolute_faiss_path = str(Path(faiss_path).resolve())
            
            # FAISS 벡터 DB 로드 (동기 방식으로 안정성 개선)
            self.vector_db = FAISS.load_local(
                absolute_faiss_path, self.embeddings, allow_dangerous_deserialization=True
            )
            
            self.all_docs = [self.vector_db.docstore.search(doc_id) 
                             for doc_id in self.vector_db.index_to_docstore_id.values() 
                             if self.vector_db.docstore.search(doc_id) is not None]
            
            # --- ✨ [핵심 복원] 목차 검색을 위한 별도 DB 생성 ---
            self.toc_docs = [
                doc for doc in self.all_docs 
                if doc.metadata.get('중분류') == '목차' and doc.metadata.get('항목') == '세부목차'
            ]
            # 목차 전용 DB 생성 (동기 방식으로 안정성 개선)
            self.toc_db = FAISS.from_documents(self.toc_docs, self.embeddings) if self.toc_docs else None
            logging.info(f"DEBUG: FAISS 벡터 DB 로드 성공. 전체 {len(self.all_docs)}개 문서, 목차 {len(self.toc_docs)}개 항목.")
            
        except Exception as e:
            logging.error(f"!!! FAISS 벡터 DB 로드 실패: {e}")
            raise FileNotFoundError(
                f"FAISS 인덱스 파일을 로드하는 데 실패했습니다: {faiss_path}\n"
                f"FAISS 파일이 존재하고 손상되지 않았는지, 그리고 임베딩 모델과 호환되는지 확인해주세요.\n"
                f"상세 오류: {e}"
            )

    def get_schema_context(self) -> Dict[str, any]:
        """
        [✨ 개선안] LLM의 검색 설계를 돕기 위해 '대분류-중분류' 전체 계층 구조를 포함한 컨텍스트를 제공합니다.
        """
        logging.debug("DEBUG: DB의 전체 '대분류-중분류' 계층 구조 컨텍스트 추출 중...")
        if not self.all_docs:
            return {'context_string': '', 'service_names': []}
        
        all_meta = [doc.metadata for doc in self.all_docs if doc and doc.metadata]
        
        # 전체 사업명 목록 추출
        service_names = sorted(list(set(m.get('사업명') for m in all_meta if m.get('사업명'))))
        
        category_hierarchy = OrderedDict()

        for meta in all_meta:
            major_cat = meta.get('대분류')
            minor_cat = meta.get('중분류')
            
            if major_cat and minor_cat:
                if major_cat not in category_hierarchy:
                    category_hierarchy[major_cat] = set()
                category_hierarchy[major_cat].add(minor_cat)

        context_parts = ["# [전체 카테고리 목록]"]
        for major, minors in category_hierarchy.items():
            context_parts.append(f"## {major}")
            for minor in sorted(list(minors)):
                context_parts.append(f"- {minor}")
            context_parts.append("")
        
        context_string = "\n".join(context_parts)
        
        logging.debug("LLM에 전달될 카테고리 계층 구조 컨텍스트:\n" + context_string)

        return {'context_string': context_string, 'service_names': service_names}
    
    def _search_by_metadata_filters(self, filters: Dict) -> List[Document]:
        """
        [내부 헬퍼] metadata_filters의 여러 '중분류' 조건과 일치하는 모든 문서를 반환합니다.
        """
        if not filters or '중분류' not in filters or not filters['중분류']:
            return self.all_docs

        target_categories = set(filters['중분류'])
        logging.debug(f"DEBUG: 메타데이터 필터링 시작 (대상 중분류: {target_categories})")
        
        matched_docs = [
            doc for doc in self.all_docs
            if doc.metadata.get('중분류') in target_categories
        ]
        
        logging.debug(f"DEBUG: 메타데이터 필터링 결과 {len(matched_docs)}개 문서 발견.")
        return matched_docs

    def advanced_search(self, filters: Dict, keywords: List[str], k: int = 15) -> List[Document]:
        """
        [새로운 핵심 검색 함수] 메타데이터로 1차 필터링 후, 키워드로 2차 정밀 검색을 수행합니다.
        """
        logging.debug(f"고급 검색 시작 (필터: {filters}, 키워드: {keywords})")

        primary_docs = self._search_by_metadata_filters(filters)

        if not primary_docs:
            logging.warning("메타데이터 필터링 결과, 검색할 문서가 없습니다.")
            return []

        logging.debug(f"메타데이터 필터링으로 검색 범위가 {len(primary_docs)}개 문서로 좁혀졌습니다.")

        search_query = " ".join(keywords)
        
        # 임시 DB 생성 (동기 방식으로 안정성 개선)
        temp_db = FAISS.from_documents(primary_docs, self.embeddings)
        
        # 유사도 검색 (동기 방식으로 안정성 개선)
        final_docs = temp_db.similarity_search(query=search_query, k=k)


        logging.debug("\n" + "="*50)
        logging.debug(f"🕵️  [DB_SERVICE] 최종 검색 결과 (상위 {len(final_docs)}개)")
        logging.debug(f"   - 적용된 필터(중분류): {filters.get('중분류', 'N/A')}")
        logging.debug(f"   - 적용된 키워드: {keywords}")
        logging.debug("="*50)
        for i, doc in enumerate(final_docs):
            minor_category = doc.metadata.get('중분류', 'N/A')
            service_name = doc.metadata.get('사업명', 'N/A')
            logging.debug(f"{i+1:02d}. 중분류: {minor_category:<40} | 사업명: {service_name}")
        logging.debug("="*50 + "\n")

        return final_docs
    
    def metadata_search(self, filter_dict: Dict) -> List[Document]:
        """특정 메타데이터 조건과 일치하는 모든 문서를 반환합니다."""
        logging.debug(f"DEBUG: 메타데이터 검색 시작 (필터: {filter_dict})")
        matched_docs = []
        for doc in self.all_docs:
            if not (doc and doc.metadata): continue
            
            is_match = True
            for key, value in filter_dict.items():
                metadata_value = doc.metadata.get(key)
                if metadata_value is None or not str(metadata_value).startswith(str(value)):
                    is_match = False
                    break
            if is_match:
                matched_docs.append(doc)
        
        logging.debug(f"DEBUG: 메타데이터 검색 결과 {len(matched_docs)}개 문서 발견.")
        return matched_docs

    def __del__(self):
        pass
