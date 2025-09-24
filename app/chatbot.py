# app/chatbot.py
import json
import re
from thefuzz import fuzz
from .llm_service import get_llm
from .db_service import DBService
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document

class WelfareChatbot:
    def __init__(self, user_id, llm_choice="gemini"):
        self.user_id = user_id
        self.db_service = DBService()
        self.llm = get_llm(llm_choice)
        self.schema_context_str = None
        self.service_names_list = []
        self._prepare_chatbot_data()

    def _prepare_chatbot_data(self):
        context_data = self.db_service.get_schema_context() #"""DB의 구조(카테고리 계층)와 사업명 목록을 미리 준비합니다."""
        self.schema_context_str = context_data.get('context_string', '') #  이제 'context_string'에 대분류-중분류 계층 정보가 모두 담겨 있습니다.
        self.service_names_list = context_data.get('service_names', [])
        print("DEBUG: LLM에 전달될 DB 카테고리 계층 및 사업명 컨텍스트가 준비되었습니다.")

    def _create_chain(self, template, parser):
        """PromptTemplate, LLM, OutputParser를 연결한 체인을 생성합니다."""
        return PromptTemplate.from_template(template) | self.llm | parser

    def chat(self, session_state):
        """사용자 메시지를 받아 지능형 RAG 파이프라인을 실행하고 답변을 반환합니다."""
        messages = session_state.get('messages', [])
        user_message = messages[-1]['content'].strip()
        chat_history = self._format_chat_history(messages)

        try:
            return self._get_intelligent_response(user_message, chat_history)
        except Exception as e:
            import traceback
            print(f"!!!!!!!!!!!! 최상위 오류 발생 in chat processing: {e} !!!!!!!!!!!!")
            traceback.print_exc()
            return "죄송합니다, 답변을 생성하는 중 예상치 못한 오류가 발생했습니다.", "NORMAL"

    def _format_chat_history(self, messages):
        """세션의 메시지 기록을 LLM 컨텍스트에 넣을 문자열로 변환합니다."""
        if len(messages) <= 1:
            return "이전 대화 기록이 없습니다."
        history = []
        for msg in messages[-6:-1]:
            role = "사용자" if msg["role"] == "user" else "지니(AI)"
            history.append(f"{role}: {msg['content']}")
        return "\n".join(history)

    def _detect_fast_track_keyword(self, user_message: str) -> str | None:
        """
        [역할 변경] 사용자 질문에서 특정 사업명을 '탐지'하여 그 이름을 반환합니다.
        """
        if not self.service_names_list: return None
        
        normalized_message = user_message.replace(" ", "")
        
        # 1단계: 정확성 우선 검사
        for service_name in self.service_names_list:
            if not service_name: continue
            normalized_service_name = service_name.replace(" ", "")
            
            # ✅ [논리 오류 수정] '서비스명'이 '사용자 질문'에 포함되어 있는지 확인해야 합니다.
            if normalized_service_name in normalized_message and len(normalized_service_name) > 2:
                 print(f"DEBUG: 🚀 Fast Track 키워드 (정확성) 탐지! -> '{service_name}'")
                 return service_name # 문서가 아닌 '사업명'을 반환
        
        # 2단계: 유사도 보조 검사
        scores = {
            name: fuzz.partial_ratio(normalized_message, name.replace(" ", "")) 
            for name in self.service_names_list if name
        }
        if not scores: return None
        
        best_match = max(scores, key=scores.get)
        if scores[best_match] >= 80: # 임계값은 조정 가능
            print(f"DEBUG: 🚀 Fast Track 키워드 (유사도) 탐지! -> '{best_match}'")
            return best_match # 문서가 아닌 '사업명'을 반환
            
        return None
    
    def _merge_and_deduplicate(self, docs1: list[Document], docs2: list[Document]) -> list[Document]:
        """
        [신규 추가] 두 문서 리스트를 병합하고 중복을 제거합니다.
        """
        merged_docs = {} # 순서 유지를 위해 딕셔너리 사용
        
        # 고유 키를 사용하여 문서를 딕셔너리에 추가
        for doc in docs1 + docs2:
            # 문서의 내용과 사업명을 조합하여 고유 키 생성
            doc_key = (doc.page_content, doc.metadata.get('사업명'))
            if doc_key not in merged_docs:
                merged_docs[doc_key] = doc
        
        final_list = list(merged_docs.values())
        print(f"DEBUG: 合併 및 중복 제거 완료. (리스트1: {len(docs1)}개, 리스트2: {len(docs2)}개 -> 최종: {len(final_list)}개)")
        return final_list

    def _generate_fallback_answer(self, user_message: str, documents: list):
        """검색 결과가 없을 때, 전체 서비스 카테고리를 안내하는 폴백 답변을 생성합니다."""
        print(f"DEBUG: 폴백 답변 생성을 위해 {len(documents)}개의 안내 문서를 컨텍스트로 사용합니다.")
        
        context_parts = []
        for doc in documents:
            try:
                # page_content가 JSON 문자열이므로 파싱합니다.
                content_data = json.loads(doc.page_content)
                # 'categories' 키에 있는 정보를 추출합니다.
                if 'categories' in content_data and content_data['categories']:
                    # categories는 리스트 형태의 JSON 문자열일 수 있으므로 다시 파싱합니다.
                    categories_list = json.loads(content_data['categories'])
                    for category in categories_list:
                        context_parts.append(f"- **{category.get('category')}**: {category.get('description')}")
            except (json.JSONDecodeError, TypeError):
                # JSON 파싱 실패 시 원본 텍스트를 사용합니다.
                context_parts.append(doc.page_content)

        context_string = "\n".join(context_parts)

        fallback_template = """
        당신은 사용자의 질문에 딱 맞는 정보를 찾지 못했을 때, 대신 어떤 종류의 복지 서비스가 있는지 친절하게 안내하는 AI 복지 컨설턴트 '지니'입니다.

        [답변 생성 지침]
        1.  "문의하신 내용에 꼭 맞는 서비스를 찾지 못했지만, 제가 도와드릴 수 있는 전체 복지 서비스 분야는 다음과 같습니다." 와 같이 먼저 상황을 설명하며 답변을 시작해주세요.
        2.  주어진 '전체 서비스 분야 정보'를 바탕으로, 각 분야의 이름과 설명을 목록 형태로 명확하게 정리하여 보여주세요.
        3.  사용자가 정보를 보고 다시 질문할 수 있도록 유도하는 문장으로 마무리해주세요.
        4.  마지막 인사는 반드시 "더 궁금한 점이 있으시면 위 분야를 참고하여 다시 질문해주세요!" 로 끝내주세요.

        --- 전체 서비스 분야 정보 ---
        {context}

        --- 사용자 질문 ---
        {question}

        --- 지니의 안내 답변 ---
        """
        fallback_chain = self._create_chain(fallback_template, StrOutputParser())
        final_response = fallback_chain.invoke({
            "question": user_message, "context": context_string
        })
        
        return f'{final_response}', "NORMAL"

    # app/chatbot.py

# ... (다른 코드는 그대로 유지) ...

    def _generate_search_plan(self, user_message: str, chat_history: str):
        """LLM을 사용하여 사용자의 질문과 대화 기록을 분석하고, 검색 계획(키워드, 필터)을 생성합니다."""
        print("DEBUG: 🕵️‍♂️ 1단계 - LLM을 활용한 검색 설계도 생성 시작...")
        parser = JsonOutputParser()
        
        # [수정 완료] '대분류' 필터링 부분을 복원한 프롬프트
        analysis_template = """
    당신은 사용자의 질문을 깊이 있게 분석하여, 가장 중요한 순서대로 검색을 실행할 수 있는 '우선순위가 적용된 검색 계획'을 JSON 형식으로 만드는 최고의 검색 전략가입니다.

    [임무]
    사용자의 최신 질문('{question}')을 바탕으로, 아래 [단계별 사고 과정]에 따라 최적의 '검색 계획'을 작성해주세요.

    [단계별 사고 과정]
    1.  **핵심 필요(Core Needs)와 사용자 맥락(User Context) 식별:** 사용자의 상황에서 가장 시급한 필요(예: 의료비, 생계비)와 사용자의 배경(예: 청년, 실직)을 구분합니다.
    2.  **대상과 질의사항 식별:** 사용자 질의에서 사용자의 대상(예: 미성년자, 임산부, 장애인, 중증질환자, 보훈대상자 등)을 식별하고 식별한 대상이 원하는 질의 사항을 구분합니다.
    3.  **검색 우선순위 결정:** 식별된 필요 사항들의 시급성과 중요도를 판단하여 검색할 순서를 정합니다. 가장 생명과 직결되거나 시급한 문제를 최우선(priority 1)으로 설정합니다.
    4.  **JSON 계획 생성:**
        -   `intent`: 사용자의 대상과 사용자의 핵심 의도를 요약합니다.
        -   `search_plan`: 2단계에서 결정한 우선순위에 따라 검색에 필요한 `keywords`와 `filters`를 묶어 리스트 형태로 만듭니다. 가장 중요한 검색 계획이 리스트의 첫 번째 요소가 되어야 합니다.

    [✨✨✨ 핵심 수정 사항 시작 ✨✨✨]
    [중요 규칙]
    -   '중분류'를 선택할 때는 아래 '전체 중분류 목록 및 상세 설명'에 있는 내용을 보고 사용자 질문을 포함하고 있는 중분류_개요를 찾고 해당 중분류_개요가 해당하는 중분류의 이름과 '정확히 일치'하는 것만 골라야 합니다.
    -   절대 새로운 이름을 만들거나 요약하거나 추론해서는 안 됩니다. 목록에 있는 그대로 사용하세요.
    -   keywords를 추출할 때 사용자 질문의 맥락을 이해하고 질문에 없는 내용을 일부 정보를 가지고 상상으로 만들지 마세요. 예를 들멸 "15세 가출 여중생 임신 걱정" 이라는 내용으로 학생이라는 단어에서 '교육비'라는 keyword를 만들면 안됩니다. 질문에 있는 맥락을 이해하고 그 안에서 keyword를 추출하세요. 
    [✨✨✨ 핵심 수정 사항 종료 ✨✨✨]


    [전체 중분류 목록 및 상세 설명]
    {schema_context}

    [이전 대화 기록]
    {chat_history}

    [출력할 JSON 구조]
    {{
        "intent": "사용자의 핵심 의도를 2~3단어로 요약",
        "search_plan": [
            {{
                "priority": 1,
                "reason": "가장 시급한 문제(예: 중증질환 의료비)에 대한 검색",
                "base_condition": 서비스 제공 대상 여부를 판단하는 사용자의 대상(예: 미성년자, 임산부, 장애인, 중증질환자, 보훈대상자 등),
                "keywords": ["1순위 검색에 사용할 핵심 키워드 배열"],
                "filters": {{ 
                    "중분류": ["(예시) 치료가 어려운 질환을 앓고 있을 때"] 
                }}
            }},
            {{
                "priority": 2,
                "reason": "그 다음으로 중요한 문제(예: 실직으로 인한 생계비)에 대한 검색",
                "base_condition": 서비스 제공 대상 여부를 판단하는 사용자의 대상(예: 미성년자, 임산부, 장애인, 중증질환자, 보훈대상자 등),
                "keywords": ["2순위 검색에 사용할 키워드 배열"],
                "filters": {{ 
                    "중분류": ["(예시) 생계를 유지하기가 힘들 때", "(예시) 실직으로 곤란을 겪고 있을 때"] 
                }}
            }}
        ]
    }}
    
    ---
    [사용자 최신 질문]
    {question}
    ---
    [검색 계획 (JSON)]
"""

        analysis_chain = self._create_chain(analysis_template, parser)
        try:
            analysis_result = analysis_chain.invoke({
                "question": user_message, "schema_context": self.schema_context_str, "chat_history": chat_history
            })
            print(f"DEBUG: LLM 분석 결과 (검색 설계도):\n{json.dumps(analysis_result, ensure_ascii=False, indent=2)}")
            return analysis_result
        except Exception as e:
            print(f"!!! LLM 질의어 분석 실패: {e}")
            return {"intent": "분석 실패", "semantic_keywords": [user_message], "metadata_filters": {}}

    
    def _get_intelligent_response(self, user_message, chat_history):
        """
        [최종 수정] '다단계 필터링' 로직을 적용한 최종 파이프라인
        """
        fast_track_docs = []
        intelligent_docs = []
        remaining_query = user_message.strip()

        # [유지] 1. 순차적 Fast Track 루프 실행
        while True:
            if not remaining_query:
                break
                
            detected_service_name = self._detect_fast_track_keyword(remaining_query)
            
            if detected_service_name:
                print(f"DEBUG: 🕵️‍♂️ 순차적 Fast Track 실행... (탐지된 사업명: {detected_service_name})")
                found_docs = self.db_service.metadata_search({"사업명": detected_service_name})
                if found_docs:
                    fast_track_docs.extend(found_docs)
                
                query_before_removal = remaining_query
                pattern_text = detected_service_name.replace(" ", "")
                pattern = r''.join(char + r'\s*' for char in pattern_text)
                remaining_query = re.sub(pattern, '', remaining_query, count=1, flags=re.IGNORECASE).strip()

                if query_before_removal == remaining_query:
                    print(f"DEBUG: ⚠️ Fast Track으로 탐지된 '{detected_service_name}'가 원본 질문에 없어 제거에 실패했습니다. Fast Track을 중단합니다.")
                    remaining_query = query_before_removal
                    break
            else:
                break

        # [전면 수정] 2. Fast Track 처리 후 남은 질문에 대한 지능형 검색 실행 (다단계 필터링)
        if remaining_query:
            print(f"DEBUG: 🚀 지능형 검색 실행 (남은 질문: '{remaining_query}')")
            
            # 2-1. [유지] 분석 - 우선순위가 포함된 검색 계획 생성
            query_analysis = self._generate_search_plan(remaining_query, chat_history)
            search_plan = query_analysis.get("search_plan", [])

            # 2-2. [신규 로직] 계획에 따라 '다단계 필터링'을 순차적으로 실행하여 결과 누적
            if search_plan:
                print(f"DEBUG: 🕵️‍♂️ 총 {len(search_plan)}개의 우선순위 계획에 따라 순차 검색을 시작합니다.")
                for i, plan in enumerate(search_plan):
                    priority = plan.get('priority', i + 1)
                    reason = plan.get('reason', 'N/A')
                    base_conditions = plan.get('base_condition', [])
                    keywords = plan.get('keywords', [])
                    # 중분류는 리스트 형태이므로 여러 개일 수 있습니다.
                    middle_categories = plan.get('filters', {}).get('중분류', [])
                    
                    if not middle_categories:
                        print(f"DEBUG: [Priority {priority}] 필터링의 기준이 되는 '중분류'가 없어 건너뜁니다.")
                        continue
                    
                    print(f"DEBUG: [Priority {priority} - {reason}] 필터링 실행...")
                    
                    # [신규] Step 1: '중분류'에 해당하는 모든 사업 문서를 DB에서 가져옵니다.
                    # 각 중분류에 대해 metadata_search를 호출하여 문서를 가져옵니다.
                    category_docs = []
                    for category in middle_categories:
                         category_docs.extend(self.db_service.metadata_search({"중분류": category}))
                    
                    if not category_docs:
                        print(f"DEBUG: ➡️  '{middle_categories}' 중분류에 해당하는 문서가 없습니다.")
                        continue
                    
                    print(f"DEBUG: ➡️  {len(category_docs)}개의 '{middle_categories}' 관련 문서를 찾았습니다. 필터링을 시작합니다.")

                    # [신규] Step 2: 1차 필터링 - 'base_condition' (사용자 대상)
                    # base_condition이 없으면 이 단계는 건너뛰고 모든 문서를 통과시킵니다.
                    first_filtered_docs = []
                    if base_conditions:
                        for doc in category_docs:
                            target_info = doc.metadata.get('대상', '').replace(" ", "")
                            # base_condition 중 하나라도 '대상' 정보에 포함되면 통과
                            if any(bc.replace(" ", "") in target_info for bc in base_conditions):
                                first_filtered_docs.append(doc)
                    else:
                        first_filtered_docs = category_docs # 조건이 없으면 모두 통과
                    
                    print(f"DEBUG: ➡️  1차 필터링('base_condition') 후 {len(first_filtered_docs)}개 문서가 남았습니다.")
                    if first_filtered_docs:
                        intelligent_docs.extend(first_filtered_docs)

                    
            else:
                print("DEBUG: ⚠️ LLM이 유효한 검색 계획을 생성하지 못했습니다.")
       
        # [유지] 3. 위기 상황 대비, '10장. 기타 위기별 상황별 지원' 사업을 추가로 검색
        print("DEBUG: 🆘 위기 상황 대비, '10장. 기타 위기별 상황별 지원' 사업을 추가로 검색합니다.")
        crisis_support_docs = self.db_service.metadata_search({
            "대분류": "10장. 기타 위기별 상황별 지원"
        })

        # [수정] 4. 모든 검색 결과 병합 (단순 결합, 중복 제거 없음)
        # _merge_and_deduplicate 함수를 사용하지 않고 리스트를 그대로 합칩니다.
        final_docs = fast_track_docs + intelligent_docs + crisis_support_docs
        print(f"DEBUG: 최종 문서 취합 완료. (FastTrack: {len(fast_track_docs)}개, 지능형: {len(intelligent_docs)}개, 위기지원: {len(crisis_support_docs)}개 -> 최종: {len(final_docs)}개)")
        
        # 수정 코드 (수정 후)
        # 5. [수정] 최종 결과 유효성 확인 및 단계적 폴백 답변 생성
        if not final_docs:
            print("DEBUG: 🕵️‍♂️ 최종 검색 결과 없음. 단계적 폴백 로직 시작...")
            
            
            fallback_docs = self.db_service.metadata_search({
                "사업명": "책 안에 어떤 내용이 담겨 있나요?",
                "항목": "sections"
            })
            if fallback_docs:
                return self._generate_fallback_answer(user_message, fallback_docs)
            else:
                # 최종 안전장치
                return "죄송합니다, 문의하신 내용과 관련된 복지서비스를 찾지 못했습니다. 조금 더 자세히 질문해주시겠어요?", "NORMAL"

        return self._generate_final_answer(user_message, chat_history, final_docs)
                   
    def _format_content(self, data, indent_level=0):
        if not data: return ""
        text_parts = []
        indent = "  " * indent_level
        if isinstance(data, dict):
            for key, value in data.items():
                if value and isinstance(value, str) and value.strip():
                    text_parts.append(f"{indent}- **{key}**: {value.strip()}")
                elif isinstance(value, (dict, list)):
                    formatted_sub_content = self._format_content(value, indent_level + 1)
                    if formatted_sub_content:
                         text_parts.append(f"{indent}- **{key}**:")
                         text_parts.append(formatted_sub_content)
        elif isinstance(data, list):
            for item in data:
                if item:
                    formatted_item = self._format_content(item, indent_level)
                    if formatted_item: text_parts.append(formatted_item)
        else:
            cleaned_data = str(data).strip()
            if cleaned_data: text_parts.append(f"{indent}- {cleaned_data}")
        return "\n".join(filter(None, text_parts))

    def _generate_final_answer(self, user_message, chat_history, documents):
        print(f"DEBUG: 최종 답변 생성을 위해 검색된 {len(documents)}개 문서를 컨텍스트로 사용합니다.")
        
        grouped_docs = {}
        for doc in documents:
            service_name = doc.metadata.get('사업명')
            if not service_name or any(keyword in service_name for keyword in ["목차", "안내", "기준", "소개", "연락처"]):
                continue
            
            if service_name not in grouped_docs:
                grouped_docs[service_name] = {'contents': set()}
            
            content_parts = []
            if doc.metadata.get('개요'): content_parts.append(f"**개요**: {doc.metadata['개요']}")
            if doc.metadata.get('대상'): content_parts.append(f"**지원 대상**: {doc.metadata['대상']}")
            if doc.metadata.get('내용'): content_parts.append(f"**지원 내용**: {doc.metadata['내용']}")
            if '지원내용' in doc.metadata: content_parts.append(f"**지원 내용**: {doc.metadata['지원내용']}")
            
            try:
                content_data = json.loads(doc.page_content)
                formatted_page_content = self._format_content(content_data)
                if formatted_page_content: content_parts.append(f"**세부 정보**:\n{formatted_page_content}")
            except (json.JSONDecodeError, TypeError):
                 if doc.page_content: content_parts.append(f"**내용**: {doc.page_content}")
            
            if doc.metadata.get('방법'): content_parts.append(f"**신청 방법**: {doc.metadata.get('방법')}")
            if doc.metadata.get('문의'):
                contact_info = self._format_content({'문의처': doc.metadata.get('문의')})
                if contact_info: content_parts.append(contact_info)
            
            full_content_str = "\n".join(part for part in content_parts if part)
            if full_content_str:
                grouped_docs[service_name]['contents'].add(full_content_str)

        context_list = []
        for service_name, data in grouped_docs.items():
            full_text = "\n\n".join(sorted(list(data['contents'])))
            context_list.append(f"### 서비스명: {service_name}\n{full_text}\n")
        
        context_string = "\n---\n".join(context_list)

        final_template = """
### 페르소나 (Persona)
당신은 대한민국 복지정책을 30년 이상 총괄해 온 베테랑 정책 담당자이자, AI 복지 전문가 '지니'입니다. 당신의 지식과 경험은 대한민국 최고 수준이며, 어떤 국민이 자신의 상황을 이야기하든 그 사람에게 최적의 복지 서비스 조합을 찾아 제안할 수 있습니다. 당신의 임무는 단순 정보 전달을 넘어, 한 사람의 삶에 실질적인 도움이 될 수 있는 최상의 해결책을 제시하는 것입니다.

### 지니의 임무 및 답변 생성 원칙

주어진 '[관련 서비스 정보]'와 '[사용자 질문]'을 바탕으로, 아래의 단계별 사고 과정을 거쳐 사용자에게 가장 도움이 되는 답변을 생성하세요.

**[1단계: 질문 의도 및 유형 분석]**
먼저 사용자 질문의 근본적인 의도와 유형을 파악합니다.
-   **상황 기반 해결책 요구**: 자신의 어려운 상황을 설명하며, 이에 맞는 전반적인 복지 서비스를 문의하는가?
-   **서비스 간 비교/차이 문의**: 두 가지 이상의 특정 서비스 간의 차이점을 묻는가?
-   **자격 조건 문의**: 특정 서비스를 받기 위한 구체적인 자격 조건을 묻는가?
-   **심화 질문**: 이전 대화에 이어, 더 구체적이거나 발전된 내용을 묻는가?

**[2단계: 핵심 정보 추출 및 우선순위 결정]**
사용자 질문에서 '상태 정보(Who/What)'와 '지원 필요 사항(Needs)'을 명확히 분리하여 추출합니다. 이 정보를 바탕으로 '[관련 서비스 정보]'에서 사용할 내용의 우선순위를 결정합니다.
-   **(예시) 질문**: "80대 노인입니다. 치료가 어려운 질환을 앓고 있어 병원비와 생계비 마련이 어려워요"
-   **(분석)**
    -   `상태 정보`: 80대 노인, 치료가 어려운 질환
    -   `지원 필요 사항`: 병원비, 생계비
-   **(우선순위 결정)**
    1.  먼저 '지원 필요 사항'인 **'병원비'와 '생계비' 지원**에 대한 내용을 '[관련 서비스 정보]'에서 찾는다.
    2.  찾아낸 서비스들의 지원 대상이 '상태 정보'인 **'80대 노인', '치료가 어려운 질환'**과 일치하는지, 혹은 관련이 있는지 교차 확인하여 답변에 사용할 핵심 정보를 최종 선별한다.

**[3단계: 답변 초안 작성 및 구조화]**
선별된 핵심 정보를 바탕으로, 아래 규칙에 따라 답변의 초안을 작성합니다.
-   **전문성과 신뢰의 톤**: "보건복지부의 '나에게 힘이되는 복지서비스'를 바탕으로 말씀드리겠습니다." 와 같이 전문가의 입장에서 신뢰를 주는 톤으로 시작합니다.
-   **정확한 사업명 제시**: 지원 받을 수 있는 사업명을 명확히 제시하고 해당 사업의 개요, 대상자 등을 알려 줍니다
-   **구조적 설명**: 사용자가 이해하기 쉽게 **어떤 혜택(What)**을, **누가(Who)**, **어떻게(How)** 받을 수 있는지 명확히 구조화하여 설명합니다.
-   **정보의 조합**: 필요한 경우, 여러 복지 서비스를 유기적으로 연결하여 사용자에게 최적화된 '서비스 조합'을 제안합니다.

**[4단계: 자격 요건 명시 및 최종 검수]**
-   **엄격한 정보 선별**: '상태 정보'와 '지원 필요 사항'에 부합하지 않는 정보는 답변에서 과감히 제외합니다.
-   **자격 요건 명시 및 안내**: 만약 주어진 정보만으로 사용자가 지원 대상인지 명확히 판단할 수 없다면, **"이 서비스를 이용하시려면 [서비스 제공 대상]에 해당하셔야 합니다."** 와 같이 검색된 서비스의 공식적인 지원 대상을 명확히 알려주고, 본인이 여기에 해당하는지 확인해야 한다고 안내합니다.
-   **마무리**: "안내해 드린 내용이 보탬이 되길 바랍니다. 추가로 궁금한 점이 있으시면 언제든지 다시 찾아주십시오." 와 같이 전문가로서 격려하며 마무리합니다.
-   **정보 부족 시**: 만약 적절한 정보가 전혀 없다면, "안타깝게도 문의하신 내용과 정확히 일치하는 서비스 정보를 찾지 못했습니다. 하지만 다른 방법이 있을 수 있으니, 거주지 주민센터나 보건복지상담센터(국번없이 129)에 문의해보시는 것을 권해드립니다."라고 대안을 제시하며 솔직하게 답변합니다.

---
[관련 서비스 정보]
{context}
---
[사용자 질문]
{question}
---
[30년 경력 복지 전문가 지니의 최종 답변]
"""
        final_chain = self._create_chain(final_template, StrOutputParser())
        final_response = final_chain.invoke({
            "question": user_message, "context": context_string, "chat_history": chat_history
        })
        
        return f'{final_response}', "NORMAL"
    
    