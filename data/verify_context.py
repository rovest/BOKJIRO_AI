import json

def generate_and_print_structured_details(file_path: str):
    """
    JSON 파일에서 '대분류'에 따라 '중분류'와 '중분류_개요'를 그룹화하고,
    파일에 명시된 순서대로 출력하는 함수입니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        print("스크립트를 'vd_base_v2_refined.json' 파일과 같은 위치에 두고 실행해주세요.")
        return

    all_docs = data
    
    # 최종 데이터를 저장할 딕셔너리 (Key: 대분류, Value: 중분류 리스트)
    # 파이썬 3.7+ 에서는 딕셔너리가 삽입 순서를 유지합니다.
    structured_details = {}
    
    # 이미 추가된 '대분류-중분류' 조합을 추적하기 위한 집합(set)
    seen_combinations = set()

    for doc in all_docs:
        metadata = doc.get('metadata', {})
        major_cat = metadata.get('대분류')
        minor_cat = metadata.get('중분류')
        overview = metadata.get('중분류_개요')

        # 모든 정보가 존재할 때만 처리
        if major_cat and minor_cat and overview:
            # (대분류, 중분류) 쌍으로 중복 여부 확인
            combination = (major_cat, minor_cat)
            if combination not in seen_combinations:
                # 처음 보는 대분류라면, 딕셔너리에 키를 추가
                if major_cat not in structured_details:
                    structured_details[major_cat] = []
                
                # 중분류와 개요를 리스트에 추가
                structured_details[major_cat].append({
                    '중분류': minor_cat,
                    '개요': overview
                })
                
                # 처리된 조합으로 기록
                seen_combinations.add(combination)

    # --- 결과 출력 ---
    print("=" * 80)
    print("✨ '대분류' 그룹별 '중분류-중분류_개요' 목록 (원본 순서 유지) ✨")
    print("=" * 80)

    total_minor_cats = 0
    for major_cat, minor_details_list in structured_details.items():
        print(f"\n📁 대분류: {major_cat}")
        print("-" * 40)
        for details in minor_details_list:
            print(f"  ▶ 중분류: {details['중분류']}")
            print(f"    └ 개요: {details['개요']}")
            total_minor_cats += 1
        
    print("\n" + "=" * 80)
    print(f"총 {len(structured_details)}개의 대분류, {total_minor_cats}개의 중분류가 성공적으로 추출되었습니다.")
    print("=" * 80)


if __name__ == "__main__":
    JSON_FILE_PATH = 'vd_base_v2_refined.json'
    # 'data' 폴더 안에 json 파일이 있는 경우 경로를 아래와 같이 수정할 수 있습니다.
    # JSON_FILE_PATH = 'data/vd_base_v2_refined.json'
    generate_and_print_structured_details(JSON_FILE_PATH)