import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

load_dotenv()

def get_llm(model_name: str = "gemini"):
    model_name = model_name.lower()
    
    if model_name == "gemini":
        print("DEBUG: Google Gemini 모델을 로딩합니다.")
        # [수정] 보스의 요청에 따라 gemini-2.0-flash 모델로 변경
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.2, 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif model_name == "gemma":
        # Gemma 모델은 메모리를 많이 사용하므로, 더 가벼운 llama3.2로 대체합니다.
        print("DEBUG: Gemma 모델 요청 확인. 메모리 안정을 위해 경량 모델(llama3.2)을 로딩합니다.")
        return Ollama(model="gemma3:latest") 
            
    elif model_name == "exaone":
        # Exaone 모델은 메모리를 많이 사용하므로, 더 가벼운 llama3.2로 대체합니다.
        print("DEBUG: Exaone 모델 요청 확인. 메모리 안정을 위해 경량 모델(llama3.2)을 로딩합니다.")
        return Ollama(model="exaone3.5:latest")
    else:
        print(f"경고: '{model_name}'은(는) 지원하지 않는 모델입니다. 기본 Gemini 모델을 사용합니다.")
        return get_llm("gemini")
