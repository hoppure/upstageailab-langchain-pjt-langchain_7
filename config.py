import os
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangSmith 공식 환경변수 방식
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default-project")

# langchain_teddynote 편의 함수 방식
USE_TEDDYNOTE = os.getenv("USE_TEDDYNOTE", "false").lower() == "true"

def set_lang_smith_env():

    if USE_TEDDYNOTE:
        try:
            from langchain_teddynote import logging
            # .env에 LANGCHAIN_API_KEY가 없더라도 logging.langsmith()에서 세팅 가능
            logging.langsmith(LANGCHAIN_PROJECT)
            print(f"[LangSmith] teddynote 방식으로 '{LANGCHAIN_PROJECT}' 프로젝트 연결 완료")
        except ImportError:
            print("[경고] USE_TEDDYNOTE=true 이지만 langchain_teddynote 패키지가 설치되어 있지 않습니다.")
    else:
        # 공식 방식: 환경변수 기반 → LangChain이 자동 인식
        if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
            print(f"[LangSmith] 공식 방식으로 '{LANGCHAIN_PROJECT}' 프로젝트 연결 완료")
        else:
            print("[LangSmith] 추적 비활성화 상태 (환경변수 미설정)")
