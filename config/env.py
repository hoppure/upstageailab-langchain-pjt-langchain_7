import os
from dotenv import load_dotenv

# .env 파일을 모듈 로드 시점에 바로 로드합니다.
load_dotenv()

# LangSmith 관련 설정 변수들을 모듈 레벨에서 정의합니다.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default-project")
USE_TEDDYNOTE = os.getenv("USE_TEDDYNOTE", "false").lower() == "true"

def load_environment() -> None:
    """필수 환경 변수가 설정되었는지 확인합니다."""
    # 필수 API 키 확인
    required_keys = ["UPSTAGE_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise ValueError(f"다음 환경 변수가 누락되었습니다: {', '.join(missing_keys)}")

def configure_langsmith() -> None:
    """LangSmith 추적 설정"""
    if USE_TEDDYNOTE:
        try:
            from langchain_teddynote import logging
            logging.langsmith(LANGSMITH_PROJECT)
            print(f"[LangSmith] teddynote 방식으로 '{LANGSMITH_PROJECT}' 프로젝트 연결 완료")
        except ImportError:
            print("[경고] langchain_teddynote 패키지가 설치되어 있지 않습니다.")
    else:
        if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
            print(f"[LangSmith] 공식 방식으로 '{LANGSMITH_PROJECT}' 프로젝트 연결 완료")
        else:
            print("[LangSmith] 추적 비활성화 상태 (환경변수 미설정)")