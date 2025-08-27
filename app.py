import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from config import load_environment, configure_langsmith
from core import LLMManager, VectorStoreManager, QAMemoryChain


def initialize_qa_chain():
    # api key 로드
    load_environment()
    configure_langsmith()

    # 기존 설정 가져오기
    llm_provider = "upstage"
    embedding_provider = "upstage"
    vector_store_path = Path("upstage_vectorstore")

    # LLM 매니저 초기화
    llm_manager = LLMManager(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        model_name="solar-pro2-250710",
        embedding_name="solar-embedding-1-large",
        temperature=0
    )

    # 벡터스토어 매니저 초기화
    vector_manager = VectorStoreManager(llm_manager.get_embeddings(), store_path=vector_store_path)
    retriever = vector_manager.load_store()

    # QA 체인 반환
    return QAMemoryChain(retriever, llm_manager.get_llm(), vector_manager)


def create_new_chat():
    """새로운 채팅 세션을 생성합니다."""
    chat_id = int(time.time())
    st.session_state.current_chat_id = chat_id
    st.session_state.chats[chat_id] = {
        "title": "새로운 대화",
        "messages": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    return chat_id

def main():
    st.title("기업 리포트 QA 시스템")
    
    # 세션 상태 초기화
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = initialize_qa_chain()
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'current_chat_id' not in st.session_state:
        create_new_chat()

    # 사이드바 구성
    with st.sidebar:
        st.button("새 대화", on_click=create_new_chat)
        
        st.markdown("### 대화 기록")
        for chat_id, chat in sorted(st.session_state.chats.items(), reverse=True):
            # 첫 메시지나 제목으로 대화 제목 설정
            if len(chat["messages"]) > 0:
                first_msg = chat["messages"][0]["content"]
                title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
                chat["title"] = title
            
            # 대화 선택 버튼
            if st.button(
                f"{chat['title']}\n{chat['created_at']}",
                key=f"chat_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()

        if st.button("모든 대화 삭제", type="secondary"):
            st.session_state.chats = {}
            create_new_chat()
            st.rerun()

    # 현재 채팅 표시
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # 저장된 채팅 히스토리 표시
    for message in current_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가
        current_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    response = st.session_state.qa_chain.run(prompt)
                    answer = response['answer']
                    
                    # 응답 메시지를 구성
                    message_content = answer
                    if 'source_documents' in response:
                        sources = "\n\n**참고 문서:**"
                        for doc in response['source_documents']:
                            source = doc.metadata.get('source', '알 수 없음')
                            sources += f"\n- {source}"
                        message_content += sources
                    
                    st.markdown(answer)
                    with st.expander("참고한 문서 보기"):
                        for doc in response['source_documents']:
                            st.markdown(f"**출처:** {doc.metadata.get('source', '알 수 없음')}")
                            st.markdown(f"**내용:** {doc.page_content[:200]}...")
                    
                    # AI 응답을 현재 채팅에 저장
                    current_chat["messages"].append({
                        "role": "assistant",
                        "content": message_content
                    })
                
                except Exception as e:
                    error_message = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_message)
                    current_chat["messages"].append({
                        "role": "assistant",
                        "content": error_message
                    })

if __name__ == "__main__":
    main()