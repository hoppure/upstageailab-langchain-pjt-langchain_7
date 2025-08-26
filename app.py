import streamlit as st
from pathlib import Path
from pipeline.config import load_api_keys
from pipeline.qa_pipeline import LLMManager, VectorStoreManager, process_pdfs_in_folder, QAChain, QAMemoryChain


def initialize_qa_chain():
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


def main():
    st.title("기업 리포트 QA 시스템")
    
    # 세션 스테이트에 QA 체인 저장
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = initialize_qa_chain()
        st.session_state.messages = []  # 메시지 히스토리 초기화

    # 사이드바에 대화 초기화 버튼 추가
    if st.sidebar.button("대화 내용 초기화"):
        st.session_state.messages = []
        st.session_state.qa_chain.clear_memory()
        st.rerun()

    # 사이드바에 설명 추가
    st.sidebar.markdown("""
    ### 사용 방법
    1. 질문을 입력하세요
    2. Enter를 누르거나 'Ask' 버튼을 클릭하세요
    3. AI가 답변을 생성합니다
    """)

    # 저장된 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    response = st.session_state.qa_chain.run(prompt)
                    answer = response['answer']
                    
                    # 응답 메시지를 세션에 저장
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
                    
                    # AI 응답을 세션 상태에 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": message_content
                    })
                
                except Exception as e:
                    error_message = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

if __name__ == "__main__":
    main()