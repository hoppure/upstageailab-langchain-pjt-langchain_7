# %%
from pathlib import Path

from pipeline.qa_pipeline import LLMManager, VectorStoreManager, process_pdfs_in_folder, QAChain


# %%
# 1. 각종 변수 세팅
# TODO: 나중에는 yaml로 옮기든 파라미터 넣어주는 식으로 바꾸든지 해야함.
llm_provider = "upstage"  # openai / claude / upstage / ollama
embedding_provider = "upstage" # openai / huggingface / upstage
vector_store_path = Path("upstage_vectorstore")

# %%
# 2. LLM/Embedding 설정
llm_manager = LLMManager(
    llm_provider=llm_provider,
    embedding_provider=embedding_provider,
    model_name="solar-pro2-250710",
    embedding_name="solar-embedding-1-large",
    temperature=0
)

# 3. 폴더 안에 있는 문서 처리
retriever = process_pdfs_in_folder(
    folder_path=Path("data"),       # PDF 폴더 경로
    embeddings=llm_manager.get_embeddings(),
    store_path=vector_store_path,
    chunk_size=1000,
    chunk_overlap=100
)

# 4. 벡터스토어 불러오기
vector_manager = VectorStoreManager(llm_manager.get_embeddings(), store_path=vector_store_path)
retriever = vector_manager.load_store()


# 5. QA 체인 생성
qa_chain = QAChain(retriever, llm_manager.get_llm())

# %%
# 6. 질문 실행
print(qa_chain.run("이 기업의 시가총액은 얼마야?"))
# %%
print(qa_chain.run("어떤 기업을 분석해줄 수 있어?"))

# %%
print(qa_chain.run("원익 기업의 리포트 요약해줘."))
# %%
print(qa_chain.run("삼성전자 기업의 리포트 요약해줘"))
# %%
