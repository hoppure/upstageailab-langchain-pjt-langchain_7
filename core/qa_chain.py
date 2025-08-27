from typing import Dict, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from utils.text_utils import format_docs_with_citations


class QAChain:
    def __init__(self, retriever, llm, vectorstore):
        self.retriever = retriever
        self.llm = llm
        self.vectorstore = vectorstore
        num_companies = self.vectorstore.get_num_companies()
        # 예시의 {source}, {page}는 프롬프트 변수로 해석되지 않도록 이스케이프
        self.prompt = PromptTemplate.from_template(
            """You are a financial analyst assistant. Use the retrieved context to answer accurately in Korean.
If unsure, say '모르겠습니다.' Include source citations like '출처: {{source}}, 페이지 {{page}}'.

Chat History:
{chat_history}

Current Question: {question}
Context:
{context}

Answer step-by-step: 1. 요약, 2. 세부 설명, 3. 결론.
만약 질문이 '어떤 기업 분석해줄 수 있어?'와 같은 광범위한 경우, '현재 """ + str(num_companies) + """개 기업 자료 있어요. 기업 이름이나 업종 알려주세요!'로 유도해 주세요."""
        )

    def run(self, question):
        context_chain = self.retriever | RunnableLambda(format_docs_with_citations)
        chain = (
            {"context": context_chain, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)


class QAMemoryChain:
    def __init__(self, retriever, llm, vectorstore):
        self.retriever = retriever
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        self.vectorstore = vectorstore
        num_companies = self.vectorstore.get_num_companies()
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate.from_template(
                    """You are a financial analyst assistant. Use the retrieved context to answer accurately in Korean.
If unsure, say '모르겠습니다.' Include source citations like '출처: {{source}}, 페이지 {{page}}'.

Chat History:
{chat_history}

Current Question: {question}
Context:
{context}

Answer step-by-step: 1. 요약, 2. 세부 설명, 3. 결론.
만약 질문이 '어떤 기업 분석해줄 수 있어?'와 같은 광범위한 경우, '현재 """ + str(num_companies) + """개 기업 자료 있어요. 기업 이름이나 업종 알려주세요!'로 유도해 주세요."""
                )
            }
        )

    def run(self, question: str) -> Dict[str, Any]:
        """
        질문을 처리하고 답변과 소스 문서를 반환합니다.
        """
        # 기업 수 가져오기
        num_companies = self.vectorstore.get_num_companies()
        
        response = self.qa_chain({"question": question})
        
        # 소스 문서에서 인용 정보 추출
        citations = []
        for doc in response['source_documents']:
            src = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            citations.append(f"(출처: {src}, 페이지 {page})")


        # 답변에 인용 정보 추가 및 기업 수 동적 삽입
        response['answer'] = response['answer'].format(num_companies=num_companies)
        response['answer'] += f"\n\n참고 문서:\n" + "\n".join(citations)
        return response
    
    def clear_memory(self) -> None:
        """대화 히스토리를 초기화합니다."""
        self.memory.clear()
        
    def get_chat_history(self) -> List[Dict[str, str]]:
        """현재까지의 대화 히스토리를 반환합니다."""
        return self.memory.chat_memory.messages


    