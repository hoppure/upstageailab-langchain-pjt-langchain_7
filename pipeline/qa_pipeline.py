from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatAnthropic  # Claude
from langchain_upstage import ChatUpstage   # Upstage
from langchain_community.chat_models import ChatOllama    # 로컬 LLM 예시

# TODO: dataclass로 만들까 했는데, 사용가능한 모델 말고 더 뭐가 필요하지 떠오르는게 없어서 일단 dict으로 진행~
AVAILABLE_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "claude": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "upstage": ["solar-mini-250422", "solar-pro2-250710"],
    "ollama": ["llama2", "mistral", "gemma"]
}

# TODO: dataclass로 만들까 했는데, 사용가능한 모델 말고 더 뭐가 필요하지 떠오르는게 없어서 일단 dict으로 진행~
AVAILABLE_MODELS_EMBEDDINGS = {
    "openai": ["text-embedding-3-small"],
    "huggingface": ["sentence-transformers/all-MiniLM-L6-v2"],
    "upstage": ["solar-embedding-1-large"]
}

def check_model_name(provider, model_name, available_models: dict):
    if provider not in available_models:
        raise ValueError(f"지원하지 않는 LLM provider: {provider}")
    if model_name not in available_models[provider]:
        raise ValueError(f"지원하지 않는 모델: {model_name}")




class LLMManager:
    def __init__(self, llm_provider="upstage", embedding_provider="openai", **kwargs):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.kwargs = kwargs

    def get_llm(self):
        check_model_name(self.llm_provider, self.kwargs.get("model_name", "-"), AVAILABLE_MODELS)
        if self.llm_provider == "openai":
            return ChatOpenAI(model_name=self.kwargs.get("model_name", "gpt-4o"),
                              temperature=self.kwargs.get("temperature", 0))
        elif self.llm_provider == "claude":
            return ChatAnthropic(model=self.kwargs.get("model_name", "claude-3-opus-20240229"),
                                 temperature=self.kwargs.get("temperature", 0))
        elif self.llm_provider == "upstage":
            return ChatUpstage(model=self.kwargs.get("model_name", "solar-pro2-250710"),
                               temperature=self.kwargs.get("temperature", 0))
        elif self.llm_provider == "ollama":
            return ChatOllama(model=self.kwargs.get("model_name", "llama2"),
                              temperature=self.kwargs.get("temperature", 0))
        else:
            raise ValueError(f"지원하지 않는 LLM provider: {self.llm_provider}")

    def get_embeddings(self):
        check_model_name(self.embedding_provider, self.kwargs.get("embedding_name", "-"), AVAILABLE_MODELS_EMBEDDINGS)
        if self.embedding_provider == "openai":
            return OpenAIEmbeddings(model=self.kwargs.get("embedding_name", "text-embedding-3-small"))
        elif self.embedding_provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.kwargs.get("embedding_name", "sentence-transformers/all-MiniLM-L6-v2"))
        elif self.embedding_provider == "upstage":
            return UpstageEmbeddings(model=self.kwargs.get("embedding_name", "solar-embedding-1-large"),)
        else:
            raise ValueError(f"지원하지 않는 Embedding provider: {self.embedding_provider}")


class DocumentProcessor:
    def __init__(self, file_path: Path, chunk_size=1000, chunk_overlap=50):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self):
        """PDF 로드 후 청크 분할 + source/page 메타데이터 정규화"""
        loader = PyMuPDFLoader(str(self.file_path))  # Path 객체를 문자열로 변환
        docs = loader.load()

        # 파일명/페이지 메타데이터 추가 및 1-based 보정
        for doc in docs:
            # 파일명 메타데이터
            doc.metadata["source"] = self.file_path.name

            # page 키 정규화 (0-based 가능성 처리)
            page0 = doc.metadata.get("page", doc.metadata.get("page_number"))
            if page0 is not None:
                try:
                    doc.metadata["page"] = int(page0) + 1  # 1-based로 보정
                except Exception:
                    # 숫자로 캐스팅 실패 시 원본 유지하되 최소 1로 폴백
                    doc.metadata["page"] = 1
            else:
                # 페이지 정보가 없다면 1로 기본값
                doc.metadata["page"] = 1

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        # 청크 분할 시 메타데이터는 자동 복제됨
        chunks = splitter.split_documents(docs)
        return chunks


class VectorStoreManager:
    def __init__(self, embeddings, store_path: Path = Path("vectorstore")):
        self.embeddings = embeddings
        self.vectorstore = None
        self.store_path = store_path

    def build_store(self, documents):
        """새로 벡터스토어 생성 + 저장"""
        self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        self.vectorstore.save_local(str(self.store_path))
        return self.vectorstore.as_retriever()

    def load_store(self):
        """기존 벡터스토어 불러오기"""
        if self.store_path.exists():
            self.vectorstore = FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return self.vectorstore.as_retriever()
        else:
            raise FileNotFoundError("저장된 벡터스토어가 없습니다.")

    def add_documents(self, documents):
        """기존 스토어에 문서 추가"""
        if self.vectorstore is None:
            self.load_store()
        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local(str(self.store_path))


def process_pdfs_in_folder(
    folder_path: Path,
    embeddings,
    store_path: Path = Path("vectorstore"),
    chunk_size=1000,
    chunk_overlap=50
):
    """
    폴더 내 새 PDF만 감지해서 벡터스토어에 추가
    processed_files.json은 store_path 내부에 저장
    """
    # store_path 폴더 생성 (없으면)
    store_path.mkdir(parents=True, exist_ok=True)

    # processed_files.json 경로를 store_path 안으로 이동
    processed_log = store_path / "processed_files.json"

    # 기존 처리된 파일 목록 불러오기
    if processed_log.exists():
        with open(processed_log, "r", encoding="utf-8") as f:
            processed_files = set(json.load(f))
    else:
        processed_files = set()

    # 폴더 내 모든 PDF 파일
    pdf_files = list(folder_path.glob("*.pdf"))

    # 새로 추가된 PDF만 필터링
    new_pdfs = [pdf for pdf in pdf_files if pdf.name not in processed_files]

    if not new_pdfs:
        print("새로운 PDF가 없습니다. 업데이트 불필요.")
        return

    print(f"새로 감지된 PDF: {[pdf.name for pdf in new_pdfs]}")

    # 새 PDF 임베딩
    all_chunks = []
    for pdf in new_pdfs:
        processor = DocumentProcessor(pdf, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = processor.load_and_split()
        all_chunks.extend(chunks)

    manager = VectorStoreManager(embeddings, store_path=store_path)

    if store_path.exists() and any(store_path.iterdir()):
        # 기존 스토어에 추가
        manager.load_store()
        manager.add_documents(all_chunks)
    else:
        # 새 스토어 생성
        manager.build_store(all_chunks)

    # 처리된 파일 목록 업데이트
    processed_files.update([pdf.name for pdf in new_pdfs])
    with open(processed_log, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, ensure_ascii=False, indent=2)

    print("벡터스토어 업데이트 완료.")


class QAChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        # 예시의 {source}, {page}는 프롬프트 변수로 해석되지 않도록 이스케이프
        self.prompt = PromptTemplate.from_template(
            """You are a financial analyst assistant. Use the retrieved context to answer accurately in Korean.
If unsure, say '모르겠습니다.' Include source citations like '출처: {{source}}, 페이지 {{page}}'.
Question: {question}
Context:
{context}
Answer step-by-step: 1. 요약, 2. 세부 설명, 3. 결론."""
        )

    @staticmethod
    def _format_docs_with_citations(docs):
        parts = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown")
            pg = d.metadata.get("page", "?")
            parts.append(f"[{i}] {d.page_content}\n(출처: {src}, 페이지 {pg})")
        return "\n\n".join(parts)

    def run(self, question):
        context_chain = self.retriever | RunnableLambda(self._format_docs_with_citations)
        chain = (
            {"context": context_chain, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)
