from pathlib import Path
import json

from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    def __init__(self, embeddings, store_path: Path = Path("vectorstore")):
        self.embeddings = embeddings
        self.vectorstore = None
        self.store_path = store_path
        self.num_companies = 0  # 고유 기업 수를 저장하는 변수
        self._company_log = store_path / "company_log.json"  # 기업 로그 파일

    def build_store(self, documents):
        """새로 벡터스토어 생성 + 저장 및 기업 수 계산"""
        self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        self.vectorstore.save_local(str(self.store_path))
        
        # 고유 기업 수 계산 (메타데이터 'source' 기반)
        unique_sources = set(doc.metadata.get('source', '') for doc in documents)
        self.num_companies = len(unique_sources)
        
        # 기업 로그 저장
        with open(self._company_log, 'w', encoding='utf-8') as f:
            json.dump(list(unique_sources), f, ensure_ascii=False, indent=2)
        
        return self.vectorstore.as_retriever()

    def load_store(self):
        """기존 벡터스토어 불러오기 및 기업 수 로드"""
        if self.store_path.exists():
            self.vectorstore = FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            # 기존 기업 로그 로드
            if self._company_log.exists():
                with open(self._company_log, 'r', encoding='utf-8') as f:
                    self.num_companies = len(json.load(f))
            return self.vectorstore.as_retriever()
        else:
            raise FileNotFoundError("저장된 벡터스토어가 없습니다.")

    def add_documents(self, documents):
        """기존 스토어에 문서 추가 및 기업 수 업데이트"""
        if self.vectorstore is None:
            self.load_store()
        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local(str(self.store_path))
        
        # 기업 수 업데이트
        unique_sources = set(doc.metadata.get('source', '') for doc in documents)
        current_sources = set()
        if self._company_log.exists():
            with open(self._company_log, 'r', encoding='utf-8') as f:
                current_sources = set(json.load(f))
        new_sources = unique_sources - current_sources
        self.num_companies += len(new_sources)
        
        # 업데이트된 기업 로그 저장
        with open(self._company_log, 'w', encoding='utf-8') as f:
            json.dump(list(current_sources.union(unique_sources)), f, ensure_ascii=False, indent=2)

    def get_num_companies(self):
        """현재 저장된 고유 기업 수 반환"""
        return self.num_companies