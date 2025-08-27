from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


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