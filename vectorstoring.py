import os
from typing import List, Optional, Dict, Tuple
import uuid
import sys

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


INDEX_DIR = "faiss_index"
PDF_DIR = "./pdfs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NS = "reports"  # 컬렉션/네임스페이스 흉내



def load_from_pdf(file_path: str, mode: str = "page") -> Tuple[List[Document], Dict]:
    """
    PDF를 로드해 페이지별 Document 리스트와 '파일 단위' 메타데이터를 함께 반환.
    file_meta는 첫 페이지 metadata를 기준으로 공통 필드를 추려 구성.
    """
    loader = PyMuPDFLoader(file_path, mode=mode)
    docs: List[Document] = loader.load()

    if not docs:
        return [], {}

    # 첫 문서의 메타데이터를 파일 공통 메타로 사용(필드명은 PDF에 따라 대소문자 변형 존재)
    m = docs.metadata.copy()
    # 표준화 키 매핑(있을 때만)
    file_meta = {
        "source": m.get("source") or m.get("file_path"),
        "file_path": m.get("file_path") or m.get("source"),
        "total_pages": m.get("total_pages"),
        "format": m.get("format"),
        "title": m.get("title", ""),
        "author": m.get("author", ""),
        "subject": m.get("subject", ""),
        "keywords": m.get("keywords", ""),
        "creator": m.get("creator", ""),
        "producer": m.get("producer", ""),
        "creationdate": m.get("creationdate") or m.get("creationDate"),
        "moddate": m.get("moddate") or m.get("modDate"),
        "trapped": m.get("trapped", ""),
    }
    return docs, file_meta


def split_docs(doc: List[Document], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # 기본 분리 규칙
    )
    return text_splitter.split_documents(doc)


def ensure_pdf_dir(path: str) -> List[str]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"PDF 디렉터리가 없습니다: {path}")
    pdfs = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"경고: {path} 안에 PDF가 없습니다.", file=sys.stderr)
    return pdfs


def load_or_init_faiss(embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
    try:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception:
        return None
    

def add_documents_with_ids(vectorstore: FAISS, docs: List[Document]) -> int:
    # 중복 방지를 위해 없는 경우에만 id 생성
    # LangChain FAISS는 내부적으로 id를 관리하지만, 명시 ID를 줄 수도 있음
    # 여기서는 메타데이터에 uuid를 넣어 추후 식별
    for d in docs:
        d.metadata.setdefault("uid", str(uuid.uuid4()))
    vectorstore.add_documents(docs)
    return len(docs)


def build_embeddings(model_name: str = EMBED_MODEL) -> HuggingFaceEmbeddings:
    # 필요 시 encode_kwargs로 device 지정 가능: {"device": "cuda"}
    # or normalize_embeddings=True 등 설정
    return HuggingFaceEmbeddings(model_name=model_name)


def count_docs(vectorstore: Optional[FAISS]) -> int:
    if vectorstore is None:
        return 0
    # InMemoryDocstore: 내부 dict 길이
    return len(getattr(vectorstore.docstore, "_dict", {}))


def save_faiss(vectorstore: FAISS):
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)


def ingest_pdfs_to_faiss(
    pdf_dir: str = PDF_DIR,
    ns: str = DEFAULT_NS,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    embed_model: str = EMBED_MODEL,
):
    pdf_files = ensure_pdf_dir(pdf_dir)
    embeddings = build_embeddings(embed_model)

    vectorstore = load_or_init_faiss(embeddings)
    if vectorstore is not None:
        print(f"로딩된 벡터스토어 문서 수: {count_docs(vectorstore)}")
    else:
        print("벡터스토어가 로드되지 않았거나 비어 있습니다.")

    all_splits: List[Document] = []
    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            docs, metadata = load_from_pdf(pdf_file)
            splits = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_splits.extend(splits)
            print(f"[{i}/{len(pdf_files)}] {os.path.basename(pdf_file)} -> 분할 {len(splits)} 청크")
        except Exception as e:
            print(f"경고: {pdf_file} 처리 중 오류: {e}", file=sys.stderr)

    if not all_splits:
        print("경고: 추가할 분할 문서가 없습니다. 종료합니다.", file=sys.stderr)
        return

    if vectorstore is None:
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        added = len(all_splits)
    else:
        added = add_documents_with_ids(vectorstore, all_splits)

    save_faiss(vectorstore)
    print(f"저장 완료. 현재 벡터스토어 문서 수: {count_docs(vectorstore)} (+{added})")

    # 검증 로드
    reloaded = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    print(f"재로딩 검증: 문서 수 {count_docs(reloaded)}")


def search_in_namespace(
    query: str,
    ns: str = DEFAULT_NS,
    k_fetch: int = 50,      # 먼저 넉넉히 가져오기
    k_return: int = 5,      # 필터 후 상위 N
    embed_model: str = EMBED_MODEL,
) -> List[Document]:
    # 컬렉션 흉내: ns 메타로 필터링
    embeddings = build_embeddings(embed_model)
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    docs_and_scores = vs.similarity_search_with_score(query, k=k_fetch)
    filtered: List[Document] = []
    for d, score in docs_and_scores:
        if d.metadata.get("ns") == ns:
            # 점수 보존 시 d.metadata["score"]=score 등으로 저장 가능
            filtered.append(d)
            if len(filtered) >= k_return:
                break
    return filtered


if __name__ == "__main__":

    pdf_dir = "./pdfs"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except:
        vectorstore = None

    if vectorstore is not None:
        print(f"로딩된 벡터스토어 문서 수: {len(vectorstore.docstore._dict)}")
    else:
        print("벡터스토어가 로드되지 않았거나 비어 있습니다.")

    docs = []
    for i, pdf_file in enumerate(pdf_files):
        doc = load_from_pdf(pdf_file)
        splits_documents = chunk(doc)
        docs.extend(splits_documents)
        # if i == 3:
        #     break


    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        vectorstore.add_documents(docs)

    vectorstore.save_local("faiss_index")

    loaded_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print(f"로딩된 벡터스토어 문서 수: {len(loaded_vectorstore.docstore._dict)}")

