from pathlib import Path
import json

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager


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