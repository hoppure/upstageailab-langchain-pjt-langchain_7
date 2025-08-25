import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_from_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return docs

def chunk(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(doc)
    return split_documents


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

    loaded_vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print(f"로딩된 벡터스토어 문서 수: {len(loaded_vs.docstore._dict)}")

