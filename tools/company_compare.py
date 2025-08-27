from langchain_core.documents import Document
from langchain.tools import tool
import re


@tool
def compare_companies(retriever, company1: str, company2: str) -> str:
    """
    두 기업의 주요 지표를 비교합니다.
    """
    query1 = f"{company1} PER EPS 매출"
    query2 = f"{company2} PER EPS 매출"
    docs1 = retriever.get_relevant_documents(query1)
    docs2 = retriever.get_relevant_documents(query2)

    if not docs1 or not docs2:
        return f"{company1} 또는 {company2}에 대한 데이터가 부족합니다."

    def extract_metrics(docs):
        per = eps = revenue = "N/A"
        for doc in docs:
            text = doc.page_content.lower()
            per_match = re.search(r'per\s*(\d+\.?\d*)', text)
            eps_match = re.search(r'eps\s*(\d+\.?\d*)', text)
            rev_match = re.search(r'매출\s*(\d{1,3}(?:,\d{3})*)', text)
            if per_match: per = per_match.group(1)
            if eps_match: eps = eps_match.group(1)
            if rev_match: revenue = rev_match.group(1)
        return {"PER": per, "EPS": eps, "매출": revenue}

    metrics1 = extract_metrics(docs1)
    metrics2 = extract_metrics(docs2)

    return (f"{company1} vs {company2} 비교:\n"
            f"- PER: {metrics1['PER']}x vs {metrics2['PER']}x\n"
            f"- EPS: {metrics1['EPS']}원 vs {metrics2['EPS']}원\n"
            f"- 매출: {metrics1['매출']}억 vs {metrics2['매출']}억\n"
            f"출처: 각 문서 메타데이터 참조")