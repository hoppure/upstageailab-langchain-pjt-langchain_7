import requests
from bs4 import BeautifulSoup
import os
import time  # 추가


def crawling(p, url):
        
    # 타겟 URL
    url = f"https://finance.naver.com/research/company_list.naver?page={p}"

    # 요청 및 파싱
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # PDF 링크를 추출 (링크 내에 '.pdf'가 포함된 경우만)
    pdf_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if ".pdf" in href:
            # 네이버 금융의 경우, 상대경로가 많으므로 절대경로 변환 필요
            if href.startswith("/"):
                href = "https://finance.naver.com" + href
            pdf_links.append(href)
    # 저장할 폴더 생성
    os.makedirs("pdfs", exist_ok=True)

    # PDF 파일 다운로드
    for link in pdf_links:
        file_name = link.split("/")[-1]
        file_path = os.path.join("pdfs", file_name)
        r = requests.get(link)
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(r.content)
            print(f"Downloaded: {file_name}")

    print(f"{p}페이지, PDF 크롤링 완료!")


if __name__ == "__main__":
    for i in range(50):
        crawling(i+1)
        time.sleep(5)  # 5초 대기 (원하는 초 단위로 변경 가능)