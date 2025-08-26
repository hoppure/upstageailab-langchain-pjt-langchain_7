import requests
from bs4 import BeautifulSoup
import os
import time  # 추가
from datetime import datetime 
from pypdf import PdfReader, PdfWriter


def write_pdf_metadata_pypdf(src_path, dst_path, title, author, subject, keywords, created_dt, custom=None):
    # src_path에서 읽고 메타데이터 추가 후 dst_path로 저장
    reader = PdfReader(src_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    # 기존 메타데이터 유지하고 싶다면 복사
    if reader.metadata is not None:
        writer.add_metadata(reader.metadata)

    # PDF 날짜 형식 D:YYYYMMDDHHmmSS±HH'mm'
    # 간단히 로컬시간 기준으로 기록 (타임존 생략 가능)
    time_str = created_dt.strftime("D:%Y%m%d%H%m%S")

    meta = {
        "/Title": title or "",
        "/Subject": subject or "",
        "/Author": author or "",
        "/Keywords": keywords or "",
        "/Creator": "crawler-script",
        "/Producer": "crawler-script",
        "/CreationDate": time_str,
        "/ModDate": time_str,
    }

    writer.add_metadata(meta)

    with open(dst_path, "wb") as f:
        writer.write(f)


def get_tds_from_table_row(tr):
    # tr 내부의 td들 추출
    tds = tr.find_all("td")

    ticker = tds[0].get_text(strip=True) if tds[0] else ""
    title = tds[1].get_text(strip=True) if len(tds) >= 1 else ""
    date = tds[-2].get_text(strip=True) if len(tds) >= 2 else ""
    company = tds[-4].get_text(strip=True) if len(tds) >= 1 else ""
    date_dt = datetime.strptime(date, "%y.%m.%d")

    return ticker, title, company, date_dt


def get_row_from_a_tag(a_tag):
        # 입력된 a_tag 에서 가장 가까운 상위 tr 가져오기
        tr = a_tag.find_parent("tr")
        if tr is None:
            # tr을 못 찾는 경우 상위로 한 단계 더 타고 올라가며 탐색
            parent = a_tag.parent
            while parent and parent.name != "tr":
                parent = parent.parent
            tr = parent
        return tr


def crawling(p):
        
    # 타겟 URL
    url = f"https://finance.naver.com/research/company_list.naver?page={p}"

    # 요청 및 파싱
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # PDF 링크를 추출 (링크 내에 '.pdf'가 포함된 경우만)
    pdfs = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if ".pdf" in href:
            # 네이버 금융의 경우, 상대경로가 많으므로 절대경로 변환 필요
            if href.startswith("/"):
                href = "https://finance.naver.com" + href
            
            tablerow = get_row_from_a_tag(a_tag)
            ticker, title, company, date_dt = get_tds_from_table_row(tablerow)

            pdfs.append({"link": href, 'ticker': ticker, 'title':title, 'author': company, 'date': date_dt})
    # 저장할 폴더 생성
    os.makedirs("pdfs", exist_ok=True)

    # PDF 파일 다운로드
    for pdf in pdfs:
        file_name = pdf['link'].split("/")[-1]
        file_path = os.path.join("pdfs", file_name)
        r = requests.get(pdf['link'])
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(r.content)
                     
            # 메타 데이터 덮어쓰기
            write_pdf_metadata_pypdf(
                src_path=file_path,
                dst_path=file_path,
                title=pdf['title'],
                author=pdf['author'],
                subject=pdf['ticker'],
                keywords=pdf['ticker'],
                created_dt=pdf['date'],  # datetime 객체
            )
            print(f"Downloaded: {file_name}")


    print(f"{p}페이지, PDF 크롤링 완료!")


if __name__ == "__main__":
    for i in range(50):
        crawling(i+1)
        time.sleep(5)  # 5초 대기 (원하는 초 단위로 변경 가능)
        break
