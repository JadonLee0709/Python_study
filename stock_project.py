# stock_project.py
# update 0915 (auto code grab from Naver search page + manual fallback)

import time
import re
from typing import List, Optional
from urllib.parse import quote
import webbrowser

import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# GUI
import tkinter as tk
from tkinter import simpledialog, messagebox

# -----------------------------
# 공통 설정
# -----------------------------
BASE_URL = "https://finance.naver.com/item/sise_day.naver?code={code}&page={page}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://finance.naver.com/",
}

SEARCH_HEADERS = {
    **HEADERS,
    "Referer": "https://search.naver.com/",
}

def _set_korean_font():
    # 시스템에 있는 한글 폰트로 자동 설정
    try:
        plt.rc("font", family="AppleGothic")      # macOS
    except Exception:
        try:
            plt.rc("font", family="Malgun Gothic")  # Windows
        except Exception:
            plt.rc("font", family="NanumGothic")    # Linux
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

# -----------------------------
# 브라우저로 네이버(또는 검색) 열기
# -----------------------------
def open_finance_search(query: str):
    # 요청대로 www.naver.com 을 기본으로 열고 싶다면 아래 1줄 사용
    webbrowser.open_new_tab("https://www.naver.com")
    # 바로 검색 결과를 띄우고 싶으면 아래 주석을 해제
    # url = f"https://search.naver.com/search.naver?query={quote(query)}"
    # webbrowser.open_new_tab(url)

# 붙여넣은 텍스트에서 6자리 코드 추출
CODE_RE = re.compile(r"\b(\d{6})\b")
def pick_code_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = CODE_RE.search(text)
    return m.group(1) if m else None

# -----------------------------
# 로컬 폴백 사전(안정성↑)
# -----------------------------
FALLBACK_MAP = {
    "삼성전자": "005930",
    "삼성전자우": "005935",
    "네이버": "035420",
    "NAVER": "035420",
    "카카오": "035720",
    "SK하이닉스": "000660",
    "현대차": "005380",
    "기아": "000270",
}

# -----------------------------
# 네이버 '검색' 페이지에서 종목코드 자동 추출
# -----------------------------
def get_code_from_naver_search(query: str) -> Optional[str]:
    """
    검색.naver.com 결과 HTML에서 6자리 종목코드를 최대한 뽑는다.
    1) finance.naver.com/item/* 링크의 ?code=XXXXXX
    2) 임베디드 JSON의 "code":"XXXXXX" / "stockCd":"XXXXXX"
    3) 전체 텍스트에서 KOSPI/KOSDAQ/KRX 주변 6자리
    """
    url = f"https://search.naver.com/search.naver?query={quote(query)}"
    try:
        r = requests.get(url, headers=SEARCH_HEADERS, timeout=12)
        r.raise_for_status()
    except Exception as e:
        print(f"[WARN] naver search request fail: {e}")
        return None

    html = r.text

    # 1) 링크 패턴: /item/main.naver?code=XXXXXX, /item/coinfo.naver, /item/sise_day.naver 등
    m = re.findall(r"/item/(?:main|coinfo|sise_day)\.naver\?[^\"'>]*\bcode=(\d{6})", html)
    if m:
        return m[0]

    # 2) 임베디드 JSON 내 코드 필드
    m = re.findall(r'"(?:code|stockCd)"\s*:\s*"(\d{6})"', html)
    if m:
        return m[0]

    # 3) 텍스트에서 KOSPI/KOSDAQ/KRX 근처 6자리
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ")
    m = re.search(r"\b(\d{6})\b(?=[^\n]{0,40}\b(?:KOSPI|KOSDAQ|KRX)\b)", text, re.I)
    if m:
        return m.group(1)

    # 마지막으로, 페이지 전역에서 6자리 숫자 후보 (노이즈 가능성↑)
    m = CODE_RE.findall(text)
    if m:
        # 중복 제거 후 반환
        seen = []
        for c in m:
            if c not in seen:
                seen.append(c)
        return seen[0] if seen else None

    return None

# -----------------------------
# 일별 시세 크롤링 (견고 버전)
# -----------------------------
def Daily_prices_naver(code: str, pages: int = 10, pause: float = 0.35) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in range(1, pages + 1):
        url = BASE_URL.format(code=code, page=p)
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
            tables = pd.read_html(res.text, match="날짜")  # '날짜'가 있는 표만
            if not tables:
                break
            df = tables[0]
        except Exception as e:
            print(f"[WARN] page {p} skip: {e}")
            continue

        df = df.dropna(how="any")
        if "날짜" not in df.columns or df.empty:
            continue
        df = df[df["날짜"].astype(str).str.contains(r"\d{4}\.\d{2}\.\d{2}", na=False)]
        if df.empty:
            break

        df = df.rename(
            columns={
                "날짜": "Date",
                "종가": "Close",
                "전일비": "Change",
                "시가": "Open",
                "고가": "High",
                "저가": "Low",
                "거래량": "Volume",
            }
        )

        for col in ["Close", "Open", "High", "Low", "Volume"]:
            s = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("-", "0", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")

        df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d", errors="coerce")
        df = df.dropna(subset=["Date"])
        frames.append(df[["Date", "Close", "Open", "High", "Low", "Volume"]])
        time.sleep(pause)

    if not frames:
        return pd.DataFrame(columns=["Date", "Close", "Open", "High", "Low", "Volume"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out

# -----------------------------
# 이동평균선 계산
# -----------------------------
def _add_moving_averages(df: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"MA{w}"] = out["Close"].rolling(window=w, min_periods=w).mean()
    return out

def stock_calculator(code: str, pages: int = 20) -> pd.DataFrame:
    prices = Daily_prices_naver(code=code, pages=pages)
    prices = _add_moving_averages(prices, windows=(20, 50))
    return prices

# -----------------------------
# 그래프
# -----------------------------
def graph_operator(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.title("20일선 & 50일선 추세")
    plt.xlabel("날짜")
    plt.ylabel("주가")
    plt.grid(True)
    plt.plot(df["Date"], df["Close"], label="종가", color="blue")
    plt.plot(df["Date"], df["MA20"], label="20일선", color="red", linestyle="--")
    plt.plot(df["Date"], df["MA50"], label="50일선", color="green", linestyle="-.")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 이름/코드 입력 → 종목코드 결정
# -----------------------------
def resolve_to_code(user_input: str) -> Optional[str]:
    s = (user_input or "").strip()
    # 1) 6자리면 즉시
    c = pick_code_from_text(s)
    if c:
        return c
    # 2) 폴백 사전
    if s in FALLBACK_MAP:
        return FALLBACK_MAP[s]
    # 3) 네이버 검색 페이지에서 자동 추출
    return get_code_from_naver_search(s)

# -----------------------------
# GUI 모드 (자동 추출 + 수동 폴백)
# -----------------------------
def run_app():
    text = simpledialog.askstring(
        "종목 입력",
        "종목코드(6자리) 또는 종목명을 입력하세요\n예) 005930 또는 삼성전자",
    )
    if not text:
        return

    code = resolve_to_code(text)
    if not code:
        # 자동 추출 실패 시: 브라우저 열고 사용자가 코드/URL 붙여넣기
        open_finance_search(text)
        messagebox.showinfo(
            "코드 입력 안내",
            "열린 브라우저에서 해당 종목을 클릭하세요.\n"
            "주소창의 6자리 코드(또는 페이지 URL)를 복사해 아래에 붙여넣어 주세요.",
        )
        pasted = simpledialog.askstring("코드 붙여넣기", "6자리 코드 또는 종목 페이지 URL:")
        code = pick_code_from_text(pasted)

    if not code:
        messagebox.showerror("입력 오류", f"유효한 6자리 코드를 찾지 못했습니다. (입력: {text})")
        return

    pages = simpledialog.askinteger("페이지 수", "몇 페이지 가져올까요?", minvalue=1, maxvalue=100)
    if not pages:
        return

    try:
        df = stock_calculator(code, pages)
        if df.empty:
            messagebox.showerror("데이터 없음", f"code={code} 에 대한 일별 시세를 가져오지 못했습니다.")
            return
    except Exception as e:
        messagebox.showerror("데이터 수집 오류", f"code={code}\n에러: {e}")
        return

    graph_operator(df)

def start_gui():
    root = tk.Tk()
    root.title("주식 그래프 앱")
    button = tk.Button(root, text="실행", command=run_app, width=20)
    button.pack(pady=20)
    root.mainloop()

# -----------------------------
# CLI 모드
# -----------------------------
if __name__ == "__main__":
    mode = input("실행 모드 선택 (1: CLI, 2: GUI) → ").strip()
    if mode == "1":
        text = input("종목코드(6자리) 또는 종목명을 입력하세요: ").strip()
        code = resolve_to_code(text)
        if not code:
            open_finance_search(text)
            print("열린 브라우저에서 종목을 클릭하고, 6자리 코드 또는 URL을 복사해서 붙여넣으세요.")
            pasted = input("코드/URL 붙여넣기: ").strip()
            code = pick_code_from_text(pasted)

        if not code:
            print(f"❌ 유효한 6자리 코드를 찾지 못했습니다. (입력: {text})")
        else:
            pages = int(input("몇 페이지를 가져올까요? (예: 20): ").strip() or "20")
            df = stock_calculator(code, pages)
            if df.empty:
                print(f"❌ 데이터가 비어있습니다. code={code}")
            else:
                print(df.tail())
                graph_operator(df)
    else:
        start_gui()