# update 0915 (ver.10, code-or-name input + robust search)

import time
import re
import json
from typing import List, Optional
from urllib.parse import quote

import requests
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog
from bs4 import BeautifulSoup

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

def _set_korean_font():
    try:
        plt.rc("font", family="AppleGothic")      # mac
    except Exception:
        try:
            plt.rc("font", family="Malgun Gothic")  # windows
        except Exception:
            plt.rc("font", family="NanumGothic")    # linux
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

def _http_get(url: str, headers: dict, timeout: int = 10, retries: int = 2) -> Optional[requests.Response]:
    """간단 재시도 GET"""
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.ok and r.text:
                return r
        except Exception as e:
            last_err = e
        time.sleep(0.25)
    return None

# -----------------------------
# 종목명 → 종목코드 변환 (3단계 폴백)
# -----------------------------
def get_stock_code(name: str) -> Optional[str]:
    if not name:
        return None
    q = quote(name.strip())

    # 1) 모바일 검색 API(JSON)
    try:
        m_url = f"https://m.stock.naver.com/api/search/stock?keyword={q}"
        r = _http_get(
            m_url,
            headers={**HEADERS, "Referer": "https://m.stock.naver.com/", "Accept": "application/json,*/*"},
            timeout=10,
            retries=2,
        )
        if r:
            js = r.json()
            candidates = []
            if isinstance(js, dict):
                for key in ("result", "stocks", "items", "list", "data"):
                    if key in js:
                        val = js[key]
                        if isinstance(val, dict) and "stocks" in val:
                            candidates = val["stocks"]
                            break
                        if isinstance(val, list):
                            candidates = val
                            break
            if not candidates and isinstance(js, dict):
                for v in js.values():
                    if isinstance(v, list):
                        candidates = v
                        break

            best = None
            for it in candidates or []:
                code = (it.get("itemCode") or it.get("symbol") or it.get("code") or "").strip()
                nm = (it.get("stockName") or it.get("name") or it.get("stock_nm") or "").strip()
                if code.isdigit() and len(code) == 6:
                    if nm == name.strip():
                        return code
                    if best is None:
                        best = code
            if best:
                return best
    except Exception:
        pass

    # 2) 데스크톱 검색 페이지(HTML, EUC-KR) 파싱
    try:
        d_url = f"https://finance.naver.com/search/searchList.naver?query={q}"
        r = _http_get(d_url, headers=HEADERS, timeout=15, retries=2)
        if r and r.text:
            if not r.encoding or r.encoding.lower() in ("utf-8", "iso-8859-1"):
                r.encoding = "cp949"  # EUC-KR 강제
            soup = BeautifulSoup(r.text, "html.parser")
            links = soup.select("td.tit a[href*='code=']")

            for a in links:  # 정확 이름 우선
                txt = a.get_text(strip=True)
                href = a.get("href", "")
                if "/item/main.naver" in href and txt == name.strip():
                    return href.split("code=")[-1][:6]

            for a in links:  # 첫 결과 fallback
                href = a.get("href", "")
                if "/item/main.naver" in href:
                    return href.split("code=")[-1][:6]
    except Exception:
        pass

    # 3) 자동완성 API(JSONP)
    try:
        ac_url = f"https://ac.finance.naver.com/ac?st=111&r_lt=111&q={q}&q_enc=UTF-8&t_koreng=1"
        r = _http_get(ac_url, headers={**HEADERS, "Referer": "https://finance.naver.com/"}, timeout=10, retries=2)
        if r and r.text:
            m = re.search(r"\((\[.*\])\)\s*$", r.text, re.S)
            if m:
                # 코드는 6자리 숫자
                codes = re.findall(r'"(\d{6})"', m.group(1))
                if codes:
                    return codes[0]
    except Exception:
        pass

    return None

# -----------------------------
# 코드/이름 입력 → 코드 해석
# -----------------------------
def resolve_to_code(user_input: str) -> Optional[str]:
    """6자리 숫자면 그대로 코드, 아니면 이름으로 검색"""
    if not user_input:
        return None
    s = user_input.strip()
    if re.fullmatch(r"\d{6}", s):
        return s
    return get_stock_code(s)

# -----------------------------
# 일별 시세 크롤링
# -----------------------------
def Daily_prices_naver(code: str, pages: int = 10) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in range(1, pages + 1):
        url = BASE_URL.format(code=code, page=p)
        html = requests.get(url, headers=HEADERS, timeout=15).text
        df = pd.read_html(html)[0]

        df = df.dropna(how="any")
        if "날짜" in df.columns:
            df = df[df["날짜"].astype(str).str.contains(r"\d{4}\.\d{2}\.\d{2}", na=False)]

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
            df[col] = pd.to_numeric(s, errors="coerce").fillna(0).round(0).astype("int64")

        df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d")
        frames.append(df[["Date", "Close", "Open", "High", "Low", "Volume"]])
        time.sleep(0.35)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out

# -----------------------------
# 이동평균선
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
# GUI 모드
# -----------------------------
def run_app():
    text = simpledialog.askstring("종목 입력", "종목코드(6자리) 또는 종목명을 입력하세요\n예) 005930 또는 삼성전자")
    code = resolve_to_code(text)
    if not code:
        print(f"❌ 종목코드를 찾을 수 없습니다. (입력: {text})")
        return
    pages = simpledialog.askinteger("페이지 수", "몇 페이지 가져올까요?", minvalue=1, maxvalue=100)
    if not pages:
        return
    df = stock_calculator(code, pages)
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
        text = input("종목코드(6자리) 또는 종목명을 입력하세요 (예: 005930 또는 삼성전자): ").strip()
        code = resolve_to_code(text)
        if not code:
            print(f"❌ 종목코드를 찾을 수 없습니다. (입력: {text})")
        else:
            pages = int(input("몇 페이지를 가져올까요? (예: 20): ").strip() or "20")
            df = stock_calculator(code, pages)
            print(df.tail())
            graph_operator(df)
    else:
        start_gui()