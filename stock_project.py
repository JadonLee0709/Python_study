# stock_project.py
# update_251002_dynamic_MA

import time
import re
from typing import List, Optional, Tuple
from urllib.parse import quote
import webbrowser
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import concurrent.futures
import multiprocessing

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
# 브라우저 열기
# -----------------------------
def open_finance_search(query: str):
    webbrowser.open_new_tab("https://www.naver.com")

# 붙여넣은 텍스트에서 6자리 코드 추출
CODE_RE = re.compile(r"\b(\d{6})\b")
def pick_code_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = CODE_RE.search(text)
    return m.group(1) if m else None

# -----------------------------
# 로컬 폴백 사전
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
# 유틸: MA 윈도우 파싱 (예: "20,60" -> (20, 60))
# -----------------------------
def parse_windows(s: str, default: Tuple[int, ...] = (20, 50)) -> Tuple[int, ...]:
    try:
        wins = tuple(int(x) for x in s.replace(" ", "").split(",") if x)
        if not wins or any(w <= 0 for w in wins):
            return default
        return wins
    except Exception:
        return default

# -----------------------------
# 네이버 검색에서 종목코드 추출
# -----------------------------
def get_code_from_naver_search(query: str) -> Optional[str]:
    url = "https://search.naver.com/search.naver?query={}".format(quote(query))
    try:
        r = requests.get(url, headers=SEARCH_HEADERS, timeout=12)
        r.raise_for_status()
    except Exception as e:
        print("[WARN] naver search request fail: {}".format(e))
        return None

    html = r.text
    m = re.findall(r"/item/(?:main|coinfo|sise_day)\.naver\?[^\"'>]*\bcode=(\d{6})", html)
    if m:
        return m[0]

    m = re.findall(r'"(?:code|stockCd)"\s*:\s*"(\d{6})"', html)
    if m:
        return m[0]

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ")
    m = re.search(r"\b(\d{6})\b(?=[^\n]{0,40}\b(?:KOSPI|KOSDAQ|KRX)\b)", text, re.I)
    if m:
        return m.group(1)

    m = CODE_RE.findall(text)
    if m:
        seen = []
        for c in m:
            if c not in seen:
                seen.append(c)
        return seen[0] if seen else None
    return None

# -----------------------------
# 페이지 단위 크롤링 함수 (멀티프로세싱용)
# -----------------------------
def _fetch_page(code: str, page: int) -> pd.DataFrame:
    url = BASE_URL.format(code=code, page=page)
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        tables = pd.read_html(res.text, match="날짜")
        if not tables:
            return pd.DataFrame()
        df = tables[0]
    except Exception:
        return pd.DataFrame()

    df = df.dropna(how="any")
    if "날짜" not in df.columns or df.empty:
        return pd.DataFrame()
    df = df[df["날짜"].astype(str).str.contains(r"\d{4}\.\d{2}\.\d{2}", na=False)]
    if df.empty:
        return pd.DataFrame()

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
    return df[["Date", "Close", "Open", "High", "Low", "Volume"]]

# -----------------------------
# 멀티프로세싱 데이터 수집
# -----------------------------
def Daily_prices_naver(code: str, pages: int = 10, workers: int = None) -> pd.DataFrame:
    if workers is None:
        workers = multiprocessing.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_page, code, p) for p in range(1, pages + 1)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    frames = [df for df in results if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["Date", "Close", "Open", "High", "Low", "Volume"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out

# -----------------------------
# 이동평균선 계산 (동적 창 길이)
# -----------------------------
def _add_moving_averages(df: pd.DataFrame, windows: Tuple[int, ...] = (20, 50)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        col = "MA{}".format(w)
        out[col] = out["Close"].rolling(window=w, min_periods=w).mean()
    return out

def stock_calculator(code: str, pages: int = 20, workers: int = None,
                     ma_windows: Tuple[int, ...] = (20, 50)) -> pd.DataFrame:
    prices = Daily_prices_naver(code=code, pages=pages, workers=workers)
    prices = _add_moving_averages(prices, windows=ma_windows)
    return prices

# -----------------------------
# 그래프 (동적 창 길이)
# -----------------------------
def graph_operator(df: pd.DataFrame, windows: Tuple[int, ...] = (20, 50)):
    plt.figure(figsize=(12, 6))

    if len(windows) == 2:
        title = "{}일선 & {}일선 추세".format(windows[0], windows[1])
    else:
        title = "이동평균선({})".format(", ".join(str(w) for w in windows))
    plt.title(title)

    plt.xlabel("날짜")
    plt.ylabel("주가")
    plt.grid(True)

    plt.plot(df["Date"], df["Close"], label="종가", color="blue")

    for w in windows:
        col = "MA{}".format(w)
        if col in df.columns:
            plt.plot(df["Date"], df[col], label="{}일선".format(w), linestyle="--")

    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 알람 기능 (동적: 가장 짧은 MA vs 가장 긴 MA 교차)
# -----------------------------
def alarm_operator(df: pd.DataFrame, windows: Tuple[int, ...] = (20, 50)) -> str:
    if not windows:
        return "⚖️ 창 길이 미지정"

    short = min(windows)
    long_ = max(windows)
    cs = "MA{}".format(short)
    cl = "MA{}".format(long_)

    if cs not in df.columns or cl not in df.columns:
        return "⚖️ 필요한 이동평균 컬럼이 없습니다."

    # 두 MA가 유효한(NA 아님) 최근 2개 구간 확보
    tmp = df.dropna(subset=[cs, cl])
    if len(tmp) < 2:
        return "⚖️ 데이터가 부족합니다."

    last = tmp.iloc[-1]
    prev = tmp.iloc[-2]

    if (last[cs] > last[cl]) and (prev[cs] <= prev[cl]):
        return "📈 골든크로스({}↗{}).".format(short, long_)
    elif (last[cs] < last[cl]) and (prev[cs] >= prev[cl]):
        return "📉 데드크로스({}↘{}).".format(short, long_)
    else:
        return "⚖️ 특별한 신호 없음"

# -----------------------------
# 코드 해석
# -----------------------------
def resolve_to_code(user_input: str) -> Optional[str]:
    s = (user_input or "").strip()
    c = pick_code_from_text(s)
    if c:
        return c
    if s in FALLBACK_MAP:
        return FALLBACK_MAP[s]
    return get_code_from_naver_search(s)

# -----------------------------
# GUI
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
        open_finance_search(text)
        messagebox.showinfo("코드 입력 안내", "브라우저에서 종목 코드 확인 후 붙여넣으세요.")
        pasted = simpledialog.askstring("코드 붙여넣기", "6자리 코드 또는 종목 URL:")
        code = pick_code_from_text(pasted)

    if not code:
        messagebox.showerror("입력 오류", "유효한 코드가 없습니다. (입력: {})".format(text))
        return

    pages = simpledialog.askinteger("페이지 수", "몇 페이지 가져올까요?", minvalue=1, maxvalue=100)
    if not pages:
        return

    raw_win = simpledialog.askstring("이동평균", "창 길이(쉼표, 예: 20,60) [기본 20,50]:") or ""
    ma_windows = parse_windows(raw_win, default=(20, 50))

    workers = simpledialog.askinteger(
        "코어 수",
        "사용할 코어 개수 (기본: 최대치)",
        minvalue=1,
        maxvalue=multiprocessing.cpu_count()
    )

    df = stock_calculator(code, pages, workers=workers, ma_windows=ma_windows)
    if df.empty:
        messagebox.showerror("데이터 없음", "code={} 데이터 없음".format(code))
        return

    graph_operator(df, windows=ma_windows)
    signal = alarm_operator(df, windows=ma_windows)
    messagebox.showinfo("알림", signal)

def start_gui():
    root = tk.Tk()
    root.title("주식 그래프 앱")
    button = tk.Button(root, text="실행", command=run_app, width=20)
    button.pack(pady=20)
    root.mainloop()

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    mode = input("실행 모드 선택 (1: CLI, 2: GUI) → ").strip()
    if mode == "1":
        text = input("종목코드(6자리) 또는 종목명을 입력하세요: ").strip()
        code = resolve_to_code(text)

        if not code:
            print("❌ 코드 확인 불가")
        else:
            pages = int(input("몇 페이지를 가져올까요? (예: 20): ").strip() or "20")
            raw_win = input("이동평균 창 길이(쉼표, 예: 20,60) [기본 20,50]: ").strip()
            ma_windows = parse_windows(raw_win, default=(20, 50))

            workers = input("몇 개 코어를 사용할까요? (최대={}): ".format(multiprocessing.cpu_count())).strip()
            workers = int(workers) if workers else None

            df = stock_calculator(code, pages, workers=workers, ma_windows=ma_windows)
            if df.empty:
                print("❌ 데이터 없음 (code={})".format(code))
            else:
                print(df.tail())
                signal = alarm_operator(df, windows=ma_windows)
                print(signal)
                graph_operator(df, windows=ma_windows)
    else:
        start_gui()