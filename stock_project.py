# stock_project.py
# update_251110_fixed_MAs + math_analysis_derivatives

import time
import re
from typing import Optional, List
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

# 고정 이동평균 창 길이 (요청사항)
FIXED_WINDOWS = (5, 20, 60, 120)


def _set_korean_font():
    try:
        plt.rc("font", family="AppleGothic")  # macOS
    except Exception:
        try:
            plt.rc("font", family="Malgun Gothic")  # Windows
        except Exception:
            plt.rc("font", family="NanumGothic")  # Linux
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
        return m[0]

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
# 이동평균선 계산 (고정: 5, 20, 60, 120)
# -----------------------------
def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for w in FIXED_WINDOWS:
        col = f"MA{w}"
        out[col] = out["Close"].rolling(window=w, min_periods=w).mean()
    return out


def stock_calculator(code: str, pages: int = 20, workers: int = None) -> pd.DataFrame:
    prices = Daily_prices_naver(code=code, pages=pages, workers=workers)
    prices = _add_moving_averages(prices)
    return prices


# -----------------------------
# 그래프 (고정: 5, 20, 60, 120)
# -----------------------------
def graph_operator(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.title("이동평균선(5, 20, 60, 120)")
    plt.xlabel("날짜")
    plt.ylabel("주가")
    plt.grid(True)

    # 종가
    plt.plot(df["Date"], df["Close"], label="종가", color="black")

    # MAs
    for w in FIXED_WINDOWS:
        col = f"MA{w}"
        if col in df.columns:
            plt.plot(df["Date"], df[col], label=f"{w}일선", linestyle="--")

    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# 알람 기능 (고정: 5 vs 120 교차)
# -----------------------------
def alarm_operator(df: pd.DataFrame) -> str:
    cs, cl = "MA5", "MA120"

    if cs not in df.columns or cl not in df.columns:
        return "⚖️ 필요한 이동평균 컬럼(MA5/MA120)이 없습니다."

    tmp = df.dropna(subset=[cs, cl])
    if len(tmp) < 2:
        return "⚖️ 데이터가 부족합니다."

    last = tmp.iloc[-1]
    prev = tmp.iloc[-2]

    if (last[cs] > last[cl]) and (prev[cs] <= prev[cl]):
        return "📈 골든크로스(5↗120)."
    elif (last[cs] < last[cl]) and (prev[cs] >= prev[cl]):
        return "📉 데드크로스(5↘120)."
    else:
        return "⚖️ 특별한 신호 없음"


# -----------------------------
# 수학적 분석 (미분 기반 특징 + 패턴 통계)
# -----------------------------
def _add_derivative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Close와 MA(5,20,60,120)에 대해 1차/2차 차분(=미분 근사)을 계산해서
    d1_*, d2_* 컬럼을 추가한다.
    """
    out = df.copy()

    # 가격 자체의 변화율(1차, 2차)
    out["d1_Close"] = out["Close"].diff()       # 1차 미분 근사
    out["d2_Close"] = out["d1_Close"].diff()    # 2차 미분 근사

    for w in FIXED_WINDOWS:
        col = f"MA{w}"
        if col in out.columns:
            v_col = f"d1_{col}"  # 1차 미분(기울기)
            a_col = f"d2_{col}"  # 2차 미분(기울기 변화, 곡률)
            out[v_col] = out[col].diff()
            out[a_col] = out[v_col].diff()

    return out


def math_analysis_report(df: pd.DataFrame, horizon: int = 5) -> str:
    """
    - 미분 기반 특징을 추가하고
    - '상승 추세(20/60 우상향) + 단기 조정(5일선 하락 & 가격이 5/20 아래)' 패턴에서
      h일 뒤 수익률 통계를 계산해 리포트 문자열을 반환한다.
    """
    if df.empty or len(df) < 150:
        return "데이터가 부족하여 수학적 분석을 수행할 수 없습니다. (최소 150개 이상 필요)"

    work = _add_derivative_features(df.copy())

    # 향후 horizon일 수익률
    future_col = f"future_ret_{horizon}"
    work[future_col] = work["Close"].shift(-horizon) / work["Close"] - 1.0

    # 레짐 조건: 60일선, 20일선이 모두 우상향 & 20일선이 60일선 위
    cond_regime = (
        (work.get("d1_MA60", 0) > 0)
        & (work.get("d1_MA20", 0) > 0)
        & (work.get("MA20", 0) > work.get("MA60", 0))
    )

    # 단기 조정: 5일선 기울기 < 0, 가격이 5/20일선 아래
    cond_pullback = (
        (work.get("d1_MA5", 0) < 0)
        & (work["Close"] < work.get("MA5", work["Close"] * 0 + float("inf")))
        & (work["Close"] < work.get("MA20", work["Close"] * 0 + float("inf")))
    )

    pattern = cond_regime & cond_pullback
    pattern_df = work.loc[pattern].dropna(subset=[future_col])

    total = len(pattern_df)

    # 현재 상태
    current = work.iloc[-1]
    now_regime = bool(cond_regime.iloc[-1])
    now_pullback = bool(cond_pullback.iloc[-1])

    lines: List[str] = []
    lines.append(f"[수학적 분석 요약 - 향후 {horizon}거래일 기준]")

    # 1) 현재 기울기 상태
    lines.append("1) 현재 이동평균선 기울기 상태:")
    for w in FIXED_WINDOWS:
        v_col = f"d1_MA{w}"
        if v_col in work.columns and pd.notna(current.get(v_col)):
            val = float(current[v_col])
            if val > 0:
                direction = "상승(우상향)"
            elif val < 0:
                direction = "하락(우하향)"
            else:
                direction = "거의 보합"
            lines.append(f"   - {w:3}일선: {direction} (최근 기울기: {val:.2f})")

    # 2) 현재 패턴 분류
    lines.append("\n2) 현재 패턴 판단:")
    if now_regime and now_pullback:
        lines.append("   - 상태: '상승 추세 속 단기 조정' 패턴에 해당합니다.")
    elif now_regime and not now_pullback:
        lines.append("   - 상태: 상승 추세는 유지 중이지만, 단기 조정 구간은 아닙니다.")
    elif (not now_regime) and now_pullback:
        lines.append("   - 상태: 단기 조정처럼 보이지만, 장기 추세(20/60 기준)가 명확한 상승은 아닙니다.")
    else:
        lines.append("   - 상태: 상승 추세도, 전형적인 단기 조정 패턴도 아닙니다.")

    # 3) 과거 통계
    lines.append("\n3) 과거 데이터에서 같은 패턴의 성과:")

    if total < 5:
        lines.append(f"   - 동일 패턴 발생 횟수: {total}회 (5회 미만 → 통계적 신뢰도 낮음)")
        return "\n".join(lines)

    avg_ret = float(pattern_df[future_col].mean())
    win_rate = float((pattern_df[future_col] > 0).mean())
    max_ret = float(pattern_df[future_col].max())
    min_ret = float(pattern_df[future_col].min())

    lines.append(f"   - 동일 패턴 발생 횟수: {total}회")
    lines.append(f"   - 평균 {horizon}일 수익률: {avg_ret * 100:.2f}%")
    lines.append(f"   - 상승한 비율(승률): {win_rate * 100:.1f}%")
    lines.append(f"   - 최고 {horizon}일 수익률: {max_ret * 100:.2f}%")
    lines.append(f"   - 최저 {horizon}일 수익률: {min_ret * 100:.2f}%")

    lines.append(
        "\n※ 위 통계는 과거 패턴 분석 결과일 뿐, "
        "미래 수익을 보장하지 않습니다. (거래비용/슬리피지 미반영)"
    )

    return "\n".join(lines)


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
        messagebox.showerror("입력 오류", f"유효한 코드가 없습니다. (입력: {text})")
        return

    pages = simpledialog.askinteger("페이지 수", "몇 페이지 가져올까요?", minvalue=1, maxvalue=100)
    if not pages:
        return

    workers = simpledialog.askinteger(
        "코어 수",
        "사용할 코어 개수 (기본: 최대치)",
        minvalue=1,
        maxvalue=multiprocessing.cpu_count(),
    )

    df = stock_calculator(code, pages, workers=workers)
    if df.empty:
        messagebox.showerror("데이터 없음", f"code={code} 데이터 없음")
        return

    graph_operator(df)
    signal = alarm_operator(df)

    # 수학적 분석 리포트 생성
    report = math_analysis_report(df, horizon=5)

    messagebox.showinfo("알림", f"{signal}\n\n{report}")


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
            workers = input(
                f"몇 개 코어를 사용할까요? (최대={multiprocessing.cpu_count()}): "
            ).strip()
            workers = int(workers) if workers else None

            df = stock_calculator(code, pages, workers=workers)
            if df.empty:
                print(f"❌ 데이터 없음 (code={code})")
            else:
                print(df.tail())
                signal = alarm_operator(df)
                print(signal)

                # 수학적 분석 리포트 출력 (기본 5거래일 기준)
                report = math_analysis_report(df, horizon=5)
                print("\n=== 수학적 분석(미분 기반) ===")
                print(report)

                graph_operator(df)
    else:
        start_gui()