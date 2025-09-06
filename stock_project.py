#update 0906(ver.2)

import time
from typing import List
import requests
import pandas as pd


# ❗ BASE_URL은 code/page 자리표시자가 있어야 .format이 동작함
BASE_URL = "https://finance.naver.com/item/sise_day.naver?code={code}&page={page}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def Daily_prices_naver(code: str, pages: int = 10) -> pd.DataFrame:
    """
    네이버 금융 일별시세에서 종가 기준 데이터 수집
    Returns
    --------------
    pd.DataFrame
    columns = [Date, Close, Open, High, Low, Volume]
    Date 오름차순 정렬
    """
    frames: List[pd.DataFrame] = []
    for p in range(1, pages + 1):
        url = BASE_URL.format(code=code, page=p)
        html = requests.get(url, headers=HEADERS, timeout=15).text

        # 테이블 읽기
        df = pd.read_html(html)[0]  # flavor 생략(lxml/html5lib 중 설치된 파서 사용)

        # 공백/합계 행 제거 + '날짜'가 실제 날짜인 행만 남기기(헤더 반복 방지)
        df = df.dropna(how="any")
        if "날짜" in df.columns:
            df = df[df["날짜"].astype(str).str.contains(r"\d{4}\.\d{2}\.\d{2}", na=False)]

        # 컬럼명 영문으로 통일
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

        # 숫자 컬럼: 쉼표/하이픈/소수 안전 처리 후 정수로 변환
        for col in ["Close", "Open", "High", "Low", "Volume"]:
            s = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)   # 71,600 → 71600
                .str.replace("-", "0", regex=False)  # '-' → '0'
                .str.strip()
            )
            s = pd.to_numeric(s, errors="coerce")    # '71600.0' → 71600.0, 처리불가 → NaN
            df[col] = s.fillna(0).round(0).astype("int64")

        df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d")
        frames.append(df[["Date", "Close", "Open", "High", "Low", "Volume"]])
        time.sleep(0.4)

    out = pd.concat(frames, ignore_index=True)
    # ❗ drop_duplicates / sort_values는 반환값을 받아야 적용됨
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out

def _add_moving_averages(df: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    """Close 기준 이동평균 열(MA20, MA50 등) 추가 후 반환"""
    out = df.copy()
    for w in windows:
        out[f"MA{w}"] = out["Close"].rolling(window=w, min_periods=w).mean()
    return out  # for 바깥

def stock_calculator(code: str, pages: int = 20) -> pd.DataFrame:
    """
    네이버 금융에서 종가를 스크래핑하여 20일선/50일선을 계산해 반환.
    """
    prices = Daily_prices_naver(code=code, pages=pages)
    prices = _add_moving_averages(prices, windows=(20, 50))
    return prices

if __name__ == "__main__":
    code = input("종목 코드를 입력하세요 (예: 삼성전자 005930): ")
    pages = int(input("몇 페이지를 가져올까요? (예: 20): "))

    df = stock_calculator(code=code, pages=pages)
    print(df.tail())