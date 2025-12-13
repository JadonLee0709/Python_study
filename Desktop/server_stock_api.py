# server_stock_api.py
# Stock analysis API for iOS (Route A): Python server + iOS app(SwiftUI)

import re
import time
from typing import Optional, List, Dict, Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =============================
# Config
# =============================
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
SEARCH_HEADERS = {**HEADERS, "Referer": "https://search.naver.com/"}

FIXED_WINDOWS = (5, 20, 60, 120)

CODE_RE = re.compile(r"\b(\d{6})\b")

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

# 간단 캐시 (서버 부하/네이버 차단 방지용)
_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SEC = 60  # 같은 code/payload 요청이 60초 내면 캐시 반환


# =============================
# Utils: code resolving
# =============================
def pick_code_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = CODE_RE.search(text)
    return m.group(1) if m else None


def get_code_from_naver_search(query: str) -> Optional[str]:
    url = f"https://search.naver.com/search.naver?query={quote(query)}"
    try:
        r = requests.get(url, headers=SEARCH_HEADERS, timeout=12)
        r.raise_for_status()
    except Exception:
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
        # 중복 제거 후 첫 번째
        seen = []
        for c in m:
            if c not in seen:
                seen.append(c)
        return seen[0] if seen else None

    return None


def resolve_to_code(user_input: str) -> Optional[str]:
    s = (user_input or "").strip()
    c = pick_code_from_text(s)
    if c:
        return c
    if s in FALLBACK_MAP:
        return FALLBACK_MAP[s]
    return get_code_from_naver_search(s)


# =============================
# Crawling (Thread-based)
# =============================
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


def daily_prices_naver(code: str, pages: int = 30, workers: int = 8) -> pd.DataFrame:
    pages = max(1, int(pages))
    workers = max(1, int(workers))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_fetch_page, code, p) for p in range(1, pages + 1)]
        results = []
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass

    frames = [df for df in results if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["Date", "Close", "Open", "High", "Low", "Volume"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out


# =============================
# Indicators / Signals / Analysis
# =============================
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for w in FIXED_WINDOWS:
        out[f"MA{w}"] = out["Close"].rolling(window=w, min_periods=w).mean()
    return out


def add_derivative_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["d1_Close"] = out["Close"].diff()
    out["d2_Close"] = out["d1_Close"].diff()

    for w in FIXED_WINDOWS:
        col = f"MA{w}"
        if col in out.columns:
            out[f"d1_{col}"] = out[col].diff()
            out[f"d2_{col}"] = out[f"d1_{col}"].diff()
    return out


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
    return "⚖️ 특별한 신호 없음"


def detect_local_extrema(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    n = len(work)
    is_peak = [False] * n
    is_trough = [False] * n

    window = max(1, int(window))
    for i in range(window, n - window):
        center = work.at[i, "Close"]
        left = work.at[i - window, "Close"]
        right = work.at[i + window, "Close"]

        if center > left and center >= right:
            is_peak[i] = True
        if center < left and center <= right:
            is_trough[i] = True

    work["is_peak"] = is_peak
    work["is_trough"] = is_trough
    return work


def add_trend_labels(df: pd.DataFrame) -> pd.DataFrame:
    work = add_moving_averages(df.copy())
    work = add_derivative_features(work)

    labels = []
    for _, row in work.iterrows():
        d1_ma60 = row.get("d1_MA60", 0)
        d1_ma120 = row.get("d1_MA120", 0)
        ma60 = row.get("MA60", np.nan)
        ma120 = row.get("MA120", np.nan)

        label = "transition"
        if pd.notna(ma60) and pd.notna(ma120):
            if (d1_ma60 > 0) and (d1_ma120 > 0) and (ma60 > ma120):
                label = "bull"
            elif (d1_ma60 < 0) and (d1_ma120 < 0) and (ma60 < ma120):
                label = "bear"
        labels.append(label)

    work["trend_label"] = labels
    return work


def math_analysis_report(df: pd.DataFrame, horizon: int = 5) -> str:
    if df.empty or len(df) < 150:
        return "데이터가 부족하여 수학적 분석을 수행할 수 없습니다. (최소 150개 이상 필요)"

    horizon = max(1, int(horizon))

    work = add_derivative_features(df.copy())
    future_col = f"future_ret_{horizon}"
    work[future_col] = work["Close"].shift(-horizon) / work["Close"] - 1.0

    cond_regime = (
        (work.get("d1_MA60", 0) > 0)
        & (work.get("d1_MA20", 0) > 0)
        & (work.get("MA20", 0) > work.get("MA60", 0))
    )

    cond_pullback = (
        (work.get("d1_MA5", 0) < 0)
        & (work["Close"] < work.get("MA5", work["Close"] * 0 + float("inf")))
        & (work["Close"] < work.get("MA20", work["Close"] * 0 + float("inf")))
    )

    pattern = cond_regime & cond_pullback
    pattern_df = work.loc[pattern].dropna(subset=[future_col])

    total = len(pattern_df)
    current = work.iloc[-1]
    now_regime = bool(cond_regime.iloc[-1])
    now_pullback = bool(cond_pullback.iloc[-1])

    lines: List[str] = []
    lines.append(f"[수학적 분석 요약 - 향후 {horizon}거래일 기준]")
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

    lines.append("\n2) 현재 패턴 판단:")
    if now_regime and now_pullback:
        lines.append("   - 상태: '상승 추세 속 단기 조정' 패턴에 해당합니다.")
    elif now_regime and not now_pullback:
        lines.append("   - 상태: 상승 추세는 유지 중이지만, 단기 조정 구간은 아닙니다.")
    elif (not now_regime) and now_pullback:
        lines.append("   - 상태: 단기 조정처럼 보이지만, 장기 추세(20/60 기준)가 명확한 상승은 아닙니다.")
    else:
        lines.append("   - 상태: 상승 추세도, 전형적인 단기 조정 패턴도 아닙니다.")

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
    lines.append("\n※ 위 통계는 과거 패턴 분석 결과일 뿐, 미래 수익을 보장하지 않습니다.")
    return "\n".join(lines)


# =============================
# Backtest (MA20/60)
# =============================
def _max_drawdown(equity: pd.Series) -> float:
    cum_max = equity.cummax()
    dd = (equity - cum_max) / cum_max
    return float(dd.min())


def backtest_ma20_60_strategy(
    df: pd.DataFrame,
    fee_rate: float = 0.0005,
    initial_cash: float = 10_000_000.0
) -> dict:
    work = df.copy()
    if "MA20" not in work.columns or "MA60" not in work.columns:
        work = add_moving_averages(work)
    work = add_derivative_features(work)

    for col in ["Close", "MA20", "MA60", "d1_MA20"]:
        if col not in work.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    work = work.dropna(subset=["Close", "MA20", "MA60", "d1_MA20"]).reset_index(drop=True)

    cash = float(initial_cash)
    shares = 0.0
    equity_list: List[float] = []
    position_flag: List[int] = []
    trades: List[dict] = []

    in_position = False
    entry_price = None
    entry_idx = None

    for i, row in work.iterrows():
        price = float(row["Close"])
        ma20 = float(row["MA20"])
        ma60 = float(row["MA60"])
        d1_ma20 = float(row["d1_MA20"])

        if in_position:
            if (price < ma20) or (ma20 < ma60):
                sell_price = price * (1 - fee_rate)
                cash = shares * sell_price
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": sell_price,
                    "ret": sell_price / entry_price - 1.0,
                })
                shares = 0.0
                in_position = False
                entry_price = None
                entry_idx = None
        else:
            if (price > ma20) and (ma20 > ma60) and (d1_ma20 > 0):
                buy_price = price * (1 + fee_rate)
                shares = cash / buy_price
                cash = 0.0
                in_position = True
                entry_price = buy_price
                entry_idx = i

        equity = cash + shares * price
        equity_list.append(equity)
        position_flag.append(1 if in_position else 0)

    work["equity"] = equity_list
    work["position"] = position_flag
    work["equity_ret"] = work["equity"].pct_change().fillna(0.0)

    total_return = work["equity"].iloc[-1] / initial_cash - 1.0
    mdd = _max_drawdown(work["equity"])

    mean_ret = work["equity_ret"].mean()
    std_ret = work["equity_ret"].std()
    sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

    n_trades = len(trades)
    if n_trades > 0:
        rets = np.array([t["ret"] for t in trades], dtype=float)
        win_rate = float((rets > 0).mean())
        avg_trade_ret = float(rets.mean())
    else:
        win_rate = 0.0
        avg_trade_ret = 0.0

    equity_curve = work[["Date", "equity", "position"]].copy()
    equity_curve["Date"] = equity_curve["Date"].dt.strftime("%Y-%m-%d")

    return {
        "total_return": float(total_return),
        "MDD": float(mdd),
        "sharpe": float(sharpe),
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "avg_trade_ret": float(avg_trade_ret),
        "trades": trades,
        "equity_curve": {
            "date": equity_curve["Date"].tolist(),
            "equity": equity_curve["equity"].astype(float).tolist(),
            "position": equity_curve["position"].astype(int).tolist(),
        },
    }


# =============================
# High-level analyze (for iOS)
# =============================
def stock_calculator(code: str, pages: int = 30, workers: int = 8) -> pd.DataFrame:
    prices = daily_prices_naver(code=code, pages=pages, workers=workers)
    prices = add_moving_averages(prices)
    return prices


def _cache_key(code: str, pages: int, horizon: int) -> str:
    return f"{code}|p={pages}|h={horizon}"


def analyze(code_or_name: str, pages: int = 30, workers: int = 8, horizon: int = 5,
            tail_n: int = 260, extrema_window: int = 2) -> dict:
    code = resolve_to_code(code_or_name)
    if not code:
        return {"ok": False, "error": "cannot_resolve_code", "input": code_or_name}

    pages = max(1, int(pages))
    horizon = max(1, int(horizon))
    tail_n = max(50, int(tail_n))

    key = _cache_key(code, pages, horizon)
    now = time.time()
    if key in _CACHE and (now - _CACHE[key]["ts"] < CACHE_TTL_SEC):
        return _CACHE[key]["data"]

    df = stock_calculator(code, pages=pages, workers=workers)
    if df.empty or len(df) < 10:
        return {"ok": False, "error": "no_data", "code": code}

    signal = alarm_operator(df)
    trend_df = add_trend_labels(df)
    trend_label = str(trend_df["trend_label"].iloc[-1])

    report = math_analysis_report(df, horizon=horizon)

    ext_df = detect_local_extrema(df, window=extrema_window)
    recent_ext = ext_df.tail(60)
    peaks = recent_ext[recent_ext["is_peak"]].tail(10)
    troughs = recent_ext[recent_ext["is_trough"]].tail(10)

    tail = df.tail(tail_n).copy()
    tail["Date"] = tail["Date"].dt.strftime("%Y-%m-%d")

    data = {
        "ok": True,
        "code": code,
        "signal": signal,
        "trend_label": trend_label,
        "report": report,
        "meta": {
            "pages": pages,
            "horizon": horizon,
            "tail_n": tail_n,
            "windows": list(FIXED_WINDOWS),
            "last_date": str(tail["Date"].iloc[-1]),
            "last_close": int(tail["Close"].iloc[-1]),
        },
        "series": {
            "date": tail["Date"].tolist(),
            "close": tail["Close"].astype(int).tolist(),
            "ma5": tail["MA5"].round(3).where(~tail["MA5"].isna(), None).tolist(),
            "ma20": tail["MA20"].round(3).where(~tail["MA20"].isna(), None).tolist(),
            "ma60": tail["MA60"].round(3).where(~tail["MA60"].isna(), None).tolist(),
            "ma120": tail["MA120"].round(3).where(~tail["MA120"].isna(), None).tolist(),
        },
        "extrema_recent": {
            "peaks": [
                {"date": str(row["Date"].date()), "close": int(row["Close"])}
                for _, row in peaks.iterrows()
            ],
            "troughs": [
                {"date": str(row["Date"].date()), "close": int(row["Close"])}
                for _, row in troughs.iterrows()
            ],
        },
    }

    _CACHE[key] = {"ts": now, "data": data}
    return data


# =============================
# FastAPI App
# =============================
app = FastAPI(title="Stock Analysis API (Naver)", version="1.0")

# iOS 앱에서 호출 편하게 CORS 허용 (배포 시에는 origin 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


@app.get("/analyze")
def api_analyze(
    q: str = Query(..., description="종목코드(6자리) 또는 종목명 (예: 005930 / 삼성전자)"),
    pages: int = Query(30, ge=1, le=200),
    workers: int = Query(8, ge=1, le=32),
    horizon: int = Query(5, ge=1, le=60),
    tail_n: int = Query(260, ge=50, le=2000),
):
    data = analyze(q, pages=pages, workers=workers, horizon=horizon, tail_n=tail_n)
    if not data.get("ok", False):
        raise HTTPException(status_code=400, detail=data)
    return data


@app.get("/backtest")
def api_backtest(
    q: str = Query(..., description="종목코드(6자리) 또는 종목명"),
    pages: int = Query(80, ge=1, le=300),
    workers: int = Query(8, ge=1, le=32),
    fee_rate: float = Query(0.0005, ge=0.0, le=0.01),
    initial_cash: float = Query(10_000_000.0, ge=1_000.0, le=1_000_000_000.0),
):
    code = resolve_to_code(q)
    if not code:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "cannot_resolve_code", "input": q})

    df = stock_calculator(code, pages=pages, workers=workers)
    if df.empty or len(df) < 100:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "no_data_or_too_short", "code": code})

    try:
        result = backtest_ma20_60_strategy(df, fee_rate=fee_rate, initial_cash=initial_cash)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"ok": False, "error": "backtest_failed", "message": str(e)})

    return {"ok": True, "code": code, "result": result}


# =============================
# Run (local)
# =============================
# 터미널:
#   pip install fastapi uvicorn requests pandas numpy beautifulsoup4 lxml
#   uvicorn server_stock_api:app --host 0.0.0.0 --port 8000
#
# 테스트:
#   http://localhost:8000/health
#   http://localhost:8000/analyze?q=삼성전자
#   http://localhost:8000/analyze?q=005930&pages=60
#   http://localhost:8000/backtest?q=005930