# ------------------------
# 1. HTTP 요청 & 파싱
# ------------------------
import requests
from bs4 import BeautifulSoup
from lxml import etree

# ------------------------
# 2. 데이터 처리 & 변환
# ------------------------
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import pytz  # 또는 from zoneinfo import ZoneInfo

# ------------------------
# 3. 저장 (Parquet / SQLite)
# ------------------------
import pyarrow as pa
import pyarrow.parquet as pq
import sqlite3

# ------------------------
# 4. 스케줄링 & CLI
# ------------------------
import schedule
# 또는
# from apscheduler.schedulers.background import BackgroundScheduler

# ------------------------
# 5. 검증 & 로깅
# ------------------------
from pydantic import BaseModel, validator
from loguru import logger

# ------------------------
# 6. 시각화 (옵션)
# ------------------------
import matplotlib.pyplot as plt
import mplfinance as mpf