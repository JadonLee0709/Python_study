import time

print("카운트다운 시작")
for i in range(5,0,-1) :
    print(i)
    time.sleep(1)
print('발사')

from typing import List
def total(nums: List[int]) -> int :
    return sum(nums)

print(total([1,2,3]))

import requests
res = requests.get("https://jsonplaceholder.typicode.com/todos/1")
print("상태 코드 : ",res.status_code)
print("응답 JSON :",res.json())

import pandas as pd
data = {"이름" : ["철수","영희","민수"], "점수": [90,85,77]}
df = pd.DataFrame(data).rename(index=lambda x : x+1)

print(df)
print("평균 점수:",df["점수"].mean())

