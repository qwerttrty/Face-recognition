import json
import requests

url = "http://127.0.0.1:7002/delete"

payload = {
    "entityID" : "lqh60203251", #人员编号
    "projectCode": "test004" #项目数据库编号
}

response = requests.post(url, json=payload)

print("状态码:", response.status_code)

try:
    print("返回内容:", json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("⚠️ 返回内容解析失败:", e)
    print("原始响应文本:", response.text)