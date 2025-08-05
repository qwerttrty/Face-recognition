import base64
import json
import requests


file_path = 'image_dir/test002/lqh.png'
with open(file_path, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

# 2. 接口地址（本地 FastAPI 服务）
url = "http://127.0.0.1:7001/AddFace"

payload = {
    "faceImage": img_base64,
    "projectCode": "test004", #项目数据库编号
    "entityID":"lqh60203251"
}

# 4. 发起 POST 请求
response = requests.post(url, json=payload)

# 5. 打印响应结果
print("状态码:", response.status_code)

try:
    print("返回内容:", json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("⚠️ 返回内容解析失败:", e)
    print("原始响应文本:", response.text)