import base64
import json
import requests
from onnxruntime.transformers.shape_infer_helper import file_path

# 1. 读取图片并转 base64
# file_path = './image/100005.jpg'
file_path = 'image_dir/test002/tyx14.jpg'
with open(file_path, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

# 2. 接口地址（本地 FastAPI 服务）
url = "http://127.0.0.1:7777/infer"

payload = {
    "faceImage": img_base64,
    "projectCode": "test001" #项目数据库编号
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
