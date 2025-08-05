#port 7000 for main compare and detect
import base64
import io
from contextlib import asynccontextmanager
from http.client import HTTPException
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
import os
from fastapi import FastAPI,Request
from pydantic import BaseModel
from alibabacloud_facebody20191230.client import Client
from alibabacloud_facebody20191230.models import SearchFaceAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions
from starlette.responses import JSONResponse
import threading




#确保只初次加载模型，防止反复加载
@asynccontextmanager
async def lifespan(app: FastAPI):
    face_model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_model.prepare(ctx_id=-1)

    app.state.face_model = face_model
    app.state.recognition_model = face_model.models['recognition'] #使用 ArcFaceONNX
    yield

#模型初始化
def init_models():
    face_model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_model.prepare(ctx_id=-1)
    return face_model, face_model.models['recognition']

# FastAPI初始化
app = FastAPI(lifespan=lifespan)
app.state.cached_token = None
app.state.token_expire_time = 0

# 接收请求体
class ImageInput(BaseModel):
    faceImage: str
    projectCode: str

#异步保存
def save_image_async(path, image):
    threading.Thread(target=cv2.imwrite, args=(path, image)).start()

#转换base64为图片
def base64_to_cv_image(base64_str):
    # 去掉前缀
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

#tenengrad函数
def tenengrad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0.5)  # 降低光照影响
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) #对x轴求导，卷积核3*3
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) #对y轴求导，卷积核3*3
    g = np.sqrt(gx ** 2 + gy ** 2) #计算magnitude
    return np.mean(g) #计算E(X)

#laplacian函数
def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() #方差


# 统一归一化到 0~100，标准化3个方法，使其在同一维度进行比较
def normalize_score(score, min_val, max_val):
    score = np.clip(score, min_val, max_val) # score<min -> 0, score>max -> max
    return 100 * (score - min_val) / (max_val - min_val)

#面积函数
def area_score(image, min_area=60*60, max_area=200*200 , base_score=100.0):
    h, w = image.shape[:2]
    area = h * w
    if area < min_area:
        return 0.0  # 面积太小不给分
    area = np.clip(area, min_area, max_area)
    ratio = (area - min_area) / (max_area - min_area)
    bonus_score = ratio * base_score  # 映射到 0 ~ base_score
    return bonus_score

# 清晰度核心判断
def is_clear(image, threshold=30.0, min_size=50, method='tenengrad',return_score=False):
    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        print("❌ 被判为模糊：尺寸过小")
        return (False, 0.0) if return_score else False
    score = None

    if method == 'laplacian':
        score = laplacian_score(image)
    elif method == 'tenengrad':
        score = tenengrad(image)
    elif method == 'combined':
        ten = tenengrad(image)
        lap = laplacian_score(image)
        area = area_score(image,base_score=100.0)

        ten_norm = normalize_score(ten, 20, 40)
        lap_norm = normalize_score(lap, 30, 100)
        area_norm = area  # 已是0~100
        #阳光下lap容易虚高，比值调低，当前公式适应大部分情况，暂时忽略截取图又大又不清晰这一情况
        score = 0.3* ten_norm + 0.1 * lap_norm +0.6*area_norm# 权重比值
        print(f" 🔍 Tenengrad: {ten:.2f}, Laplacian: {lap:.2f}, Area: {area:.2f}",)
        print(f" 🎉 Tenengrad_norm: {ten_norm:.2f}, Laplacian-norm: {lap_norm:.2f}, Area_norm: {area_norm:.2f}", )
    else:
        raise ValueError("method must be one of: 'laplacian', 'tenengrad', or 'combined'")

    print(f" 📏 清晰度评分 ({method}): {score:.2f}")
    return (score > threshold, score) if return_score else (score > threshold)


#log日志记录自动累加
def get_next_face_index(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    existing = [f for f in os.listdir(log_dir) if f.startswith("face_") and f.endswith(".jpg")]
    indexes = []

    for name in existing:
        try:
            idx = int(name.split("_")[1].split(".")[0])
            indexes.append(idx)
        except:
            continue

    return max(indexes) + 1 if indexes else 1


#扩充截取矩形，避免图像裁剪小被过滤，图像不要包含边框
def expand_bbox(x1, y1, x2, y2, image_shape, margin=10):
    h, w = image_shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return x1, y1, x2, y2

#检查倾斜角度是否符合标准
def is_frontal_face(face, image, yaw_thresh=40, pitch_thresh=40, blur_thresh=30.0,
                    method='combined', visualize=False, save_dir="Human_faces", face_index=None):
    yaw, pitch, roll = face.pose
    x1, y1, x2, y2 = map(int, face.bbox)
    x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, image.shape, margin=10)
    face_crop = image[y1:y2, x1:x2]
    print(f"🔹 角度 yaw={yaw:.2f}, pitch={pitch:.2f}")
    print(f"🔹 face_crop 尺寸: {face_crop.shape[:2]}")

    timestamp = int(time.time() * 1000)
    face_id = f"{face_index}" if face_index is not None else f"{timestamp}"

    # 保存用于上传人脸库的图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"face_bbox_{face_id}.jpg")
    cv2.imwrite(save_path, face_crop)
    print(f"💾 已保存用于人脸库上传：{save_path}")

    # 显示图像聚焦点（可视化）
    if visualize:
        img_with_bbox = image.copy()
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Face Crop", face_crop)
        cv2.imshow("Image with Face BBox", img_with_bbox)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 进行清晰度判断
    is_clear_result,final_score = is_clear(face_crop, threshold=blur_thresh, method=method,return_score=True)
    passed = abs(yaw) < yaw_thresh and abs(pitch) < pitch_thresh and is_clear_result

    # 保存日志图像（通过和过滤）
    log_dir = "logs/passed_faces" if passed else "logs/filtered_faces"
    os.makedirs(log_dir, exist_ok=True)
    face_index = get_next_face_index(log_dir)
    if face_index is not None:
        log_filename = f"face_{face_index}_{final_score:.2f}_bbox.jpg"
    else:
        log_filename = f"face_{int(time.time() * 1000)}_{final_score:.2f}_bbox.jpg"
    log_path = os.path.join(log_dir, log_filename)
    cv2.imwrite(log_path, face_crop)
    print(f"📝 日志图像已保存至：{log_path}")
    return passed

#多倍率人脸检测，用来检查是否为人类
def detect_faces_with_scaling(image, face_model, scales=(1.0, 1.5, 2.0)):
    for scale in scales:
        resized = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1.0 else image.copy()
        faces = face_model.get(resized)
        if faces:
            print(f"✅ 是人脸（缩放倍率：{scale}x）")
            return faces, resized, scale
    print("❌ 所有缩放倍率均检测失败，不包含人脸特征")
    return [], image, 1.0

# def imread_unicode(path):
#     """兼容中文路径的图像读取函数"""
#     stream = np.fromfile(path, dtype=np.uint8)
#     image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
#     return image
#
# def imwrite_unicode(filename, image):
#     """兼容中文路径的图像保存函数"""
#     ext = os.path.splitext(filename)[1]  # 例如 ".jpg"
#     success, encoded_img = cv2.imencode(ext, image)
#     if success:
#         encoded_img.tofile(filename)
#     else:
#         raise ValueError(f"❌ 图像编码失败，无法保存: {filename}")

#如果通过检验再调用特征向量提取和比对，否则直接pass
#提取人脸特征向量

# def extract_embedding(face_img,recognition_model):
#     face_img = cv2.resize(face_img, (112, 112))  #规范到insightface输入图像
#     face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2BGR) #色彩空间转换
#     face_img = np.transpose(face_img, (2, 0, 1))  # (3,112,112)
#     face_img = np.expand_dims(face_img, axis=0)   # (1,3,112,112)
#     face_img = face_img.astype("float32")
#     face_embedding = recognition_model.forward(face_img)[0] #提取特征512维向量
#     return face_embedding / np.linalg.norm(face_embedding) #计算向量L2平方和根号，即向量X/||x||,得到向量X的方向
#
# #人脸库基于人员照片库/当前需要的比较对象库去裁剪,并且保存特征向量
# def batch_extract_embeddings(image_dir, db_root_dir, project_name,
#                              face_detector, extract_embedding, recognition_model,
#                              is_cropped=False):
#     db_dir = os.path.join(db_root_dir, project_name)
#     os.makedirs(db_dir, exist_ok=True)
#
#     for filename in os.listdir(image_dir):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#
#         img_path = os.path.join(image_dir, filename)
#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"❌ 无法读取图像: {filename}")
#             continue
#
#         # 如果图片未裁剪，裁剪
#         if not is_cropped:
#             faces = face_detector.get(image)
#             if not faces:
#                 print(f"❌ 未检测到人脸: {filename}")
#                 continue
#
#             face = faces[0]
#             x1, y1, x2, y2 = map(int, face.bbox)
#             x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, image.shape, margin=10)
#             face_crop = image[y1:y2, x1:x2]
#         else:
#             face_crop = image  # 图像已是裁剪人脸
#
#         try:
#             embedding = extract_embedding(face_crop, recognition_model)
#             entity_id = os.path.splitext(filename)[0]
#
#             np.save(os.path.join(db_dir, f"{entity_id}.npy"), embedding)
#             cv2.imwrite(os.path.join(db_dir, f"{entity_id}.jpg"), face_crop)
#
#             print(f"✅ 成功保存: {entity_id}")
#         except Exception as e:
#             print(f"❌ 特征提取失败: {filename} -> {e}")

#人脸位置数组替代Face_crop,face_crop无法统一图片尺寸到训练模板，用来做检验合适，不适合做人脸比对提取前的处理
#[x1,y1],[x2,y2]...[x5,y5]
REFERENCE_FIVE_POINTS = np.array([ #5个特征位置超参数，可调整
    [38.2946, 51.6963], # 左眼中心
    [73.5318, 51.5014], # 右眼中心
    [56.0252, 71.7366], # 鼻尖
    [41.5493, 92.3655], # 左嘴角
    [70.7299, 92.2041] # 右嘴角
], dtype=np.float32)

#将五个特征对齐图像人脸，输出112*112的图
def align_face_by_landmark(image, landmarks, output_size=(112, 112)):
    landmarks = np.array(landmarks, dtype=np.float32)
    tfm = cv2.estimateAffinePartial2D(landmarks, REFERENCE_FIVE_POINTS, method=cv2.LMEDS)[0]
    #计算仿射变换矩阵，LMEDS提高对异常点的容忍度,[0]取M，p=A*vertical matrix[x y 1]
    aligned = cv2.warpAffine(image, tfm, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    #应用仿射变换矩阵，flag双线性插值，平滑缩放效果，borderMode超出边界像素填充方式，获得对齐后的图像,
    return aligned

def extract_embedding(face_img, recognition_model):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # BGR → RGB
    face_img = cv2.resize(face_img, (112, 112))    #统一到尺寸112，112，和训练模型一致
    face_img = np.transpose(face_img, (2, 0, 1))           # CHW
    face_img = np.expand_dims(face_img, axis=0)           # NCHW
    face_img = face_img.astype("float32")
    embedding = recognition_model.forward(face_img)[0]    # 提取特征
    return embedding / np.linalg.norm(embedding)          # L2归一化，x/||x||即x^

def batch_extract_embeddings(image_dir, db_root_dir, projectCode,
                             face_detector, recognition_model,entityID,force_override: bool = False,get_entity: bool=False):
    db_dir = os.path.join(db_root_dir, projectCode)
    os.makedirs(db_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if not get_entity:
            entity_id = os.path.splitext(filename)[0]
            vector_path = os.path.join(db_dir, f"{entity_id}.npy")
        else:
            entity_id = entityID
            vector_path = os.path.join(db_dir, f"{entity_id}.npy")  # ✅ 向量保存路径

        if os.path.exists(vector_path) and not force_override:
            print(f"⏭️ 已存在向量文件，跳过: {entity_id}")
            continue

        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 无法读取图像: {filename}")
            continue

        # 人脸检测
        faces = face_detector.get(image)
        if not faces:
            print(f"❌ 未检测到人脸: {filename}")
            continue

        face = faces[0]
        if not hasattr(face, "kps"):
            print(f"❌ 无法获取五点关键点: {filename}")
            continue

        try:
            # 五点对齐，face.kps找当前人脸五点坐标
            aligned_face = align_face_by_landmark(image, face.kps)

            # 提取特征向量
            embedding = extract_embedding(aligned_face, recognition_model)

            # 保存向量与对齐图像
            np.save(os.path.join(db_dir, f"{entity_id}.npy"), embedding)
            cv2.imwrite(os.path.join(db_dir, f"{entity_id}.jpg"), aligned_face)

            print(f"✅ 成功保存: {entity_id}")

        except Exception as e:
            print(f"❌ 特征提取失败: {filename} -> {e}")


def map_confidence(sim: float, second_best: float = 0.0) -> float:
    # 不映射的区域
    if sim <= 0.45 or sim >= 0.9:
        return round(sim, 4)

    diff = sim - second_best
    if diff >= 0.3 and sim>=0.4:
        # ✅ 无论 sim 是不是高，只要差距够大，也认为很自信
        mapped = 0.8 + (sim - 0.45) * (0.1 / 0.45)

    elif diff>=0.25:
        mapped = 0.7 + (sim - 0.45) * (0.2 / 0.45)

    elif sim >= 0.45:

        if diff>=0.20:
            mapped = 0.7 + (sim - 0.45) * (0.2 / 0.45)

        elif diff >= 0.15:
            mapped = 0.6 + (sim - 0.45) * (0.3 / 0.45)

        else:
            mapped = 0.5 + (sim - 0.45) * (0.4 / 0.45)
    else:
        # ❌ 分数低、差距也不大：保留原值
        return round(sim, 4)

    return round(mapped, 4)

#1:N人脸比对
def compare_faces(query_emb: np.ndarray, db_dir: str, projectCode, top_k=3):
    db_embeddings = [] #特征向量
    db_ids = [] #confidence序号
    db_dir_vector = os.path.join(db_dir, projectCode)
    for vector in os.listdir(db_dir_vector):
        if vector.endswith(".npy"):
            emb = np.load(os.path.join(db_dir_vector, vector))#遍历人脸库特征向量
            db_embeddings.append(emb)
            db_ids.append(os.path.splitext(vector)[0]) #获取分割后的向量名字a，（a,.npy）

    db_matrix = np.vstack(db_embeddings)  # shape: (N, 512)

    # 计算相似度（已归一化，可用点积）
    similarities = db_matrix @ query_emb  # X^*Y^

    # 获取 Top-K 相似向量索引
    top_k_indices = similarities.argsort()[::-1][:top_k]

    print(f"🗝️\n查询人脸向量比对结果:")
    result = []
    for i, idx in enumerate(top_k_indices):
        sim = float(similarities[idx])
        if i == 0 and len(top_k_indices) > 1:
            second_best = float(similarities[top_k_indices[1]])
            mapped_sim = map_confidence(sim, second_best=second_best)
        else:
            mapped_sim = round(sim, 4)
        matched_id = db_ids[idx]
        # score: {sim: .4f}
        print(f"  entityId: {matched_id}  confidence:{mapped_sim} DbName: {projectCode}",)
        result.append({"entityId": matched_id, "DbName": projectCode,"confidence":round(mapped_sim,4)})

    return result


#将human_faces中的图片做特征向量提取保存在当前文件



#构建余弦相似度函数



#特征向量循环db_name 一一比对，返回置信度


#将输入图片截取后提取特征向量，与对应人脸数据库编号的向量库中提取所有向量进行比较，k = top3


#后续将main替换为infer
# if __name__ == "__main__":
#     # 初始化模型（CPU）
#     app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
#     app.prepare(ctx_id=-1)
#
#     # 读取图像
#     img = cv2.imread("image/test34.jpg")
#     if img is None:
#         raise FileNotFoundError("❌ 图像读取失败")
#
#     # 多倍率检测
#     Human_faces, img_rescaled, used_scale = detect_faces_with_scaling(img, app)
#
#     for i, face in enumerate(Human_faces):
#         print(f"\n🔍 Face #{i + 1}")
#         if is_frontal_face(face, img_rescaled, blur_thresh=25.0, method='tenengrad', visualize=True,face_index=i):
#             print("✅ 是清晰正脸（可用于识别）")
#             #将图片与人脸库比对后续
#         else:
#             print("❌ 非正脸（角度或清晰度不达标）")

#
#调用人脸库

#定义infer
@app.post("/infer")
async def infer(input_data: ImageInput):
    projectCode = input_data.projectCode.strip()
    #检测用face-crop,人脸比对提取用核心人脸位置对齐，不与存入的crop检验图片冲突.重新创建位置存原始图片
    image = base64_to_cv_image(input_data.faceImage)#BASE64转化图片

    if image is None:
        raise FileNotFoundError("❌ 图像读取失败")

    face_model, recognition_model = init_models() #初始化模型

    # 多倍率检测
    faces, img_rescaled, used_scale = detect_faces_with_scaling(image, face_model)
    print("faces 的类型是：", type(faces))
    if not faces:
        return {"ret": 300, "msg": "检测未包含人脸特征"}#输出
    for i, face in enumerate(faces):
        print(f"\n🔍 Face #{i + 1}")
        # 可在这里调节阈值，限制即使lap和ten满分，但图过于小也没有参考意义
        if is_frontal_face(face, img_rescaled, blur_thresh=40.0, method='combined', visualize=False, face_index=i):
            print("✅ 是清晰正脸（可用于识别）")

            # 通过检验后将原始图片存入文件，用于特征向量
            save_dir = "face_origin"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"face.jpg")
            cv2.imwrite(save_path, image)
            print(f"😀 已保存用于五点特征提取：{save_path}")
            #更新特征向量库，vector_db,如果有新的数据源



            #提取输入图片中的特征向量，face_origin文件始终保持最新内容
            try:
                batch_extract_embeddings(
                    image_dir="face_origin",
                    db_root_dir="face_origin/face_vector",
                    projectCode="",
                    entityID="",
                    face_detector=face_model,
                    recognition_model=recognition_model,
                    force_override=True
                )
                query_embedding = np.load("face_origin/face_vector/face.npy")
                result =compare_faces(query_embedding, db_dir="vector_db", projectCode=projectCode, top_k=5)
                return JSONResponse(content={"ret": 200, "msg": "识别成功","data":result},status_code=200)

            except Exception as error:
                print(error)
                return JSONResponse(content={"ret": 500, "msg": "人脸搜索失败"})





            #通过检验，调用阿里人脸库
            # config = Config(
            #     access_key_id="",
            #     access_key_secret="",
            #     endpoint='facebody.cn-shanghai.aliyuncs.com',
            #     region_id='cn-shanghai'
            # )
            # runtime_option = RuntimeOptions()
            # image_path = os.path.join("Human_faces", f"face_bbox_{i}.jpg")
            # with open(image_path, 'rb') as stream0:
            #     search_face_request = SearchFaceAdvanceRequest()
            #     search_face_request.image_url_object = stream0
            #     search_face_request.db_name = projectCode
            #     search_face_request.limit = 5
            #
            #     try:
            #         client = Client(config)
            #         response = client.search_face_advance(search_face_request, runtime_option)
            #         print(response.body)
            #         return JSONResponse(content={
            #             "ret": 200,
            #             "msg": "识别成功",
            #             "data": response.body.to_map()
            #         }, status_code=200)
            #     except Exception as error:
            #         print(error)
            #         return JSONResponse(content={"ret": 500, "msg": "阿里云人脸搜索失败"})
        else:
            print("❌ 非正脸（角度或清晰度不达标）")
            return JSONResponse(content={"ret": 300, "msg": "人脸未通过检验，过于模糊或非正脸"})



