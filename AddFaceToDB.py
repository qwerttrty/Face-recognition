#port 7001 for add,构建fastapi
#payload:base64pic, ProjectCode, EntityID
#ProjectCode 没有即在vector_db创建新文件夹
#引用base64pic-——>pic，pic存额外文件夹（非vector_db）,pic存完清空temp
#引用recognition,仿射变换矩阵 112*112根据EntityID 命名图片文件和npy向量
#batch_extract_embedding构建向量库和112*112照片库

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
import os
import cv2
from faceRecognize import (
    base64_to_cv_image,
    batch_extract_embeddings,
    init_models
)

face_model, recognition_model = init_models()
app = FastAPI()

TEMP_IMAGE_DIR = "face_temp"
VECTOR_DB_ROOT = os.path.abspath("vector_db")  # 确保是绝对路径


class AddFaceRequest(BaseModel):
    faceImage: str
    projectCode: str
    entityID: str

@app.post("/AddFace")
def add_face(req: AddFaceRequest):
    entity_id = req.entityID.strip()
    project_code = req.projectCode.strip()
    base64_image = req.faceImage.strip()

    #检验输入
    if not all([entity_id, project_code, base64_image]):
        raise HTTPException(status_code=500, detail="缺少必要字段")


    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

    #添加新的projectCode文件夹
    # target_dir 仍需路径安全校验 ✅
    target_dir = os.path.abspath(os.path.join(VECTOR_DB_ROOT, project_code))
    if not target_dir.startswith(VECTOR_DB_ROOT + os.sep):
        # raise HTTPException(status_code=400, detail="非法的 ProjectCode 路径")
        return JSONResponse(content={"ret": 500, "msg": "非法的 ProjectCode 路径"})

    # ✅ 如果目录不存在，则创建；如果已存在，继续使用（允许添加/覆盖 entityID）
    os.makedirs(target_dir, exist_ok=True)

    # 将 base64 图像转为 cv2 图像并保存为临时原图
    try:
        image = base64_to_cv_image(base64_image)
        temp_image_path = os.path.join(TEMP_IMAGE_DIR, f"{entity_id}.jpg")
        cv2.imwrite(temp_image_path, image)
    except Exception as e:
        # raise HTTPException(status_code=400, detail=f"图像解码失败: {e}")
        return JSONResponse(content={"ret": 500, "msg": "图像解码失败"})
    try:
        batch_extract_embeddings(
            image_dir=TEMP_IMAGE_DIR,
            db_root_dir=VECTOR_DB_ROOT,
            projectCode=project_code,
            entityID=entity_id,
            face_detector=face_model,
            recognition_model=recognition_model,
            force_override=True #覆盖同名entity图和向量，用作更新
        )
    except Exception as e:
        # raise HTTPException(status_code=500, detail=f"特征提取失败: {e}")
        return JSONResponse(content={"ret": 500, "msg": "特征提取失败"})

    try:
        os.remove(temp_image_path)
    except:
        pass

    return {
        "ret":200,
        "msg": "人脸添加成功",
        "entityID": entity_id,
        "projectCode": project_code
    }

