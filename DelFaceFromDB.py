#port 7002 for del，构建fastapi
#payload:EntityID,ProjectCode
#根据输入的ProjectCode进入vector_db,删除对应前缀entityID的图和npy

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse
import os
import glob

# 数据库路径
VECTOR_DB_ROOT = "./vector_db"

# 构建FastAPI
app = FastAPI()

# 定义请求体结构
class DeleteRequest(BaseModel):
    entityID: str
    projectCode: str

@app.post("/delete")
async def delete_entity(req: DeleteRequest):
    entity_id = req.entityID.strip()
    project_code = req.projectCode.strip()

    if not all([entity_id, project_code]):
        raise HTTPException(status_code=500, detail="缺少必要字段")

    # 拼接路径，并进入对应项目的文件夹
    target_dir = os.path.join(VECTOR_DB_ROOT, project_code)

    if not os.path.exists(target_dir):
        # raise HTTPException(status_code=404, detail=f"项目 {project_code} 不存在")
        return JSONResponse(content={"ret": 500, "msg": "项目路径不存在"})

    # 构建待删除文件的前缀路径 pattern
    pattern = os.path.join(target_dir, f"{entity_id}*")

    # 找到所有匹配的文件
    files_to_delete = glob.glob(pattern)

    if not files_to_delete:
        # raise HTTPException(status_code=404, detail=f"未找到 EntityID 为 {entity_id} 的文件")
        return JSONResponse(content={"ret": 500, "msg": "当前库未找到entityID"})

    deleted_files = []
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_files.append(os.path.basename(file_path))
        except Exception as e:
            # raise HTTPException(status_code=500, detail=f"删除 {file_path} 失败: {e}")
            return JSONResponse(content={"ret": 500, "msg": "文件删除失败"})

    return {"ret":200, "msg": f"成功删除 {len(deleted_files)} 个文件", "files": deleted_files}
