import os
import numpy as np
import pandas as pd
from faceRecognize import compare_faces

QUERY_PROJECT = "test004"
TARGET_PROJECT = "test001"
VECTOR_DB = "vector_db"
TOP_K = 3

query_dir = os.path.join(VECTOR_DB, QUERY_PROJECT)

if __name__ == "__main__":
    records = []
    for filename in os.listdir(query_dir):
        if filename.endswith(".npy"):
            entity_id = os.path.splitext(filename)[0]
            query_path = os.path.join(query_dir, filename)

            try:
                query_embedding = np.load(query_path)
            except Exception as e:
                print(f"❌ 加载失败: {filename}")
                continue

            print(f"\n🗝️🗝️ 查询实体: {entity_id}")

            try:
                results = compare_faces(
                    query_embedding,
                    db_dir=VECTOR_DB,
                    projectCode=TARGET_PROJECT,
                    top_k=TOP_K
                )
                print("比对返回结果样例：", results[0])
            except Exception as e:
                print(f"❌ 比对失败: {entity_id}")
                continue

            for rank, r in enumerate(results, start=1):
                records.append({
                    "查询实体": entity_id,
                    "匹配实体": r["entityId"],
                    "相似度": round(r["confidence"], 4),
                    "数据库": r["DbName"],
                    "rank": rank
                })

    # 保存 Excel
    df = pd.DataFrame(records)
    df.to_excel("比对结果.xlsx", index=False, engine='openpyxl')
    print("✅ 比对结果已保存为 Excel 文件：比对结果.xlsx")
            # 输出比对结果

