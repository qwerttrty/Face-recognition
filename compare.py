from faceRecognize import compare_faces
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# img = Image.open("image_compare/image_compare_vector/z1.jpg")
# plt.imshow(img)
# plt.title("是否是对齐后的 112×112 RGB 人脸图")
# plt.axis('off')
# plt.show()
if __name__ == "__main__":
    query_embedding = np.load("image_compare/image_compare_vector/lj4.npy")
    compare_faces(query_embedding, db_dir="vector_db", projectCode="test001",top_k=10)