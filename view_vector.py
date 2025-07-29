import numpy as np

embedding = np.load("vector_db/test001/lj2.npy")

print("向量 shape:", embedding.shape)
print("向量内容（前50维）:", embedding[:50])