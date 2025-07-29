from insightface.app import FaceAnalysis
from faceRecognize import batch_extract_embeddings, extract_embedding

def init_models():
    face_model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_model.prepare(ctx_id=-1)
    return face_model, face_model.models['recognition']


if __name__ == "__main__":
    face_detector, recognition_model = init_models()

    #提取Human_faces特征向量
    # batch_extract_embeddings(
    #     image_dir="image_compare",
    #     db_root_dir="image_compare/image_compare_vector",
    #     projectCode="",
    #     face_detector=face_detector,
    #     recognition_model=recognition_model,
    #     # is_cropped=True
    # )

    batch_extract_embeddings(
        image_dir="image_dir/test001",
        db_root_dir="vector_db",
        projectCode="test001",
        face_detector=face_detector,
        recognition_model=recognition_model,
        force_override=False
    )
