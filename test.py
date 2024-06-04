from imutils import paths
from AdaFaceFeature import AdaFaceFeature
from FaceDatabase import FaceDatabase
from pathlib import Path

if __name__ == '__main__':
    # 创建一个 FaceDatabase 实例
    face_db = FaceDatabase()
    print("获取特征开始")
    adaface =  AdaFaceFeature() # 创建一个 Adaface 实例
    adaface.load_pretrained_model()
    #处理过的人脸照片，以做面部对齐处理,大小： 112*112
    #图片注册
    DIR_PATH = Path.cwd() / "test_reg"
    for img_path in paths.list_images(DIR_PATH):
        print(img_path)
        feature = adaface.byte_get_represent(img_path)
        face_db.addFace(img_path, feature.detach().numpy()[0]) 
    
    #图片检测测试
    TEST_PATH = Path.cwd() / "test" / "img1.jpg"
    query_vector = adaface.byte_get_represent(TEST_PATH) 
    
    #人脸验证
    threshold = -1000000000000000000
    result = face_db.searchSimilarFaces(query_vector.detach().numpy()[0], threshold)
    if result:
        similar_id, distance = result
        print(f"The most similar face is {similar_id} with distance {distance}")
    else:
        print("No similar face found")