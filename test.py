import numpy as np
import net
import torch
import utils
from imutils import paths
from face_alignment import align
from AdaFaceFeature import AdaFaceFeature
from FaceDatabase import FaceDatabase

if __name__ == '__main__':
    # 创建一个 FaceDatabase 实例
    face_db = FaceDatabase()
    print("获取特征开始")
    adaface =  AdaFaceFeature() # 创建一个 Adaface 实例
    adaface.load_pretrained_model()
    #处理过的人脸照片，以做面部对齐处理,大小： 112*112
    dir_path = "./test_reg/"
    #byte 测试
    for img_path in paths.list_images(dir_path):
        print(img_path)
        feature = adaface.byte_get_represent(img_path)
        #print("feature = ",feature.detach().numpy()[0])
        #print("feature.shape = ",len(feature.detach().numpy()[0]))
        face_db.addFace(img_path, feature.detach().numpy()[0]) 
        #print(utils.feature2json(feature))
        #print(feature[0].size())
    
    
    #query_vector, norm = adaface.model(torch.randn(2,3,112,112))
    
    test_path = "./test/img1.jpg"
    query_vector = adaface.byte_get_represent(test_path)
    #test_path = "./mtcnn/7cc64_147.88.jpg"
    #feature = adaface.byte_get_represent(utils.get_image_path_to_byte(test_path))
    #print(feature == query_vector)
    #query_vector = adaface.byte_get_represent(utils.get_image_path_to_byte())
    
    #人脸验证
    threshold = -1000000000000000000
    result = face_db.searchSimilarFaces(query_vector.detach().numpy()[0], threshold)
    if result:
        similar_id, distance = result
        print(f"The most similar face is {similar_id} with distance {distance}")
    else:
        print("No similar face found")