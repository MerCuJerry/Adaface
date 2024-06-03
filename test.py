import numpy as np
import net
import torch
import utils
from imutils import paths
from face_alignment import align
from .AdaFaceFeature import AdaFaceFeature
from .FaceDatabase import FaceDatabase

adaface_models = {
    'ir_18':"models/adaface_ir18_webface4m.ckpt",
}

def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

if __name__ == '__main__':
    # 创建一个 FaceDatabase 实例
    face_db = FaceDatabase()
    print("获取特征开始")
    adaface =  AdaFaceFeature() # 创建一个 Adaface 实例
    adaface.load_pretrained_model()
    #处理过的人脸照片，以做面部对齐处理,大小： 112*112
    dir_path = "./mtcnn/"
    #byte 测试
    for img_path in paths.list_images(dir_path):
        feature = adaface.byte_get_represent(utils.get_image_path_to_byte(img_path))
        #print("feature = ",feature.detach().numpy()[0])
        #print("feature.shape = ",len(feature.detach().numpy()[0]))
        face_db.addFace(img_path, feature.detach().numpy()[0]) 
        #print(utils.feature2json(feature))
        #print(feature[0].size())
    
    #人脸验证
    query_vector, norm = adaface(torch.randn(2,3,112,112))
    
    test_path = "./test/img1.jpg"
    aligned_rgb_img = align.get_aligned_face(test_path)
    bgr_tensor_input = to_input(aligned_rgb_img)
    query_vector, _ = adaface(bgr_tensor_input)
    #query_vector = adaface.byte_get_represent(utils.get_image_path_to_byte())
    
    
    threshold = -1000000000000000000
    result = face_db.searchSimilarFaces(query_vector.detach().numpy()[0], threshold)
    if result:
        similar_id, distance = result
        print(f"The most similar face is {similar_id} with distance {distance}")
    else:
        print("No similar face found")