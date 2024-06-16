from imutils import paths
from src.AdaFaceFeature import AdaFaceFeature
from src.FaceDatabase import FaceDatabase
from pathlib import Path
import asyncio
import multiprocessing

async def test():
    # 创建一个 FaceDatabase 实例
    face_db = FaceDatabase()
    print("获取特征开始")
    adaface =  AdaFaceFeature() # 创建一个 Adaface 实例
    adaface.load_pretrained_model()
    #处理过的人脸照片，以做面部对齐处理,大小： 112*112
    pool = multiprocessing.Pool(processes=12)
    #图片注册
    DIR_PATH = Path.cwd() / "test_reg"
    for img_path in paths.list_images(DIR_PATH):
        print(img_path)
        feature = adaface.byte_get_represent(img_path)
        face_db.addFace(img_path, feature.detach().numpy()[0]) 
    
    #图片检测测试
    TEST_PATH = Path.cwd() / "test"
    threshold = -1000
    for img_path in paths.list_images(TEST_PATH):
        print(img_path)
        query_vector = adaface.byte_get_represent(img_path)
        thisresult = pool.apply_async(face_db.searchSimilarFaces, args=(query_vector.detach().numpy()[0], threshold)).get()
        print(f"The most similar face of {img_path} is {thisresult[0]} with distance {thisresult[1]}")
        

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test()) 