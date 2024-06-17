from .AdaFaceFeature import AdaFaceFeature
from .FaceDatabase import FaceDatabase
from multiprocessing import Pool
from .__init__ import FAISS_DATABASE_PATH, INDEX_DATABASE_PATH
import asyncio

class Config:
    ada_face_feature: AdaFaceFeature
    face_database: FaceDatabase

    async def startup_event(self, faiss_path = FAISS_DATABASE_PATH, index_path = INDEX_DATABASE_PATH):
        self.ada_face_feature = AdaFaceFeature()
        self.face_database = FaceDatabase(faiss_path=faiss_path, index_path=index_path)
        self.pool = Pool(processes=12)
        self.ada_face_feature.load_pretrained_model()
    
    async def shutdown_event(self, faiss_path = FAISS_DATABASE_PATH):
        self.face_database.saveDatabase(faiss_path)
        self.pool.close()
        self.pool.join()
        asyncio.sleep(1)

config = Config()