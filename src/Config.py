from .AdaFaceFeature import AdaFaceFeature
from .FaceDatabase import FaceDatabase
from multiprocessing import Pool
from pathlib import Path
import asyncio

DATABASE_PATH = Path.cwd() / "data" / "face_db.bin"

class Config:
    ada_face_feature: AdaFaceFeature
    face_database: FaceDatabase

    async def startup_event(self, path = DATABASE_PATH):
        self.ada_face_feature = AdaFaceFeature()
        self.face_database = FaceDatabase()
        if DATABASE_PATH.exists():
            self.face_database.loadDatabase(DATABASE_PATH)
        self.pool = Pool(processes=12)
        self.ada_face_feature.load_pretrained_model()
    
    async def shutdown_event(self):
        self.face_database.saveDatabase(self.DATABASE_PATH)
        self.pool.close()
        self.pool.join()
        asyncio.sleep(1)

config = Config()