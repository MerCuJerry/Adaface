from face_hnfnu.AdaFaceFeature import AdaFaceFeature
from face_hnfnu.FaceDatabase import FaceDatabase
from multiprocessing import Pool
import asyncio
from face_hnfnu.Config import server_config

class AdafaceServer():
    ada_face_feature: AdaFaceFeature
    face_database: FaceDatabase

    async def startup_event(self):
        self.ada_face_feature = AdaFaceFeature(config=server_config)
        self.face_database = FaceDatabase(config=server_config)
        self.pool = Pool(processes=server_config.THREAD_COUNT)
        self.ada_face_feature.load_pretrained_model()
    
    async def shutdown_event(self):
        self.face_database.saveDatabase()
        self.pool.close()
        self.pool.join()
        await asyncio.sleep(1)

adaface = AdafaceServer()