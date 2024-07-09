from face_hnfnu.AdaFaceFeature import AdaFaceFeature
from face_hnfnu.FaceDatabase import FaceDatabase
from multiprocessing import Pool as Pool, set_start_method
import asyncio
from face_hnfnu.Config import server_config


class ProcPool:
    async def startup_event(self):
        set_start_method("forkserver")
        self.pool = Pool(processes=server_config.THREAD_COUNT)

    async def shutdown_event(self):
        self.pool.close()
        self.pool.join()
        await asyncio.sleep(1)


class AdafaceServer:
    ada_face_feature: AdaFaceFeature
    face_database: FaceDatabase

    async def startup_event(self):
        self.ada_face_feature = AdaFaceFeature(config=server_config)
        self.face_database = FaceDatabase(config=server_config)
        self.ada_face_feature.load_pretrained_model()

    async def shutdown_event(self):
        self.face_database.saveDatabase()
        await asyncio.sleep(1)

    def verify_face(self, img, threshold):
        q_vec = self.ada_face_feature.byte_get_represent(img)
        result = self.face_database.searchSimilarFaces(q_vec.numpy()[0], threshold)
        return result


adaface = AdafaceServer()
procpool = ProcPool()
