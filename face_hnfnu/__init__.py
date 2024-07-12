from multiprocessing import Pool, set_start_method
import signal
from face_hnfnu.AdaFaceFeature import AdaFaceFeature
from face_hnfnu.FaceDatabase import FaceDatabase
from face_hnfnu.Config import server_config


class ProcPool:
    def startup_event(self):
        set_start_method("spawn")
        self.pool = Pool(
            processes=server_config.THREAD_COUNT,
            initializer=signal.signal,
            initargs=(signal.SIGINT, signal.SIG_IGN),
        )

    def shutdown_event(self):
        self.pool.terminate()
        self.pool.join()


class AdafaceServer:
    ada_face_feature: AdaFaceFeature
    face_database: FaceDatabase

    def startup_event(self):
        self.ada_face_feature = AdaFaceFeature(config=server_config)
        self.face_database = FaceDatabase(config=server_config)
        self.ada_face_feature.load_pretrained_model()

    def shutdown_event(self):
        self.face_database.saveDatabase()

    def verify_face(self, img, threshold):
        q_vec = self.ada_face_feature.byte_get_represent(img)
        result = self.face_database.searchSimilarFaces(q_vec.numpy()[0], threshold)
        return result


adaface = AdafaceServer()
procpool = ProcPool()
