from multiprocessing import Pool, set_start_method
import signal
from face_hnfnu.AdaFaceFeature import AdaFaceFeature
from face_hnfnu.FaceDatabase import FaceDatabase
from face_hnfnu.Config import server_config


class ProcPool:
    def startup_event(self):
        """
        Called when server is starting up. Initializes the multiprocessing pool.
        """
        set_start_method("spawn")
        self.pool = Pool(
            processes=server_config.THREAD_COUNT,
            initializer=signal.signal,
            initargs=(signal.SIGINT, signal.SIG_IGN),  # ignore sigint signal
        )

    def shutdown_event(self):
        """
        Called when server is shutting down
        """
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

    def verify_face(self, img, threshold) -> tuple:
        try:
            result = self.face_database.searchSimilarFaces(
                self.ada_face_feature.byte_get_represent(img).detach().numpy(), threshold
            )
        except Exception as err:
            result = (None, err)
        return result


adaface = AdafaceServer()
procpool = ProcPool()
