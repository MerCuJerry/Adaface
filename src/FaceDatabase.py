import numpy as np
import faiss
from pathlib import Path

class FaceDatabase:
    def __init__(self, dimension=512):
        """
        初始化 FaceDatabase 类
        Parameters:
        dimension (int): 人脸向量的维度，默认为 512
        """
        self.dimension = dimension  # 保存人脸向量的维度
        self.index = faiss.IndexFlatIP(dimension)  # 创建 Faiss 的余弦相似度索引

    def addFace(self, face_id, face_vector):
        """
        将人脸向量添加进数据库
        Parameters:
        face_id: 人脸 id
        face_vector: 人脸向量
        """
        if face_vector.shape[0] != self.dimension:
            raise ValueError(
                "Face vector dimension does not match the database dimension"
            )  # 抛出维度不匹配的异常
        self.index.add_with_ids(
            np.expand_dims(face_vector, axis=0),
            xids=face_id # 记录人脸向量的 id
        )  # 向 Faiss 索引中添加人脸向量

    def searchSimilarFaces(self, query_vector, threshold) -> tuple | None:
        """
        查询相似的人脸向量
        Parameters:
        query_vector: 查询向量
        threshold: 相似度阈值
        Returns:
        tuple or None: 返回超过阈值的最相似人脸向量的id和对应的距离，如果没有超过阈值的，则返回 None
        """
        distances, indices = self.index.search(
            np.expand_dims(query_vector, axis=0), len(self.ids)
        )  # 使用 Faiss 进行搜索
        # print(distances, indices)
        if distances[0][0] > threshold:
            return self.ids[indices[0][0]], distances[0][0]
        else:
            return None  # 如果没有超过阈值的则返回 None

    def removeFaceById(self, face_id):
        """
        根据人脸 id 删除数据库内的人脸向量
        Parameters:
        face_id: 待删除的人脸 id
        """
        self.index.remove_ids(face_id)  # 从 Faiss 索引中删除人脸向量

    def __len__(self):
        """
        获取数据库内存储的人脸向量数量
        Returns:
        int: 人脸向量数量
        """
        return self.index.ntotal  # 返回保存的人脸向量数量

    def clearDatabase(self):
        """
        清空数据库，删除所有的人脸向量
        """
        self.index.reset()  # 重置 Faiss 索引
        
    def saveDatabase(self, path: Path):
        """
        保存数据库到指定路径

        Args:
            path (_type_): _description_
        """
        faiss.write_index(self.index, path.open('wb'))

    def loadDatabase(self, path):
        """
        从指定路径加载数据库

        Args:
            path (_type_): _description_
        """
        self.index = faiss.read_index(path)
