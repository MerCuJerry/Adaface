import numpy as np
import faiss
import sqlite3
from pathlib import Path
from face_hnfnu.Config import ConfigModel


class FaceDatabase:
    def __init__(self, dimension=512, config: ConfigModel = None):
        """
        初始化 FaceDatabase 类
        Parameters:
        dimension (int): 人脸向量的维度，默认为 512
        """
        self.faiss_path = Path(config.FAISS_DATABASE_PATH)  # 保存 Faiss 数据库路径
        self.index_path = Path(config.INDEX_DATABASE_PATH)  # 保存人脸 id 表路径
        self.dimension = dimension  # 保存人脸向量的维度
        if self.faiss_path.exists():
            self.loadDatabase(self.faiss_path)  # 从指定路径加载数据库
        else:
            self.index = faiss.IndexFlatIP(dimension)  # 创建 Faiss 的余弦相似度索引
        if not self.index_path.is_file():
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_path.touch(exist_ok=True)
            self.query_database(
                "CREATE TABLE ids (id INTEGER PRIMARY KEY NOT NULL, name TEXT UNIQUE);",
                (),
            )  # 创建人脸 id 表

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
        try:
            sql = "INSERT INTO ids (id, name) VALUES (?,?)"
            query = self.__len__(), str(face_id)
            self.query_database(sql, query)  # 插入人脸 id 表
            self.index.add(
                np.expand_dims(face_vector, axis=0)
            )  # 向 Faiss 索引中添加人脸向量
        except sqlite3.IntegrityError as err:
            raise ValueError(
                f"Face id {face_id} already exists in the database"
            ) from err  # 抛出 id 已存在的异常

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
            np.expand_dims(query_vector, axis=0), self.index.ntotal
        )  # 使用 Faiss 进行搜索
        if distances[0][0] > threshold:
            name = self.query_database(
                "SELECT name FROM ids WHERE id = ?", (int(indices[0][0]),)
            )
            return name[0], float(distances[0][0])  # 返回人脸 id 和距离
        else:
            return None  # 如果没有超过阈值的则返回 None

    def removeFaceById(self, face_id: str):
        """
        根据人脸 id 删除数据库内的人脸向量
        Parameters:
        face_id: 待删除的人脸 id
        """
        print(face_id)
        id = self.query_database("SELECT id FROM ids WHERE name = ?", (face_id,))
        self.index.remove_ids(np.array([id[0]]))  # 从 Faiss 索引中删除人脸向量
        self.query_database(
            "DELETE FROM ids WHERE id = ?", (int(id[0]),)
        )  # 删除人脸 id 表中的人脸 id

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

    def saveDatabase(self):
        """
        保存数据库到指定路径

        Args:
            path (_type_): _description_
        """
        faiss.write_index(self.index, self.faiss_path.open("wb"))

    def loadDatabase(self):
        """
        从指定路径加载数据库

        Args:
            path (_type_): _description_
        """
        self.index = faiss.read_index(self.index_path)

    def query_database(self, sql, query):
        """访问数据库

        Args:
            sql (_type_): SQL语句
            query (_type_): 参数(可选)

        Returns:
           result: 访问结果(Nonable)
        """
        conn = sqlite3.connect(self.index_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(sql, query)
        result = cursor.fetchone()
        conn.commit()
        conn.close()
        return result
