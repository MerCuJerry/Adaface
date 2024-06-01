

import faiss  # 导入 faiss 库
import numpy as np
class FaceDatabase:
    def __init__(self, dimension=512):
        """
        初始化 FaceDatabase 类
        
        Parameters:
        dimension (int): 人脸向量的维度，默认为 512
        """
        self.dimension = dimension  # 保存人脸向量的维度
        self.index = faiss.IndexFlatIP(dimension)  # 创建 Faiss 的余弦相似度索引
        self.ids = []  # 保存人脸向量对应的 id

    def addFace(self, face_id, face_vector):
        """
        将人脸向量添加进数据库
        
        Parameters:
        face_id: 人脸 id
        face_vector: 人脸向量
        """
        if face_vector.shape[0] != self.dimension:
            raise ValueError("Face vector dimension does not match the database dimension")  # 抛出维度不匹配的异常
        self.index.add(np.expand_dims(face_vector, axis=0))  # 向 Faiss 索引中添加人脸向量
        self.ids.append(face_id)  # 记录人脸向量的 id

    def searchSimilarFaces(self, query_vector, threshold):
        """
        查询相似的人脸向量
            
        Parameters:
        query_vector: 查询向量
        threshold: 相似度阈值
        Returns:
        tuple or None: 返回超过阈值的最相似人脸向量和对应的距离，如果没有超过阈值的，则返回 None
        """
        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), len(self.ids))  # 使用 Faiss 进行搜索4
        print(distances)
        similar_faces = [(self.ids[i], distances[0][i]) for i in range(len(self.ids)) if distances[0][i] > threshold]  # 保存超过阈值的最相似人脸向量
        if similar_faces:
            most_similar_face = max(similar_faces, key=lambda x: x[1])  # 找到最相似的人脸
            return most_similar_face  # 返回最相似的人脸和对应的距离
        else:
            return None  # 如果没有超过阈值的则返回 None

    def removeFaceById(self, face_id):
        """
        根据人脸 id 删除数据库内的人脸向量
        
        Parameters:
        face_id: 待删除的人脸 id
        """
        #先删除高索引的元素，再删除低索引的元素，避免索引错位的问题。
        index_to_remove = [i for i, stored_id in enumerate(self.ids) if stored_id == face_id]  # 找到需要删除的人脸向量的索引
        for i in sorted(index_to_remove, reverse=True):
            self.index.remove_ids(np.array([i]))  # 从 Faiss 索引中删除对应的人脸向量
            del self.ids[i]  # 从 id 列表中删除对应的 id

    def getNumFaces(self):
        """
        获取数据库内存储的人脸向量数量
        
        Returns:
        int: 人脸向量数量
        """
        return len(self.ids)  # 返回保存的人脸向量数量

    def clearDatabase(self):
        """
        清空数据库，删除所有的人脸向量
        """
        self.index.reset()  # 重置 Faiss 索引
        self.ids.clear()  # 清空 id 列表
        
if __name__ == '__main__':
   # 创建一个 FaceDatabase 实例
    face_db = FaceDatabase()

    # 添加人脸向量和对应的 id
    id1 = '001'
    vector1 = np.random.rand(512).astype('float32')  # 随机生成一个512维向量
    face_db.addFace(id1, vector1)

    id2 = '002'
    vector2 = np.random.rand(512).astype('float32')
    face_db.addFace(id2, vector2)

    id3 = '003'
    vector3 = np.random.rand(512).astype('float32')
    face_db.addFace(id3, vector3)

    # 搜索相似人脸
    query_vector = np.random.rand(512).astype('float32')  # 随机生成一个512维向量作为查询向量
    print("query_vector = ",query_vector)
    print("query_vector.shape = ", len(query_vector))
    threshold = -1000000000000000000
    result = face_db.searchSimilarFaces(query_vector, threshold)
    if result:
        similar_id, distance = result
        print(f"The most similar face is {similar_id} with distance {distance}")
    else:
        print("No similar face found")

    # 删除人脸
    #face_db.removeFaceById(id2)
    #print(f"Number of faces in the database: {face_db.getNumFaces()}")

    # 清空数据库
    #face_db.clearDatabase()
    #print(f"Number of faces in the database: {face_db.getNumFaces()}")
