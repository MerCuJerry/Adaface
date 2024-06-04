#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   flask_http_server.py
@Desc    :   flask 版本的 httpd 服务
"""

# here put the import lib
from flask import  Flask  # 导入Flask类
from flask import Flask, render_template, request,jsonify,Response
import utils
import logging
from flask import current_app
from pathlib import Path
from functools import wraps
from flask import abort
from concurrent.futures import ThreadPoolExecutor
from AdaFaceFeature import AdaFaceFeature
import yaml

DEFAULT_PATH = Path.cwd() / "config" / "config.yaml"
app = Flask(__name__)  # 实例化并命名为app实例

logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
with DEFAULT_PATH.open() as file:
    config = yaml.safe_load(file.read())
flask_config = config['flask']



class MyContextObject:
    """
    @Desc    :   全局上下文对象
    """

    def __init__(self, adaface, ready_mark=False):
        self.adaface = adaface
        self.ready_mark = ready_mark


def init_obj():
    """
    @Desc    :   上下文对象处理
    """

    adaface = AdaFaceFeature()
    logging.info('🚀🚀🚀🚀 人脸特征提取相关模型加载')
    adaface.load_pretrained_model()
    logging.info('🚀🚀🚀🚀 构建上下文对象')
    my_context = MyContextObject(adaface, ready_mark=True)
    app.my_context = my_context
    logging.info('🚀🚀🚀🚀🚀 \033[32m服务启动成功\033[0m')

with app.app_context():
       init_obj()

# 定义装饰器函数，用于验证 Token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        valid_tokens = flask_config["token"]
        # 如果 Token 无效，返回未授权的响应
        if token not in valid_tokens:
             return jsonify({'message': 'Token is invalid'}), 401

        # 如果 Token 有效，继续处理请求
        return f(*args, **kwargs)

    return decorated


@app.route("/")
def index():
    """
    @Desc    :   欢迎页
    """

    return {'result': "Hello, face"}


@app.route("/livez")
def livez():
    """
    @Desc    :   服务存活探针接口
    """
    return {'result': " live  ^_^"}


@app.route("/token")
def token():
    return jsonify({
            "code": 200,
            "message": "获取 token",
            "token": flask_config["token"],
        })


@app.route("/readyz")
def readyz():
    """
    @Desc    :   服务就绪探针接口
    """
    context = app.my_context
    if context.ready_mark:
        return {'result': " ready ^_^ "} 
    else:
        abort(503, "Service Unavailable")


@app.route("/face_mtcnn_represent", methods=["POST"])
@token_required
def face_mtcnn_represent():
    """
    @Desc    :   解析 mtcnn 返回的  face 数据
    """
    faces = request.json
    if faces is not None and faces['face_efficient_total_resp'] != 0:
        resp = faces['resp']
        for face in resp:
            logging.info(face['face_id'])
            face_b64 = face['face_align_images_b64']
            try:
                context = current_app.my_context
                feature =  context.adaface.b64_get_represent(face_b64)
                if feature  is not None:
                    face['face_vetor'] = utils.feature2json(feature)
         
            except Exception as exc:
                logging.info(f"adaface 获取人脸信息 调用异常：{exc}")
                return {"status": 400, "message": f"mtcnn 获取人脸信息 调用异常：{exc}"}
        
        return {"status": 200, "message": "提取人脸特征向量成功", "data": faces}
    else:
        return {"status": 400, "message": f"{faces['image_id']} 没有符合特征提取条件的人脸","data":faces }


    



@app.route("/b64_represent_byte", methods=["POST"])
@token_required
def b64representBYTE():
    """
    @Desc    :   返回二进制特征向量
    """
    
    base64_data = request.get_data(as_text=True)
    if base64_data  is not None:
        context = current_app.my_context
        feature =  context.adaface.b64_get_represent(base64_data)
        if feature  is not None:
            return Response(utils.feature2byte(feature), content_type='application/octet-stream')
        else:
            abort(400, "feature fail")
    else:
        abort(503, "Service Unavailable")
    

@app.route("/b64_represent_json", methods=["POST"])
@token_required
def b64representJSON():
    """
    @Desc    :   返回JSON 特征向量
    """
    base64_data = request.get_data(as_text=True)
    if base64_data  is not None:
        context = current_app.my_context
        feature =  context.adaface.b64_get_represent(base64_data)
        if feature  is not None:
            return Response(utils.feature2json_str(feature), content_type='application/json')
        else:
            abort(400, "feature fail")
    else:
        abort(503, "Service Unavailable")


@app.route("/byte_represent_byte", methods=["POST"])
@token_required
def byterepresentBYTE():
    """
    @Desc    :   返回二进制特征向量
    """
    
    base64_data = request.get_data(as_text=True)
    if base64_data  is not None:
        context = current_app.my_context
        feature =  context.adaface.byte_get_represent(base64_data)
        if feature  is not None:
            return Response(utils.feature2byte(feature), content_type='application/octet-stream')
        else:
            abort(400, "feature fail")
    else:
        abort(503, "Service Unavailable")
    

@app.route("/byte_represent_json", methods=["POST"])
@token_required
def byterepresentJSON():
    """
    @Desc    :   返回JSON 特征向量
    """
    base64_data = request.get_data(as_text=True)
    if base64_data  is not None:
        context = current_app.my_context
        feature =  context.adaface.byte_get_represent(base64_data)
        if feature  is not None:
            return Response(utils.feature2json_str(feature), content_type='application/json')
        else:
            abort(400, "feature fail")
    else:
        abort(503, "Service Unavailable")


if __name__ == "__main__":
    
    app.run(port=flask_config['port'], host="0.0.0.0")  # 调用run方法，设定端口号，启动服务

