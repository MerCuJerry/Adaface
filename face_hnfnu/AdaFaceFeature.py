from face_hnfnu.net import build_model
import torch
import numpy as np
from PIL import Image
from face_hnfnu.Config import ConfigModel
from yuface import detect
import cv2


class FaceAlignment:
    def align_process(img, bbox, landmark, image_size):
        """
        crop and align face
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
            desired_size: default 256
            padding: default 0
        Retures:
        -------
        crop_imgs: list, n
            cropped and aligned faces
        """
        M = None
        if landmark is not None:
            assert len(image_size) == 2
            src = np.array(
                [
                    [30.2946, 51.6963],
                    [65.5318, 51.5014],
                    [48.0252, 71.7366],
                    [33.5493, 92.3655],
                    [62.7299, 92.2041],
                ],
                dtype=np.float32,
            )

            src[:, 0] += 8
            src[:, 1] -= 8

            if image_size[0] == image_size[1] and image_size[0] != 112:
                src = src / 112 * image_size[0]

            dst = landmark.astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(
                dst.reshape(1, 5, 2), src.reshape(1, 5, 2)
            )

        if M is None:
            if bbox is None:  # use center crop
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1] * 0.0625)
                det[1] = int(img.shape[0] * 0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            margin = 44
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
            bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
            ret = img[bb[1] : bb[3], bb[0] : bb[2], :]
            if len(image_size) > 0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret
        else:  # do align using landmark
            assert len(image_size) == 2
            warped = cv2.warpAffine(
                img, M, (image_size[1], image_size[0]), borderValue=0.0
            )
            return warped


class AdaFaceFeature:
    """AdaFace 人脸特征值预测"""

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, config: ConfigModel) -> None:
        """初始化配置"""
        self.adaface_config = config.ADAFACE_MODEL
        self.adaface_models = {config.ADAFACE_MODEL: config.ADAFACE_MODEL_FILE}

    def load_pretrained_model(self):
        """加载模型"""

        # load model and pretrained statedict
        architecture = self.adaface_config
        assert architecture in self.adaface_models.keys()
        model = build_model(architecture)
        statedict = torch.load(
            self.adaface_models[architecture], map_location=torch.device("cpu"), weights_only=True
        )["state_dict"]
        model_statedict = {
            key[6:]: val for key, val in statedict.items() if key.startswith("model.")
        }
        model.load_state_dict(model_statedict)
        model.eval()
        self.model = model
        return self

    def to_input(self, pil_rgb_image):
        """PIL RGB图像对象转换为PyTorch模型的输入张量"""
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
        # tensor = torch.tensor([brg_img.transpose(2, 0,1)]).float()
        tensor = torch.tensor(np.array([brg_img.transpose(2, 0, 1)])).float()
        return tensor

    def byte_get_represent(self, image: Image.Image):
        """获取脸部特征向量"""
        try:
            w, h = image.size
            if w > 960 or h > 960:
                if w < h:
                    aspect_ratio = w / h
                    image.thumbnail((960 * aspect_ratio, 960))
                else:
                    aspect_ratio = h / w
                    image.thumbnail((960, 960 * aspect_ratio))
            _conf, bboxes, landmark = detect(np.array(image), conf=0.75)
            if len(bboxes) == 0:
                raise ValueError("未检测到人脸")
            aligned_rgb_img = FaceAlignment.align_process(
                np.array(image), bboxes, landmark, image_size=[112, 112]
            )
            bgr_tensor_input = self.to_input(aligned_rgb_img)
            if bgr_tensor_input is not None:
                feature, _ = self.model(bgr_tensor_input)
            return feature
        except Exception as err:
            raise ValueError(f"无法提取脸部特征向量, caused by:{err}") from err
