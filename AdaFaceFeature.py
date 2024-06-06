import net
import torch
import numpy as np
from pathlib import Path
import yaml
import utils
from face_alignment import align


DEFAULT_PATH = Path.cwd() / "config" / "config.yaml"


class AdaFaceFeature:
    """
    AdaFace 人脸特征值预测
    """

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, file_name = DEFAULT_PATH) -> None:
        """
        初始化配置
        """
        with file_name.open() as file:
            self.config = yaml.safe_load(file.read())
        self.adaface_config = self.config["adaface"]["zero"]
        self.adaface_models = {
            self.adaface_config["model"]: self.adaface_config["model_file"],
        }
        pass

    def load_pretrained_model(self):
        """
        加载模型
        """

        # load model and pretrained statedict
        architecture = self.adaface_config["model"]
        assert architecture in self.adaface_models.keys()
        model = net.build_model(architecture)
        statedict = torch.load(
            self.adaface_models[architecture], map_location=torch.device("cpu")
        )["state_dict"]
        model_statedict = {
            key[6:]: val for key, val in statedict.items() if key.startswith("model.")
        }
        model.load_state_dict(model_statedict)
        model.eval()
        self.model = model
        return self

    def to_input(self, pil_rgb_image):
        """
        PIL RGB图像对象转换为PyTorch模型的输入张量
        """
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
        # tensor = torch.tensor([brg_img.transpose(2, 0,1)]).float()
        tensor = torch.tensor(np.array([brg_img.transpose(2, 0, 1)])).float()
        return tensor

    def b64_get_represent(self, path):
        """
        Base64获取脸部特征向量, Need Fix
        """

        feature = None

        try:
            aligned_rgb_img = align.get_aligned_face(
                image_path=None,
                rgb_pil_image=utils.get_base64_to_Image(path).convert("RGB"),
            )
            bgr_tensor_input = self.to_input(aligned_rgb_img)
            if bgr_tensor_input is not None:
                feature, _ = self.model(bgr_tensor_input)
            else:
                print("无法提取脸部特征向量")
            return feature
        except Exception as err:
            raise Exception("无法提取脸部特征向量") from err
        
    def byte_get_represent(self, path):
        """
        获取脸部特征向量
        """

        try:
            aligned_rgb_img = align.get_aligned_face(path)
            bgr_tensor_input = self.to_input(aligned_rgb_img)
            if bgr_tensor_input is not None:
                feature, _ = self.model(bgr_tensor_input)
            return feature
        except Exception as err:
            raise Exception("无法提取脸部特征向量") from err