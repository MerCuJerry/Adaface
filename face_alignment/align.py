from face_alignment import mtcnn
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn_model = mtcnn.MTCNN(device="cpu", crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None):
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0]
    except Exception as e:
        raise Exception(f"Face detection Failed due to {e}") from e

    return face


