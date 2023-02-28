import cv2
import numpy as np
from openvino.model_zoo.model_api.models.utils import resize_image
from PIL import Image
from settings import (
    FRAME_MICRO,
    FRAME_SOUND_LVL,
)


def crop(frame, roi):
    p1 = roi.position.astype(int)
    p1 = np.clip(p1, [0, 0], [frame.shape[1], frame.shape[0]])
    p2 = (roi.position + roi.size).astype(int)
    p2 = np.clip(p2, [0, 0], [frame.shape[1], frame.shape[0]])
    return frame[p1[1] : p2[1], p1[0] : p2[0]]


def cut_rois(frame, rois):
    return [crop(frame, roi) for roi in rois]


def resize_input(image, target_shape, nchw_layout):
    if nchw_layout:
        _, _, h, w = target_shape
    else:
        _, h, w, _ = target_shape
    resized_image = resize_image(image, (w, h))
    if nchw_layout:
        resized_image = resized_image.transpose((2, 0, 1))  # HWC->CHW
    resized_image = resized_image.reshape(target_shape)
    return resized_image


def convert_frame_to_time(fps: int, frames_number: np.ndarray) -> float:
    if fps == 0:
        return 0
    minutes = frames_number / (fps * 60)
    # remaining_frames = frames % (fps * 60)
    # seconds = remaining_frames // fps
    return format_decimal_number(minutes)


def format_decimal_number(number: float) -> float:
    return float(f"{number:.2f}")


def crop_microphone_area(webcam_area: np.ndarray) -> np.ndarray:
    return webcam_area[
        FRAME_MICRO[0][0] : FRAME_MICRO[0][1],
        FRAME_MICRO[1][0] : FRAME_MICRO[1][1],
    ]


def crop_soundbar_area(webcam_area: np.ndarray) -> np.ndarray:
    return webcam_area[
        FRAME_SOUND_LVL[0][0] : FRAME_SOUND_LVL[0][1],
        FRAME_SOUND_LVL[1][0] : FRAME_SOUND_LVL[1][1],
    ]


def convert_cv2_2_pil(image: np.ndarray) -> Image:
    """
    convert image in cv2 format to PIL format
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def convert_pil_2_cv2(image: Image) -> np.ndarray:
    """
    convert image in PIL format to cv2 format
    """
    image = np.array(image)
    # Convert RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
