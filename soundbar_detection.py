import numpy as np
import torch
from settings import (
    WEBCAM_HEIGHT,
    WEBCAM_WIDTH,
    NMS_THRESHOLD,
    IMGSZ,
    CONF_THRES,
    IOU_THRES,
    CLASSES, 
    IMGSZ_DETECT_SOUNDBAR,
    DEVICE_SOUNDBAR_DETECTION,
    SCALE_IMG
)
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from yolov7.utils.torch_utils import select_device


class SoundbarDetection:
    def __init__(
        self,
        model,
        frame_width=WEBCAM_WIDTH,
        frame_height=WEBCAM_HEIGHT,
    ):
        # Parameters
        self.nms_threshold = NMS_THRESHOLD  
        self.imgsz = IMGSZ  # inference size (height, width)
        self.conf_thres = CONF_THRES  # confidence threshold
        self.iou_thres = IOU_THRES  
        self.classes = CLASSES  # colours soundbar
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = SCALE_IMG
        self.model = model

    def detect_soundbar(
        self,
        source,
        model,
        imgsz= IMGSZ_DETECT_SOUNDBAR, 
        device= DEVICE_SOUNDBAR_DETECTION,
        conf_thres=CONF_THRES,  # confidence threshold
        iou_thres=IOU_THRES,  # NMS IOU threshold
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
    ):
        device = select_device(device)
        half = device.type != DEVICE_SOUNDBAR_DETECTION  # half precision only supported on CUDA
        # Load model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        img = letterbox(source, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img *= self.scale  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:  
            img = img.unsqueeze(0)
        # Calculating gradients would cause a GPU memory leak
        with torch.no_grad():
            pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
        )
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ""
            s += "%gx%g " % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(
                    img.shape[2:],
                    det[:, :4],
                    source.shape,
                ).round()
        return source, det

    def get_soundbar(self, frame):
        # fillter objects in frame
        # Inference
        _, outs = self.detect_soundbar(frame, model=self.model)
        height_soundbar = 0
        for detection in outs:
            confidence = detection[4]
            if float(confidence) >= self.conf_thres:
                # x_min = int(detection[0])
                y_min = int(detection[1])
                # x_max = int(detection[2])
                y_max = int(detection[3])
                height_soundbar = y_max - y_min
        return height_soundbar
