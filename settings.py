import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

# Tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# List of devices can be used for training/recognizing
DEVICE_KINDS = ["CPU", "GPU", "MYRIAD", "HETERO", "HDDL"]
DEVICE_SOUNDBAR_DETECTION = "cpu"
# Models
MODEL_FACE_DETECTION = (
    "models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml"
)
MODEL_FACE_REIDENTIFICATION = "models/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml"  # noqa: E501
MODEL_LANDMARKS = "models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml"  # noqa: E501
MODEL_OCR = "models/vietocr/vietocr.model"
MODEL_SOUNDBAR_DETECTION = "models/soundbar/soundbar_vgg16.h5"
MODEL_SOUNDBAR_YOLOV7 = "models/soundbar_detection_yolov7/soundbar_yolov7.pt"

# Face settings:
IMAGE_EXTENSIONS = ["jpg", "png"]
FACE_INTERVAL_SECONDS = 3
REGISTERED_FACE = "registered_faces/"

# Face_Landmark
POINTS_NUMBER = 5


# Voice settings:
DIFFERENT_PERCENT = 0.05
SOUNDBAR_TEMPLATE_DIR = "models/soundbar-template/"
SOUNDBAR_COLORS = ["green", "yellow"]
SOUNDBAR_MAX = 15
SOUNDBAR_MIN = 3
SOUNDBAR_EDGE = 2
SOUNDBAR_LEFT = 5
SOUNDBAR_TOP = 4
SOUNDBAR_PADDING = 3
VOICE_INTERVAL_SECONDS = 1


# Name settings:
FONT_SIZE = 15
FONT_PATH = "font/Roboto-Regular.ttf"  # Vietnamese Font


# OCR
CLASS_DURATION = 31
OCR_NETWORK = "vgg_transformer"
OCR_DEVICE = "cpu"  # gpu:0


# Area
# [[y, y + height], [x, x + width]]
BACKGROUND_1_MEAN_COLOR = 210.47
BACKGROUND_2_MEAN_COLOR = 75.60
BACKGROUND_MEAN_COLOR_DIF = 0.03  # 3%

# Default resolution
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# Webcam location (720p & 1080p)
WEBCAM_TOP = 20
WEBCAM_BOT = 130
WEBCAM_LEFT_2 = 460  # for people = 2
WEBCAM_LEFT_3 = 369  # for people = 3
WEBCAM_PADDING = 2
WEBCAM_WIDTH = 180
WEBCAM_HEIGHT = 110
WEBCAM_MID = 400

# Checking existing webcam
WEBCAM_CHECKING_BOT = 125
WEBCAM_CHECKING_LEFT_2 = 3
WEBCAM_CHECKING_LEFT_3 = 2
WEBCAM_CHECKING_TOP = 100
WEBCAM_CHECKING_WIDTH = 2
WEBCAM_MAX_HEIGHT = 106  # 101
WEBCAM_MAX_WIDTH = 182  # 177
WEBCAM_MIN_HEIGHT = 96  # 101
WEBCAM_MIN_WIDTH = 172  # 177

# Crop from the whole frame
CLASS_STARTING_AREA = [[5, 20], [600, 850]]

# Crop from webcam area
FRAME_MICRO = [[90, 105], [3, 13]]
FRAME_NAME = [[90, 107], [18, 150]] #TODO delete
FRAME_SOUND_LVL = [[30, 91], [3, 13]]

# Crop from sound_lvl
TEMPLATE_COMPARATION = [[59, 61], [4, 8]]

HEIGHT_SOUNDBAR = [4, 6]

MAX_PEOPLE = 6

# Const of yolov7. Do not change
NMS_THRESHOLD = 0.4
IMGSZ = (416, 416)
CONF_THRES = 0.70 
IOU_THRES = 0.45 
CLASSES = ["yellow", "green"]

IMGSZ_DETECT_SOUNDBAR = 192
SCALE_IMG = 1/255

OSCILLATE_RANGE = 25 #webcam

#Size of name area
NAME_LEFT = 18
NAME_WIDTH = 132
NAME_BOT = -3
NAME_TOP = -3 - 17

#CONST OF RESNET MODEL
#DON'T CHANGE

RESNET_RESIZE = 200
RESNET_CROP_CENTER = 140
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]
