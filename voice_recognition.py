import cv2
import numpy as np
import tensorflow as tf
from log_info import setup_logger
from settings import (
    DIFFERENT_PERCENT,
    HEIGHT_SOUNDBAR,
    MODEL_SOUNDBAR_YOLOV7,
    SOUNDBAR_COLORS,
    SOUNDBAR_EDGE,
    SOUNDBAR_LEFT,
    SOUNDBAR_MIN,
    SOUNDBAR_PADDING,
    SOUNDBAR_TEMPLATE_DIR,
    SOUNDBAR_TOP,
    VOICE_INTERVAL_SECONDS,
    DEVICE_SOUNDBAR_DETECTION,
)
from soundbar_detection import SoundbarDetection
from utilities import crop_soundbar_area
from video_config import ClassConfig
from yolov7.models.experimental import attempt_load

logger = setup_logger("voice_recognition")


class VoiceProcessor:
    def __init__(self, args, class_config: ClassConfig):
        if args.d_soundbar == "CPU":
            with tf.device("/cpu:0"):
                # self.soundbar_detection = keras.models.load_model(
                #     MODEL_SOUNDBAR_DETECTION,
                # )
                self.attempt_load = attempt_load(
                    MODEL_SOUNDBAR_YOLOV7,
                    map_location= DEVICE_SOUNDBAR_DETECTION,
                )  # load FP32 model
                self.soundbar_detection_model = SoundbarDetection(
                    model=self.attempt_load,
                )
                logger.info(f"Loaded {MODEL_SOUNDBAR_YOLOV7} to CPU")
        else:
            # self.soundbar_detection = keras.models.load_model(
            #     MODEL_SOUNDBAR_DETECTION,
            # )
            self.attempt_load = attempt_load(
                MODEL_SOUNDBAR_YOLOV7,
                map_location="gpu",
            )  # load FP32 model
            self.soundbar_detection_model = SoundbarDetection(
                model=self.attempt_load,
            )
            logger.info(f"Loaded {MODEL_SOUNDBAR_YOLOV7} to GPU")
        self.voices = {}
        self.current_frame = set()
        self.current_interval = set()
        self.whole_video = []
        self.frame_count_in_an_interval = 0
        self.interval_frames = args.frames_per_sec * VOICE_INTERVAL_SECONDS
        self.ratio_interval_frames = int(
            class_config.fps / args.frames_per_sec,
        )
        self.template = self.load_templates()
        self.total_soundbar = [0, 0, 0]

    def load_templates(self) -> dict:
        """
        Load color template of soundbar (yellow and green)
        """
        soundbar_template_dir = SOUNDBAR_TEMPLATE_DIR
        soundbar_colors = SOUNDBAR_COLORS
        templates = {}
        for color in soundbar_colors:
            templates[color] = cv2.imread(
                soundbar_template_dir + color + ".png",
                0,
            )
        return templates

    def update_totally_talking_time(self):
        """
        Calculate talking time of each person up to the current frame
        """
        if self.frame_count_in_an_interval <= self.interval_frames:
            for id in self.current_frame:
                if id not in self.current_interval:
                    self.current_interval.add(id)
            self.frame_count_in_an_interval += 1
        if (
            self.frame_count_in_an_interval == self.interval_frames
        ):  # after an interval
            for id in self.current_interval:
                self.whole_video[id] += (
                    self.interval_frames * self.ratio_interval_frames
                )
            self.frame_count_in_an_interval = 0
            self.current_interval = set()
        # return self.voices, voice_frame_count

    def get_detected_voices(self, webcams: np.ndarray):
        """
        Get detected voices and their talking times
        """
        id = 0
        self.current_frame = set()
        for id in range(len(webcams)):
            if id >= len(self.whole_video):
                self.whole_video.append(0)
            # This method may be updated
            soundbars = self.count_soundbar_using_yolov7(webcams[id])
            self.total_soundbar[id] = soundbars
            if soundbars >= SOUNDBAR_MIN:
                self.current_frame.add(id)
        # estimate talking time
        self.update_totally_talking_time()

    def count_soundbar_using_vgg16(self, webcam):
        # TODO: define numbers
        webcam = cv2.resize(webcam, (180, 110))
        webcam = np.expand_dims(webcam, axis=0)
        prediction = self.soundbar_detection.predict(
            webcam,
            batch_size=None,
            steps=1,
            verbose=0,
        )
        if np.argmax(prediction[0]) > 0:
            return 4
        else:
            return 1
        # # return total_soundbar
        # return np.argmax(prediction[0])

    def count_soundbar_using_yolov7(self, webcam):

        # webcam = cv2.resize(webcam, (180, 110))
        SIZE_SOUNDBAR = HEIGHT_SOUNDBAR[0]
        if webcam.shape[0] == 165: #165 is height of webcam in full HD video
            SIZE_SOUNDBAR = HEIGHT_SOUNDBAR[1] # 6 is height_soundbar of Nhu_y.mp4
        height_soundbar = self.soundbar_detection_model.get_soundbar(
            frame=webcam,
        )
        return int((height_soundbar / SIZE_SOUNDBAR))


# TODO: following three functions are not be used currently
def count_soundbar_using_adaptiveThreshold(webcam: np.ndarray) -> int:
    soundbar_area = crop_soundbar_area(webcam)
    # Count number of soundbar in each frame using adaptiveThreshold
    soundbar = 0
    gray = cv2.cvtColor(
        soundbar_area,
        cv2.COLOR_RGB2GRAY,
    )  # why RGB is better than BGR
    _, std = cv2.meanStdDev(gray)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        int(std),
    )
    contours = cv2.findContours(
        thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]
    for c in contours:
        if 2.0 <= cv2.contourArea(c) <= 3.0:
            soundbar += 1
    return soundbar


def select_template(templates, soundbar_frame):
    for template in templates:
        tem_a = np.average(templates[template])
        if (
            np.average(soundbar_frame[59:61, 5:7]) * 1.1 >= tem_a
            and np.average(soundbar_frame[59:61, 5:7]) * 0.9 <= tem_a
        ):
            return template
    return None


def detect_talking_templateMatching(templates, soundbar_area):
    soundbar_area = cv2.cvtColor(soundbar_area, cv2.COLOR_BGR2GRAY)
    color = select_template(templates, soundbar_area)
    soundbar_count = 0
    if color is None:
        return 0
    template = np.average(templates[color])
    for i in range(SOUNDBAR_MIN):
        x, y = (SOUNDBAR_LEFT, SOUNDBAR_PADDING + SOUNDBAR_TOP * i)
        imgCrop = soundbar_area[y : y + SOUNDBAR_EDGE, x : x + SOUNDBAR_EDGE]
        if (
            np.average(imgCrop) * (1 - DIFFERENT_PERCENT)
            < template
            < np.average(imgCrop) * (1 + DIFFERENT_PERCENT)
        ):
            soundbar_count += 1
    return soundbar_count

