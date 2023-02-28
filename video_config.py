import torch
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from log_info import setup_logger
from ocr import OCR
from settings import (
    BACKGROUND_1_MEAN_COLOR,
    BACKGROUND_2_MEAN_COLOR,
    BACKGROUND_MEAN_COLOR_DIF,
    CLASS_DURATION,
    CLASS_STARTING_AREA,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WEBCAM_BOT,
    WEBCAM_CHECKING_BOT,
    WEBCAM_CHECKING_LEFT_2,
    WEBCAM_CHECKING_LEFT_3,
    WEBCAM_CHECKING_TOP,
    WEBCAM_CHECKING_WIDTH,
    WEBCAM_LEFT_2,
    WEBCAM_LEFT_3,
    WEBCAM_MAX_HEIGHT,
    WEBCAM_MAX_WIDTH,
    WEBCAM_MID,
    WEBCAM_MIN_HEIGHT,
    WEBCAM_MIN_WIDTH,
    WEBCAM_PADDING,
    WEBCAM_TOP,
    WEBCAM_WIDTH,
    MAX_PEOPLE,
    CONF_THRES,
    OSCILLATE_RANGE,
    RESNET_RESIZE,
    RESNET_CROP_CENTER,
    RESNET_MEAN,
    RESNET_STD,
)
from utilities import convert_cv2_2_pil


logger = setup_logger("main_logger")

#tracking webcams by Resnet
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(RESNET_RESIZE),
        transforms.CenterCrop(RESNET_CROP_CENTER),
        transforms.ToTensor(),
        transforms.Normalize(mean= RESNET_MEAN, std= RESNET_STD),
    ]
)
feature_map = [None for i in range(MAX_PEOPLE)]
coordinate_webcam = [(0, 0) for i in range(MAX_PEOPLE)]
class ClassConfig:
    def __init__(self, args):
        self.video = cv2.VideoCapture(args.input)
        if self.video.open is False:
            raise RuntimeError("Can't open video. Please check input.")
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        if args.frames_per_sec > self.fps:
            args.frames_per_sec = self.fps
        self.interval_frames = int(self.fps / args.frames_per_sec)

        self.background = 1  # default wall, 2: blue
        self.webcam_top = WEBCAM_TOP
        self.webcam_bot = WEBCAM_BOT
        self.webcam_left_2 = WEBCAM_LEFT_2  # for people = 2
        self.webcam_left_3 = WEBCAM_LEFT_3  # for people = 3
        self.webcam_width = WEBCAM_WIDTH
        self.webcam_mid = WEBCAM_MID

        self.class_start_flag = False
        self.class_duration = 0

        self.frame_start = 0
        self.frame_class = 0

        self.updated_participants = 0
        self.participants = 0
        self.participants_change = 0
        self.max_participants = 0

        self.current_frame = []
        self.webcam_frames = []
        self.webcam_positions = []

        self.model = YOLO("models/webcam_detection/model_detect_webcam.pt")


    def read_current_frame(self, frame_num):
        """
        Method for reading the current frame in the input video
        """
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, self.current_frame = self.video.read()
        if self.current_frame is not None:
            self.current_frame = cv2.resize(
                self.current_frame,
                (VIDEO_WIDTH, VIDEO_HEIGHT),
            )
    def get_webcam_frame_option_OFF(self):
        """
        Extract frames of webcams from a given number of participants
        """
        # return get_webcam_frame_using_yolov8(self)
        wc_padding = WEBCAM_PADDING
        wc_width = self.webcam_width
        wc_top = self.webcam_top
        wc_bot = self.webcam_bot
        wc_left = self.webcam_left_2  # for people = 2
        if self.participants == 1:
            wc_left = self.webcam_left_3 + wc_width + wc_padding
        elif self.participants == 3:
            wc_left = self.webcam_left_3

        self.webcam_frames = []
        for each in range(0, self.participants):
            wc_right = wc_left + wc_width
            web_frame = self.current_frame[wc_top:wc_bot, wc_left:wc_right]
            self.webcam_frames.append(web_frame)
            wc_left = wc_right + wc_padding

        if self.participants_change != 0:
            self.webcam_positions = []
            wc_left = self.webcam_left_2  # for people = 2
            if self.participants == 3:
                wc_left = self.webcam_left_3
            for each in range(0, self.participants):
                wc_right = wc_left + wc_width
                web_frame = self.current_frame[wc_top:wc_bot, wc_left:wc_right]
                self.webcam_positions.append(
                    [
                        self.webcam_top,
                        self.webcam_bot,
                        wc_left,
                        wc_right,
                    ],
                )
                wc_left = wc_right + wc_padding
    def get_webcam_frame_option_ON(self):
        threshold = CONF_THRES
        check_object = [0 for i in range(MAX_PEOPLE)]

        results = self.model(self.current_frame, verbose = False)
        detections = []
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                web_frame = self.current_frame[y1:y2, x1:x2]

                if class_id == 0 and score > threshold: # 0 is class_id of webcam
                    detections.append([web_frame, x1, y1, x2, y2])
        self.webcam_frames = [None for j in detections]
        self.webcam_positions = [None for j in detections]

        for detection in detections:
            i = get_index(detection, check_object)
            if i >= len(self.webcam_frames):
                continue
            feature_map[i] = embed_image_to_extract_features(detection[0])
            coordinate_webcam[i] = (detection[1], detection[2])
            check_object[i] = 1
            self.webcam_frames[i] = detection[0] #img
            self.webcam_top = detection[2]
            self.webcam_bot = detection[4]
            self.wc_left = detection[1]
            self.wc_right = detection[3]

            self.webcam_positions[i] = [
                                    self.webcam_top,
                                    self.webcam_bot,
                                    self.wc_left,
                                    self.wc_right,
            ]



    def get_class_duration(self, ocr_engine: OCR):
        """
        Extract an area in the last frame and recognize the time.
        Keywords are:
        - "Start in": count down time before class starts
        - "Started": time of the class (0 -> 25 mins)
        - "Remaining": remaining time of the class (26 -> 30 mins)
        - "Extra time": added time when the class is over 30 mins

        Time will be round up in minute.
        """
        time_area = self.current_frame[
            CLASS_STARTING_AREA[0][0] : CLASS_STARTING_AREA[0][1],
            CLASS_STARTING_AREA[1][0] : CLASS_STARTING_AREA[1][1],
        ]

        time_area = convert_cv2_2_pil(time_area)
        text, _ = ocr_engine.recognize(time_area)

        if "Remain" in text:
            text = text.split("Remain")[1]
        elif "Extra" in text:
            text = text.split("Extra")[1]

        text = re.findall(r"\d{2}", text)[0]
        self.class_duration = (
            int(text) + CLASS_DURATION + 1
        )  # time in minutes (round up)

        logger.info(f"Class duration: {self.class_duration}")

    def detect_starting_time(self, ocr_engine: OCR, frame_num: int) -> bool:
        """
        Detect the first frame that contains "started" keyword,
            then recognize the time in that frame.
        Keywords are:
        - "Start in": count down time before class starts
        - "Started": time of the class (0 -> 25 mins)
        - "Remaining": remaining time of the class (26 -> 30 mins)
        - "Extra time": added time when the class is over 30 mins

        Time will be round up in minute.
        """
        class_time_area = self.current_frame[
            CLASS_STARTING_AREA[0][0] : CLASS_STARTING_AREA[0][1],
            CLASS_STARTING_AREA[1][0] : CLASS_STARTING_AREA[1][1],
        ]

        class_time_area = convert_cv2_2_pil(class_time_area)
        text, _ = ocr_engine.recognize(class_time_area)
        if (
            "Started" in text
            or "Remain" in text
            or "Đã" in text
            or "00:00" in text
        ):
            self.class_start_flag = True
            self.frame_start = frame_num
            logger.info("Class starts!")
        else:
            logger.info(text[-5:])

    def count_webcam_by_vertical_edges(self, edges):
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

        vertical_line_count = 0
        if linesP is not None:
            for i in range(0, len(linesP)):
                line = linesP[i][0]
                if line[0] < 300 or line[0] > 900:
                    continue
                if line[2] < 300 or line[2] > 900:
                    continue

                # l: x1,y1,x2,y2
                p1 = np.array([line[0], line[1]])
                p2 = np.array([line[2], line[3]])
                p3 = np.subtract(p2, p1)  # translate p2 by p1

                angle_radiants = math.atan2(p3[1], p3[0])
                angle_degree = angle_radiants * 180 / math.pi

                # TODO: check CONS numbers
                if -93 < angle_degree < -87 or 87 < angle_degree < 93:
                    if abs(line[1] - line[3]) > 80:
                        vertical_line_count += 1

        return int(vertical_line_count / 2)

    def count_webcam_by_contour(self) -> int: #TODO change to len(webcam_frames) when option == 'ON'
        webcams_area = self.current_frame[self.webcam_top : self.webcam_bot, :]
        edges = cv2.Canny(webcams_area, 100, 200)
        contours = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        webcam_count = 0
        for c in contours:
            # get the bounding rect
            _, _, w, h = cv2.boundingRect(c)
            if WEBCAM_MIN_HEIGHT < h < WEBCAM_MAX_HEIGHT:  # 101
                if WEBCAM_MIN_WIDTH < w < WEBCAM_MAX_WIDTH:  # 177
                    webcam_count += 1

        """
        # TODO: recheck the number
        if webcam_count == 0:
            satisfy_contour_count = 0
            min_x = 1000
            max_x = 0
            contours = merge_contour(contours)
            for c in contours:
                x, _, w, h = c[0], c[1], c[2], c[3]
                if 300 < x < 900:
                    satisfy_contour_count += 1
                    if min_x > x:
                        min_x = x
                    if max_x < x:
                        max_x = x
            if (max_x - min_x) > 190 and satisfy_contour_count > 1:
                webcam_count = 2
            else:
                webcam_count = 1
        """
        if webcam_count == 0:
            webcam_count = self.count_webcam_by_vertical_edges(edges)

        return webcam_count

    def detect_default_background(self) -> bool:
        frame_left = WEBCAM_LEFT_3 - 6
        frame_right = WEBCAM_LEFT_3 - 1
        bg = self.current_frame[
            self.webcam_top : self.webcam_bot,
            frame_left:frame_right,
        ]
        if (
            abs(bg.mean() - BACKGROUND_1_MEAN_COLOR)
            < BACKGROUND_1_MEAN_COLOR * BACKGROUND_MEAN_COLOR_DIF
        ):
            return True
        return False

    def get_total_participants(self) -> int:  #TODO Change to len(webcam_frames) when option == 'ON'
        """
        Get number of persons in the current frame.
        Update the change of that number (comparing to the previous frame)
        """

        # self.updated_participants = self.count_webcam_by_contour()
        if self.detect_default_background() is True:
            bg_left_3_1 = self.current_frame[
                self.webcam_top : self.webcam_bot,
                self.webcam_left_3 : self.webcam_mid,
            ]
            bg_left_2_1 = self.current_frame[
                self.webcam_top : self.webcam_bot,
                self.webcam_left_2 : self.webcam_left_2 + 10,
            ]
            if (
                abs(bg_left_3_1.mean() - BACKGROUND_1_MEAN_COLOR)
                < BACKGROUND_1_MEAN_COLOR * BACKGROUND_MEAN_COLOR_DIF
            ):
                if (
                    abs(bg_left_2_1.mean() - BACKGROUND_1_MEAN_COLOR)
                    < BACKGROUND_1_MEAN_COLOR * BACKGROUND_MEAN_COLOR_DIF
                ):
                    self.updated_participants = 1
                else:
                    self.updated_participants = 2
            else:
                self.updated_participants = 3
        else:
            self.background = 2
            # Background 2
            # TODO: add case of 3 participants
            bg_mid = self.current_frame[
                self.webcam_top : self.webcam_bot,
                638:642,
            ]
            if (
                abs(bg_mid.mean() - BACKGROUND_2_MEAN_COLOR)
                < BACKGROUND_2_MEAN_COLOR * BACKGROUND_MEAN_COLOR_DIF
            ):
                self.updated_participants = 2
            else:
                self.updated_participants = 1

        if self.updated_participants != self.participants:
            self.participants_change = (
                self.updated_participants - self.participants
            )
            self.participants = self.updated_participants
        else:
            self.participants_change = 0

        if self.updated_participants > self.max_participants:
            self.max_participants = self.updated_participants

    def check_existing_webcam(
        self,
        participant_id: int,
    ) -> bool:
        """
        Check if a given webcam is in the correct position
        Assume that webcam #1 is always in the correct position
        """

        if participant_id == 0:
            return True

        """ Case: 2 webcams
        Order of webcam is counted from left to right
        apply for webcam #1"""
        if self.participants == 2:  # webcam #2
            if participant_id == 1:
                left_area = (
                    self.webcam_left_2
                    + self.webcam_width
                    + WEBCAM_CHECKING_LEFT_2
                )  # noqa: E501
                check_area = self.current_frame[
                    WEBCAM_CHECKING_TOP:WEBCAM_CHECKING_BOT,
                    left_area : left_area + WEBCAM_CHECKING_WIDTH,
                ]
                if self.background == 1:
                    if (
                        self.check_background(
                            check_area,
                            BACKGROUND_1_MEAN_COLOR,
                        )
                        is False
                    ):
                        return False
                    return True
                elif self.background == 2:
                    if (
                        self.check_background(
                            check_area,
                            BACKGROUND_2_MEAN_COLOR,
                        )
                        is False
                    ):
                        return False
                    return True

        """ Case: 3 webcams
        Order of webcam is counted from left to right
        Apply for webcam #2 & #3"""
        if self.participants == 3:
            left_area = (
                self.webcam_left_3
                + (self.webcam_width + WEBCAM_CHECKING_LEFT_3) * participant_id
            )  # noqa: E501
            check_area = self.current_frame[
                WEBCAM_CHECKING_TOP:WEBCAM_CHECKING_BOT,
                left_area : left_area + WEBCAM_CHECKING_WIDTH,
            ]
            if self.background == 1:
                if (
                    self.check_background(check_area, BACKGROUND_1_MEAN_COLOR)
                    is False
                ):
                    return False
                return True
            elif self.background == 2:
                if (
                    self.check_background(check_area, BACKGROUND_2_MEAN_COLOR)
                    is False
                ):
                    return False
                return True

        return False

    def check_background(self, check_area, background):
        if (
            abs(check_area.mean() - background)
            < background * BACKGROUND_MEAN_COLOR_DIF
        ):
            return False
        return True
    
    
def getXFromRect(item):
    return item[0]


def merge_contour(cnts):
    xThr = WEBCAM_PADDING - 1
    # Array of initial bounding rects
    rects = []

    # Bool array indicating which initial bounding rect has
    # already been used
    rectsUsed = []

    # Just initialize bounding rects and set all bools to false
    for cnt in cnts:
        rects.append(cv2.boundingRect(cnt))
        rectsUsed.append(False)

    # Sort bounding rects by x coordinate
    rects.sort(key=getXFromRect)

    # Array of accepted rects
    acceptedRects = []

    # xThr: Merge threshold for x coordinate distance

    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if rectsUsed[supIdx] is False:

            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(
                rects[(supIdx + 1) :],
                start=(supIdx + 1),
            ):

                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if candxMin <= currxMax + xThr:

                    # Reset coordinates of current rect
                    currxMax = candxMax
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)

                    # Merge candidate (bounding rect) is used
                    rectsUsed[subIdx] = True
                else:
                    break

            # No more merge candidates possible, accept current rect
            acceptedRects.append(
                [currxMin, curryMin, currxMax - currxMin, curryMax - curryMin],
            )
    return acceptedRects


def embed_image_to_extract_features(image: np.ndarray): #embedding image 
    image = Image.fromarray(image)

    # Apply the preprocessing transform to the input image
    input_tensor = preprocess(image)

    # Add a batch dimension to the input tensor
    input_batch = input_tensor.unsqueeze(0)

    # Use the ResNet model to extract embeddings from the input image
    with torch.no_grad():
        embeddings = resnet18(input_batch)
    return np.array(embeddings)
def get_index(detection, check_object): #return index webcam
    threshold = CONF_THRES
    img, x, y, xmax, ymax = detection
    feature_img = embed_image_to_extract_features(img)
    cnt = -1
    for i in range(len(feature_map)):
        if check_object[i] == 0:
            cosine = cosine_similarity(feature_img, feature_map[i])
            if abs(cosine) > threshold:
                return i

            else:
                # cosine1 = cosine_similarity(feature_img, feature_without_webcam)
                x_old, y_old = coordinate_webcam[i]
                x_min = x_old - OSCILLATE_RANGE  
                x_max = x_old + OSCILLATE_RANGE  
                y_min = y_old - OSCILLATE_RANGE
                y_max = y_old + OSCILLATE_RANGE

                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return i
            cnt = i
    return cnt + 1

