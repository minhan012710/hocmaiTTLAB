import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings('ignore')
import sys

# DEPLOYMENT_PLACEHOLDER_1
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
# import cv2_ext
import numpy as np
from helpers import resolution
from log_info import setup_logger
from moviepy.editor import VideoFileClip
from name_identifier import IdentityProcessor
from ocr import OCR
from PIL import (
    ImageDraw,
    ImageFont,
)
from settings import (
    DEVICE_KINDS,
    FONT_PATH,
    FONT_SIZE,
    MODEL_FACE_DETECTION,
    MODEL_FACE_REIDENTIFICATION,
    MODEL_LANDMARKS,
    ROOT_DIR,
)
from utilities import (
    convert_cv2_2_pil,
    convert_frame_to_time,
    convert_pil_2_cv2,
    format_decimal_number,
)
from video_config import ClassConfig
from voice_recognition import VoiceProcessor

from face_recognition import FaceProcessor

sys.path.append(str(Path(__file__).resolve().parents[2] / "common/python"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "source/yolov7"))
logger = setup_logger("main_logger", log_file="logging.txt")
vietnamese_font = ImageFont.truetype(FONT_PATH, FONT_SIZE, encoding="unic")
# Tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument(
        "-i",
        "--input",
        required=True,
        help="Required. An input to process. The input must be a single image,"
        "a folder of images, video file or camera id.",
    )
    general.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Optional. Enable reading the input in a loop.",
    )
    general.add_argument(
        "-o",
        "--output",
        help="Optional. Name of the output file(s) to save.",
    )
    general.add_argument(
        "-json",
        "--output_json",
        default="output/",
        type=Path,
        help="Optional. Directory to save the json file.",
    )
    general.add_argument(
        "-limit",
        "--output_limit",
        default=0,
        type=int,
        help="Optional. Number of frames to store in output. "
        "If 0 is set, all frames are stored.",
    )
    general.add_argument(
        "--output_resolution",
        default=None,
        type=resolution,
        help="Optional. Specify the maximum output window resolution "
        "in (width x height) format. Example: 1280x720. "
        "Input frame size used by default.",
    )
    general.add_argument(
        "--no_show",
        action="store_true",
        help="Optional. Don't show output.",
    )
    general.add_argument(
        "--crop_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Crop the input stream to this resolution.",
    )
    general.add_argument(
        "--match_algo",
        default="HUNGARIAN",
        choices=("HUNGARIAN", "MIN_DIST"),
        help="Optional. Algorithm for face matching. Default: HUNGARIAN.",
    )
    general.add_argument(
        "-u",
        "--utilization_monitors",
        default="",
        type=str,
        help="Optional. List of monitors to show initially.",
    )
    general.add_argument(
        "--option_detect_webcam",
        default = 'OFF',
        type = str,
        help= "Optional. Choose the way detect webcam",
    )
    general.add_argument(
        "--frames_per_sec",
        default=15,
        type=int,
        help="Optional. Number of frames to be process in a second.",
    )

    gallery = parser.add_argument_group("Faces database")
    gallery.add_argument(
        "-fg",
        default="",
        help="Optional. Path to the face images directory.",
    )
    gallery.add_argument(
        "--run_detector",
        action="store_true",
        help="Optional. Use Face Detection model to find faces "
        "on the face images, otherwise use full images.",
    )
    gallery.add_argument(
        "--allow_grow",
        action="store_true",
        help="Optional. Allow to grow faces gallery and to dump on disk. "
        "Available only if --no_show option is off.",
    )

    models = parser.add_argument_group("Models")
    models.add_argument(
        "-m_fd",
        default=MODEL_FACE_DETECTION,
        type=Path,
        required=False,
        help="Required. Path to an .xml file with Face Detection model.",
    )
    models.add_argument(
        "-m_lm",
        default=MODEL_LANDMARKS,
        type=Path,
        required=False,
        help="Required. Path to an .xml file with Facial Landmarks Detection model.",  # noqa: E501
    )
    models.add_argument(
        "-m_reid",
        default=MODEL_FACE_REIDENTIFICATION,
        type=Path,
        required=False,
        help="Required. Path to an .xml file with Face Reidentification model.",  # noqa: E501
    )
    models.add_argument(
        "--fd_input_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Specify the input size of detection model for "
        "reshaping. Example: 500 700.",
    )

    infer = parser.add_argument_group("Inference options")
    infer.add_argument(
        "-d_fd",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Face Detection model. "
        "Default value is CPU.",
    )
    infer.add_argument(
        "-d_lm",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Facial Landmarks Detection "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-d_reid",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Face Reidentification "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-d_soundbar",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Soundbar Detection "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Optional. Be more verbose.",
    )
    infer.add_argument(
        "-t_fd",
        metavar="[0..1]",
        type=float,
        default=0.6,
        help="Optional. Probability threshold for face detections.",
    )
    infer.add_argument(
        "-t_id",
        metavar="[0..1]",
        type=float,
        default=0.3,
        help="Optional. Cosine distance threshold between two vectors "
        "for face identification.",
    )
    infer.add_argument(
        "-exp_r_fd",
        metavar="NUMBER",
        type=float,
        default=1.15,
        help="Optional. Scaling ratio for bboxes passed to face recognition.",
    )
    return parser


def main():
    start_time = time.time()
    args = build_argparser().parse_args()

    class_config = ClassConfig(args)
    face_processor = FaceProcessor(args, class_config)
    voice_processor = VoiceProcessor(args, class_config)
    identity_processor = IdentityProcessor()
    ocr = OCR()

    frame_num = 0

    if args.output:
        height = int(class_config.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(class_config.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            class_config.fps,
            (width, height),
        )
        if video_writer.open is False:
            raise RuntimeError("Can't open video writer")

    frame_start = 0
    frame_num = frame_start
    class_config.video.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    while class_config.video.isOpened():
        # process only some frames every second
        if frame_num % class_config.interval_frames > 0:
            frame_num += 1
            continue
        class_config.read_current_frame(frame_num)

        if class_config.current_frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break

        # Detect starting time of the class
        if class_config.class_start_flag is False:
            if frame_num % class_config.fps == 0:
                class_config.detect_starting_time(ocr, frame_num)

            frame_num += 1
            continue

        # Get class duration by extracting the ending time
        if class_config.frames - 1 == frame_num:
            class_config.get_class_duration(ocr)

        # Get number of people in the frame
        class_config.get_total_participants()

        # Get current webcam frame and position
        if args.option_detect_webcam == 'ON':
            class_config.get_webcam_frame_option_ON()
        elif args.option_detect_webcam == 'OFF':
            class_config.get_webcam_frame_option_OFF()

        # Recognize identities
        identity_processor.recognize_names(class_config, ocr)

        # Detect faces
        face_processor.get_detected_faces(class_config.webcam_frames)
        # TODO: not recognize icon face when awarding
        # -> use face_reidentification

        # Detect talking persons
        voice_processor.get_detected_voices(class_config.webcam_frames)

        # Write ouput
        if args.output:
            frame = draw_detections(
                class_config,
                identity_processor,
                face_processor,
                voice_processor,
            )
            video_writer.write(frame)
            # cv2.imshow("Face recognition demo", frame)

            # # Press Q on keyboard to exit
            # if cv2.waitKey(25) & 0xFF == ord("q"):
            #     break

        # Show processing status
        processing_status(
            frame_num,
            class_config,
            identity_processor,
            face_processor,
            voice_processor,
        )

        frame_num += 1

    processing_time = time.time() - start_time
    # if args.output:
    # add_audio2video(args)

    return format_output(
        args,
        class_config,
        identity_processor,
        face_processor,
        voice_processor,
        processing_time,
    )


def processing_status(
    frame_num: int,
    class_config: ClassConfig,
    identity_processor: IdentityProcessor,
    face_processor: FaceProcessor,
    voice_processor: VoiceProcessor,
):
    class_config.frame_class = frame_num - class_config.frame_start
    if (
        class_config.frame_class
        % (
            face_processor.interval_frames
            * face_processor.ratio_interval_frames
        )
        == 0  # noqa: E501
        and class_config.frame_class > 0
    ):
        logger.info(
            f"Class time: {convert_frame_to_time(class_config.fps, class_config.frame_class)} - frame: {class_config.frame_class}/{frame_num} - {class_config.max_participants} participants",  # noqa: E501
        )

        for id in range(class_config.participants):
            face_time = convert_frame_to_time(
                class_config.fps,
                face_processor.whole_video[id],
            )
            voice_time = convert_frame_to_time(
                class_config.fps,
                voice_processor.whole_video[id],
            )
            info = (
                f"{identity_processor.identities[id]}"
                # f"\tface: {face_time}"  # noqa: E501
                f"\tvoice: {voice_time}"  # noqa: E501
            )
            logger.info(info)


def format_output(
    args,
    class_config,
    identity_processor,
    face_processor: FaceProcessor,
    voice_processor,
    processing_time,
) -> dict:
    """
    Reformat the output of the video analysis
    OUTPUT format: dictionary
    {
        name_1:
        {
            "face_showing": {total_frame, total_time, percentage}
            "talking": {total_frame, total_time, percentage}
        },
        ... ,
    }
    processing_time: float
    """
    recognition = {}
    for id in range(class_config.max_participants):
        face_frame = face_processor.whole_video[id]
        face_time = convert_frame_to_time(
            class_config.fps,
            face_processor.whole_video[id],
        )
        face_percentage = (
            face_processor.whole_video[id] / class_config.frame_class * 100
        )
        face_percentage = format_decimal_number(face_percentage)

        voice_frame = voice_processor.whole_video[id]
        voice_time = convert_frame_to_time(
            class_config.fps,
            voice_processor.whole_video[id],
        )
        voice_percentage = (
            voice_processor.whole_video[id] / class_config.frame_class * 100
        )
        voice_percentage = format_decimal_number(voice_percentage)

        recognition[identity_processor.identities[id]] = {
            "face_showing": {
                "frame": face_frame,
                "time": face_time,
                "percentage": face_percentage,
                # "photo": face_processor.photos[id][1].tolist(),
            },
            "talking": {
                "frame": voice_frame,
                "time": voice_time,
                "percentage": voice_percentage,
            },
        }

    processing_time = format_decimal_number(processing_time / 60)

    results = {
        "recognition": recognition,
        "video_duration": convert_frame_to_time(
            class_config.fps,
            class_config.frames,
        ),
        "class_duration": convert_frame_to_time(
            class_config.fps,
            class_config.frame_class,
        ),
        "processing_time": processing_time,
    }

    # write results to json file
    video_id = os.path.basename(args.input)
    video_id = os.path.splitext(video_id)[0]
    file_name = video_id + ".json"
    json_dir = os.path.join(
        ROOT_DIR,
        args.output_json,
        file_name,
    )

    with open(json_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Export results to {json_dir}")

    # write image
    for id in range(len(face_processor.photos)):
        if face_processor.photos[id][0] > 0:
            filename = (
                video_id + "_" + identity_processor.identities[id] + ".png"
            )
            image_dir = os.path.join(
                ROOT_DIR,
                args.output_json,
                filename,
            )
            cv2_ext.imwrite(image_dir, face_processor.photos[id][1])

    return json.dumps(results, indent=4, sort_keys=False)


def draw_detections(
    class_config: ClassConfig,
    identity_processor: IdentityProcessor,
    face_processor: FaceProcessor,
    voice_processor: VoiceProcessor,
) -> np.ndarray:
    """
    Method to draw detection of the following things:
    - Voice detection
    - Face detection
    - Identity recognition
    """
    output_frame = class_config.current_frame
    output_frame = draw_voice_detection(
        class_config,
        voice_processor,
        output_frame,
    )
    output_frame = draw_face_detection(
        class_config,
        identity_processor,
        face_processor,
        output_frame,
    )
    return output_frame


def draw_voice_detection(
    class_config: ClassConfig,
    voice_processor: VoiceProcessor,
    output_frame: np.ndarray,
) -> np.ndarray:
    """
    Draw bounding boxes around the webcam of a talking person
    """

    for i in range(class_config.participants):
        top = class_config.webcam_positions[i][0] + 5
        bot = class_config.webcam_positions[i][1] - 3
        left = class_config.webcam_positions[i][2]
        right = class_config.webcam_positions[i][3] - 2

        # Draw bounding box of the webcam
        if i in voice_processor.current_frame:
            cv2.rectangle(
                output_frame,
                (left, top),
                (right, bot),
                (0, 0, 255),  # BGR
                2,
            )

        """ Draw number of detected soundbars in each webcam """
        text = f"soundbars = {voice_processor.total_soundbar[i]}"

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            thickness=1,
        )

        # Fill text background
        cv2.rectangle(
            output_frame,
            (left, top),
            (left + text_width, top - text_height),
            (255, 255, 0),
            cv2.FILLED,
        )

        # Write text
        cv2.putText(
            output_frame,
            text=text,
            org=(left, top),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(0, 0, 0),
            thickness=1,
        )

    return output_frame


def draw_face_detection(
    class_config: ClassConfig,
    identity_processor: IdentityProcessor,
    face_processor: FaceProcessor,
    output_frame: np.ndarray,
) -> np.ndarray:
    """
    Draw bounding boxes around the detected faces with confident score
    Write the name of person in the webcam
    """

    size = output_frame.shape[:2]
    # Draw face recognition
    for i in range(len(face_processor.detections)):
        detection = face_processor.detections[i]
        for roi, landmarks, identity in zip(*detection):
            xmin = max(int(roi.position[0]), 0)
            ymin = max(int(roi.position[1]), 0)
            xmax = min(int(roi.position[0] + roi.size[0]), size[1])
            ymax = min(int(roi.position[1] + roi.size[1]), size[0])

            ymin += class_config.webcam_positions[i][0]  # top
            ymax += class_config.webcam_positions[i][0]  # bot
            xmin += class_config.webcam_positions[i][2]  # left
            xmax += class_config.webcam_positions[i][2]  # right
            cv2.rectangle(
                output_frame,
                (xmin, ymin),
                (xmax, ymax),
                (0, 220, 0),
                2,
            )

            for point in landmarks:
                x = xmin
                y = ymin
                cv2.circle(output_frame, (int(x), int(y)), 1, (0, 255, 255), 2)

            # Get name and confident score
            if len(identity_processor.identities) < i + 1:
                text = ""
            else:
                text = identity_processor.identities[i]
            # if identity.id != FaceIdentifier.UNKNOWN_ID:
            # detection score in percentage
            text += " %.2f%%" % (100.0 * (1 - identity.distance))
            output_frame = write_recognized_name(
                output_frame,
                text,
                xmin,
                ymin,
            )

    return output_frame


def write_recognized_name(
    output_frame: np.ndarray,
    text: str,
    xmin: int,
    ymin: int,
) -> np.ndarray:
    """write name and confident score of person"""
    # this method can write utf-8 text

    text_width, text_height = vietnamese_font.getsize(text)

    cv2.rectangle(
        output_frame,
        (xmin, ymin),
        (xmin + text_width, ymin - text_height),
        (255, 255, 0),
        cv2.FILLED,
    )

    img_pil = convert_cv2_2_pil(output_frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((xmin, ymin - text_height), text, "black", vietnamese_font)
    return convert_pil_2_cv2(img_pil)


def add_audio2video(args):
    # def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    """
    Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module
    """
    # Read videos
    output_video = VideoFileClip(args.output)
    original_video = VideoFileClip(args.input)

    # Extract duration
    output_duration = output_video.duration
    original_duration = original_video.duration

    # Trim original audio
    trimmed_video = original_video.subclip(
        original_duration - output_duration,
        original_duration,
    )

    # Extract audio from trimmed video
    trimmed_audio = trimmed_video.audio

    # Add audio to output video
    output_video = output_video.set_audio(trimmed_audio)

    # Write audio with audio
    output_video.write_videofile(args.output)


if __name__ == "__main__":
    sys.exit(main() or 0)
