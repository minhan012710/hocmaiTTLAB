import numpy as np
from face_detector import FaceDetector
from face_identifier import FaceIdentifier
from faces_database import FacesDatabase
from landmarks_detector import LandmarksDetector
from log_info import setup_logger
from openvino.runtime import (
    Core,
    get_version,
)
from settings import (
    FACE_INTERVAL_SECONDS,
    REGISTERED_FACE,
)
from utilities import crop
from video_config import ClassConfig

logger = setup_logger("face_recognition")


class FaceProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args, class_config: ClassConfig):
        self.allow_grow = args.allow_grow and not args.no_show

        logger.info(f"OpenVINO Runtime\tbuild: {get_version()}")
        core = Core()
        self.face_detector = FaceDetector(
            core,
            args.m_fd,
            args.fd_input_size,
            confidence_threshold=args.t_fd,
            roi_scale_factor=args.exp_r_fd,
        )
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(
            core,
            args.m_reid,
            match_threshold=args.t_id,
            match_algo=args.match_algo,
        )
        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        args.fg = REGISTERED_FACE
        logger.debug(
            f"Building faces database using images from {args.fg}",
        )
        self.faces_database = FacesDatabase(
            args.fg,
            self.face_identifier,
            self.landmarks_detector,
            self.face_detector if args.run_detector else None,
            args.no_show,
        )
        self.face_identifier.set_faces_database(self.faces_database)
        logger.info(
            f"Database is built, registered {len(self.faces_database)} identities",  # noqa: E501
        )

        self.faces = {}
        self.current_frame = set()
        self.current_interval = set()
        self.whole_video = []
        self.interval_frames = args.frames_per_sec * FACE_INTERVAL_SECONDS
        self.ratio_interval_frames = int(
            class_config.fps / args.frames_per_sec,
        )
        self.frame_count_in_an_interval = 0
        self.detections = []
        self.photos = []

    def process(self, frame: np.ndarray) -> np.ndarray:
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            logger.warning(
                "Too many faces for processing. Will be processed only {} of {}".format(  # noqa: E501
                    self.QUEUE_SIZE,
                    len(rois),
                ),
            )
            rois = rois[: self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer(
            (frame, rois, landmarks),
        )
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images
                # in the boundary of images
                if (
                    rois[i].position[0] == 0.0
                    or rois[i].position[1] == 0.0
                    or (
                        rois[i].position[0] + rois[i].size[0]
                        > orig_image.shape[1]
                    )
                    or (
                        rois[i].position[1] + rois[i].size[1]
                        > orig_image.shape[0]
                    )
                ):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(
                        crop_image,
                        face_identities[i].descriptor,
                        name,
                    )
                    face_identities[i].id = id

        return [rois, landmarks, face_identities]

    def get_detected_faces(
        self,
        webcams: np.ndarray,
    ):
        """
        Get detected faces and their showing times
        """
        # Detect faces
        total_participants = len(webcams)
        self.current_frame = set()
        self.detections = []

        for id in range(total_participants):
            if id >= len(self.whole_video):
                self.whole_video.append(0)
            detection = self.process(webcams[id])
            if len(detection[1]) > 0:
                self.current_frame.add(id)
            self.detections.append(detection)

            self.get_face_photo(webcams[id], detection, id)

        # Calculate showing time of each face
        self.update_total_showing_time()

    def get_face_photo(
        self,
        webcam: np.ndarray,
        detection: np.ndarray,
        id: int,
    ):
        # Get photo of the face
        size = webcam.shape[:2]
        if id >= len(self.photos):
            self.photos.append([0, np.array([], dtype=np.uint8)])

        # Save the face photo
        for roi, _, identity in zip(*detection):
            score = 100.0 * (1 - identity.distance)
            # Update image if the new score is higher then the saved score
            if score > self.photos[id][0]:
                xmin = max(int(roi.position[0]) - int(roi.size[0] * 0.2), 0)
                ymin = max(int(roi.position[1]) - int(roi.size[1] * 0.2), 0)
                xmax = min(
                    int(roi.position[0] + int(roi.size[0] * 1.2)),
                    size[1],
                )
                ymax = min(
                    int(roi.position[1] + int(roi.size[1] * 1.2)),
                    size[0],
                )

                self.photos[id] = [score, webcam[ymin:ymax, xmin:xmax].copy()]

    def update_total_showing_time(self):
        """
        Calculate showing time of each person up to the current frame
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
