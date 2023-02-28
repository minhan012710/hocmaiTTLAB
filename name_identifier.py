import numpy as np
from log_info import setup_logger
from ocr import OCR
from settings import (
    FRAME_NAME, 
    NAME_LEFT,
    NAME_WIDTH,
    NAME_BOT,
    NAME_TOP,
)
from utilities import convert_cv2_2_pil
from video_config import ClassConfig

logger = setup_logger("name_identifier")


class IdentityProcessor:
    def __init__(self):
        self.identities = []
        self.reidentified_list = set()

    def recognize_names(self, class_config: ClassConfig, ocr: OCR):
        """
        Recognize name in each newly updated webcam
        """
        if len(self.reidentified_list) > 0:
            self.recognize_unregcognized_name(class_config, ocr)

        if len(self.identities) == class_config.updated_participants:
            return

        if class_config.participants_change > 0:
            for i in range(class_config.participants_change):
                # TODO: what if a student outs and rejoin the class?
                # TODO: a student outs, another student joins?
                idx = (
                    class_config.participants
                    - class_config.participants_change
                    + i
                )
                identity = ""

                # Check if webcam is in the correct position
                # (webcam may be moved by teacher)
                if class_config.check_existing_webcam(idx) is True: 
                    identity = recognize_name(
                        ocr,
                        class_config.webcam_frames[idx],
                    )
                    identity = identity.strip()

                if identity not in self.identities:
                    self.identities.append(identity)
                    logger.info(f"self.identities={self.identities}")
                if identity == "":
                    self.reidentified_list.add(
                        len(self.identities) - 1,
                    )

    def recognize_unregcognized_name(
        self,
        class_config: ClassConfig,
        ocr: OCR,
    ):
        """
        Recognize the unregcognized name.
        Reason for unregcognized name:
        - Webcam is not in the correct posistion
        - Wrongly recognition (all number, blank)
        """
        new_identitfied_list = set()

        for i in self.reidentified_list:
            # Check if webcam is in the correct position
            # (webcam may be moved by teacher)
            if class_config.check_existing_webcam(i) is True:
                self.identities[i] = recognize_name(
                    ocr,
                    class_config.webcam_frames[i],
                )

            if self.identities[i] != "":
                new_identitfied_list.add(i)

        # Remove the sucessfully recognized name from the list
        self.reidentified_list = self.reidentified_list.difference(
            new_identitfied_list,
        )


def crop_name_area(image) -> np.ndarray:
    """
    Crop the area containing nickname in a webcam
    """
    height = image.shape[0]
    
    return image[
        (height + NAME_TOP) : (height + NAME_BOT),    
        NAME_LEFT : (NAME_LEFT + NAME_WIDTH),  
    ]


def recognize_name(ocr_engine: OCR, webcam: np.ndarray) -> str:
    """
    Recognize nickname in a webcam
    """
    name_area = crop_name_area(webcam)
    name_area_pil = convert_cv2_2_pil(name_area)
    recognized_text, prob = ocr_engine.recognize(name_area_pil)
    # import cv2_ext
    # cv2_ext.imwrite(f"output/{recognized_text}.png", name_area)

    if prob < 0.5:
        return ""
    return recognized_text
