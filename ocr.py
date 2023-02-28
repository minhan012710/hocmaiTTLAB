import sys
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from settings import (
    MODEL_OCR,
    OCR_DEVICE,
    OCR_NETWORK,
)
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


class OCR:
    def __init__(self):
        config = Cfg.load_config_from_name(OCR_NETWORK)
        config["weights"] = MODEL_OCR
        config["cnn"]["pretrained"] = False
        config["device"] = OCR_DEVICE
        config["predictor"]["beamsearch"] = False

        self.recognizer = Predictor(config)

    def recognize(self, image: np.ndarray) -> str:
        """
        Method for recognizing text in an image
        """
        recognized_text, prob = self.recognizer.predict(
            image,
            return_prob=True,
        )
        return recognized_text, prob


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
        "-m",
        "--model",
        required=True,
        help="Required. Model to recognize text.",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    config = Cfg.load_config_from_name("vgg_transformer")
    config["weights"] = args.model
    config["cnn"]["pretrained"] = False
    config["device"] = "cpu"  # gpu:0
    config["predictor"]["beamsearch"] = False

    detector = Predictor(config)

    img = Image.open(args.input)
    recognized_text = detector.predict(img)
    return recognized_text


if __name__ == "__main__":
    sys.exit(main() or 0)
