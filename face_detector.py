import numpy as np
from ie_module import Module
from log_info import setup_logger
from openvino.runtime import PartialShape
from utilities import resize_input

logger = setup_logger("3rd_logger")


class FaceDetector(Module):
    class Result:
        OUTPUT_SIZE = 7

        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4]))  # (x, y)
            self.size = np.array((output[5], output[6]))  # (w, h)

        def rescale_roi(self, roi_scale_factor=1.0):
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = np.clip(self.position, min, max)
            self.size[:] = np.clip(self.size, min, max)

    def __init__(
        self,
        core,
        model,
        input_size,
        confidence_threshold=0.5,
        roi_scale_factor=1.15,
    ):
        super().__init__(core, model, "Face Detection")

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        if input_size[0] > 0 and input_size[1] > 0:
            self.model.reshape(
                {self.input_tensor_name: PartialShape([1, 3, *input_size])},
            )
        elif not (input_size[0] == 0 and input_size[1] == 0):
            raise ValueError(
                "Both input height and width should be positive for Face Detector reshape",  # noqa: E501
            )

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        self.output_shape = self.model.outputs[0].shape
        if (
            len(self.output_shape) != 4
            or self.output_shape[3] != self.Result.OUTPUT_SIZE
        ):
            raise RuntimeError(
                f"The model expects output shape with {self.Result.OUTPUT_SIZE} outputs",  # noqa: E501
            )

        if confidence_threshold > 1.0 or confidence_threshold < 0:
            raise ValueError(
                "Confidence threshold is expected to be in range [0; 1]",
            )
        if roi_scale_factor < 0.0:
            raise ValueError("Expected positive ROI scale factor")

        self.confidence_threshold = confidence_threshold
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        self.input_size = frame.shape
        return resize_input(frame, self.input_shape, self.nchw_layout)

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super().enqueue({self.input_tensor_name: input})

    def postprocess(self):
        outputs = self.get_outputs()[0]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]

        results = []
        for output in outputs[0][0]:
            result = FaceDetector.Result(output)
            if result.confidence < self.confidence_threshold:
                break  # results are sorted by confidence decrease

            result.resize_roi(self.input_size[1], self.input_size[0])
            result.rescale_roi(self.roi_scale_factor)
            result.clip(self.input_size[1], self.input_size[0])
            results.append(result)

        return results
