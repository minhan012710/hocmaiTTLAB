from log_info import setup_logger
from openvino.runtime import AsyncInferQueue

logger = setup_logger("ie_module")


class Module:
    def __init__(self, core, model_path, model_type):
        self.core = core
        self.model_type = model_type
        logger.info(f"Reading {model_type} model {model_path}")
        self.model = core.read_model(model_path)
        self.model_path = model_path
        self.active_requests = 0
        self.clear()

    def deploy(self, device, max_requests=1):
        self.max_requests = max_requests
        compiled_model = self.core.compile_model(self.model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_queue = AsyncInferQueue(compiled_model, self.max_requests)
        self.infer_queue.set_callback(self.completion_callback)
        logger.info(
            "The {} model {} is loaded to {}".format(
                self.model_type,
                self.model_path,
                device,
            ),
        )

    def completion_callback(self, infer_request, id):
        self.outputs[id] = infer_request.results[self.output_tensor]

    def enqueue(self, input):
        self.clear()

        if self.max_requests <= self.active_requests:
            logger.warning("Processing request rejected - too many requests")
            return False

        self.infer_queue.start_async(input, self.active_requests)
        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return
        self.infer_queue.wait_all()
        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        return [v for _, v in sorted(self.outputs.items())]

    def clear(self):
        self.outputs = {}

    def infer(self, inputs):
        self.clear()
        self.start_async(*inputs)
        return self.postprocess()
