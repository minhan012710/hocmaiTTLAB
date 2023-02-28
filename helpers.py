from log_info import setup_logger

logger = setup_logger("helpers")


def resolution(value: str):
    """
    get resolution from input text
    e.g., 10x15 -> [10, 15]
    """
    try:
        result = [int(v) for v in value.split("x")]
        if len(result) != 2:
            raise RuntimeError(
                'Сorrect format of --output_resolution parameter is "width"x"height".',  # noqa: E501
            )
    except ValueError:
        raise RuntimeError(
            'Сorrect format of --output_resolution parameter is "width"x"height".',  # noqa: E501
        )
    return result


def log_latency_per_stage(*pipeline_metrics):
    stages = (
        "Decoding",
        "Preprocessing",
        "Inference",
        "Postprocessing",
        "Rendering",
    )
    for stage, latency in zip(stages, pipeline_metrics):
        logger.info(f"\t{stage}:\t{latency:.1f} ms")
