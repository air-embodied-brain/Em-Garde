from vlm2vec.arguments import ModelArguments as VLM2VecModelArguments
from vlm2vec.arguments import DataArguments as VLM2VecDataArguments

class ModelArguments:
    def __init__(self,
                detector_type: str = 'vlm2vec',
                proposer_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                responder_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                detector_config = None,
                history_length: int = 15,
                history_fps: int = 2,
                streaming_fps: int = 5,
                streaming_length: int = 2,
                should_respond: bool = False,
                detect_threshold: float = 0.04
                ):
        self.detector_type = detector_type
        self.proposer_model_name = proposer_model_name
        self.responder_model_name = responder_model_name
        self.detector_config = detector_config
        self.history_length = history_length
        self.history_fps = history_fps
        self.streaming_fps = streaming_fps
        self.streaming_length = streaming_length
        self.should_respond = should_respond
        self.detect_threshold = detect_threshold