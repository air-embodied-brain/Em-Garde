import torch

from vlm2vec.arguments import ModelArguments, DataArguments
from vlm2vec.model.model import MMEBModel
from vlm2vec.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS, VLM_IMAGE_TOKENS
from vlm2vec.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info

from embedding_models.ops_mm_embedding_v1 import OpsMMEmbeddingV1
from transformers import SiglipProcessor, SiglipModel
# TODO: vlm detector seems too "specialized". Better restructure needed.
from src.vlm_detector import VLMDetector
import time

class VLM2VecDetector:
    def __init__(self, model_args: ModelArguments, data_args: DataArguments, device='cuda'):
        self.device = device
        
        processor = load_processor(model_args, data_args)
        model = MMEBModel.load(model_args)
        model = model.to(device, dtype=torch.bfloat16)
        model.eval()
        
        self.processor = processor
        self.model = model
        
    @torch.no_grad()
    def encode_image(self, image, question=None):
        question = "What is in the image" if question is None else question
        inputs = self.processor(text=f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: {question}.',
                   images=[image],
                   return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
        qry_output = self.model(qry=inputs)["qry_reps"]
        return qry_output
        
    @torch.no_grad()
    def encode_video(self, video_frames, question=None):
        start_time = time.time()
        
        # print(video_frames.shape)
        if question is None:
            inputs = self.processor(
            text=[f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.'],
            videos=[video_frames],
            return_tensors="pt"
        )
        else:
            # inputs = self.processor(
            #     text=[f'{VLM_VIDEO_TOKENS[QWEN2_VL]} {question}.'],
            #     videos=[video_frames],
            #     return_tensors="pt"
            # )
            inputs = self.processor(
                text=[f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Answer a question based on the content of a video: {question}.'],
                videos=[video_frames],
                return_tensors="pt"
            )
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
        inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        start_time2 = time.time()
        qry_output = self.model(qry=inputs)["qry_reps"]
        end_time = time.time()
        # print("video encoding time:", end_time - start_time)
        # print("model forward time:", end_time - start_time2)
        return qry_output
        # print(self.curr_qry_rep.shape)
    
    @torch.no_grad()
    def encode_targets(self, targets):
        self.curr_tgt_rep = None
        
        inputs = self.processor(text=targets,
                images=None,
                return_tensors="pt",
                padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        tgt_output = self.model(tgt=inputs)["tgt_reps"]
        return tgt_output
        # print(self.curr_tgt_rep.shape)
        
    # def compute_similarity(self):
    #     similarities = torch.cosine_similarity(self.curr_qry_rep, self.curr_tgt_rep, dim=-1)
        
    #     return similarities
    
class OPSMMEmbeddingV1Detector:
    def __init__(self,model_name: str = "OpenSeachAI/Ops-MM-embedding-v1-2B/", device='cuda', attn_implementation="flash_attention_2"):
        self.device = device
        self.model = OpsMMEmbeddingV1(model_name, device=device, attn_implementation=attn_implementation)
        self.model.eval()
        self.past_visual_embed = None
        
    def encode_image(self, image, question=None):
        if question is None:
            embeds = self.model.get_image_embeddings([image])
        else:
            instruction = "Answer the question given the video frames."
            embeds = self.model.get_fused_embeddings(texts=[question],images=[image],instruction=instruction)
        return embeds
    
    def encode_video(self, new_frames, question=None, frames_length=10):
        start_time = time.time()
        if question is None:
            embeds = self.model.streaming_embed(new_frames, frames_length=frames_length)
        else:
            pass
            # instruction = "Answer the question given the video frames."
            # embeds = self.model.get_fused_embeddings(texts=[question],images=[list(video_frames)],instruction=instruction)
        end_time = time.time()
        # print("video encoding time:", end_time - start_time)
        return embeds
    
    def encode_targets(self, targets):
        embeds = self.model.get_text_embeddings(targets)
        return embeds
    
    def reset(self):
        self.model.reset()
    
class SiglipDetector:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device='cuda'):
        self.device = device
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device).eval()
    
    @torch.no_grad()
    def encode_image(self, image, question=None):
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    
    @torch.no_grad()  
    def encode_targets(self, targets):
        inputs = self.processor(text=targets, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        return outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
    @torch.no_grad()
    def encode_video(self, video_frames, question=None):
        raise ValueError("SiglipDetector does not support video encoding.")
        
    # def compute_similarity(self):
    #     similarities = torch.cosine_similarity(self.curr_qry_rep, self.curr_tgt_rep, dim=-1)
        
    #     return similarities
    
# detector configs    
from dataclasses import dataclass, field

@dataclass
class VLM2VecConfig:
    model_args: dict = field(default_factory=dict)
    data_args: dict = field(default_factory=dict)

@dataclass
class OPSMMConfig:
    model_name: str = "OpenSearchAI/Ops-MM-embedding-v1-2B/"

@dataclass
class SiglipConfig:
    model_name: str = "google/siglip-base-patch16-224"

@dataclass
class VLMConfig:
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    
import yaml
    
def load_detector_config(detector_type, path):
    with open(path) as f:
        raw = yaml.safe_load(f)

    if detector_type == "vlm2vec":
        return VLM2VecConfig(**raw)
    elif detector_type == "siglip":
        return SiglipConfig(**raw)
    elif detector_type == "opsmm_embedding_v1":
        return OPSMMConfig(**raw)
    elif detector_type == "vlm":
        return VLMConfig(**raw)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

def build_detector(detector_type: str, config, device="cuda"):
    if detector_type == "vlm2vec":
        from vlm2vec.arguments import ModelArguments, DataArguments

        model_args = ModelArguments(**config.model_args)
        data_args = DataArguments(**config.data_args)

        return VLM2VecDetector(
            model_args=model_args,
            data_args=data_args,
            device=device,
        )

    elif detector_type == "siglip":
        return SiglipDetector(
            model_name=config.model_name,
            device=device,
        )

    elif detector_type == "opsmm_embedding_v1":
        return OPSMMEmbeddingV1Detector(
            model_name=config.model_name,
            device=device,
        )

    elif detector_type == "vlm":
        return VLMDetector(
            model_name=config.model_name,
            device=device,
        )

    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    
    
if __name__ == "__main__":
    model_args = ModelArguments(
        model_name='/data/VLM2Vec',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=False
    )
    data_args = DataArguments()
    
    video_path = "assets/bacon_and_eggs.mp4"
    from src.video_utils import read_video_decord
    video_frames = read_video_decord(video_path, 146, 148, 5)
        
        