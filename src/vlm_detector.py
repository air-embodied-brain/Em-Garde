from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
import json
import re
import time

class VLMDetector:
    def __init__(self, model_name: str, device: str = 'cuda'):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = "flash_attention_2"
        if '2_5' in model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
        self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
    
    @torch.no_grad()    
    def detect(self, video_frames, proposal_list):
        prompt = "Does one of the following descriptions in the list match what's happening in the above video? If ANY one or more of them matches the video, answer 'Yes'. Otherwise, answer 'No'. Only answer in 'Yes' or 'No'. Here are the descriptions: {}\n Your answer:".format(proposal_list)
        # match = re.findall(r"""(['"])(.*?)\1""", proposal_list[0])
        # print("Proposal to check:", proposal_list[0], "Extracted output:", match[0][1] if match else "N/A")
        # if match:
        #     output = match[0][1]
        # prompt = "{} Is it the right time to output \"{}\"? You can only answer yes or no.".format(proposal_list[0], output)
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": "dummy_video"},  # Placeholder for video input
                {"type": "text", "text": prompt}
            ]
            }
        ]
            
        message = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print("Generated message:", message)
        inputs = self.processor(text=[message], videos=[video_frames], return_tensors="pt", padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        # start_time = time.time()        
        response = self.model.generate(**inputs, max_new_tokens=1)
        # end_time = time.time()
        # print("Generation time:", end_time - start_time)
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[:, input_len:]
        
        outputs = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        print("Model output:", outputs)
        
        return True if 'yes' in outputs[0].lower() else False