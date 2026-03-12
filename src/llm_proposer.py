from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
import json
import re
import time

from .video_utils import read_video_decord

def safe_json_loads(s):
    """
    Attempt to parse a JSON list of strings from an LLM output.
    Tries to fix common formatting errors such as:
    - Missing commas
    - Single quotes instead of double quotes
    - Extra text before/after JSON
    - Missing brackets
    """
    s_origin = s
    s = s.strip()

    # 1. Try direct load first
    try:
        data = json.loads(s)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass

    # 2. Extract JSON-like content between [ and ]
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if match:
        s = match.group(0)
    else:
        raise ValueError("No JSON-like list found.")

    # 3. Replace single quotes with double quotes (safe enough for simple text)
    s = re.sub(r"(?<!\\)'", '"', s)

    # 4. Fix trailing commas
    s = re.sub(r",\s*\]", "]", s)

    # 5. Remove newlines and excessive whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    inner = s.strip("[]").strip()
    if not inner:
        return []

    # If no comma → treat as one string
    if "," not in inner:
        return [inner.strip('"').strip("'").strip()]

    # --- Key addition: quote unquoted items ---
    # Example: [A, B, C] → ["A", "B", "C"]
    # Handles already-quoted strings safely
    parts = [p.strip() for p in inner.split(",")]
    fixed_parts = []
    for p in parts:
        if not (p.startswith('"') and p.endswith('"')):
            if not (p.startswith("'") and p.endswith("'")):
                p = '"' + p.strip('"').strip("'").strip() + '"'
        fixed_parts.append(p)
    repaired = "[" + ", ".join(fixed_parts) + "]"

    # 6. Retry parsing
    try:
        data = json.loads(repaired)
        # ensure it's a list of strings
        if isinstance(data, list):
            data = [str(x) for x in data]
            return data
        else:
            raise ValueError("Parsed JSON is not a list.")
    except Exception as e:
        raise ValueError(f"Failed to clean and parse JSON: {e}\nCleaned string:\n{repaired}, original string:\n{s_origin}")
    
    

def recover_truncated_json_list(text: str):
    """
    Recover a truncated JSON list of objects.
    If the list is missing a closing ']', truncate to the last valid object and close the list.
    """

    # Fast path: already valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip leading/trailing whitespace
    s = text.strip()

    # Must look like a list
    if not s.startswith("["):
        raise ValueError("Input does not start with '['")

    # Scan to find last valid object boundary
    stack = []
    last_valid_end = None
    in_string = False
    escape = False

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                stack.append("{")
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        # closed a full top-level object
                        last_valid_end = i

    if last_valid_end is None:
        raise ValueError("No complete JSON object found")

    # Cut after the last complete object
    recovered = s[: last_valid_end + 1]

    # Remove trailing commas
    recovered = recovered.rstrip(", \n\t")

    # Close the list
    recovered += "\n]"
    
    return json.loads(recovered)

class LLMProposer:
    def __init__(self, model_name: str, device: str = 'cuda'):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = "flash_attention_2"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
        self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        
    @torch.no_grad()
    def propose(self, video_frames, query: str):
        # video_timestamps, video_frames = read_video_decord(video_path, current_timestamp - history_length, current_timestamp, fps, max_pixels = 280*280)
        
#         prompt = '''You are an AI assistant specialized in streaming video understanding. The above video frames shows the recent history of a video stream. Next you will see a query that may depend on future content to answer. Your task is to propose a description that might help identifying key frames to answer the question. Later, we will use an embedding model to compare your description with future frames to identify the key frames. Follow these rules:
# 1. Imagine what will the scene be like when the query is addressed. If the query mentions specific objects, actions, or places, use the history frames to retrieve detailed descriptions of them, and imagine how they might evolve in the future.
# 2. Always propose something that may happen in the future given the objects, environment, and actions in the history frames. Do not propose something that is impossible, unrelated to the situation or highly unlikely.
# 3. Include detailed possible visual descriptions, like colors, shapes, interactions, spatial relations and scene context of salient elements. Keep the descriptions informative for better similarity scores.
# 4. If there are vast possibilities, keep the uncertain part general while focusing on the detailed description of the more certain elements.
# 5. Describe the whole possible scene instead of directly answering the query.
# 6. Provide one proposal to describe one possibility related to the query. Only output the description, do not include any additional explanations or notes.
# Query:{}
# Your answer:'''.format(query)

#         prompt = '''You are an AI assistant specialized in streaming video understanding. The above video frames shows the recent history of a video stream. Next you will see a query that may depend on future content to answer. Your task is to propose some descriptions that might help identifying key frames to answer the question. Later, we will use an embedding model to compare your description with future frames to identify the key frames. Follow these rules:
# 1.  Fully utilize the video as your context to imagine what will the scene be like when the query is addressed. Convert object names, actions, place names in the query to detailed descriptions based on the video.
# 2. Always propose something that may happen in the future given the objects, environment, and actions in the history frames. Do not propose something that is impossible, unrelated to the situation or highly unlikely.
# 3. Always describe in visual details, simple spatial relationships and simple actions. Avoid high-level action or event names that needs cognitive ability or context understanding to recognize.
# 4. Describe the whole possible scene including main objects and the background. Focus the details on main objects or objects that you’re certain of. Do not directly answer the query.
# 5. If no video content is provided, use your common sense to imagine possible scenarios related to the query.
# 6. Provide {} proposals to describe different possibilities related to the query. Only output the description, do not include any additional explanations or notes. If there are less than {} possibilities, just use similar sentences with slightly different wording.
# Arrange your answer in the following json format. Make sure you only output one closed list, each string is closed with double quotations, and use single quotations in each string.:
# ["First description","Second description",..."{}th description"]
# Query:{}
# Your answer:'''.format(num_proposals,num_proposals, num_proposals, query)

#         prompt = '''
# You are an AI assistant specialized in streaming-video understanding and future-moment retrieval. Task Goal: You will see a short video segment representing the recent history of a video stream. Then you will receive a Query that can only be answered using future frames. Your job is to anticipate what visual clues are likely to appear at the moment when the Query becomes answerable. These clues will be converted into text embeddings and matched against future video segments. High-similarity segments will be selected for answering the Query.
# Rules for generating visual clues:
# 1.	Imagine different possible scenes when the query becomes answrable. Then produce several visual clues, each describing a short caption of an event, an object, an action, or a spatial relationship. Each clue must be a stand-alone descriptive phrase.
# 2.	Combine clues using logical operators: AND, OR, NOT. Use AND for multiple clues in one possible scene. Use OR for different possible scenes. Use NOT for hard negatives: clues that are visually plausible but not associated with the answer. When creating a NOT clue, modify an object, action, or location so it is realistic but clearly unrelated.
# 3.	Be specific where possible. Use “someone” or “something” only when necessary. For the important and relevant elements, include detailed visual attributes such as colors, shapes, positions, or interactions.
# 4.	Always make clues informative for embedding models. Each clue should contain enough visual detail for a frame to rank highly by cosine similarity.
# Output Format: Produce a single logical expression composed of multiple clues connected by AND, OR, or NOT. Write it in one line only, for example: <Clue 1> AND <Clue 2> OR <Clue 3> AND NOT <Clue 4>.
# Query:{}
# Your Proposal:'''.format(query)

#         prompt = '''You are an AI assistant specialized in streaming-video understanding and future-moment retrieval. Task Goal: You will see a short video segment representing the recent history of a video stream. Then you will receive a Query that can only be answered using future frames. Your job is to anticipate what visual clues are likely to appear at the moment when the Query becomes answerable. These clues will be converted into text embeddings and matched against future video segments. High-similarity segments will be selected for answering the Query. 
# Rules for generate visual clues:
# 1. Imagine different possible scenes when the query becomes answerable. Then generate both positive clues and negative clues. 
# 2. Positive clues depict possible scenarios. Include multiple scenarios to include possibilities comprehensively. Clues can be a caption, an object, action, or spatial relationship. If you need several clues to appear together in one positive clue, use "AND" to connect them.
# 3. Negative clues shows clues which might happen in the video, contain similar clues as positive clues, but SHOULD NOT answer the query. Each clue can also contain sub-clues connected with AND.
# 4. Include visual details in the clues. Each clue should contain enough visual detail for a frame to rank highly by cosine similarity. If an element has too many possibilities or is too uncertain, use "someone" or "something" to keep it general.
# 5. The final criterion to determine that a segment answers the query is: one of positive is true and all negative are false.
# Output format: [{"positive": "<clue> AND <clue>"},{"positive": "<clue>},{"negative":<clue>},...]''' + '''Query: {}
# Your Proposal:'''.format(query)
        INSTRUCTION = '''You are an expert video understanding model. Above is a short video segment representing the recent history of a streaming video. Next you'll see a query that's related to future frames. Based on the video segment, propose some visual clues to help identify the frames that answer the query.\nRespond in the following json format:[\n  {\n    \"positive\": \"clue1\"\n  },\n  {\n    \"positive\": \"clue2\"\n  },...,\n  {\n    \"negative\": \"clue1\"\n  }\n,...]'''
        query_prompt = f"Query: {query}"
        prompt = INSTRUCTION + "\n" + query_prompt
        if video_frames is None:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]
                }
            ]
            message = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print("Generated message:", message)
            inputs = self.processor(text=[message], return_tensors="pt", padding=True)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
        else:                    
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
            inputs['second_per_grid_ts'] = inputs['second_per_grid_ts'].tolist()
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
        
        start_time = time.time()
        with torch.inference_mode():
            response = self.model.generate(**inputs, max_new_tokens=200,do_sample=False)
        # response = self.model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, num_return_sequences=5)
        end_time = time.time()
        # print("Generation time:", end_time - start_time)
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[:, input_len:]
        
        outputs = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        # print(outputs[0])

        try:
            outputs = recover_truncated_json_list(outputs[0])
        except Exception:
            outputs = {"json_parse_error": outputs[0]}
        # print(outputs)
        return outputs

if __name__ == "__main__":
    import time
    video_path = "assets/cat.mp4"
    current_timestamp = 5
    history_length = 5
    fps = 2
    query = "What is the cat eating?"
    model_name = 'fredzheng/Em-Garde-7B'
    
    llm_proposal = LLMProposer(model_name, device='cuda')
    # start_time = time.time()
    video_timestamps, video_frames = read_video_decord(video_path, current_timestamp - history_length, current_timestamp, fps, max_pixels = 280*280)
    start_time1 = time.time()
    proposals = llm_proposal.propose(video_frames, query)
    end_time1 = time.time()
    
    
    
    
    
    
    
    
    