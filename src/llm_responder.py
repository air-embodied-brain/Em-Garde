from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
import json
import re
import time

class LLMResponder:
    def __init__(self, model_name: str, device: str = 'cuda'):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = "flash_attention_2"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
        self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
    
    @torch.no_grad()
    def response_check(self, video_frames, question:str, last_response: str = None) -> str:
        instruction = '''You are an AI assistant specialized in streaming video understanding. Above is a short video segment representing the recent history of a streaming video. Next you will see a question, and your last response to the question. Based on this video segment, please decide whether it is appropriate to provide an answer to the user's question at this moment.
        follow these rules:
        1. First focus on the last 10 frames. Extract important information and events. Then refer to the longer context for relevant information if needed.
        2. When responding, first check if the question directly tells when to answer. If so, follow its instruction. Answer "YES" if the current event meets the condition, otherwise answer "NO".
        3. If the question does not specify when to answer, analyze whether the recent video segment contains sufficient information to answer the question accurately. If it does, answer "YES". If not, answer "NO".
        4. If you have already answered before (i.e., last response is not empty), consider whether the new information in the video segment allows you to provide a new or updated answer. If it does, answer "YES". If not, answer "NO".
        NOTE: In this task, you are a second-stage verifier. The timestamps are already filtered and should somehow relate to the question. Just check for wrong relationships, insufficient information, or dupluicate responses. If no such issues are found, please prefer answering "YES".
        Only answer "YES" or "NO". Do not provide any additional explanation.'''
        if last_response is None:
            last_response = ""
        prompt = f"{instruction}\nQuestion: {question}\nLast Response: {last_response}\n Your Answer (YES or NO):"
        
        messages = [
            {"role": "user", 
             "content": 
                [
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
        response = self.model.generate(**inputs, max_new_tokens=1,)
        # response = self.model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, num_return_sequences=5)
        end_time = time.time()
        # print("Generation time:", end_time - start_time)
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[:, input_len:]
        
        outputs = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        print("Response check output:", outputs[0])
        
        if outputs[0].strip().upper() == "YES":
            return True
        else:
            return False
    
    @torch.no_grad()    
    def respond(self, video_frames, question:str, last_response: str = None) -> str:
        instruction = '''You are an AI assistant specialized in streaming video understanding. You're doing a dialogue with the user about a streaming video. Above is a short video segment representing the recent history of the video. Next you will see a question. The recent history of the video should provide sufficient information to answer the question. Please give a concise and accurate answer. If you have answered the question before (i.e., last response is not empty), please incorporate the new information from the video segment to provide a new or updated answer. Please output your answer directly without any additional explanation.
        Remember: you are doing a dialogue and may answer in multiple turns. You don't need to provide a complete answer in one turn. Just answer based on the current content. For example, if you are asked to describe the steps of a tutorial, just answer the current step shown in the video segment.'''
        if last_response is None:
            last_response = ""
        prompt = f"{instruction}\nQuestion: {question}\nLast Response: {last_response}\n Your Answer:"
        
        messages = [
            {"role": "user", 
             "content": 
                [
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
        response = self.model.generate(**inputs, max_new_tokens=30)
        end_time = time.time()
        # print("Generation time:", end_time - start_time)
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[:, input_len:]
        
        outputs = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return outputs[0]
    
    @ torch.no_grad()
    def respond_and_check(self, video_frames, question:str, last_response: str = None) -> Optional[str]:
        instruction = '''You are an AI assistant specialized in streaming video question answering, not video description.

        You are participating in a multi-turn dialogue about a streaming video.
        The video segment above shows only the most recent moments of a longer video.

        A question will follow.
        Your task is to output only the answer to the question, not a description of the video.

        Follow these rules strictly:

        1. Answer-first rule:
        Your output must be a direct answer to the question (e.g., a step, decision, or conclusion).

        2. Last-moment grounding:
        Base your answer primarily on the last few frames of the video.
        Use earlier context only if necessary to interpret what is happening now.

        3. Partial answering is expected:
        This is a streaming, multi-turn setting.
        If the full answer is not yet available, output only the part that can be concluded from the current video segment.

        4. Abstraction over narration:
        When the question asks for steps, actions, or decisions, summarize the event into an abstract step.
        Do NOT narrate or restate visual details.
        Do interpret what the action means in the context of the question.

        5. No-evidence rule:
        If the current video segment does not provide enough evidence to answer any part of the question, output exactly:

        Insufficient information.

        6. Forbidden behaviors:

        No scene descriptions

        No speculation or common sense

        No explanations or justifications

        Output format:
        One concise sentence.
        Output only the final answer.'''
        # instruction = '''You are an AI assistant specialized in streaming video question answering.

        # You are participating in a multi-turn dialogue about a streaming video.
        # The video segment above shows only the most recent moments of a longer video.

        # A question will follow.
        # Your task is to output only the answer to the question.
        
        # Note these rules:
        # 1. The information needed to answer the question is supposed to appear in the last few frames of the video segment.
        # 2. Only answer based on the video segment. It's possible that only a partial answer can be given, in that case, give a partial answer (e.g. a step for a question asking for steps). Don't make up any information that is not in the video. Don't speculate or use common sense.
        # 3. Answer the question directly with a concise sentence. Don't include any explanations or justifications.
        # 4. If you did not see a target action or event directly to the query, or you cannot recognize the key information needed to answer the query, say "Insufficient information."

        # Output only the final answer.'''
        if last_response is None:
            last_response = ""
        prompt = f"{instruction}\nQuestion: {question}\n Your answer to the Question:"
        messages = [
            {"role": "user", 
             "content": 
                [
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
        response = self.model.generate(**inputs, max_new_tokens=30)
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[:, input_len:]
        
        outputs = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        
        # check_instruction = '''You'll see a question and two rounds of response to the question based on a video. Your task is to decide whether the second answer is appropriate.
        # If the second response directly addresses the question, and is different from the first response, answer "YES". If it is vague, meaningless, irrelevant to the question, or convey mostly the same meaning as the first response, answer "NO".
        # Note these rules:
        # 1. The rounds are part of a multi-turn dialogue. Incomplete answers are acceptable if they address the question.
        # 2. "conveys mostly the same meaning" includes different wordings of the same concept, slightly different mentioning of unimportant details, or minor elaborations. The key point is whether the core information provided in the second response is new compared to the first response.
        # Only answer "YES" or "NO". Do not provide any additional explanation.'''
        
        check_instruction = '''You are an automatic evaluator for streaming video question answering.

        You will be given:

        A question

        A first response (Might be None)

        A second response

        The video is not shown in this round.
        Do not infer or assume any visual content.

        Your task is to decide whether the second response is appropriate.

        Strictly follow the rules:

        Answer "YES" only if both conditions are met:

        The second response directly answers the question.

        The second response provides new core information compared to the first response.

        Otherwise, answer "NO".

        Clarifications:

        “New core information” means a new step, action, decision, outcome, or fact, not a rephrasing.

        The following are considered NOT new information:

        Paraphrases or rewordings

        Added adjectives or minor details

        More specific wording of the same idea

        Restating the same step with different phrasing

        Forbidden considerations:

        Do NOT judge correctness or factual accuracy.

        Do NOT judge style, fluency, or verbosity.

        Output format:

        Answer only one token:

        YES or NO'''
        check_prompt = f"{check_instruction}\nQuestion: {question}\nFirst Response: {last_response}\nSecond Response: {outputs}\n Your Answer (YES or NO):"
        check_messages = [
            {"role": "user", 
             "content": 
                [
                    {"type": "text", "text": check_prompt}
                ]
            }
        ]
        check_message = self.processor.tokenizer.apply_chat_template(check_messages, tokenize=False, add_generation_prompt=True)
        check_inputs = self.processor(text=[check_message], return_tensors="pt", padding=True)
        for k, v in check_inputs.items():
            if isinstance(v, torch.Tensor):
                check_inputs[k] = v.to(self.device)
        check_response = self.model.generate(**check_inputs, max_new_tokens=1,)
        new_tokens = check_response[:, check_inputs["input_ids"].shape[1]:]
        check_outputs = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        print("answer:", outputs, " check output:", check_outputs[0])
        if check_outputs[0].strip().upper() == "YES":
            return outputs
        else:
            return None
          
    def respond_with_context(self, video_frames, context_frames, question:str) -> str:
        instruction = "You are an AI assistant specialized in video understanding. Next you'll see a video segment, several short segments of context video, and a question. Please answer the question based on the video segment and the context segments. First look at the main video segment. If you cannot find the answer, refer to the context segments to help you answer the question."
        messages = [
            {"role": "user",
             "content":[
                    {"type":"text", "text": instruction},  # Placeholder for video input
                ]
            }
        ]
        context_frame_flattened = []
        for i, ctx in enumerate(context_frames):
            messages[0]["content"].append({"type":"text", "text": f"Context Segment {i+1}:"})
            messages[0]["content"].extend([{"type":"image"}] * len(ctx))
            context_frame_flattened.extend(ctx)
            
        messages[0]["content"].append({"type":"text", "text": f"Main Video Segment:"})
        messages[0]["content"].append({"type":"video", "video": "dummy_video"})
        
        question_part = {"type":"text", "text": f"Question: {question}\n Your Answer:"}
        messages[0]["content"].append(question_part)
        input_message = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if context_frame_flattened:
            inputs = self.processor(text=[input_message], videos=[video_frames], images=context_frame_flattened, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=[input_message], videos=[video_frames], return_tensors="pt", padding=True)
        inputs['second_per_grid_ts'] = inputs['second_per_grid_ts'].tolist()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        
        start_time = time.time()
        response = self.model.generate(**inputs, max_new_tokens=30)
        end_time = time.time()
        # print("Generation time:", end_time - start_time)
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[:, input_len:]
        
        outputs = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return outputs[0]
        
        
        
        
    def respond_offline(self, video_frames, question:str) -> str:
        # instruction = '''You are an AI assistant specialized in video understanding. Above is a short video segment. Next you will see a question and a list of proposals. Please answer the question following its instructions based on the video segment. The list of proposals include visual clues that might help you locate the relevant information and generate the answer. If you see some content related to the proposals, you can pay more attention to them.'''
        # proposals_str = "\n".join([f"{i+1}. {p}" for i, p in enumerate(proposals)])
        # prompt = f"{instruction}\nQuestion: {question}\nProposals:\n{proposals_str}\n Your Answer:"
        
        messages = [
            {"role": "user", 
             "content": 
                [
                    {"type": "video", "video": "dummy_video"},  # Placeholder for video input
                    {"type": "text", "text": question}
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
        response = self.model.generate(**inputs, max_new_tokens=30)
        end_time = time.time()
        # print("Generation time:", end_time - start_time)
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = response[:, input_len:]
        
        outputs = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return outputs[0]

# from StreamForest.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from StreamForest.llava.conversation import conv_templates, SeparatorStyle
# from StreamForest.llava.model.builder import load_pretrained_model
# from StreamForest.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    
# class StreamForestResponder:
#     def __init__(self, model_path: str, device: str = 'cuda'):
        
#         model_name = get_model_name_from_path(model_path)
        
#         # cfg_pretrained = AutoConfig.from_pretrained(model_path)
#         current_device = torch.cuda.current_device()
#         device_map = {"": current_device}
#         tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, multimodal=True, attn_implementation="flash_attention_2", device_map=device_map)
#         model.to(torch.bfloat16)
#         self.model = model
#         self.model.eval()
#         self.tokenizer = tokenizer
#         self.image_processor = image_processor
#         self.context_len = context_len
        
#     def build_input(self, video_frames, question):
#         if video_frames is None:
#             conv = conv_templates["qwen_2"].copy()
#             conv.append_message(conv.roles[0], question.strip())
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()
#             input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
#             if self.tokenizer.pad_token_id is None:
#                 if "qwen" in self.tokenizer.name_or_path.lower():
#                     print("Setting pad token to bos token for qwen model.")
#                     self.tokenizer.pad_token_id = 151643
#             attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
#             stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            
#             return input_ids, attention_masks, None, None, stop_str
#         video_frames, time_msg = video_frames
#         # len_sec = len(video_frames)
#         qs = f'{time_msg.strip()}\n{question.strip()}'
#         if self.model.config.mm_use_im_start_end:
#             qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
#         conv = conv_templates["qwen_2"].copy()
#         conv.append_message(conv.roles[0], qs)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()
#         input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
#         if self.tokenizer.pad_token_id is None:
#             if "qwen" in self.tokenizer.name_or_path.lower():
#                 print("Setting pad token to bos token for qwen model.")
#                 self.tokenizer.pad_token_id = 151643
                
#         attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

#         stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#         # keywords = [stop_str]
#         # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
#         frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(torch.bfloat16).cuda()
#         video = [frames]
#         image_sizes = [frames[0].shape[:2]]
        
#         return input_ids, attention_masks, video, image_sizes, stop_str
        
     
#     @ torch.no_grad()    
#     def respond_and_check(self, video_frames, question:str, last_response: str = None) -> Optional[str]:
#         instruction = '''You are an AI assistant specialized in streaming video question answering.

#         You are participating in a multi-turn dialogue about a streaming video.

#         A question will follow.
#         Your task is to output only the answer to the question, not a description of the video.

#         Follow these rules strictly:

#         1. Answer-first rule:
#         Your output must be a direct answer to the question (e.g., a step, decision, or conclusion).

#         2. Last-moment grounding:
#         Base your answer primarily on the last few frames of the video.
#         Use earlier context only if necessary to interpret what is happening now.

#         3. Partial answering is expected:
#         This is a streaming, multi-turn setting.
#         If the full answer is not yet available, output only the part that can be concluded from the current video segment.

#         4. Abstraction over narration:
#         When the question asks for steps, actions, or decisions, summarize the event into an abstract step.
#         Do NOT narrate or restate visual details.
#         Do interpret what the action means in the context of the question.

#         5. No-evidence rule:
#         If the current video segment does not provide enough evidence to give any answer the question, output exactly:

#         Insufficient information.

#         6. Forbidden behaviors:

#         No scene descriptions

#         No speculation or common sense

#         No explanations or justifications

#         Output format:
#         One concise sentence.
#         Output only the final answer.'''
        
#         qs = instruction + f"\nQuestion: {question}\n Your Answer to the question:"
        
#         input_ids, attention_masks, video, image_sizes, stop_str = self.build_input(video_frames, qs)
        
#         with torch.inference_mode():
#             output_ids = self.model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities=["video"], do_sample=False, temperature=0.0, max_new_tokens=30, top_p=0.1, num_beams=1, use_cache=True)
        
#         outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
#         if outputs.endswith(stop_str):
#             outputs = outputs[: -len(stop_str)]
            
#         outputs = outputs.strip()
        
#         check_instruction = '''You are an automatic evaluator for streaming video question answering.

#         You will be given:

#         A question

#         A first response (Might be None)

#         A second response

#         The video is not shown in this round.
#         Do not infer or assume any visual content.

#         Your task is to decide whether the second response is appropriate.

#         Strictly follow the rules:

#         Answer "YES" only if both conditions are met:

#         The second response directly addresses the question (partial answer is ok, since this is a multi-turn dialogue).

#         The second response provides new core information compared to the first response.

#         Otherwise, answer "NO".

#         Clarifications:

#         Incomplete or partial answers are acceptable if they add new information relevant to the question.

#         “New core information” means a new step, action, decision, outcome, or fact, not a rephrasing.

#         The following are considered NOT new information:

#         Paraphrases or rewordings

#         Added adjectives or minor details

#         More specific wording of the same idea

#         Restating the same step with different phrasing

#         Forbidden considerations:

#         Do NOT judge correctness or factual accuracy.

#         Do NOT judge style, fluency, or verbosity.

#         Do NOT compare answers to what should be in the video.

#         Output format:

#         Answer only one token:

#         YES or NO'''
        
#         check_prompt = f"{check_instruction}\nQuestion: {question}\nFirst Response: {last_response}\nSecond Response: {outputs}\n Your Answer (YES or NO):"
#         input_ids, attention_masks, _, _, stop_str = self.build_input(None, check_prompt)
        
#         with torch.inference_mode():
#             check_response_ids = self.model.generate(inputs=input_ids, attention_mask=attention_masks, do_sample=False, temperature=0.0, max_new_tokens=1, top_p=0.1, num_beams=1, use_cache=True)
        
#         check_outputs = self.tokenizer.batch_decode(check_response_ids, skip_special_tokens=True)[0].strip()
#         print("answer:", outputs, " check output:", check_outputs)
#         if check_outputs[0].strip().upper() == "YES":
#             return outputs
#         else:
#             return None
        
#     def respond_offline(self, video_frames, question) -> str:
#         input_ids, attention_masks, video, image_sizes, stop_str = self.build_input(video_frames, question)
#         with torch.inference_mode():
#             output_ids = self.model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities=["video"], do_sample=False, temperature=0.0, max_new_tokens=30, top_p=0.1, num_beams=1, use_cache=True)
#         outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#         if outputs.endswith(stop_str):
#             outputs = outputs[: -len(stop_str)]
        
#         return outputs.strip()
        
        