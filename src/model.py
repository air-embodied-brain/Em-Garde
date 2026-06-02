from .llm_proposer import LLMProposer
from .llm_responder import LLMResponder
from .embedding_comparison import build_detector
from .video_utils import read_video_decord, read_until_end, ffmpeg_once
from .proposal import Proposal
from .trigger_detection import trigger_detection

import torch
import torchvision
from typing import List

import time

class QueryInProgress():
    def __init__(self, query: str, proposal: Proposal, proposal_embeddings: torch.Tensor, history: List[torch.Tensor]):
        self.query = query
        self.proposal = proposal
        self.proposal_embeddings = proposal_embeddings
        self.history = history
        self.last_triggered_embedding = None
        self.last_triggered_time = None
        self.last_response = None

class StreamingModel:
    def __init__(self, model_args, device='cuda'):
        self.detector_type = model_args.detector_type
        self.detector = build_detector(model_args.detector_type, model_args.detector_config, device=device)
        self.proposer = LLMProposer(model_name=model_args.proposer_model_name, device=device)
        self.responder = LLMResponder(model_name=model_args.responder_model_name, device=device)
        self.responder_type = "qwen"
        self.video_tensor = None
        self.start_time = 0
        self.first_step = True
        
        self.fps = model_args.streaming_fps
        self.history_fps = model_args.history_fps
        self.video_embed_length = model_args.streaming_fps * model_args.streaming_length
        self.streaming_length = model_args.streaming_length
        
        self.queries = []
        self.respond = model_args.should_respond
        self.detect_threshold = model_args.detect_threshold
        self.video_cache_dir = "/data/Em_garde/video_cache"       
        
    def init_video(self, video_path, start_time=0, end_time=99999, delay_load=False):
        if delay_load:
            self.video_path = video_path  # for offline response
            self.first_step = True
            self.detector.reset()
            self.queries = []
        else:
            # video_name = video_path.split("/")[-1].split(".")[0]
            # ffmpeg_path = f"{self.video_cache_dir}/{video_name}_{self.fps}fps.mp4"
            # if not os.path.exists(ffmpeg_path):
            #     ffmpeg_once(video_path, ffmpeg_path, fps=self.fps)
            _, video_tensor = read_video_decord(video_path, start_time, end_time, self.fps, allow_end_truncation=True, max_pixels=280*280)
            self.video_tensor = video_tensor
            self.start_time = start_time
            
            self.first_step = True
            self.detector.reset()
            self.queries = []
    
    # def reset_video(self):
    #     self.video_tensor = None
    #     self.start_time = 0
    #     self.first_step = True
    #     self.detector.reset()
    #     self.queries = []
        
    def get_streaming_frames(self, start_timestamp, end_timestamp, fps):
        start_index = int(round((start_timestamp - self.start_time) * self.fps))
        end_index = int(round((end_timestamp - self.start_time) * self.fps))
        if start_index >= len(self.video_tensor) or end_index <= 0 or start_index > end_index:
            return None
        stride = max(1, int(self.fps / fps))       # assume that history fps can be divided by streaming fps
        if start_index == end_index:
            end_index +=1
        else:
            start_index += 1
            end_index += 1
        return self.video_tensor[start_index:end_index:stride]
        
    def get_video_frames(self, start_timestamp, end_timestamp, fps, max_pixels=280*280, responding=False):
        if end_timestamp == -1:
            return read_until_end(self.video_path, start_timestamp, fps, max_pixels=max_pixels)
        return read_video_decord(self.video_path, start_timestamp, end_timestamp, fps, max_pixels=max_pixels)
    
    def process_query(self, current_timestamp, history_length, fps, query):
        start_timestamp = max(0, current_timestamp - history_length)
        end_timestamp = current_timestamp
        video_frames = self.get_streaming_frames(start_timestamp, end_timestamp, fps)
        proposals = self.proposer.propose(video_frames, query)
        proposal = Proposal(proposals)
        if self.detector_type in ['vlm2vec','siglip','opsmm_embedding_v1']:
            proposal_embeddings = self.detector.encode_targets(proposal.get_literals())
        else:
            proposal_embeddings = None
        self.queries.append(QueryInProgress(query, proposal, proposal_embeddings, []))
        return proposals
    
    def update_query(self, current_timestamp, history_length, fps, query: str):
        start_timestamp = max(0, current_timestamp - history_length)
        end_timestamp = current_timestamp
        video_frames = self.get_streaming_frames(start_timestamp, end_timestamp, fps)
        proposals = self.proposer.propose(video_frames, query)
        proposal = Proposal(proposals)
        if self.detector_type in ['vlm2vec','siglip','opsmm_embedding_v1']:
            proposal_embeddings = self.detector.encode_targets(proposal.get_literals())
        else:
            proposal_embeddings = None
        for q in self.queries:
            if q.query == query:
                print("updating query proposals...")
                q.proposal = proposal
                q.proposal_embeddings = proposal_embeddings
                q.history = []
                break
        return proposals
    
    # ! temporal, only for testing !!!
    def encode_proposal(self, proposals):
        proposal = Proposal(proposals)
        if self.detector_type in ['vlm2vec','siglip','opsmm_embedding_v1']:
            proposal_embeddings = self.detector.encode_targets(proposal.get_literals())
        else:
            proposal_embeddings = None
        self.queries.append(QueryInProgress("DUMMY", proposal, proposal_embeddings, []))
    
    def compute_similarity(self, video_embeddings, proposal_embeddings):
        similarities = torch.matmul(video_embeddings, proposal_embeddings.T)
        return similarities
    
    def process_streaming_video(self, timestamp, question = None, force_respond=False):
        start_time = time.time()
        if self.first_step:
            new_frames = self.get_streaming_frames(max(0, timestamp - self.streaming_length), timestamp, self.fps)
            self.first_step = False
        else:
            new_frames = self.get_streaming_frames(timestamp, timestamp, self.fps)
            # new_frames = self.get_streaming_frames(max(0, timestamp - self.streaming_length), timestamp, self.fps)
        end_time = time.time()
        # print("video loading time:", end_time - start_time)
        if new_frames is None:
            return None
        start_time_2 = time.time()
        triggered_queries = []
        all_similarities = []
        if self.detector_type in ['opsmm_embedding_v1']:
            start_time = time.time()
            # print(new_frames.shape)
            video_embeddings = self.detector.encode_video(new_frames, question=question, frames_length = self.video_embed_length)
            end_time = time.time()
            # print("video encoding time:", end_time - start_time)
            for query in self.queries:
                similarities = self.compute_similarity(video_embeddings, query.proposal_embeddings)
                all_similarities.append(similarities)
                query.history.append(similarities)
                if len(query.history) > 600:
                    query.history = query.history[-500:]
                
                if self.evaluate_proposal(query, trigger_scheme="surge", surge_threshold=self.detect_threshold, fps=self.fps) or force_respond:
                    if not query.last_triggered_time or self.compute_similarity(query.last_triggered_embedding, video_embeddings).item() < 0.8 or timestamp - query.last_triggered_time > 5:             # avoid duplicate triggers
                        query.last_triggered_embedding = video_embeddings
                        query.last_triggered_time = timestamp
                        if self.respond:
                            answer = self.handle_response(query, timestamp+1)
                            if answer:
                                print(f"Response generated for query: {query.query} at time {timestamp}s: {answer}")
                                triggered_queries.append({"query":query.query, "response": answer})
                            else:
                                triggered_queries.append({"query":query.query, "response": None})
                        else:
                            print(f"Trigger detected for query: {query.query} at time {timestamp}s")
                            triggered_queries.append(query.query)
            end_time3 = time.time()
            processing_time = end_time3 - start_time_2
            # print("similarity processing time:", end_time3 - start_time3)
        else:
            raise NotImplementedError("Only embedding-based detectors are implemented in this version.")
            # for query in self.queries:
            #     detection = self.detector.detect(video_frames,query.proposal.get_positive_literals())
            #     all_similarities.append(torch.tensor([1.0]) if detection else torch.tensor([0.0]))
            #     query.history.append(torch.tensor([1.0]) if detection else torch.tensor([0.0]))
            #     trigger = trigger_detection(torch.cat(query.history, dim=0), trigger_scheme="binary")
            #     if trigger:
            #         if not query.last_triggered_time or self.compute_similarity(query.last_triggered_embedding, video_embeddings).item() < 0.9 or end_timestamp - query.last_triggered_time > 5:             # avoid duplicate triggers
            #             query.last_triggered_embedding = video_embeddings
            #             query.last_triggered_time = end_timestamp
            #             print(f"Trigger detected for query: {query.query} at time {end_timestamp}s")
            #             triggered_queries.append(query.query)
                        
        end_time = time.time()
        # print("total processing time for segment:", end_time - start_time) 
        # print("video loading time:", start_time_2 - start_time)               
        return triggered_queries, all_similarities, processing_time
    
    def handle_response(self, query: QueryInProgress, timestamp):
        answer_history = 3
        answer_fps = 2
        start_timestamp = max(0, timestamp - answer_history)
        end_timestamp = timestamp
        # video_frames = self.get_video_frames(start_timestamp, end_timestamp, answer_fps, responding=True)
        video_frames = self.get_streaming_frames(start_timestamp, end_timestamp, answer_fps)
        # if self.responder_type != "streamforest":
        #     video_timestamps, video_frames = video_frames
        # fallback_check = self.responder.response_check(video_frames, query.query, query.last_response)
        # if fallback_check:
        #     print("Fallback check passed. responding...")
        #     answer = self.responder.respond(video_frames, query.query)
        #     print("Generated answer:", answer)
        #     query.last_response = answer
        #     return answer    
        # else:
        #     return None 
        # save_video(video_frames, f"debug_{query.query}_{timestamp}.mp4", fps=answer_fps)   
        response = self.responder.respond_and_check(video_frames, query.query, query.last_response)
        if response is not None:
            # print("Generated answer:", response)
            query.last_response = response
        return response

    def process_frame(self, timestamp, question=None):
        video_timestamps, video_frames = self.get_video_frames(timestamp, timestamp + 1 / 30, fps=30)
        if self.detector_type in ['vlm2vec','siglip','opsmm_embedding_v1']:
            frame_embeddings = self.detector.encode_image(video_frames[0], question=question)
            triggered_queries = []
            all_similarities = []
            for query in self.queries:
                similarities = self.compute_similarity(frame_embeddings, query.proposal_embeddings)
                all_similarities.append(similarities)
                query.history.append(similarities)
                if len(query.history) > 600:
                    query.history = query.history[-500:]
                
                if self.evaluate_proposal(query):
                    print(f"Trigger detected for query: {query.query} at time {timestamp}s")
                    triggered_queries.append(query.query)
            
        
    
    def evaluate_proposal(self, query: QueryInProgress, trigger_scheme="surge", **kwargs):
        similarity_history = torch.cat(query.history, dim=0)
        triggers = trigger_detection(similarity_history, trigger_scheme=trigger_scheme, **kwargs)
        triggers.cpu().tolist()
        trigger_dict = {k: v for k, v in zip(query.proposal.get_literals(), triggers)}
        return query.proposal.evaluate(trigger_dict)
    
    # def offline_respond(self, query_for_proposal, query_for_response, proposal_history_length, respond_history_length, proposal_fps, response_fps, proposal_time, response_time):
    #     # Propose
    #     start_timestamp = proposal_history_length if proposal_time == -1 else max(0, proposal_time - proposal_history_length)
    #     end_timestamp = proposal_time
    #     video_timestamps, video_frames = self.get_video_frames(start_timestamp, end_timestamp, proposal_fps)
    #     proposals = self.proposer.propose(video_frames, query_for_proposal)
    #     proposal = Proposal(proposals)
    #     proposal = proposal.get_positive_literals()
        
    #     # Respond
    #     start_timestamp = respond_history_length if response_time == -1 else max(0, response_time - respond_history_length)
    #     end_timestamp = response_time
    #     video_timestamps, video_frames = self.get_video_frames(start_timestamp, end_timestamp, response_fps)
    #     answer = self.responder.respond_offline(video_frames, query_for_response, proposals=proposal)
    #     return proposal,answer
    
    # def respond_with_context(self, response_time, response_fps, respond_history_length, context_time, query_for_response):
        # start_timestamp = respond_history_length if response_time == -1 else max(0, response_time - respond_history_length)
        # end_timestamp = response_time
        # video_timestamps, video_frames = self.get_video_frames(start_timestamp, end_timestamp, response_fps)
        
        # # Get context frames
        # context_frames = []
        # for (st, et) in context_time:
        #     _, frames = self.get_video_frames(st, et, response_fps)
        #     if frames is not None:
        #         context_frames.append(frames)
        
        # answer = self.responder.respond_with_context(video_frames, context_frames, query_for_response)
        # return answer
        
    def offline_respond(self, query: str, history_length: int, fps: int, response_time: int):
        if response_time == -1:
            start_timestamp = history_length
        else:
            start_timestamp = max(0, response_time - history_length)
        _, video_frames = self.get_video_frames(start_timestamp, response_time, fps, max_pixels=560*560, responding=True)
        answer = self.responder.respond_offline(video_frames, query)
        return answer


def save_video(video_tensor, path, fps=2):
    """
    video_tensor: [T, C, H, W], float in [0,1] or uint8 in [0,255]
    """
    # Move to CPU
    video_tensor = video_tensor.detach().cpu()

    # Convert to uint8 if needed
    if video_tensor.dtype != torch.uint8:
        video_tensor = (video_tensor).clamp(0, 255).to(torch.uint8)

    # Convert from [T, C, H, W] → [T, H, W, C]
    video_tensor = video_tensor.permute(0, 2, 3, 1)

    import cv2
    video_np = video_tensor.cpu().numpy()
    T, H, W, C = video_np.shape
    
    if C == 3:
        video_np = cv2.cvtColor(video_np.reshape(-1, H, W, C), cv2.COLOR_RGB2BGR).reshape(T, H, W, C)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for i in range(T):
        out.write(video_np[i])
    out.release()