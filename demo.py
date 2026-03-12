import json
import tqdm
from src.model_args import ModelArguments
from src.model import StreamingModel
from src.embedding_comparison import load_detector_config
import torch
import re
import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import time

import yaml
from typing import List, TypedDict


class Query(TypedDict):
    text: str
    query_time: float
    response_time: List[float]


class VideoLLMRequest(TypedDict):
    video_path: str
    queries: List[Query]
    start_time: float
    end_time: float
    
def load_requests(yaml_path: str) -> VideoLLMRequest:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def run_streaming(video_path, output_path, plot_path, query_list, start_time=2, end_time=99999, model_args=None):
    model = StreamingModel(model_args, device='cuda')
    
    model.init_video(video_path, start_time = max(0, start_time-model_args.history_length), end_time = end_time)
    curr_t = start_time
    interval = 1.0 / model_args.streaming_fps
    
    queries_in_progress = []
    
    # efficiency_log_path = os.path.join(output_path, "efficiency_log.txt")
    
    while curr_t <= end_time:
        time1 = time.time()
        query = query_list[0] if len(query_list) > 0 else None
        if query and query['query_time'] <= curr_t:
            query_list.pop(0)
            proposals = model.process_query(curr_t, model_args.history_length, model_args.history_fps, query['text'])
            print(f"At time {curr_t:.2f}s, processed query: {query['text']}, given proposals: {proposals}")
            response_time = query.get('response_time', [])
            queries_in_progress.append({'query': query['text'],'start_time': curr_t, 'detections': [], 'ground_truth_times': response_time, "checked_detections":[]})
        res = model.process_streaming_video(curr_t, question=None)
        time2 = time.time()
        print(f"Processing timestep:{curr_t:.2f}s, Time taken for processing: {time2 - time1:.4f}s")
        if res is None:
            break
        else:
            triggered_queries, similarities, processing_time = res
            # with open(efficiency_log_path, 'a') as f:
            #     f.write(f"Time {curr_t:.2f}s: Processing time {processing_time:.4f}s\n")
            for i, q in enumerate(queries_in_progress):
                similarities = torch.round(similarities[i] * 1000) / 1000.0
                similarities = similarities.cpu().tolist()
                # q['similarities'].append({"curr_time": curr_t, "similarities": similarities})
                if len(triggered_queries)>0:
                    q['detections'].append(curr_t)
                    if triggered_queries[0]["response"]:
                        q["checked_detections"].append({
                            "time": curr_t,
                            "response": triggered_queries[0]["response"],
                        })
                    
        if abs(round(curr_t)-curr_t) < 1e-2 and int(round(curr_t-start_time)) % 30 == 0:
            for i, q in enumerate(queries_in_progress):
                proposals = model.update_query(curr_t, model_args.history_length, model_args.history_fps, q['query'])
                # q['proposals'] = proposals
                print(f"At time {curr_t:.2f}s, updated proposals for query '{q['query']}': {proposals}")
                
        # cuda_mem(f"Time {curr_t:.2f}s")
        curr_t += interval
    
    for i, q in enumerate(queries_in_progress):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_path, f"{video_name}_query_{i}.json")
        with open(output_path, 'w') as f:
            json.dump(q, f, indent=4)
        # plot_path = os.path.join(plot_path, f"{video_name}_query_{i}.png")
        # plot(q['similarities'], q['proposals'], q['detections'], plot_path)
        
    #     gt_response_times = q['ground_truth_times']
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # plot(q['similarities'], q['proposals'], gt_response_times, q['detections'], output_path)
                
                
def plot(similarities, proposals, detections, output_path):
    
    seen = defaultdict(set)
    deduped = []
    for d in proposals:
        (k, v), = d.items()   # assumes exactly one key per dict
        if v not in seen[k]:
            seen[k].add(v)
            deduped.append(d)
        proposals = deduped
    
    curves = []
    width = len(similarities) / 10
    plt.figure(figsize=(width, 6))
    x = [s["curr_time"] for s in similarities]
    fig_width = len(similarities) / 10
    if isinstance(similarities[0]["similarities"][0],float):       # vlm style detection
        y = [s["similarities"][0] for s in similarities]
    else:
        surges = []
        plt.figure(figsize=(fig_width, 6))
        chunk_size = 150 
        for i in range(0, len(similarities), chunk_size):
            offside = 5 if i !=0 else 0
            chunk_x = x[i-offside:i+chunk_size-5]
            num_proposals = len(similarities[i-offside]["similarities"][0])
            for idx, s in enumerate(similarities[i-offside:i+chunk_size-5]):
                if idx<5:
                    surge_max = 0
                else:
                    surge_max=0
                    for j in range(num_proposals):
                        surge = s["similarities"][0][j] - similarities[i-offside+idx-5]["similarities"][0][j]
                        if surge > surge_max:
                            surge_max = surge
                surges.append(surge_max)
        plt.plot(x, surges, linestyle="solid")
        plt.xlabel("Video Time (s)")
        plt.ylabel("Temporal surge signal")
    # for i in range(len(proposals)):
    #     proposal = proposals[i]
    #     sim = []
    #     for s in similarities:
    #         # print(s["similarities"])
    #         sim.append(s["similarities"][0][i])
    #     curves.append(sim)
    #     linestyle = "solid" if "positive" in proposal else "dashed"
    #     label = proposal.get("positive", proposal.get("negative", f"Proposal {i}"))
    #     plt.plot(x, sim, linestyle=linestyle, label=label)
        
    # for gt_time in gt_response_times:
    #     plt.axvline(gt_time, color="black", linestyle="--")
    
    for det_time in detections:
        plt.axvline(det_time, color="red", linestyle=":")
    
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.savefig(output_path, bbox_inches="tight")
 
def cuda_mem(tag):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"[{tag:>25}] "
        f"allocated={alloc:8.1f}MB | "
        f"reserved={reserv:8.1f}MB | "
        f"max_alloc={max_alloc:8.1f}MB"
    ) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--yaml-path", type=str, default="configs/demo/demo_ego4d.yaml", help="Path to the YAML file containing evaluation requests")
    parser.add_argument("--output-path", type=str, default="demo/", help="Path to save evaluation results")
    parser.add_argument("--plot-path", type=str, default="demo/", help="Path to save plots")
    parser.add_argument("--detector-type", type=str, default="opsmm_embedding_v1", help="Type of detector to use")
    parser.add_argument("--detector-config", type=str, default="configs/detector/ops_mm_v1_2B.yaml", help="Path to detector config file")
    parser.add_argument("--proposer-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--responder-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path for response generation")
    parser.add_argument("--streaming-fps", type=int, default=5, help="Streaming FPS for evaluation")
    parser.add_argument("--history-length", type=int, default=5, help="History length for proposal processing")
    parser.add_argument("--history-fps", type=int, default=1, help="History FPS for proposal processing")
    args = parser.parse_args()
    
    request = load_requests(args.yaml_path)
    
    model_args = ModelArguments()
    model_args.detector_type = args.detector_type
    model_args.proposer_model_name = args.proposer_model_name
    model_args.should_respond = True
    model_args.detector_config = load_detector_config(args.detector_type, args.detector_config)
    model_args.streaming_fps = args.streaming_fps
    model_args.history_length = args.history_length
    model_args.history_fps = args.history_fps
    model_args.responder_model_name = args.responder_model_name
    
    os.makedirs(args.output_path, exist_ok=True)
    
    run_streaming(request['video_path'], args.output_path, args.plot_path, request['queries'], request.get('start_time', 2), request.get('end_time', 99999), model_args) 