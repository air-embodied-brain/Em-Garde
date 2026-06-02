"""
本demo用于在ARM架构上运行Em-Garde视频理解模型，使用以下替代方案：

1. 用FFmpeg代替decord：创建FFmpegVideoReader类，实现与decord.VideoReader兼容的API
   - 修改文件：src/video_utils.py, vlm2vec/data/eval_dataset/mvbench_dataset.py, vlm2vec/model/vlm_backbone/qwen2_vl/qwen_vl_utils.py
   - 实现方法：__init__, __len__, __getitem__, get_avg_fps, get_frame_timestamp, get_frame_index, get_batch
   - 注意：FFmpegVideoReader.get_batch() 直接返回 numpy array，无需调用 .asnumpy()
   - 修复：src/video_utils.py 中移除了 .asnumpy() 调用并添加了类型检查

2. 用OpenCV代替torchvision.io.write_video：创建自定义write_video函数
   - 修改文件：vlm2vec/data/utils/video_transforms.py, vlm2vec/data/utils/vision_utils.py, train/rl/data/data_process.py, src/model.py
   - 实现方法：使用cv2.VideoWriter写入视频，支持mp4(h264)和avi格式

3. 检测器模型本地加载：修改配置文件使用本地模型路径
   - 修改文件：configs/detector/ops_mm_v1_2B.yaml
   - 配置：将 model_name 从 "OpenSearch-AI/Ops-MM-embedding-v1-2B" 改为本地路径
   - 在本文件中，需要指定Em-Garde-7B的模型路径
   - 修改proposer-model-name和responder-model-name为本地路径
"""
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
    # 使用CPU设备（ARM架构可能没有CUDA）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = StreamingModel(model_args, device=device)
    
    model.init_video(video_path, start_time = max(0, start_time-model_args.history_length), end_time = end_time)
    curr_t = start_time
    interval = 1.0 / model_args.streaming_fps
    
    queries_in_progress = []
    
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
            for i, q in enumerate(queries_in_progress):
                similarities = torch.round(similarities[i] * 1000) / 1000.0
                similarities = similarities.cpu().tolist()
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
                print(f"At time {curr_t:.2f}s, updated proposals for query '{q['query']}': {proposals}")
                
        curr_t += interval
    
    for i, q in enumerate(queries_in_progress):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_path, f"{video_name}_query_{i}.json")
        with open(output_path, 'w') as f:
            json.dump(q, f, indent=4)
                
def plot(similarities, proposals, detections, output_path):
    
    seen = defaultdict(set)
    deduped = []
    for d in proposals:
        (k, v), = d.items()
        if v not in seen[k]:
            seen[k].add(v)
            deduped.append(d)
        proposals = deduped
    
    curves = []
    width = len(similarities) / 10
    plt.figure(figsize=(width, 6))
    x = [s["curr_time"] for s in similarities]
    fig_width = len(similarities) / 10
    if isinstance(similarities[0]["similarities"][0],float):
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
    
    for det_time in detections:
        plt.axvline(det_time, color="red", linestyle=":")
    
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.savefig(output_path, bbox_inches="tight")
 
def cuda_mem(tag):
    if torch.cuda.is_available():
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
    else:
        print(f"[{tag:>25}] CUDA not available")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Em-Garde ARM Demo - Video Understanding on ARM Architecture")
    
    parser.add_argument("--yaml-path", type=str, default="configs/demo/demo_ego4d.yaml", help="Path to the YAML file containing evaluation requests")
    parser.add_argument("--output-path", type=str, default="demo/", help="Path to save evaluation results")
    parser.add_argument("--plot-path", type=str, default="demo/", help="Path to save plots")
    parser.add_argument("--detector-type", type=str, default="opsmm_embedding_v1", help="Type of detector to use")
    parser.add_argument("--detector-config", type=str, default="configs/detector/ops_mm_v1_2B.yaml", help="Path to detector config file")
    parser.add_argument("--proposer-model-name", type=str, default="/home/aiot/mingjuwang/Models/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--responder-model-name", type=str, default="/home/aiot/mingjuwang/Models/Em-Garde-7B", help="LLM model name or path for response generation")
    parser.add_argument("--streaming-fps", type=int, default=5, help="Streaming FPS for evaluation")
    parser.add_argument("--history-length", type=int, default=5, help="History length for proposal processing")
    parser.add_argument("--history-fps", type=int, default=1, help="History FPS for proposal processing")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Em-Garde ARM Demo")
    print("=" * 60)
    print(f"Using FFmpeg for video reading (replaces decord)")
    print(f"Using OpenCV for video writing (replaces torchvision.io.write_video)")
    print("=" * 60)
    
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
