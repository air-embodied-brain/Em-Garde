import json
import tqdm
from src.model_args import ModelArguments
from src.model import StreamingModel
from src.embedding_comparison import load_detector_config
from src.video_utils import ffmpeg_once
import torch
import re
import os
from pathlib import Path
import argparse
from .plot_similarity import plot


def prepare_videos(video_dir, fps):
    video_root = os.path.dirname(video_dir)
    ffmpeg_path = os.path.join(video_root, f"ffmpeg_videos_{fps}fps")
    os.makedirs(ffmpeg_path, exist_ok=True)
    for path in Path(video_dir).rglob("*"):
        if path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            relative_path = path.relative_to(video_dir)
            output_path = os.path.join(ffmpeg_path, relative_path)
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            ffmpeg_once(
                src_path=str(path),
                dst_path=output_path,
                fps=fps
            )                 

def eval_one_question(
    model,
    model_args,
    subset,
    qid,
    video_root            
):

    video = subset["video"]

    ffmpeg_path = os.path.dirname(video_root) + f"/ffmpeg_videos_{model_args.streaming_fps}fps/" + video

    model.init_video(ffmpeg_path)
    
    task = subset['task']
    
    # for REC and SSR, we start from 0. for CRR, we start from ask_time
    start_time = 0 if task in ['REC','SSR'] else subset['ask_time']
    end_time = subset["test_info"][-1]["realtime"] # stop at the last testpoint
    
    if task=='REC':
        question = "How many times does the event: {} happen?".format(subset['activity'])
    elif task=='SSR':
        question = "Describe the steps of {}".format(re.sub(r'(?<!^)(?=[A-Z])', ' ', subset["tutorial"]).lower())
    else:
        question = subset["question"]

    proposals = model.process_query(
        start_time,
        model_args.history_length,
        model_args.history_fps,
        question,
    )
    
    if task=='CRR':
        gt_timestamps = [subset['clue_time']]
    elif task=='REC':
        gt_timestamps = [t+2 for t in subset["start_times"]]
    else:
        gt_timestamps = [t+2 for t in subset["start_time"]]
    # gt_timestamps = [t+2 for t in subset["start_time"]] if task in ['REC','SSR'] else [subset["clue_time"]]

    res = {
        "id": qid,
        "task": task,
        "video_path": ffmpeg_path,
        "question": question,
        "proposals": proposals,
        "similarities": [],
        "detections": [],
        "checked_detections": [],
        "gt_timestamp": gt_timestamps,
    }

    curr_time = start_time + 1

    while curr_time <= end_time:
        if task in ['REC','SSR','CRR']:
            if abs(round(curr_time) - curr_time) < 1e-4 and int(round(curr_time)) % model_args.proposal_update_duration == 0:         # for REC and SSR, update proposals every 30s
                proposals = model.update_query(
                    curr_time,
                    model_args.history_length,
                    model_args.history_fps,
                    question,
                )
        ret = model.process_streaming_video(
            curr_time, question=None
        )
        if ret is None:
            break

        triggered_queries, similarities, processing_time = ret
        similarities = torch.round(similarities[0] * 1000) / 1000.0
        similarities = similarities.cpu().tolist()

        res["similarities"].append({
            "curr_time": curr_time,
            "similarities": similarities,
            "processing_time": processing_time,
        })

        if triggered_queries:
            res["detections"].append(curr_time)
        
        if model_args.should_respond and triggered_queries:
            if triggered_queries[0]["response"]:
                res["checked_detections"].append({
                    "time": curr_time,
                    "response": triggered_queries[0]["response"],
                })

        curr_time += 1 / model_args.streaming_fps

    return res

def gpu_worker(
    gpu_id,
    task_queue,
    result_queue,
    model_args,
):
    import os

    import torch
    torch.cuda.set_device(gpu_id)

    model = StreamingModel(
        model_args,
        device="cuda",
    )

    while True:
        task = task_queue.get()
        if task is None:
            break

        subset, qid, video_root = task
        # try:
        res = eval_one_question(
            model, model_args, subset, qid, video_root
        )
        result_queue.put(res)
        # except Exception as e:
        #     result_queue.put({
        #         "id": qid,
        #         "error": str(e),
        #     })    
            
def eval_multi_gpu(
    model_args,
    data_path,
    video_root,
    output_path,
    plot_path=None,
    gpu_ids=(0, 1, 2, 3),
):
    import json
    import multiprocessing as mp
    import tqdm
    
    prepare_videos(video_root, model_args.streaming_fps)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(data_path, "r") as f:
        data = json.load(f)
        
    data = [subset for subset in data if subset['task'] in ['SSR','REC','CRR']]
    
    tasks = []
    qid = 0
    for subset in data:
        tasks.append((subset, qid, video_root))
        qid += 1

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for t in tasks:
        task_queue.put(t)

    for _ in gpu_ids:
        task_queue.put(None)

    workers = []
    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                model_args
            ),
        )
        p.start()
        workers.append(p)

    results = []
    with open(output_path, "w") as f_out:
        for _ in tqdm.tqdm(range(len(tasks))):
            res = result_queue.get()
            results.append(res)
            f_out.write(json.dumps(res) + "\n")
        
    recall_CRR = []
    precision_CRR = []
    recall_REC = []
    precisions_REC = []
    recall_SSR =  []
    precisions_SSR = []
    
    for r in results:
        if "error" in r:
            print(f"Error in question {r['id']}: {r['error']}")
            continue
        det = r["detections"] if not model_args.should_respond else [d["time"] for d in r["checked_detections"]]
        det_copy = det.copy()
        gt_times = r["gt_timestamp"]
        correct = 0
        for gt in gt_times:
            for i, t in enumerate(det_copy):
                if abs(t-gt) <=2:
                    correct += 1
                    det_copy.remove(t)
                    break
        recall = correct / len(gt_times)
        precision = correct / len(det) if len(det) > 0 else 0.0
        
        if r["task"] == 'CRR':
            recall_CRR.append(recall)
            precision_CRR.append(precision)
        elif r["task"] == 'REC':
            recall_REC.append(recall)
            precisions_REC.append(precision)
        elif r["task"] == 'SSR':
            recall_SSR.append(recall)
            precisions_SSR.append(precision)
    recall_rate_CRR = sum(recall_CRR) / len(recall_CRR) if len(recall_CRR) > 0 else 0.0
    precision_rate_CRR = sum(precision_CRR) / len(precision_CRR) if len(precision_CRR) > 0 else 0.0
    recall_rate_REC = sum(recall_REC) / len(recall_REC) if len(recall_REC) > 0 else 0.0
    precision_rate_REC = sum(precisions_REC) / len(precisions_REC) if len(precisions_REC) > 0 else 0.0
    recall_rate_SSR = sum(recall_SSR) / len(recall_SSR) if len(recall_SSR) > 0 else 0.0
    precision_rate_SSR = sum(precisions_SSR) / len(precisions_SSR) if len(precisions_SSR) > 0 else 0.0
    
    F1_score_CRR = 2 * recall_rate_CRR * precision_rate_CRR / (recall_rate_CRR + precision_rate_CRR) if (recall_rate_CRR + precision_rate_CRR) > 0 else 0.0
    F1_score_SSR = 2 * recall_rate_SSR * precision_rate_SSR / (recall_rate_SSR + precision_rate_SSR) if (recall_rate_SSR + precision_rate_SSR) > 0 else 0.0
    F1_score_REC = 2 * recall_rate_REC * precision_rate_REC / (recall_rate_REC + precision_rate_REC) if (recall_rate_REC + precision_rate_REC) > 0 else 0.0
    
    Average_F1_score = (F1_score_CRR + F1_score_SSR + F1_score_REC) / 3
    
    print(f"CRR - Recall@2s: {recall_rate_CRR:.4f}, Precision@2s: {precision_rate_CRR:.4f}, F1@2s: {F1_score_CRR:.4f}")
    print(f"REC - Recall@2s: {recall_rate_REC:.4f}, Precision@2s: {precision_rate_REC:.4f}, F1@2s: {F1_score_REC:.4f}")
    print(f"SSR - Recall@2s: {recall_rate_SSR:.4f}, Precision@2s: {precision_rate_SSR:.4f}, F1@2s: {F1_score_SSR:.4f}")
    
    print(f"Average F1@2s: {Average_F1_score:.4f}")
            
    for p in workers:
        p.join()
    # if plot_path is not None:
        # plot(output_path, plot_path)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm-proposer", action='store_true', help="Whether to use LLM proposer for generating proposals")
    parser.add_argument("--data-path", type=str, default="/data/OVO-Bench/data/ovo_bench_new.json", help="Path to the dataset")
    parser.add_argument("--video-root", type=str, default="/data/OVO-Bench/data/src_videos", help="Root directory of source videos")
    parser.add_argument("--output-path", type=str, default="eval/ovo_bench/results/ovo_bench_results.jsonl", help="Path to save evaluation results")
    parser.add_argument("--plot-path", type=str, default="eval/ovo_bench/plots/plots_ovo_bench/", help="Path to save plots")
    parser.add_argument("--detector-type", type=str, default="opsmm_embedding_v1", help="Type of detector to use")
    parser.add_argument("--detector-config", type=str, default="configs/detector/ops_mm_v1_2B.yaml", help="Path to detector config file")
    parser.add_argument("--proposer-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--responder-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--streaming-fps", type=int, default=2, help="Streaming FPS for evaluation")
    parser.add_argument("--history-length", type=int, default=5, help="History length for proposal processing")
    parser.add_argument("--history-fps", type=int, default=1, help="History FPS for proposal processing")
    parser.add_argument("--should-respond", action='store_true', help="Whether to enable LLM response check and actual response")
    parser.add_argument("--detect-threshold", type=float, default=0.04, help="Threshold for detector triggering")
    parser.add_argument("--proposal-update-duration", type=int, default=9999, help="Duration (in seconds) to update proposals")
    parser.add_argument("--gpu-ids", type=str, default="4,5,6,7", help="Comma separated GPU IDs to use for evaluation")
    
    args = parser.parse_args()
    model_args = ModelArguments()
    model_args.detector_type = args.detector_type
    model_args.proposer_model_name = args.proposer_model_name
    model_args.responder_model_name = args.responder_model_name
    model_args.detector_config = load_detector_config(args.detector_type, args.detector_config)
    model_args.streaming_fps = args.streaming_fps
    model_args.history_length = args.history_length
    model_args.history_fps = args.history_fps
    model_args.should_respond = args.should_respond
    model_args.detect_threshold = args.detect_threshold
    model_args.proposal_update_duration = args.proposal_update_duration
    eval_multi_gpu(model_args, args.data_path, args.video_root, args.output_path, args.plot_path, gpu_ids=[int(x) for x in args.gpu_ids.split(",")])