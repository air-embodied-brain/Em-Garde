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

def prepare_videos(video_root, fps):
    original_path = os.path.join(video_root, "videos")
    ffmpeg_path = os.path.join(video_root, f"ffmpeg_videos_{fps}fps")
    os.makedirs(ffmpeg_path, exist_ok=True)
    for video_file in os.listdir(original_path):
        if video_file.endswith("proactive.mp4"):
            src_path = os.path.join(original_path, video_file)
            dst_path = os.path.join(ffmpeg_path, video_file)
            if not os.path.exists(dst_path):
                ffmpeg_once(src_path, dst_path, fps=fps)
    

def eval_one_question(
    model,
    model_args,
    subset,
    question,
    qid,
    use_llm_proposer=False,
):

    video_path = subset["video_path"]
    video_dir = os.path.dirname(video_path)
    ffmpeg_dir = video_dir.replace("videos", f"ffmpeg_videos_{model_args.streaming_fps}fps")
    video_path = os.path.join(ffmpeg_dir, os.path.basename(video_path))

    timestamp = question["time_stamp"]
    ground_truth_timestamp = question["ground_truth_time_stamp"]

    start_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))
    ground_truth_time = sum(int(x) * 60 ** i for i, x in enumerate(reversed(ground_truth_timestamp.split(":"))))
    max_time = ground_truth_time + 4

    model.init_video(video_path)

    if use_llm_proposer:
        proposals = model.process_query(
            start_time,
            model_args.history_length,
            model_args.history_fps,
            question["question"],
        )
    else:
        match = re.search(r'(?i)\bwhen\s+(.*?)(?=,)', question["question"])
        proposals = [{"positive": match.group(1).strip()}]
        model.encode_proposal(proposals)

    res = {
        "id": qid,
        "video_path": video_path,
        "question": question["question"],
        "proposals": proposals,
        "similarities": [],
        "detections": [],
        "checked_detections": [],
        "gt_timestamp": ground_truth_time,
    }

    curr_time = start_time + 1

    while curr_time <= max_time:
        # ret = model.process_video_segment(
        #     st, curr_time, model_args.streaming_fps, model_args.process_rate, question=None
        # )
        ret = model.process_streaming_video(curr_time, question=None)
        if ret is None:
            break

        triggered_queries, similarities, _ = ret
        # print(len(similarities))
        similarities = torch.round(similarities[0] * 1000) / 1000.0
        similarities = similarities.cpu().tolist()

        res["similarities"].append({
            "curr_time": curr_time,
            "similarities": similarities,
        })

        if triggered_queries:
            res["detections"].append(curr_time)
        
        if model_args.should_respond and len(triggered_queries) > 0:
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
    use_llm_proposer,
):
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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

        subset, question, qid = task
        try:
            res = eval_one_question(
                model, model_args, subset, question, qid, use_llm_proposer
            )
            result_queue.put(res)
        except Exception as e:
            result_queue.put({
                "id": qid,
                "error": str(e),
            })
            
            
def eval_multi_gpu(
    model_args,
    data_path,
    output_path,
    plot_path=None,
    use_llm_proposer=False,
    gpu_ids=(0, 1, 2, 3),
):
    import json
    import multiprocessing as mp
    import tqdm

    video_root = os.path.dirname(data_path)
    prepare_videos(video_root, model_args.streaming_fps)
    
    with open(data_path, "r") as f:
        data = json.load(f)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    # Optional subset
    # data = data[:10]

    # Flatten questions
    tasks = []
    qid = 0
    for subset in data:
        for question in subset["questions"]:
            tasks.append((subset, question, qid))
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
                model_args,
                use_llm_proposer,
            ),
        )
        p.start()
        workers.append(p)

    recalls_2 = recalls_1 = 0
    accurate_2 = accurate_1 = 0

    with open(output_path, "w") as f_out:
        for _ in tqdm.tqdm(range(len(tasks))):
            res = result_queue.get()
            f_out.write(json.dumps(res) + "\n")

            if "detections" not in res:
                continue

            gt = res["gt_timestamp"]
            rec_2 = rec_1 = False
            
            if model_args.should_respond:
                for i, det in enumerate(res["checked_detections"]):
                    t = det["time"]
                    if abs(t - gt) <= 2:
                        rec_2 = True
                        if i == 0:
                            accurate_2 += 1
                    if abs(t - gt) <= 1:
                        rec_1 = True
                        if i == 0:
                            accurate_1 += 1
            else:
                for i, t in enumerate(res["detections"]):
                    if abs(t - gt) <= 2:
                        rec_2 = True
                        if i == 0:
                            accurate_2 += 1
                    if abs(t - gt) <= 1:
                        rec_1 = True
                        if i == 0:
                            accurate_1 += 1

            recalls_2 += rec_2
            recalls_1 += rec_1

    for p in workers:
        p.join()

    total = len(tasks)
    print(
        f"Recall@2s: {recalls_2/total:.4f}, "
        f"Accuracy@2s: {accurate_2/total:.4f}, "
        f"Recall@1s: {recalls_1/total:.4f}, "
        f"Accuracy@1s: {accurate_1/total:.4f}"
    )
    if plot_path is not None:
        plot(output_path, plot_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm-proposer", action='store_true', help="Whether to use LLM proposer for generating proposals")
    parser.add_argument("--data-path", type=str, default="/data/StreamingBench/src/data/questions_proactive.json", help="Path to the dataset")
    parser.add_argument("--output-path", type=str, default="eval/streamingbench/results/streamingbench_results_opsmm1.jsonl", help="Path to save evaluation results")
    parser.add_argument("--plot-path", type=str, default="eval/streamingbench/plots/plots_opsmm/", help="Path to save plots")
    parser.add_argument("--detector-type", type=str, default="opsmm_embedding_v1", help="Type of detector to use")
    parser.add_argument("--detector-config", type=str, default="configs/detector/ops_mm_v1_2B.yaml", help="Path to detector config file")
    parser.add_argument("--proposer-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--responder-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--streaming-fps", type=int, default=2, help="Streaming FPS for evaluation")
    parser.add_argument("--history-length", type=int, default=5, help="History length for proposal processing")
    parser.add_argument("--history-fps", type=int, default=1, help="History FPS for proposal processing")
    parser.add_argument("--detect-threshold", type=float, default=0.025, help="Threshold for detector triggering")
    parser.add_argument("--should-respond", action='store_true', help="Whether to enable LLM response check and actual response")
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
    # model_args.process_rate = args.process_rate
    eval_multi_gpu(model_args, args.data_path, args.output_path, args.plot_path, use_llm_proposer = args.use_llm_proposer, gpu_ids=[int(x) for x in args.gpu_ids.split(",")])