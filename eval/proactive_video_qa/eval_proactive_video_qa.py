import json
import tqdm
from src.model_args import ModelArguments
from src.model import StreamingModel
from src.embedding_comparison import load_detector_config
from src.video_utils import ffmpeg_once
import torch
import re
import os
import argparse

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
    print("Video preparation completed.")

def eval_one_question(
    model,
    model_args,
    video_path,
    question,
    question_time,
    qid,
    end_time = 99999,
):

    start_time = max(question_time, 2)

    model.init_video(video_path)

    proposals = model.process_query(
        start_time,
        model_args.history_length,
        model_args.history_fps,
        question,
    )

    res = {
        "id": qid,
        "video_path": video_path,
        "question": question,
        "proposals": proposals,
        "similarities": [],
        "detections": [],
        "checked_detections": [],
    }
    res_for_eval = {
        "question_id": qid,
        "model_response_list": []
    }                                    # For PAUC evaluation
    

    curr_time = start_time

    while curr_time <= end_time:
        if abs(int(curr_time) - curr_time) < 1e-4 and int(curr_time) % 30 == 0:         # update proposals every 30s
            proposals = model.update_query(
                curr_time,
                model_args.history_length,
                model_args.history_fps,
                question,
            )
            
        st = max(0, curr_time - 2)
        force_respond = True if curr_time==2 else False
        ret = model.process_streaming_video(curr_time,question=None, force_respond=force_respond)
        if ret is None:
            break

        triggered_queries, similarities, _ = ret
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
                res_for_eval["model_response_list"].append({
                    "time": curr_time,
                    "content": triggered_queries[0]["response"],
                })

        curr_time += 1 / model_args.process_rate

    return res, res_for_eval

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

        video_path, question, question_time, qid, end_time = task
        try:
            res, res_for_eval = eval_one_question(
                model, model_args, video_path, question, question_time, qid, end_time
            )
            result_queue.put(res_for_eval)
        except Exception as e:
            result_queue.put({
                "id": qid,
                "error": str(e),
            })
            
            
def eval_multi_gpu(
    model_args,
    data_path,
    output_path,
    video_root,
    plot_path=None,
    use_llm_proposer=False,
    gpu_ids=(0, 1, 2, 3),
):
    import json
    import multiprocessing as mp
    import tqdm

    with open(data_path, "r") as f:
        data = json.load(f)
        
    prepare_videos(video_root, model_args.streaming_fps)

    # Optional subset
    # data = data[:10]

    tasks = []
    for item in data:
        qid = item["question_id"]
        video_name = item["video"]
        video_path = os.path.join(video_root, f"ffmpeg_videos_{model_args.streaming_fps}fps", video_name)
        question = item["conversation"][0]["content"]
        question_time = item["conversation"][0]["time"]
        end_time =item["duration"]
        task = (video_path, question, question_time, qid, end_time)
        tasks.append(task)

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
        
    results = []
    with open(output_path, "w") as f_out:
        for _ in tqdm.tqdm(range(len(tasks))):
            res = result_queue.get()
            results.append(res)
            f_out.write(json.dumps(res) + "\n")

    for p in workers:
        p.join()

    total = len(tasks)
    # if plot_path is not None:
    #     plot(output_path, plot_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm-proposer", action='store_true', help="Whether to use LLM proposer for generating proposals")
    parser.add_argument("--data-path", type=str, default="/data/proactive_video_qa/EGO/anno.json", help="Path to the dataset")
    parser.add_argument("--output-path", type=str, default="eval/proactive_video_qa/results/ego_results_opsmm_rl.jsonl", help="Path to save evaluation results")
    parser.add_argument("--plot-path", type=str, default="eval/proactive_video_qa/plots/plots_opsmm_rl/", help="Path to save plots")
    parser.add_argument("--detector-type", type=str, default="opsmm_embedding_v1", help="Type of detector to use")
    parser.add_argument("--detector-config", type=str, default="configs/detector/ops_mm_v1_2B.yaml", help="Path to detector config file")
    parser.add_argument("--proposer-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--responder-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path for response generation")
    parser.add_argument("--streaming-fps", type=int, default=2, help="Streaming FPS for evaluation")
    parser.add_argument("--history-length", type=int, default=5, help="History length for proposal processing")
    parser.add_argument("--history-fps", type=int, default=1, help="History FPS for proposal processing")
    parser.add_argument("--process-rate", type=int, default=2, help="Processing rate (in Hz) for streaming evaluation")
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
    model_args.process_rate = args.process_rate
    model_args.should_respond = args.should_respond
    video_root = os.path.dirname(args.data_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)
    eval_multi_gpu(model_args, args.data_path, args.output_path, video_root, args.plot_path, use_llm_proposer = args.use_llm_proposer, gpu_ids=[int(x) for x in args.gpu_ids.split(",")])