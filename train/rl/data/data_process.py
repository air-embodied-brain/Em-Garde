# Data processing script
# TODO:
# 1. Convert question into the form with <video> token and instructions
# 2. Chunk and resize videos from query and target timestamps, store the results into the target path, and record the paths in query_videos and target_videos
# 3. convert the results into parquet format for future loading

import os
import json
from typing import List
import multiprocessing as mp
from datasets import Dataset
import traceback
import time


import torch
import torchvision.io as tvio
from datasets import Dataset
from fractions import Fraction

from src.video_utils import read_video_decord, read_video_decord_strict, read_video_ffmpeg_fixed

def save_video_tensor(video: torch.Tensor, path: str, fps: float):
    """
    video: [T, C, H, W], float or uint8
    """
    print(f"[PID {os.getpid()}] writing {path} frames={video.shape[0]}", flush=True)
    fps = Fraction(float(fps))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if video.dtype != torch.uint8:
        video = (video.clamp(0, 255)).to(torch.uint8)

    # torchvision expects [T, H, W, C]
    video = video.permute(0, 2, 3, 1)

    tvio.write_video(
        path,
        video,
        fps=fps,
        video_codec="libx264",
        options={
            "crf": "23",
            "preset": "ultrafast",
            "pix_fmt": "yuv420p",
        }
    )
    print(f"[PID {os.getpid()}] done {path}", flush=True)
    
def is_valid_video(v):
    if v is None:
        return False
    if v.ndim != 4:
        return False
    if v.shape[0] < 2:  # ffmpeg hates 0 or 1 frame
        return False
    if not torch.isfinite(v).all():
        return False
    return True

def process(id, video_path, question, query_time, history_length, history_fps, tgt_times, streaming_length, streaming_fps, answer, events, video_root_dir):
    target_start_time = query_time - 1.0
    end_index = min(events[-1]["timestamp_indices"][-1], len(tgt_times)-1)
    target_end_time = tgt_times[end_index] + 10.0  # add some buffer
    query_video = read_video_ffmpeg_fixed(video_path, query_time-history_length, query_time, history_fps, max_pixels=560*560)
    tgt_video = read_video_ffmpeg_fixed(video_path, target_start_time, target_end_time, streaming_fps, max_pixels=560*560)
    # store the videos
    query_video_path = os.path.join(video_root_dir, id, f"query_{os.path.basename(video_path).split('.')[0]}_{query_time}.mp4")
    tgt_video_path = os.path.join(video_root_dir, id, f"tgt_{os.path.basename(video_path).split('.')[0]}.mp4")
    
    save_video_tensor(query_video, query_video_path, history_fps)
    save_video_tensor(tgt_video, tgt_video_path, streaming_fps)
            
    
    
    INSTRUCTION = "You are an expert video understanding model. Above is a short video segment representing the recent history of a streaming video. Next you'll see a query that's related to future frames. Based on the video segment, propose some visual clues to help identify the frames that answer the query."
    FORMAT_INSTRUCTION = '''Respond in the following json format:[\n  {\n    \"positive\": \"clue1\"\n  },\n  {\n    \"positive\": \"clue2\"\n  },...,\n  {\n    \"negative\": \"clue1\"\n  }\n,...]'''
    human_text = (
        "<video>\n"
        + INSTRUCTION
        + "\n"
        + FORMAT_INSTRUCTION
        + f"\nQuery: {question}"
    )
    
    positive_indices = [event["timestamp_indices"][0] for event in events]
    positive_timestamps = [tgt_times[idx] - query_time + 1 for idx in positive_indices]
    data_item = {
        "problem": human_text,
        "answer": answer,
        "query_video": {"video": query_video_path},
        "target_videos": {"video": tgt_video_path},
        "positive_timestamps": positive_timestamps,
        # "events": events
    }
    print(f"Processed data item ID {id} with query video at {query_video_path}.")
    
    return data_item

def _worker_process(queue, args):
    """
    Runs process(...) in an isolated subprocess.
    Reports either:
      ("ok", result)
      ("error", traceback)
    """
    try:
        result = process(*args)
        queue.put(("ok", result))
    except Exception:
        queue.put(("error", traceback.format_exc()))


# -----------------------------
# Safe call with hard timeout
# -----------------------------

def run_process_safely(args, timeout_sec=120):
    """
    Returns:
      dict  -> success
      None  -> failure (error or hang)
    """
    ctx = mp.get_context("spawn")  # REQUIRED for decord / ffmpeg
    queue = ctx.Queue()
    p = ctx.Process(target=_worker_process, args=(queue, args))

    p.start()
    p.join(timeout_sec)

    # Case 1: hard hang
    if p.is_alive():
        print("Worker hung, terminating process...")
        p.terminate()
        p.join()
        return None

    # Case 2: process exited but returned nothing (rare)
    if queue.empty():
        print("Worker exited without result.")
        return None

    status, payload = queue.get()

    if status == "ok":
        return payload

    # Case 3: Python exception inside worker
    print("Worker error:\n", payload)
    return None


# -----------------------------
# DROP-IN REPLACEMENT
# -----------------------------

def process_data(
    input_json_path: str,
    output_parquet_path: str,
    output_json_path: str,
    video_root_dir: str,
    timeout_sec: int = 60,
):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    history_length = 10.0
    history_fps = 1.0
    streaming_length = 2.0
    streaming_fps = 5.0

    processed_data = []
    skipped = 0

    os.makedirs(video_root_dir, exist_ok=True)

    for idx, item in enumerate(data):
        print(f"\n[{idx+1}/{len(data)}] Processing ID {item['id']}")

        args = (
            str(item["id"]),
            item["video_path"],
            item["question"],
            item["query_time"],
            history_length,
            history_fps,
            item["target_timestamps"],
            streaming_length,
            streaming_fps,
            item.get("answer", ""),
            item["events"],
            video_root_dir,
        )

        result = run_process_safely(args, timeout_sec=timeout_sec)

        if result is None:
            skipped += 1
            print(f"Skipped ID {item['id']}")
            continue

        processed_data.append(result)

    print("\n==============================")
    print(f"Finished processing.")
    print(f"Valid samples: {len(processed_data)}")
    print(f"Skipped samples: {skipped}")
    print("==============================\n")

    if not processed_data:
        raise RuntimeError("No valid data processed. All samples failed.")

    dataset = Dataset.from_list(processed_data)
    dataset.to_parquet(output_parquet_path)

    with open(output_json_path, "w") as f_out:
        for item in processed_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

# def process_data(input_json_path: str, output_parquet_path: str, output_json_path: str, video_root_dir: str):
#     with open(input_json_path, "r") as f:
#         data = json.load(f)
    
#     history_length = 10.0  # seconds
#     history_fps = 1.0  # fps
#     streaming_length = 2.0  # seconds
#     streaming_fps = 5.0  # fps
    
#     processed_data = []
#     for item in data:
#         id = item["id"]
#         video_path = item["video_path"]
#         question = item["question"]
#         query_time = item["query_time"]
#         # pos_tgt_times = item["pos_tgt_times"]
#         # neg_tgt_times = item["neg_tgt_times"]
#         tgt_times = item["target_timestamps"]
#         answer = item["answer"] if "answer" in item else ""
#         events = item["events"]
        
#         if int(id)>50:
#             processed_item = process(
#                 str(id),
#                 video_path,
#                 question,
#                 query_time,
#                 history_length,
#                 history_fps,
#                 tgt_times,
#                 streaming_length,
#                 streaming_fps,
#                 answer,
#                 events,
#                 video_root_dir
#             )
#             processed_data.append(processed_item)
        
#     # TODO: train-test split
    
#     dataset = Dataset.from_list(processed_data)
#     dataset.to_parquet(output_parquet_path)
#     with open(output_json_path, "w") as f_out:
#         for item in processed_data:
#             f_out.write(json.dumps(item, indent=4, ensure_ascii=False) + "\n")
    
if __name__ == "__main__":
    input_json_path = "train/rl/data/filtered.json"
    output_parquet_path = "/data/Em_garde/rl_format.parquet"
    output_json_path = "train/rl/data/rl.jsonl"
    video_root_dir = "/data/Em_garde/processed_videos2/"
    
    process_data(input_json_path, output_parquet_path, output_json_path, video_root_dir)
    