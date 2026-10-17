import os
import json
import subprocess
from pathlib import Path

# ===========================
# CONFIG
# ===========================

INPUT_JSON = "train/sft/llm_data_generation/data/data_ego4d.jsonl"
OUTPUT_JSONL = "sft_data_qwen.jsonl"
OUTPUT_VIDEO_DIR = "data/sft_videos"
INSTRUCTION = "You are an expert video understanding model. Above is a short video segment representing the recent history of a streaming video. Next you'll see a query that's related to future frames. Based on the video segment, propose some visual clues to help identify the frames that answer the query."
FORMAT_INSTRUCTION = '''Respond in the following json format:[\n  {\n    \"positive\": \"clue1\"\n  },\n  {\n    \"positive\": \"clue2\"\n  },...,\n  {\n    \"negative\": \"clue1\"\n  }\n,...]'''
HISTORY_SECONDS = 15  # amount of video before query_time
FPS = 1

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# ===========================
# VIDEO CUTTING FUNCTION
# ===========================

def cut_video_segment(input_path, start_time, end_time, output_path, fps=1):
    """
    Cut a video segment AND convert it to 1 FPS.
    Show ffmpeg stderr only when something goes wrong.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", input_path,
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("\n FFmpeg ERROR:")
        print("Input video:", input_path)
        print("Start:", start_time, "End:", end_time)
        print(result.stderr.decode("utf-8"))

# ===========================
# MAIN
# ===========================

with open(INPUT_JSON, "r") as input_file:
    if INPUT_JSON.endswith(".jsonl"):
        data = [json.loads(line) for line in input_file if line.strip()]
    else:
        data = json.load(input_file)
outfile = open(OUTPUT_JSONL, "a")
for item in data:
    video_path = item["video_path"]
    # minutes, seconds = item["query_time"].split(":")
    # query_time = int(minutes) * 60 + float(seconds)
    query_time = item["query_time"]

    # segment start
    start_time = max(0, query_time - HISTORY_SECONDS)
    end_time = query_time

    # output clipped video path
    vid_name = Path(video_path).stem
    out_video = f"{OUTPUT_VIDEO_DIR}/{vid_name}_{start_time:.2f}_{end_time:.2f}.mp4"

    # cut_video_segment(video_path, start_time, end_time, out_video, FPS)

    # human message
    human_text = (
        "<video>\n"
        + INSTRUCTION
        + "\n"
        + FORMAT_INSTRUCTION
        + f"\nQuery: {item['question']}"
    )

    # proposals → convert to readable string
    # Example: "- positive: xxx\n- negative: yyy"
    gpt_text = gpt_text = json.dumps(item["proposals"], ensure_ascii=False, indent=2)
        
    output_entry = {
        "video": out_video,
        "conversations": [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": gpt_text},
        ]
    }
    outfile.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
    
outfile.close()

print("Done. Output written to", OUTPUT_JSONL)