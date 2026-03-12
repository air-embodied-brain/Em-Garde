import cv2
import base64
import os
from openai import AzureOpenAI
import time
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# ---- Config ----
MODEL = "gpt-5"                     # vision-capable model
FPS_SAMPLE = 1                       # sample 2 frames per second

endpoint = "replace_with_your_endpoint"
model_name = "gpt-5"
deployment = "gpt-5"

subscription_key = "your_subscription_key_here"
api_version = "your_api_version_here"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

write_lock = threading.Lock()


def smart_resize(frame, max_width=480):
    """
    Resize an OpenCV image to have width <= max_width,
    preserving the original aspect ratio.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame  # no need to resize

    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def get_video_frames(video_path):
# ---- Extract frames ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(round(fps / FPS_SAMPLE))

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps

    frames = []
    timestamps = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Compute timestamp (in seconds)
            frame = smart_resize(frame, max_width=480)
            
            timestamp_sec = frame_idx / fps
            timestamps.append(timestamp_sec)

            # Add timestamp text to frame
            txt = f"{timestamp_sec:.2f}s"
            cv2.putText(frame, txt, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

            # Encode to base64 JPEG
            _, buf = cv2.imencode(".jpg", frame)
            b64_img = base64.b64encode(buf).decode("utf-8")
            frames.append(b64_img)
            # cv2.imwrite(f"images/frame_{frame_idx}.jpg", frame)  # Optional: save frame as image file

        frame_idx += 1
        
    width = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 480)
    height = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(480 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print(f"Extracted {len(frames)} frames from video ({width}x{height}, {duration_sec:.2f}s)")
    cap.release()
    return frames, timestamps

def chunk_frames(frames, timestamps, chunk_size=20):
    """Yield successive n-sized chunks from frames and timestamps."""
    for i in range(0, len(frames), chunk_size):
        yield frames[i:i + chunk_size], timestamps[i:i + chunk_size]
        
def sample_frames(frames, timestamps, sample_num=10, chunk_size=20):
    """Yield sample_num video segments each with chunk_size frames."""
    total_chunks = len(frames) // chunk_size
    sampled_chunk_indices = random.sample(range(total_chunks), min(sample_num, total_chunks))
    for idx in sampled_chunk_indices:
        start_idx = idx * chunk_size
        yield frames[start_idx:start_idx + chunk_size], timestamps[start_idx:start_idx + chunk_size]
        
def get_frames_in_time_range(frames, timestamps, start_time, end_time):
    """Get frames and their timestamps within a specific time range."""
    selected_frames = []
    selected_timestamps = []
    for frame, timestamp in zip(frames, timestamps):
        if start_time <= timestamp <= end_time:
            selected_frames.append(frame)
            selected_timestamps.append(timestamp)
    return selected_frames, selected_timestamps

def recursive_find_video_paths(root_dir, video_extensions={".mp4", ".avi", ".mov", ".mkv"}):
    video_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_paths.append(os.path.join(dirpath, filename))
    return video_paths

def generate_question_from_frames(frames, timestamps, task_category = "action_recognition"):
    task_info_path = 'train/sft/llm_data_generation/tasks/{}.jsonl'.format(task_category)
    with open(task_info_path,'r') as f:
        task_info = json.load(f)
        
    task_name = task_info['task_name']
    task_description = task_info['description']
    example_question_format = task_info['example_question_format']
    generation_guideline = task_info['generation_guideline']
    
    prompt = f'''
    You are an expert video understanding AI model. Your task is to generate a question based on the content of the provided video frames. You should provide the question, the query time as well as the response times. The question should be proactive: the query time is BEFORE the event that provides the answer. When a model is given the query, it should be able to watch the video from the query time and find the answer in the subsequent frames.
    Below are the details of the question you need to generate:
    task type: {task_name},
    task description: {task_description},
    example question format: {example_question_format},
    guideline for question generation: {generation_guideline}.
    To generate the question, first summarize the video and extract main events and objects. Note the event boundaries and key actions. They should help you propose informative questions. Then, formulate a question that can be answered by observing the video after the query time. After that, select a suitable query time (in seconds) that is BEFORE the event that provides the answer, but at least 10 seconds from the start of the video so that the model that answers the question can understand the video. Finally choose the response time based on the timestamp that the answer is revealed. Note that one question might be responded at one timestamp or multiple timestamps, so use a list to include them. The timestamps are already written on the frames.
    The example question formats are only for reference and to help you understand the task. When providing the question, try to ask in novel and creative ways. Imagine yourself to be a viewer with different interests and backgrounds, and ask questions that such a viewer might want to know about the content after your query time.
    Keep the questions short (mostly in one sentence) and high-level. DO NOT include many details and instructions. Imagine it is asked by someone who is watching the video. Especially, DO NOT add details that cannot be seen before the query time or might help answering the question, e.g. the purpose of a tool, the detail of an action yet to be performed, the shape/color of an object that is not shown yet, etc.
    Provide your response in the following format:
    video summary: <your summary here>
    key events and objects: <your extracted events and objects here>
    Generated Question:
    {{
        "question": "<your generated question here>",
        "query_time": <query time in seconds, float>,
        "response_time": [<times when the query should be answered, float>]
    }}
    It is possible that the video clip cannot produce relevant and valuable question. In that case, simply respond with "No question can be generated." after summarizing and extracting information.
    The following are the video frames (in base64-encoded JPEG format) with timestamps. Generate your question based on these frames.'''
    
    content = [{"type": "text", "text": prompt}]
    for t, img in zip(timestamps, frames):
        content.append({
            "type": "text",
            "text": f"Frame at {t:.2f}s:"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url":f"data:image/jpeg;base64,{img}"}
        })
        
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        max_completion_tokens=10000,  # limit response length
    )
    response = response.choices[0].message.content
    
    if "No question can be generated" in response:
        return None

    # Extract the JSON-like block after "Generated Question:"
    pattern = r"Generated Question:\s*\{\s*(.*?)\s*\}"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None

    block = "{" + match.group(1) + "}"

    # Try to convert into valid JSON
    # Replace single quotes with double quotes if necessary
    cleaned = block.replace("'", '"')

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to fix malformed JSON like trailing commas, etc.
        # You can add more repair logic here if your model outputs vary.
        return None

    # Validate required fields
    question = obj.get("question", None)
    query_time = obj.get("query_time", None)
    response_time = obj.get("response_time", None)

    if question is None or query_time is None or response_time is None:
        return None

    # Convert query_time to float if possible
    try:
        query_time = float(query_time)
    except Exception:
        return None

    return {"question": question, "query_time": query_time, "response_time": response_time}

def generate_target_timestamps(frames, timestamps, query):
    prompt = '''You are an expert video understanding AI model. Next you will see a sequence of video frames with timestamps and a user query. Your task is to identify the timestamps in the provided video frames that are relevant (and irrelevant) to answering the given query and sample from these timestamps. Please follow the instructions below:
    1. First analyze the video frames and extract events from them. Events are defined as a sequence of frames that can be grouped together and described with a simple sentence. Note that inconsecutive frames that share the same description should be grouped into the SAME event.
    2. Filter events that can directly answer the query. Imagine you are watching the video, when will you shout out the answer to the query? These events are considered relevant to the query. For most queries, ther should be only ONE relevant event. For some (describe the steps, ...), there might be multiple relevant events. Note: ignore words like "start to", "for the first time", "continue to", etc. when determining relevant events. As long as the visual content can provide the answer, it is relevant.
    3. Sample 10~30 timestamps throughout the video, from both relevant and irrelevant events. Make sure the sampled timestamps are evenly distributed across events and capture the diversity of the video content. Each relevant event should have at least 3 sampled timestamps. If there is only one or two relevant events, this number can increased to 5 or more. Also make sure to include timestamps from diverse irrelevant events to provide contrast.
    4. In your answer, first provide extracted events with their descriptions. Then provide the answer in a json format showing the sampled timestamps and each relevant event, with indices of corresponding timestamps, as shown below.
    5. It is possible the the question is badly formed and cannot be answered by the video content. In that case, simply respond with "The query cannot be answered based on the video content." after extracting events.
    Answer format:
    Events:
    <Your extracted events here>
    Answers:
    {
        "target_timestamps": [<list of sampled timestamps, float>],
        "events":[
            {"description": "<event 1 description>", "timestamp_indices": [<list of corresponding indices in target_timestamps>]},
            {"description": "<event 2 description>", "timestamp_indices": [<list of corresponding indices in target_timestamps>]},
            ...
        ]
    }
    Below are the video frames.
    '''
    content = [{"type": "text", "text": prompt}]
    for t, img in zip(timestamps, frames):
        content.append({
            "type": "text",
            "text": f"Frame at {t:.2f}s:"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url":f"data:image/jpeg;base64,{img}"}
        })
    query = f"User Query: {query}" + "\nYour answer:"
    content.append({"type": "text", "text": query})
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        max_completion_tokens=10000,  # limit response length
    )
    response = response.choices[0].message.content
    
    if "The query cannot be answered based on the video content." in response:
        return None
    
    pattern = r"Answers:\s*\{\s*(.*?)\s\}"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    block = "{" + match.group(1) + "}"
    cleaned = block.replace("'", '"')
    print(cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    target_timestamps = obj.get("target_timestamps", None)
    events = obj.get("events", None)
    if target_timestamps is None or events is None:
        return None
    return {"target_timestamps": target_timestamps, "events": events}
    

def process_video(video_path, output_path, task_category="action_recognition", sample_num=None):
    try:
        frames, timestamps = get_video_frames(video_path)
        all_questions = []
        if sample_num:
            for chunk_frames_, chunk_timestamps in sample_frames(frames, timestamps, sample_num=sample_num, chunk_size=30):
                result = generate_question_from_frames(chunk_frames_, chunk_timestamps, task_category=task_category)
                if result:
                    all_questions.append({
                        "video_path": video_path,
                        "question": result["question"],
                        "task_category": task_category,
                        "query_time": result["query_time"],
                        "response_time": result["response_time"]
                    })
        else:
            for chunk_frames_, chunk_timestamps in chunk_frames(frames, timestamps, chunk_size=30):
                result = generate_question_from_frames(chunk_frames_, chunk_timestamps, task_category=task_category)
                if result:
                    all_questions.append({
                        "video_path": video_path,
                        "question": result["question"],
                        "task_category": task_category,
                        "query_time": result["query_time"],
                        "response_time": result["response_time"]
                    })
        # Write results to output file
        with write_lock:
            with open(output_path, "a") as f_out:
                for qa in all_questions:
                    f_out.write(json.dumps(qa, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process video {video_path} for category {task_category}: {e}")
        return False
        
def store_frames_for_analysis(frames, timestamps, event_timestamps, analysis_path):
    os.makedirs(analysis_path, exist_ok=True)
    for i, t in enumerate(event_timestamps):
        closest_frame_index = min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - t))
        frame_b64 = frames[closest_frame_index]
        frame_data = base64.b64decode(frame_b64)
        with open(os.path.join(analysis_path, f"event_frame_{i}_at_{t:.2f}s.jpg"), "wb") as f_img:
            f_img.write(frame_data)
        

def process_question(id, video_path, query_time, question, output_path, analysis_path=None):
    try:
        frames, timestamps = get_video_frames(video_path)
        if isinstance(query_time, float):
            start_time = query_time
        else:
            # query time is in mm:ss. convert to float.
            mm, ss = map(int, query_time.split(":"))
            start_time = float(mm * 60 + ss) 
        end_time = start_time + 45.0  
        frames, timestamps = get_frames_in_time_range(frames, timestamps, start_time, end_time)
        
        result = generate_target_timestamps(frames, timestamps, question)
        if not result:
            return id, False
        with write_lock:
            with open(output_path, "a") as f_out:
                data_item = {
                    "id": id,
                    "video_path": video_path,
                    "question": question,
                    "query_time": query_time,
                    "target_timestamps": result["target_timestamps"],
                    "events": result["events"]
                }
                if analysis_path:
                    sampled_timestamps = result["target_timestamps"]
                    irrelevant_timestamps = set(sampled_timestamps)
                    for i, event in enumerate(result["events"]):
                        event_timestamps = [sampled_timestamps[idx] for idx in event["timestamp_indices"]]
                        store_frames_for_analysis(frames, timestamps, event_timestamps, os.path.join(analysis_path, f"event_{i}"))
                        irrelevant_timestamps -= set(event_timestamps)
                    store_frames_for_analysis(frames, timestamps, list(irrelevant_timestamps), os.path.join(analysis_path, "irrelevant"))
                
                f_out.write(json.dumps(data_item, ensure_ascii=False) + "\n")
        return id, True
    except Exception as e:
        print(f"[ERROR] Failed to process question for video {video_path}: {e}")
        return id, False

task_categories = ["action_recognition", "object_recognition", "attribute_recognition", "temporal_reasoning", "spatial_reasoning", "clue_revealing", "step_recognition"]
    
def pipeline_generate_question(video_root, output_path, max_workers=8, max_video_num=100,frame_sample_num=None):
    
    video_paths = recursive_find_video_paths(video_root)
    # randomly sample 100 videos
    video_paths = random.sample(video_paths, min(max_video_num, len(video_paths)))
    
    print(f"[INFO] Found {len(video_paths)} video files.")
    
    with open(output_path, "w") as f_out:
        pass
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_path in video_paths:
            for task_category in task_categories:
                futures.append(executor.submit(process_video, video_path, output_path, task_category, sample_num=frame_sample_num))
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                print("[INFO] Successfully processed a video-task pair.")
            else:
                print("[WARN] A video-task pair failed to process.")
                
def pipeline_process_question(questions_path, output_path, analysis_path, max_workers=8):
    if questions_path.endswith('.jsonl'):
        with open(questions_path, "r") as f_in:
            lines = f_in.readlines()
            data = [json.loads(line) for line in lines]
    else:
        with open(questions_path, "r") as f_in:
            data = json.load(f_in)
    
    with open(output_path, "w") as f_out:
        pass
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for id, item in enumerate(data):
            video_path = item["video_path"]
            question = item["question"] if "question" in item else item["query"]
            query_time = item["query_time"]
            futures.append(executor.submit(process_question, id, video_path, query_time, question, output_path, os.path.join(analysis_path, f"{id}") if analysis_path else None))
        
        for future in as_completed(futures):
            id, result = future.result()
            if result:
                print(f"[INFO] Successfully processed question id {id}.")
            else:
                print(f"[WARN] Question id {id} failed to process.")
                
if __name__ == "__main__":
    video_path = "path_to_your_videos"
    dataset_name = "coin"
    question_path = "train/.json"
    output_path = "train/rl/llm_data_generation/rl_data_.jsonl"
    analysis_path = "train/rl/llm_data_generation/analysis3"
    pipeline_generate_question(video_path, question_path, max_workers=16, max_video_num=100)
    pipeline_process_question(question_path, output_path, analysis_path, max_workers=16)