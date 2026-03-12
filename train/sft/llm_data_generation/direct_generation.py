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

endpoint = "your_endpoint_here"
model_name = "gpt-5"
deployment = "gpt-5"

subscription_key = "your_api_key_here"
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
    task_info_path = './tasks/{}.jsonl'.format(task_category)
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
    Keep the questions short (mostly in one sentence). DO NOT include many details and instructions. Imagine it is asked by someone who is watching the video.
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

def generate_response_from_question(frames, timestamps, question, query_time, task_category="temporal_reasoning"):
    task_info_path = 'tasks/{}.jsonl'.format(task_category)
    with open(task_info_path,'r') as f:
        task_info = json.load(f)
        
    task_qa_example = task_info['task_qa_example']
    randomized_instruction_list = {
        "proposal length":["In general, keep proposals shorter. 1~3 positive clues and 1~2 negative clues are enough.","Try to keep proposals concise. 2~5 positive clues and 2~4 negative clues are sufficient.", "You can enumerate more possibilities in proposals. 4~7 positive clues and 3~5 negative clues are acceptable."],
        "visual detail":["Use high-level ideas instead of visual details in your clues, For example, use 'a person running' instead of 'a person wearing a red shirt running on grass'. Do not mention text or numbers","You can include some visually distinct details in your clues to enrich them, such as colors, clothing, background, etc. But avoid overloading the clues with too many details. No numbers or text allowed","Feel free to add visual details in your clues, like colors, clothing, background, actions, interactions, etc. Avoid high-level concepts that are hard to visualize.", "You can mix high-level concepts and visual details in your clues. For example, 'a person in a blue shirt jumping over a fence'."],
        "logical structure":["Avoid including logic that needs parsing in each clue, For example, don't use 'and', 'or', or 'not' in one clue.","You can use simple logical structures in your clues, such as 'a person jumping or running'. Avoid 'no' or 'not'. Use negative clues for negation","Feel free to use logical structures and negations in your clues, like 'a person either sitting or standing while holding an object'."]
    }
    prompt = f'''
    You are an expert video understanding AI model with reasoning capabilities. Next you will see the recent history of a video frame and a question asked at the end of the video frame. The question is relevant to future frames of the video. In other words, you CANNOT directly answer the question. But you can reason about the video content and help a small visual embedding model that keeps watching the video find the key frames to answer the query. Here is how you can help:
    Your task is to rewrite the query into a few visual clues that can be converted into text embeddings and matched against future video segments. The visual clues are called "proposals". They include positive clues and negative clues. Segments with high similarity to one positive clue while having low similarity to all negative clues are likely to be the key frames that answer the question.
    Follow these rules to generate proposals:
    First give a summarization of the video frames. Extract main events and objects. They help you understand the current video and reason about the future. Then provide positive and negativ clues.
    You can provide clues creatively. They can be video captions, name of actions, certain objects, etc. You might add visual details, or just use high-level concepts. You can also use various sentence structures.
    Positive clues are descriptions of visual content that are likely to appear in the future video segments that answer the question. You can provide multiple clues to enumerate different possibilities, or just use one high-level description to include them all. Each clue should be a stand-alone descriptive phrase. They SHOULD NOT be slightly different descriptions of a same possibility. They SHOULD NOT be direct answers to the question.
    Negative clues are descriptions of visual content that are visually plausible, or might appear in the future video segments, but CANNOT help answer the question. They are hard negatives that might confuse the embedding model. Propose them creatively. Each negative clue should be a stand-alone descriptive phrase. They SHOULD NOT be direct negations of positive clues. They SHOULD NOT be possibilities to answer the query that you did not include in the positive clues. Also, you can provide multiple negative clues or just one.
    Other Specific rules about the response:
    1. proposal length: {random.choice(randomized_instruction_list["proposal length"])}
    2. visual detail: {random.choice(randomized_instruction_list["visual detail"])}
    3. logical structure: {random.choice(randomized_instruction_list["logical structure"])}
    provide your answers in the following format:
    video summary: <your summary here>
    key events and objects: <your extracted events and objects here>
    Proposals:
    [
        {{"positive": <positive clue 1 here>}},
        {{"positive": <positive clue 2 here>}},
        ...
        {{"negative": <negative clue 1 here>}},
        {{"negative": <negative clue 2 here>}},
        ...
    ]
    It is possible that the question is totally unrelated to the video content, or you can directly answer the question with the current frames. In that case, simply respond with "Bad question." after summarizing and extracting information.
    
    Here are several examples:
    {task_qa_example}
    Below are the video frames (in base64-encoded JPEG format) with timestamps. Generate your proposals based on these frames.'''
    
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
        
    query = "Here is the question asked at the end of the above frames: " + question + f" The query time is {query_time:.2f}s. Generate proposals based on the video frames and the question."
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
    
    if "Bad question" in response:
        return None
      
    pattern = r"Proposals:\s*\[\s*(.*?)\s*\]"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None

    block = "[" + match.group(1) + "]"
    cleaned = block.replace("'", '"')
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to fix malformed JSON like trailing commas, etc.
        # You can add more repair logic here if your model outputs vary.
        return None
    return obj
        

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
    
def process_question(video_path, output_path, question, task_category, query_time, response_time):
    try:
        frames, timestamps = get_video_frames(video_path)
        selected_frames, selected_timestamps = get_frames_in_time_range(frames, timestamps, query_time - 15, query_time)
        proposals = generate_response_from_question(selected_frames, selected_timestamps, question, query_time, task_category=task_category)
        if proposals:
            result = {
                "video_path": video_path,
                "question": question,
                "task_category": task_category,
                "query_time": query_time,
                "proposals": proposals,
                "response_time": response_time
            }
            with write_lock:
                with open(output_path, "a") as f_out:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            return True
        else:
            return False
    except Exception as e:
        print(f"[ERROR] Failed to process question for video {video_path}: {e}")
        return False
            
task_categories = ["action_recognition", "object_recognition", "attribute_recognition", "temporal_reasoning", "spatial_reasoning", "clue_revealing", "step_recognition"]

def pipeline_generate(video_root, output_path, max_workers=8, max_video_num=100,frame_sample_num=10):
    
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
                
def pipeline_answer(questions_path, output_path, max_workers=8):
    with open(output_path, "w") as f_out:
        pass
    
    with open(questions_path, "r") as f:
        questions = [json.loads(line) for line in f]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for question in questions:
            video_path = question["video_path"]
            q_text = question["question"]
            task_category = question["task_category"]
            query_time = question["query_time"]
            response_time = question["response_time"]
            futures.append(executor.submit(process_question, video_path, output_path, q_text, task_category, query_time, response_time))
        for future in as_completed(futures):
            result = future.result()
            if result:
                print("[INFO] Successfully processed a question.")
            else:
                print("[WARN] A question failed to process.")
                
if __name__ == "__main__":
    dataset_name = "coin"
    video_root = "path_to_your_video_directory_here"
    question_path = f"data/generated_video_questions_{dataset_name}.jsonl"
    output_path = f"data/data_{dataset_name}.jsonl"
    pipeline_generate(video_root, question_path, max_workers=16, max_video_num=50)
    pipeline_answer(question_path, output_path, max_workers=16)
    
    
    
    
    
    
    