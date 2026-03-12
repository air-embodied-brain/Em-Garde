import json
import tqdm
from src.model_args import ModelArguments
from src.model import StreamingModel
from src.embedding_comparison import load_detector_config
import torch
import re
import os
import argparse
from .constant import *

def process(answer):
    option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
    match = option_regex.match(answer.strip())
    
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    if match:
            # If matched, return the option letter in uppercase
        return match.group(1).upper()
    else:
        # If no match, process the answer as before
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = processPunctuation(answer)
        answer = answer.strip("'")
        answer = answer.strip('"')
        answer = answer.strip(")")
        answer = answer.strip("(")
        answer = answer.strip().lower()

        # Try to find any single letter (A-E) in the processed answer
        letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

        return answer
        
def eval_one_question(
    model,
    model_args,
    task,             
):
    qid = task["id"]
    task_type = task["task"]
    query_for_response = task["query_for_response"]
    response_time = task["response_time"]
    video_path = task["video_path"]
    respond_history_length = 32
    response_fps = 1
    
    model.init_video(video_path, delay_load=True)
    answer = model.offline_respond(query_for_response, respond_history_length, response_fps, response_time)
    # proposal, answer = model.offline_respond(query_for_proposal,query_for_response,proposal_history_length,respond_history_length,proposal_fps,response_fps,proposal_time,response_time)
    
    answer = process(answer)
    res = {
        "id": qid,
        "video_path": video_path,
        "answer": answer,
        "gt": task["gt"],
        "task": task_type,
    } 
    gt = task["gt"]
    if answer.strip() == gt.strip():
        res["correct"] = True
    else:
        res["correct"] = False
    
    return res

def gpu_worker(
    gpu_id,
    task_queue,
    result_queue,
    model_args,
    use_llm_proposer,
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

        
        # try:
        res = eval_one_question(
            model, model_args, task
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
    output_path,
    plot_path=None,
    video_root=None,
    use_llm_proposer=False,
    gpu_ids=(0, 1, 2, 3),
):
    import json
    import multiprocessing as mp
    import tqdm

    with open(data_path, "r") as f:
        data = json.load(f)
    
    tasks = []
    data = [item for item in data if item['task'] in BACKWARD_TASKS + REAL_TIME_TASKS]
    for item in data:
        task = item['task']
        if task in BACKWARD_TASKS or task in REAL_TIME_TASKS:
            id = item["id"]
            video_path = os.path.join(video_root, f"{id}.mp4")
            question = item["question"]
            options = item["options"]
            question_for_response = build_prompt(task, question, options, None, None)
            proposal_time= response_time = -1
            gt = chr(65 + item["gt"])
            tasks.append({
                "id": id,
                "task": task,
                "video_path": video_path,
                "query_for_proposal": question,          # not used in backward tasks
                "query_for_response": question_for_response,
                "proposal_time": proposal_time,
                "response_time": response_time,
                "gt": gt,
            })
        else:
            id = item["id"]
            test_info = item["test_info"]
            for i, test in enumerate(test_info):
                video_path = os.path.join(video_root, f"{id}_{i}.mp4")
                question_for_response = build_prompt(task = task, question = None, options = None, _anno_ = item, index = i)
                if task=='REC':
                    question = "How many times does the event: {} happen?".format(item['activity'])
                elif task=='SSR':
                    question = "Describe the steps of {}".format(re.sub(r'(?<!^)(?=[A-Z])', ' ', item["tutorial"]).lower())
                else:
                    question = item["question"]
                video_path = os.path.join(video_root, f"{id}_{i}.mp4")
                qid = str(id) + f"_{i}"
                if task == "CRR":
                    proposal_time = item["ask_time"]
                    response_time = -1
                else:
                    proposal_time = response_time = -1
                    
                if task == "REC":
                    gt = str(test["count"])
                else:
                    gt = "Yes" if test["type"] == 1 else "No"
                    
                tasks.append({
                    "id": qid,
                    "task": task,
                    "video_path": video_path,
                    "query_for_proposal": question,          # not used in backward tasks
                    "query_for_response": question_for_response,
                    "proposal_time": proposal_time,
                    "response_time": response_time,
                    "gt": gt,
                })
                

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
            f_out.flush()
            
    for task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        correct = sum(1 for r in results if r["task"] == task and r["correct"])
        total = sum(1 for r in results if r["task"] == task)
        if total > 0:
            accuracy = correct / total * 100
            print(f"Task: {task}, Accuracy: {accuracy:.2f}% ({correct}/{total})")
        else:
            print(f"Task: {task}, No samples found.")
            
            
def build_prompt(task, question, options, _anno_, index):
        if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
            formatted_options = '\n'.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
            prompt = BR_PROMPT_TEMPLATE.format(question, formatted_options)
            
        elif task == "REC":
            activity = _anno_["activity"]
            question = "How many times did they " + activity + "?"
            prompt = REC_PROMPT_TEMPLATE.format(question)
        elif task == "SSR":
            step = _anno_["test_info"][index]["step"]
            prompt = SSR_PROMPT_TEMPLATE.format(step)
        elif task == "CRR":
            question = _anno_["question"]
            prompt = CRR_PROMPT_TEMPLATE.format(question)
        return prompt
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm-proposer", action='store_true', help="Whether to use LLM proposer for generating proposals")
    parser.add_argument("--data-path", type=str, default="/data/OVO-Bench/data/ovo_bench_new.json", help="Path to the dataset")
    parser.add_argument("--output-path", type=str, default="eval/ovo_bench/results/ovo_bench_results_offline.jsonl", help="Path to save evaluation results")
    parser.add_argument("--plot-path", type=str, default="eval/ovo_bench/plots/plots_ovo_bench_offline/", help="Path to save plots")
    parser.add_argument("--detector-type", type=str, default="opsmm_embedding_v1", help="Type of detector to use")
    parser.add_argument("--detector-config", type=str, default="configs/detector/ops_mm_v1_2B.yaml", help="Path to detector config file")
    parser.add_argument("--proposer-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--responder-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--streaming-fps", type=int, default=5, help="Streaming FPS for evaluation")
    parser.add_argument("--history-length", type=int, default=5, help="History length for proposal processing")
    parser.add_argument("--history-fps", type=int, default=1, help="History FPS for proposal processing")
    parser.add_argument("--should-respond", action='store_true', help="Whether to enable LLM response check and actual response")
    parser.add_argument("--gpu-ids", type=str, default="2,3,4,5", help="Comma separated GPU IDs to use for evaluation")
    
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
    video_root = os.path.join(os.path.dirname(args.data_path), "chunked_videos")
    os.makedirs(args.plot_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    eval_multi_gpu(model_args, args.data_path, args.output_path, args.plot_path, video_root, use_llm_proposer = args.use_llm_proposer, gpu_ids=[int(x) for x in args.gpu_ids.split(",")])