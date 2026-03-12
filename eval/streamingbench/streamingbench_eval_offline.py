import json
import tqdm
from src.model_args import ModelArguments
from src.model import StreamingModel
from src.embedding_comparison import load_detector_config
import torch
import re
import os
import argparse
from .plot_similarity import plot

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
    subset,
    question,
    qid,
    use_llm_proposer=False,
    process_rate = 2,
):

    context_time = 16
    video_path = subset["video_path"]
    timestamp = question["time_stamp"]
            # convert timestamps like "00:03:10" to seconds
    timestamp = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(":"))))

    if context_time > 0:
        start_time = max(0, timestamp - context_time)
    else:
        start_time = 0

    model.init_video(video_path, delay_load=True)
        
    PROMPT_TEMPLATE = '''{} Provide the best answer choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

    Options:
    {}
    {}
    {}
    {}'''
    
    if "options" in question.keys():
        options = question["options"]
        if not options[0].startswith("A."):
            options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
    query_for_response = PROMPT_TEMPLATE.format(
        question["question"],
        *options
    )
        
    answer = model.offline_respond(query_for_response, context_time, model_args.history_fps, timestamp)
    
    answer = process(answer)
    res = {
        "id": qid,
        "video_path": video_path,
        "task": question["task_type"],
        "answer": answer,
        "gt": question["answer"],
    }

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

    with open(data_path, "r") as f:
        data = json.load(f)

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

    # recalls_2 = recalls_1 = 0
    # precisions_2 = precisions_1 = 0

    with open(output_path, "w") as f_out:
        for _ in tqdm.tqdm(range(len(tasks))):
            res = result_queue.get()
            f_out.write(json.dumps(res) + "\n")
            f_out.flush()

    for p in workers:
        p.join()

    # total = len(tasks)
    # print(
    #     f"Recall@2s: {recalls_2/total:.4f}, "
    #     f"Precision@2s: {precisions_2/total:.4f}, "
    #     f"Recall@1s: {recalls_1/total:.4f}, "
    #     f"Precision@1s: {precisions_1/total:.4f}"
    # )
    # if plot_path is not None:
    #     plot(output_path, plot_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm-proposer", action='store_true', help="Whether to use LLM proposer for generating proposals")
    parser.add_argument("--data-path", type=str, default="/data/StreamingBench/src/data/questions_real.json", help="Path to the dataset")
    parser.add_argument("--output-path", type=str, default="eval/streamingbench/results/streamingbench_results_offline.jsonl", help="Path to save evaluation results")
    parser.add_argument("--plot-path", type=str, default="eval/streamingbench/plots/plots_offline/", help="Path to save plots")
    parser.add_argument("--detector-type", type=str, default="opsmm_embedding_v1", help="Type of detector to use")
    parser.add_argument("--detector-config", type=str, default="configs/detector/ops_mm_v1_2B.yaml", help="Path to detector config file")
    parser.add_argument("--proposer-model-name", type=str, default="fredzheng/Em-Garde-7B", help="LLM model name or path")
    parser.add_argument("--responder-model-name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="LLM model name or path for response generation")
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
    eval_multi_gpu(model_args, args.data_path, args.output_path, args.plot_path, use_llm_proposer = args.use_llm_proposer, gpu_ids=[int(x) for x in args.gpu_ids.split(",")])