import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def plot(input_path, output_path):
    with open(input_path, "r") as f_in:
        data_lines = [json.loads(line) for line in f_in]
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    for data in data_lines:
        id = data["id"]
        try:
            video_path = data["video_path"]
        except KeyError:
            print("No video path in data:", data)
            continue
        video_name = Path(video_path).stem
        similarities = data["similarities"]
        x = [s["curr_time"] for s in similarities]
        y_pos= []
        y_pos_labels = []
        y_neg = []
        y_neg_labels = []
        fig_width = len(similarities) / 10
        if isinstance(similarities[0]["similarities"][0],float):       # vlm style detection
            y = [s["similarities"][0] for s in similarities]
            plt.figure(figsize=(fig_width, 6))
            plt.plot(x, y, linestyle="solid", label=data["proposals"][0])
        else:
            plt.figure(figsize=(fig_width, 6))
            chunk_size = 150 if data['task'] in ['SSR', 'REC'] else 30000
            for i in range(0, len(similarities), chunk_size):
                offside = 5 if i !=0 else 0
                chunk_x = x[i-offside:i+chunk_size-5]
                num_proposals = len(similarities[i-offside]["similarities"][0])
                chunk_ys = [[] for _ in range(num_proposals)]
                for  s in similarities[i-offside:i+chunk_size-5]:
                    for j in range(num_proposals):
                        chunk_ys[j].append(s["similarities"][0][j])
                for j in range(num_proposals):
                    label = None if data["task"] in ['SSR', 'REC'] else data["proposals"][j].get("positive")
                    plt.plot(chunk_x, chunk_ys[j],label=label)
                
            
            
            # proposals = data["proposals"]
            # seen = defaultdict(set)
            # deduped = []

            # for d in proposals:
            #     (k, v), = d.items()   # assumes exactly one key per dict
            #     if v not in seen[k]:
            #         seen[k].add(v)
            #         deduped.append(d)
            # proposals = deduped
            
            # for i in range(len(proposals)):
            #     proposal = proposals[i]
            #     if "positive" in proposal:
            #         sim = []
            #         for s in similarities:
            #             sim.append(s["similarities"][0][i])
            #         y_pos.append(sim)
            #         y_pos_labels.append(proposal["positive"])             #! Here the proposals are the last updated proposala. Need to be changed for better visualization.
            #     else:
            #         sim = []
            #         for s in similarities:
            #             sim.append(s["similarities"][0][i])
            #         y_neg.append(sim)
            #         y_neg_labels.append(proposal["negative"])
            # plt.figure(figsize=(fig_width, 6))
            # for y in y_pos:
            #     plt.plot(x, y, linestyle="solid", label=y_pos_labels[y_pos.index(y)])
            # for y in y_neg:
            #     plt.plot(x, y, linestyle="dashed", label=y_neg_labels[y_neg.index(y)])
                

        for gt in data["gt_timestamp"]:
            plt.axvline(gt, color="black", linestyle="--")
        for det in data["checked_detections"]:
            plt.axvline(det["time"], color="green", linestyle=":")
        for det_time in data["detections"]:
            if det_time not in [d["time"] for d in data["checked_detections"]]:
                plt.axvline(det_time, color="red", linestyle=":")
            
        plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.savefig(output_path / f"{id}.jpg", bbox_inches="tight")
        
        
if __name__ == "__main__":
    input_path = "results/ovo_bench_results_check1.jsonl"
    output_path = "plots/plots_ovo_bench_check1/"
    plot(input_path, output_path)
    
        