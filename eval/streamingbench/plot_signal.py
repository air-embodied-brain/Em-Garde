import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 25,
    "lines.linewidth": 3,
    "axes.linewidth": 2,
})

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
        fig_width = len(similarities) / 5
        if isinstance(similarities[0]["similarities"][0],float):       # vlm style detection
            y = [s["similarities"][0] for s in similarities]
            plt.figure(figsize=(fig_width, 6))
            plt.plot(x, y, linestyle="solid", label=data["proposals"][0])
        else:
            proposals = data["proposals"]
            seen = defaultdict(set)
            deduped = []

            for d in proposals:
                (k, v), = d.items()   # assumes exactly one key per dict
                if v not in seen[k]:
                    seen[k].add(v)
                    deduped.append(d)
            proposals = deduped
            
            surges = []
            
            for i in range(len(similarities)):
                surge_max = 0
                if i >=5:
                    for j in range(len(proposals)):
                        if j==0:
                        # if "positive" in proposals[j]:
                            surge = similarities[i]["similarities"][0][j] - similarities[i-5]["similarities"][0][j]
                            if surge > surge_max:
                                surge_max = surge
                surges.append(surge_max)
            plt.figure(figsize=(fig_width, 6))
            plt.plot(x, surges, linestyle="solid")
                

        plt.axvline(data['gt_timestamp'], color="black", linestyle="--", label="Ground Truth Response Time")
        # for det in data["checked_detections"]:
        #     plt.axvline(det["time"], color="green", linestyle=":")
        for det_time in data["detections"]:
            # if det_time not in [d["time"] for d in data["checked_detections"]]:
            plt.axvline(det_time, color="red", linestyle=":")
        plt.axvline(-10, color="red", linestyle=":", label="Model Response Time")
        plt.xlim(x[0], x[-1])
        
        plt.xlabel("Video Time (s)")
        plt.ylabel("Temporal surge signal")
            
        plt.legend()
        plt.savefig(output_path / f"{id}_surge0.svg", bbox_inches="tight")
        
if __name__ == "__main__":
    input_path = "results/streamingbench_results_opsmm_rl_drgrpo.jsonl"
    output_path = "plots/plots_opsmm_rl_drgrpo/"
    plot(input_path, output_path)