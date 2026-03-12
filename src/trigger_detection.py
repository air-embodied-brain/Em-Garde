import torch
from typing import List

def trigger_detection(similarities: torch.Tensor,
                      trigger_scheme="surge",
                      **kwargs):
    """
    similarities: Tensor [T, N]
    returns: Bool Tensor [N]
    """

    if trigger_scheme == "surge":
        fps = kwargs.get('fps', 5)
        surge_threshold = kwargs.get('surge_threshold', 0.04)

        T, N = similarities.shape

        # Not enough history → no trigger for all samples
        if T < fps + 1:
            return torch.zeros(N, dtype=torch.bool, device=similarities.device)

        # Current similarity: similarities[T-1] → [N]
        curr_sim = similarities[-1]              # [N]

        # Similarity fps steps before last: similarities[T - fps - 1]
        last_sim = similarities[-fps-1]         # [N]

        # Average of all but the last `fps`
        # similarities[:T-fps] → shape [(T-fps), N]
        avg_last_sim = similarities[:-fps].mean(dim=0)  # [N]

        cond1 = (curr_sim - last_sim) > surge_threshold
        # cond2 = (curr_sim - avg_last_sim) > (0.5 * surge_threshold)
        cond2 = (curr_sim - avg_last_sim) > 0
        return cond1 & cond2

    elif trigger_scheme == "binary":
        threshold = kwargs.get('threshold', 0.5)
        fps = kwargs.get('fps', 5)
        T =  similarities.shape[0]

        if T < fps + 1:
            return False

        curr_sim = similarities[-1].item()  # Assuming N=1 for binary scheme
        last_sim = similarities[-2].item()
        past_sim = torch.mean(similarities[-2 * fps-1:-1]).item()
        return curr_sim > threshold and past_sim <= threshold and last_sim <= threshold
    
    else:
        raise ValueError(f"Unknown trigger scheme: {trigger_scheme}")