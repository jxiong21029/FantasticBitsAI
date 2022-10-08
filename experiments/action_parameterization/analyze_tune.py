import json
import os

for action_param in ("von_mises", "euclidean", "normed_euclidean"):
    data = []
    root = f"../ray_results/action_param4_{action_param}/"
    for trialdir in os.listdir(root):
        if not os.path.isdir(os.path.join(root, trialdir)):
            continue
        try:
            with open(os.path.join(root, trialdir, "params.json")) as f:
                params = json.load(f)
            with open(os.path.join(root, trialdir, "result.json")) as f:
                results = json.load(f)
        except json.JSONDecodeError:
            continue
        data.append((results["mo3_eval_goals_scored_mean"], params))
    data.sort(reverse=True, key=lambda d: d[0])
    print(f"{action_param} top 3 results:")
    for i in range(3):
        print(data[i])
