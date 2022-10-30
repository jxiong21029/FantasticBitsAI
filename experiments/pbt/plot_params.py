import json
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import PROJECT_DIR

matplotlib.use("Tkagg")

RESULT_DIR = os.path.join(PROJECT_DIR, "experiments/ray_results/limit_test_nops")
trials = [
    dirname
    for dirname in os.listdir(RESULT_DIR)
    if os.path.isdir(os.path.join(RESULT_DIR, dirname))
]
trials.sort()

results = [defaultdict(list) for _ in range(len(trials))]

for i, trial in enumerate(trials):
    with open(os.path.join(RESULT_DIR, trial, "result.json")) as f:
        for line in f.readlines():
            cfg = json.loads(line)["config"]
            for k, v in cfg.items():
                results[i][k].append(v)

colors = [None for _ in range(len(trials))]
for k in results[0].keys():
    fig, ax = plt.subplots()
    for i, result in enumerate(results):
        ax: plt.Axes
        (line,) = ax.plot(np.array(result[k]) * (1 + 0.002 * i), c=colors[i])
        colors[i] = line.get_color()
    fig.savefig(f"param_plots/{k}.png")
