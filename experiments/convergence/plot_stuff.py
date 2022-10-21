import random
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt

from utils import Logger

matplotlib.use("Tkagg")
fig, axes = plt.subplots(nrows=2, ncols=3)
length = 30

logger = Logger()
betas = [0.75, 0.9]
color_refs = {}


axes[0][0].set_title("Exponential Convergence")
axes[0][0].set_ylabel("Data")
axes[0][1].set_ylabel("Convergence")

dataset = [-(1 - 0.3**i) for i in range(length)]
axes[0][0].plot(dataset)
results = defaultdict(list)
for beta in betas:
    for n in range(1, length):
        logger.cumulative_data["data"] = dataset[:n]
        results[beta].append(logger.estimate_convergence("data", smoothing=beta))
    if beta not in color_refs:
        (color_refs[beta],) = axes[1][0].plot(results[beta], label=str(beta))
    else:
        axes[1][0].plot(results[beta], c=color_refs[beta].get_color(), label=str(beta))

axes[0][1].set_title("Noisy Exponential Convergence")
dataset = [1 - 0.5**i + random.random() / 15 for i in range(length)]
axes[0][1].plot(dataset)

results = defaultdict(list)
for beta in betas:
    for n in range(1, length):
        logger.cumulative_data["data"] = dataset[:n]
        results[beta].append(logger.estimate_convergence("data", smoothing=beta))
    axes[1][1].plot(results[beta], c=color_refs[beta].get_color(), label=str(beta))

axes[0][2].set_title("Noisy Power Law Convergence")
dataset = [(i + 1) ** -0.8 + random.gauss(0, 0.03) for i in range(30)]
axes[0][2].plot(dataset)

results = defaultdict(list)
for beta in betas:
    for n in range(1, len(dataset)):
        logger.cumulative_data["data"] = dataset[:n]
        results[beta].append(logger.estimate_convergence("data", smoothing=beta))
    axes[1][2].plot(results[beta], c=color_refs[beta].get_color(), label=str(beta))
    axes[1][2].legend()

plt.show()
