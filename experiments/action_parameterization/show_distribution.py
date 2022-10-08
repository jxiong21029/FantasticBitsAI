import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions

concentrations = [0.01, 0.03, 0.1, 0.3]
theta = np.linspace(0, 2 * np.pi, 1000)

matplotlib.use("Tkagg")
fig, axes = plt.subplots(
    ncols=len(concentrations),
    subplot_kw={"projection": "polar"},
    constrained_layout=True,
)
for i in range(len(concentrations)):
    distr = distributions.VonMises(loc=0, concentration=concentrations[i])
    r = torch.exp(distr.log_prob(torch.tensor(theta))).numpy()
    axes[i].set_title(f"concentration={concentrations[i]}")
    axes[i].plot(theta, r)
plt.show()
