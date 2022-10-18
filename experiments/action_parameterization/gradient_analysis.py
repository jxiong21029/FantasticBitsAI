import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions
import tqdm

theta = np.linspace(0, 2 * np.pi, 1000)
concentrations = [0.01, 0.1, 1, 10]

r_loc = [np.zeros_like(theta) for _ in range(len(concentrations))]
r_con = [np.zeros_like(theta) for _ in range(len(concentrations))]

for j in range(len(concentrations)):
    for i, t in tqdm.tqdm(enumerate(theta)):
        loc = torch.tensor(0.0, requires_grad=True)
        con = torch.tensor(float(concentrations[j]), requires_grad=True)
        distr = distributions.VonMises(loc=loc, concentration=con)

        (distr.log_prob(torch.tensor(t))).backward()
        r_loc[j][i] = loc.grad.item()
        r_con[j][i] = con.grad.item()

matplotlib.use("TkAgg")
fig, axes = plt.subplots(
    nrows=2,
    ncols=len(concentrations),
    subplot_kw={"projection": "polar"},
    constrained_layout=True,
)

for j, con in enumerate(concentrations):
    axes[0][j].set_title(f"grad of loc, con={con}")
    axes[0][j].plot(theta[r_loc[j] >= 0], r_loc[j][r_loc[j] >= 0], c="green")
    axes[0][j].plot(theta[r_loc[j] < 0], -r_loc[j][r_loc[j] < 0], c="red")

    boundary = min(range(len(theta)), key=lambda idx: abs(r_con[j][idx]))
    axes[1][j].plot(
        [theta[boundary], theta[boundary]], [0, max(np.abs(r_con[j]))], c="grey"
    )
    axes[1][j].plot(
        [-theta[boundary], -theta[boundary]], [0, max(np.abs(r_con[j]))], c="grey"
    )

    axes[1][j].set_title(f"grad of con, con={con}")
    axes[1][j].plot(theta[r_con[j] >= 0], r_con[j][r_con[j] >= 0], c="green")
    axes[1][j].plot(theta[r_con[j] < 0], -r_con[j][r_con[j] < 0], c="red")


plt.show()
