import footsteps
import torch
atlas = torch.load("../results/mean_initialized-1/atlas.torch")

import matplotlib.pyplot as plt

for sl in range(0, 100, 10):
    plt.imshow(atlas[0, 0, sl].detach())
    plt.savefig(footsteps.output_dir + f"{sl:03}.png")
    plt.clf
