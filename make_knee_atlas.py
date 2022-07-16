import random

import footsteps
import icon_registration as icon
import itk
import torch
import matplotlib.pyplot as plt
from icon_registration.pretrained_models import OAI_knees_gradICON_model
from icon_registration.train import write_stats

from torch.utils.tensorboard import SummaryWriter
footsteps.initialize()
writer = SummaryWriter(
    footsteps.output_dir + "/" + "loss_curve",
    flush_secs=30,
)

BATCH_SIZE = 4

knee_network = OAI_knees_gradICON_model()

knee_batch = torch.load("results/grab_batch/atlas_sample.pytorch")


atlas = torch.nn.Parameter(torch.randn(tuple(knee_network.input_shape), device="cuda"))

with torch.no_grad():
    atlas[:] = torch.mean(torch.cat(knee_batch), axis=0, keepdims=True)

atlas_exp = atlas.expand(BATCH_SIZE, -1, -1, -1, -1).cuda()
params = [atlas]


optim = torch.optim.Adam(params, lr=0.08)

for step in range(90):
    plt.imshow(atlas_exp[0, 0, 50].cpu().detach().numpy())
    plt.savefig(footsteps.output_dir + "step" + str(step) + ".png")
    plt.clf()
    for _ in range(5):
        batch = torch.cat(random.choices(knee_batch, k=BATCH_SIZE), axis=0).cuda()
        optim.zero_grad()
        loss_obj = knee_network(atlas_exp, batch)
        # (torch.mean((batch - netGrad.warped_image_A)**2))
        loss_obj.similarity_loss.backward()
        optim.step()
        print(icon.losses.to_floats(loss_obj))
        write_stats(writer, loss_obj, 5 * step + _)
torch.save(atlas.cpu(), footsteps.output_dir + "atlas.torch")

itk.imsave(itk.image_from_array(np.array(atlas.detach().cpu())), "atlas.nrrd")
