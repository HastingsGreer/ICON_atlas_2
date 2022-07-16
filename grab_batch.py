import torch

knees = torch.load("/playpen-ssd/tgreer/knees_big_2xdownsample_train_set")
import random
random.seed(99)
atlas_sample = random.choices(knees, k=64)

import footsteps

torch.save(atlas_sample, footsteps.output_dir + "atlas_sample.pytorch")

