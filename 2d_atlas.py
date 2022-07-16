#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/HastingsGreer/ICON_atlas_2/blob/master/2d_atlas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[26]:


get_ipython().system('pip install --no-dependencies git+https://github.com/uncbiag/icon')


# In[27]:


import icon_registration.data as data
import icon_registration.networks as networks
import icon_registration.visualize as visualize
import icon_registration as icon


# In[28]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import pickle
from IPython.display import Image, display
import torch.nn.functional as F


# In[29]:


batch_size = 64
d1, d2 = data.get_dataset_triangles("train", hollow=True)
d1_t, d2_t = data.get_dataset_triangles("test", hollow=True)

lmbda = .2


# In[30]:


inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=2))
for _ in range(3):
  inner_net = icon.TwoStepRegistration(icon.DownsampleRegistration(inner_net, 2), icon.FunctionFromVectorField(networks.tallUNet2(dimension=2)))

netGrad = icon.GradientICON(
    inner_net,
    # Our image similarity metric. The last channel of x and y is whether the value is interpolated or extrapolated, 
    # which is used by some metrics but not this one
    #inverseConsistentNet.LNCC(sigma=5),
    icon.ssd,
    lmbda,
)


# In[31]:


input_shape = next(iter(d1))[0].size()
netGrad.assign_identity_map(input_shape)
netGrad.cuda()
optimizerGrad = torch.optim.Adam(netGrad.parameters(), lr=0.001)
netGrad.train()
0


# In[32]:


y_grad = np.array(icon.train_datasets(netGrad, optimizerGrad, d1, d2, epochs=12))


# In[33]:


plt.plot(y_grad[:, :3])
#plt.plot(y_001[:, :3])


# In[34]:


y_grad


# In[35]:


# First registration done all blurry

image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
with torch.no_grad():
  for N in range(3):
      visualize.visualizeRegistration(
          netGrad,
          image_A,
          image_B,
          N,
          f"test{N}.png"
      )
      display(Image(f"test{N}.png"))


# 

# In[36]:


batch = next(iter(d1))[0].cuda()


# In[37]:


atlas = torch.nn.Parameter(torch.randn(1, 1, 128, 128, device="cuda"))
#with torch.no_grad():
#  atlas[:] = torch.mean(batch, axis=0, keepdims=True)

atlas_exp = atlas.expand(128, -1, -1, -1).cuda()
params = [atlas]


# In[38]:


optim = torch.optim.Adam(params, lr=.3)


# In[39]:


for _ in range(730):
  optim.zero_grad()
  loss_obj = netGrad(atlas_exp, batch)
  #(torch.mean((batch - netGrad.warped_image_A)**2))
  loss_obj.all_loss.backward()
  optim.step()
  print(icon.losses.to_floats(loss_obj))


# In[40]:


plt.imshow(atlas[0, 0].cpu().detach())
plt.colorbar()


# In[41]:


plt.imshow(netGrad.warped_image_A[4, 0].detach().cpu())


# In[42]:


plt.imshow(batch[4, 0].detach().cpu())


# In[42]:




