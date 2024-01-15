## Example of generating data and training the TT channel
# %%
from multiprocessing import cpu_count
import os
os.environ["JULIA_NUM_THREADS"] = str(cpu_count())
from juliacall import Main as jl
jl.Pkg.activate("/home/zaldivar/Documents/Androniki/Github/GWBackFinder.jl")
jl.seval("using GWBackFinder")
import torch
from sbi import utils as utils
from torch.distributions.normal import Normal
import sys
import h5py
from tqdm import tqdm
import numpy as np
sys.path.append("/home/zaldivar/Documents/Androniki/Github/GWBackFinder_python")
from src.GWBackFinder import train as noise_train

# %%
prior_c = Normal(torch.tensor([15.]), torch.tensor([0.2*15]))
prior, *_ = utils.process_prior(prior_c)  
z=prior.sample((1000,))
#np.save("/data/users/Androniki/Dani/z_noise.npy",z)

# %%
f=jl.range(3*1e-5, 0.5, step=1e-6)
# %%
gw_total_noise=[]
for i in (range(5)):
    Data_total = jl.GWBackFinder.model_noise_train_data(f, z[i].numpy()[0])
    gw_total_noise.append(np.array(Data_total))
    
## convert to tensor 
gw_total_noise=torch.tensor(np.array(gw_total_noise) , dtype=torch.float32)# %%
thetas=torch.tensor(z[0:len(z)])

# %%
print(gw_total_noise.shape)
print(thetas.shape)
# %%
## train
noise_train.train(thetas=thetas, gw_total=gw_total_noise, prior=prior, resume_training=False, validation_fraction=0.2,
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=200, 
          path_saved=None,path_inference="/data/users/Androniki/", name_file="train_200_noise.pkl", model_type="nsf", hidden_features=64, num_transforms=3)

# %%
## get posterior
noise_train.get_posterior("/data/users/Androniki/train_200_noise.pkl","posterior_noise.pkl")# %%


# %%
