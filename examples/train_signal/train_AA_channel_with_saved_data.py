## Example of training the AA channel with data that can be downloaded from Kaggle
## from 

# %%
import os
from multiprocessing import cpu_count
os.environ["JULIA_NUM_THREADS"] = str(cpu_count())
from juliacall import Main as jl
jl.Pkg.activate("/home/zaldivar/Documents/Androniki/Github/GWBackFinder.jl")
jl.seval("using GWBackFinder")
from tqdm import tqdm
import sys
import numpy as np
sys.path.append("/home/zaldivar/Documents/Androniki/Github/GWBackFinder_python")
from src.GWBackFinder import train as GW_train
from src.GWBackFinder import prior as GW_prior
import torch
import h5py

# %%
## load the prior
custom_prior=GW_prior.get_prior()

# %%
## load the parameters using the correct path
z=np.load("/data/users/Androniki/Dani/z_new.npy")

gw_total_list=[]
for i in tqdm(range(1,430000)):
    f = h5py.File("/data/users/Androniki/Dani_new/"+str(i-1)+".jld2", "r")    
    gw_total_list.append(np.array(f["data"]))
  
# %%      
## convert to tensor 
gw_total=torch.tensor(np.array(gw_total_list) , dtype=torch.float32)
thetas=torch.tensor(z[0:429999])

# %% 
## check shapes
print(gw_total.shape)
print(thetas.shape)

# %%
## train and save inference in the file train_200.pkl
GW_train.train(thetas=thetas, gw_total=gw_total, prior=custom_prior, resume_training=False, validation_fraction=0.2, 
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=200, 
          path_saved=None, path_inference="/data/users/Androniki/", name_file="train_200.pkl", model_type="nsf", hidden_features=64, num_transforms=3)

# %%
## get posterior from the train_200.pkl and save it in posterior.pkl
GW_train.get_posterior("/data/users/Androniki/train_200.pkl","posterior.pkl")

