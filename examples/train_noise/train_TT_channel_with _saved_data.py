## Example of training the TT channel with data that can be downloaded from Kaggle
## from 

# %%
import torch
from scipy import stats
from sbi.utils import process_prior
from sbi import utils as utils
from torch.distributions.normal import Normal
import sys
import h5py
from tqdm import tqdm
import numpy as np
sys.path.append("/home/zaldivar/Documents/Androniki/Github/GWBackFinder_python")
from src.GWBackFinder import train as noise_train

# %% 
## Define prior

prior_c = Normal(torch.tensor([15.]), torch.tensor([0.2*15]))
prior, *_ = utils.process_prior(prior_c)  

# %% 
## load the parameters using the correct path
z=np.load("/data/users/Androniki/Dani/z_noise.npy")
gw_total_noise=[]
for i in tqdm(range(1,430000)):
    f = h5py.File("/data/users/Androniki/Dani_new_noise/"+str(i-1)+".jld2", "r")    
    gw_total_noise.append(np.array(f["data"]))
    
## convert to tensor 
gw_total_noise=torch.tensor(np.array(gw_total_noise) , dtype=torch.float32)
thetas=torch.tensor(z[0:429999])

# %%
print(gw_total_noise.shape)
print(thetas.shape)
# %%
## train
noise_train.train(thetas=thetas, gw_total=gw_total_noise, prior=prior, resume_training=False, validation_fraction=0.2, 
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=200, 
          path_saved=None, path_inference="/data/users/Androniki/", name_file="train_200_noise.pkl", model_type="nsf", hidden_features=64, num_transforms=3)
# %%
## get posterior
noise_train.get_posterior("/data/users/Androniki/train_200_noise.pkl","posterior_noise.pkl")# %%