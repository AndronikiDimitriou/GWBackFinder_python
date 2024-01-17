#%%
import sys
import numpy as np
sys.path.append("/home/zaldivar/Documents/Androniki/Github/GWBackFinder_python")
from src.GWBackFinder import prior as GW_prior
from src.GWBackFinder import train as GW_train
import torch
import h5py
import os
from tqdm import tqdm
basepath="."
custom_prior=GW_prior.get_prior()
thetas=custom_prior.sample(100)
thetas_numpy=thetas.numpy()

# Create or open an HDF5 file in write mode
# %%
dataset_name = "thetas"
with h5py.File(basepath+"/parameters.h5", "w") as file:
    # Create a dataset (replace "dataset_name" and data with your actual dataset name and content)
    data = thetas_numpy
    # Save the data to the HDF5 file
    file.create_dataset(dataset_name, data=data)

# %%
# use os to run the julia script

os.system(f"julia -t 5 {basepath}/data_generation_AA_channel.jl -i parameters.h5 -o data_signal -d {dataset_name} -f {len(thetas_numpy)}")

#%%
gw_total_list=[]
for i in tqdm(range(1,len(thetas_numpy)+1)):
    f = h5py.File(basepath+"/data_gen/"+str(i-1)+".jld2", "r")    
    gw_total_list.append(np.array(f["data"]))
  
#%%      
## convert to tensor 
gw_total=torch.tensor(np.array(gw_total_list) , dtype=torch.float32)

print(thetas.shape)
print(gw_total.shape)

# %%

## train and save inference in the file train_200.pkl
GW_train.train(thetas=thetas, gw_total=gw_total, prior=custom_prior, resume_training=False, validation_fraction=0.2, 
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=200, 
          path_saved=None, path_inference=basepath+"/", name_file="train_200_11.pkl", model_type="nsf", hidden_features=64, num_transforms=3)

# %%
## get posterior from the train_200.pkl and save it in posterior.pkl
GW_train.get_posterior(basepath+"/"+"train_200_11.pkl","posterior.pkl")

# %%
