# %%
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

# %%
## load the prior and save inference in the file train_400.pkl
custom_prior=GW_prior.get_prior()

## resume training 
GW_train.train(thetas=None, gw_total=None, prior=custom_prior, resume_training=True, validation_fraction=0.2, 
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=400, 
          path_saved="/data/users/Androniki/train_200.pkl", path_inference="/data/users/Androniki/", name_file="train_400.pkl", 
          model_type="nsf", hidden_features=64, num_transforms=3)

# %%
## get posterior from the train_400.pkl and save it in posterior.pkl
GW_train.get_posterior("/data/users/Androniki/train_400.pkl","posterior")



# %%
