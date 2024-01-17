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
basepath="."
# %%
prior_c = Normal(torch.tensor([15.]), torch.tensor([0.2*15]))
prior, *_ = utils.process_prior(prior_c)  

## resume training 
GW_train.train(thetas=None, gw_total=None, prior=prior, resume_training=True, validation_fraction=0.2, 
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=400,
          path_saved=basepath+"train_200_noise.pkl",path_inference=basepath+"/", name_file="train_400_noise.pkl", model_type="nsf", hidden_features=64, num_transforms=3)

# %%
## get posterior from the train_400.pkl and save it in posterior.pkl
GW_train.get_posterior(basepath+"/train_400_noise.pkl","posterior.pkl")
