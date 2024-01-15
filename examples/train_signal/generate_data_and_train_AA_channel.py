## Example of generating data and training the AA channel
# %%
from multiprocessing import cpu_count
import os
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

# %%
# ## load the prior sample 1 million points and save

custom_prior=GW_prior.get_prior()
z=custom_prior.sample(1000000)
#z=np.save("./samples.npy",z)
# %%

##Split in 26 bins.
##shows in which bin the frequency belongs to and
##are the boundaries of the bins . After we coarse grain the frequencies from
##Hz to the maximum frequency
##Hz), i.e., we bin them in 1000 intervals of equal log-spacing. idx shows in which of the 1000 bins the frequency belongs to and logbins the boundaries of the 1000 bins.

f=jl.range(3*1e-5, 0.5, step=1e-6)
idx,idx26,logbins_26,logbins,f_filtered = jl.GWBackFinder.binning(f)
Sb1, Sb2, Sb3, Sb4, Sb5, Sb6, Sb7, Sb8, Sb9, Sb10, Sb11, Sb12, Sb13, Sb14, Sb15, Sb16, Sb17, Sb18, Sb19, Sb20, Sb21, Sb22, Sb23, Sb24, Sb25, Sb26 = logbins_26[0],logbins_26[1], logbins_26[2], logbins_26[3], logbins_26[4], logbins_26[5], logbins_26[6], logbins_26[7], logbins_26[8], logbins_26[9], logbins_26[10], logbins_26[11], logbins_26[12], logbins_26[13], logbins_26[14], logbins_26[15], logbins_26[16], logbins_26[17], logbins_26[18], logbins_26[19], logbins_26[20], logbins_26[21], logbins_26[22], logbins_26[23], logbins_26[24], logbins_26[25]


# %%
gw_total_list=[]
for i in tqdm(range(len(z))):
    z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23, z24, z25, z26 = -12 +z[i,0:26].numpy()*(12-(-12))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    Amp =-14+z[i,26].numpy()*(-6-(-14))
    A = z[i,27].numpy()
    Data_total = jl.GWBackFinder.model_train_data(z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, 
                                                  z16, z17, z18, z19, z20, z21, z22, z23, z24, z25, z26, Amp, A, f, idx,
                                                  f_filtered,logbins, Sb1, Sb2, Sb3, Sb4, Sb5, Sb6, Sb7, Sb8, Sb9, Sb10, 
                                                  Sb11, Sb12, Sb13, Sb14, Sb15, Sb16, Sb17, Sb18, Sb19, Sb20, Sb21, Sb22, Sb23, Sb24, Sb25)
    gw_total_list.append(np.array(Data_total))

## convert to tensor 
gw_total=torch.tensor(np.array(gw_total_list) , dtype=torch.float32)
thetas=torch.tensor(z[0:len(z)])
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
