# %%
from multiprocessing import cpu_count
os.environ["JULIA_NUM_THREADS"] = str(cpu_count())
from juliacall import Main as jl
jl.Pkg.activate("/home/zaldivar/Documents/Androniki/Github/GWBackFinder2.jl")
jl.seval("using GWBackFinder")
from tqdm import tqdm
import sys
import numpy as np
sys.path.append("/home/zaldivar/Documents/Androniki/Github/GWBackFinder2")
from src.GWBackFinder import train as GW_train
from src.GWBackFinder import prior as GW_prior
import torch
import h5py

# %%
custom_prior=GW_prior.get_prior()

#z=custom_prior.sample(1000000)
#z.shape
#z=np.save("/data/users/Androniki/Dani/z_new.npy",z)
z=np.load("/data/users/Androniki/Dani/z_new.npy")

# %%
f=jl.range(3*1e-5, 0.5, step=1e-6)
idx,idx27,logbins_27,logbins,f_filtered = jl.GWBackFinder.binning(f)
Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3, Sb26_new3, Sb27_new3 = logbins_27[0],logbins_27[1], logbins_27[2], logbins_27[3], logbins_27[4], logbins_27[5], logbins_27[6], logbins_27[7], logbins_27[8], logbins_27[9], logbins_27[10], logbins_27[11], logbins_27[12], logbins_27[13], logbins_27[14], logbins_27[15], logbins_27[16], logbins_27[17], logbins_27[18], logbins_27[19], logbins_27[20], logbins_27[21], logbins_27[22], logbins_27[23], logbins_27[24], logbins_27[25], logbins_27[26]

##sample from prior
#z=prior.sample(1000000)
## load prior
#z=np.load("/data/users/Androniki/Dani/z.npy")
#z=jl.Vector(z)
#Data_total = jl.GWBackFinder.model(np.array(z[100, :]),f,idx,f_filtered,logbins_27,logbins, Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3, Sb26_new3)
#np.save("/data/users/Androniki/Dani/z.npy",z)
# %%

'''
# %%
for i in tqdm(range(len(z))):
    Data_total = jl.GWBackFinder.model(np.array(z[i, :]),f,idx,f_filtered,logbins_27,logbins, Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3, Sb26_new3)

    np.save("/data/users/Androniki/Dani/"+str(i)+".npy",Data_total)
# %%


for i in tqdm(range(len(z))):
    np.save("/data/users/Androniki/Dani/"+str(i)+".npy",z)
# %%
gw_total
'''
# %%


gw_total_list=[]
for i in tqdm(range(1,430000)):
    f = h5py.File("/data/users/Androniki/Dani_new/"+str(i-1)+".jld2", "r")    
    gw_total_list.append(np.array(f["data"]))
    gw_total_list
    
## convert to tensor 
gw_total=torch.tensor(np.array(gw_total_list) , dtype=torch.float32)# %%
thetas=torch.tensor(z[0:429999])
# %%
print(gw_total.shape)
print(thetas.shape)

# %%
## train
GW_train.train(thetas=thetas, gw_total=gw_total, prior=custom_prior, resume_training=False, validation_fraction=0.2, 
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=200, name_file="train_200", 
          path=None, model_type="nsf", hidden_features=64, num_transforms=3)
# %%
## get posterior
GW_train.get_posterior("/data/users/Androniki/train_200.pkl","posterior")
