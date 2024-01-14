#%%
import sys
sys.path.append("/home/zaldivar/Documents/Androniki/Github/GWBackFinder_python")
import numpy as np
from juliacall import Main as jl
#import os
#import os.path
##jl.Pkg.add(url="https://github.com/AndronikiDimitriou/GWBackFinder.jl")
#jl.seval("using GWBackFinder")
jl.Pkg.activate("/home/zaldivar/Documents/Androniki/Github/GWBackFinder.jl")
jl.seval("using GWBackFinder")
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import pickle
import matplotlib.pyplot as plt
from sbi.analysis import check_sbc, run_sbc, get_nltp, sbc_rank_plot
from statistics import mode
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils import process_prior
from src.GWBackFinder import prior as GW_prior
from src.GWBackFinder import plot_signal as pl

#%%
f=jl.range(3*1e-5, 0.5, step=1e-6)
idx,idx26,logbins_26,logbins,f_filtered = jl.GWBackFinder.binning(f)
Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3, Sb26_new3 = logbins_27[0],logbins_27[1], logbins_27[2], logbins_27[3], logbins_27[4], logbins_27[5], logbins_27[6], logbins_27[7], logbins_27[8], logbins_27[9], logbins_27[10], logbins_27[11], logbins_27[12], logbins_27[13], logbins_27[14], logbins_27[15], logbins_27[16], logbins_27[17], logbins_27[18], logbins_27[19], logbins_27[20], logbins_27[21], logbins_27[22], logbins_27[23], logbins_27[24], logbins_27[25]
#%%
len(f_filtered)
print(idx27)
#%%
print(idx)
z_power_law_test         =  jl.GWBackFinder.zPowerLaw(np.array([-12.,0.6]))
z_peak_test              =  jl.GWBackFinder.zPeak(np.array([-9.,0.002,0.2]))
z_wiggly_test            =  jl.GWBackFinder.zWiggly(np.array([-10,2,0.1]))
z_Broken_powerlaw_test   =  jl.GWBackFinder.zBroken_powerlaw(np.array([-11,-1,2/3,0.002]))
z_Double_peaks_test      =  jl.GWBackFinder.zDouble_peaks(np.array([-11.,-10.,0.001,0.01,0.1,0.1]))
z_three_peaks_test       =  jl.GWBackFinder.zThree_peaks(np.array([-10.,-10.,-10.,5*10**(-4),2*10**(-3),8*10**(-3),0.1,0.1,0.1]))
z_noise                  =  jl.GWBackFinder.zNoise(np.array([15.,3.]))

#%%
plt.figure()
Data_A,Data_T,f_total_A,f_total_T = jl.GWBackFinder.different_signals(z_power_law_test,z_noise,f,f_filtered,logbins,idx)
plt.plot(f_total_A, Data_A[0:3970])
plt.plot(f_total_T, Data_T[0:3970])
plt.show()
#%%
plt.figure()
Data_A,Data_T,f_total_A,f_total_T= jl.GWBackFinder.different_signals(z_peak_test,z_noise,f,f_filtered,logbins,idx)
plt.plot(f_total_A, Data_A[0:3970])
plt.plot(f_total_T, Data_T[0:3970])
plt.show()
#%%
plt.figure()
Data_A,Data_T,f_total_A,f_total_T = jl.GWBackFinder.different_signals(z_wiggly_test,z_noise,f,f_filtered,logbins,idx)
plt.plot(f_total_A, Data_A[0:3970])
plt.plot(f_total_T, Data_T[0:3970])
plt.show()
#%%
plt.figure()
Data_A,Data_T,f_total_A,f_total_T= jl.GWBackFinder.different_signals(z_Broken_powerlaw_test,z_noise,f,f_filtered,logbins,idx)
plt.plot(f_total_A, Data_A[0:3970])
plt.plot(f_total_T, Data_T[0:3970])
plt.show()
#%%
plt.figure()
Data_A,Data_T,f_total_A,f_total_T  = jl.GWBackFinder.different_signals(z_Double_peaks_test,z_noise,f,f_filtered,logbins,idx)
plt.plot(f_total_A, Data_A[0:3970])
plt.plot(f_total_T, Data_T[0:3970])
plt.show()
#%%
plt.figure()
Data_A,Data_T,f_total_A,f_total_T  = jl.GWBackFinder.different_signals(z_three_peaks_test,z_noise,f,f_filtered,logbins,idx)
plt.plot(f_total_A, Data_A[0:3970])
plt.plot(f_total_T, Data_T[0:3970])
plt.show()
#%%
custom_prior=GW_prior.get_prior()

#%%
with open("/home/zaldivar/Documents/Androniki/Github/GWBackFinder_python/examples/train_signal/data_signal/posterior.pkl", "rb") as handle:
    posterior= pickle.load(handle)
    
with open("/home/zaldivar/Documents/Androniki/Github/GWBackFinder_python/examples/train_noise/data_noise/posterior_noise.pkl", "rb") as handle:
    posterior_noise= pickle.load(handle)
# %%

##RECONSTRUCTION##


z_power_law_test         =  jl.GWBackFinder.zPowerLaw(np.array([-10.,0.6]))
z_noise                  =  jl.GWBackFinder.zNoise(np.array([13.,2.]))

Data_A,Data_T,f_total_A,f_total_T = jl.GWBackFinder.different_signals(z_power_law_test,z_noise,f,f_filtered,logbins,idx)
# %%
plt.plot(f_total_A, Data_A[0:3970])
# %%
# %%
nsamples=500
samples_TT=posterior_noise.sample((nsamples,), x=Data_T[:2970])
# %%
print(samples_TT.shape)
plt.hist(samples_TT.reshape(500),bins=200)
plt.xlim(12.5,13.5)
plt.axvline(13)
# %%
samples=np.array([posterior.sample((1,), x=np.concatenate((Data_A[:3970],samples_TT[i]))).reshape(28) for i in range(nsamples)])
# %%
##convert the samples to their real values, everything here is from (0,1) apart from the z3 and z4 which are the noise parameters and amplitude for galactic
l=np.arange(26)
z=samples[:,l]
z=-12+z*(12+12)
z2=samples[:,26]
z2=-14+z2*(-6-(-14))
z3=samples[:,27]

# %%
plt.hist(z3)
# %%
#exp=[jl.GWBackFinder.f26(fi, Sb13_new3, z[:,0].reshape(len(z[:,0]),1),z[:,1].reshape(len(z[:,0]),1),z[:,2].reshape(len(z[:,0]),1),z[:,3].reshape(len(z[:,0]),1),z[:,4].reshape(len(z[:,0]),1),z[:,5].reshape(len(z[:,0]),1),z[:,6].reshape(len(z[:,0]),1),z[:,7].reshape(len(z[:,0]),1),z[:,8].reshape(len(z[:,0]),1),z[:,9].reshape(len(z[:,0]),1),z[:,10].reshape(len(z[:,0]),1),z[:,11].reshape(len(z[:,0]),1),z[:,12].reshape(len(z[:,0]),1),z[:,13].reshape(len(z[:,0]),1),z[:,14].reshape(len(z[:,0]),1),z[:,15].reshape(len(z[:,0]),1),z[:,16].reshape(len(z[:,0]),1),z[:,17].reshape(len(z[:,0]),1),z[:,18].reshape(len(z[:,0]),1),z[:,19].reshape(len(z[:,0]),1),z[:,20].reshape(len(z[:,0]),1),z[:,21].reshape(len(z[:,0]),1),z[:,22].reshape(len(z[:,0]),1),z[:,23].reshape(len(z[:,0]),1),z[:,24].reshape(len(z[:,0]),1),z[:,25].reshape(len(z[:,0]),1), 10**(z2.reshape(len(z[:,0]),1)), Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3).reshape((500)) for fi in f]
exp=[jl.GWBackFinder.f26(fi, Sb13_new3, z[:,0].mean(),z[:,1].mean(),z[:,2].mean(),z[:,3].mean(),z[:,4].mean(),z[:,5].mean(),z[:,6].mean(),z[:,7].mean(),z[:,8].mean(),z[:,9].mean(),z[:,10].mean(),z[:,11].mean(),z[:,12].mean(),z[:,13].mean(),z[:,14].mean(),z[:,15].mean(),z[:,16].mean(),z[:,17].mean(),z[:,18].mean(),z[:,19].mean(),z[:,20].mean(),z[:,21].mean(),z[:,22].mean(),z[:,23].mean(),z[:,24].mean(),z[:,25].mean(), 10**(z2.mean()), Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3) for fi in f]
    
# %%
##find the 68% etc hdi for the posteriors
int2low, int1low, int0low, int2high, int1high, int0high=pl.bounds(z,z2)

# %%
exp2_high=[jl.GWBackFinder.f26(fi, Sb13_new3, int2high[0],int2high[1],int2high[2],int2high[3],int2high[4],int2high[5],int2high[6],int2high[7],int2high[8],int2high[9],int2high[10],int2high[11],int2high[12],int2high[13],int2high[14],int2high[15],int2high[16],int2high[17],int2high[18],int2high[19],int2high[20],int2high[21],int2high[22],int2high[23],int2high[24],int2high[25],10**(int2high[26]), Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3) for fi in f]
exp2_low=[jl.GWBackFinder.f26(fi, Sb13_new3, int2low[0],int2low[1],int2low[2],int2low[3],int2low[4],int2low[5],int2low[6],int2low[7],int2low[8],int2low[9],int2low[10],int2low[11],int2low[12],int2low[13],int2low[14],int2low[15],int2low[16],int2low[17],int2low[18],int2low[19],int2low[20],int2low[21],int2low[22],int2low[23],int2low[24],int2low[25],10**(int2low[26]), Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3) for fi in f]
exp1_low =[jl.GWBackFinder.f26(fi, Sb13_new3, int1low[0] ,int1low[1] ,int1low[2] ,int1low[3] ,int1low[4], int1low[5], int1low[6], int1low[7], int1low[8], int1low[9], int2low[10], int1low[11], int1low[12],int1low[13],int1low[14],int1low[15],int1low[16],int1low[17],int1low[18],int1low[19],int1low[20],int1low[21],int1low[22],int1low[23],int1low[24],int1low[25],10**(int1low[26]), Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3) for fi in f]
exp1_high=[jl.GWBackFinder.f26(fi, Sb13_new3, int1high[0],int1high[1],int1high[2],int1high[3],int1high[4],int1high[5],int1high[6],int1high[7],int1high[8],int1high[9],int1high[10],int1high[11],int1high[12],int1high[13],int1high[14],int1high[15],int1high[16],int1high[17],int1high[18],int1high[19],int1high[20],int1high[21],int1high[22],int1high[23],int1high[24],int1high[25],10**(int1high[26]), Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3) for fi in f]
exp0_low =[jl.GWBackFinder.f26(fi, Sb13_new3, int0low[0] ,int0low[1] ,int0low[2] ,int0low[3] ,int0low[4], int0low[5], int0low[6], int0low[7], int0low[8], int0low[9], int0low[10], int0low[11], int0low[12],int0low[13],int0low[14],int0low[15],int0low[16],int0low[17],int0low[18],int0low[19],int0low[20],int0low[21],int0low[22],int0low[23],int0low[24],int0low[25],10**(int0low[26]), Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3) for fi in f]
exp0_high=[jl.GWBackFinder.f26(fi, Sb13_new3,int0high[0],int0high[1],int0high[2],int0high[3],int0high[4],int0high[5],int0high[6],int0high[7],int0high[8],int0high[9],int0high[10],int0high[11],int0high[12],int0high[13],int0high[14],int0high[15],int0high[16],int0high[17],int0high[18],int0high[19],int0high[20],int0high[21],int0high[22],int0high[23],int0high[24],int0high[25],10**(int0high[26]), Sb1_new3, Sb2_new3, Sb3_new3, Sb4_new3, Sb5_new3, Sb6_new3, Sb7_new3, Sb8_new3, Sb9_new3, Sb10_new3, Sb11_new3, Sb12_new3, Sb13_new3, Sb14_new3, Sb15_new3, Sb16_new3, Sb17_new3, Sb18_new3, Sb19_new3, Sb20_new3, Sb21_new3, Sb22_new3, Sb23_new3, Sb24_new3, Sb25_new3) for fi in f]
# %%
L=2.5e9
c=3.e8
P=13
A=2
h2Omega_nn_TT = [jl.GWBackFinder.Omega_noiseh2_TT(fi,L,c,P,A) for fi in f]
h2Omega_nn_AA = [jl.GWBackFinder.Omega_noiseh2_AA(fi,L,c,P,A) for fi in f]
# %%
# %%
pls = np.genfromtxt('plis_LISA_new_noise.dat')
fig, ax = plt.subplots()
ax.tick_params(labelsize=11)
plt.xscale('log')
plt.yscale('log')

plt.loglog((10**pls[:,0][15:]),(10**pls[:,1][15:]), color='black',
           linestyle='dashed',label='PLS')
plt.plot(f,h2Omega_nn_TT,label='TT_channel',ls='--')

plt.plot(f,h2Omega_nn_AA,color='green',label='central noise',ls='--')
plt.plot(f,10**(-10)*(np.array(f)/0.001)**0.6,color='red',label='theoretical signal',ls='--')
plt.plot(f,exp,color='blue',label='reconstructed signal')
plt.fill_between(f,np.array(exp2_low),np.array(exp2_high),alpha=0.7,color= "#838B8B",label='68% CI')
plt.fill_between(f,np.array(exp1_low),np.array(exp1_high),alpha=0.5,color= "#808080",label='95% CI')
plt.xlabel('Frequency [Hz]',fontsize=12)
plt.ylabel(r'$h^2\Omega$',fontsize=12)
plt.legend(loc='lower right')
#plt.ylim(10**(-13),10**(-7))
# %%
