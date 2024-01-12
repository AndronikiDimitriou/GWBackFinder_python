import numpy as np
import arviz as az

def bounds(z,z2):
    int2low=[]
    int1low=[]
    int0low=[]
    int2high=[]
    int1high=[]
    int0high=[]
    for i in range(26):
        int2low.append(az.hdi(z[:,i],0.68268)[0])
        int1low.append(az.hdi(z[:,i],0.95450)[0])
        int0low.append(az.hdi(z[:,i],0.99730)[0])
        int2high.append(az.hdi(z[:,i],0.68268)[1])
        int1high.append(az.hdi(z[:,i],0.95450)[1])
        int0high.append(az.hdi(z[:,i],0.99730)[1])

    int2low.append(az.hdi(z2,0.68268)[0])
    int1low.append(az.hdi(z2,0.95450)[0])
    int0low.append(az.hdi(z2,0.99730)[0])
    int2high.append(az.hdi(z2,0.68268)[1])
    int1high.append(az.hdi(z2,0.95450)[1])
    int0high.append(az.hdi(z2,0.99730)[1])

    return int2low, int1low, int0low, int2high, int1high, int0high