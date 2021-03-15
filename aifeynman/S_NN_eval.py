from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
import time


bs = 2048


def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))

    return torch.sqrt(F.mse_loss(pred, targ))/denom


def NN_eval(pathdir, filename, device='cpu', results_path='./'):
    try:
        n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+filename, usecols=(0,))

        if n_variables==0:
            return 0
        elif n_variables==1:
            variables = np.reshape(variables,(len(variables),1))
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+filename, usecols=(j,))
                variables = np.column_stack((variables,v))

        f_dependent = np.loadtxt(pathdir + filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent, (len(f_dependent),1))

        # eval data
        factors_val = torch.from_numpy(variables[int(5*len(variables)/6):int(len(variables))])
        factors_val = factors_val.float()
        product_val = torch.from_numpy(f_dependent[int(5*len(variables)/6):int(len(variables))])
        product_val = product_val.float()


        # eval data to device
        factors_val = factors_val.to(device)
        product_val = product_val.to(device)

        class SimpleNet(nn.Module):
            def __init__(self, ni):
                super().__init__()
                self.linear1 = nn.Linear(ni, 128)
                self.linear2 = nn.Linear(128, 128)
                self.linear3 = nn.Linear(128, 64)
                self.linear4 = nn.Linear(64,64)
                self.linear5 = nn.Linear(64,1)

            def forward(self, x):
                x = F.tanh(self.linear1(x))
                x = F.tanh(self.linear2(x))
                x = F.tanh(self.linear3(x))
                x = F.tanh(self.linear4(x))
                x = self.linear5(x)
                return x

        model.load_state_dict(torch.load(f"{results_path}/NN_trained_models/models/"+filename+".h5"))
        model = model.to(device)
        model.eval()

        return(rmse_loss(model(factors_val), product_val), model)

    except Exception as e:
        print(e)
        return (100,0)





