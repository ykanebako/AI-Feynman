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
from matplotlib import pyplot as plt
import torch.utils.data as utils
import time
import os

bs = 2048
wd = 1e-2


def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom


def NN_train(pathdir, filename, epochs=1000, device='cpu', lrs=1e-2, N_red_lr=4, pretrained_path="", results_path='./'):

    os.makedirs(f"{results_path}/NN_trained_models/models/", exist_ok=True)

    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))

        """
        epochs = 200*n_variables
        if len(variables)<5000:
            epochs = epochs*3
        """

        if n_variables==0 or n_variables==1:
            return 0

        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
                variables = np.column_stack((variables,v))

        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables)
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        product = product.float()

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

        my_dataset = utils.TensorDataset(factors,product) # create your datset
        my_dataloader = utils.DataLoader(my_dataset, batch_size=bs, shuffle=True) # create your dataloader

        model_feynman = SimpleNet(n_variables).to(device)

        if pretrained_path!="":
            model_feynman.load_state_dict(torch.load(pretrained_path))

        check_es_loss = 10000

        for i_i in range(N_red_lr):
            optimizer_feynman = optim.Adam(model_feynman.parameters(), lr = lrs)
            for epoch in range(epochs):
                model_feynman.train()
                for i, data in enumerate(my_dataloader):
                    optimizer_feynman.zero_grad()

                    fct = data[0].float().to(device)
                    prd = data[1].float().to(device)

                    loss = rmse_loss(model_feynman(fct),prd)
                    loss.backward()
                    optimizer_feynman.step()
                if epoch%10 == 0:
                    print(f'lr_reduce:{i_i}/{N_red_lr} \t epoch:{epoch}/{epochs} \t loss:{float(loss)}')

                '''
                # Early stopping
                if epoch%20==0 and epoch>0:
                    if check_es_loss < loss:
                        break
                    else:
                        torch.save(model_feynman.state_dict(), "results/NN_trained_models/models/" + filename + ".h5")
                        check_es_loss = loss
                if epoch==0:
                    if check_es_loss < loss:
                        torch.save(model_feynman.state_dict(), "results/NN_trained_models/models/" + filename + ".h5")
                        check_es_loss = loss
                '''
                torch.save(model_feynman.state_dict(), f"{results_path}/NN_trained_models/models/" + filename + ".h5")
            lrs = lrs/10

        return model_feynman

    except NameError:
        print("Error in file: %s" %filename)
        raise


