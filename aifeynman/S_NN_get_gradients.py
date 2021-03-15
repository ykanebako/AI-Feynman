# SAve a file with 2*(n-1) columns contaning the (n-1) independent variables and the (n-1) gradients of the trained NN with respect these variables

import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
is_cuda = torch.cuda.is_available()

def evaluate_derivatives(pathdir, filename, model, device):
    try:
        data = np.loadtxt(pathdir+filename)[:,0:-1]
        pts = np.loadtxt(pathdir+filename)[:,0:-1]
        pts = torch.tensor(pts)
        pts = pts.clone().detach()
        is_cuda = torch.cuda.is_available()
        grad_weights = torch.ones(pts.shape[0], 1)

        pts = pts.float().to(device)
        model = model.to(device)
        grad_weights = grad_weights.to(device)

        pts.requires_grad_(True)
        outs = model(pts)
        grad = torch.autograd.grad(outs, pts, grad_outputs=grad_weights, create_graph=True)[0]
        save_grads = grad.detach().data.cpu().numpy()
        save_data = np.column_stack((data,save_grads))
        np.savetxt("results/gradients_comp_%s.txt" %filename,save_data)
        return 1
    except:
        return 0
