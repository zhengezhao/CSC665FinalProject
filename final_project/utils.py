import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os.path
import numpy.linalg as linalg
import os
from scipy.stats import ortho_group
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm
from sklearn.decomposition import PCA
from matplotlib import colors
import matplotlib.gridspec as gridspec
import sys
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from sklearn.cluster.bicluster import SpectralBiclustering


def plotConfusionMatrices(file):
    data = np.load(file)
    range_list = list(range(1,11))
    range_list += [20,30,40,50,100]


    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(6,10))
    for i,ax in enumerate(axes.flat):
        matrix = data[i]
        im = ax.imshow(matrix,aspect='auto',cmap = plt.cm.Greys, vmin =0, vmax=1)
        ax.xaxis.set_ticks(np.arange(0,4,1))
        ax.yaxis.set_ticks(np.arange(0,4,1))
        ax.set_xticklabels(['c1','c2','f1','o'])
        ax.set_yticklabels(['c1','c2','f1','o'])


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.savefig(file[:-4]+'.png')







if __name__ == '__main__':
    # plotConfusionMatrices("c1_layer_sim.npy")
    # plotConfusionMatrices("c2_layer_sim.npy")
    for i in range(6):
        plotConfusionMatrices("nn{0}_sim.npy".format(i+1))


