import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import createModel as createModel
import os
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from skimage import color
import torch.nn as nn
import torch.nn.functional as F
from math import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import itertools
import operator
from sklearn.metrics.cluster import adjusted_rand_score


def most_common(L):
  # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
  # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

classes = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')

cwd = os.getcwd()

true_label= np.load(os.path.join(cwd,'true_label.npy'))

full_data_path ='/Users/zhengezhao/Desktop/ArizonaPhDstudy/DeepCompare/picture_wise_vis/static/data/fashion-mnist/data_full_layer'

#print(true_label)

def clustering(nn_idx, epoch,layer):
    layer_data = np.load(os.path.join(full_data_path,'data_nn{0}_epoch{1}.npy'.format(nn_idx,epoch)))
    predictions = np.array(layer_data[1])

    #print(predictions)


    data_matrix = np.array(layer_data[0][layer])
    # print(data_matrix.shape)

    data_matrix = data_matrix.reshape(1000,-1)
    #tsne = TSNE(n_jobs=6,n_components=2)
    pca = PCA(n_components=10)
    trans_data = pca.fit_transform(data_matrix)
    kmeans=  KMeans(n_clusters=10, random_state=0).fit(trans_data)

    labels = kmeans.labels_
    labels_indices = np.argsort(labels)

    #print(labels)

    bin_counts = np.insert(np.cumsum(np.bincount(labels)) ,0,0)
    #print(bin_counts)
    my_predictions = []
    for i in range(10):

        cluster_array = true_label[labels_indices[bin_counts[i]:bin_counts[i+1]]]
        #print(labels[labels_indices[bin_counts[i]:bin_counts[i+1]]])
        common_label = most_common(cluster_array)
        #print(common_label)
        for j in range(len(cluster_array)):
            my_predictions.append(common_label)

    #print(adjusted_rand_score(my_predictions,true_label[labels_indices]))

    unsorted_list = np.zeros(1000)

    for i,value in enumerate(labels_indices):
        unsorted_list[value] = int(my_predictions[i])

    #print(unsorted_list)
    print(adjusted_rand_score(true_label,unsorted_list))

    return unsorted_list



    # predictions[labels_indices]
    # cum_counts = np.cumsum()
    # for i in range(10):

    # print(labels_indices)
    # print(np.bincount(labels[labels_indices]))

def nn_similarity(layer):
    l = layer
    o_data_results = []
    range_list = list(range(1,11))
    range_list += [20,30,40,50,100]

    for epoch in range_list:
        similarity_matrix = [[0.0]*6 for i in range(6)]
        for nn_1 in range(6):
            a = clustering(nn_1+1,epoch,l)
            for nn_2 in range(nn_1,6):
                b= clustering(nn_2+1,epoch,l)
                score = adjusted_rand_score(a,b)
                similarity_matrix[nn_1][nn_2] = score
                similarity_matrix[nn_2][nn_1] = score
        o_data_results.append(similarity_matrix)


    np.save('{0}_layer_sim.npy'.format(l),o_data_results)

def within_nn(nn_idx):
    o_data_results = []
    layers = ['c1','c2','f1','o']
    range_list = list(range(1,11))
    range_list += [20,30,40,50,100]
    for epoch in range_list:
        similarity_matrix = [[0.0]*4 for i in range(4)]
        for l_1 in range(len(layers)):
            a = clustering(nn_idx,epoch,layers[l_1])
            for l_2 in range(l_1,len(layers)):
                b= clustering(nn_idx,epoch,layers[l_2])
                score = adjusted_rand_score(a,b)
                similarity_matrix[l_1][l_2] = score
                similarity_matrix[l_2][l_1] = score
        o_data_results.append(similarity_matrix)


    np.save('nn{0}_sim.npy'.format(nn_idx),o_data_results)





if __name__ == '__main__':
    for i in range(1,7):
        within_nn(i)







