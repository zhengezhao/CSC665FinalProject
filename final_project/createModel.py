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
# from flask import Flask, render_template, request
# from flask_bootstrap import Bootstrap
# from flask_sqlalchemy import SQLAlchemy
import os
from scipy.stats import ortho_group
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
from matplotlib import cm
from sklearn.decomposition import PCA
from matplotlib import colors
# import numpy as np
# import sys
# from numpy.fft import fft
# import matplotlib as plot
# from flask import Flask, render_template, request, url_for, jsonify
# from flask_cors import CORS, cross_origin
# import json

# app = Flask(__name__)



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)



def filter_trainset(trainset,label):
    count = 0
    filtertrainset = []
    for i in range(len(trainset)):
        if trainset[i][1] == label and count <5000:
            count+=1
        else:
            filtertrainset.append(trainset[i])
    return filtertrainset

###prodcue the same results
torch.backends.cudnn.deterministic = True #if we are using GPU
torch.manual_seed(0)
#filter_4 = filter_trainset(trainset,4)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=5)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=5)

classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_of_nn = 2
num_of_epoch=2
std_1 = 0.02
std_2 =0.1
mean = 0.0
w_sigma_s = 0.1
w_sigma_l = 0.9




cwd = os.getcwd()
# path_to_model = os.path.join(cwd,'model_10_epoch_sim')
# path_to_layer_epoch =os.path.join(path_to_model,'layerXepoch')
# path_to_model_sim =os.path.join(path_to_model,'model_within')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean,std_1)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(mean,std_2)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean,std_1)
        m.bias.data.fill_(0)

def weights_init_noise(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        noise = np.random.normal(mean,std_1,m.weight.data.shape)
        m.weight.data = m.weight.data*w_sigma_l+ torch.from_numpy(noise).type(torch.FloatTensor)*w_sigma_s
    elif classname.find('Linear') != -1:
        noise = np.random.normal(mean,std_2,m.weight.data.shape)
        m.weight.data = m.weight.data*w_sigma_l+ torch.from_numpy(noise).type(torch.FloatTensor)*w_sigma_s

def TrainNN(net, net_name):
     # set up the loss function momentum to avoid local optimal
    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss(size_average = False)
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.3)
    net.to(device)

    if os.path.isfile("model_100_epoch/" + net_name):
        print(net_name + "has been trained.")
        return


    # train the network
    for epoch in range(num_of_epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()



            # forward + backward + optimize
            inputs, labels = inputs.to(device), labels.to(device)
            _,_,_,outputs = net(inputs)
            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 125 == 124:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        #save the intermedial model in each epoch
        if epoch != num_of_epoch - 1:
            net_name_epoch = net_name + '_epoch' + str(epoch + 1)
        else:
            net_name_epoch = net_name
        #torch.save(net.state_dict(), "model_100_epoch/" + net_name_epoch)

    print('Finished Training')



# F=1, PAD=0; F=3, PAD=1; F=5, PAD=2; F=7, PAD=3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        a = nn.Conv2d(1, 10, 5)
        layers = [nn.Conv2d(1, 10, 5), nn.MaxPool2d(2), nn.ReLU()]
        self.conv1 = nn.Sequential(*layers)
        layers = [nn.Conv2d(10, 20, 5), nn.Dropout2d(), nn.MaxPool2d(2), nn.ReLU()]
        self.conv2 = nn.Sequential(*layers)
        layers = [nn.Linear(320, 50), nn.ReLU()]
        self.fc1 = nn.Sequential(*layers)
        self.drop = nn.Dropout(p=0.2)
        layers = [nn.Linear(50, 10), nn.ReLU()]
        self.fc2 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        c1 = x
        x = self.conv2(x)
        c2 = x
        x = x.view(-1, 320)
        x = self.fc1(x)
        f1 = x
        x = self.drop(x)
        x = self.fc2(x)
        return c1,c2,f1,x


# this function is used to trained the models and record intermedia model durning each epoch
# training 10 nn using same structure and the same but shuffled data
def createModels():
    if not os.path.isfile('cluster_1_init_state'):
    ##create initializations for two clusters
        net_cluster_1 = Net()
        net_cluster_1.apply(weights_init)
        torch.save(net_cluster_1.state_dict(),'cluster_1_init_state')
    if not os.path.isfile('cluster_2_init_state'):
        net_cluster_2 = Net()
        net_cluster_2.apply(weights_init)
        torch.save(net_cluster_2.state_dict(),'cluster_2_init_state')


    for i in range(1,2):
        for j in range(num_of_nn):
            net = Net()
            net.load_state_dict(torch.load('cluster_'+str(i)+'_init_state', map_location = 'cpu'))
            net.apply(weights_init_noise)
            net_name = 'net_' + str((i-1)*num_of_nn+j+1)
            TrainNN(net, net_name)


if __name__ == '__main__':
    createModels()

