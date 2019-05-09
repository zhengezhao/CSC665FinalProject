import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os.path
import numpy.linalg as linalg
import os
from sklearn.decomposition import PCA
import createModel as createModel

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

num_of_nn = 200
### every 50 NNs is a cluster

num_of_clusters = 4


cwd = os.getcwd()


model_path =os.path.join(cwd,'model')

full_data_path =os.path.join(cwd,'data_full_layer')






trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

testloader_all = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False, num_workers=4)

testloader_one= torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=1)

trainloader_one = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=False, num_workers=1)

trainloader_all = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                              shuffle=False, num_workers=4)


def SaveTrueLabel():
    for index,data in enumerate(testloader_all): ### just load it once
        images,label = data
        labels  = label.numpy()
        np.save('true_label.npy',labels)



def data_full_layer_vectors():
    iamges= None


    for index,data in enumerate(testloader_all): ### just load it once
        images,_ = data
        break

    for i in range(num_of_nn):
        results=[]
        matrix = []
        predicts= None
        net = createModel.Net()
        net_name = 'net_{}_epoch50'.format(i+1)
        if os.path.isfile(os.path.join(model_path,net_name)):
            net.load_state_dict(torch.load(os.path.join(model_path,net_name), map_location = 'cpu'))
        else:
            print(net_name + " is not trained")

        with torch.no_grad():
            net.eval()
            c1,c2,f1,outputs = net(images)  ##outputs shape : number_of_inputs X 10
            _,predicted = torch.max(outputs,1)
            matrix.append({'f1': f1.numpy(), 'o': outputs.numpy()})
            predicts = predicted.numpy()

            results.append(matrix)
            results.append(predicts)

        np.save(os.path.join(full_data_path,'data_nn{0}_epoch50.npy'.format(i+1)),results)



    return results







if __name__ == '__main__':
    #SaveTrueLabel()
    #data_full_layer_vectors()


