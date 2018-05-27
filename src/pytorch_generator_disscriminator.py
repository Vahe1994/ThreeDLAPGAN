import numpy as np
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


def sample_noise(N,NOISE_DIM = 128):
    return np.random.normal(size=(N,NOISE_DIM)).astype(np.float32)


class Generator(nn.Module):
    def __init__(self, noise_dim, out_dim, hidden_dim=128):
        super(Generator, self).__init__()
        
          
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.constant(self.fc1.bias, 0.0)

        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal(self.fc2.weight)
        nn.init.constant(self.fc2.bias, 0.0)

        self.bn2 = nn.BatchNorm1d(out_dim)
        

    def forward(self, z):
        """
            Generator takes a vector of noise and produces sample
        """
        h1 = F.relu(self.fc1(z))
        h1 = self.bn1(h1)
        y_gen = self.fc2(h1)
        y_gen = self.bn2(y_gen)
        
        return y_gen


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim=100):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, 256)
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.constant(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(256, 512)
        nn.init.xavier_normal(self.fc2.weight)
        nn.init.constant(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(512, 1)
        nn.init.xavier_normal(self.fc3.weight)
        nn.init.constant(self.fc3.bias, 0.0)
        

    def forward(self, x,TASK =2):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        if TASK == 1 or TASK ==2:
            score = F.sigmoid(self.fc3(h2))
        else:
            score = self.fc3(h2)
        return score
class UnitNormClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data

            w.clamp_(-0.001,0.001)
def g_loss(input_1,TASK = 2):
    # if TASK == 1: 
    #     do something
    if TASK ==1 :
        return torch.mean(torch.log(1-input_1)).cuda()
    if TASK == 2:
        return -torch.mean(torch.log(input_1)).cuda()
    if TASK == 3 or TASK == 4:
        return -torch.mean(input_1)
    return # TODO
def d_loss(input_1,input_2,TASK = 2,penalty = None) :
    if TASK == 1 or TASK == 2: 
        return -torch.mean(torch.log(input_2)).cuda() - torch.mean(torch.log(1-input_1)).cuda()

    if TASK==3 or TASK == 4:
        if(penalty is None):
            return -torch.mean(input_2).cuda() + torch.mean(input_1).cuda()
        else:
            return -torch.mean(input_2).cuda() + torch.mean(input_1).cuda() + penalty.cuda()
################################
def iterate_minibatches(X, batchsize, y=None):
    perm = np.random.permutation(X.shape[0])
    
    for start in range(0, X.shape[0], batchsize):
        end = min(start + batchsize, X.shape[0])
        if y is None:
            yield X[perm[start:end]], y
        else:
            yield X[perm[start:end]], y[perm[start:end]]