"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import matplotlib as mpl  
import matplotlib.pyplot as plt 
plt.style.use('seaborn-darkgrid')

import torch
import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
# def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    # op = partial(torch.einsum, "bixy,ioxy->boxy")
    # return torch.stack([
        # op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        # op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)



class LowRank2d(nn.Module):
    def __init__(self, in_channels, out_channels, s, width, rank):
        super(LowRank2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.n = s*s
        self.rank = rank

        self.phi = DenseNet([3, 64, 128, 256, width*width*rank], torch.nn.ReLU)
        self.psi = DenseNet([3, 64, 128, 256, width*width*rank], torch.nn.ReLU)


    def forward(self, v, a):
        # a (batch, n, 3)
        # v (batch, n, f)
        batch_size = v.shape[0]

        phi_eval = self.phi(a).reshape(batch_size, self.n, self.out_channels, self.in_channels, self.rank)
        psi_eval = self.psi(a).reshape(batch_size, self.n, self.out_channels, self.in_channels, self.rank)

        # print(psi_eval.shape, v.shape, phi_eval.shape)
        v = torch.einsum('bnoir,bni,bmoir->bmo', psi_eval, v, phi_eval) / self.n

        return v



class MyNet(torch.nn.Module):
    def __init__(self, s, width=32, rank=1):
        super(MyNet, self).__init__()
        self.s = s
        self.width = width
        self.rank = rank

        self.fc0 = nn.Linear(3, self.width)

        self.net1 = LowRank2d(self.width, self.width, s, width, rank=self.rank)
        self.net2 = LowRank2d(self.width, self.width, s, width, rank=self.rank)
        self.net3 = LowRank2d(self.width, self.width, s, width, rank=self.rank)
        self.net4 = LowRank2d(self.width, self.width, s, width, rank=self.rank)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)
        self.w4 = nn.Linear(self.width, self.width)

        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)
        self.bn4 = torch.nn.BatchNorm1d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, v):
        batch_size, n = v.shape[0], v.shape[1]
        a = v.clone()

        v = self.fc0(v)

        v1 = self.net1(v, a)
        v2 = self.w1(v)
        v = v1+v2
        v = self.bn1(v.reshape(-1, self.width)).view(batch_size,n,self.width)
        v = F.relu(v)

        v1 = self.net2(v, a)
        v2 = self.w2(v)
        v = v1+v2
        v = self.bn2(v.reshape(-1, self.width)).view(batch_size,n,self.width)
        v = F.relu(v)

        v1 = self.net3(v, a)
        v2 = self.w3(v)
        v = v1+v2
        v = self.bn3(v.reshape(-1, self.width)).view(batch_size,n,self.width)
        v = F.relu(v)

        v1 = self.net4(v, a)
        v2 = self.w4(v)
        v = v1+v2
        v = self.bn4(v.reshape(-1, self.width)).view(batch_size,n,self.width)


        v = self.fc1(v)
        v = F.relu(v)
        v = self.fc2(v)

        return v.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c




TRAIN_PATH = 'piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'piececonst_r421_N1024_smooth2.mat'

ntrain = 1000
ntest = 100

batch_size = 10

r = 5
h = int(((421 - 1)/r) + 1)
s = h

learning_rate = 0.00025

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s].reshape(ntrain,s*s)
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s].reshape(ntrain,s*s)

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s].reshape(ntest,s*s)
y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s].reshape(ntest,s*s)


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s*s,2)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain,s*s,1), grid.repeat(ntrain,1,1)], dim=2)
x_test = torch.cat([x_test.reshape(ntest,s*s,1), grid.repeat(ntest,1,1)], dim=2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

model = MyNet(s).cuda()
# model = MyNet_old(s).cuda()

print(model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
epochs = 20

train_history = []
test_history = []

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s*s)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        mse.backward()

        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        # loss.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s*s)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    # print(ep, t2-t1, train_mse, train_l2, test_l2)
    print(f"epoch: {ep} | time: {t2-t1} | train_l2: {train_mse} | test_l2: {test_l2}")
    train_history.append(train_l2)
    test_history.append(test_l2)

plt.plot(train_history, label="train l2")
plt.plot(test_history, label="test_l2")
plt.legend()

plt.show()

print("\nTRAINING DONE")
print("Saving model to ")

torch.save(model, "darcy2d_trained")


