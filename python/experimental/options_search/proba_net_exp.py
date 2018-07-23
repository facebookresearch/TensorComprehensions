import numpy as np
import ipdb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import ipdb

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.affine1 = nn.Linear(1, 128)
        self.affine2 = nn.Linear(128, 1)
        self.affine3 = nn.Linear(1, 128)
        self.affine4 = nn.Linear(128, 1)
    
    def pred(self, mu, sigma, x):
        diff = mu-x
        diff = diff*diff
        diff = -diff/(2*sigma*sigma+0.0000001)
        diff = torch.exp(diff)
        return diff/(sigma * np.sqrt(2.*np.pi))

    def forward(self, x):
        tmp1 = F.relu(self.affine1(x))
        tmp2 = F.relu(self.affine3(x))
        mu = self.affine2(tmp1)
        sigma = self.affine4(tmp2)
        return self.pred(mu, sigma, x)

predictor = Predictor()
optimizer = optim.Adam(predictor.parameters(), lr=1e-3)

def f(x):
    return x + 0.1*np.random.rand()

Xtrain = torch.rand(500)
Ytrain = f(Xtrain)

dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

nb_epochs=100

for epoch in range(nb_epochs):
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        predictor.zero_grad()
        #ipdb.set_trace()
        out = predictor(inputs.unsqueeze(1))
        loss = -torch.mean(torch.log(out+0.000000001))
        print(loss)
        loss.backward()
        optimizer.step()

