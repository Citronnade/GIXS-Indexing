import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import fit_d
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
#from logger import Logger

class SimpleNet(torch.nn.Module):
    """
    SimpleNet is a fairly shallow fully connected network with batch normalization.
    """
    def __init__(self, num_spacings=8):
        super(SimpleNet, self).__init__()
        self.bn1 = nn.BatchNorm1d(350) #batchnorm layers aid in training
        self.bn2 = nn.BatchNorm1d(350)
        self.bn3 = nn.BatchNorm1d(250)
        self.bn4 = nn.BatchNorm1d(100)

        self.linear1 = nn.Linear(num_spacings, 350)
        self.hidden1 = nn.Linear(350, 350)
        self.hidden2 = nn.Linear(350, 250)
        self.hidden3 = nn.Linear(250, 100)
        self.linear2 = nn.Linear(100, 3)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(self.bn1(x))

        x = self.hidden1(x)
        x = F.relu(self.bn2(x))

        x = self.hidden2(x)
        x = F.relu(self.bn3(x))

        x = self.hidden3(x)
        x = F.relu(self.bn4(x))

        x = self.linear2(x)
        return x


class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(1, 50, 2, batch_first=True)
        self.hidden = None
        self.last_linear = nn.Linear(30*50, 3)

    def forward(self, x):
        out, self.hidden = self.lstm(x)
        out = out.contiguous().view(out.shape[0], -1)
        out = self.last_linear(out)
        return out


if __name__ == '__main__':
    noise = 0.01
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20*10, gamma=0.25)
    generator = fit_d.dSpaceGenerator()
    yTr = torch.Tensor(fit_d.gen_input(5000))

    xTr = torch.Tensor(generator(yTr)) #first generate input to scale
    scaler = preprocessing.StandardScaler().fit(xTr) #0-1 normalization is essential
    xTr = torch.Tensor(scaler.transform(xTr))
    dataset = torch.utils.data.TensorDataset(xTr, yTr) # sets up the data for torch
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=96, shuffle=True) #minibatch size of 96


    known_y = np.array([.65, 1.2, np.radians(111)]).reshape(1,-1)
    known_x = generator(known_y)#np.array(fit_d.gen_d_vector(*known_y[0]))

    known_x = scaler.transform(known_x.reshape(1,-1))
    known_x = torch.Tensor(known_x)
    print(known_x)


    for epoch in range(75):
        # The primary training loop
        #1000 iterations in an epoch?
        running_loss = 0
        yTr = torch.Tensor(fit_d.gen_input(15000)) #each loop we generate 15000 training inputs
        xTr = torch.Tensor(generator(yTr))
        #scaler = preprocessing.StandardScaler().fit(xTr)
        xTr = torch.Tensor(scaler.transform(xTr))
        dataset = torch.utils.data.TensorDataset(xTr, yTr)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=192, shuffle=True)
        #hidden = None
        for inputs, labels in dataloader:
            #Iteration through the dataset
            optimizer.zero_grad() # must clear gradients from previous iteration before current

            #inputs = inputs.unsqueeze(-1)
            outputs = model(inputs) # raw logits from the model
            #outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, labels) # get the error
            loss.backward() # backpropagate the loss
            optimizer.step() # update model parameters based off the backpropagated loss
            running_loss += loss.item() # and accumulate the loss so we can track it
        running_loss /= len(dataloader)
        info = {
            'loss_' : running_loss
        }
        print("Epoch {0}: loss {1}".format(epoch, running_loss))
        scheduler.step() #scheduler steps every epoch for learning rate decay

    yTe = torch.Tensor(fit_d.gen_input(1000))
    xTe = torch.Tensor(generator(yTe))
    xTe = torch.Tensor(scaler.transform(xTe))

    test_dataset = torch.utils.data.TensorDataset(xTe, yTe)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)


    torch.save(model.state_dict(), "model.pth") # save model weights to a file to be loaded later
    model.eval() # make sure to set model to eval mode before any predictions--batchnorm has different behavior
    preds = torch.Tensor()
    i = 0
    for inputs, labels in test_dataloader:
        #inputs = inputs.unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        preds = torch.cat((preds, outputs), 0)
        i+=1

    print("average test error:", torch.mean(preds - yTe))

    xs = np.array([])
    ys = np.array([])
    xs2 = np.array([])
    ys2 = np.array([])
    


    a = 0.65
    b = 1.2
    g = np.radians(111)
    #known_x = known_x.unsqueeze(-1)
    predicted_params = model(known_x).detach().numpy()[0]
    print("params:", predicted_params)
    a2 = predicted_params[0]
    b2 = predicted_params[1]
    g2 = predicted_params[2]
    actual_qs = 1/ generator(np.array([a,b,g]).reshape(1, -1))
    pred_qs = 1 / generator(predicted_params.reshape(1,-1))
    for M in range(-3, 3):
        for N in range(-3, 3):
            #if M == 0 and N == 0:
            #    continue
            xs = np.append(xs, M*a + N*b*np.cos(g))
            ys = np.append(ys, N*b*np.sin(g))
    for M in range(-3, 3):
        for N in range(-3, 3):
            #if M == 0 and N == 0:
            #    continue
            xs2 = np.append(xs2, M*a2 + N*b2*np.cos(g2))
            ys2 = np.append(ys2, N*b2*np.sin(g2))

    plt.scatter(xs, ys)
    plt.scatter(xs2, ys2)
    plt.show()

    plt.clf()
    print(actual_qs.shape)
    plt.scatter(actual_qs.reshape(-1), [0.5]*len(actual_qs.reshape(-1)))
    for q in pred_qs.reshape(-1):
        plt.axvline(x=q)
    plt.show()
