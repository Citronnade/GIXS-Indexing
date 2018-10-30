import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import fit_d
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
#from logger import Logger

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(30, 150)
        self.bn1 = nn.BatchNorm1d(150)
        self.bn2 = nn.BatchNorm1d(150)
        self.bn3 = nn.BatchNorm1d(150)
        self.bn4 = nn.BatchNorm1d(100)
        self.hidden1 = nn.Linear(150, 150)
        self.hidden2 = nn.Linear(150, 150)
        self.hidden3 = nn.Linear(150, 100)
        self.linear2 = nn.Linear(100, 3)
        



    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        #x = F.relu(self.bn1(x))
        x = self.hidden1(x)
        #x = F.relu(self.bn2(x))
        x = F.relu(x)
        x = self.hidden2(x)
        #x = F.relu(self.bn3(x))
        x = F.relu(x)
        x = self.hidden3(x)
        #x = F.relu(self.bn4(x))
        x = F.relu(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    #logger = Logger("logs")
    yTr = torch.Tensor(fit_d.gen_input(100000))
    xTr = torch.Tensor(list(map(lambda x: fit_d.gen_d_vector(*x, noise=0.02), yTr)))

    model = SimpleNet()
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.99, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    scaler = preprocessing.StandardScaler().fit(xTr)
    xTr = torch.Tensor(scaler.transform(xTr))
    print(xTr)
    dataset = torch.utils.data.TensorDataset(xTr, yTr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True)

    yTe = torch.Tensor(fit_d.gen_input(1000))
    xTe = torch.Tensor(list(map(lambda x: fit_d.gen_d_vector(*x, noise=0.01), yTe)))
    xTe = torch.Tensor(scaler.transform(xTe))

    test_dataset = torch.utils.data.TensorDataset(xTe, yTe)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)


    known_y = np.array([.65, 1.2, np.radians(111)]).reshape(1,-1)
    known_x = np.array(fit_d.gen_d_vector(*known_y[0]))
    known_x = scaler.transform(known_x.reshape(1,-1))
    known_x = torch.Tensor(known_x)
    print(known_x)
    
    for epoch in range(50):
        running_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        info = {
            'loss_' : running_loss
            }
        #for tag, value in info.items():
        #    logger.scalar_summary(tag, value, epoch+1)
        print("Epoch {0}: loss {1}".format(epoch, running_loss))
        scheduler.step()

    model.eval()
    preds = torch.Tensor()
    for inputs, labels in test_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        preds = torch.cat((preds, outputs), 0)

    print("average test error:", torch.mean(preds - yTe))

    xs = np.array([])
    ys = np.array([])
    xs2 = np.array([])
    ys2 = np.array([])
    


    a = 0.65
    b = 1.2
    g = np.radians(111)

    predicted_params = model(known_x).detach().numpy()[0]
    a2 = predicted_params[0]
    b2 = predicted_params[1]
    g2 = predicted_params[2]
    actual_qs = 1/ np.array(fit_d.gen_d_vector(a,b,g))
    pred_qs = 1 / np.array(fit_d.gen_d_vector(a2,b2,g2, H_max = 20, K_max = 20))
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
    plt.scatter(actual_qs, [0.5]*len(actual_qs))
    for q in pred_qs:
        plt.axvline(x=q)
    plt.show()
