from sklearn import preprocessing
from sklearn.externals import joblib
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import fit_d


class SimpleNet(torch.nn.Module):
    """
    :param num_spacings: The number of d-spacings per input the network should be trained on.

    SimpleNet is a fairly shallow fully connected network with batch normalization.
    """

    def __init__(self, num_spacings=8):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(num_spacings, 350) # input linear layer
        self.bn1 = nn.BatchNorm1d(350)  # batchnorm layers aid in training.  The dimension of the input here (350) must match the dimension of the output of the previous layer.
        self.hidden1 = nn.Linear(350, 350)
        self.bn2 = nn.BatchNorm1d(350)
        self.hidden2 = nn.Linear(350, 250)
        self.bn3 = nn.BatchNorm1d(250)
        self.hidden3 = nn.Linear(250, 100)
        self.bn4 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 3) # linear layer with no activation or softmax, known as a "regression layer" in MATLAB.

    # Defines the operations to perform in the forwards pass.
    def forward(self, x):
        x = self.linear1(x) # First linear layer
        x = F.relu(self.bn1(x)) # Apply batchnorm to linear output, then apply a Relu activation.
        # in Torch, it's common to apply parameterless layers (like ReLU) using torch.nn.functional rather than saving it as a layer in the model.
        x = self.hidden1(x)
        x = F.relu(self.bn2(x))

        x = self.hidden2(x)
        x = F.relu(self.bn3(x))

        x = self.hidden3(x)
        x = F.relu(self.bn4(x))

        x = self.linear2(x)
        # no activation function on the output layer
        return x

    # Pytorch automatically symbolically differentiates the forward pass with autograd to generate the corresponding backwards pass.  We don't need to do anything!

def train_model(num_epochs=75, path="model.pth", use_cuda=False, gamma_scheduler=0.25, scheduler_step_size=20, batch_size=192, use_qs=False, lr=1e-3, num_spacings=8):
    """
    :param num_epochs: Number of epochs to train model for
    :param path: Path to save trained model state dict in
    :param use_cuda: Whether or not to use CUDA when training (not yet implemented)
    :param gamma_scheduler: Factor to decay learning rate by
    :param scheduler_step_size: How many epochs to wait between learning rate decays
    :param batch_size: Batch size during training
    :param use_qs: Whether or not to use q values as inputs rather than d values
    :param lr: Base learning rate to use during training
    :param num_spacings: Number of d spacings in each input the model should be trained with
    :return: None
    :Side effects: When training has finished, the final model state is saved into path and a dump of the scaler is saved in "scaler.save"

    """
    model = SimpleNet(num_spacings=num_spacings)
    criterion = nn.MSELoss() # mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma_scheduler) # decays the learning rate by a factor of gamma_scheduler every scheduler_step_size epochs

    generator = fit_d.dSpaceGenerator(gen_q=use_qs) # generates d-spacing vectors from a,b,gamma vectors
    yTr = torch.Tensor(fit_d.gen_input(5000)) # generate y's for scaling
    xTr = torch.Tensor(generator(yTr))  # generate x's for scaling
    scaler = preprocessing.StandardScaler().fit(xTr)  # 0-1 normalization is essential
    joblib.dump(scaler, "scaler.save") # save the scaler for use during evaluation

    for epoch in range(num_epochs):
        # The primary training loop
        running_loss = 0
        yTr = torch.Tensor(fit_d.gen_input(15000))  # each loop we generate 15000 training inputs
        xTr = torch.Tensor(generator(yTr)) # generate X's
        xTr = torch.Tensor(scaler.transform(xTr)) # normalize X's
        dataset = torch.utils.data.TensorDataset(xTr, yTr) # sets up the dataset, required for the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # dataloader provides an iterator over data as (input minibatch, label minibatch) tuples
        # hidden = None
        for inputs, labels in dataloader:
            # Iteration through the dataset
            # Batch size is the first dimension
            optimizer.zero_grad()  # must clear gradients from previous iteration before current
            outputs = model(inputs)  # raw logits from the model
            loss = criterion(outputs, labels)  # get the error
            loss.backward()  # backpropagate the loss
            optimizer.step()  # update model parameters based off the backpropagated loss
            running_loss += loss.item()  # and accumulate the loss so we can track it
        running_loss /= len(dataloader) # normalize the loss
        print("Epoch {0}: loss {1}".format(epoch, running_loss))
        scheduler.step()  # scheduler steps every epoch for learning rate decay

    # Generate 1000 points for validation
    yTe = torch.Tensor(fit_d.gen_input(1000))
    xTe = torch.Tensor(generator(yTe))
    xTe = torch.Tensor(scaler.transform(xTe))

    test_dataset = torch.utils.data.TensorDataset(xTe, yTe)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    torch.save(model.state_dict(), path)  # save model weights to a file to be loaded later
    model.eval()  # make sure to set model to eval mode before any predictions--batchnorm has different behavior
    preds = torch.Tensor() # an empty tensor
    i = 0
    for inputs, labels in test_dataloader:
        # inputs = inputs.unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        if i == 5:
            print("actual: ", labels)
            print("predicted: ", outputs)
        preds = torch.cat((preds, outputs), 0) # append/concatenate to the empty tenosr
        i += 1


    print("average test error:", torch.mean(preds - yTe))
