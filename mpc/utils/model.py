import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F

# Classes to define various models used in the MPC project

# Define the arbitary neural network model

class NNController(nn.Module):
    def __init__(self, n_state, hidden_size, n_ctrl):
        super(NNController, self).__init__()
        self.type = 'LinReLux4'
        # Initialize weights and biases for all layers
        self.fc1 = nn.Linear(n_state, hidden_size)

        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))

        self.act2 = nn.ReLU()
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(int(hidden_size/2), n_ctrl)


    def forward(self, x):  # x: (n_batch, n_state)
        out = self.fc1(x)

        out = self.act1(out)
        out = self.fc2(out)

        out = self.act2(out)
        # out = self.fc3(out)
        # out = self.act3(out)
        x = self.fc4(out)

        return x


# Define the loss function
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.norm(predictions-targets)
        return loss


# Define the linear controller 
class LinearController(nn.Module):
    def __init__(self, n_state, n_ctrl):
        super(LinearController, self).__init__()
        self.type = 'linear'
        self.fc = nn.Linear(n_state, n_ctrl)

    def forward(self, x):
        return self.fc(x)
    


    