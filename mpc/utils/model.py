import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F


# Define the neural network model

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
    
class LinearController(nn.Module):
    def __init__(self, n_state, n_ctrl):
        super(LinearController, self).__init__()
        self.type = 'linear'
        self.fc = nn.Linear(n_state, n_ctrl)

    def forward(self, x):
        return self.fc(x)
    

class RNNController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNController, self).__init__()
        self.type = 'rnn' + str(num_layers) + 'layers'
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden state to zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through RNN
        out, _ = self.rnn(x, h0)
        
        # Pass the last output from RNN to the fully connected layer
        out = self.fc(out[:, -1, :])  # out[:, -1, :] takes the last time step
        
        return out
    