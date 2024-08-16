import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim

import numpy as np
import numpy.random as npr

import os
import matplotlib.pyplot as plt

import pickle as pkl

# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '../data'

# Load tensors from the specified directory
x_train = torch.load(os.path.join(data_dir, 'x_train.pt'))
x_test = torch.load(os.path.join(data_dir, 'x_test.pt'))
u_train = torch.load(os.path.join(data_dir, 'u_train.pt'))
u_test = torch.load(os.path.join(data_dir, 'u_test.pt'))



# Create TensorDatasets
train_dataset = torch.utils.data.TensorDataset(x_train, u_train)
test_dataset = torch.utils.data.TensorDataset(x_test, u_test)

# Create DataLoaders
batch_size = 5
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


hidden_size = 32
#Change rate of learning
learning_rate = 1e-4
num_epochs = 500



n_state, n_ctrl = x_train.size(1), u_train.size(1)
n_sc = n_state + n_ctrl

sizes={
    'n_state': n_state,
    'n_ctrl': n_ctrl,
    'hidden_size': hidden_size,
    'n_sc': n_sc
}

fname = os.path.join(data_dir, 'sizes.pkl')
with open(fname, 'wb') as f:
    pkl.dump(sizes, f)

class NNController(nn.Module):
    def __init__(self, n_state, hidden_size, n_ctrl):
        super(NNController, self).__init__()
        self.hidden_size = hidden_size

        # Initialize weights and biases for all layers
        self.fc1 = nn.Linear(n_state, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, n_ctrl)

    def forward(self, x):  # x: (n_batch, n_state)
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        return out


# Define the loss function
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.norm(predictions-targets)
        return loss


# Construct the NN model
model = NNController(n_state, hidden_size, n_ctrl)

# Loss and optimizer
criterion = get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# store the loss values
loss_values = []
epoch_losses =[]
epochs = []

# Train the model!
total_step = len(train_loader)
for epoch in range(num_epochs):  # episode size
    for i, (x_train, u_train) in enumerate(train_loader):
        # Move tensors to the configured device

        x_train = x_train.to(device)
        u_train = u_train.to(device)

        # Forward pass
        predictions = model(x_train)
        loss = criterion(predictions, u_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store the loss value
        loss_values.append(loss.item())

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i + 1, total_step, loss.item()))
    epochs.append(epoch)
    epoch_losses.append(loss.item())
    



# Test the model
with torch.no_grad():
    loss_value = []
    for x_test, u_test in test_loader:
        x_test = x_test.to(device)
        u_test = u_test.to(device)
        predictions = model(x_test)
        loss_value.append(criterion(predictions, u_test))

    print('Test Loss: {:.4f}'.format(np.mean(loss_value)))

data_dir = '../data/model'
torch.save(model.state_dict(), os.path.join(data_dir, 'model.ckpt'))

epochs = np.array(epochs)
epoch_losses = np.array(epoch_losses)

np.savez('../data/model/modelloss' , epochs = epochs, epoch_losses = epoch_losses)


plt.plot(epochs, epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()