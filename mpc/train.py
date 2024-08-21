import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import numpy.random as npr

import os
import matplotlib.pyplot as plt

import time
import pickle as pkl

import threading
from utils import model as md

# cuda
device = 'cpu'
data_dir = '../data'

# Load tensors from the specified directory
x_train = torch.load(os.path.join(data_dir, 'x_train.pt'))
x_test = torch.load(os.path.join(data_dir, 'x_test.pt'))
u_train = torch.load(os.path.join(data_dir, 'u_train.pt'))
u_test = torch.load(os.path.join(data_dir, 'u_test.pt'))


#Hyperparameters
hidden_size = 256
learning_rate = 1e-5
num_epochs = 400
weight_decay = 1e-4
batch_size = 20

# Create TensorDatasets
train_dataset = torch.utils.data.TensorDataset(x_train, u_train)
test_dataset = torch.utils.data.TensorDataset(x_test, u_test)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)





n_state, n_ctrl = x_train.size(1), u_train.size(1)
n_sc = n_state + n_ctrl

# Save the sizes
sizes={
    'n_state': n_state,
    'n_ctrl': n_ctrl,
    'hidden_size': hidden_size,
    'n_sc': n_sc
}

fname = os.path.join(data_dir, 'sizes.pkl')
with open(fname, 'wb') as f:
    pkl.dump(sizes, f)




# Construct the NN model
model = md.NNController(n_state, hidden_size, n_ctrl)
model.to(device)
# Loss and optimizer
criterion = md.get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# store the loss values
loss_values = []
train_losses =[]
val_losses = []
epochs = []

# Function for testing the model
def test_model(model, test_loader):
    with torch.no_grad():
        loss_value = []
        for x_test, u_test in test_loader:
            x_test = x_test.to(device)
            u_test = u_test.to(device)
            predictions = model(x_test)
            loss_value.append(criterion(predictions, u_test))
    return np.mean(loss_value)

# Train the model
total_step = len(train_loader)
start = time.time()

#Start thread to monitor user input for early exit
keep_training = True
exit_event = threading.Event()

#Function to check for user input
def check_user_input():
    global keep_training
    input("Press Enter to stop training here...\n")
    keep_training = False
    exit_event.set()


input_thread = threading.Thread(target=check_user_input)
input_thread.start()


for epoch in range(num_epochs):  # episode size
    if keep_training == False:
        break
    model.eval()
    val_loss = test_model(model, test_loader)
    val_losses.append(val_loss.item())
    TimeRemaining = ((time.time()-start) / (epoch+1)) * (num_epochs - epoch)
    model.train()
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
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}, Time Remaining: {} mins {} s, stop(ENTER)'.format(epoch+1, num_epochs, i + 1, total_step, loss.item(), val_loss, int(TimeRemaining/60), round(TimeRemaining%60)))
    epochs.append(epoch)
    train_losses.append(loss.item())
    
# Signal the thread to exit if it hasn't already
exit_event.set()

# Wait for the input thread to finish
input_thread.join()


print('Test Loss: {:.4f}'.format(test_model(model, test_loader)))

data_dir = '../data/model'
torch.save(model.state_dict(), os.path.join(data_dir, 'model.ckpt'))

epochs = np.array(epochs)
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

np.savez('../data/model/modelloss' , epochs = epochs, train_losses = train_losses, val_losses = val_losses)

fig, axs = plt.subplots(1, 2, figsize=(15, 8))

axs[0].plot(epochs, train_losses)
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss')

axs[1].plot(epochs, val_losses)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_title('Validation Loss')

plt.show()