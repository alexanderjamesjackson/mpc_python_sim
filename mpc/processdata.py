import numpy as np
import torch
import os

results = np.load('../data/simresults.npz')

x0_obs = results['x0_obs']
xd_obs = results['xd_obs']
u_sim = results['u_sim']
print(u_sim.shape)

x0_obs = x0_obs.squeeze()
xd_obs = xd_obs.squeeze()


x_data = np.hstack((x0_obs, xd_obs))

#Concatenate x_data and u_data along the second axis
combined_data = np.hstack((x_data, u_sim))

np.random.seed(42)
#Shuffle the combined data randomly
np.random.shuffle(combined_data)

#Split the combined data back into x and u components

x_data_shuffled = combined_data[:, :x_data.shape[1]]
u_data_shuffled = combined_data[:, x_data.shape[1]:]

#Split the shuffled data into training and testing sets
train_size = int(0.8 * combined_data.shape[0])


x_train = torch.tensor(x_data_shuffled[:train_size])
x_test = torch.tensor(x_data_shuffled[train_size:])

u_train = torch.tensor(u_data_shuffled[:train_size])
u_test = torch.tensor(u_data_shuffled[train_size:])


data_dir = '../data/'

torch.save(x_train.float(), os.path.join(data_dir, 'x_train.pt'))
torch.save(x_test.float(), os.path.join(data_dir, 'x_test.pt'))
torch.save(u_train.float(), os.path.join(data_dir, 'u_train.pt'))
torch.save(u_test.float(), os.path.join(data_dir, 'u_test.pt'))