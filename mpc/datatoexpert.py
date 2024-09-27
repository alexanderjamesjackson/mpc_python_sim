import numpy as np
import torch
import os

# Helper function for a situation where you have a dataset and you want to convert it to expert data


data_dir = '../data'
x_train = torch.load(os.path.join(data_dir, 'x_train.pt')).numpy()
x_test = torch.load(os.path.join(data_dir, 'x_test.pt')).numpy()
u_train = torch.load(os.path.join(data_dir, 'u_train.pt')).numpy()
u_test = torch.load(os.path.join(data_dir, 'u_test.pt')).numpy()

combined_data_train = np.hstack((x_train, u_train))
combined_data_test = np.hstack((x_test, u_test))

combined_data = np.vstack((combined_data_train, combined_data_test))

expert_data_dir = os.path.join(data_dir, 'expert_data.npy')
np.save(expert_data_dir, combined_data)



