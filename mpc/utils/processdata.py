import numpy as np
import torch
import os

#Helper function to construct the training and testing datasets

def process_data_shuff(x0_obs, xd_obs, u_sim, data_dir, use_dagger):

    x_data = np.hstack((x0_obs, xd_obs))
    #Concatenate x_data and u_data along the second axis
    combined_data = np.hstack((x_data, u_sim))
    expert_data_dir = os.path.join(data_dir, 'expert_data.npy')
    

    if use_dagger:
        #Add the expert data to the combined data
        
        if os.path.exists(expert_data_dir):
            expert_data = np.load(expert_data_dir)
            combined_data = np.vstack((combined_data, expert_data))
        
    np.save(expert_data_dir, combined_data)
    
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

    torch.save(x_train.float(), os.path.join(data_dir, 'x_train.pt'))
    torch.save(x_test.float(), os.path.join(data_dir, 'x_test.pt'))
    torch.save(u_train.float(), os.path.join(data_dir, 'u_train.pt'))
    torch.save(u_test.float(), os.path.join(data_dir, 'u_test.pt'))


def process_data_sequential(x0_obs, xd_obs, u_sim, data_dir, n_samples, use_dagger):
    expert_data_dir = os.path.join(data_dir, 'expert_data.npz')
    x_data = np.hstack((x0_obs, xd_obs))
    if use_dagger:
        expert_data = np.load(expert_data_dir)
        x_expert = expert_data['x']
        u_expert = expert_data['u']
        x_data = np.vstack((x_data, x_expert))
        u_sim = np.vstack((u_sim, u_expert))

    np.savez(expert_data_dir, x=x_data, u=u_sim)

    n_traj = int(x_data.shape[0]/n_samples)
    n_train = int(n_traj*0.8)
    n_test = n_traj - n_train
    sequence_length = 5


    x_train = np.array([x_data[i:i+sequence_length] for i in range(n_train*n_samples-sequence_length)])
    u_train = np.array([u_sim[i+sequence_length] for i in range(n_train*n_samples-sequence_length)])
    x_test = np.array([x_data[i:i+sequence_length] for i in range(n_train*n_samples, n_traj*n_samples-sequence_length)])
    u_test = np.array([u_sim[i+sequence_length] for i in range(n_train*n_samples, n_traj*n_samples-sequence_length)])

    x_train = torch.tensor(x_train)
    u_train = torch.tensor(u_train)
    x_test = torch.tensor(x_test)
    u_test = torch.tensor(u_test)

    torch.save(x_train.float(), os.path.join(data_dir, 'x_train.pt'))
    torch.save(x_test.float(), os.path.join(data_dir, 'x_test.pt'))
    torch.save(u_train.float(), os.path.join(data_dir, 'u_train.pt'))
    torch.save(u_test.float(), os.path.join(data_dir, 'u_test.pt'))


