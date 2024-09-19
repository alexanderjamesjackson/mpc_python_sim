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

import pickle as pkl
import scipy.sparse as sparse
from scipy.io import loadmat
from utils import sim_mpc as sim
from utils import model as md
from utils import processdata as process

from utils import diamond_I_configuration_v5 as DI
import pandas as pd
import time

#Helper function to generate random disturbancemodes

def randModes(seed, RM, id_to_bpm, TOT_BPM, u_mag):
    np.random.seed(seed)
    UR, SR, VR = np.linalg.svd(RM)
    weighted_combination = np.zeros_like(UR[:, 0])  
    doff_tmp = np.zeros((TOT_BPM, 1))
    for i in range(len(SR)):
        #Generate random pertubation
        pert = np.random.uniform(-1,1,1)
        #Change the ith mode by the pertubation
        weighted_combination += pert * UR[:, i] * u_mag
    doff_tmp[id_to_bpm] = weighted_combination[:, np.newaxis]
    doff = doff_tmp * np.ones((1,n_samples))
    #Add noise to the disturbance modes
    # for i in range(n_samples):
    #     if i % 50 == 0:
    #         doff[id_to_bpm, i] = doff[id_to_bpm, i] + np.random.normal(0, 1, 1)
    return doff




start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load model parameters
size_file_path = '../data/sizes.pkl'
with open(size_file_path, 'rb') as f:
    sizes = pkl.load(f)

n_state = sizes['n_state']
n_ctrl = sizes['n_ctrl']
hidden_size = sizes['hidden_size']
n_sc = sizes['n_sc']
num_layers = sizes['num_layers']
sequence_length = sizes['sequence_length']
sequence_length = 3
model_path = '../data/model/model.ckpt'
nnparams = torch.load(model_path)


# options
fname_RM = '../orms/GoldenBPMResp_DIAD.mat'
pick_dir = 1
dirs = ['horizontal','vertical']
pick_direction = dirs[pick_dir]
sim_IMC = False
use_FGM = True
#Simulates multiple modes of disturbance to get training data
train = False
trainRNN = False    #///DO NOT USE IMP BROKEN
#Toggle for comparing nn performance and mpc performance
compare = False
#Toggle for using DAGGER ///DO NOT USE IMP BROKEN
use_dagger = False
#Toggle for LQR limits
use_lqr = False

#Hardlimits
fname_correctors = '../data/corrector_data.csv'
correctors = pd.read_csv(fname_correctors)
hardlimits = correctors['MaxAmps'].iloc[:172]
hardlimits = hardlimits.to_numpy().reshape(172, 1)

#Configure Diamond-I storage ring
Rmat = loadmat(fname_RM)
RMorigx = Rmat['Rmat'][0][0][0]
ny_x = np.size(RMorigx, 0)
nu_x = np.size(RMorigx, 1) 
RMorigy = Rmat['Rmat'][1][1][0]
ny_y = np.size(RMorigy, 0) 
nu_y = np.size(RMorigy, 1)
assert ny_x == ny_y
assert nu_x == nu_y
TOT_BPM = np.size(RMorigx, 0)
TOT_CM = np.size(RMorigx, 1)
square_config = True
id_to_bpm_x, id_to_cm_x, id_to_bpm_y, id_to_cm_y = DI.diamond_I_configuration_v5(RMorigx, RMorigy, square_config)

#first n_include BPMs and CMs active for testing

n_include = 165


id_to_bpm_x = id_to_bpm_x[:n_include]
id_to_cm_x = id_to_cm_x[:n_include]
id_to_bpm_y = id_to_bpm_y[:n_include]
id_to_cm_y = id_to_cm_y[:n_include]

RMx = RMorigx[np.ix_(id_to_bpm_x, id_to_cm_x)]
RMy = RMorigy[np.ix_(id_to_bpm_y, id_to_cm_y)]

# condition_number = np.linalg.cond(RMy)
# print(condition_number)
# UR, SR, VR = np.linalg.svd(RMy)
# print(SR[1])

fname = '../data/systems/165statesystemnd8.mat'
#OnlyValidforND8
mat_data = loadmat(fname)

#Observer and Regulator
n_delay = 8
# fname = f'../data/mpc_data_13092022_nd{n_delay}.mat'
# mat_data = loadmat(fname)


Fs = 10000
Ts = 1/Fs
# fname = fname = f'../data/mpc_data_13092022_nd{n_delay}.mat'
# mat_data.update(loadmat(fname))

if pick_direction == 'vertical':
    id_to_bpm = id_to_bpm_y
    id_to_cm = id_to_cm_y
    RM = RMy
    aI_Hz = 700
    #Observer
    Ao = mat_data['Ao_y']
    Bo = mat_data['Bo_y']
    Co = mat_data['Co_y']
    Ad = mat_data['Ad_y']
    Cd = mat_data['Cd_y']
    #Plant with all BPMs and CMs
    Ap = mat_data['Ap_y']
    Bp = mat_data['Bp_y']
    Cp = mat_data['Cp_y']
    Kfd = mat_data['Kfd_y'] #Observer gain for disturbance
    Kfx = mat_data['Kfx_y'] #Observer gain for state
    P_mpc = mat_data['P_y'] #Terminal cost
    Q_mpc = mat_data['Qlqr_y'] #State weighting
    R_mpc = mat_data['Rlqr_y'] #Input weighting
    #SOFB

else:
    id_to_bpm = id_to_bpm_x
    id_to_cm = id_to_cm_x
    RM = RMx
    aI_Hz = 500
    #Observer
    Ao = mat_data['Ao_x']
    Bo = mat_data['Bo_x']
    Co = mat_data['Co_x']
    Ad = mat_data['Ad_x']
    Cd = mat_data['Cd_x']
    #Plant with all BPMs and CMs
    Ap = mat_data['Ap_x']
    Bp = mat_data['Bp_x']
    Cp = mat_data['Cp_x']
    Kfd = mat_data['Kfd_x']
    Kfx = mat_data['Kfx_x']
    P_mpc = mat_data['P_x']
    Q_mpc = mat_data['Qlqr_x']
    R_mpc = mat_data['Rlqr_x']
    #SOFB

ny = np.size(RM, 0)
nu = np.size(RM, 1)
nx = nu

#Observer
Lxd_obs = Kfd
Lx8_obs = Kfx
S_sp_pinv = np.linalg.pinv(np.block([[np.eye(nx) - Ao, -Bo], [Co, np.zeros((ny, nu))]]))
S_sp_pinv = S_sp_pinv[:,nx:]

#MPC
horizon = 1
u_rate_scalar = 1*1000
u_rate = np.ones((nu, 1)) * u_rate_scalar
u_max = hardlimits[id_to_cm] * 1000
y_max_scalar = np.infty
y_max = np.ones((id_to_bpm.size, 1)) * y_max_scalar
J_mpc = np.transpose(Bo) @ P_mpc @ Bo + R_mpc
S_sp_pinv_x = S_sp_pinv[:nx,:]
S_sp_pinv_u = S_sp_pinv[nx:,:]
q_mat_x0 = np.transpose(Bo) @ P_mpc @ Ao
q_mat_xd = (np.hstack((Bo.T @ P_mpc, R_mpc)) @ np.vstack((S_sp_pinv_x, S_sp_pinv_u)) @ Cd)
q_mat = np.hstack((q_mat_x0, q_mat_xd))

#Set up FGM
beta_fgm = 0

if use_FGM:
    eigmax = np.max(np.linalg.eigvals(J_mpc))
    eigmin = np.min(np.linalg.eigvals(J_mpc))   
    J_mpc = np.eye(J_mpc.shape[0]) - J_mpc / eigmax
    beta_fgm = (np.sqrt(eigmax) - np.sqrt(eigmin)) / (np.sqrt(eigmax) + np.sqrt(eigmin))
    q_mat = np.hstack((q_mat_x0, q_mat_xd)) / eigmax


#Rate limiter on VME processors
if pick_direction == 'vertical':
    mat_data.update(loadmat('../data/awrSSy.mat'))

else:
    mat_data.update(loadmat('../data/awrSSx.mat'))


n_samples = 15000



#Initialise array of seeds for pertubations
n_traj = 10
trainseeds = np.linspace(1, n_traj*n_include, n_traj*n_include).astype(int)
u_mags = np.linspace(1, 100, n_traj*n_include)
n_tests = n_traj * n_include

#Storage for training data
u_sim_train = np.zeros((n_samples * n_tests, n_include))
xd_obs_train = np.zeros((n_samples * n_tests, n_include))
x0_obs_train = np.zeros((n_samples * n_tests, n_include))
y_sim_train = np.zeros((n_samples * n_tests , n_include))

#SOFB setpoints
SOFB_setp = np.zeros((nu, 1))

if train:
    k = 0
    n_complete = 0
    for seed, u_mag in zip(trainseeds, u_mags):
        print('[{}/{}]'.format(n_complete, n_tests))
        #Generate random disturbance modes based on seed
        doff = randModes(seed, RM, id_to_bpm, TOT_BPM, u_mag)

        #Simulation
        endt = n_samples*Ts - Ts
        Lsim = n_samples*Ts
        t = np.arange(0, endt + Ts, Ts)

    
        #Initialise mpc
        mpc = sim.Mpc(
            n_samples, n_delay, doff,
            Ap, Bp, Cp, 
            Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
            J_mpc, q_mat, y_max,
            u_max, u_rate,
            id_to_bpm, id_to_cm,
            mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
            SOFB_setp, beta_fgm, use_lqr=use_lqr)

        if use_dagger:
            model = md.NNController(n_state, hidden_size, n_ctrl)
            model.load_state_dict(nnparams)
            #Simulate and store trajectory
            u_sim_expert, x0_obs_dggr, xd_obs_dggr, n_simulated = mpc.sim_dagger(model, device, sequence_length=sequence_length, RNN=False)
            
            try:
                assert(n_simulated < n_samples)
                u_sim_train[k:k+n_simulated, :] = u_sim_expert[:,id_to_cm]
                xd_obs_train[k:k+n_simulated, :] = xd_obs_dggr
                x0_obs_train[k:k+n_simulated, :] = x0_obs_dggr
            except AssertionError as e:
                print("Did not converge")
                k -= n_simulated
        else:
            #Simulate and store trajectory
            y_sim_fgm ,u_sim_fgm, x0_obs_fgm, xd_obs_fgm, _ , n_simulated = mpc.sim_mpc(use_FGM)
            #Check that mpc has converged
            try:
                assert(n_simulated < n_samples)
                u_sim_train[k:k+n_simulated, :] = u_sim_fgm[:,id_to_cm]
                xd_obs_train[k:k+n_simulated, :] = xd_obs_fgm
                x0_obs_train[k:k+n_simulated, :] = x0_obs_fgm
                y_sim_train[k:k+n_simulated, :] = y_sim_fgm[:,id_to_bpm]
            except AssertionError as e:
                print("Did not converge")
                k -= n_simulated

        k += n_simulated
        n_complete += 1

    u_sim_train = u_sim_train[:k, :]
    xd_obs_train = xd_obs_train[:k, :]
    x0_obs_train = x0_obs_train[:k, :]
    # y_sim_train = y_sim_train[:k, :]
    # plt.plot(u_sim_train * 0.001)
    # plt.show()
    # plt.plot(xd_obs_train)
    # plt.show()
    # plt.plot(x0_obs_train)
    # plt.show()

else:
    #Generate random disturbance modes based on testing seed
    doff = randModes(4220, RM, id_to_bpm, TOT_BPM, 100)
    # mag_u = 1000
    # UR,SR,VR = np.linalg.svd(RM)
    # imode = 2
    # tmp = UR[:, imode - 1] * SR[imode - 1] * mag_u
    # doff_tmp = np.zeros((TOT_BPM, 1))
    # doff_tmp[id_to_bpm] = tmp[:, np.newaxis]
    # doff = doff_tmp * np.ones((1,n_samples))
    #Simulation
    endt = n_samples*Ts - Ts
    Lsim = n_samples*Ts
    t = np.arange(0, endt + Ts, Ts)
    #Initialise mpc
    mpc = sim.Mpc(
        n_samples, n_delay, doff,
        Ap, Bp, Cp, 
        Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
        J_mpc, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
        SOFB_setp, beta_fgm, use_lqr = use_lqr)

    #Simulate and store trajectory

    y_sim_fgm ,u_sim_fgm, x0_obs_fgm, xd_obs_fgm,x_sim, n_simulated = mpc.sim_mpc(use_FGM)


if train:
    data_dir = '../data'
    if trainRNN:
        process.process_data_sequential(x0_obs_train, xd_obs_train, u_sim_train, data_dir, n_samples, use_dagger)
    else:  
        process.process_data_shuff(x0_obs_train, xd_obs_train, u_sim_train, data_dir, use_dagger)
    


if compare:
    #Initialise mpc 
    mpc = sim.Mpc(
        n_samples, n_delay, doff,
        Ap, Bp, Cp, 
        Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
        J_mpc, q_mat, y_max,
        u_max, u_rate,
        id_to_bpm, id_to_cm,
        mat_data['A'][:n_include,:n_include], mat_data['B'][:n_include,:n_include], mat_data['C'][:n_include,:n_include], mat_data['D'][:n_include,:n_include],
        SOFB_setp, beta_fgm, use_lqr=use_lqr)
    #Load nn model
    model = md.LinearController(n_state,n_ctrl)
    model.load_state_dict(nnparams)
    #Simulate using nn model
    y_sim_nn ,u_sim_nn, x0_obs_nn, xd_obs_nn = mpc.sim_nn(model, device, RNN=False, sequence_length=sequence_length)
    y_nn_longterm = y_sim_nn[:, id_to_bpm]
    y_sim_nn = y_sim_nn[:n_simulated, :]
    u_sim_nn = u_sim_nn[:n_simulated, :]
    x0_obs_nn = x0_obs_nn[:n_simulated, :]
    xd_obs_nn = xd_obs_nn[:n_simulated, :]

    y_err = (y_sim_fgm - y_sim_nn) 
    u_err = (u_sim_fgm - u_sim_nn) 





#Unpack Loss Data






# Plotting

if not use_dagger:
    loss_data = np.load('../data/model/modelloss.npz')
    y_plt_fgm = y_sim_fgm[:, id_to_bpm]
    u_plt_fgm = u_sim_fgm[:, id_to_cm]
    if compare:
        y_plt_nn = y_sim_nn[:, id_to_bpm]
        u_plt_nn = u_sim_nn[:, id_to_cm]
        u_plt_err = u_err[:, id_to_cm]
        y_plt_err = y_err[:, id_to_bpm]

    scale_u = 0.001

    fig, axs = plt.subplots(2, 4, figsize=(15, 8))

    # Subplot 1: Disturbance
    axs[0, 0].plot(doff[id_to_bpm, :].T)
    axs[0, 0].set_title('Disturbance')


    # Subplot 2: Input
    axs[0, 1].plot(u_plt_fgm * scale_u, linestyle='-')  # solid line for u_sim_fgm
    if compare:
        axs[0, 1].plot(u_plt_nn * scale_u, linestyle='--')  # dashed line for u_sim_nn
    axs[0, 1].set_title('Input')



    start_index = int(n_simulated/4)
    n_plt = np.linspace(0, n_simulated, n_simulated)

    # # Subplot 3: % nn Steady State
    if compare:
        # axs[0, 2].plot(n_plt[start_index:],y_plt_nn[start_index:])  # dashed line for y_sim_nn
        axs[0, 2].plot(n_plt[start_index:],y_plt_nn[start_index:], linestyle='--')
        # axs[0, 2].plot(y_nn_longterm, linestyle='-.')
        axs[0, 2].set_title('NN Steady State Output')

    #Subplot 4: MPC Steady State Output
    
    axs[0, 3].plot(n_plt[start_index:],y_plt_fgm[start_index:])  # solid line for y_sim_fgm
    
    axs[0, 3].set_title('MPC Steady State Output')




    # Subplot 4: Output
    axs[1, 0].plot(y_plt_fgm, linestyle='-')  # solid line for y_sim_fgm
    if compare:
        axs[1, 0].plot(y_plt_nn, linestyle='--')  # dashed line for y_sim_nn
    axs[1, 0].set_title('Output')


    # Subplot 5: % Error in Output
    if compare:
        axs[1, 1].plot(y_plt_err, linestyle='-')  # solid line for y_sim_fgm
    axs[1, 1].set_title('Output Error')

    #Subplot 6 : Training Loss
    if compare:
        axs[1, 2].plot(loss_data['epochs'], loss_data['train_losses'])
        axs[1, 2].set_xlabel('Epoch')
        axs[1, 2].set_ylabel('Loss')
        axs[1, 2].set_title('Training Loss')
        
    #Subplot 7: Validation Loss
    if compare:
        #Only plot once model has settled
        axs[1, 3].plot(loss_data['epochs'][10:], loss_data['val_losses'][10:])
        axs[1, 3].set_xlabel('Epoch')
        axs[1, 3].set_ylabel('Loss')
        axs[1, 3].set_title('Validation Loss')




    
    print("Time taken: ", time.time() - start)
    # # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()






    







