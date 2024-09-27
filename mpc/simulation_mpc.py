import numpy as np
import scipy.sparse as sparse
from scipy.io import loadmat
from utils import sim_mpc as sim
from utils import diamond_I_configuration_v5 as DI
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import os
# Script to simulate the MPC controller on the full Diamond Light Source storage ring

start = time.time()

# options
# Load Orbit Response Matrix
fname_RM = '../orms/GoldenBPMResp_DIAD.mat'
pick_dir = 1
dirs = ['horizontal','vertical']
pick_direction = dirs[pick_dir]
do_step = True
sim_IMC = False
use_FGM = True

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
RMx = RMorigx[np.ix_(id_to_bpm_x, id_to_cm_x)]
RMy = RMorigy[np.ix_(id_to_bpm_y, id_to_cm_y)]

#Observer and Regulator
n_delay = 8
# Ensure that the correct file is loaded here! 
fname = f'../data/systems/165statesystemnd8.mat'


mat_data = loadmat(fname)



Fs = 10000
Ts = 1/Fs


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
y_max_scalar = 200
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

#Measurement Data
if do_step:
    n_samples = 2000
    doff = np.ones((TOT_BPM, 1)) * np.ones((1, n_samples)) * 10
    don = doff

else:
    print("no data at this time")

imode = 100
n_samples = 6000

if pick_direction == 'vertical':
    UR, SR, VR = np.linalg.svd(RMy)
else:
    UR, SR, VR = np.linalg.svd(RMx)

mag_u = 1000
tmp = UR[:, imode - 1] * SR[imode - 1] * mag_u
doff_tmp = np.zeros((TOT_BPM, 1))
doff_tmp[id_to_bpm] = tmp[:, np.newaxis]
doff = doff_tmp * np.ones((1,n_samples))
don = doff
y_max = np.ones((id_to_bpm.size, 1)) * 850
#Simulation



endt = n_samples*Ts - Ts
Lsim = n_samples*Ts
t = np.arange(0, endt + Ts, Ts)

SOFB_setpoints = np.zeros((1,172))
SOFB_setp = np.transpose(SOFB_setpoints[:,id_to_cm])
SOFB_setp = np.where(SOFB_setp > u_max, u_max, SOFB_setp)
SOFB_setp = np.where(SOFB_setp < -u_max, -u_max, SOFB_setp)




mpc = sim.Mpc(
    n_samples, n_delay, doff,
    Ap, Bp, Cp, 
    Ao, Bo, Co, Ad, Cd, Lx8_obs, Lxd_obs,
    J_mpc, q_mat, y_max,
    u_max, u_rate,
    id_to_bpm, id_to_cm,
    mat_data['A'], mat_data['B'], mat_data['C'], mat_data['D'],
    SOFB_setp, beta_fgm)




y_sim ,u_sim, x0_obs, xd_obs, _, n_samples = mpc.sim_mpc(use_FGM)


# Plotting


scale_u = 0.001

fig, axs = plt.subplots(2, 4, figsize=(15, 8))

# Subplot 1: Disturbance
axs[0, 0].plot(doff[id_to_bpm, :].T)
axs[0, 0].set_title('Disturbance')

# Subplot 2: Disturbance Mode Space
axs[0, 1].plot((UR.T @ doff[id_to_bpm, :]).T)
axs[0, 1].set_title('Disturbance Mode Space')

# Subplot 3: Input
axs[0, 2].plot(u_sim[:, id_to_cm] * scale_u)
axs[0, 2].set_title('Input')

# Subplot 4: Input Mode Space
axs[0, 3].plot(scale_u * u_sim[:, id_to_cm] @ VR)
axs[0, 3].set_title('Input Mode Space')

# Subplot 5: Output
axs[1, 0].plot(y_sim[:, id_to_bpm])
axs[1, 0].set_title('Output')

# Subplot 6: Output Mode Space
axs[1, 1].plot(y_sim[:, id_to_bpm] @ UR)
axs[1, 1].set_title('Output Mode Space')

print("Time taken: ", time.time() - start)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()









